from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common import project_config
from ppo_simple import mappo_baseline, policy, primitive_control, rollout, trainer
from scripts.training import mappo_stage3_pool_contract


class FactorizedCacheKeepIdentityTests(unittest.TestCase):
    def test_factorized_keep_rows_have_ordered_identity_and_keep_other_features_zero(self) -> None:
        node_ids = ("sv-0", "rsu-0", "uav-0")
        roles = ("SV", "RSU", "UAV")
        context = SimpleNamespace(
            action_spec=SimpleNamespace(cache_node_ids=node_ids),
            cache_nodes={"node_ids": node_ids, "node_roles": roles},
            cache_cfg=SimpleNamespace(target_candidate_capacity=3),
            observation_spec=SimpleNamespace(observation_dim=1),
            service_catalog={"num_services": 2},
        )
        step = SimpleNamespace(
            scenario_slice={
                "arrival": np.zeros((1, 1, 1), dtype=np.int8),
                "service_id": np.zeros((1, 1, 1), dtype=np.int64),
                "task_vehicle_ids": ("tv-0",),
                "vehicle_ids": ("tv-0",),
            },
            prediction_horizon_mask=np.ones(1, dtype=bool),
        )
        current = np.zeros((len(node_ids), 2), dtype=np.int8)
        primitive_dim = 18 + 6 * context.service_catalog["num_services"]

        def proposals_for_node(*args, **kwargs):
            node_index = int(args[3])
            identity = primitive_control.cache_node_identity_features(
                node_index, len(node_ids), roles[node_index]
            )
            row = np.zeros((1, primitive_dim), dtype=np.float32)
            row[0, :4] = identity
            return [SimpleNamespace(target_node_cache=np.asarray([1, 0], dtype=np.int8))], row

        with (
            patch.object(
                rollout.uav_configuration,
                "cache_nodes_with_uav_positions",
                return_value=context.cache_nodes,
            ),
            patch.object(
                rollout,
                "_target_cache_proposals_with_features",
                side_effect=proposals_for_node,
            ),
            patch.object(
                primitive_control.cache_configuration,
                "predicted_vehicle_service_demand_by_horizon",
                return_value=np.ones((1, 1, 2), dtype=np.float64),
            ),
            patch.object(
                primitive_control.cache_configuration,
                "_node_predicted_serviceability_weights",
                return_value=np.ones((1, 1), dtype=np.float64),
            ),
        ):
            candidates, features, masks = rollout._cache_node_candidates(
                current,
                {},
                step,
                context,
                SimpleNamespace(cache_feature_dim=primitive_dim),
            )

        feature_dim = 2 * current.size + 1 + primitive_control.CACHE_ECONOMIC_FEATURE_DIM
        self.assertEqual(features.shape, (3, 2, feature_dim))
        self.assertEqual(masks.shape, (3, 2))
        self.assertTrue(np.all(masks))
        self.assertEqual(len(candidates), 3)
        self.assertTrue(np.isfinite(features[masks]).all())

        economic_offset = feature_dim - primitive_control.CACHE_ECONOMIC_FEATURE_DIM
        expected_roles = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        keep_identities = []
        for node_index in range(len(node_ids)):
            keep = features[node_index, 0, economic_offset:]
            self.assertEqual(float(keep[0]), 1.0)
            self.assertAlmostEqual(float(keep[1]), node_index / 2.0)
            np.testing.assert_array_equal(keep[2:5], expected_roles[node_index])
            np.testing.assert_array_equal(keep[5:], np.zeros(14, dtype=np.float32))
            keep_identities.append(tuple(keep[1:5].tolist()))
        self.assertEqual(len(set(keep_identities)), 3)

        # UPDATE economic identities retain the pre-existing primitive encoding.
        for node_index in range(len(node_ids)):
            update = features[node_index, 1, economic_offset:]
            self.assertAlmostEqual(float(update[1]), node_index / 2.0)
            np.testing.assert_array_equal(update[2:5], expected_roles[node_index])

    def test_update_feature_identity_encoder_matches_existing_formula(self) -> None:
        node_ids = ("sv-0", "rsu-0", "uav-0")
        roles = ("SV", "RSU", "UAV")
        context = SimpleNamespace(
            service_catalog={"num_services": 2, "service_sizes_mb": np.asarray([5.0, 8.0])},
            cache_nodes={
                "node_ids": node_ids,
                "node_roles": roles,
                "capacities_mb": np.asarray([20.0, 20.0, 20.0]),
            },
            cache_cfg=SimpleNamespace(),
        )
        current = np.zeros((3, 2), dtype=np.int8)
        expected_roles = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

        for node_index, role in enumerate(roles):
            target = np.asarray([1, 0], dtype=np.int8)
            candidate = SimpleNamespace(
                node_index=node_index,
                before_node_cache=current[node_index].copy(),
                target_node_cache=target,
            )
            with patch.object(
                primitive_control.cache_configuration,
                "cache_update_cost",
                return_value=(0.0, np.zeros(3, dtype=np.float64)),
            ):
                values = primitive_control._cache_target_configuration_feature(
                    candidate,
                    current,
                    current,
                    np.zeros(3, dtype=np.int8),
                    np.zeros(2, dtype=np.float64),
                    context,
                )
            self.assertAlmostEqual(float(values[0]), node_index / 2.0)
            np.testing.assert_array_equal(values[1:4], expected_roles[node_index])


class CachePolicyVersionTests(unittest.TestCase):
    def test_only_factorized_and_sparse_schemas_advance(self) -> None:
        import tomllib

        root = Path(__file__).resolve().parents[1]
        system = tomllib.loads(
            (root / "configs/proposed/system_config.toml").read_text(encoding="utf-8-sig")
        )
        sparse = tomllib.loads(
            (root / "configs/proposed/sparse_coordinated_cache_v1.toml").read_text(
                encoding="utf-8-sig"
            )
        )
        self.assertEqual(
            system["algorithm"]["ppo"]["policy_schema"],
            project_config.MULTI_NODE_FACTORIZED_POLICY_SCHEMA,
        )
        self.assertEqual(
            sparse["algorithm"]["ppo"]["policy_schema"],
            project_config.SPARSE_COORDINATED_CACHE_POLICY_SCHEMA,
        )
        system_path = root / "configs/proposed/system_config.toml"
        system_config = project_config.parse_toml_file(system_path)
        self.assertEqual(
            project_config.raw_config_sha256(system_path),
            mappo_stage3_pool_contract.CURRENT_FORMAL_CONFIG_SHA256,
        )
        self.assertEqual(
            project_config.semantic_config_hash(system_config),
            mappo_stage3_pool_contract.CURRENT_FORMAL_SEMANTIC_SHA256,
        )
        self.assertEqual(
            mappo_stage3_pool_contract.LEGACY_POOL_FIELD_DIFFERENCES[
                "algorithm.ppo.policy_schema"
            ][1],
            project_config.MULTI_NODE_FACTORIZED_POLICY_SCHEMA,
        )
        self.assertIn("v5_multi_node_cache_factorized", project_config.MULTI_NODE_FACTORIZED_POLICY_SCHEMA)
        self.assertIn("v6_sparse_coordinated_cache", project_config.SPARSE_COORDINATED_CACHE_POLICY_SCHEMA)
        self.assertIn("keep_node_identity_v1", project_config.MULTI_NODE_FACTORIZED_POLICY_SCHEMA)
        self.assertIn("keep_node_identity_v1", project_config.SPARSE_COORDINATED_CACHE_POLICY_SCHEMA)
        self.assertTrue(project_config.PROPOSED_POLICY_SCHEMA.endswith("v1"))
        self.assertEqual(policy.MultiNodeFactorizedActorCritic.architecture_version, 36)
        self.assertEqual(policy.SparseCoordinatedCacheActorCritic.architecture_version, 37)
        self.assertEqual(policy.AdaptiveMultiNodeFactorizedActorCritic.architecture_version, 2)
        self.assertEqual(policy.AdaptiveSparseCoordinatedCacheActorCritic.architecture_version, 2)
        self.assertEqual(policy.AdaptiveDualTimescaleActorCritic.architecture_version, 1)

    def test_old_factorized_checkpoint_architecture_is_rejected(self) -> None:
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "old_factorized.pt"
            torch.save(
                {
                    "checkpoint_version": 1,
                    "architecture_version": 1,
                    "task_execution_model": "PER_SLOT_ANALYTIC",
                },
                checkpoint_path,
            )
            fake_trainer = SimpleNamespace(
                device=torch.device("cpu"),
                checkpoint_version=1,
                policy=SimpleNamespace(architecture_version=2),
            )
            with self.assertRaisesRegex(ValueError, "architecture/action semantics"):
                trainer.PPOTrainer.load_checkpoint(fake_trainer, checkpoint_path)

    def test_old_factorized_checkpoint_contract_is_rejected_without_global_bump(self) -> None:
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "old_factorized_baseline.pt"
            torch.save(
                {
                    "checkpoint_version": mappo_baseline.MAPPO_CHECKPOINT_VERSION,
                    "baseline_schema": mappo_baseline.MAPPO_BASELINE_SCHEMA,
                    # Old contracts did not identify the Cache candidate input semantics.
                    "model_contract": {
                        "baseline_schema": mappo_baseline.MAPPO_BASELINE_SCHEMA,
                        "macro_candidate_feature_dim": 19,
                    },
                },
                checkpoint_path,
            )
            fake_trainer = SimpleNamespace(
                device=torch.device("cpu"),
                stage3_cache_value_config=None,
                _checkpoint_contract=lambda: {
                    "baseline_schema": mappo_baseline.MAPPO_BASELINE_SCHEMA,
                    "macro_candidate_feature_dim": 19,
                    "policy_schema": policy.AdaptiveMultiNodeFactorizedActorCritic.architecture_id,
                    "architecture_version": policy.AdaptiveMultiNodeFactorizedActorCritic.architecture_version,
                },
            )
            self.assertEqual(
                mappo_baseline.MAPPO_BASELINE_SCHEMA,
                "mappo_shared_actor_global_critic_fixed_macro_2_v2_group_balanced_ops",
            )
            with self.assertRaisesRegex(ValueError, "model/config contract mismatch"):
                mappo_baseline.MAPPOBaselineTrainer.load_checkpoint(
                    fake_trainer, checkpoint_path
                )


if __name__ == "__main__":
    unittest.main()
