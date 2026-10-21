from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common import project_config
from decision import cache_configuration, slot_environment
from ppo_simple import policy, primitive_control, rollout, trainer


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
        current = np.tile(np.asarray([1, 0], dtype=np.int8), (len(node_ids), 1))
        primitive_dim = 18 + 6 * context.service_catalog["num_services"]

        def proposals_for_node(*args, **kwargs):
            node_index = int(args[3])
            identity = primitive_control.cache_node_identity_features(
                node_index, len(node_ids), roles[node_index]
            )
            targets = (
                [
                    np.asarray([0, 0], dtype=np.int8),
                    np.asarray([1, 1], dtype=np.int8),
                    np.asarray([1, 0], dtype=np.int8),
                ]
                if node_index == 0
                else [
                    np.asarray([0, 1], dtype=np.int8),
                    np.asarray([0, 0], dtype=np.int8),
                    np.asarray([1, 1], dtype=np.int8),
                    np.asarray([1, 0], dtype=np.int8),
                ]
            )
            row = np.zeros((len(targets), primitive_dim), dtype=np.float32)
            row[:, :4] = identity
            return [SimpleNamespace(target_node_cache=target) for target in targets], row

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
        np.testing.assert_array_equal(
            masks,
            np.asarray([[True, False], [True, True], [True, True]], dtype=bool),
        )
        self.assertEqual(len(candidates), 3)
        self.assertEqual([len(node_candidates) for node_candidates in candidates], [1, 2, 2])
        self.assertTrue(np.isfinite(features[masks]).all())
        for node_index, node_candidates in enumerate(candidates):
            np.testing.assert_array_equal(node_candidates[0], current)
            for target in node_candidates[1:]:
                changed_rows = np.flatnonzero(np.any(target != current, axis=1))
                np.testing.assert_array_equal(changed_rows, np.asarray([node_index]))
                self.assertTrue(
                    cache_configuration.is_legal_cache_replacement(
                        current[node_index], target[node_index]
                    )
                )

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
        for node_index in (1, 2):
            update = features[node_index, 1, economic_offset:]
            self.assertAlmostEqual(float(update[1]), node_index / 2.0)
            np.testing.assert_array_equal(update[2:5], expected_roles[node_index])

        actor = policy.MultiNodeFactorizedActorCritic(
            observation_dim=1,
            macro_cache_feature_dim=features.shape[-1],
            micro_uav_feature_dim=1,
            adaptive_holding_min_slots=1,
            adaptive_holding_max_slots=2,
            hidden_sizes=(8,),
        )
        selected = actor.sample_macro_nodes(np.zeros(1, dtype=np.float32), features, masks)
        np.testing.assert_array_equal(selected.cache_masks, masks)
        self.assertTrue(np.all(selected.node_probabilities[~masks] == 0.0))
        for node_index, candidate_index in enumerate(selected.cache_actions):
            self.assertTrue(masks[node_index, candidate_index])
            if candidate_index == 0:
                np.testing.assert_array_equal(candidates[node_index][0], current)
            else:
                self.assertTrue(
                    cache_configuration.is_legal_cache_replacement(
                        current[node_index], candidates[node_index][candidate_index][node_index]
                    )
                )

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
        current = np.tile(np.asarray([1, 0], dtype=np.int8), (3, 1))
        expected_roles = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

        for node_index, role in enumerate(roles):
            target = np.asarray([0, 1], dtype=np.int8)
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


class CacheReplacementRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service_catalog = {
            "num_services": 4,
            "service_sizes_mb": np.ones(4, dtype=np.float64),
        }
        self.cache_nodes = {
            "node_ids": ("sv-0",),
            "node_roles": ("SV",),
            "capacities_mb": np.asarray([3.0], dtype=np.float64),
            "cache_cost_normalization": {"reference_kind": "test"},
        }
        self.before = np.asarray([[1, 1, 0, 0]], dtype=np.int8)

    def _build_policy_target(self, final_cache: np.ndarray, trigger: int):
        search = SimpleNamespace(
            position_tolerance_m=1.0e-6,
            movement_energy_reference_j=0.0,
            beta_movement=0.0,
        )
        cache_cost_details = {
            "cache_cost": 0.0,
            "cache_download_delay_s": 0.0,
            "cache_download_delay_normalized": 0.0,
            "cache_download_delay_reference_s": 6000.0,
            "cache_cost_beta": 0.12,
            "cache_download_delay_s_by_node": np.zeros(1, dtype=np.float64),
            "cache_cost_by_node": np.zeros(1, dtype=np.float64),
        }
        with (
            patch.object(
                slot_environment.uav_configuration,
                "validate_uav_configuration_transition",
                return_value=None,
            ),
            patch.object(
                slot_environment.uav_configuration,
                "uav_search_config_from_project",
                return_value=search,
            ),
            patch.object(
                slot_environment.uav_configuration,
                "movement_summary",
                return_value=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            patch.object(
                slot_environment.cache_configuration,
                "cache_update_cost_details",
                return_value=cache_cost_details,
            ),
        ):
            return slot_environment.build_policy_target_configuration(
                self.before,
                {},
                final_cache,
                {},
                np.asarray([trigger], dtype=np.int8),
                {},
                self.service_catalog,
                {},
                self.cache_nodes,
                None,
                (0.0, 0.0, 1.0, 1.0),
            )

    def test_bounded_candidates_are_replacements_and_allow_unequal_counts(self) -> None:
        config = SimpleNamespace(
            target_candidate_capacity=12,
            target_intermediate_fractions=(0.25, 0.5, 0.75),
        )
        cache_nodes = {
            "node_ids": ("sv-0",),
            "capacities_mb": np.asarray([3.0], dtype=np.float64),
        }
        demand = np.asarray([0.01, 0.01, 1.0, 0.9], dtype=np.float64)
        with patch.object(
            cache_configuration,
            "node_specific_predicted_demand",
            return_value=demand,
        ):
            candidates = cache_configuration.generate_bounded_cache_target_configurations(
                self.before,
                0,
                self.service_catalog,
                cache_nodes,
                {},
                config,
            )

        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertTrue(
                cache_configuration.is_legal_cache_replacement(
                    self.before[0], candidate.target_node_cache
                )
            )
            self.assertLessEqual(
                float(candidate.target_node_cache @ self.service_catalog["service_sizes_mb"]),
                float(cache_nodes["capacities_mb"][0]),
            )
        self.assertTrue(
            any(
                np.count_nonzero((self.before[0] == 1) & (candidate.target_node_cache == 0))
                != np.count_nonzero((self.before[0] == 0) & (candidate.target_node_cache == 1))
                for candidate in candidates
            ),
            "candidate generation must not require equal removal/addition counts",
        )

        empty_cache = np.zeros_like(self.before)
        self.assertFalse(
            cache_configuration.has_legal_cache_target_configuration(
                empty_cache, 0, self.service_catalog, cache_nodes
            )
        )
        with patch.object(
            cache_configuration,
            "node_specific_predicted_demand",
            return_value=demand,
        ):
            self.assertEqual(
                cache_configuration.generate_bounded_cache_target_configurations(
                    empty_cache,
                    0,
                    self.service_catalog,
                    cache_nodes,
                    {},
                    config,
                ),
                [],
            )

    def test_formal_target_builder_enforces_keep_and_replace(self) -> None:
        keep = self._build_policy_target(self.before.copy(), 0)
        self.assertEqual(keep.accepted_cache_updates, 0)
        replacement = np.asarray([[1, 0, 1, 1]], dtype=np.int8)
        changed = self._build_policy_target(replacement, 1)
        self.assertEqual(changed.accepted_cache_updates, 1)
        applied = cache_configuration.apply_cache_target_configuration(
            self.before.copy(),
            0,
            replacement[0],
            self.service_catalog,
            self.cache_nodes,
        )
        np.testing.assert_array_equal(applied["target_node_cache"], replacement[0])

        invalid_targets = (
            np.asarray([[1, 0, 0, 0]], dtype=np.int8),  # pure deletion
            np.asarray([[1, 1, 1, 0]], dtype=np.int8),  # pure addition
            self.before.copy(),  # UPDATE with no change
        )
        for target in invalid_targets:
            with self.subTest(target=target.tolist()), self.assertRaises(ValueError):
                self._build_policy_target(target, 1)
            with self.subTest(apply_target=target[0].tolist()), self.assertRaises(ValueError):
                cache_configuration.apply_cache_target_configuration(
                    self.before.copy(),
                    0,
                    target[0],
                    self.service_catalog,
                    self.cache_nodes,
                )


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

if __name__ == "__main__":
    unittest.main()
