from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from common import project_config, system_model
from decision import cache_configuration, uav_configuration
from scripts.training import training_pool_contract


class CostFormulaMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configs = {
            name: project_config.load_config(path, require_artifacts=False)
            for name, path in (
                ("factorized", "configs/proposed/system_config.toml"),
                ("sparse", "configs/proposed/sparse_coordinated_cache_v1.toml"),
            )
        }

    def test_proposed_configs_use_new_cost_contract_without_legacy_weights(self) -> None:
        for name, config in self.configs.items():
            with self.subTest(config=name):
                objective = config["objective"]
                self.assertNotIn("cache_cost_weight", objective)
                self.assertNotIn("uav_movement_cost_weight", objective)
                self.assertEqual(float(objective["lambda_cache"]), 0.01)
                self.assertAlmostEqual(float(objective["lambda_movement"]), 8.8e-6)
                cache_config = cache_configuration.cache_config_from_project(
                    config
                )
                movement_config = uav_configuration.uav_search_config_from_project(config)
                self.assertFalse(hasattr(cache_config, "cache_cost_weight"))
                self.assertFalse(hasattr(movement_config, "movement_cost_weight"))

    def test_deprecated_outer_cost_weights_are_rejected(self) -> None:
        config = deepcopy(self.configs["factorized"])
        config["objective"]["cache_cost_weight"] = 1.0
        with self.assertRaisesRegex(ValueError, "cache_cost_weight 已废弃"):
            project_config.validate_config(
                config, require_artifacts=False
            )

        config = deepcopy(self.configs["factorized"])
        config["objective"]["uav_movement_cost_weight"] = 1.0
        with self.assertRaisesRegex(ValueError, "uav_movement_cost_weight 已废弃"):
            project_config.validate_config(
                config, require_artifacts=False
            )

    def test_training_pool_compatibility_uses_generation_contract(self) -> None:
        manifest = {
            "config_path": training_pool_contract.CONFIG_RELATIVE.as_posix(),
            "target_generation_config_fingerprint": (
                training_pool_contract.CURRENT_GENERATION_FINGERPRINT
            ),
            # Recorded full-config hashes remain provenance only.
            "config_sha256": "pool-config-hash",
            "semantic_config_sha256": "pool-semantic-hash",
        }
        compatibility = training_pool_contract._validate_pool_config_compatibility(
            manifest,
            generation_fingerprint=training_pool_contract.CURRENT_GENERATION_FINGERPRINT,
        )
        self.assertEqual(
            compatibility["policy"], "TRAINING_DATA_GENERATION_COMPATIBILITY_V1"
        )

        manifest["target_generation_config_fingerprint"] = "different-generation"
        with self.assertRaisesRegex(ValueError, "generation contract mismatch"):
            training_pool_contract._validate_pool_config_compatibility(
                manifest,
                generation_fingerprint=training_pool_contract.CURRENT_GENERATION_FINGERPRINT,
            )

    def test_cache_keep_add_delete_and_replace_costs_match_old_formula(self) -> None:
        service_sizes = np.asarray([100.0, 250.0], dtype=np.float64)
        old_xi = np.asarray([1.0e-3, 0.7e-3, 1.2e-3], dtype=np.float64)
        old_eta = np.asarray([0.20, 0.10, 0.15], dtype=np.float64)
        new_eta = np.asarray([2.0, 1.0, 1.5], dtype=np.float64)
        old_weight = 0.1
        old_lambda = 0.1
        new_lambda = 0.01
        cases = (
            ("keep", 0, [1, 0], [1, 0]),
            ("add", 0, [0, 0], [1, 0]),
            ("delete", 1, [1, 0], [0, 0]),
            ("replace", 2, [1, 0], [0, 1]),
        )
        for label, node_index, old_row, new_row in cases:
            with self.subTest(case=label):
                previous = np.asarray([old_row], dtype=np.int8)
                final = np.asarray([new_row], dtype=np.int8)
                changed = int(not np.array_equal(previous[0], final[0]))
                trigger = np.asarray([changed], dtype=np.int8)
                old_trigger = np.asarray([changed], dtype=np.float64)
                loading_proxy = old_xi[node_index] * float(
                    np.maximum(final[0] - previous[0], 0) @ service_sizes
                )
                old_cost = old_weight * (
                    old_trigger[0] * old_eta[node_index]
                    + old_lambda * loading_proxy
                )
                result = system_model.cache_update_cost(
                    previous,
                    final,
                    trigger,
                    service_sizes,
                    np.asarray([new_eta[node_index]]),
                    np.asarray([old_xi[node_index]]),
                    new_lambda,
                )
                self.assertEqual(int(result["actual_trigger"][0]), changed)
                self.assertAlmostEqual(float(result["loaded_mb"][0]), float(
                    np.maximum(final[0] - previous[0], 0) @ service_sizes
                ))
                self.assertAlmostEqual(float(result["loading_energy"][0]), loading_proxy)
                self.assertAlmostEqual(
                    float(result["loading_energy_proxy"][0]), loading_proxy
                )
                self.assertAlmostEqual(float(result["cost_by_node"][0]), old_cost, places=14)
                self.assertAlmostEqual(float(result["total_cost"]), old_cost, places=14)
                if label in {"keep", "delete"}:
                    self.assertEqual(float(result["loaded_mb"][0]), 0.0)

    def test_uav_stay_and_move_costs_match_old_formula(self) -> None:
        config = self.configs["factorized"]
        movement = uav_configuration.uav_search_config_from_project(config)
        old_weight = 0.50
        old_lambda = 1.76e-5
        old_eta = 0.00176
        energy_coefficient = movement.movement_energy_coefficient_j_per_m

        stay = system_model.uav_movement_cost(
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            0,
            movement.dt_s,
            movement.max_horizontal_speed_mps,
            energy_coefficient,
            movement.movement_fixed_cost,
            movement.lambda_movement,
            tolerance_m=movement.position_tolerance_m,
        )
        self.assertEqual(stay["actual_z_q"], 0)
        self.assertEqual(stay["movement_energy_j"], 0.0)
        self.assertEqual(stay["movement_cost"], 0.0)

        distance = min(10.0, movement.max_move_distance_m)
        move = system_model.uav_movement_cost(
            np.asarray([0.0, 0.0]),
            np.asarray([distance, 0.0]),
            1,
            movement.dt_s,
            movement.max_horizontal_speed_mps,
            energy_coefficient,
            movement.movement_fixed_cost,
            movement.lambda_movement,
            tolerance_m=movement.position_tolerance_m,
        )
        old_energy = energy_coefficient * distance
        old_cost = old_weight * (old_eta + old_lambda * old_energy)
        self.assertEqual(move["actual_z_q"], 1)
        self.assertAlmostEqual(float(move["movement_energy_j"]), old_energy)
        self.assertAlmostEqual(float(move["movement_cost"]), old_cost, places=14)
        self.assertAlmostEqual(
            float(move["movement_cost"]),
            movement.lambda_movement * (movement.movement_fixed_cost + old_energy),
            places=14,
        )

if __name__ == "__main__":
    unittest.main()
