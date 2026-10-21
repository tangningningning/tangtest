from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from common import project_config, system_model
from decision import cache_configuration, uav_configuration


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

    def test_formal_config_requires_explicit_backhaul_rates_and_cost_weights(self) -> None:
        config = self.configs["factorized"]
        objective = config["objective"]
        self.assertNotIn("cache_cost_weight", objective)
        self.assertNotIn("uav_movement_cost_weight", objective)
        self.assertEqual(float(objective["lambda_cache"]), 2.0e-5)
        self.assertAlmostEqual(float(objective["lambda_movement"]), 5.0e-6)
        for role in ("sv", "rsu", "uav"):
            self.assertEqual(
                float(config["nodes"][role]["cache_backhaul_rate_bps"]),
                200.0e6,
            )
        cache_config = cache_configuration.cache_config_from_project(config)
        movement_config = uav_configuration.uav_search_config_from_project(config)
        service_vehicle_count = int(config["simulation"]["num_service_vehicles"])
        reference_cache_nodes = {
            "node_ids": [
                *(f"SV_{index}" for index in range(service_vehicle_count)),
                *(f"RSU_{index}" for index in range(cache_config.num_rsu)),
                *(f"UAV_{index}" for index in range(cache_config.num_uav)),
            ],
            "capacities_mb": np.asarray(
                [cache_config.sv_capacity_mb] * service_vehicle_count
                + [cache_config.rsu_capacity_mb] * cache_config.num_rsu
                + [cache_config.uav_capacity_mb] * cache_config.num_uav,
                dtype=np.float64,
            ),
            "cache_backhaul_rate_bps": np.asarray(
                [cache_config.sv_cache_backhaul_rate_bps] * service_vehicle_count
                + [cache_config.rsu_cache_backhaul_rate_bps] * cache_config.num_rsu
                + [cache_config.uav_cache_backhaul_rate_bps] * cache_config.num_uav,
                dtype=np.float64,
            ),
        }
        cost_reference = cache_configuration.initialize_cache_cost_normalization(
            reference_cache_nodes,
            np.asarray([1.0], dtype=np.float64),
            cache_config.lambda_cache,
        )
        self.assertAlmostEqual(float(cost_reference["reference_s"]), 6000.0)
        self.assertAlmostEqual(float(cost_reference["beta_cache"]), 0.12)
        self.assertAlmostEqual(movement_config.movement_energy_reference_j, 450.0)
        self.assertAlmostEqual(movement_config.beta_movement, 0.00225)
        self.assertFalse(hasattr(cache_config, "sv_eta_ca"))
        self.assertFalse(hasattr(cache_config, "sv_xi_ca"))
        self.assertFalse(hasattr(movement_config, "movement_cost_weight"))

        with self.assertRaisesRegex(ValueError, "缺少必需配置 nodes.sv.cache_backhaul_rate_bps"):
            cache_configuration.cache_config_from_project(self.configs["sparse"])

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

    def test_cache_keep_delete_and_add_costs_use_download_delay_only(self) -> None:
        service_sizes = np.asarray([2000.0], dtype=np.float64)
        backhaul_rate_bps = np.asarray([200.0e6], dtype=np.float64)
        lambda_cache = 2.0e-5
        cache_nodes = {
            "node_ids": ["SV_0", "RSU_0"],
            "capacities_mb": np.asarray([2000.0, 2000.0], dtype=np.float64),
            "cache_backhaul_rate_bps": backhaul_rate_bps.repeat(2),
        }
        reference = cache_configuration.initialize_cache_cost_normalization(
            cache_nodes, service_sizes, lambda_cache
        )
        self.assertEqual(reference["reference_kind"], "conservative_capacity_upper_bound")
        self.assertAlmostEqual(float(reference["reference_s"]), 160.0)
        self.assertAlmostEqual(float(reference["beta_cache"]), 0.0032)
        cases = (
            ("keep", [1], [1], 0.0, 0.0),
            ("add", [0], [1], 80.0, 0.0016),
            ("delete", [1], [0], 0.0, 0.0),
        )
        for label, old_row, new_row, expected_delay_s, expected_cost in cases:
            with self.subTest(case=label):
                previous = np.asarray([old_row], dtype=np.int8)
                final = np.asarray([new_row], dtype=np.int8)
                changed = int(not np.array_equal(previous[0], final[0]))
                trigger = np.asarray([changed], dtype=np.int8)
                result = system_model.cache_update_cost(
                    previous,
                    final,
                    trigger,
                    service_sizes,
                    backhaul_rate_bps,
                    reference["reference_s"],
                    reference["beta_cache"],
                )
                self.assertEqual(int(result["actual_trigger"][0]), changed)
                self.assertAlmostEqual(
                    float(result["cache_download_delay_s_by_node"][0]),
                    expected_delay_s,
                )
                self.assertAlmostEqual(float(result["cache_cost_by_node"][0]), expected_cost)
                self.assertAlmostEqual(float(result["cache_download_delay_s"]), expected_delay_s)
                self.assertAlmostEqual(float(result["cache_cost"]), expected_cost)

        multi_node_result = system_model.cache_update_cost(
            np.asarray([[0], [0]], dtype=np.int8),
            np.asarray([[1], [1]], dtype=np.int8),
            np.asarray([1, 1], dtype=np.int8),
            service_sizes,
            np.asarray([200.0e6, 200.0e6], dtype=np.float64),
            reference["reference_s"],
            reference["beta_cache"],
        )
        np.testing.assert_allclose(
            multi_node_result["cache_download_delay_s_by_node"], [80.0, 80.0]
        )
        np.testing.assert_allclose(multi_node_result["cache_cost_by_node"], [0.0016, 0.0016])
        self.assertAlmostEqual(multi_node_result["cache_download_delay_s"], 160.0)
        self.assertAlmostEqual(multi_node_result["cache_cost"], 0.0032)
        self.assertAlmostEqual(
            multi_node_result["cache_download_delay_normalized"], 1.0
        )

    def test_zero_normalization_references_do_not_divide_by_zero(self) -> None:
        cache_nodes = {
            "node_ids": ["SV_0"],
            "capacities_mb": np.asarray([100.0], dtype=np.float64),
            "cache_backhaul_rate_bps": np.asarray([200.0e6], dtype=np.float64),
        }
        reference = cache_configuration.initialize_cache_cost_normalization(
            cache_nodes,
            np.asarray([2000.0], dtype=np.float64),
            2.0e-5,
        )
        self.assertEqual(float(reference["reference_s"]), 0.0)
        self.assertEqual(float(reference["beta_cache"]), 0.0)
        keep = system_model.cache_update_cost(
            np.asarray([[0]], dtype=np.int8),
            np.asarray([[0]], dtype=np.int8),
            np.asarray([0], dtype=np.int8),
            np.asarray([2000.0], dtype=np.float64),
            np.asarray([200.0e6], dtype=np.float64),
            reference["reference_s"],
            reference["beta_cache"],
        )
        self.assertEqual(float(keep["cache_download_delay_normalized"]), 0.0)
        self.assertEqual(float(keep["cache_cost"]), 0.0)
        with self.assertRaisesRegex(ValueError, "zero reference"):
            system_model.cache_update_cost(
                np.asarray([[0]], dtype=np.int8),
                np.asarray([[1]], dtype=np.int8),
                np.asarray([1], dtype=np.int8),
                np.asarray([2000.0], dtype=np.float64),
                np.asarray([200.0e6], dtype=np.float64),
                reference["reference_s"],
                reference["beta_cache"],
            )

    def test_uav_stay_and_move_costs_use_movement_energy_only(self) -> None:
        config = self.configs["factorized"]
        movement = uav_configuration.uav_search_config_from_project(config)
        energy_coefficient = movement.movement_energy_coefficient_j_per_m
        self.assertEqual(energy_coefficient, 15.0)
        self.assertEqual(movement.movement_fixed_cost, 0.0)
        self.assertEqual(movement.lambda_movement, 5.0e-6)

        stay = system_model.uav_movement_cost(
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            0,
            movement.dt_s,
            movement.max_horizontal_speed_mps,
            energy_coefficient,
            movement.movement_fixed_cost,
            movement.movement_energy_reference_j,
            movement.beta_movement,
            tolerance_m=movement.position_tolerance_m,
        )
        self.assertEqual(stay["actual_z_q"], 0)
        self.assertEqual(stay["movement_energy_j"], 0.0)
        self.assertEqual(stay["movement_cost"], 0.0)

        distance = min(5.0, movement.max_move_distance_m)
        move = system_model.uav_movement_cost(
            np.asarray([0.0, 0.0]),
            np.asarray([distance, 0.0]),
            1,
            movement.dt_s,
            movement.max_horizontal_speed_mps,
            energy_coefficient,
            movement.movement_fixed_cost,
            movement.movement_energy_reference_j,
            movement.beta_movement,
            tolerance_m=movement.position_tolerance_m,
        )
        expected_energy = energy_coefficient * distance
        expected_cost = movement.lambda_movement * expected_energy
        self.assertEqual(move["actual_z_q"], 1)
        self.assertAlmostEqual(float(move["movement_energy_j"]), expected_energy)
        self.assertAlmostEqual(float(move["movement_cost"]), expected_cost, places=14)
        self.assertAlmostEqual(
            float(move["movement_cost"]),
            movement.lambda_movement * expected_energy,
            places=14,
        )
        self.assertAlmostEqual(float(move["movement_energy_j"]), 75.0)
        self.assertAlmostEqual(float(move["movement_cost"]), 0.000375)

        zero_reference_stay = system_model.uav_movement_cost(
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            0,
            movement.dt_s,
            movement.max_horizontal_speed_mps,
            energy_coefficient,
            movement.movement_fixed_cost,
            0.0,
            0.0,
            tolerance_m=movement.position_tolerance_m,
        )
        self.assertEqual(zero_reference_stay["movement_cost"], 0.0)

if __name__ == "__main__":
    unittest.main()
