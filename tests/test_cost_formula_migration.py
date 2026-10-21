from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from common import project_config, system_model
from decision import cache_configuration, uav_configuration


class CostFormulaMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = project_config.load_config(
            "configs/proposed/alternating_actor_updates_v1.toml",
            require_artifacts=False,
        )

    def test_formal_config_and_node_level_references(self) -> None:
        config = self.config
        objective = config["objective"]
        self.assertNotIn("cache_cost_weight", objective)
        self.assertNotIn("uav_movement_cost_weight", objective)
        self.assertEqual(float(objective["lambda_cache"]), 2.0e-5)
        self.assertAlmostEqual(float(objective["lambda_movement"]), 5.0e-6)
        self.assertEqual(
            config["objective"]["cost_normalization"]["contract_version"],
            "node_level_cost_upper_bounds_v1",
        )
        for role in ("sv", "rsu", "uav"):
            self.assertEqual(
                float(config["nodes"][role]["cache_backhaul_rate_bps"]),
                200.0e6,
            )

        cache_config = cache_configuration.cache_config_from_project(config)
        movement_config = uav_configuration.uav_search_config_from_project(config)
        service_vehicle_count = int(config["simulation"]["num_service_vehicles"])
        cache_nodes = {
            "node_ids": [
                *(f"SV_{index}" for index in range(service_vehicle_count)),
                *(f"RSU_{index}" for index in range(cache_config.num_rsu)),
                *(f"UAV_{index}" for index in range(cache_config.num_uav)),
            ],
            "node_roles": [
                *(["SV"] * service_vehicle_count),
                *(["RSU"] * cache_config.num_rsu),
                *(["UAV"] * cache_config.num_uav),
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
        reference = cache_configuration.initialize_cache_cost_normalization(
            cache_nodes,
            np.asarray([2000.0], dtype=np.float64),
            cache_config.lambda_cache,
        )
        np.testing.assert_allclose(
            reference["reference_s_by_node"],
            [480.0] * 5 + [1000.0] * 2 + [800.0] * 2,
        )
        self.assertAlmostEqual(float(reference["reference_s_total"]), 6000.0)
        self.assertAlmostEqual(float(reference["beta_cache"]), 0.12)
        self.assertEqual(
            movement_config.movement_energy_reference_j_by_uav,
            {"UAV_0": 225.0, "UAV_1": 225.0},
        )
        self.assertAlmostEqual(movement_config.movement_energy_reference_j, 450.0)
        self.assertAlmostEqual(movement_config.beta_movement, 0.00225)
        self.assertFalse(hasattr(cache_config, "sv_eta_ca"))
        self.assertFalse(hasattr(cache_config, "sv_xi_ca"))
        self.assertFalse(hasattr(movement_config, "movement_cost_weight"))

        missing_rate = deepcopy(config)
        del missing_rate["nodes"]["sv"]["cache_backhaul_rate_bps"]
        with self.assertRaisesRegex(
            ValueError, "缺少必需配置 nodes.sv.cache_backhaul_rate_bps"
        ):
            cache_configuration.cache_config_from_project(missing_rate)

    def test_deprecated_outer_cost_weights_are_rejected(self) -> None:
        config = deepcopy(self.config)
        config["objective"]["cache_cost_weight"] = 1.0
        with self.assertRaisesRegex(ValueError, "cache_cost_weight 已废弃"):
            project_config.validate_config(config, require_artifacts=False)

        config = deepcopy(self.config)
        config["objective"]["uav_movement_cost_weight"] = 1.0
        with self.assertRaisesRegex(ValueError, "uav_movement_cost_weight 已废弃"):
            project_config.validate_config(config, require_artifacts=False)

    def test_cache_cost_uses_each_nodes_reference_and_actual_additions(self) -> None:
        service_sizes = np.asarray([2000.0, 2000.0], dtype=np.float64)
        rates = np.asarray([200.0e6, 200.0e6], dtype=np.float64)
        cache_nodes = {
            "node_ids": ["SV_0", "RSU_0"],
            "node_roles": ["SV", "RSU"],
            "capacities_mb": np.asarray([2000.0, 2000.0], dtype=np.float64),
            "cache_backhaul_rate_bps": rates,
        }
        reference = cache_configuration.initialize_cache_cost_normalization(
            cache_nodes, service_sizes, 2.0e-5
        )
        cache_config = cache_configuration.cache_config_from_project(self.config)
        self.assertEqual(
            reference["reference_kind"],
            "per_node_conservative_capacity_upper_bound",
        )
        np.testing.assert_allclose(reference["reference_s_by_node"], [80.0, 80.0])
        self.assertAlmostEqual(float(reference["reference_s_total"]), 160.0)
        self.assertAlmostEqual(float(reference["beta_cache"]), 0.0032)

        cases = (
            ("keep", [[1, 0], [0, 0]], [[1, 0], [0, 0]], [0, 0], [0.0, 0.0]),
            ("replace", [[1, 0], [0, 0]], [[0, 1], [0, 0]], [1, 0], [80.0, 0.0]),
            ("delete", [[1, 0], [0, 0]], [[0, 0], [0, 0]], [1, 0], [0.0, 0.0]),
        )
        for label, old, new, trigger, expected_by_node in cases:
            with self.subTest(case=label):
                result = cache_configuration.cache_update_cost_details(
                    np.asarray(old, dtype=np.int8),
                    np.asarray(new, dtype=np.int8),
                    np.asarray(trigger, dtype=np.int8),
                    service_sizes,
                    cache_nodes,
                    cache_config,
                )
                np.testing.assert_allclose(
                    result["cache_download_delay_s_by_node"], expected_by_node
                )
                normalized = np.asarray(expected_by_node) / np.asarray(
                    reference["reference_s_by_node"]
                )
                np.testing.assert_allclose(
                    result["cache_download_delay_normalized_by_node"], normalized
                )
                self.assertAlmostEqual(
                    float(result["cache_download_delay_normalized"]),
                    float(np.sum(normalized)),
                )
                self.assertAlmostEqual(
                    float(result["cache_cost"]), float(np.sum(normalized)) * 0.0032
                )

        both_nodes_update = cache_configuration.cache_update_cost_details(
            np.asarray([[1, 0], [1, 0]], dtype=np.int8),
            np.asarray([[0, 1], [0, 1]], dtype=np.int8),
            np.asarray([1, 1], dtype=np.int8),
            service_sizes,
            cache_nodes,
            cache_config,
        )
        np.testing.assert_allclose(
            both_nodes_update["cache_download_delay_normalized_by_node"], [1.0, 1.0]
        )
        np.testing.assert_allclose(both_nodes_update["cache_cost_by_node"], [0.0032, 0.0032])
        self.assertAlmostEqual(both_nodes_update["cache_download_delay_normalized"], 2.0)
        self.assertAlmostEqual(both_nodes_update["cache_cost"], 0.0064)

    def test_zero_node_reference_rejects_positive_delay_without_division(self) -> None:
        rates = np.asarray([200.0e6], dtype=np.float64)
        reference = cache_configuration.initialize_cache_cost_normalization(
            {
                "node_ids": ["SV_0"],
                "node_roles": ["SV"],
                "capacities_mb": np.asarray([0.0], dtype=np.float64),
                "cache_backhaul_rate_bps": rates,
            },
            np.asarray([2000.0], dtype=np.float64),
            2.0e-5,
        )
        self.assertEqual(reference["reference_s_by_node"], [0.0])
        self.assertEqual(float(reference["beta_cache"]), 0.0)
        keep = system_model.cache_update_cost(
            np.asarray([[0]], dtype=np.int8),
            np.asarray([[0]], dtype=np.int8),
            np.asarray([0], dtype=np.int8),
            np.asarray([2000.0], dtype=np.float64),
            rates,
            np.asarray(reference["reference_s_by_node"]),
            reference["beta_cache"],
        )
        self.assertEqual(float(keep["cache_download_delay_normalized"]), 0.0)
        self.assertEqual(float(keep["cache_cost"]), 0.0)
        with self.assertRaisesRegex(ValueError, "zero node reference"):
            system_model.cache_update_cost(
                np.asarray([[0]], dtype=np.int8),
                np.asarray([[1]], dtype=np.int8),
                np.asarray([1], dtype=np.int8),
                np.asarray([2000.0], dtype=np.float64),
                rates,
                np.asarray(reference["reference_s_by_node"]),
                reference["beta_cache"],
            )

    def test_uav_cost_uses_per_uav_reference_and_actual_horizontal_movement(self) -> None:
        movement = uav_configuration.uav_search_config_from_project(self.config)
        self.assertEqual(movement.movement_energy_coefficient_j_per_m, 15.0)
        self.assertEqual(movement.movement_fixed_cost, 0.0)
        self.assertEqual(movement.lambda_movement, 5.0e-6)
        self.assertEqual(
            movement.movement_energy_reference_j_by_uav,
            {"UAV_0": 225.0, "UAV_1": 225.0},
        )

        stay = system_model.uav_movement_cost(
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            0,
            movement.dt_s,
            movement.max_horizontal_speed_mps,
            movement.movement_energy_coefficient_j_per_m,
            movement.movement_fixed_cost,
            movement.movement_energy_reference_j_by_uav["UAV_0"],
            movement.beta_movement,
            tolerance_m=movement.position_tolerance_m,
        )
        self.assertEqual(stay["actual_z_q"], 0)
        self.assertEqual(stay["movement_energy_j"], 0.0)
        self.assertEqual(stay["movement_cost"], 0.0)

        move = system_model.uav_movement_cost(
            np.asarray([0.0, 0.0]),
            np.asarray([5.0, 0.0]),
            1,
            movement.dt_s,
            movement.max_horizontal_speed_mps,
            movement.movement_energy_coefficient_j_per_m,
            movement.movement_fixed_cost,
            movement.movement_energy_reference_j_by_uav["UAV_0"],
            movement.beta_movement,
            tolerance_m=movement.position_tolerance_m,
        )
        self.assertEqual(move["actual_z_q"], 1)
        self.assertAlmostEqual(float(move["movement_energy_j"]), 75.0)
        self.assertAlmostEqual(float(move["movement_energy_normalized"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(move["movement_cost"]), 0.00075)

        joint = uav_configuration.movement_summary(
            {"UAV_0": np.asarray([0.0, 0.0]), "UAV_1": np.asarray([30.0, 30.0])},
            {"UAV_0": np.asarray([15.0, 0.0]), "UAV_1": np.asarray([30.0, 45.0])},
            {"UAV_0": 1, "UAV_1": 1},
            movement,
        )
        self.assertAlmostEqual(joint[0], 450.0)
        self.assertAlmostEqual(joint[1], 0.0045)
        self.assertAlmostEqual(joint[5], 2.0)

        zero_reference_stay = system_model.uav_movement_cost(
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            0,
            movement.dt_s,
            movement.max_horizontal_speed_mps,
            movement.movement_energy_coefficient_j_per_m,
            movement.movement_fixed_cost,
            0.0,
            0.0,
            tolerance_m=movement.position_tolerance_m,
        )
        self.assertEqual(zero_reference_stay["movement_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
