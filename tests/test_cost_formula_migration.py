from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from common import project_config, system_model
from decision import cache_configuration, uav_configuration
from ppo_simple import rollout


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
        self.assertEqual(float(objective["beta_ca"]), 1.0)
        self.assertAlmostEqual(float(objective["beta_q"]), 0.00225)
        # Kept only for existing analysis/report contracts; reward Cache cost
        # no longer reads or multiplies this legacy parameter.
        self.assertEqual(float(objective["lambda_cache"]), 2.0e-5)
        self.assertAlmostEqual(float(objective["lambda_movement"]), 5.0e-6)
        self.assertEqual(
            config["objective"]["cost_normalization"]["contract_version"],
            "node_level_cost_fixed_reference_explicit_weights_v2",
        )
        self.assertEqual(
            float(config["objective"]["cost_normalization"]["cache_reference_s_per_node"]),
            800.0,
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
            float(config["objective"]["cost_normalization"]["cache_reference_s_per_node"]),
        )
        np.testing.assert_allclose(
            reference["reference_s_by_node"],
            [800.0] * 9,
        )
        self.assertAlmostEqual(float(reference["reference_s_total"]), 7200.0)
        self.assertEqual(
            reference["reference_kind"],
            "fixed_per_node_design_normalization_scale",
        )
        np.testing.assert_allclose(
            reference["capacity_upper_bound_s_by_node"],
            [480.0] * 5 + [1000.0] * 2 + [800.0] * 2,
        )
        self.assertEqual(reference["additional_cost_multiplier"], "none")
        self.assertFalse(hasattr(cache_config, "lambda_cache"))
        self.assertEqual(
            movement_config.movement_energy_reference_j_by_uav,
            {"UAV_0": 225.0, "UAV_1": 225.0},
        )
        self.assertAlmostEqual(movement_config.movement_energy_reference_j, 450.0)
        self.assertFalse(hasattr(movement_config, "beta_movement"))
        self.assertFalse(hasattr(movement_config, "lambda_movement"))
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

    def test_cache_cost_uses_fixed_per_node_reference_and_actual_additions(self) -> None:
        service_sizes = np.asarray([2000.0, 2000.0], dtype=np.float64)
        rates = np.asarray([200.0e6, 200.0e6], dtype=np.float64)
        cache_nodes = {
            "node_ids": ["SV_0", "RSU_0"],
            "node_roles": ["SV", "RSU"],
            "capacities_mb": np.asarray([4000.0, 4000.0], dtype=np.float64),
            "cache_backhaul_rate_bps": rates,
        }
        reference = cache_configuration.initialize_cache_cost_normalization(
            cache_nodes, service_sizes, 800.0
        )
        cache_config = cache_configuration.cache_config_from_project(self.config)
        self.assertEqual(
            reference["reference_kind"],
            "fixed_per_node_design_normalization_scale",
        )
        np.testing.assert_allclose(reference["reference_s_by_node"], [800.0, 800.0])
        self.assertAlmostEqual(float(reference["reference_s_total"]), 1600.0)

        cases = (
            ("keep", [[1, 0], [0, 0]], [[1, 0], [0, 0]], [0, 0], [0.0, 0.0]),
            ("replace", [[1, 0], [0, 0]], [[0, 1], [0, 0]], [1, 0], [80.0, 0.0]),
            ("delete", [[1, 0], [0, 0]], [[0, 0], [0, 0]], [1, 0], [0.0, 0.0]),
            ("multiple_additions", [[0, 0], [0, 0]], [[1, 1], [0, 0]], [1, 0], [160.0, 0.0]),
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
                    float(result["cache_cost"]), float(np.sum(normalized))
                )
                np.testing.assert_allclose(result["cache_cost_by_node"], normalized)

        both_nodes_update = cache_configuration.cache_update_cost_details(
            np.asarray([[1, 0], [1, 0]], dtype=np.int8),
            np.asarray([[0, 1], [0, 1]], dtype=np.int8),
            np.asarray([1, 1], dtype=np.int8),
            service_sizes,
            cache_nodes,
            cache_config,
        )
        np.testing.assert_allclose(
            both_nodes_update["cache_download_delay_normalized_by_node"], [0.1, 0.1]
        )
        np.testing.assert_allclose(both_nodes_update["cache_cost_by_node"], [0.1, 0.1])
        self.assertAlmostEqual(both_nodes_update["cache_download_delay_normalized"], 0.2)
        self.assertAlmostEqual(both_nodes_update["cache_cost"], 0.2)

        # The 800 s denominator is a design scale, not a physical delay cap.
        large_node = {
            "node_ids": ["RSU_0"],
            "node_roles": ["RSU"],
            "capacities_mb": np.asarray([30000.0], dtype=np.float64),
            "cache_backhaul_rate_bps": np.asarray([200.0e6], dtype=np.float64),
        }
        large_reference = cache_configuration.initialize_cache_cost_normalization(
            large_node, np.asarray([22000.0], dtype=np.float64), 800.0
        )
        np.testing.assert_allclose(large_reference["reference_s_by_node"], [800.0])
        over_scale = cache_configuration.cache_update_cost_details(
            np.asarray([[0]], dtype=np.int8),
            np.asarray([[1]], dtype=np.int8),
            np.asarray([1], dtype=np.int8),
            np.asarray([22000.0], dtype=np.float64),
            large_node,
            cache_config,
        )
        self.assertGreater(float(over_scale["cache_download_delay_s"]), 800.0)
        self.assertAlmostEqual(float(over_scale["cache_cost"]), 1.1)

    def test_zero_node_reference_is_rejected(self) -> None:
        rates = np.asarray([200.0e6], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "必须全部为有限正数"):
            system_model.cache_update_cost(
                np.asarray([[0]], dtype=np.int8),
                np.asarray([[0]], dtype=np.int8),
                np.asarray([0], dtype=np.int8),
                np.asarray([2000.0], dtype=np.float64),
                rates,
                np.asarray([0.0], dtype=np.float64),
            )

        with self.assertRaisesRegex(ValueError, "容量必须为正"):
            cache_configuration.initialize_cache_cost_normalization(
                {
                    "node_ids": ["SV_0"],
                    "node_roles": ["SV"],
                    "capacities_mb": np.asarray([0.0], dtype=np.float64),
                    "cache_backhaul_rate_bps": rates,
                },
                np.asarray([2000.0], dtype=np.float64),
                800.0,
            )

    def test_uav_cost_uses_per_uav_reference_and_actual_horizontal_movement(self) -> None:
        movement = uav_configuration.uav_search_config_from_project(self.config)
        self.assertEqual(movement.movement_energy_coefficient_j_per_m, 15.0)
        self.assertEqual(movement.movement_fixed_cost, 0.0)
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
            tolerance_m=movement.position_tolerance_m,
        )
        self.assertEqual(move["actual_z_q"], 1)
        self.assertAlmostEqual(float(move["movement_energy_j"]), 75.0)
        self.assertAlmostEqual(float(move["movement_energy_normalized"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(move["movement_cost"]), 1.0 / 3.0)
        self.assertAlmostEqual(
            system_model.aggregate_system_utility(
                1.0, 0.0, float(move["movement_cost"]), 1.0, 0.00225
            ),
            1.0 - 0.00225 / 3.0,
        )

        joint = uav_configuration.movement_summary(
            {"UAV_0": np.asarray([0.0, 0.0]), "UAV_1": np.asarray([30.0, 30.0])},
            {"UAV_0": np.asarray([15.0, 0.0]), "UAV_1": np.asarray([30.0, 45.0])},
            {"UAV_0": 1, "UAV_1": 1},
            movement,
        )
        self.assertAlmostEqual(joint[0], 450.0)
        self.assertAlmostEqual(joint[1], 2.0)
        self.assertAlmostEqual(joint[5], 2.0)
        self.assertAlmostEqual(
            0.00225 * joint[1],
            0.0045,
        )

        zero_reference_stay = system_model.uav_movement_cost(
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            0,
            movement.dt_s,
            movement.max_horizontal_speed_mps,
            movement.movement_energy_coefficient_j_per_m,
            movement.movement_fixed_cost,
            0.0,
            tolerance_m=movement.position_tolerance_m,
        )
        self.assertEqual(zero_reference_stay["movement_cost"], 0.0)

    def test_explicit_weight_aggregation_and_training_reward_check(self) -> None:
        q = 0.73
        cache_cost = 0.4
        movement_cost = 0.8
        beta_ca = float(self.config["objective"]["beta_ca"])
        beta_q = float(self.config["objective"]["beta_q"])
        cases = (
            (0.0, 0.0, q),
            (cache_cost, 0.0, q - beta_ca * cache_cost),
            (0.0, movement_cost, q - beta_q * movement_cost),
            (
                cache_cost,
                movement_cost,
                q - beta_ca * cache_cost - beta_q * movement_cost,
            ),
        )
        for cache_value, movement_value, expected in cases:
            with self.subTest(cache=cache_value, movement=movement_value):
                actual = system_model.aggregate_system_utility(
                    q, cache_value, movement_value, beta_ca, beta_q
                )
                self.assertAlmostEqual(actual, expected)

        result = type(
            "TrainingRewardFixture",
            (),
            {
                "system_utility": cases[-1][2],
                "actual_task_utility": q,
                "cache_cost": cache_cost,
                "movement_cost": movement_cost,
                "beta_ca": beta_ca,
                "beta_q": beta_q,
            },
        )()
        self.assertAlmostEqual(rollout._check_reward(result), cases[-1][2])


if __name__ == "__main__":
    unittest.main()
