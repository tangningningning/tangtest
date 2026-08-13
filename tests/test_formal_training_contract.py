"""正式论文训练入口的静态/轻量回归测试；任何测试都不得调用 model.learn。"""

from __future__ import annotations

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import train_ppo
from data import load_config
from environment import MECEnvironment, NodeType, cache_update_cost, task_utility
from mobility import TraceMobilityProvider
from solver import (
    predicted_cache_utility,
    solve_cooperative_cache,
    transfer_exchange_matching,
)


class FormalConfigurationTests(unittest.TestCase):
    def test_frozen_formal_dimensions(self) -> None:
        result = train_ppo.validate_formal_config(train_ppo.prepare_ppo_config())
        self.assertEqual(result["services"], 10)
        self.assertEqual(result["action_dimension"], 8)
        self.assertEqual(result["observation_dimension"], 95)
        self.assertEqual(result["parameter_drift"], 0)

    def test_gamma_matches_finite_horizon_p0_and_vecnormalize(self) -> None:
        config = train_ppo.prepare_ppo_config()
        vector = train_ppo.build_vecnormalize_environment(
            config,
            episode_seed_base=int(config["ppo"]["train_seed_base"]),
        )
        try:
            model = train_ppo.build_sb3_ppo(vector, config)
            self.assertEqual(float(config["ppo"]["gamma"]), 1.0)
            self.assertEqual(float(model.gamma), 1.0)
            self.assertEqual(float(vector.gamma), 1.0)
            self.assertEqual(float(model.gae_lambda), 0.95)
        finally:
            vector.close()

    def test_service_count_drift_is_rejected(self) -> None:
        config = copy.deepcopy(train_ppo.prepare_ppo_config())
        config["services"]["count"] = 20
        with self.assertRaisesRegex(ValueError, "formal configuration drift"):
            train_ppo.validate_formal_config(config)

    def test_all_causal_and_identifier_diagnostics_are_registered(self) -> None:
        required = {
            "future_request_leakage",
            "future_trajectory_leakage",
            "future_mobility_leakage",
            "invalid_service_ids",
            "invalid_node_ids",
        }
        self.assertTrue(required.issubset(train_ppo.LEARNING_CHECK_HARD_KEYS))


class CausalTraceTests(unittest.TestCase):
    def test_future_slot_rejection_is_counted_and_reset(self) -> None:
        config = load_config()
        provider = TraceMobilityProvider(config, seed=20260811)
        try:
            provider.reset(provider._times)
            with self.assertRaisesRegex(RuntimeError, "must start at slot 0"):
                provider.states_at(1)
            self.assertEqual(provider.diagnostics()["future_access_count"], 1)

            provider.reset(provider._times)
            provider.states_at(0)
            provider.states_at(1)
            with self.assertRaisesRegex(RuntimeError, "cannot skip a slot"):
                provider.states_at(3)
            self.assertEqual(provider.diagnostics()["future_access_count"], 1)
        finally:
            provider.close()


class FormalPipelineGuardTests(unittest.TestCase):
    def test_training_preflight_does_not_parse_test_seeds(self) -> None:
        config = train_ppo.prepare_ppo_config()
        with mock.patch.object(
            train_ppo,
            "_formal_test_seeds",
            side_effect=AssertionError("test seeds accessed before best checkpoint"),
        ):
            result = train_ppo.preflight_training_traces(
                config,
                total_timesteps=256,
            )
        self.assertTrue(result["passed"])
        self.assertTrue(result["test_deferred_until_best"])
        self.assertEqual(result["test_traces"], 0)

    def test_configured_diagnostic_triggers_hard_stop(self) -> None:
        config = train_ppo.prepare_ppo_config()
        with self.assertRaisesRegex(RuntimeError, "future_mobility_leakage"):
            train_ppo._raise_on_hard_stop(
                {"future_mobility_leakage": 1},
                config,
                phase="unit test",
            )

    def test_missing_artifact_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            present = root / "present.txt"
            present.write_text("ok", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing=.*best_model"):
                train_ppo._validate_nonempty_artifacts(
                    root,
                    {
                        "present": present,
                        "best_model": root / "best_model.zip",
                    },
                )

    def test_cli_dispatch_never_uses_a_second_training_implementation(self) -> None:
        with mock.patch.object(train_ppo, "run_training") as formal:
            train_ppo.main([])
            formal.assert_called_once_with(total_timesteps=None, output_dir=None)
        with mock.patch.object(train_ppo, "run_training") as formal:
            train_ppo.main(["--train"])
            formal.assert_called_once_with(total_timesteps=None, output_dir=None)

        with mock.patch.object(train_ppo, "run_training") as formal:
            with redirect_stdout(io.StringIO()):
                train_ppo.main(["--config-check"])
            formal.assert_not_called()


class MathematicalConsistencyTests(unittest.TestCase):
    def test_task_utility_scalar_and_vector_references_are_equivalent(self) -> None:
        arguments = {
            "arrived": np.asarray([True, True]),
            "completion_delay_s": np.asarray([0.5, 1.0]),
            "vehicle_energy_j": np.asarray([0.05, 0.1]),
            "delay_weight": 0.5,
            "energy_weight": 0.5,
        }
        scalar = task_utility(
            **arguments,
            delay_reference_s=2.0,
            energy_reference_j=0.2,
        )
        vector = task_utility(
            **arguments,
            delay_reference_s=np.asarray([2.0, 2.0]),
            energy_reference_j=np.asarray([0.2, 0.2]),
        )
        np.testing.assert_allclose(
            scalar["per_task_utility"], vector["per_task_utility"]
        )
        self.assertAlmostEqual(
            scalar["total_task_utility"], vector["total_task_utility"]
        )

    def test_unchanged_cache_has_no_update_cost(self) -> None:
        cache = np.asarray([1.0, 0.0])
        result = cache_update_cost(
            cache,
            cache,
            np.asarray([8.0e6, 8.0e6]),
            fixed_cost_per_update=1.0,
            loading_energy_per_bit=1.0e-9,
            loading_cost_weight=1.0,
        )
        self.assertFalse(result["effective_trigger"])
        self.assertEqual(result["total_cost"], 0.0)

    def test_trigger_can_be_net_benefit_noop(self) -> None:
        config = load_config()
        config["cache"]["fixed_update_cost"] = {
            "service_vehicle": 10.0,
            "rsu": 10.0,
            "uav": 10.0,
        }
        environment = MECEnvironment(config)
        previous = np.zeros((1, 10), dtype=float)
        previous[0, 0] = 1.0
        demand = np.zeros((1, 10), dtype=float)
        demand[0, 1] = 1.0
        result = solve_cooperative_cache(
            previous,
            np.asarray([True]),
            np.ones(10, dtype=float),
            np.asarray([1.0]),
            cache_prediction={"predicted_demand": demand},
            node_types=(NodeType.RSU,),
            environment=environment,
            task_vehicle_cache=np.zeros((1, 10), dtype=float),
            task_vehicle_ids=("v0",),
            integer_scale=1,
        )
        np.testing.assert_array_equal(result["candidate_cache"], previous)
        self.assertFalse(bool(result["actual_cache_update"][0]))
        self.assertEqual(result["predicted_cache_update_cost"], 0.0)

    def test_local_cache_reduces_edge_marginal_to_zero(self) -> None:
        class StubEnvironment:
            config = {
                "workload": {
                    "task_input_size_min": 1.0,
                    "task_input_size_max": 1.0,
                    "task_input_size_unit": "bit",
                    "task_cpu_cycles_min": 1.0,
                    "task_cpu_cycles_max": 1.0,
                    "task_cpu_cycles_unit": "cycle",
                }
            }

            @staticmethod
            def execute_task(kind: NodeType, **_kwargs: object) -> dict[str, float]:
                delay = {
                    NodeType.LOCAL: 1.0,
                    NodeType.RSU: 2.0,
                    NodeType.CLOUD: 3.0,
                }[kind]
                return {"completion_delay_s": delay, "vehicle_energy_j": 0.0}

            @staticmethod
            def task_utility(arrived: np.ndarray, delays: np.ndarray, energies: np.ndarray) -> dict[str, object]:
                return task_utility(
                    arrived,
                    delays,
                    energies,
                    delay_weight=1.0,
                    energy_weight=0.0,
                    delay_reference_s=10.0,
                    energy_reference_j=1.0,
                )

            @staticmethod
            def ground_rate(*_args: object, **_kwargs: object) -> float:
                return 1.0

        prediction = {
            "predicted_coverage": np.ones((1, 1, 1)),
            "vehicle_ids": ("v0",),
            "vehicle_preferences": {"v0": np.asarray([1.0])},
            "horizon_weights": np.asarray([1.0]),
            "node_positions_m": np.zeros((1, 1, 2)),
            "vehicle_positions_m": np.zeros((1, 1, 2)),
        }
        value = predicted_cache_utility(
            np.asarray([[1.0]]),
            cache_prediction=prediction,
            node_types=(NodeType.RSU,),
            environment=StubEnvironment(),
            task_vehicle_cache=np.asarray([[1.0]]),
            task_vehicle_ids=("v0",),
        )
        self.assertEqual(value, 0.0)

    def test_matching_final_q_uses_kkt_reference_vectors(self) -> None:
        environment = MECEnvironment(load_config())
        delay_refs = np.asarray([7.0])
        energy_refs = np.asarray([3.0])
        result = transfer_exchange_matching(
            ((2,),),
            environment=environment,
            task_arrivals=np.asarray([True]),
            task_input_bits=np.asarray([1.0e6]),
            task_cpu_cycles=np.asarray([1.0e9]),
            task_vehicle_positions=np.asarray([[0.0, 0.0]]),
            service_node_positions=np.asarray([[1.0, 1.0]]),
            service_node_types=(NodeType.RSU,),
            local_columns=(0,),
            service_columns=(1,),
            cloud_column=2,
            base_rates_bps=np.asarray([[1.0e6]]),
            node_frequencies_hz=np.asarray([1.0e9]),
            transmit_power_w=np.asarray([0.1]),
            reference_delays_s=delay_refs,
            reference_energies_j=energy_refs,
            omega_time=0.25,
            omega_energy=0.75,
            epsilon_compute=0.01,
            epsilon_bandwidth=0.01,
        )
        task_result = result["task_results"][0]
        expected = environment.task_utility(
            np.asarray([True]),
            np.asarray([task_result["completion_delay_s"]]),
            np.asarray([task_result["vehicle_energy_j"]]),
            delay_reference_s=delay_refs,
            energy_reference_j=energy_refs,
            delay_weight=0.25,
            energy_weight=0.75,
        )
        self.assertAlmostEqual(
            result["total_task_utility"], expected["total_task_utility"]
        )
        self.assertEqual(
            result["candidate_diagnostics"]["candidate_evaluation_count"], 1
        )
        self.assertEqual(result["kkt_failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
