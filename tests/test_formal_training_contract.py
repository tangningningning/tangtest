"""正式论文训练入口的静态/轻量回归测试；任何测试都不得调用 model.learn。"""

from __future__ import annotations

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import train_ppo
from data import load_config
from mobility import TraceMobilityProvider


class FormalConfigurationTests(unittest.TestCase):
    def test_frozen_formal_dimensions(self) -> None:
        result = train_ppo.validate_formal_config(train_ppo.prepare_ppo_config())
        self.assertEqual(result["services"], 10)
        self.assertEqual(result["action_dimension"], 8)
        self.assertEqual(result["observation_dimension"], 95)
        self.assertEqual(result["parameter_drift"], 0)

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


if __name__ == "__main__":
    unittest.main()
