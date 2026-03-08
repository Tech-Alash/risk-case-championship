from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "tune_catboost.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("tune_catboost_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load tune_catboost.py module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tune_catboost_module"] = module
    spec.loader.exec_module(module)
    return module


class _DummyTrial:
    def suggest_int(self, name: str, low: int, high: int) -> int:
        return int(low)

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        return float(low)


class TestTuneCatboostModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_module()

    def test_suggest_params_full_mode_contains_frequency_and_severity(self) -> None:
        params = self.mod._suggest_catboost_params(_DummyTrial(), phase="fine", mode="full")
        self.assertIn("iterations", params)
        self.assertIn("learning_rate", params)
        self.assertIn("depth", params)
        self.assertIn("reg_iterations", params)
        self.assertIn("reg_learning_rate", params)

    def test_suggest_params_severity_only_mode_contains_only_severity_space(self) -> None:
        params = self.mod._suggest_catboost_params(_DummyTrial(), phase="fine", mode="severity_only")
        self.assertIn("reg_iterations", params)
        self.assertIn("reg_learning_rate", params)
        self.assertIn("reg_depth", params)
        self.assertIn("tweedie_variance_power", params)
        self.assertEqual(params.get("severity_loss_function"), "TWEEDIE")
        self.assertNotIn("iterations", params)
        self.assertNotIn("learning_rate", params)
        self.assertNotIn("depth", params)

    def test_parse_args_supports_dependent_candidate(self) -> None:
        old_argv = list(sys.argv)
        try:
            sys.argv = [str(SCRIPT_PATH), "--candidate", "catboost_dep_freq_sev", "--n_trials", "1"]
            args = self.mod.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.candidate, "catboost_dep_freq_sev")
        self.assertEqual(args.n_trials, 1)


if __name__ == "__main__":
    unittest.main()
