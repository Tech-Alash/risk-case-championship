from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.models.benchmark import (
    CatBoostFrequencySeverityModel,
    DependentCatBoostFrequencySeverityModel,
    _build_candidate_model,
)


class TestCatBoostCandidateParams(unittest.TestCase):
    def test_catboost_candidate_uses_extended_params(self) -> None:
        model = _build_candidate_model(
            candidate_name="catboost_freq_sev",
            model_max_iter=120,
            model_ridge_alpha=1.0,
            random_state=42,
            candidate_params={
                "catboost_freq_sev": {
                    "iterations": 333,
                    "learning_rate": 0.03,
                    "depth": 8,
                    "l2_leaf_reg": 7.5,
                    "random_strength": 1.3,
                    "bagging_temperature": 0.7,
                    "border_count": 128,
                    "reg_iterations": 500,
                    "reg_learning_rate": 0.02,
                    "reg_depth": 5,
                    "reg_l2_leaf_reg": 9.0,
                    "reg_random_strength": 0.6,
                    "reg_bagging_temperature": 1.2,
                    "reg_border_count": 96,
                    "severity_loss_function": "TWEEDIE",
                    "tweedie_variance_power": 1.35,
                }
            },
        )

        self.assertIsInstance(model, CatBoostFrequencySeverityModel)
        self.assertEqual(model.iterations, 333)
        self.assertAlmostEqual(model.learning_rate, 0.03)
        self.assertEqual(model.depth, 8)
        self.assertAlmostEqual(model.l2_leaf_reg, 7.5)
        self.assertAlmostEqual(model.random_strength, 1.3)
        self.assertAlmostEqual(model.bagging_temperature, 0.7)
        self.assertEqual(model.border_count, 128)
        self.assertEqual(model.reg_iterations, 500)
        self.assertAlmostEqual(model.reg_learning_rate, 0.02)
        self.assertEqual(model.reg_depth, 5)
        self.assertAlmostEqual(model.reg_l2_leaf_reg, 9.0)
        self.assertAlmostEqual(model.reg_random_strength, 0.6)
        self.assertAlmostEqual(model.reg_bagging_temperature, 1.2)
        self.assertEqual(model.reg_border_count, 96)
        self.assertEqual(model.severity_loss_function, "TWEEDIE")
        self.assertAlmostEqual(model.tweedie_variance_power, 1.35)

    def test_catboost_candidate_defaults_reg_params_to_none(self) -> None:
        model = _build_candidate_model(
            candidate_name="catboost_freq_sev",
            model_max_iter=120,
            model_ridge_alpha=1.0,
            random_state=7,
            candidate_params={},
        )

        self.assertIsInstance(model, CatBoostFrequencySeverityModel)
        self.assertIsNone(model.reg_iterations)
        self.assertIsNone(model.reg_learning_rate)
        self.assertIsNone(model.reg_depth)
        self.assertIsNone(model.reg_l2_leaf_reg)
        self.assertIsNone(model.reg_random_strength)
        self.assertIsNone(model.reg_bagging_temperature)
        self.assertIsNone(model.reg_border_count)
        self.assertEqual(model.severity_loss_function, "RMSE")
        self.assertAlmostEqual(model.tweedie_variance_power, 1.5)

    def test_dependent_catboost_candidate_uses_dependent_params(self) -> None:
        model = _build_candidate_model(
            candidate_name="catboost_dep_freq_sev",
            model_max_iter=120,
            model_ridge_alpha=1.0,
            random_state=11,
            candidate_params={
                "catboost_dep_freq_sev": {
                    "iterations": 280,
                    "reg_iterations": 420,
                    "dep_oof_folds": 7,
                    "dep_frequency_signal_name": "my_freq_signal",
                    "dep_use_frequency_signal": True,
                }
            },
        )

        self.assertIsInstance(model, DependentCatBoostFrequencySeverityModel)
        self.assertEqual(model.iterations, 280)
        self.assertEqual(model.reg_iterations, 420)
        self.assertEqual(model.dep_oof_folds, 7)
        self.assertEqual(model.dep_frequency_signal_name, "my_freq_signal")
        self.assertTrue(model.dep_use_frequency_signal)

    def test_dependent_catboost_candidate_defaults_extra_params(self) -> None:
        model = _build_candidate_model(
            candidate_name="catboost_dep_freq_sev",
            model_max_iter=120,
            model_ridge_alpha=1.0,
            random_state=13,
            candidate_params={},
        )

        self.assertIsInstance(model, DependentCatBoostFrequencySeverityModel)
        self.assertEqual(model.dep_oof_folds, 5)
        self.assertEqual(model.dep_frequency_signal_name, "freq_risk_signal")
        self.assertTrue(model.dep_use_frequency_signal)

    def test_dependent_catboost_candidate_fit_predict_contract_when_catboost_available(self) -> None:
        try:
            import catboost  # noqa: F401
        except Exception:
            self.skipTest("catboost is not available")

        rng = np.random.default_rng(7)
        rows = 80
        is_claim = np.zeros(rows, dtype=int)
        is_claim[:24] = 1
        claim_amount = np.where(is_claim == 1, rng.gamma(shape=2.2, scale=800.0, size=rows), 0.0)
        df = pd.DataFrame(
            {
                "contract_number": [f"contract_{i // 2}" for i in range(rows)],
                "premium": rng.uniform(2000.0, 12000.0, rows),
                "premium_wo_term": rng.uniform(1800.0, 11000.0, rows),
                "engine_power": rng.uniform(70.0, 220.0, rows),
                "region_name": rng.choice(["01", "02", "03"], rows),
                "vehicle_type_name": rng.choice(["sedan", "truck"], rows),
                "is_claim": is_claim,
                "claim_amount": claim_amount,
                "claim_cnt": is_claim,
            }
        )

        model = _build_candidate_model(
            candidate_name="catboost_dep_freq_sev",
            model_max_iter=120,
            model_ridge_alpha=1.0,
            random_state=17,
            candidate_params={
                "catboost_dep_freq_sev": {
                    "iterations": 20,
                    "reg_iterations": 20,
                    "depth": 4,
                    "reg_depth": 4,
                    "thread_count": 1,
                    "dep_oof_folds": 4,
                }
            },
        )
        pred = model.fit(df).predict(df.head(12))

        self.assertEqual(list(pred.columns), ["p_claim", "expected_severity", "expected_loss"])
        self.assertEqual(len(pred), 12)
        self.assertIsNotNone(model.train_oof_p_claim_)
        self.assertEqual(len(model.train_oof_p_claim_), rows)
        self.assertIsNotNone(model.severity_schema)
        assert model.severity_schema is not None
        self.assertIn("freq_risk_signal", model.severity_schema.numeric_cols)


if __name__ == "__main__":
    unittest.main()
