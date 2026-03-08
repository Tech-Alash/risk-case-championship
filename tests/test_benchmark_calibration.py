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

from risk_case.models.benchmark import BenchmarkConfig, CalibratedFrequencySeverityModel, run_model_benchmark


class TestBenchmarkCalibration(unittest.TestCase):
    def _build_frame(self, rows: int = 220) -> pd.DataFrame:
        rng = np.random.default_rng(23)
        engine_power = rng.uniform(60.0, 320.0, rows)
        premium = rng.uniform(2500.0, 16000.0, rows)
        score = -2.0 + 0.008 * engine_power + 0.00006 * premium
        p_claim = 1.0 / (1.0 + np.exp(-score))
        is_claim = (rng.random(rows) < p_claim).astype(int)
        claim_amount = is_claim * rng.gamma(shape=2.0, scale=620.0, size=rows)
        return pd.DataFrame(
            {
                "contract_number": [f"c_{i}" for i in range(rows)],
                "premium": premium,
                "premium_wo_term": premium * rng.uniform(0.75, 0.95, rows),
                "is_claim": is_claim,
                "claim_amount": claim_amount,
                "claim_cnt": is_claim,
                "region_name": rng.choice(["01", "02", "03"], rows),
                "vehicle_type_name": rng.choice(["sedan", "truck"], rows),
                "engine_power_mean": engine_power,
            }
        )

    def test_probability_calibration_applies_and_updates_metadata(self) -> None:
        df = self._build_frame()
        train_df = df.iloc[:170].copy()
        valid_df = df.iloc[170:].copy()
        config = BenchmarkConfig.from_dict(
            {
                "enabled": True,
                "candidates": ["baseline_freq_sev"],
                "constraints": {
                    "max_violations": 1000,
                    "lr_total_min": 0.0,
                    "lr_total_max": 5.0,
                    "share_group1_min": 0.0,
                    "share_group1_max": 1.0,
                },
                "calibration": {
                    "enabled": True,
                    "method": "platt",
                    "oof_folds": 3,
                    "group_column": "contract_number",
                    "min_samples": 50,
                },
            }
        )

        result = run_model_benchmark(
            train_df=train_df,
            valid_df=valid_df,
            benchmark_config=config,
            pricing_target_lr=0.7,
            pricing_alpha_grid=np.linspace(-0.6, 2.0, 12),
            pricing_beta_grid=np.linspace(1.0, 1.0, 1),
            pricing_target_band=(0.69, 0.71),
            model_max_iter=120,
            model_ridge_alpha=1.0,
        )

        self.assertEqual(result.winner_name, "baseline_freq_sev")
        self.assertIsInstance(result.winner_model, CalibratedFrequencySeverityModel)
        candidate = result.results[0]
        self.assertEqual(candidate.status, "ok")
        self.assertIsNotNone(candidate.metadata)
        metadata = candidate.metadata or {}
        self.assertEqual(metadata.get("calibration_status"), "applied")
        self.assertEqual(metadata.get("calibration_method"), "platt")
        self.assertIsNotNone(metadata.get("calibration_oof_brier_raw"))
        self.assertIsNotNone(metadata.get("calibration_oof_brier_calibrated"))
        record = candidate.to_record()
        self.assertEqual(record.get("calibration_status"), "applied")
        self.assertEqual(record.get("calibration_method"), "platt")


if __name__ == "__main__":
    unittest.main()
