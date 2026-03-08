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

from risk_case.models.benchmark import BenchmarkConfig, run_model_benchmark


class TestBenchmarkFailover(unittest.TestCase):
    def _build_frame(self, rows: int = 120) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        is_claim = (rng.random(rows) < 0.20).astype(int)
        claim_amount = is_claim * rng.gamma(shape=2.0, scale=600.0, size=rows)

        return pd.DataFrame(
            {
                "contract_number": [f"c_{i}" for i in range(rows)],
                "premium": rng.uniform(2500.0, 15000.0, rows),
                "premium_wo_term": rng.uniform(2000.0, 14000.0, rows),
                "is_claim": is_claim,
                "claim_amount": claim_amount,
                "claim_cnt": is_claim,
                "region_name": rng.choice(["01", "02", "03"], rows),
                "vehicle_type_name": rng.choice(["sedan", "truck"], rows),
                "engine_power_mean": rng.uniform(70.0, 280.0, rows),
            }
        )

    def test_failed_candidate_does_not_break_benchmark(self) -> None:
        df = self._build_frame()
        train_df = df.iloc[:90].copy()
        valid_df = df.iloc[90:].copy()

        config = BenchmarkConfig.from_dict(
            {
                "enabled": True,
                "candidates": ["missing_model", "baseline_freq_sev"],
                "fallback_candidate": "baseline_freq_sev",
                "constraints": {
                    "max_violations": 1000,
                    "lr_total_min": 0.0,
                    "lr_total_max": 5.0,
                    "share_group1_min": 0.0,
                    "share_group1_max": 1.0,
                },
            }
        )

        result = run_model_benchmark(
            train_df=train_df,
            valid_df=valid_df,
            benchmark_config=config,
            pricing_target_lr=0.7,
            pricing_alpha_grid=np.linspace(-0.6, 2.0, 20),
            pricing_beta_grid=np.linspace(1.0, 1.0, 1),
            pricing_target_band=(0.69, 0.71),
            model_max_iter=120,
            model_ridge_alpha=1.0,
        )

        failed = [item for item in result.results if item.status == "failed"]
        ok = [item for item in result.results if item.status == "ok"]
        self.assertEqual(result.winner_name, "baseline_freq_sev")
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(ok), 1)

    def test_all_failed_candidates_raise(self) -> None:
        df = self._build_frame()
        train_df = df.iloc[:90].copy()
        valid_df = df.iloc[90:].copy()

        config = BenchmarkConfig.from_dict(
            {
                "enabled": True,
                "candidates": ["missing_a", "missing_b"],
                "fallback_candidate": "baseline_freq_sev",
            }
        )

        with self.assertRaises(ValueError):
            run_model_benchmark(
                train_df=train_df,
                valid_df=valid_df,
                benchmark_config=config,
                pricing_target_lr=0.7,
                pricing_alpha_grid=np.linspace(-0.6, 2.0, 10),
                pricing_beta_grid=np.linspace(1.0, 1.0, 1),
                pricing_target_band=(0.69, 0.71),
                model_max_iter=120,
                model_ridge_alpha=1.0,
            )


if __name__ == "__main__":
    unittest.main()
