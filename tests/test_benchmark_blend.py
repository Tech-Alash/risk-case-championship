from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.models.benchmark import BenchmarkConfig, run_model_benchmark


class TestBenchmarkBlend(unittest.TestCase):
    def _build_frame(self, rows: int = 120) -> pd.DataFrame:
        rng = np.random.default_rng(17)
        is_claim = (rng.random(rows) < 0.20).astype(int)
        claim_amount = is_claim * rng.gamma(shape=2.0, scale=650.0, size=rows)
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

    def test_oof_blend_candidate_runs_and_writes_metadata(self) -> None:
        df = self._build_frame()
        train_df = df.iloc[:90].copy()
        valid_df = df.iloc[90:].copy()

        config = BenchmarkConfig.from_dict(
            {
                "enabled": True,
                "selection_metric": "policy_score",
                "candidates": ["oof_blend_freq_sev", "baseline_freq_sev"],
                "constraints": {
                    "max_violations": 1000,
                    "lr_total_min": 0.0,
                    "lr_total_max": 5.0,
                    "share_group1_min": 0.0,
                    "share_group1_max": 1.0,
                },
                "candidate_params": {
                    "oof_blend_freq_sev": {
                        "base_candidates": ["baseline_freq_sev"],
                        "oof_folds": 3,
                        "oof_group_column": "contract_number",
                        "weight_grid_step": 0.25,
                    }
                },
            }
        )

        result = run_model_benchmark(
            train_df=train_df,
            valid_df=valid_df,
            benchmark_config=config,
            pricing_target_lr=0.7,
            pricing_alpha_grid=np.linspace(-0.6, 2.0, 15),
            pricing_beta_grid=np.linspace(1.0, 1.0, 1),
            pricing_target_band=(0.69, 0.71),
            model_max_iter=120,
            model_ridge_alpha=1.0,
        )

        blend_result = next(item for item in result.results if item.candidate_name == "oof_blend_freq_sev")
        self.assertEqual(blend_result.status, "ok")
        self.assertIsNotNone(blend_result.metadata)
        metadata = blend_result.metadata or {}
        self.assertEqual(metadata.get("base_candidates"), ["baseline_freq_sev"])
        self.assertEqual(metadata.get("weights", {}).get("baseline_freq_sev"), 1.0)

        record = blend_result.to_record()
        self.assertEqual(record.get("blend_base_candidates"), "baseline_freq_sev")
        self.assertIn("baseline_freq_sev", record.get("blend_weights", ""))

    def test_oof_blend_checkpoint_is_written_and_can_resume(self) -> None:
        df = self._build_frame()
        train_df = df.iloc[:90].copy()
        valid_df = df.iloc[90:].copy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "oof_checkpoint"
            config = BenchmarkConfig.from_dict(
                {
                    "enabled": True,
                    "selection_metric": "policy_score",
                    "candidates": ["oof_blend_freq_sev"],
                    "constraints": {
                        "max_violations": 1000,
                        "lr_total_min": 0.0,
                        "lr_total_max": 5.0,
                        "share_group1_min": 0.0,
                        "share_group1_max": 1.0,
                    },
                    "candidate_params": {
                        "oof_blend_freq_sev": {
                            "base_candidates": ["baseline_freq_sev"],
                            "oof_folds": 3,
                            "oof_group_column": "contract_number",
                            "weight_grid_step": 0.25,
                            "checkpoint_dir": str(checkpoint_dir),
                        }
                    },
                }
            )

            first_result = run_model_benchmark(
                train_df=train_df,
                valid_df=valid_df,
                benchmark_config=config,
                pricing_target_lr=0.7,
                pricing_alpha_grid=np.linspace(-0.6, 2.0, 15),
                pricing_beta_grid=np.linspace(1.0, 1.0, 1),
                pricing_target_band=(0.69, 0.71),
                model_max_iter=120,
                model_ridge_alpha=1.0,
            )
            self.assertEqual(first_result.results[0].status, "ok")
            self.assertTrue((checkpoint_dir / "state.json").exists())
            self.assertTrue((checkpoint_dir / "oof_predictions.joblib").exists())

            resumed_config = BenchmarkConfig.from_dict(
                {
                    "enabled": True,
                    "selection_metric": "policy_score",
                    "candidates": ["oof_blend_freq_sev"],
                    "constraints": {
                        "max_violations": 1000,
                        "lr_total_min": 0.0,
                        "lr_total_max": 5.0,
                        "share_group1_min": 0.0,
                        "share_group1_max": 1.0,
                    },
                    "candidate_params": {
                        "oof_blend_freq_sev": {
                            "base_candidates": ["baseline_freq_sev"],
                            "oof_folds": 3,
                            "oof_group_column": "contract_number",
                            "weight_grid_step": 0.25,
                            "resume_from_checkpoint": str(checkpoint_dir),
                        }
                    },
                }
            )

            resumed_result = run_model_benchmark(
                train_df=train_df,
                valid_df=valid_df,
                benchmark_config=resumed_config,
                pricing_target_lr=0.7,
                pricing_alpha_grid=np.linspace(-0.6, 2.0, 15),
                pricing_beta_grid=np.linspace(1.0, 1.0, 1),
                pricing_target_band=(0.69, 0.71),
                model_max_iter=120,
                model_ridge_alpha=1.0,
            )
            resumed_metadata = resumed_result.results[0].metadata or {}
            self.assertEqual(resumed_result.results[0].status, "ok")
            self.assertEqual(resumed_metadata.get("checkpoint_dir"), str(checkpoint_dir.resolve()))


if __name__ == "__main__":
    unittest.main()
