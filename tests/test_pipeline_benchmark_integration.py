from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.orchestration.run_pipeline import run_experiment


class TestPipelineBenchmarkIntegration(unittest.TestCase):
    def test_pipeline_writes_benchmark_artifacts(self) -> None:
        rng = np.random.default_rng(123)
        rows = 180
        is_claim = (rng.random(rows) < 0.18).astype(int)
        claim_amount = is_claim * rng.gamma(shape=2.0, scale=700.0, size=rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_dir = tmp / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            train_df = pd.DataFrame(
                {
                    "unique_id": [f"id_{i}" for i in range(rows)],
                    "contract_number": [f"contract_{i // 2}" for i in range(rows)],
                    "premium": rng.uniform(3000.0, 18000.0, rows),
                    "premium_wo_term": rng.uniform(2500.0, 17000.0, rows),
                    "is_claim": is_claim,
                    "claim_amount": claim_amount,
                    "claim_cnt": is_claim,
                    "region_name": rng.choice(["01", "02", "03"], rows),
                    "vehicle_type_name": rng.choice(["sedan", "truck"], rows),
                    "engine_power": rng.uniform(55.0, 260.0, rows),
                }
            )
            test_df = train_df.drop(columns=["is_claim", "claim_amount", "claim_cnt"]).head(30).copy()

            train_path = data_dir / "train.csv"
            test_path = data_dir / "test.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            config = {
                "paths": {
                    "train_csv": str(train_path),
                    "test_csv": str(test_path),
                    "artifacts_dir": str(tmp / "artifacts"),
                },
                "split": {"test_size": 0.25, "random_state": 42},
                "model": {"max_iter": 150, "ridge_alpha": 1.0},
                "benchmark": {
                    "enabled": True,
                    "candidates": ["baseline_freq_sev"],
                    "selection_metric": "policy_score",
                    "fallback_strategy": "best_metric",
                    "constraints": {
                        "max_violations": 1000,
                        "lr_total_min": 0.0,
                        "lr_total_max": 5.0,
                        "share_group1_min": 0.0,
                        "share_group1_max": 1.0,
                    },
                    "fallback_candidate": "baseline_freq_sev",
                    "random_state": 42,
                },
                "pricing": {
                    "target_lr": 0.7,
                    "alpha_grid": {"start": -0.6, "stop": 1.8, "num": 30},
                    "beta_grid": {"start": 1.0, "stop": 1.0, "num": 1},
                    "target_band": {"min": 0.69, "max": 0.71},
                    "optimization": {
                        "method": "slsqp",
                        "slsqp": {"maxiter": 40, "ftol": 1e-6, "eps": 1e-3}
                    },
                    "retention": {
                        "enabled": True,
                        "base_retention": 0.9,
                        "elasticity": 4.0,
                        "center": 0.0,
                        "floor": 0.05,
                        "cap": 0.99
                    }
                },
                "diagnostics": {"enabled": True, "deciles": 5},
            }
            config_dir = tmp / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "default.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_experiment(config_path)
            run_dir = Path(result["run_dir"])

            self.assertEqual(result["status"], "ok")
            self.assertTrue((run_dir / "benchmark" / "results.csv").exists())
            self.assertTrue((run_dir / "benchmark" / "results.json").exists())
            self.assertTrue((run_dir / "benchmark" / "winner.json").exists())
            self.assertTrue((run_dir / "benchmark" / "failed_candidates.json").exists())
            self.assertTrue((run_dir / "pricing_policy.json").exists())
            self.assertTrue((run_dir / "ablation_results.csv").exists())
            self.assertTrue((run_dir / "double_lift_table.csv").exists())
            self.assertTrue((run_dir / "ae_by_risk_decile.csv").exists())
            self.assertTrue((run_dir / "ae_by_segment.csv").exists())
            self.assertTrue(result["metrics"]["benchmark"]["enabled"])
            self.assertEqual(result["metrics"]["benchmark"]["winner_name"], "baseline_freq_sev")
            self.assertIn("beta", result["metrics"]["pricing"])
            self.assertIn("in_target", result["metrics"]["pricing"])
            self.assertEqual(result["metrics"]["pricing"]["pricing_policy_kind"], "scalar")
            self.assertEqual(result["metrics"]["pricing"]["pricing_policy_path"], str(run_dir / "pricing_policy.json"))
            self.assertIn("ablation", result["metrics"])
            self.assertIn("diagnostics", result["metrics"])
            self.assertEqual(result["metrics"]["pricing"]["optimization_method"], "slsqp")

            latest_pointer = json.loads((tmp / "artifacts" / "latest_run.json").read_text(encoding="utf-8"))
            self.assertEqual(latest_pointer["pricing_policy_path"], str(run_dir / "pricing_policy.json"))

    def test_pipeline_supports_stratified_pricing_policy(self) -> None:
        rng = np.random.default_rng(321)
        rows = 220
        is_claim = (rng.random(rows) < 0.22).astype(int)
        claim_amount = is_claim * rng.gamma(shape=2.0, scale=900.0, size=rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_dir = tmp / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            train_df = pd.DataFrame(
                {
                    "unique_id": [f"id_{i}" for i in range(rows)],
                    "contract_number": [f"contract_{i // 2}" for i in range(rows)],
                    "premium": rng.uniform(3000.0, 20000.0, rows),
                    "premium_wo_term": rng.uniform(2500.0, 19000.0, rows),
                    "is_claim": is_claim,
                    "claim_amount": claim_amount,
                    "claim_cnt": is_claim,
                    "region_name": rng.choice(["01", "02", "03", "04"], rows),
                    "vehicle_type_name": rng.choice(["sedan", "truck", "suv"], rows),
                    "engine_power": rng.uniform(55.0, 300.0, rows),
                }
            )
            train_path = data_dir / "train.csv"
            train_df.to_csv(train_path, index=False)

            config = {
                "paths": {
                    "train_csv": str(train_path),
                    "test_csv": None,
                    "artifacts_dir": str(tmp / "artifacts"),
                },
                "split": {"test_size": 0.25, "random_state": 42},
                "model": {"max_iter": 150, "ridge_alpha": 1.0},
                "benchmark": {
                    "enabled": True,
                    "candidates": ["baseline_freq_sev"],
                    "selection_metric": "policy_score",
                    "fallback_strategy": "best_metric",
                    "constraints": {
                        "max_violations": 1000,
                        "lr_total_min": 0.0,
                        "lr_total_max": 5.0,
                        "share_group1_min": 0.0,
                        "share_group1_max": 1.0,
                    },
                    "fallback_candidate": "baseline_freq_sev",
                    "random_state": 42,
                },
                "pricing": {
                    "target_lr": 0.7,
                    "alpha_grid": {"start": -0.4, "stop": 1.4, "num": 20},
                    "beta_grid": {"start": 0.9, "stop": 1.2, "num": 10},
                    "optimization": {"method": "stratified_grid"},
                    "stratified": {
                        "enabled": True,
                        "n_buckets": 4,
                        "coordinate_passes": 2,
                        "min_bucket_size": 10,
                        "enforce_monotonic": True,
                    },
                    "target_band": {"min": 0.0, "max": 5.0},
                },
            }
            config_dir = tmp / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "default.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_experiment(config_path)
            run_dir = Path(result["run_dir"])
            pricing_policy = json.loads((run_dir / "pricing_policy.json").read_text(encoding="utf-8"))

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["metrics"]["pricing"]["optimization_method"], "stratified_grid")
            self.assertIn(result["metrics"]["pricing"]["pricing_policy_kind"], {"scalar", "stratified"})
            self.assertEqual(pricing_policy["kind"], result["metrics"]["pricing"]["pricing_policy_kind"])
            if pricing_policy["kind"] == "stratified":
                self.assertGreaterEqual(len(pricing_policy["bucket_params"]), 2)


if __name__ == "__main__":
    unittest.main()
