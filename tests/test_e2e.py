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


class TestE2EPipeline(unittest.TestCase):
    def test_pipeline_creates_artifacts(self) -> None:
        rng = np.random.default_rng(42)
        rows = 140

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_dir = tmp / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            is_claim = (rng.random(rows) < 0.15).astype(int)
            claim_amount = is_claim * rng.gamma(shape=2.0, scale=700.0, size=rows)

            train_df = pd.DataFrame(
                {
                    "unique_id": [f"id_{i}" for i in range(rows)],
                    "contract_number": [f"contract_{i // 2}" for i in range(rows)],
                    "premium": rng.uniform(3000, 20000, rows),
                    "premium_wo_term": rng.uniform(2500, 18000, rows),
                    "is_claim": is_claim,
                    "claim_amount": claim_amount,
                    "claim_cnt": is_claim,
                    "is_individual_person_name": rng.choice(["A", "B"], rows),
                    "is_residence_name": rng.choice(["R", "NR"], rows),
                    "region_name": rng.choice(["01", "02", "03"], rows),
                    "age_experience_name": rng.choice(["young", "mature"], rows),
                    "vehicle_type_name": rng.choice(["sedan", "truck"], rows),
                    "car_age": rng.choice(["new", "old"], rows),
                    "bonus_malus": rng.integers(1, 14, rows),
                    "age_experience_id": rng.integers(1, 6, rows),
                    "experience_year": rng.integers(0, 30, rows),
                    "vehicle_type_id": rng.integers(1, 4, rows),
                    "car_year": rng.integers(1998, 2024, rows),
                    "engine_volume": rng.uniform(1.0, 5.0, rows),
                    "engine_power": rng.uniform(45, 300, rows),
                    "SCORE_1_1": rng.uniform(0, 10, rows),
                    "SCORE_1_2": rng.uniform(0, 10, rows),
                }
            )
            test_df = train_df.drop(columns=["is_claim", "claim_amount", "claim_cnt"]).head(20).copy()

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
                "model": {"max_iter": 200, "ridge_alpha": 1.0},
                "pricing": {
                    "target_lr": 0.7,
                    "alpha_grid": {"start": -0.6, "stop": 1.8, "num": 30},
                    "beta_grid": {"start": 1.0, "stop": 1.0, "num": 1},
                    "target_band": {"min": 0.69, "max": 0.71},
                },
            }
            config_dir = tmp / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "default.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_experiment(config_path)
            run_dir = Path(result["run_dir"])

            self.assertEqual(result["status"], "ok")
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "model.joblib").exists())
            self.assertTrue((run_dir / "valid_predictions.csv").exists())
            self.assertTrue((run_dir / "ablation_results.csv").exists())
            self.assertTrue((run_dir / "double_lift_table.csv").exists())
            self.assertTrue((run_dir / "ae_by_risk_decile.csv").exists())
            self.assertTrue((run_dir / "ae_by_segment.csv").exists())
            self.assertTrue((run_dir / "submission.csv").exists())
            self.assertTrue((run_dir / "preprocessed" / "train_policy_preprocessed.csv").exists())
            self.assertTrue((run_dir / "preprocessed" / "train_split_preprocessed.csv").exists())
            self.assertTrue((run_dir / "preprocessed" / "valid_split_preprocessed.csv").exists())
            self.assertTrue((run_dir / "preprocessed" / "preprocess_metadata.json").exists())
            self.assertTrue((run_dir / "preprocessed" / "quality_report.json").exists())
            self.assertTrue((run_dir / "preprocessed" / "inference_policy_preprocessed.csv").exists())

            submission = pd.read_csv(run_dir / "submission.csv")
            self.assertEqual(len(submission), len(test_df))


if __name__ == "__main__":
    unittest.main()
