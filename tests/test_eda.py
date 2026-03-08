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

from risk_case.eda import EDAConfig, run_eda


class TestEDA(unittest.TestCase):
    def test_run_eda_creates_expected_artifacts(self) -> None:
        rng = np.random.default_rng(7)
        rows = 120

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_dir = tmp / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            train = pd.DataFrame(
                {
                    "unique_id": [f"id_{i}" for i in range(rows)],
                    "contract_number": [f"contract_{i // 3}" for i in range(rows)],
                    "premium": rng.uniform(3000, 12000, rows),
                    "premium_wo_term": rng.uniform(2000, 10000, rows),
                    "is_claim": (rng.random(rows) < 0.1).astype(int),
                    "claim_cnt": (rng.random(rows) < 0.1).astype(int),
                    "claim_amount": rng.gamma(2, 500, rows),
                    "engine_power": rng.uniform(55, 320, rows),
                    "engine_volume": rng.uniform(1.0, 5.0, rows),
                    "region_name": rng.choice(["A", "B", "C"], rows),
                    "driver_iin": [f"d_{i%20}" for i in range(rows)],
                }
            )
            test = train.drop(columns=["is_claim", "claim_cnt", "claim_amount"]).copy()

            train_path = data_dir / "train.csv"
            test_path = data_dir / "test.csv"
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            (tmp / "reports").mkdir(parents=True, exist_ok=True)
            cfg = EDAConfig(
                train_csv=train_path,
                test_csv=test_path,
                output_dir=tmp / "artifacts" / "eda",
                sample_nrows=None,
                export_figures=True,
            )
            result = run_eda(config=cfg, project_root=tmp)

            self.assertEqual(result["status"], "ok")
            self.assertTrue((tmp / "artifacts" / "eda" / "tables" / "missing_top20.csv").exists())
            self.assertTrue((tmp / "artifacts" / "eda" / "tables" / "policy_vs_driver_kpis.csv").exists())
            self.assertTrue((tmp / "artifacts" / "eda" / "figures" / "target_distribution.png").exists())
            self.assertTrue((tmp / "artifacts" / "eda" / "metadata" / "eda_profile.json").exists())
            self.assertTrue((tmp / "reports" / "eda_summary.md").exists())
            self.assertTrue((tmp / "artifacts" / "eda" / "feature_selection" / "feature_whitelist.csv").exists())
            self.assertTrue((tmp / "artifacts" / "eda" / "feature_selection" / "feature_droplist.csv").exists())
            self.assertTrue((tmp / "artifacts" / "eda" / "feature_selection" / "feature_review_list.csv").exists())
            self.assertTrue((tmp / "artifacts" / "eda" / "feature_selection" / "feature_selection_summary.json").exists())
            self.assertTrue((tmp / "reports" / "feature_selection_report.md").exists())

            profile = json.loads((tmp / "artifacts" / "eda" / "metadata" / "eda_profile.json").read_text(encoding="utf-8"))
            self.assertIn("claim_rate", profile)
            self.assertIn("premium_duplication_factor", profile)
            self.assertIn("feature_selection", profile)


if __name__ == "__main__":
    unittest.main()
