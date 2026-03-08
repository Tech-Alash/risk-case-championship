from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.eda.feature_selection import FeatureSelectionConfig, build_feature_selection_spec


class TestFeatureSelection(unittest.TestCase):
    def test_strict_missing_rule_and_force_keep(self) -> None:
        rows = 200
        df = pd.DataFrame(
            {
                "contract_number": [f"c_{i}" for i in range(rows)],
                "is_claim": [0] * rows,
                "premium": [1000.0] * rows,
                "premium_wo_term": [800.0] * rows,
                "dense_feature": list(range(rows)),
                "sparse_feature": [None] * 190 + [1.0] * 10,
                "driver_iin": [f"d_{i}" for i in range(rows)],
            }
        )
        cfg = FeatureSelectionConfig(
            missing_drop_threshold=0.90,
            force_keep=["premium", "premium_wo_term"],
            force_drop=["driver_iin", "is_claim", "contract_number"],
        )
        result = build_feature_selection_spec(df, cfg)
        drop_features = set(result.droplist["feature"].tolist())
        keep_features = set(result.whitelist["feature"].tolist())

        self.assertIn("sparse_feature", drop_features)
        self.assertIn("premium", keep_features)
        self.assertIn("premium_wo_term", keep_features)
        self.assertNotIn("driver_iin", keep_features)


if __name__ == "__main__":
    unittest.main()

