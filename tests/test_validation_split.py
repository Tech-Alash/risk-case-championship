from __future__ import annotations

import logging
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.features.preprocessing import PreprocessingConfig
from risk_case.models.benchmark import BenchmarkConfig
from risk_case.orchestration.run_pipeline import RunConfig, _split_policy_train_valid


class TestValidationSplit(unittest.TestCase):
    def _build_config(self, scheme: str = "group_time") -> RunConfig:
        return RunConfig(
            train_csv=Path("train.csv"),
            test_csv=None,
            artifacts_dir=Path("artifacts"),
            split_test_size=0.2,
            split_random_state=42,
            validation_scheme=scheme,
            validation_group_column="contract_number",
            validation_time_column="operation_date",
            validation_time_holdout_start="2024-05-01",
            validation_group_kfold_n_splits=5,
            model_max_iter=100,
            model_ridge_alpha=1.0,
            pricing_target_lr=0.7,
            pricing_alpha_start=-0.6,
            pricing_alpha_stop=1.8,
            pricing_alpha_num=20,
            preprocessing=PreprocessingConfig(),
            benchmark=BenchmarkConfig.from_dict({"enabled": False}),
            logging_level="INFO",
        )

    def test_group_time_split_respects_cutoff(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "contract_number": [f"c{i}" for i in range(20)],
                "operation_date": pd.date_range("2024-01-01", periods=20, freq="15D").astype(str),
                "is_claim": (rng.random(20) < 0.2).astype(int),
                "claim_amount": rng.uniform(0, 1000, 20),
                "claim_cnt": (rng.random(20) < 0.2).astype(int),
            }
        )

        logger = logging.getLogger("test.split")
        train_df, valid_df, meta = _split_policy_train_valid(df, self._build_config(), logger)

        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(valid_df), 0)
        self.assertEqual(meta["scheme"], "group_time")
        self.assertTrue((pd.to_datetime(valid_df["operation_date"]) >= pd.Timestamp("2024-05-01")).all())
        self.assertTrue(set(train_df["contract_number"]).isdisjoint(set(valid_df["contract_number"])))

    def test_group_time_falls_back_to_group_when_time_missing(self) -> None:
        df = pd.DataFrame(
            {
                "contract_number": [f"c{i}" for i in range(15)],
                "is_claim": [0, 1] * 7 + [0],
                "claim_amount": [0.0] * 15,
                "claim_cnt": [0, 1] * 7 + [0],
            }
        )
        logger = logging.getLogger("test.split.fallback")
        train_df, valid_df, meta = _split_policy_train_valid(df, self._build_config(), logger)
        self.assertEqual(meta["scheme"], "group")
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(valid_df), 0)
        self.assertTrue(set(train_df["contract_number"]).isdisjoint(set(valid_df["contract_number"])))


if __name__ == "__main__":
    unittest.main()
