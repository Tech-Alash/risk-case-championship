from __future__ import annotations

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

from risk_case.data.policy_aggregation import aggregate_to_policy_level
from risk_case.features.preprocessing import (
    PreprocessingConfig,
    build_oof_target_encoding_features,
    fit_preprocessor,
    transform_with_preprocessor,
)


class TestPreprocessing(unittest.TestCase):
    def test_policy_aggregation_outputs_unique_contract(self) -> None:
        df = pd.DataFrame(
            {
                "contract_number": ["c1", "c1", "c2"],
                "premium": [100.0, 100.0, 150.0],
                "premium_wo_term": [90.0, 90.0, 140.0],
                "is_claim": [0, 1, 0],
                "claim_amount": [0.0, 200.0, 0.0],
                "claim_cnt": [0, 1, 0],
                "engine_power": [80, 120, 110],
                "region_name": ["A", "B", "A"],
            }
        )
        policy_df = aggregate_to_policy_level(df)
        self.assertEqual(policy_df["contract_number"].nunique(), len(policy_df))
        self.assertEqual(len(policy_df), 2)
        self.assertIn("driver_count", policy_df.columns)

    def test_preprocessor_drops_forbidden_from_features(self) -> None:
        df = pd.DataFrame(
            {
                "contract_number": ["c1", "c2"],
                "premium": [100.0, None],
                "premium_wo_term": [90.0, 80.0],
                "is_claim": [0, 1],
                "claim_amount": [0.0, 400.0],
                "claim_cnt": [0, 1],
                "driver_iin": ["x", "y"],
                "region_name": ["A", None],
            }
        )

        cfg = PreprocessingConfig.from_dict(
            {
                "grain": "contract_number",
                "drop_columns": ["driver_iin"],
                "forbidden_feature_columns": ["contract_number", "driver_iin"],
                "target_columns": ["is_claim", "claim_amount", "claim_cnt"],
            }
        )
        state = fit_preprocessor(df, cfg)
        processed = transform_with_preprocessor(df, state)

        self.assertNotIn("driver_iin", processed.columns)
        self.assertIn("premium", processed.columns)
        self.assertIn("region_name", processed.columns)
        self.assertTrue(any(col.endswith("_is_missing") for col in processed.columns))

    def test_preprocessor_uses_whitelist_and_droplist_files(self) -> None:
        df = pd.DataFrame(
            {
                "contract_number": ["c1", "c2", "c3"],
                "premium": [100.0, 120.0, 140.0],
                "premium_wo_term": [90.0, 100.0, 130.0],
                "is_claim": [0, 1, 0],
                "claim_amount": [0.0, 150.0, 0.0],
                "claim_cnt": [0, 1, 0],
                "engine_power_mean": [80.0, 110.0, 95.0],
                "region_name": ["A", "B", "A"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            whitelist = pd.DataFrame({"feature": ["premium", "engine_power_mean"]})
            droplist = pd.DataFrame({"feature": ["engine_power_mean"]})
            whitelist_path = tmp / "whitelist.csv"
            droplist_path = tmp / "droplist.csv"
            whitelist.to_csv(whitelist_path, index=False)
            droplist.to_csv(droplist_path, index=False)

            cfg = PreprocessingConfig.from_dict(
                {
                    "grain": "contract_number",
                    "feature_whitelist_path": str(whitelist_path),
                    "feature_droplist_path": str(droplist_path),
                    "selection_rules": {"force_keep": ["premium", "premium_wo_term"]},
                }
            )
            state = fit_preprocessor(df, cfg)
            features = set(state.feature_columns)
            self.assertIn("premium", features)
            self.assertIn("premium_wo_term", features)
            self.assertNotIn("engine_power_mean", features)

    def test_preprocessor_builds_date_encoding_and_interaction_features(self) -> None:
        df = pd.DataFrame(
            {
                "contract_number": ["c1", "c2", "c3", "c4"],
                "operation_date": ["2024-01-15", "2024-02-18", "2024-03-01", "2024-03-27"],
                "premium": [100.0, 120.0, 140.0, 180.0],
                "premium_wo_term": [90.0, 105.0, 130.0, 160.0],
                "driver_count": [1, 2, 1, 3],
                "engine_power_mean": [80.0, 100.0, 120.0, 90.0],
                "region_name": ["A", "B", "A", "B"],
                "vehicle_type_name": ["sedan", "truck", "sedan", "truck"],
                "model": ["m1", "m2", "m1", "m2"],
                "mark": ["x", "y", "x", "y"],
                "ownerkato": ["k1", "k2", "k1", "k2"],
                "car_year": [2019, 2020, 2021, 2018],
                "bonus_malus": [5, 6, 4, 7],
                "is_claim": [0, 1, 0, 1],
                "claim_amount": [0.0, 150.0, 0.0, 220.0],
                "claim_cnt": [0, 1, 0, 1],
            }
        )
        cfg = PreprocessingConfig.from_dict(
            {
                "grain": "contract_number",
                "date_features": {"columns": ["operation_date"], "features": ["month", "sin_month", "cos_month"]},
                "target_encoding": {"enabled": True, "columns": ["model", "mark"], "smoothing": 5.0},
                "freq_encoding": {"enabled": True, "columns": ["model", "mark"]},
                "interaction_features": {
                    "enabled": True,
                    "definitions": ["premium_per_driver", "premium_per_power", "region_x_vehicle_type"],
                },
            }
        )
        state = fit_preprocessor(df, cfg)
        processed = transform_with_preprocessor(df, state)

        self.assertIn("operation_date_month", processed.columns)
        self.assertIn("operation_date_sin_month", processed.columns)
        self.assertIn("model_te", processed.columns)
        self.assertIn("mark_freq", processed.columns)
        self.assertIn("premium_per_driver", processed.columns)
        self.assertIn("premium_per_power", processed.columns)
        self.assertIn("region_x_vehicle_type", processed.columns)
        self.assertTrue(np.isfinite(processed["premium_per_driver"]).all())

    def test_preprocessor_builds_missing_aggregates(self) -> None:
        df = pd.DataFrame(
            {
                "contract_number": ["c1", "c2", "c3", "c4"],
                "premium": [100.0, 120.0, 140.0, 160.0],
                "premium_wo_term": [90.0, 100.0, 120.0, 150.0],
                "SCORE_4_1": [1.0, np.nan, 3.0, np.nan],
                "SCORE_11_1": [np.nan, np.nan, 0.5, 1.0],
                "is_claim": [0, 1, 0, 1],
                "claim_amount": [0.0, 100.0, 0.0, 200.0],
                "claim_cnt": [0, 1, 0, 1],
            }
        )
        cfg = PreprocessingConfig.from_dict(
            {
                "grain": "contract_number",
                "missing": {
                    "add_missing_flags": True,
                    "add_missing_aggregates": True,
                    "missing_flag_threshold": 0.0,
                    "missing_aggregate_prefixes": ["SCORE_4_", "SCORE_11_", "SCORE_"],
                },
            }
        )
        state = fit_preprocessor(df, cfg)
        processed = transform_with_preprocessor(df, state)

        self.assertIn("score_missing_cnt_total", processed.columns)
        self.assertIn("score_missing_share", processed.columns)
        self.assertIn("score_4_missing_cnt", processed.columns)
        self.assertIn("score_11_missing_cnt", processed.columns)
        self.assertTrue((processed["score_missing_cnt_total"] >= 0).all())
        self.assertTrue((processed["score_missing_share"] >= 0).all())

    def test_interaction_features_mvp_generates_and_filters_by_max_features(self) -> None:
        rows = 80
        rng = np.random.default_rng(9)
        df = pd.DataFrame(
            {
                "contract_number": [f"c{i}" for i in range(rows)],
                "operation_date": pd.date_range("2022-01-01", periods=rows, freq="D").astype(str),
                "premium": rng.uniform(100.0, 300.0, rows),
                "premium_wo_term": rng.uniform(90.0, 260.0, rows),
                "engine_power_mean": rng.uniform(70.0, 180.0, rows),
                "car_age_mean": rng.uniform(1.0, 15.0, rows),
                "bonus_malus_mean": rng.uniform(2.0, 8.0, rows),
                "SCORE_1_1": rng.normal(loc=0.0, scale=1.0, size=rows),
                "SCORE_2_1": rng.normal(loc=0.2, scale=1.2, size=rows),
                "is_claim": (rng.random(rows) < 0.25).astype(int),
                "claim_amount": rng.gamma(shape=2.0, scale=100.0, size=rows),
                "claim_cnt": (rng.random(rows) < 0.25).astype(int),
            }
        )
        cfg = PreprocessingConfig.from_dict(
            {
                "grain": "contract_number",
                "interaction_features": {"enabled": False, "definitions": []},
                "interaction_features_mvp": {
                    "enabled": True,
                    "definitions": [
                        "score_x_premium_ratio",
                        "premium_x_power",
                        "score_dispersion_x_premium",
                    ],
                    "max_features": 1,
                    "corr_filter_threshold": 1.1,
                    "psi_filter_threshold": 99.0,
                    "require_business_whitelist": True,
                },
                "target_encoding": {"enabled": False, "columns": []},
                "freq_encoding": {"enabled": False, "columns": []},
            }
        )
        state = fit_preprocessor(df, cfg)
        processed = transform_with_preprocessor(df, state)

        mvp_report = state.feature_pruning_report.get("interaction_mvp_report", {})
        retained = mvp_report.get("retained", [])
        self.assertEqual(len(retained), 1)
        self.assertIn(retained[0], state.feature_columns)
        self.assertIn(retained[0], processed.columns)

    def test_interaction_features_mvp_respects_business_whitelist(self) -> None:
        rows = 40
        rng = np.random.default_rng(10)
        df = pd.DataFrame(
            {
                "contract_number": [f"c{i}" for i in range(rows)],
                "premium": rng.uniform(100.0, 300.0, rows),
                "premium_wo_term": rng.uniform(90.0, 260.0, rows),
                "engine_power_mean": rng.uniform(70.0, 180.0, rows),
                "SCORE_1_1": rng.normal(loc=0.0, scale=1.0, size=rows),
                "is_claim": (rng.random(rows) < 0.25).astype(int),
                "claim_amount": rng.gamma(shape=2.0, scale=100.0, size=rows),
                "claim_cnt": (rng.random(rows) < 0.25).astype(int),
            }
        )
        cfg = PreprocessingConfig.from_dict(
            {
                "grain": "contract_number",
                "interaction_features": {"enabled": False, "definitions": []},
                "interaction_features_mvp": {
                    "enabled": True,
                    "definitions": ["score_x_premium_ratio", "custom_interaction_not_allowed"],
                    "require_business_whitelist": True,
                },
                "target_encoding": {"enabled": False, "columns": []},
                "freq_encoding": {"enabled": False, "columns": []},
            }
        )
        state = fit_preprocessor(df, cfg)

        mvp_report = state.feature_pruning_report.get("interaction_mvp_report", {})
        self.assertIn("custom_interaction_not_allowed", mvp_report.get("dropped_by_whitelist", []))
        self.assertNotIn("custom_interaction_not_allowed", state.feature_columns)

    def test_oof_target_encoding_features_shape_and_columns(self) -> None:
        df = pd.DataFrame(
            {
                "contract_number": [f"c{i}" for i in range(12)],
                "premium": np.linspace(100.0, 220.0, 12),
                "premium_wo_term": np.linspace(90.0, 200.0, 12),
                "model": ["m1", "m2", "m1", "m3", "m2", "m1", "m3", "m2", "m1", "m2", "m3", "m1"],
                "mark": ["a", "b", "a", "c", "b", "a", "c", "b", "a", "b", "c", "a"],
                "is_claim": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                "claim_amount": [0.0, 120.0, 0.0, 90.0, 0.0, 80.0, 0.0, 0.0, 100.0, 0.0, 75.0, 0.0],
                "claim_cnt": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            }
        )
        cfg = PreprocessingConfig.from_dict(
            {
                "grain": "contract_number",
                "target_encoding": {"enabled": True, "columns": ["model", "mark"], "smoothing": 5.0},
                "freq_encoding": {"enabled": False, "columns": []},
            }
        )
        state = fit_preprocessor(df, cfg)
        oof = build_oof_target_encoding_features(
            df=df,
            state=state,
            target_column="is_claim",
            n_splits=3,
            random_state=42,
            group_column="contract_number",
        )
        self.assertEqual(len(oof), len(df))
        self.assertIn("model_te", oof.columns)
        self.assertIn("mark_te", oof.columns)
        self.assertTrue(np.isfinite(oof["model_te"]).all())

    def test_numeric_like_cleanup_for_bonus_malus_and_car_year(self) -> None:
        df = pd.DataFrame(
            {
                "contract_number": ["c1", "c2", "c3", "c4"],
                "premium": [100.0, 120.0, 140.0, 160.0],
                "premium_wo_term": [90.0, 100.0, 120.0, 150.0],
                "bonus_malus": ["5", "M", " 7 ", None],
                "car_year": ["2\u00a000", "1\u00a098", "2020", "bad"],
                "region_name": ["A", "B", "A", "B"],
                "is_claim": [0, 1, 0, 1],
                "claim_amount": [0.0, 100.0, 0.0, 200.0],
                "claim_cnt": [0, 1, 0, 1],
            }
        )
        cfg = PreprocessingConfig.from_dict(
            {
                "grain": "contract_number",
                "target_encoding": {"enabled": False, "columns": []},
                "freq_encoding": {"enabled": False, "columns": []},
            }
        )
        with self.assertLogs("risk_case.pipeline.preprocessing", level="INFO") as captured:
            state = fit_preprocessor(df, cfg)
            processed = transform_with_preprocessor(df, state)

        self.assertIn("bonus_malus", processed.columns)
        self.assertIn("car_year", processed.columns)
        self.assertEqual(processed.loc[0, "bonus_malus"], "5")
        self.assertEqual(processed.loc[1, "bonus_malus"], "missing")
        self.assertEqual(processed.loc[0, "car_year"], "2000")
        self.assertEqual(processed.loc[1, "car_year"], "1998")
        self.assertEqual(processed.loc[3, "car_year"], "missing")
        self.assertTrue(any("Numeric-like cleanup[fit]" in line for line in captured.output))
        self.assertTrue(any("Numeric-like cleanup[transform]" in line for line in captured.output))

    def test_feature_pruning_removes_duplicate_missing_flags(self) -> None:
        df = pd.DataFrame(
            {
                "contract_number": ["c1", "c2", "c3", "c4", "c5"],
                "premium": [100.0, 120.0, 140.0, 160.0, 180.0],
                "premium_wo_term": [90.0, 110.0, 120.0, 150.0, 170.0],
                "SCORE_1_1": [1.0, np.nan, 3.0, np.nan, 5.0],
                "SCORE_1_2": [10.0, np.nan, 30.0, np.nan, 50.0],
                "is_claim": [0, 1, 0, 1, 0],
                "claim_amount": [0.0, 100.0, 0.0, 200.0, 0.0],
                "claim_cnt": [0, 1, 0, 1, 0],
            }
        )
        cfg = PreprocessingConfig.from_dict(
            {
                "grain": "contract_number",
                "missing": {
                    "add_missing_flags": True,
                    "add_missing_aggregates": True,
                    "missing_flag_threshold": 0.0,
                    "missing_aggregate_prefixes": ["SCORE_1_", "SCORE_"],
                },
                "feature_pruning": {
                    "enabled": True,
                    "drop_exact_duplicates": True,
                    "drop_missing_share": True,
                    "corr_threshold": 0.999,
                },
                "target_encoding": {"enabled": False, "columns": []},
                "freq_encoding": {"enabled": False, "columns": []},
            }
        )
        state = fit_preprocessor(df, cfg)
        processed = transform_with_preprocessor(df, state)

        self.assertIn("score_missing_cnt_total", processed.columns)
        self.assertNotIn("score_missing_share", state.feature_columns)
        score1_missing_flags = [col for col in state.feature_columns if col.startswith("SCORE_1_") and col.endswith("_is_missing")]
        self.assertEqual(len(score1_missing_flags), 1)
        report = state.feature_pruning_report
        self.assertTrue(report.get("enabled"))
        self.assertTrue(report.get("applied"))
        self.assertGreater(int(report.get("dropped_total", 0)), 0)

    def test_drift_pruning_drops_time_unstable_feature(self) -> None:
        rows = 180
        rng = np.random.default_rng(123)
        drift_values = np.concatenate(
            [
                rng.normal(loc=0.0, scale=0.5, size=rows // 2),
                rng.normal(loc=5.0, scale=0.5, size=rows - rows // 2),
            ]
        )
        stable_values = rng.normal(loc=1.0, scale=0.5, size=rows)
        df = pd.DataFrame(
            {
                "contract_number": [f"c{i}" for i in range(rows)],
                "operation_date": pd.date_range("2022-01-01", periods=rows, freq="D").astype(str),
                "premium": rng.uniform(100.0, 250.0, rows),
                "premium_wo_term": rng.uniform(90.0, 230.0, rows),
                "stable_feature": stable_values,
                "drift_feature": drift_values,
                "is_claim": (rng.random(rows) < 0.2).astype(int),
                "claim_amount": rng.gamma(shape=2.0, scale=120.0, size=rows),
                "claim_cnt": (rng.random(rows) < 0.2).astype(int),
            }
        )
        cfg = PreprocessingConfig.from_dict(
            {
                "grain": "contract_number",
                "feature_pruning": {
                    "enabled": False,
                },
                "drift_pruning": {
                    "enabled": True,
                    "time_column": "operation_date",
                    "reference_share": 0.5,
                    "psi_threshold": 0.2,
                    "bins": 8,
                    "min_rows": 40,
                },
                "target_encoding": {"enabled": False, "columns": []},
                "freq_encoding": {"enabled": False, "columns": []},
                "interaction_features": {"enabled": False, "definitions": []},
            }
        )
        state = fit_preprocessor(df, cfg)

        self.assertNotIn("drift_feature", state.feature_columns)
        self.assertIn("stable_feature", state.feature_columns)
        report = state.feature_pruning_report
        dropped = report.get("dropped_drift_psi", [])
        self.assertTrue(any(item.get("dropped") == "drift_feature" for item in dropped))
        self.assertTrue(report.get("applied"))

    def test_drift_pruning_respects_exclusion_patterns(self) -> None:
        rows = 180
        rng = np.random.default_rng(124)
        drift_values = np.concatenate(
            [
                rng.normal(loc=0.0, scale=0.5, size=rows // 2),
                rng.normal(loc=5.0, scale=0.5, size=rows - rows // 2),
            ]
        )
        df = pd.DataFrame(
            {
                "contract_number": [f"c{i}" for i in range(rows)],
                "operation_date": pd.date_range("2022-01-01", periods=rows, freq="D").astype(str),
                "premium": rng.uniform(100.0, 250.0, rows),
                "premium_wo_term": rng.uniform(90.0, 230.0, rows),
                "drift_feature": drift_values,
                "is_claim": (rng.random(rows) < 0.2).astype(int),
                "claim_amount": rng.gamma(shape=2.0, scale=120.0, size=rows),
                "claim_cnt": (rng.random(rows) < 0.2).astype(int),
            }
        )
        cfg = PreprocessingConfig.from_dict(
            {
                "grain": "contract_number",
                "feature_pruning": {
                    "enabled": False,
                },
                "drift_pruning": {
                    "enabled": True,
                    "time_column": "operation_date",
                    "reference_share": 0.5,
                    "psi_threshold": 0.2,
                    "bins": 8,
                    "min_rows": 40,
                    "exclude_patterns": ["*drift_feature"],
                },
                "target_encoding": {"enabled": False, "columns": []},
                "freq_encoding": {"enabled": False, "columns": []},
                "interaction_features": {"enabled": False, "definitions": []},
            }
        )
        state = fit_preprocessor(df, cfg)

        self.assertIn("drift_feature", state.feature_columns)
        report = state.feature_pruning_report
        dropped = report.get("dropped_drift_psi", [])
        self.assertFalse(any(item.get("dropped") == "drift_feature" for item in dropped))
        self.assertIn("drift_feature", report.get("drift", {}).get("excluded_columns", []))


if __name__ == "__main__":
    unittest.main()
