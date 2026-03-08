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

from risk_case.pricing.evaluator import (
    RetentionConfig,
    StratifiedPricingConfig,
    estimate_retention_probabilities,
    evaluate_pricing,
    select_best_pricing,
)
from risk_case.pricing.policy import apply_pricing_policy, apply_pricing_policy_artifact


class TestPricing(unittest.TestCase):
    def test_pricing_respects_constraints(self) -> None:
        df = pd.DataFrame(
            {
                "premium": [100.0, 200.0, 300.0],
                "premium_wo_term": [100.0, 200.0, 300.0],
                "claim_amount": [0.0, 1000.0, 0.0],
                "claim_cnt": [0, 1, 0],
                "is_claim": [0, 1, 0],
            }
        )
        expected_loss = pd.Series([10.0, 300.0, 20.0])
        new_premium = apply_pricing_policy(df, expected_loss, alpha=2.0)

        self.assertTrue((new_premium >= 0.0).all())
        self.assertTrue((new_premium <= 3.0 * df["premium"]).all())

        metrics = evaluate_pricing(df, new_premium, target_lr=0.7)
        self.assertGreaterEqual(metrics.share_group1, 0.0)
        self.assertLessEqual(metrics.share_group1, 1.0)

    def test_select_best_pricing_prefers_in_target_band(self) -> None:
        df = pd.DataFrame(
            {
                "premium": [100.0, 100.0, 100.0, 100.0],
                "premium_wo_term": [100.0, 100.0, 100.0, 100.0],
                "claim_amount": [70.0, 70.0, 70.0, 70.0],
                "claim_cnt": [1, 1, 1, 1],
                "is_claim": [1, 1, 1, 1],
            }
        )
        expected_loss = pd.Series([10.0, 10.0, 10.0, 10.0])

        best_alpha, best_beta, _, best_eval = select_best_pricing(
            df=df,
            expected_loss=expected_loss,
            target_lr=0.7,
            alpha_grid=np.asarray([0.0]),
            beta_grid=np.asarray([0.8, 1.0, 1.2]),
            target_band=(0.69, 0.71),
        )

        self.assertEqual(best_alpha, 0.0)
        self.assertAlmostEqual(best_beta, 1.0)
        self.assertTrue(best_eval.in_target)

    def test_retention_probabilities_monotonic_wrt_price_delta(self) -> None:
        base = pd.Series([100.0, 100.0, 100.0, 100.0])
        new = pd.Series([80.0, 100.0, 120.0, 160.0])
        cfg = RetentionConfig(
            enabled=True,
            base_retention=0.90,
            elasticity=5.0,
            center=0.0,
            floor=0.05,
            cap=0.99,
        )

        retention = estimate_retention_probabilities(base, new, retention_config=cfg)
        self.assertGreater(retention.iloc[0], retention.iloc[1])
        self.assertGreater(retention.iloc[1], retention.iloc[2])
        self.assertGreater(retention.iloc[2], retention.iloc[3])

    def test_slsqp_method_is_not_worse_than_grid_key(self) -> None:
        df = pd.DataFrame(
            {
                "premium": [90.0, 110.0, 130.0, 160.0, 200.0],
                "premium_wo_term": [90.0, 110.0, 130.0, 160.0, 200.0],
                "claim_amount": [20.0, 120.0, 220.0, 100.0, 180.0],
                "claim_cnt": [0, 1, 1, 0, 1],
                "is_claim": [0, 1, 1, 0, 1],
            }
        )
        expected_loss = pd.Series([15.0, 70.0, 120.0, 60.0, 100.0])
        alpha_grid = np.asarray([-0.4, 0.0, 0.4, 0.8], dtype=float)
        beta_grid = np.asarray([0.8, 1.0, 1.2], dtype=float)

        grid_alpha, grid_beta, _, grid_eval = select_best_pricing(
            df=df,
            expected_loss=expected_loss,
            target_lr=0.7,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            target_band=(0.6, 0.8),
            method="grid",
        )
        slsqp_alpha, slsqp_beta, _, slsqp_eval = select_best_pricing(
            df=df,
            expected_loss=expected_loss,
            target_lr=0.7,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            target_band=(0.6, 0.8),
            method="slsqp",
            slsqp_options={"maxiter": 60, "ftol": 1e-6, "eps": 1e-3},
        )

        self.assertGreaterEqual(slsqp_alpha, float(alpha_grid.min()))
        self.assertLessEqual(slsqp_alpha, float(alpha_grid.max()))
        self.assertGreaterEqual(slsqp_beta, float(beta_grid.min()))
        self.assertLessEqual(slsqp_beta, float(beta_grid.max()))

        grid_key = (1 if grid_eval.in_target else 0, float(grid_eval.score), float(grid_eval.share_group1))
        slsqp_key = (1 if slsqp_eval.in_target else 0, float(slsqp_eval.score), float(slsqp_eval.share_group1))
        self.assertGreaterEqual(slsqp_key, grid_key)

    def test_stratified_grid_returns_serializable_policy_and_respects_monotonicity(self) -> None:
        df = pd.DataFrame(
            {
                "premium": [80.0, 95.0, 110.0, 140.0, 180.0, 220.0, 260.0, 320.0],
                "premium_wo_term": [80.0, 95.0, 110.0, 140.0, 180.0, 220.0, 260.0, 320.0],
                "claim_amount": [20.0, 50.0, 40.0, 80.0, 160.0, 220.0, 260.0, 320.0],
                "claim_cnt": [0, 0, 0, 1, 1, 1, 1, 1],
                "is_claim": [0, 0, 0, 1, 1, 1, 1, 1],
            }
        )
        expected_loss = pd.Series([10.0, 15.0, 20.0, 60.0, 90.0, 140.0, 210.0, 280.0], dtype=float)

        warm_alpha, warm_beta, new_premium, pricing_eval = select_best_pricing(
            df=df,
            expected_loss=expected_loss,
            target_lr=0.7,
            alpha_grid=np.asarray([0.0, 0.3, 0.6, 0.9], dtype=float),
            beta_grid=np.asarray([0.9, 1.0, 1.1], dtype=float),
            target_band=(0.4, 1.2),
            method="stratified_grid",
            stratified_config=StratifiedPricingConfig(
                enabled=True,
                n_buckets=3,
                min_bucket_size=2,
                coordinate_passes=2,
                enforce_monotonic=True,
            ),
        )

        self.assertIsNotNone(pricing_eval.pricing_policy)
        pricing_policy = pricing_eval.pricing_policy
        assert pricing_policy is not None
        self.assertEqual(pricing_policy.kind, "stratified")
        self.assertEqual(pricing_policy.method, "stratified_grid")
        self.assertAlmostEqual(float(pricing_policy.global_init_alpha), float(warm_alpha))
        self.assertAlmostEqual(float(pricing_policy.global_init_beta), float(warm_beta))
        self.assertGreaterEqual(len(pricing_policy.bucket_params), 2)

        reapplied = apply_pricing_policy_artifact(df=df, expected_loss=expected_loss, pricing_policy=pricing_policy)
        self.assertTrue(np.allclose(new_premium.values, reapplied.values))
        self.assertTrue((new_premium >= 0.0).all())
        self.assertTrue((new_premium <= 3.0 * df["premium"]).all())

        bucket_ids = np.digitize(expected_loss.to_numpy(dtype=float), np.asarray(pricing_policy.bucket_edges, dtype=float), right=False)
        base_multiplier = np.where(df["premium"].to_numpy(dtype=float) > 0, new_premium.to_numpy(dtype=float) / df["premium"].to_numpy(dtype=float), 0.0)
        bucket_means = pd.Series(base_multiplier).groupby(bucket_ids).mean().to_numpy(dtype=float)
        self.assertTrue(np.all(np.diff(bucket_means) >= -1e-9))


if __name__ == "__main__":
    unittest.main()
