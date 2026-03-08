from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.models.benchmark import BenchmarkConfig, CandidateResult, select_benchmark_winner


class TestBenchmarkSelection(unittest.TestCase):
    def test_selects_best_compliant_by_policy_score(self) -> None:
        config = BenchmarkConfig.from_dict(
            {
                "enabled": True,
                "selection_metric": "policy_score",
                "fallback_candidate": "baseline_freq_sev",
                "must_pass_constraints": False,
            }
        )
        results = [
            CandidateResult(
                candidate_name="baseline_freq_sev",
                status="ok",
                ml={"gini": 0.20},
                pricing={"policy_score": -0.40},
                passes_constraints=True,
            ),
            CandidateResult(
                candidate_name="xgboost_freq_sev",
                status="ok",
                ml={"gini": 0.35},
                pricing={"policy_score": -0.15},
                passes_constraints=True,
            ),
        ]

        winner_name, reason = select_benchmark_winner(results, config)
        self.assertEqual(winner_name, "xgboost_freq_sev")
        self.assertEqual(reason, "best_compliant_candidate")

    def test_tie_breaks_by_gini(self) -> None:
        config = BenchmarkConfig.from_dict({"enabled": True, "selection_metric": "policy_score"})
        results = [
            CandidateResult(
                candidate_name="candidate_a",
                status="ok",
                ml={"gini": 0.11},
                pricing={"policy_score": -0.20},
                passes_constraints=True,
            ),
            CandidateResult(
                candidate_name="candidate_b",
                status="ok",
                ml={"gini": 0.17},
                pricing={"policy_score": -0.20},
                passes_constraints=True,
            ),
        ]

        winner_name, _ = select_benchmark_winner(results, config)
        self.assertEqual(winner_name, "candidate_b")

    def test_uses_best_metric_when_no_compliant_candidate(self) -> None:
        config = BenchmarkConfig.from_dict(
            {
                "enabled": True,
                "selection_metric": "policy_score",
                "fallback_candidate": "baseline_freq_sev",
                "must_pass_constraints": False,
            }
        )
        results = [
            CandidateResult(
                candidate_name="baseline_freq_sev",
                status="ok",
                ml={"gini": 0.10},
                pricing={"policy_score": -1.00},
                passes_constraints=False,
                constraint_reasons=["lr_total>1.02"],
            ),
            CandidateResult(
                candidate_name="lightgbm_freq_sev",
                status="ok",
                ml={"gini": 0.25},
                pricing={"policy_score": -0.50},
                passes_constraints=False,
                constraint_reasons=["share_group1<0.35"],
            ),
        ]

        winner_name, reason = select_benchmark_winner(results, config)
        self.assertEqual(winner_name, "lightgbm_freq_sev")
        self.assertEqual(reason, "best_available_no_compliant_candidates")

    def test_can_use_configured_fallback_when_requested(self) -> None:
        config = BenchmarkConfig.from_dict(
            {
                "enabled": True,
                "selection_metric": "policy_score",
                "fallback_strategy": "configured_candidate",
                "fallback_candidate": "baseline_freq_sev",
                "must_pass_constraints": False,
            }
        )
        results = [
            CandidateResult(
                candidate_name="baseline_freq_sev",
                status="ok",
                ml={"gini": 0.10},
                pricing={"policy_score": -1.00},
                passes_constraints=False,
                constraint_reasons=["lr_total>0.71"],
            ),
            CandidateResult(
                candidate_name="lightgbm_freq_sev",
                status="ok",
                ml={"gini": 0.25},
                pricing={"policy_score": -0.50},
                passes_constraints=False,
                constraint_reasons=["share_group1<0.35"],
            ),
        ]

        winner_name, reason = select_benchmark_winner(results, config)
        self.assertEqual(winner_name, "baseline_freq_sev")
        self.assertEqual(reason, "fallback_no_compliant_candidates")

    def test_raises_when_constraints_are_mandatory(self) -> None:
        config = BenchmarkConfig.from_dict(
            {
                "enabled": True,
                "selection_metric": "policy_score",
                "must_pass_constraints": True,
            }
        )
        results = [
            CandidateResult(
                candidate_name="candidate_a",
                status="ok",
                ml={"gini": 0.21},
                pricing={"policy_score": -0.10},
                passes_constraints=False,
                constraint_reasons=["lr_total>1.02"],
            )
        ]

        with self.assertRaises(ValueError):
            select_benchmark_winner(results, config)


if __name__ == "__main__":
    unittest.main()
