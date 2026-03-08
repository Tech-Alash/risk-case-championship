from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
import unittest

import pandas as pd
from joblib import dump

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional test dependency in local env
    TestClient = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.api.main import app
from risk_case.pricing.artifacts import PricingPolicyArtifact


class _DummyModel:
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_loss = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0).clip(lower=0.0)
        p_claim = (expected_loss / expected_loss.max()).fillna(0.0) if len(expected_loss) else expected_loss
        expected_severity = pd.Series(1000.0, index=df.index)
        return pd.DataFrame(
            {
                "p_claim": p_claim,
                "expected_severity": expected_severity,
                "expected_loss": expected_loss,
            },
            index=df.index,
        )


class TestApiPricingPolicy(unittest.TestCase):
    def setUp(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi TestClient/httpx is not available")
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.prev_cwd = os.getcwd()
        self.addCleanup(os.chdir, self.prev_cwd)
        os.chdir(self.tmpdir.name)

        artifacts = Path("artifacts")
        run_dir = artifacts / "runs" / "20260308_000000"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.client = TestClient(app)

        dump(_DummyModel(), run_dir / "model.joblib")
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "pricing": {
                        "alpha": 0.2,
                        "beta": 1.0,
                        "optimization_method": "grid",
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_latest_pointer(self, pricing_policy_path: Path | None = None) -> None:
        payload = {
            "run_id": "20260308_000000",
            "run_dir": str(self.run_dir),
            "model_path": str(self.run_dir / "model.joblib"),
            "metrics_path": str(self.run_dir / "metrics.json"),
        }
        if pricing_policy_path is not None:
            payload["pricing_policy_path"] = str(pricing_policy_path)
        (Path("artifacts") / "latest_run.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def test_reprice_uses_persisted_scalar_policy(self) -> None:
        pricing_policy_path = self.run_dir / "pricing_policy.json"
        PricingPolicyArtifact.scalar(alpha=0.5, beta=1.1, method="grid").save(pricing_policy_path)
        self._write_latest_pointer(pricing_policy_path=pricing_policy_path)

        response = self.client.post(
            "/reprice",
            json={"records": [{"premium": 100.0, "risk_score": 20.0}, {"premium": 120.0, "risk_score": 50.0}]},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["pricing_policy_kind"], "scalar")
        self.assertAlmostEqual(payload["alpha"], 0.5)
        self.assertAlmostEqual(payload["beta"], 1.1)

    def test_reprice_uses_persisted_stratified_policy(self) -> None:
        pricing_policy_path = self.run_dir / "pricing_policy.json"
        PricingPolicyArtifact(
            kind="stratified",
            method="stratified_grid",
            alpha=0.1,
            beta=1.0,
            bucket_edges=[30.0],
            bucket_params=[
                {"bucket_id": 0, "alpha": 0.0, "beta": 0.9, "mean_expected_loss": 15.0, "count": 1},
                {"bucket_id": 1, "alpha": 0.6, "beta": 1.2, "mean_expected_loss": 60.0, "count": 1},
            ],
            global_init_alpha=0.1,
            global_init_beta=1.0,
        ).save(pricing_policy_path)
        self._write_latest_pointer(pricing_policy_path=pricing_policy_path)

        response = self.client.post(
            "/reprice",
            json={"records": [{"premium": 100.0, "risk_score": 10.0}, {"premium": 100.0, "risk_score": 80.0}]},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["pricing_policy_kind"], "stratified")
        premiums = [item["new_premium"] for item in payload["predictions"]]
        self.assertNotEqual(premiums[0], premiums[1])

    def test_reprice_manual_override_keeps_backward_compatible_scalar_path(self) -> None:
        pricing_policy_path = self.run_dir / "pricing_policy.json"
        PricingPolicyArtifact(
            kind="stratified",
            method="stratified_grid",
            alpha=0.1,
            beta=1.0,
            bucket_edges=[30.0],
            bucket_params=[
                {"bucket_id": 0, "alpha": 0.0, "beta": 0.9, "mean_expected_loss": 15.0, "count": 1},
                {"bucket_id": 1, "alpha": 0.6, "beta": 1.2, "mean_expected_loss": 60.0, "count": 1},
            ],
            global_init_alpha=0.1,
            global_init_beta=1.0,
        ).save(pricing_policy_path)
        self._write_latest_pointer(pricing_policy_path=pricing_policy_path)

        response = self.client.post(
            "/reprice",
            json={
                "alpha": 0.0,
                "beta": 1.0,
                "records": [{"premium": 100.0, "risk_score": 10.0}, {"premium": 100.0, "risk_score": 80.0}],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["pricing_policy_kind"], "scalar")
        premiums = [item["new_premium"] for item in payload["predictions"]]
        self.assertAlmostEqual(premiums[0], 100.0)
        self.assertAlmostEqual(premiums[1], 100.0)


if __name__ == "__main__":
    unittest.main()
