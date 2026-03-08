from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from risk_case.models.frequency_severity import FrequencySeverityModel
from risk_case.pricing.artifacts import PricingPolicyArtifact
from risk_case.pricing.policy import apply_pricing_policy_artifact


class RecordsPayload(BaseModel):
    records: list[dict[str, Any]] = Field(default_factory=list)
    alpha: float | None = None
    beta: float | None = None


def _default_artifacts_root() -> Path:
    return Path("artifacts")


def _latest_pointer(artifacts_root: Path) -> dict[str, Any]:
    pointer = artifacts_root / "latest_run.json"
    if not pointer.exists():
        raise HTTPException(status_code=404, detail="No run artifacts found. Run pipeline first.")
    return json.loads(pointer.read_text(encoding="utf-8"))


def _load_model(artifacts_root: Path) -> FrequencySeverityModel:
    latest = _latest_pointer(artifacts_root)
    model_path = Path(latest["model_path"])
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
    return FrequencySeverityModel.load(model_path)


def _load_default_pricing_policy(latest: dict[str, Any]) -> PricingPolicyArtifact:
    pricing_policy_path_raw = latest.get("pricing_policy_path")
    if pricing_policy_path_raw:
        pricing_policy_path = Path(pricing_policy_path_raw)
        if pricing_policy_path.exists():
            return PricingPolicyArtifact.load(pricing_policy_path)

    metrics_path = Path(latest["metrics_path"])
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    pricing_metrics = metrics.get("pricing", {})
    return PricingPolicyArtifact.scalar(
        alpha=float(pricing_metrics.get("alpha", 0.0)),
        beta=float(pricing_metrics.get("beta", 1.0)),
        method=str(pricing_metrics.get("optimization_method", "grid")),
    )


app = FastAPI(title="Risk Case API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
def score(payload: RecordsPayload) -> dict[str, Any]:
    if not payload.records:
        raise HTTPException(status_code=400, detail="records cannot be empty")

    artifacts_root = _default_artifacts_root()
    model = _load_model(artifacts_root)
    df = pd.DataFrame(payload.records)
    pred = model.predict(df)
    return {"predictions": pred.to_dict(orient="records")}


@app.post("/reprice")
def reprice(payload: RecordsPayload) -> dict[str, Any]:
    if not payload.records:
        raise HTTPException(status_code=400, detail="records cannot be empty")

    artifacts_root = _default_artifacts_root()
    latest = _latest_pointer(artifacts_root)
    default_policy = _load_default_pricing_policy(latest)
    alpha = payload.alpha if payload.alpha is not None else default_policy.alpha
    beta = payload.beta if payload.beta is not None else default_policy.beta
    if payload.alpha is not None or payload.beta is not None:
        pricing_policy = PricingPolicyArtifact.scalar(
            alpha=float(alpha if alpha is not None else 0.0),
            beta=float(beta if beta is not None else 1.0),
            method="manual_override",
        )
    else:
        pricing_policy = default_policy

    model = _load_model(artifacts_root)
    df = pd.DataFrame(payload.records)
    pred = model.predict(df)
    new_premium = apply_pricing_policy_artifact(
        df=df,
        expected_loss=pred["expected_loss"],
        pricing_policy=pricing_policy,
    )

    output = pred.copy()
    output["new_premium"] = new_premium.values
    return {
        "alpha": alpha,
        "beta": beta,
        "pricing_policy_kind": pricing_policy.kind,
        "pricing_policy": pricing_policy.to_summary(),
        "predictions": output.to_dict(orient="records"),
    }


@app.get("/metrics/{run_id}")
def read_metrics(run_id: str) -> dict[str, Any]:
    metrics_path = _default_artifacts_root() / "runs" / run_id / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))
