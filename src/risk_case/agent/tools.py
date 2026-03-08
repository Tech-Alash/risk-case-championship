from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _assert_keys(payload: dict[str, Any], required: list[str]) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")


def propose_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    _assert_keys(payload, ["hypothesis", "model_variant", "feature_set", "pricing_policy"])
    return {
        "status": "accepted",
        "experiment_plan": {
            "hypothesis": payload["hypothesis"],
            "model_variant": payload["model_variant"],
            "feature_set": payload["feature_set"],
            "pricing_policy": payload["pricing_policy"],
        },
    }


def run_experiment(payload: dict[str, Any], latest_run_file: Path) -> dict[str, Any]:
    _assert_keys(payload, ["experiment_id"])
    if latest_run_file.exists():
        return json.loads(latest_run_file.read_text(encoding="utf-8"))
    return {"status": "not_found", "message": "No run artifacts available yet"}


def evaluate_pricing(payload: dict[str, Any], metrics_file: Path) -> dict[str, Any]:
    _assert_keys(payload, ["run_id"])
    if not metrics_file.exists():
        return {"status": "not_found", "message": "metrics file does not exist"}
    raw = json.loads(metrics_file.read_text(encoding="utf-8"))
    return {"run_id": payload["run_id"], "pricing": raw.get("pricing", {})}


def write_report_snippet(payload: dict[str, Any], metrics_file: Path) -> str:
    _assert_keys(payload, ["run_id"])
    if not metrics_file.exists():
        return "No metrics available."
    raw = json.loads(metrics_file.read_text(encoding="utf-8"))
    pricing = raw.get("pricing", {})
    ml = raw.get("ml", {})
    return (
        f"Run {payload['run_id']} produced AUC={ml.get('auc')} and "
        f"LR_total={pricing.get('lr_total')} with share_group1={pricing.get('share_group1')}."
    )

