from __future__ import annotations

PROPOSE_EXPERIMENT_SCHEMA = {
    "type": "object",
    "required": ["hypothesis", "model_variant", "feature_set", "pricing_policy"],
    "properties": {
        "hypothesis": {"type": "string"},
        "model_variant": {"type": "string"},
        "feature_set": {"type": "string"},
        "pricing_policy": {"type": "string"},
        "expected_impact": {"type": "string"},
    },
}

RUN_EXPERIMENT_SCHEMA = {
    "type": "object",
    "required": ["experiment_id"],
    "properties": {
        "experiment_id": {"type": "string"},
    },
}

EVALUATE_PRICING_SCHEMA = {
    "type": "object",
    "required": ["run_id"],
    "properties": {
        "run_id": {"type": "string"},
    },
}

WRITE_REPORT_SNIPPET_SCHEMA = {
    "type": "object",
    "required": ["run_id"],
    "properties": {
        "run_id": {"type": "string"},
        "focus": {"type": "string"},
    },
}

