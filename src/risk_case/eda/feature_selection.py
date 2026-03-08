from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from risk_case.settings import (
    CONTRACT_COL,
    DEFAULT_FORBIDDEN_FEATURE_COLUMNS,
    DEFAULT_TARGET_COLUMNS,
    PREMIUM_COL,
    PREMIUM_NET_COL,
)


@dataclass
class FeatureSelectionConfig:
    policy_grain_col: str = CONTRACT_COL
    target_column: str = "is_claim"
    missing_drop_threshold: float = 0.90
    missing_review_threshold: float = 0.60
    corr_review_threshold: float = 0.08
    force_keep: list[str] = field(default_factory=lambda: [PREMIUM_COL, PREMIUM_NET_COL])
    force_drop: list[str] = field(default_factory=lambda: list(DEFAULT_FORBIDDEN_FEATURE_COLUMNS + DEFAULT_TARGET_COLUMNS))
    max_numeric_corr_features: int = 200

    @staticmethod
    def from_dict(raw: dict[str, Any] | None) -> "FeatureSelectionConfig":
        if not raw:
            return FeatureSelectionConfig()
        return FeatureSelectionConfig(
            policy_grain_col=str(raw.get("policy_grain_col", CONTRACT_COL)),
            target_column=str(raw.get("target_column", "is_claim")),
            missing_drop_threshold=float(raw.get("missing_drop_threshold", 0.90)),
            missing_review_threshold=float(raw.get("missing_review_threshold", 0.60)),
            corr_review_threshold=float(raw.get("corr_review_threshold", 0.08)),
            force_keep=list(raw.get("force_keep", [PREMIUM_COL, PREMIUM_NET_COL])),
            force_drop=list(raw.get("force_drop", list(DEFAULT_FORBIDDEN_FEATURE_COLUMNS + DEFAULT_TARGET_COLUMNS))),
            max_numeric_corr_features=int(raw.get("max_numeric_corr_features", 200)),
        )


@dataclass
class FeatureSelectionResult:
    whitelist: pd.DataFrame
    droplist: pd.DataFrame
    review: pd.DataFrame
    summary: dict[str, Any]


def _candidate_features(df: pd.DataFrame, config: FeatureSelectionConfig) -> list[str]:
    protected = {config.policy_grain_col, config.target_column}
    protected.update(config.force_drop)
    return [col for col in df.columns if col not in protected]


def _compute_numeric_corr_watchlist(
    df: pd.DataFrame,
    config: FeatureSelectionConfig,
    features: list[str],
) -> pd.DataFrame:
    if config.target_column not in df.columns:
        return pd.DataFrame(columns=["feature", "abs_corr", "corr"])

    y = pd.to_numeric(df[config.target_column], errors="coerce").fillna(0.0)
    numeric_candidates = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]
    numeric_candidates = numeric_candidates[: config.max_numeric_corr_features]

    rows: list[dict[str, Any]] = []
    for col in numeric_candidates:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() < 50:
            continue
        corr = x.corr(y)
        if pd.notna(corr):
            rows.append(
                {
                    "feature": col,
                    "abs_corr": float(abs(corr)),
                    "corr": float(corr),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["feature", "abs_corr", "corr"])
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def build_feature_selection_spec(
    policy_df: pd.DataFrame,
    config: FeatureSelectionConfig,
) -> FeatureSelectionResult:
    features = _candidate_features(policy_df, config)
    missing_rate = policy_df[features].isna().mean().to_dict() if features else {}
    corr_watch = _compute_numeric_corr_watchlist(policy_df, config, features)

    drop_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    keep_rows: list[dict[str, Any]] = []

    corr_map = {
        row["feature"]: (float(row["abs_corr"]), float(row["corr"]))
        for row in corr_watch.to_dict(orient="records")
    }

    force_keep_set = set(config.force_keep)
    force_drop_set = set(config.force_drop)

    for feature in features:
        if feature in force_drop_set:
            drop_rows.append({"feature": feature, "reason": "force_drop"})
            continue

        miss = float(missing_rate.get(feature, 0.0))
        abs_corr = float(corr_map.get(feature, (0.0, 0.0))[0])

        if feature in force_keep_set:
            keep_rows.append({"feature": feature, "reason": "force_keep"})
            continue

        if miss > config.missing_drop_threshold:
            drop_rows.append({"feature": feature, "reason": f"missing_rate>{config.missing_drop_threshold:.2f}"})
            continue

        review_reasons: list[str] = []
        if miss > config.missing_review_threshold:
            review_reasons.append(f"missing_rate>{config.missing_review_threshold:.2f}")
        if abs_corr >= config.corr_review_threshold:
            review_reasons.append(f"abs_corr>={config.corr_review_threshold:.2f}")

        if review_reasons:
            review_rows.append(
                {
                    "feature": feature,
                    "reason": "|".join(review_reasons),
                    "missing_rate": miss,
                    "abs_corr_with_target": abs_corr,
                }
            )
        else:
            keep_rows.append({"feature": feature, "reason": "pass_rules"})

    for forced in force_drop_set:
        if forced in policy_df.columns and forced not in [row["feature"] for row in drop_rows]:
            drop_rows.append({"feature": forced, "reason": "force_drop"})

    whitelist_df = pd.DataFrame(keep_rows).sort_values("feature") if keep_rows else pd.DataFrame(columns=["feature", "reason"])
    droplist_df = pd.DataFrame(drop_rows).drop_duplicates(subset=["feature"]).sort_values("feature") if drop_rows else pd.DataFrame(columns=["feature", "reason"])
    review_df = pd.DataFrame(review_rows).sort_values(["abs_corr_with_target", "missing_rate"], ascending=False) if review_rows else pd.DataFrame(columns=["feature", "reason", "missing_rate", "abs_corr_with_target"])

    summary = {
        "total_policy_columns": int(policy_df.shape[1]),
        "candidate_features": int(len(features)),
        "whitelist_count": int(len(whitelist_df)),
        "droplist_count": int(len(droplist_df)),
        "review_count": int(len(review_df)),
        "rules": {
            "missing_drop_threshold": config.missing_drop_threshold,
            "missing_review_threshold": config.missing_review_threshold,
            "corr_review_threshold": config.corr_review_threshold,
        },
        "force_keep": config.force_keep,
        "force_drop": config.force_drop,
    }

    return FeatureSelectionResult(
        whitelist=whitelist_df,
        droplist=droplist_df,
        review=review_df,
        summary=summary,
    )


def save_feature_selection_artifacts(
    result: FeatureSelectionResult,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    whitelist_path = output_dir / "feature_whitelist.csv"
    droplist_path = output_dir / "feature_droplist.csv"
    review_path = output_dir / "feature_review_list.csv"
    summary_path = output_dir / "feature_selection_summary.json"

    result.whitelist.to_csv(whitelist_path, index=False)
    result.droplist.to_csv(droplist_path, index=False)
    result.review.to_csv(review_path, index=False)
    summary_path.write_text(json.dumps(result.summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "whitelist_path": str(whitelist_path),
        "droplist_path": str(droplist_path),
        "review_path": str(review_path),
        "summary_path": str(summary_path),
    }


def write_feature_selection_report(
    report_path: Path,
    result: FeatureSelectionResult,
    artifact_paths: dict[str, str],
) -> None:
    lines = [
        "# Feature Selection Report",
        "",
        "## Summary",
        f"- Candidate features: {result.summary['candidate_features']}",
        f"- Whitelist: {result.summary['whitelist_count']}",
        f"- Droplist: {result.summary['droplist_count']}",
        f"- Review list: {result.summary['review_count']}",
        "",
        "## Rules",
        f"- missing_drop_threshold: {result.summary['rules']['missing_drop_threshold']}",
        f"- missing_review_threshold: {result.summary['rules']['missing_review_threshold']}",
        f"- corr_review_threshold: {result.summary['rules']['corr_review_threshold']}",
        "",
        "## Force Lists",
        f"- force_keep: {result.summary['force_keep']}",
        f"- force_drop: {result.summary['force_drop']}",
        "",
        "## Artifacts",
        f"- whitelist: `{artifact_paths['whitelist_path']}`",
        f"- droplist: `{artifact_paths['droplist_path']}`",
        f"- review list: `{artifact_paths['review_path']}`",
        f"- summary json: `{artifact_paths['summary_path']}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

