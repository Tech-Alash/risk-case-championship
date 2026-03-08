from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk_case.data.policy_aggregation import aggregate_to_policy_level
from risk_case.eda.feature_selection import (
    FeatureSelectionConfig,
    build_feature_selection_spec,
    save_feature_selection_artifacts,
    write_feature_selection_report,
)
from risk_case.settings import (
    CONTRACT_COL,
    DEFAULT_FORBIDDEN_FEATURE_COLUMNS,
    TARGET_AMOUNT_COL,
    TARGET_CLAIM_COL,
    TARGET_COUNT_COL,
    ensure_dir,
)


LOGGER = logging.getLogger("risk_case.eda")


@dataclass
class EDAConfig:
    train_csv: Path
    test_csv: Path
    output_dir: Path
    policy_grain_col: str = CONTRACT_COL
    random_state: int = 42
    sample_nrows: int | None = None
    export_figures: bool = True
    max_numeric_correlation_features: int = 120
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)

    @staticmethod
    def from_json(path: Path) -> "EDAConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        eda_cfg = raw["eda"]
        fs_cfg = FeatureSelectionConfig.from_dict(eda_cfg.get("feature_selection"))
        return EDAConfig(
            train_csv=Path(raw["paths"]["train_csv"]),
            test_csv=Path(raw["paths"]["test_csv"]),
            output_dir=Path(eda_cfg["output_dir"]),
            policy_grain_col=str(eda_cfg.get("policy_grain_col", CONTRACT_COL)),
            random_state=int(eda_cfg.get("random_state", 42)),
            sample_nrows=int(eda_cfg["sample_nrows"]) if eda_cfg.get("sample_nrows") else None,
            export_figures=bool(eda_cfg.get("export_figures", True)),
            max_numeric_correlation_features=int(eda_cfg.get("max_numeric_correlation_features", 120)),
            feature_selection=fs_cfg,
        )


def _resolve_path(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p).resolve()


def _load_data(path: Path, sample_nrows: int | None) -> pd.DataFrame:
    if sample_nrows is None:
        return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, low_memory=False, nrows=sample_nrows)


def _save_plot(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _target_distribution_plot(train_df: pd.DataFrame, out_path: Path) -> None:
    values = train_df[TARGET_CLAIM_COL].fillna(0).astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(values.index.astype(str), values.values)
    ax.set_title("Target distribution: is_claim")
    ax.set_xlabel("is_claim")
    ax.set_ylabel("Count")
    _save_plot(fig, out_path)


def _missing_top_plot(missing_top: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(missing_top.index[::-1], missing_top.values[::-1])
    ax.set_title("Top missing-rate features")
    ax.set_xlabel("Missing rate")
    _save_plot(fig, out_path)


def _rows_per_contract_plot(rows_per_contract: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(1, min(rows_per_contract.max() + 2, 25))
    ax.hist(rows_per_contract.values, bins=bins, edgecolor="black")
    ax.set_title("Rows per contract distribution")
    ax.set_xlabel("Rows per contract")
    ax.set_ylabel("Contracts")
    _save_plot(fig, out_path)


def _claim_amount_plot(train_df: pd.DataFrame, out_path: Path) -> None:
    claims = pd.to_numeric(train_df[TARGET_AMOUNT_COL], errors="coerce").fillna(0.0)
    claims = claims[claims > 0]
    if claims.empty:
        claims = pd.Series([0.0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log1p(claims.values), bins=40, edgecolor="black")
    ax.set_title("Claim amount distribution (log1p, positive only)")
    ax.set_xlabel("log1p(claim_amount)")
    ax.set_ylabel("Count")
    _save_plot(fig, out_path)


def _calc_leakage_watchlist(
    train_df: pd.DataFrame, max_numeric_features: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_numeric = [
        col
        for col in train_df.columns
        if col not in {TARGET_CLAIM_COL, TARGET_AMOUNT_COL, TARGET_COUNT_COL}
        and pd.api.types.is_numeric_dtype(train_df[col])
    ]
    candidate_numeric = candidate_numeric[:max_numeric_features]

    y = train_df[TARGET_CLAIM_COL].fillna(0).astype(float)
    rows: list[dict[str, Any]] = []
    for col in candidate_numeric:
        x = pd.to_numeric(train_df[col], errors="coerce")
        if x.notna().sum() < 30:
            continue
        corr = x.corr(y)
        if pd.notna(corr):
            rows.append(
                {
                    "feature": col,
                    "abs_corr_with_is_claim": float(abs(corr)),
                    "corr_with_is_claim": float(corr),
                }
            )

    watchlist = pd.DataFrame(rows).sort_values("abs_corr_with_is_claim", ascending=False)
    if watchlist.empty:
        watchlist = pd.DataFrame(columns=["feature", "abs_corr_with_is_claim", "corr_with_is_claim"])

    explicit_blacklist = pd.DataFrame({"feature": list(DEFAULT_FORBIDDEN_FEATURE_COLUMNS)})
    explicit_blacklist["rule"] = "forbidden_identifier_or_linkage"
    return watchlist, explicit_blacklist


def _build_eda_summary_md(
    out_path: Path,
    metrics: dict[str, Any],
    recommendations: list[str],
) -> None:
    lines = [
        "# EDA Summary",
        "",
        "## Executive Summary",
        f"- Train shape: {metrics['train_shape']}",
        f"- Test shape: {metrics['test_shape']}",
        f"- Claim rate: {metrics['claim_rate']:.6f}",
        f"- Contracts in train: {metrics['contract_nunique']}",
        f"- Mean rows per contract: {metrics['rows_per_contract_mean']:.4f}",
        "",
        "## Key Findings",
        f"- Train-only target columns: {metrics['train_only_columns']}",
        f"- Top sparse features are concentrated in SCORE_11/SCORE_12 blocks.",
        f"- Driver-level accounting can multiply financial totals if not aggregated by contract.",
        f"- Missing share for claim-related fields is high by design (sparse events).",
        "",
        "## Policy vs Driver",
        f"- Driver-level premium sum: {metrics['driver_level_premium_sum']:.2f}",
        f"- Policy-level premium sum (max per contract): {metrics['policy_level_premium_sum']:.2f}",
        f"- Duplication factor (driver/policy): {metrics['premium_duplication_factor']:.4f}",
        "",
        "## Recommended Actions",
    ]
    for rec in recommendations:
        lines.append(f"- {rec}")

    lines += [
        "",
        "## Artifacts",
        "- tables: `artifacts/eda/tables`",
        "- figures: `artifacts/eda/figures`",
        "- metadata: `artifacts/eda/metadata/eda_profile.json`",
        "- feature selection: `artifacts/eda/feature_selection`",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_eda(config: EDAConfig, project_root: Path) -> dict[str, Any]:
    train_path = _resolve_path(project_root, config.train_csv)
    test_path = _resolve_path(project_root, config.test_csv)
    out_dir = _resolve_path(project_root, config.output_dir)
    tables_dir = ensure_dir(out_dir / "tables")
    figures_dir = ensure_dir(out_dir / "figures")
    metadata_dir = ensure_dir(out_dir / "metadata")
    feature_selection_dir = ensure_dir(out_dir / "feature_selection")

    LOGGER.info("Load train from %s", train_path)
    train_df = _load_data(train_path, config.sample_nrows)
    LOGGER.info("Load test from %s", test_path)
    test_df = _load_data(test_path, config.sample_nrows)

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    train_only_cols = sorted(train_cols - test_cols)
    test_only_cols = sorted(test_cols - train_cols)

    rows_per_contract = train_df[config.policy_grain_col].value_counts()
    policy_df = aggregate_to_policy_level(train_df, contract_col=config.policy_grain_col)

    claim_rate = float(train_df[TARGET_CLAIM_COL].fillna(0).astype(float).mean())
    premium_driver_sum = float(pd.to_numeric(train_df["premium"], errors="coerce").fillna(0.0).sum())
    premium_policy_sum = float(pd.to_numeric(policy_df["premium"], errors="coerce").fillna(0.0).sum())
    premium_dup_factor = premium_driver_sum / premium_policy_sum if premium_policy_sum > 0 else float("nan")

    missing = train_df.isna().mean().sort_values(ascending=False)
    missing_top20 = missing.head(20)
    missing_top20.to_csv(tables_dir / "missing_top20.csv", header=["missing_rate"])

    numeric_summary = train_df.describe(include=[np.number]).T
    numeric_summary.to_csv(tables_dir / "numeric_summary.csv")

    categorical_cols = [c for c in train_df.columns if train_df[c].dtype == "object"]
    cat_card = pd.DataFrame(
        {
            "feature": categorical_cols,
            "nunique": [train_df[c].nunique(dropna=True) for c in categorical_cols],
            "missing_rate": [float(train_df[c].isna().mean()) for c in categorical_cols],
        }
    ).sort_values("nunique", ascending=False)
    cat_card.to_csv(tables_dir / "categorical_cardinality.csv", index=False)

    schema_diff = pd.DataFrame(
        {
            "train_only_columns": pd.Series(train_only_cols),
            "test_only_columns": pd.Series(test_only_cols),
        }
    )
    schema_diff.to_csv(tables_dir / "schema_diff.csv", index=False)

    rows_quantiles = rows_per_contract.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_frame("rows_per_contract")
    rows_quantiles.to_csv(tables_dir / "rows_per_contract_quantiles.csv")

    policy_vs_driver = pd.DataFrame(
        {
            "metric": [
                "driver_level_rows",
                "policy_level_rows",
                "driver_level_premium_sum",
                "policy_level_premium_sum",
                "premium_duplication_factor",
            ],
            "value": [
                len(train_df),
                len(policy_df),
                premium_driver_sum,
                premium_policy_sum,
                premium_dup_factor,
            ],
        }
    )
    policy_vs_driver.to_csv(tables_dir / "policy_vs_driver_kpis.csv", index=False)

    watchlist, explicit_blacklist = _calc_leakage_watchlist(
        train_df=train_df,
        max_numeric_features=config.max_numeric_correlation_features,
    )
    watchlist.to_csv(tables_dir / "leakage_watchlist_numeric_corr.csv", index=False)
    explicit_blacklist.to_csv(tables_dir / "leakage_blacklist_columns.csv", index=False)

    if config.export_figures:
        _target_distribution_plot(train_df, figures_dir / "target_distribution.png")
        _missing_top_plot(missing_top20, figures_dir / "missing_top20.png")
        _rows_per_contract_plot(rows_per_contract, figures_dir / "rows_per_contract.png")
        _claim_amount_plot(train_df, figures_dir / "claim_amount_log_hist.png")

    feature_selection_cfg = config.feature_selection
    feature_selection_cfg.policy_grain_col = config.policy_grain_col
    fs_result = build_feature_selection_spec(policy_df=policy_df, config=feature_selection_cfg)
    fs_paths = save_feature_selection_artifacts(result=fs_result, output_dir=feature_selection_dir)

    feature_report_path = project_root / "reports" / "feature_selection_report.md"
    write_feature_selection_report(feature_report_path, fs_result, fs_paths)

    recommendations = [
        "Keep policy-level as the canonical training layer for financial KPI consistency.",
        "Apply strict coverage and signal filtering for SCORE_11 and SCORE_12 blocks.",
        "Retain explicit missing flags for sparse feature groups.",
        "Keep identifier fields (iin, car_number, unique_id) in the training blacklist.",
        "Track distribution shifts introduced by driver->policy aggregation.",
    ]

    metrics = {
        "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
        "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
        "train_only_columns": train_only_cols,
        "test_only_columns": test_only_cols,
        "claim_rate": claim_rate,
        "contract_nunique": int(train_df[config.policy_grain_col].nunique()),
        "rows_per_contract_mean": float(rows_per_contract.mean()),
        "driver_level_premium_sum": premium_driver_sum,
        "policy_level_premium_sum": premium_policy_sum,
        "premium_duplication_factor": float(premium_dup_factor),
        "top_missing_features": missing_top20.to_dict(),
        "sample_nrows_used": config.sample_nrows,
        "feature_selection": fs_result.summary,
    }

    profile_path = metadata_dir / "eda_profile.json"
    profile_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = project_root / "reports" / "eda_summary.md"
    _build_eda_summary_md(summary_path, metrics, recommendations)

    return {
        "status": "ok",
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "profile_path": str(profile_path),
        "feature_selection_report_path": str(feature_report_path),
        "feature_selection_summary_path": fs_paths["summary_path"],
    }
