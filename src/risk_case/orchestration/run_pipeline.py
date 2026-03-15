from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from risk_case.data.contracts import PreprocessArtifacts
from risk_case.data.io import read_csv
from risk_case.data.policy_aggregation import aggregate_to_policy_level
from risk_case.data.quality_report import build_quality_report
from risk_case.data.validation import validate_dataset
from risk_case.features.feature_store import (
    policy_to_raw_join,
    transform_inference_feature_store,
)
from risk_case.features.preprocessing import (
    PreprocessingConfig,
    build_oof_target_encoding_features,
    fit_preprocessor,
    transform_with_preprocessor,
)
from risk_case.models.benchmark import BenchmarkConfig, run_model_benchmark
from risk_case.models.frequency_severity import FrequencySeverityModel
from risk_case.models.metrics import classification_metrics, severity_metrics
from risk_case.orchestration.logging_utils import close_run_logger, log_stage, setup_run_logger
from risk_case.pricing.artifacts import PricingPolicyArtifact
from risk_case.pricing.evaluator import RetentionConfig, StratifiedPricingConfig, select_best_pricing
from risk_case.pricing.policy import apply_pricing_policy_artifact
from risk_case.settings import CONTRACT_COL, PREMIUM_COL, TARGET_AMOUNT_COL, TARGET_CLAIM_COL, ensure_dir


@dataclass
class RunConfig:
    train_csv: Path
    test_csv: Path | None
    artifacts_dir: Path
    split_test_size: float
    split_random_state: int
    validation_scheme: str
    validation_group_column: str
    validation_time_column: str
    validation_time_holdout_start: str | None
    validation_group_kfold_n_splits: int
    model_max_iter: int
    model_ridge_alpha: float
    pricing_target_lr: float
    pricing_alpha_start: float
    pricing_alpha_stop: float
    pricing_alpha_num: int
    preprocessing: PreprocessingConfig
    benchmark: BenchmarkConfig
    logging_level: str
    pricing_beta_start: float = 1.0
    pricing_beta_stop: float = 1.0
    pricing_beta_num: int = 1
    pricing_target_band_min: float | None = None
    pricing_target_band_max: float | None = None
    ablation_enabled: bool = True
    model_severity_loss: str = "RMSE"
    model_tweedie_variance_power: float = 1.5
    pricing_optimization_method: str = "grid"
    pricing_slsqp_maxiter: int = 200
    pricing_slsqp_ftol: float = 1e-6
    pricing_slsqp_eps: float = 1e-3
    pricing_retention: RetentionConfig = field(default_factory=RetentionConfig)
    pricing_stratified: StratifiedPricingConfig = field(default_factory=StratifiedPricingConfig)
    diagnostics_enabled: bool = True
    diagnostics_deciles: int = 10

    @staticmethod
    def from_json(path: Path) -> "RunConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        paths = raw["paths"]
        split = raw["split"]
        model = raw["model"]
        pricing = raw["pricing"]
        diagnostics_cfg = raw.get("diagnostics", {})
        if not diagnostics_cfg and raw.get("reports"):
            diagnostics_cfg = raw.get("reports", {})
        alpha_grid = pricing["alpha_grid"]
        beta_grid = pricing.get("beta_grid") or {"start": 1.0, "stop": 1.0, "num": 1}
        target_band_raw = pricing.get("target_band") or {}
        pricing_optimization = pricing.get("optimization") or {}
        pricing_slsqp = pricing_optimization.get("slsqp") or {}
        retention_cfg = RetentionConfig.from_dict(pricing.get("retention"))
        stratified_cfg = StratifiedPricingConfig.from_dict(pricing.get("stratified"))
        ablation_cfg = raw.get("ablation", {})
        validation = raw.get("validation", {})
        preprocessing_cfg = PreprocessingConfig.from_dict(raw.get("preprocessing"))
        benchmark_cfg = BenchmarkConfig.from_dict(raw.get("benchmark"))
        logging_cfg = raw.get("logging", {})
        return RunConfig(
            train_csv=Path(paths["train_csv"]),
            test_csv=Path(paths["test_csv"]) if paths.get("test_csv") else None,
            artifacts_dir=Path(paths["artifacts_dir"]),
            split_test_size=float(split["test_size"]),
            split_random_state=int(split["random_state"]),
            validation_scheme=str(validation.get("scheme", "random")),
            validation_group_column=str(validation.get("group_column", CONTRACT_COL)),
            validation_time_column=str(validation.get("time_column", "operation_date")),
            validation_time_holdout_start=validation.get("time_holdout_start"),
            validation_group_kfold_n_splits=int(validation.get("group_kfold_n_splits", 5)),
            model_max_iter=int(model["max_iter"]),
            model_ridge_alpha=float(model["ridge_alpha"]),
            pricing_target_lr=float(pricing["target_lr"]),
            pricing_alpha_start=float(alpha_grid["start"]),
            pricing_alpha_stop=float(alpha_grid["stop"]),
            pricing_alpha_num=int(alpha_grid["num"]),
            preprocessing=preprocessing_cfg,
            benchmark=benchmark_cfg,
            logging_level=str(logging_cfg.get("level", "INFO")),
            pricing_beta_start=float(beta_grid.get("start", 1.0)),
            pricing_beta_stop=float(beta_grid.get("stop", 1.0)),
            pricing_beta_num=int(beta_grid.get("num", 1)),
            pricing_target_band_min=(
                float(target_band_raw["min"]) if isinstance(target_band_raw, dict) and "min" in target_band_raw else None
            ),
            pricing_target_band_max=(
                float(target_band_raw["max"]) if isinstance(target_band_raw, dict) and "max" in target_band_raw else None
            ),
            ablation_enabled=bool(ablation_cfg.get("enabled", True)),
            model_severity_loss=str(model.get("severity_loss", model.get("severity_loss_function", "RMSE"))),
            model_tweedie_variance_power=float(model.get("tweedie_variance_power", 1.5)),
            pricing_optimization_method=str(
                pricing_optimization.get("method", pricing.get("optimization_method", "grid"))
            ).strip().lower(),
            pricing_slsqp_maxiter=int(pricing_slsqp.get("maxiter", 200)),
            pricing_slsqp_ftol=float(pricing_slsqp.get("ftol", 1e-6)),
            pricing_slsqp_eps=float(pricing_slsqp.get("eps", 1e-3)),
            pricing_retention=retention_cfg,
            pricing_stratified=stratified_cfg,
            diagnostics_enabled=bool(diagnostics_cfg.get("enabled", True)),
            diagnostics_deciles=int(diagnostics_cfg.get("deciles", 10)),
        )


def _resolve_path(base: Path, maybe_relative: Path | None) -> Path | None:
    if maybe_relative is None:
        return None
    if maybe_relative.is_absolute():
        return maybe_relative
    return (base / maybe_relative).resolve()


def _resolve_target_band(config: RunConfig) -> tuple[float, float]:
    if config.pricing_target_band_min is not None and config.pricing_target_band_max is not None:
        band_min = float(config.pricing_target_band_min)
        band_max = float(config.pricing_target_band_max)
    elif config.benchmark.constraints:
        band_min = float(config.benchmark.constraints.lr_total_min)
        band_max = float(config.benchmark.constraints.lr_total_max)
    else:
        band_min = float(config.pricing_target_lr - 0.01)
        band_max = float(config.pricing_target_lr + 0.01)

    if band_min > band_max:
        band_min, band_max = band_max, band_min
    return band_min, band_max


def _build_summary(metrics: dict[str, Any], run_id: str) -> str:
    lines = [
        f"# Experiment Summary: {run_id}",
        "",
        "## Data preprocessing",
        f"- Raw rows: {metrics['preprocessing']['raw_rows']}",
        f"- Policy rows: {metrics['preprocessing']['policy_rows']}",
        f"- Feature count: {metrics['preprocessing']['feature_count']}",
        "",
        "## ML metrics",
        f"- AUC: {metrics['ml']['auc']}",
        f"- Gini: {metrics['ml']['gini']}",
        f"- Brier: {metrics['ml']['brier']}",
        f"- Severity RMSE: {metrics['severity']['rmse']}",
        f"- Severity MAE: {metrics['severity']['mae']}",
        "",
        "## Pricing metrics",
        f"- Alpha: {metrics['pricing']['alpha']}",
        f"- Beta: {metrics['pricing'].get('beta')}",
        f"- Optimization method: {metrics['pricing'].get('optimization_method')}",
        f"- Pricing policy kind: {metrics['pricing'].get('pricing_policy_kind')}",
        f"- Pricing policy path: {metrics['pricing'].get('pricing_policy_path')}",
        f"- LR total: {metrics['pricing']['lr_total']}",
        f"- LR group1: {metrics['pricing']['lr_group1']}",
        f"- LR group2: {metrics['pricing']['lr_group2']}",
        f"- Share group1: {metrics['pricing']['share_group1']}",
        f"- Retention enabled: {metrics['pricing'].get('retention_enabled')}",
        f"- Retention rate: {metrics['pricing'].get('retention_rate')}",
        f"- In target band: {metrics['pricing'].get('in_target')}",
        f"- Distance to target: {metrics['pricing'].get('distance_to_target')}",
        f"- Target band: {metrics['pricing'].get('target_band')}",
        f"- Constraint violations: {metrics['pricing']['violations']}",
    ]
    benchmark = metrics.get("benchmark")
    if benchmark:
        lines.extend(
            [
                "",
                "## Model benchmark",
                f"- Enabled: {benchmark.get('enabled')}",
                f"- Winner: {benchmark.get('winner_name')}",
                f"- Selection reason: {benchmark.get('selection_reason')}",
                f"- Candidates total: {benchmark.get('candidates_total')}",
                f"- Candidates ok: {benchmark.get('candidates_ok')}",
                f"- Candidates failed: {benchmark.get('candidates_failed')}",
            ]
        )
    ablation = metrics.get("ablation")
    if ablation:
        lines.extend(
            [
                "",
                "## Ablation diagnostics",
                f"- Path: {ablation.get('path')}",
                f"- Rows total: {ablation.get('rows_total')}",
                f"- Rows ok: {ablation.get('rows_ok')}",
                f"- Rows failed: {ablation.get('rows_failed')}",
            ]
        )
    diagnostics = metrics.get("diagnostics")
    if diagnostics:
        lines.extend(
            [
                "",
                "## Portfolio diagnostics",
                f"- Enabled: {diagnostics.get('enabled')}",
                f"- Deciles: {diagnostics.get('deciles')}",
                f"- Double lift table: {diagnostics.get('double_lift_path')}",
                f"- A/E risk table: {diagnostics.get('ae_risk_path')}",
                f"- A/E segment table: {diagnostics.get('ae_segment_path')}",
            ]
        )
    return "\n".join(lines)


def _update_leaderboard(leaderboard_path: Path, row: dict[str, Any]) -> None:
    row_df = pd.DataFrame([row])
    if leaderboard_path.exists():
        old = pd.read_csv(leaderboard_path)
        merged = pd.concat([old, row_df], ignore_index=True)
    else:
        merged = row_df
    merged.to_csv(leaderboard_path, index=False)


def _save_model(model: Any, path: Path) -> None:
    saver = getattr(model, "save", None)
    if callable(saver):
        saver(path)
        return
    dump(model, path)


def _split_by_group(
    policy_df: pd.DataFrame,
    group_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_column not in policy_df.columns:
        raise ValueError(f"Group split requires column '{group_column}'")

    groups = policy_df[group_column].astype("string").fillna("__missing_group__").astype(str)
    unique_groups = pd.Series(groups.unique(), dtype="string")
    if len(unique_groups) < 2:
        raise ValueError("Group split requires at least two distinct groups")

    train_groups, valid_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_mask = groups.isin(train_groups.astype(str))
    valid_mask = groups.isin(valid_groups.astype(str))
    train_df = policy_df.loc[train_mask].copy()
    valid_df = policy_df.loc[valid_mask].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("Group split produced empty train or valid split")
    return train_df, valid_df


def _split_policy_train_valid(
    policy_df: pd.DataFrame,
    config: RunConfig,
    logger: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scheme = (config.validation_scheme or "random").strip().lower()
    group_col = config.validation_group_column
    time_col = config.validation_time_column

    if scheme == "group_time":
        if time_col in policy_df.columns:
            parsed_time = pd.to_datetime(policy_df[time_col], errors="coerce")
            non_null_time = parsed_time.dropna()
            cutoff: pd.Timestamp | None = None
            if config.validation_time_holdout_start:
                cutoff = pd.to_datetime(config.validation_time_holdout_start, errors="coerce")
            elif not non_null_time.empty:
                quantile = float(max(0.5, min(0.95, 1.0 - config.split_test_size)))
                cutoff = pd.Timestamp(non_null_time.quantile(quantile))

            if cutoff is not None and not pd.isna(cutoff):
                valid_mask = parsed_time >= cutoff
                train_df = policy_df.loc[~valid_mask].copy()
                valid_df = policy_df.loc[valid_mask].copy()
                if group_col in policy_df.columns and not train_df.empty and not valid_df.empty:
                    train_groups = set(train_df[group_col].astype(str))
                    overlap_mask = valid_df[group_col].astype(str).isin(train_groups)
                    if overlap_mask.any():
                        valid_df = valid_df.loc[~overlap_mask].copy()

                if not train_df.empty and not valid_df.empty:
                    logger.info(
                        "Validation split scheme=group_time cutoff=%s train_rows=%d valid_rows=%d",
                        cutoff,
                        len(train_df),
                        len(valid_df),
                    )
                    return (
                        train_df,
                        valid_df,
                        {
                            "scheme": "group_time",
                            "group_column": group_col,
                            "time_column": time_col,
                            "time_holdout_start": str(cutoff),
                            "train_rows": int(len(train_df)),
                            "valid_rows": int(len(valid_df)),
                        },
                    )
                logger.warning("group_time split fallback triggered because one split is empty after filtering")
            else:
                logger.warning("group_time split fallback triggered because cutoff could not be derived")
        else:
            logger.warning("group_time split fallback: missing time column %s", time_col)

        train_df, valid_df = _split_by_group(
            policy_df=policy_df,
            group_column=group_col,
            test_size=config.split_test_size,
            random_state=config.split_random_state,
        )
        logger.info(
            "Validation split fallback scheme=group train_rows=%d valid_rows=%d",
            len(train_df),
            len(valid_df),
        )
        return (
            train_df,
            valid_df,
            {
                "scheme": "group",
                "group_column": group_col,
                "time_column": time_col,
                "time_holdout_start": None,
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
            },
        )

    if scheme == "group":
        train_df, valid_df = _split_by_group(
            policy_df=policy_df,
            group_column=group_col,
            test_size=config.split_test_size,
            random_state=config.split_random_state,
        )
        logger.info(
            "Validation split scheme=group train_rows=%d valid_rows=%d",
            len(train_df),
            len(valid_df),
        )
        return (
            train_df,
            valid_df,
            {
                "scheme": "group",
                "group_column": group_col,
                "time_column": time_col,
                "time_holdout_start": None,
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
            },
        )

    y = policy_df[TARGET_CLAIM_COL].fillna(0).astype(int) if TARGET_CLAIM_COL in policy_df.columns else None
    train_df, valid_df = train_test_split(
        policy_df,
        test_size=config.split_test_size,
        random_state=config.split_random_state,
        stratify=y if y is not None and len(np.unique(y)) > 1 else None,
    )
    logger.info(
        "Validation split scheme=random train_rows=%d valid_rows=%d",
        len(train_df),
        len(valid_df),
    )
    return (
        train_df,
        valid_df,
        {
            "scheme": "random",
            "group_column": group_col,
            "time_column": time_col,
            "time_holdout_start": None,
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
        },
    )


def _evaluate_frequency_severity_candidate(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: RunConfig,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    target_band: tuple[float, float],
    pricing_method: str,
    pricing_retention: RetentionConfig,
    pricing_slsqp_options: dict[str, Any],
    pricing_stratified_config: StratifiedPricingConfig,
) -> tuple[dict[str, Any], dict[str, Any], float, float, Any]:
    model = FrequencySeverityModel(
        max_iter=config.model_max_iter,
        ridge_alpha=config.model_ridge_alpha,
    ).fit(train_df)
    valid_pred = model.predict(valid_df)

    ml = classification_metrics(valid_df[TARGET_CLAIM_COL].fillna(0).astype(int).values, valid_pred["p_claim"].values)
    pos_mask = valid_df[TARGET_CLAIM_COL].fillna(0).astype(int) > 0
    severity = severity_metrics(
        y_true=np.log1p(pd.to_numeric(valid_df.loc[pos_mask, TARGET_AMOUNT_COL], errors="coerce").fillna(0.0).values),
        y_pred=np.log1p(valid_pred.loc[pos_mask, "expected_severity"].clip(lower=0.0).values),
    )

    alpha, beta, _, pricing_eval = select_best_pricing(
        df=valid_df,
        expected_loss=valid_pred["expected_loss"],
        target_lr=config.pricing_target_lr,
        alpha_grid=alpha_grid,
        beta_grid=beta_grid,
        target_band=target_band,
        method=pricing_method,
        retention_config=pricing_retention,
        slsqp_options=pricing_slsqp_options,
        stratified_config=pricing_stratified_config,
    )
    return ml, severity, float(alpha), float(beta), pricing_eval


def _run_ablation_diagnostics(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    preprocessor: Any,
    config: RunConfig,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    target_band: tuple[float, float],
    pricing_method: str,
    pricing_retention: RetentionConfig,
    pricing_slsqp_options: dict[str, Any],
    pricing_stratified_config: StratifiedPricingConfig,
    run_dir: Path,
    logger: Any,
) -> dict[str, Any]:
    protected_columns = {
        config.preprocessing.grain,
        TARGET_CLAIM_COL,
        TARGET_AMOUNT_COL,
        "claim_cnt",
    }

    feature_families = [
        ("full_model", []),
        ("drop_date_features", list(preprocessor.date_feature_columns)),
        ("drop_target_encoding", [f"{col}_te" for col in preprocessor.target_encoding_maps.keys()]),
        ("drop_frequency_encoding", [f"{col}_freq" for col in preprocessor.frequency_encoding_maps.keys()]),
        ("drop_interactions", list(preprocessor.interaction_feature_columns)),
        ("drop_missing_aggregates", list(preprocessor.missing_aggregate_definitions.keys())),
    ]

    rows: list[dict[str, Any]] = []
    full_policy_score: float | None = None
    for ablation_name, family_columns in feature_families:
        drop_columns = [col for col in family_columns if col in train_df.columns and col not in protected_columns]
        train_variant = train_df.drop(columns=drop_columns, errors="ignore")
        valid_variant = valid_df.drop(columns=drop_columns, errors="ignore")

        feature_columns = [col for col in train_variant.columns if col not in protected_columns]
        if not feature_columns:
            rows.append(
                {
                    "ablation_name": ablation_name,
                    "status": "failed",
                    "dropped_columns": ";".join(drop_columns),
                    "dropped_count": int(len(drop_columns)),
                    "error": "no_features_left_after_ablation",
                }
            )
            continue

        try:
            ml, severity, alpha, beta, pricing_eval = _evaluate_frequency_severity_candidate(
                train_df=train_variant,
                valid_df=valid_variant,
                config=config,
                alpha_grid=alpha_grid,
                beta_grid=beta_grid,
                target_band=target_band,
                pricing_method=pricing_method,
                pricing_retention=pricing_retention,
                pricing_slsqp_options=pricing_slsqp_options,
                pricing_stratified_config=pricing_stratified_config,
            )
            pricing_dict = pricing_eval.to_dict()
            policy_score = float(pricing_dict.get("policy_score", getattr(pricing_eval, "score", float("nan"))))
            if ablation_name == "full_model":
                full_policy_score = policy_score

            rows.append(
                {
                    "ablation_name": ablation_name,
                    "status": "ok",
                    "dropped_columns": ";".join(drop_columns),
                    "dropped_count": int(len(drop_columns)),
                    "alpha": alpha,
                    "beta": beta,
                    "policy_score": policy_score,
                    "lr_total": float(pricing_dict.get("lr_total", pricing_eval.lr_total)),
                    "lr_group1": float(pricing_dict.get("lr_group1", pricing_eval.lr_group1)),
                    "lr_group2": float(pricing_dict.get("lr_group2", pricing_eval.lr_group2)),
                    "share_group1": float(pricing_dict.get("share_group1", pricing_eval.share_group1)),
                    "violations": int(pricing_dict.get("violations", pricing_eval.violations)),
                    "auc": ml.get("auc"),
                    "gini": ml.get("gini"),
                    "brier": ml.get("brier"),
                    "severity_rmse": severity.get("rmse"),
                    "severity_mae": severity.get("mae"),
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostics should not break run
            rows.append(
                {
                    "ablation_name": ablation_name,
                    "status": "failed",
                    "dropped_columns": ";".join(drop_columns),
                    "dropped_count": int(len(drop_columns)),
                    "error": str(exc),
                }
            )

    for row in rows:
        if row.get("status") == "ok" and full_policy_score is not None:
            row["delta_vs_full_policy_score"] = float(row["policy_score"]) - full_policy_score
        else:
            row["delta_vs_full_policy_score"] = None

    ablation_path = run_dir / "ablation_results.csv"
    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(ablation_path, index=False)
    logger.info("Ablation diagnostics saved: %s rows=%d", ablation_path, len(ablation_df))

    ok_rows = [item for item in rows if item.get("status") == "ok"]
    return {
        "enabled": True,
        "path": str(ablation_path),
        "rows_total": int(len(rows)),
        "rows_ok": int(len(ok_rows)),
        "rows_failed": int(len(rows) - len(ok_rows)),
    }


def _safe_decile_buckets(series: pd.Series, deciles: int) -> pd.Series:
    if len(series) == 0:
        return pd.Series(dtype=int, index=series.index)
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.ones(len(series), dtype=int), index=series.index)
    filled = values.fillna(values.median())
    unique = int(filled.nunique(dropna=True))
    if unique <= 1:
        return pd.Series(np.ones(len(series), dtype=int), index=series.index)
    q = int(max(2, min(deciles, len(filled), unique)))
    ranked = filled.rank(method="first")
    buckets = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
    bucket_series = pd.Series(buckets, index=series.index).fillna(0).astype(int) + 1
    return bucket_series


def _build_portfolio_diagnostics(
    valid_out: pd.DataFrame,
    run_dir: Path,
    enabled: bool,
    deciles: int,
    logger: Any,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "enabled": bool(enabled),
        "deciles": int(max(2, deciles)),
        "double_lift_path": None,
        "ae_risk_path": None,
        "ae_segment_path": None,
        "double_lift_cells": 0,
        "max_abs_ae_risk_decile": None,
    }
    if not enabled:
        return diagnostics
    if valid_out.empty:
        diagnostics["enabled"] = False
        diagnostics["error"] = "valid_out_empty"
        return diagnostics

    required_cols = {"expected_loss", "new_premium", "price_delta_pct"}
    missing = sorted(required_cols.difference(valid_out.columns))
    if missing:
        diagnostics["enabled"] = False
        diagnostics["error"] = f"missing_columns:{','.join(missing)}"
        return diagnostics

    report_df = valid_out.copy()
    if TARGET_AMOUNT_COL not in report_df.columns:
        report_df[TARGET_AMOUNT_COL] = 0.0
    report_df[TARGET_AMOUNT_COL] = pd.to_numeric(report_df[TARGET_AMOUNT_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    report_df["expected_loss"] = pd.to_numeric(report_df["expected_loss"], errors="coerce").fillna(0.0).clip(lower=0.0)
    report_df["new_premium"] = pd.to_numeric(report_df["new_premium"], errors="coerce").fillna(0.0).clip(lower=0.0)
    report_df["price_delta_pct"] = pd.to_numeric(report_df["price_delta_pct"], errors="coerce").fillna(0.0)

    bucket_count = int(max(2, min(deciles, len(report_df))))
    report_df["risk_decile"] = _safe_decile_buckets(report_df["expected_loss"], bucket_count)
    report_df["delta_decile"] = _safe_decile_buckets(report_df["price_delta_pct"], bucket_count)

    double_lift_df = (
        report_df.groupby(["risk_decile", "delta_decile"], dropna=False)
        .agg(
            policies=("expected_loss", "size"),
            premium_sum=("new_premium", "sum"),
            expected_loss_sum=("expected_loss", "sum"),
            claim_amount_sum=(TARGET_AMOUNT_COL, "sum"),
            avg_price_delta=("price_delta_pct", "mean"),
            avg_expected_loss=("expected_loss", "mean"),
        )
        .reset_index()
        .sort_values(["risk_decile", "delta_decile"])
    )
    double_lift_df["actual_lr"] = np.where(
        double_lift_df["premium_sum"] > 0,
        double_lift_df["claim_amount_sum"] / double_lift_df["premium_sum"],
        0.0,
    )
    double_lift_df["expected_lr"] = np.where(
        double_lift_df["premium_sum"] > 0,
        double_lift_df["expected_loss_sum"] / double_lift_df["premium_sum"],
        0.0,
    )
    double_lift_df["ae_ratio"] = np.where(
        double_lift_df["expected_loss_sum"] > 0,
        double_lift_df["claim_amount_sum"] / double_lift_df["expected_loss_sum"],
        0.0,
    )
    double_lift_path = run_dir / "double_lift_table.csv"
    double_lift_df.to_csv(double_lift_path, index=False)

    ae_risk_df = (
        report_df.groupby(["risk_decile"], dropna=False)
        .agg(
            policies=("expected_loss", "size"),
            premium_sum=("new_premium", "sum"),
            expected_loss_sum=("expected_loss", "sum"),
            claim_amount_sum=(TARGET_AMOUNT_COL, "sum"),
        )
        .reset_index()
        .sort_values("risk_decile")
    )
    ae_risk_df["actual_lr"] = np.where(ae_risk_df["premium_sum"] > 0, ae_risk_df["claim_amount_sum"] / ae_risk_df["premium_sum"], 0.0)
    ae_risk_df["expected_lr"] = np.where(
        ae_risk_df["premium_sum"] > 0,
        ae_risk_df["expected_loss_sum"] / ae_risk_df["premium_sum"],
        0.0,
    )
    ae_risk_df["ae_ratio"] = np.where(
        ae_risk_df["expected_loss_sum"] > 0,
        ae_risk_df["claim_amount_sum"] / ae_risk_df["expected_loss_sum"],
        0.0,
    )
    ae_risk_path = run_dir / "ae_by_risk_decile.csv"
    ae_risk_df.to_csv(ae_risk_path, index=False)

    segment_frames: list[pd.DataFrame] = []
    segment_columns = [col for col in ["region_name", "vehicle_type_name", "mark", "model", "bonus_malus"] if col in report_df.columns]
    for column in segment_columns:
        segment_df = report_df.copy()
        segment_df[column] = segment_df[column].astype("string").fillna("missing").astype(str)
        top_values = segment_df[column].value_counts(dropna=False).head(20).index.astype(str)
        segment_df = segment_df.loc[segment_df[column].isin(set(top_values))].copy()
        if segment_df.empty:
            continue
        agg = (
            segment_df.groupby(column, dropna=False)
            .agg(
                policies=("expected_loss", "size"),
                premium_sum=("new_premium", "sum"),
                expected_loss_sum=("expected_loss", "sum"),
                claim_amount_sum=(TARGET_AMOUNT_COL, "sum"),
            )
            .reset_index()
            .rename(columns={column: "segment_value"})
        )
        agg["segment_column"] = column
        agg["actual_lr"] = np.where(agg["premium_sum"] > 0, agg["claim_amount_sum"] / agg["premium_sum"], 0.0)
        agg["expected_lr"] = np.where(agg["premium_sum"] > 0, agg["expected_loss_sum"] / agg["premium_sum"], 0.0)
        agg["ae_ratio"] = np.where(agg["expected_loss_sum"] > 0, agg["claim_amount_sum"] / agg["expected_loss_sum"], 0.0)
        segment_frames.append(
            agg[
                [
                    "segment_column",
                    "segment_value",
                    "policies",
                    "premium_sum",
                    "expected_loss_sum",
                    "claim_amount_sum",
                    "actual_lr",
                    "expected_lr",
                    "ae_ratio",
                ]
            ]
        )

    ae_segment_df = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame(
        columns=[
            "segment_column",
            "segment_value",
            "policies",
            "premium_sum",
            "expected_loss_sum",
            "claim_amount_sum",
            "actual_lr",
            "expected_lr",
            "ae_ratio",
        ]
    )
    ae_segment_path = run_dir / "ae_by_segment.csv"
    ae_segment_df.to_csv(ae_segment_path, index=False)

    diagnostics.update(
        {
            "double_lift_path": str(double_lift_path),
            "ae_risk_path": str(ae_risk_path),
            "ae_segment_path": str(ae_segment_path),
            "double_lift_cells": int(len(double_lift_df)),
            "max_abs_ae_risk_decile": (
                float((ae_risk_df["ae_ratio"] - 1.0).abs().max()) if not ae_risk_df.empty else None
            ),
        }
    )
    logger.info(
        "Portfolio diagnostics saved: double_lift_cells=%d risk_deciles=%d",
        len(double_lift_df),
        ae_risk_df["risk_decile"].nunique() if not ae_risk_df.empty else 0,
    )
    return diagnostics


def run_experiment(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parents[1] if config_path.parent.name == "configs" else config_path.parent
    config = RunConfig.from_json(config_path)

    if config.preprocessing.feature_whitelist_path:
        config.preprocessing.feature_whitelist_path = str(
            _resolve_path(project_root, Path(config.preprocessing.feature_whitelist_path))
        )
    if config.preprocessing.feature_droplist_path:
        config.preprocessing.feature_droplist_path = str(
            _resolve_path(project_root, Path(config.preprocessing.feature_droplist_path))
        )

    train_path = _resolve_path(project_root, config.train_csv)
    test_path = _resolve_path(project_root, config.test_csv)
    artifacts_root = _resolve_path(project_root, config.artifacts_dir)
    assert train_path is not None
    assert artifacts_root is not None

    ensure_dir(artifacts_root)
    ensure_dir(artifacts_root / "runs")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(artifacts_root / "runs" / run_id)
    preprocess_dir = ensure_dir(run_dir / "preprocessed")
    logger = setup_run_logger(run_dir=run_dir, level=config.logging_level)

    try:
        logger.info("Run initialized: run_id=%s", run_id)
        logger.info("Train path: %s", train_path)
        logger.info("Test path: %s", test_path)

        with log_stage(logger, "ingest_train_data"):
            raw_train_df = read_csv(train_path)
            logger.info("Raw train loaded: rows=%d cols=%d", len(raw_train_df), len(raw_train_df.columns))

        with log_stage(logger, "validate_raw_data"):
            validation = validate_dataset(raw_train_df)
            if not validation.ok:
                raise ValueError(f"Data validation failed: {validation.errors}")
            logger.info("Validation warnings: %d", len(validation.warnings))

        with log_stage(logger, "split_train_valid"):
            policy_train_df = aggregate_to_policy_level(raw_train_df, contract_col=config.preprocessing.grain)
            train_policy_split, valid_policy_split, split_meta = _split_policy_train_valid(
                policy_df=policy_train_df,
                config=config,
                logger=logger,
            )
            logger.info("Train policy split rows=%d; valid policy split rows=%d", len(train_policy_split), len(valid_policy_split))

        with log_stage(logger, "preprocess_train_data"):
            preprocessor = fit_preprocessor(train_policy_split, config.preprocessing)
            train_split = transform_with_preprocessor(train_policy_split, preprocessor)
            valid_split = transform_with_preprocessor(valid_policy_split, preprocessor)
            processed_train_df = transform_with_preprocessor(policy_train_df, preprocessor)

            if config.preprocessing.target_encoding_enabled and preprocessor.target_encoding_maps:
                with log_stage(logger, "oof_target_encoding_train"):
                    oof_target_encoding = build_oof_target_encoding_features(
                        df=train_policy_split,
                        state=preprocessor,
                        target_column=TARGET_CLAIM_COL,
                        n_splits=config.validation_group_kfold_n_splits,
                        random_state=config.split_random_state,
                        group_column=config.validation_group_column if "group" in split_meta.get("scheme", "") else None,
                    )
                    for column in oof_target_encoding.columns:
                        if column in train_split.columns:
                            train_split[column] = oof_target_encoding[column].values
                        else:
                            train_split[column] = oof_target_encoding[column].values
                    logger.info("OOF target encoding columns applied: %d", len(oof_target_encoding.columns))

            dataset_path = preprocess_dir / "train_policy_preprocessed.csv"
            train_split_path = preprocess_dir / "train_split_preprocessed.csv"
            valid_split_path = preprocess_dir / "valid_split_preprocessed.csv"
            metadata_path = preprocess_dir / "preprocess_metadata.json"
            quality_report_path = preprocess_dir / "quality_report.json"

            processed_train_df.to_csv(dataset_path, index=False)
            train_split.to_csv(train_split_path, index=False)
            valid_split.to_csv(valid_split_path, index=False)

            quality_report = build_quality_report(
                raw_df=raw_train_df,
                policy_df=policy_train_df,
                processed_df=processed_train_df,
                feature_columns=preprocessor.feature_columns,
                target_columns=config.preprocessing.target_columns,
                contract_col=config.preprocessing.grain,
            )
            quality_report_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")

            metadata_payload: dict[str, Any] = {
                "config": {
                    "grain": config.preprocessing.grain,
                    "target_columns": config.preprocessing.target_columns,
                    "drop_columns": config.preprocessing.drop_columns,
                    "forbidden_feature_columns": config.preprocessing.forbidden_feature_columns,
                    "winsorize": {
                        "low": config.preprocessing.winsorize_low,
                        "high": config.preprocessing.winsorize_high,
                    },
                    "missing": {
                        "numeric_default": config.preprocessing.numeric_default_strategy,
                        "financial_fill": config.preprocessing.financial_fill_value,
                        "add_missing_flags": config.preprocessing.add_missing_flags,
                        "add_missing_aggregates": config.preprocessing.add_missing_aggregates,
                        "missing_flag_threshold": config.preprocessing.missing_flag_threshold,
                    },
                    "feature_pruning": {
                        "enabled": config.preprocessing.feature_pruning_enabled,
                        "drop_exact_duplicates": config.preprocessing.feature_pruning_drop_exact_duplicates,
                        "drop_missing_share": config.preprocessing.feature_pruning_drop_missing_share,
                        "corr_threshold": config.preprocessing.feature_pruning_corr_threshold,
                    },
                    "drift_pruning": {
                        "enabled": config.preprocessing.drift_pruning_enabled,
                        "time_column": config.preprocessing.drift_pruning_time_column,
                        "reference_share": config.preprocessing.drift_pruning_reference_share,
                        "psi_threshold": config.preprocessing.drift_pruning_psi_threshold,
                        "bins": config.preprocessing.drift_pruning_bins,
                        "min_rows": config.preprocessing.drift_pruning_min_rows,
                        "exclude_columns": config.preprocessing.drift_pruning_exclude_columns,
                        "exclude_patterns": config.preprocessing.drift_pruning_exclude_patterns,
                    },
                    "categorical": {
                        "rare_threshold": config.preprocessing.rare_category_threshold,
                        "rare_min_count": config.preprocessing.rare_category_min_count,
                    },
                    "transforms": {
                        "log1p_columns": config.preprocessing.log1p_columns,
                    },
                    "date_features": {
                        "columns": config.preprocessing.date_columns,
                        "features": config.preprocessing.date_features,
                    },
                    "target_encoding": {
                        "enabled": config.preprocessing.target_encoding_enabled,
                        "columns": config.preprocessing.target_encoding_columns,
                        "smoothing": config.preprocessing.target_encoding_smoothing,
                        "min_samples_leaf": config.preprocessing.target_encoding_min_samples_leaf,
                        "noise_std": config.preprocessing.target_encoding_noise_std,
                    },
                    "freq_encoding": {
                        "enabled": config.preprocessing.frequency_encoding_enabled,
                        "columns": config.preprocessing.frequency_encoding_columns,
                    },
                    "interaction_features": {
                        "enabled": config.preprocessing.interaction_features_enabled,
                        "definitions": config.preprocessing.interaction_features,
                    },
                    "interaction_features_mvp": {
                        "enabled": config.preprocessing.interaction_features_mvp_enabled,
                        "definitions": config.preprocessing.interaction_features_mvp_definitions,
                        "max_features": config.preprocessing.interaction_features_mvp_max_features,
                        "corr_filter_threshold": config.preprocessing.interaction_features_mvp_corr_filter_threshold,
                        "psi_filter_threshold": config.preprocessing.interaction_features_mvp_psi_filter_threshold,
                        "require_business_whitelist": config.preprocessing.interaction_features_mvp_require_business_whitelist,
                    },
                    "feature_generation_version": config.preprocessing.feature_generation_version,
                    "model": {
                        "max_iter": config.model_max_iter,
                        "ridge_alpha": config.model_ridge_alpha,
                        "severity_loss": config.model_severity_loss,
                        "tweedie_variance_power": config.model_tweedie_variance_power,
                    },
                    "pricing": {
                        "target_lr": config.pricing_target_lr,
                        "optimization_method": config.pricing_optimization_method,
                        "retention": config.pricing_retention.to_dict(),
                        "stratified": config.pricing_stratified.to_dict(),
                        "slsqp": {
                            "maxiter": config.pricing_slsqp_maxiter,
                            "ftol": config.pricing_slsqp_ftol,
                            "eps": config.pricing_slsqp_eps,
                        },
                    },
                },
                "validation": split_meta,
                "preprocessor_state": preprocessor.to_dict(),
                "feature_columns": preprocessor.feature_columns,
                "target_columns": config.preprocessing.target_columns,
                "row_count": int(len(processed_train_df)),
                "train_split_row_count": int(len(train_split)),
                "valid_split_row_count": int(len(valid_split)),
            }
            metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            preprocess_artifacts = PreprocessArtifacts(
                dataset_path=dataset_path,
                metadata_path=metadata_path,
                quality_report_path=quality_report_path,
                feature_columns=preprocessor.feature_columns,
                target_columns=config.preprocessing.target_columns,
                row_count=int(len(processed_train_df)),
            )
            logger.info(
                "Processed datasets ready: full=%d train=%d valid=%d features=%d",
                len(processed_train_df),
                len(train_split),
                len(valid_split),
                len(preprocess_artifacts.feature_columns),
            )

        if TARGET_CLAIM_COL not in train_split.columns:
            raise ValueError(f"{TARGET_CLAIM_COL} is missing after preprocessing")
        if TARGET_AMOUNT_COL not in train_split.columns:
            raise ValueError(f"{TARGET_AMOUNT_COL} is missing after preprocessing")

        alpha_grid = np.linspace(
            config.pricing_alpha_start,
            config.pricing_alpha_stop,
            config.pricing_alpha_num,
        )
        beta_grid = np.linspace(
            config.pricing_beta_start,
            config.pricing_beta_stop,
            max(1, config.pricing_beta_num),
        )
        target_band = _resolve_target_band(config)
        pricing_method = (config.pricing_optimization_method or "grid").strip().lower()
        pricing_slsqp_options = {
            "maxiter": int(config.pricing_slsqp_maxiter),
            "ftol": float(config.pricing_slsqp_ftol),
            "eps": float(config.pricing_slsqp_eps),
        }
        pricing_retention = config.pricing_retention
        pricing_stratified = config.pricing_stratified
        pricing_policy_path = run_dir / "pricing_policy.json"
        if "catboost_freq_sev" in config.benchmark.candidates:
            catboost_params = config.benchmark.candidate_params.setdefault("catboost_freq_sev", {})
            catboost_params.setdefault("severity_loss_function", config.model_severity_loss)
            catboost_params.setdefault("tweedie_variance_power", config.model_tweedie_variance_power)
        if "catboost_dep_freq_sev" in config.benchmark.candidates:
            dep_params = config.benchmark.candidate_params.setdefault("catboost_dep_freq_sev", {})
            dep_params.setdefault("severity_loss_function", config.model_severity_loss)
            dep_params.setdefault("tweedie_variance_power", config.model_tweedie_variance_power)
            dep_params.setdefault("dep_oof_folds", 5)
            dep_params.setdefault("dep_frequency_signal_name", "freq_risk_signal")
            dep_params.setdefault("dep_use_frequency_signal", True)

        benchmark_metrics: dict[str, Any]
        winner_name: str
        selection_reason: str

        if config.benchmark.enabled:
            with log_stage(logger, "benchmark_models"):
                benchmark_result = run_model_benchmark(
                    train_df=train_split,
                    valid_df=valid_split,
                    benchmark_config=config.benchmark,
                    pricing_target_lr=config.pricing_target_lr,
                    pricing_alpha_grid=alpha_grid,
                    pricing_beta_grid=beta_grid,
                    pricing_target_band=target_band,
                    model_max_iter=config.model_max_iter,
                    model_ridge_alpha=config.model_ridge_alpha,
                    pricing_optimization_method=pricing_method,
                    pricing_retention=pricing_retention,
                    pricing_slsqp_options=pricing_slsqp_options,
                    pricing_stratified_config=pricing_stratified,
                    logger=logger,
                )
                model = benchmark_result.winner_model
                valid_pred = benchmark_result.winner_valid_pred
                best_alpha = benchmark_result.winner_alpha
                best_beta = benchmark_result.winner_beta
                best_premium = benchmark_result.winner_premium
                best_eval = benchmark_result.winner_pricing_eval
                ml = benchmark_result.winner_ml
                severity = benchmark_result.winner_severity
                winner_name = benchmark_result.winner_name
                selection_reason = benchmark_result.selection_reason

                benchmark_dir = ensure_dir(run_dir / "benchmark")
                results_df = pd.DataFrame([item.to_record() for item in benchmark_result.results])
                benchmark_results_csv = benchmark_dir / "results.csv"
                benchmark_results_json = benchmark_dir / "results.json"
                benchmark_winner_json = benchmark_dir / "winner.json"
                benchmark_failed_json = benchmark_dir / "failed_candidates.json"

                results_df.to_csv(benchmark_results_csv, index=False)
                benchmark_results_json.write_text(
                    json.dumps(benchmark_result.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                benchmark_winner_json.write_text(
                    json.dumps(
                        {
                            "winner_name": winner_name,
                            "selection_reason": selection_reason,
                            "selection_metric": config.benchmark.selection_metric,
                            "alpha": best_alpha,
                            "beta": best_beta,
                            "pricing_policy_kind": (
                                benchmark_result.winner_pricing_eval.pricing_policy.kind
                                if benchmark_result.winner_pricing_eval.pricing_policy is not None
                                else "scalar"
                            ),
                            "pricing_policy_path": str(pricing_policy_path),
                            "pricing_policy": (
                                benchmark_result.winner_pricing_eval.pricing_policy.to_summary()
                                if benchmark_result.winner_pricing_eval.pricing_policy is not None
                                else None
                            ),
                            "pricing": best_eval.to_dict(),
                            "ml": ml,
                            "severity": severity,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                failed_candidates = [item.to_dict() for item in benchmark_result.results if item.status != "ok"]
                benchmark_failed_json.write_text(
                    json.dumps(failed_candidates, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                candidates_ok = sum(1 for item in benchmark_result.results if item.status == "ok")
                benchmark_metrics = {
                    "enabled": True,
                    "winner_name": winner_name,
                    "selection_reason": selection_reason,
                    "selection_metric": config.benchmark.selection_metric,
                    "stability_penalty": config.benchmark.stability_penalty,
                    "must_pass_constraints": config.benchmark.must_pass_constraints,
                    "fallback_strategy": config.benchmark.fallback_strategy,
                    "fallback_candidate": config.benchmark.fallback_candidate,
                    "constraints": config.benchmark.constraints.to_dict(),
                    "calibration": config.benchmark.calibration.to_dict(),
                    "pricing_optimization_method": pricing_method,
                    "retention": pricing_retention.to_dict(),
                    "stratified": pricing_stratified.to_dict(),
                    "pricing_policy_path": str(pricing_policy_path),
                    "pricing_policy_kind": (
                        benchmark_result.winner_pricing_eval.pricing_policy.kind
                        if benchmark_result.winner_pricing_eval.pricing_policy is not None
                        else "scalar"
                    ),
                    "candidates_total": len(benchmark_result.results),
                    "candidates_ok": candidates_ok,
                    "candidates_failed": len(benchmark_result.results) - candidates_ok,
                    "results_csv_path": str(benchmark_results_csv),
                    "results_json_path": str(benchmark_results_json),
                    "winner_json_path": str(benchmark_winner_json),
                    "failed_json_path": str(benchmark_failed_json),
                }
                logger.info(
                    "Benchmark winner=%s reason=%s metric=%s",
                    winner_name,
                    selection_reason,
                    config.benchmark.selection_metric,
                )
        else:
            with log_stage(logger, "train_model"):
                model = FrequencySeverityModel(
                    max_iter=config.model_max_iter,
                    ridge_alpha=config.model_ridge_alpha,
                ).fit(train_split)

            with log_stage(logger, "predict_valid_and_metrics"):
                valid_pred = model.predict(valid_split)
                ml = classification_metrics(valid_split[TARGET_CLAIM_COL].fillna(0).values, valid_pred["p_claim"].values)
                pos_mask = valid_split[TARGET_CLAIM_COL].fillna(0).astype(int) > 0
                severity = severity_metrics(
                    y_true=np.log1p(
                        pd.to_numeric(valid_split.loc[pos_mask, TARGET_AMOUNT_COL], errors="coerce").fillna(0.0).values
                    ),
                    y_pred=np.log1p(valid_pred.loc[pos_mask, "expected_severity"].clip(lower=0.0).values),
                )
                logger.info("ML metrics: AUC=%s Gini=%s Brier=%s", ml.get("auc"), ml.get("gini"), ml.get("brier"))

            with log_stage(logger, "pricing_optimization"):
                best_alpha, best_beta, best_premium, best_eval = select_best_pricing(
                    df=valid_split,
                    expected_loss=valid_pred["expected_loss"],
                    target_lr=config.pricing_target_lr,
                    alpha_grid=alpha_grid,
                    beta_grid=beta_grid,
                    target_band=target_band,
                    method=pricing_method,
                    retention_config=pricing_retention,
                    slsqp_options=pricing_slsqp_options,
                    stratified_config=pricing_stratified,
                )
                logger.info(
                    "Pricing best alpha=%.6f beta=%.6f LR_total=%.6f share_group1=%.6f",
                    best_alpha,
                    best_beta,
                    best_eval.lr_total,
                    best_eval.share_group1,
                )
            winner_name = "baseline_freq_sev"
            selection_reason = "benchmark_disabled"
            benchmark_metrics = {
                "enabled": False,
                "winner_name": winner_name,
                "selection_reason": selection_reason,
                "selection_metric": "policy_score",
                "stability_penalty": config.benchmark.stability_penalty,
                "must_pass_constraints": config.benchmark.must_pass_constraints,
                "fallback_strategy": config.benchmark.fallback_strategy,
                "fallback_candidate": config.benchmark.fallback_candidate,
                "constraints": config.benchmark.constraints.to_dict(),
                "pricing_optimization_method": pricing_method,
                "retention": pricing_retention.to_dict(),
                "stratified": pricing_stratified.to_dict(),
                "candidates_total": 1,
                "candidates_ok": 1,
                "candidates_failed": 0,
            }

        with log_stage(logger, "save_model_and_valid_predictions"):
            pricing_policy = best_eval.pricing_policy or PricingPolicyArtifact.scalar(
                alpha=float(best_alpha),
                beta=float(best_beta),
                method=pricing_method,
            )
            pricing_policy.save(pricing_policy_path)
            valid_out = valid_split.copy()
            valid_out["p_claim"] = valid_pred["p_claim"]
            valid_out["expected_severity"] = valid_pred["expected_severity"]
            valid_out["expected_loss"] = valid_pred["expected_loss"]
            valid_out["new_premium"] = best_premium
            valid_out["price_delta_pct"] = np.where(
                valid_out[PREMIUM_COL] > 0,
                valid_out["new_premium"] / valid_out[PREMIUM_COL] - 1.0,
                0.0,
            )
            valid_out.to_csv(run_dir / "valid_predictions.csv", index=False)
            model_path = run_dir / "model.joblib"
            _save_model(model, model_path)

        # --- SHAP Explainability Report ---
        shap_metrics: dict[str, Any] = {"status": "skipped"}
        try:
            from risk_case.explainability.shap_analysis import generate_shap_report

            with log_stage(logger, "shap_explainability"):
                shap_dir = ensure_dir(run_dir / "shap")
                shap_metrics = generate_shap_report(
                    model=model,
                    df=valid_split,
                    output_dir=shap_dir,
                    max_samples=2000,
                    top_n=20,
                )
                logger.info(
                    "SHAP report: status=%s model_type=%s features=%d",
                    shap_metrics.get("status"),
                    shap_metrics.get("model_type"),
                    shap_metrics.get("n_features", 0),
                )
        except Exception as exc:
            logger.warning("SHAP report failed (non-fatal): %s", exc)
            shap_metrics = {"status": "error", "error": str(exc)}

        # --- WoE / IV Report (if WoE candidate was used or always for analysis) ---
        woe_iv_metrics: dict[str, Any] = {"status": "skipped"}
        try:
            from risk_case.models.woe_baseline import compute_woe_iv, woe_iv_report_dataframe, woe_iv_summary_dataframe

            with log_stage(logger, "woe_iv_analysis"):
                woe_features = compute_woe_iv(train_split)
                woe_report = woe_iv_report_dataframe(woe_features)
                woe_summary = woe_iv_summary_dataframe(woe_features)
                woe_dir = ensure_dir(run_dir / "woe_iv")
                woe_report.to_csv(woe_dir / "woe_report.csv", index=False)
                woe_summary.to_csv(woe_dir / "iv_summary.csv", index=False)
                top_iv = woe_summary.head(10)
                woe_iv_metrics = {
                    "status": "ok",
                    "total_features": len(woe_summary),
                    "features_iv_above_002": int((woe_summary["iv"] >= 0.02).sum()),
                    "features_iv_above_010": int((woe_summary["iv"] >= 0.10).sum()),
                    "top_features": top_iv.to_dict(orient="records"),
                    "report_path": str(woe_dir / "woe_report.csv"),
                    "summary_path": str(woe_dir / "iv_summary.csv"),
                }
                logger.info(
                    "WoE/IV analysis: %d features, %d with IV >= 0.02",
                    woe_iv_metrics["total_features"],
                    woe_iv_metrics["features_iv_above_002"],
                )
        except Exception as exc:
            logger.warning("WoE/IV analysis failed (non-fatal): %s", exc)
            woe_iv_metrics = {"status": "error", "error": str(exc)}

        # --- Bootstrap Confidence Intervals ---
        bootstrap_metrics: dict[str, Any] = {"status": "skipped"}
        try:
            from risk_case.models.bootstrap_ci import compute_bootstrap_ci, bootstrap_ci_dataframe

            with log_stage(logger, "bootstrap_ci"):
                ci_result = compute_bootstrap_ci(
                    y_true=valid_split[TARGET_CLAIM_COL].fillna(0).astype(int).values,
                    p_pred=valid_pred["p_claim"].values,
                    premiums=valid_split[PREMIUM_COL].values if PREMIUM_COL in valid_split.columns else None,
                    claims=pd.to_numeric(valid_split[TARGET_AMOUNT_COL], errors="coerce").fillna(0.0).values if TARGET_AMOUNT_COL in valid_split.columns else None,
                    new_premiums=best_premium.values if hasattr(best_premium, "values") else None,
                    n_bootstrap=500,
                    confidence=0.95,
                )
                ci_df = bootstrap_ci_dataframe(ci_result)
                ci_dir = ensure_dir(run_dir / "bootstrap_ci")
                ci_df.to_csv(ci_dir / "confidence_intervals.csv", index=False)
                (ci_dir / "bootstrap_ci.json").write_text(
                    json.dumps(ci_result, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                bootstrap_metrics = {
                    "status": "ok",
                    "n_bootstrap": ci_result.get("n_bootstrap"),
                    "auc_ci": ci_result.get("auc"),
                    "gini_ci": ci_result.get("gini"),
                }
                logger.info(
                    "Bootstrap CI: AUC=[%.4f, %.4f] Gini=[%.4f, %.4f]",
                    (ci_result.get("auc") or {}).get("lower", 0) or 0,
                    (ci_result.get("auc") or {}).get("upper", 0) or 0,
                    (ci_result.get("gini") or {}).get("lower", 0) or 0,
                    (ci_result.get("gini") or {}).get("upper", 0) or 0,
                )
        except Exception as exc:
            logger.warning("Bootstrap CI failed (non-fatal): %s", exc)
            bootstrap_metrics = {"status": "error", "error": str(exc)}

        with log_stage(logger, "portfolio_diagnostics"):
            diagnostics_metrics = _build_portfolio_diagnostics(
                valid_out=valid_out,
                run_dir=run_dir,
                enabled=config.diagnostics_enabled,
                deciles=config.diagnostics_deciles,
                logger=logger,
            )

        if config.ablation_enabled:
            with log_stage(logger, "ablation_diagnostics"):
                ablation_metrics = _run_ablation_diagnostics(
                    train_df=train_split,
                    valid_df=valid_split,
                    preprocessor=preprocessor,
                    config=config,
                    alpha_grid=alpha_grid,
                    beta_grid=beta_grid,
                    target_band=target_band,
                    pricing_method=pricing_method,
                    pricing_retention=pricing_retention,
                    pricing_slsqp_options=pricing_slsqp_options,
                    pricing_stratified_config=pricing_stratified,
                    run_dir=run_dir,
                    logger=logger,
                )
        else:
            logger.info("Ablation diagnostics disabled by config")
            ablation_metrics = {
                "enabled": False,
                "path": None,
                "rows_total": 0,
                "rows_ok": 0,
                "rows_failed": 0,
            }

        metrics = {
            "run_id": run_id,
            "validation_warnings": validation.warnings,
            "preprocessing": {
                "raw_rows": int(len(raw_train_df)),
                "policy_rows": int(preprocess_artifacts.row_count),
                "train_split_rows": int(len(train_split)),
                "valid_split_rows": int(len(valid_split)),
                "validation_scheme": split_meta,
                "feature_count": int(len(preprocess_artifacts.feature_columns)),
                "dataset_path": str(preprocess_artifacts.dataset_path),
                "metadata_path": str(preprocess_artifacts.metadata_path),
                "quality_report_path": str(preprocess_artifacts.quality_report_path),
            },
            "ml": ml,
            "severity": severity,
            "pricing": {
                "alpha": best_alpha,
                "beta": best_beta,
                "optimization_method": pricing_method,
                "retention_config": pricing_retention.to_dict(),
                "pricing_policy_kind": pricing_policy.kind,
                "pricing_policy_path": str(pricing_policy_path),
                "pricing_policy": pricing_policy.to_summary(),
                **best_eval.to_dict(),
            },
            "ablation": ablation_metrics,
            "benchmark": benchmark_metrics,
            "diagnostics": diagnostics_metrics,
            "shap": shap_metrics,
            "woe_iv": woe_iv_metrics,
            "bootstrap_ci": bootstrap_metrics,
        }

        with log_stage(logger, "save_metrics_and_summary"):
            (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            (run_dir / "summary.md").write_text(_build_summary(metrics, run_id), encoding="utf-8")

        if test_path is not None and test_path.exists():
            with log_stage(logger, "inference_and_submission"):
                raw_test_df = read_csv(test_path)
                logger.info("Raw test loaded: rows=%d cols=%d", len(raw_test_df), len(raw_test_df.columns))
                processed_test_df, policy_test_df = transform_inference_feature_store(
                    raw_df=raw_test_df,
                    preprocessor=preprocessor,
                    output_dir=preprocess_dir,
                )
                logger.info("Processed test rows=%d", len(processed_test_df))

                test_pred = model.predict(processed_test_df)
                policy_test_output = policy_test_df.copy()
                policy_test_output["p_claim"] = test_pred["p_claim"].values
                policy_test_output["expected_severity"] = test_pred["expected_severity"].values
                policy_test_output["expected_loss"] = test_pred["expected_loss"].values
                policy_test_output["new_premium"] = apply_pricing_policy_artifact(
                    df=processed_test_df,
                    expected_loss=test_pred["expected_loss"],
                    pricing_policy=pricing_policy,
                ).values
                policy_test_output.to_csv(run_dir / "test_policy_predictions.csv", index=False)

                merged_submission_base = policy_to_raw_join(
                    raw_df=raw_test_df,
                    policy_predictions=policy_test_output[[CONTRACT_COL, "new_premium"]],
                    contract_col=CONTRACT_COL,
                )
                keep_columns = [col for col in ["unique_id", CONTRACT_COL, "new_premium"] if col in merged_submission_base.columns]
                submission = merged_submission_base[keep_columns] if keep_columns else merged_submission_base[["new_premium"]]
                submission.to_csv(run_dir / "submission.csv", index=False)
                logger.info("Submission rows=%d", len(submission))

        leaderboard_row = {
            "run_id": run_id,
            "winner_model": winner_name,
            "selection_reason": selection_reason,
            "auc": ml.get("auc"),
            "gini": ml.get("gini"),
            "brier": ml.get("brier"),
            "lr_total": best_eval.lr_total,
            "lr_group1": best_eval.lr_group1,
            "lr_group2": best_eval.lr_group2,
            "share_group1": best_eval.share_group1,
            "alpha": best_alpha,
            "beta": best_beta,
            "pricing_policy_kind": pricing_policy.kind,
            "in_target": best_eval.in_target,
            "distance_to_target": best_eval.distance_to_target,
            "policy_rows": int(preprocess_artifacts.row_count),
            "feature_count": int(len(preprocess_artifacts.feature_columns)),
            "run_path": str(run_dir),
        }
        with log_stage(logger, "update_leaderboard"):
            _update_leaderboard(artifacts_root / "leaderboard.csv", leaderboard_row)

        with log_stage(logger, "write_latest_pointer"):
            latest_pointer = artifacts_root / "latest_run.json"
            latest_pointer.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "model_path": str(model_path),
                        "metrics_path": str(run_dir / "metrics.json"),
                        "pricing_policy_path": str(pricing_policy_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        logger.info("Run completed: %s", run_id)
        return {
            "status": "ok",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "metrics": metrics,
        }
    finally:
        close_run_logger(logger)
