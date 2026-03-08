from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.data.io import read_csv
from risk_case.data.policy_aggregation import aggregate_to_policy_level
from risk_case.data.validation import validate_dataset
from risk_case.features.preprocessing import (
    build_oof_target_encoding_features,
    fit_preprocessor,
    transform_with_preprocessor,
)
from risk_case.models.benchmark import BenchmarkConfig, run_model_benchmark
from risk_case.orchestration.run_pipeline import RunConfig, _resolve_target_band, _split_policy_train_valid
from risk_case.settings import TARGET_AMOUNT_COL, TARGET_CLAIM_COL, ensure_dir

LOGGER = logging.getLogger("risk_case.stability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-split stability checks for benchmark candidate")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.json",
        help="Path to pipeline config",
    )
    parser.add_argument(
        "--splits_config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "stability_splits.json",
        help="Path to stability split config with time_holdout_starts list",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="catboost_freq_sev",
        help="Benchmark candidate name",
    )
    parser.add_argument(
        "--params_json",
        type=Path,
        default=None,
        help="Optional path to candidate params json (e.g. best_params.json from tuning)",
    )
    parser.add_argument(
        "--catboost_threads",
        type=int,
        default=None,
        help="Optional CPU thread limit for catboost candidate (e.g. 4)",
    )
    parser.add_argument(
        "--objective_lambda",
        type=float,
        default=0.35,
        help="Lambda for objective=mean_policy_score-lambda*std_policy_score in summary",
    )
    return parser.parse_args()


def _resolve_path(base: Path, maybe_relative: Path | None) -> Path | None:
    if maybe_relative is None:
        return None
    if maybe_relative.is_absolute():
        return maybe_relative
    return (base / maybe_relative).resolve()


def _read_splits(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = raw.get("time_holdout_starts")
    if not isinstance(values, list):
        raise ValueError("stability splits config must contain list field: time_holdout_starts")
    cleaned = [str(item).strip() for item in values if str(item).strip()]
    if not cleaned:
        raise ValueError("time_holdout_starts must contain at least one value")
    return cleaned


def _build_single_candidate_cfg(run_cfg: RunConfig, candidate: str, params: dict[str, Any] | None) -> BenchmarkConfig:
    candidate_params = dict(params or {})
    if not candidate_params:
        candidate_params = dict(run_cfg.benchmark.candidate_params.get(candidate) or {})
    return BenchmarkConfig(
        enabled=True,
        candidates=[candidate],
        selection_metric=run_cfg.benchmark.selection_metric,
        stability_penalty=run_cfg.benchmark.stability_penalty,
        must_pass_constraints=True,
        constraints=run_cfg.benchmark.constraints,
        fallback_strategy="best_metric",
        fallback_candidate=candidate,
        random_state=run_cfg.benchmark.random_state,
        candidate_params={candidate: candidate_params} if candidate_params else {},
        calibration=run_cfg.benchmark.calibration,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config_path = args.config.resolve()
    project_root = config_path.parents[1] if config_path.parent.name == "configs" else config_path.parent
    run_cfg = RunConfig.from_json(config_path)

    if run_cfg.preprocessing.feature_whitelist_path:
        run_cfg.preprocessing.feature_whitelist_path = str(
            _resolve_path(project_root, Path(run_cfg.preprocessing.feature_whitelist_path))
        )
    if run_cfg.preprocessing.feature_droplist_path:
        run_cfg.preprocessing.feature_droplist_path = str(
            _resolve_path(project_root, Path(run_cfg.preprocessing.feature_droplist_path))
        )

    splits_path = args.splits_config.resolve()
    splits = _read_splits(splits_path)

    train_path = _resolve_path(project_root, run_cfg.train_csv)
    artifacts_root = _resolve_path(project_root, run_cfg.artifacts_dir)
    assert train_path is not None
    assert artifacts_root is not None

    candidate_params: dict[str, Any] = {}
    if args.params_json:
        params_path = args.params_json.resolve()
        candidate_params = json.loads(params_path.read_text(encoding="utf-8"))
    if args.catboost_threads is not None:
        if int(args.catboost_threads) <= 0:
            raise ValueError("--catboost_threads must be > 0")
        if args.candidate in {"catboost_freq_sev", "catboost_dep_freq_sev"}:
            candidate_params["thread_count"] = int(args.catboost_threads)
    if args.candidate == "catboost_dep_freq_sev":
        candidate_params.setdefault("dep_oof_folds", 5)
        candidate_params.setdefault("dep_frequency_signal_name", "freq_risk_signal")
        candidate_params.setdefault("dep_use_frequency_signal", True)

    target_band = _resolve_target_band(run_cfg)
    alpha_grid = np.linspace(run_cfg.pricing_alpha_start, run_cfg.pricing_alpha_stop, run_cfg.pricing_alpha_num)
    beta_grid = np.linspace(run_cfg.pricing_beta_start, run_cfg.pricing_beta_stop, max(1, run_cfg.pricing_beta_num))
    benchmark_cfg = _build_single_candidate_cfg(run_cfg, candidate=args.candidate, params=candidate_params)

    raw_train_df = read_csv(train_path)
    validation = validate_dataset(raw_train_df)
    if not validation.ok:
        raise ValueError(f"Validation failed: {validation.errors}")
    policy_df = aggregate_to_policy_level(raw_train_df, contract_col=run_cfg.preprocessing.grain)

    rows: list[dict[str, Any]] = []
    for split_value in splits:
        run_cfg.validation_time_holdout_start = split_value
        train_policy_split, valid_policy_split, split_meta = _split_policy_train_valid(
            policy_df=policy_df,
            config=run_cfg,
            logger=LOGGER,
        )

        preprocessor = fit_preprocessor(train_policy_split, run_cfg.preprocessing)
        train_df = transform_with_preprocessor(train_policy_split, preprocessor)
        valid_df = transform_with_preprocessor(valid_policy_split, preprocessor)

        if run_cfg.preprocessing.target_encoding_enabled and preprocessor.target_encoding_maps:
            oof_target_encoding = build_oof_target_encoding_features(
                df=train_policy_split,
                state=preprocessor,
                target_column=TARGET_CLAIM_COL,
                n_splits=run_cfg.validation_group_kfold_n_splits,
                random_state=run_cfg.split_random_state,
                group_column=run_cfg.validation_group_column if "group" in split_meta.get("scheme", "") else None,
            )
            for column in oof_target_encoding.columns:
                train_df[column] = oof_target_encoding[column].values

        if TARGET_CLAIM_COL not in train_df.columns:
            raise ValueError(f"{TARGET_CLAIM_COL} is missing after preprocessing")
        if TARGET_AMOUNT_COL not in train_df.columns:
            raise ValueError(f"{TARGET_AMOUNT_COL} is missing after preprocessing")

        result = run_model_benchmark(
            train_df=train_df,
            valid_df=valid_df,
            benchmark_config=benchmark_cfg,
            pricing_target_lr=run_cfg.pricing_target_lr,
            pricing_alpha_grid=alpha_grid,
            pricing_beta_grid=beta_grid,
            pricing_target_band=target_band,
            model_max_iter=run_cfg.model_max_iter,
            model_ridge_alpha=run_cfg.model_ridge_alpha,
            pricing_optimization_method=run_cfg.pricing_optimization_method,
            pricing_retention=run_cfg.pricing_retention,
            pricing_slsqp_options={
                "maxiter": int(run_cfg.pricing_slsqp_maxiter),
                "ftol": float(run_cfg.pricing_slsqp_ftol),
                "eps": float(run_cfg.pricing_slsqp_eps),
            },
            pricing_stratified_config=run_cfg.pricing_stratified,
        )
        candidate = result.results[0]
        pricing = candidate.pricing or {}
        ml = candidate.ml or {}
        severity = candidate.severity or {}
        rows.append(
            {
                "split_time_holdout_start": split_value,
                "split_scheme": split_meta.get("scheme"),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "status": candidate.status,
                "passes_constraints": bool(candidate.passes_constraints),
                "policy_score": pricing.get("policy_score"),
                "in_target": pricing.get("in_target"),
                "distance_to_target": pricing.get("distance_to_target"),
                "lr_total": pricing.get("lr_total"),
                "lr_group1": pricing.get("lr_group1"),
                "lr_group2": pricing.get("lr_group2"),
                "share_group1": pricing.get("share_group1"),
                "violations": pricing.get("violations"),
                "gini": ml.get("gini"),
                "auc": ml.get("auc"),
                "severity_rmse": severity.get("rmse"),
                "severity_mae": severity.get("mae"),
                "severity_r2": severity.get("r2"),
            }
        )

    results_df = pd.DataFrame(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(artifacts_root / "stability" / ts)
    results_csv = out_dir / "results.csv"
    summary_json = out_dir / "summary.json"
    results_df.to_csv(results_csv, index=False)

    policy_score_values = pd.to_numeric(results_df.get("policy_score"), errors="coerce")
    lr_group1_values = pd.to_numeric(results_df.get("lr_group1"), errors="coerce")
    lr_group2_values = pd.to_numeric(results_df.get("lr_group2"), errors="coerce")
    share_group1_values = pd.to_numeric(results_df.get("share_group1"), errors="coerce")
    severity_rmse_values = pd.to_numeric(results_df.get("severity_rmse"), errors="coerce")
    severity_mae_values = pd.to_numeric(results_df.get("severity_mae"), errors="coerce")
    severity_r2_values = pd.to_numeric(results_df.get("severity_r2"), errors="coerce")
    in_target_rate = float(pd.to_numeric(results_df.get("in_target"), errors="coerce").fillna(0.0).mean())
    constraints_pass_rate = float(pd.to_numeric(results_df.get("passes_constraints"), errors="coerce").fillna(0.0).mean())
    violations_series = pd.to_numeric(results_df.get("violations"), errors="coerce").fillna(0.0)
    objective_lambda = max(float(args.objective_lambda), 0.0)
    mean_policy_score = float(policy_score_values.mean()) if not policy_score_values.empty else None
    std_policy_score = float(policy_score_values.std(ddof=0)) if not policy_score_values.empty else None
    objective_value = (
        float(mean_policy_score - objective_lambda * std_policy_score)
        if mean_policy_score is not None and std_policy_score is not None
        else None
    )
    summary = {
        "timestamp_utc": ts,
        "config_path": str(config_path),
        "splits_config_path": str(splits_path),
        "candidate": args.candidate,
        "params_json_path": str(args.params_json.resolve()) if args.params_json else None,
        "n_splits": int(len(results_df)),
        "in_target_rate": in_target_rate,
        "constraints_pass_rate": constraints_pass_rate,
        "max_violations": int(violations_series.max()) if not violations_series.empty else None,
        "mean_policy_score": mean_policy_score,
        "std_policy_score": std_policy_score,
        "objective_lambda": objective_lambda,
        "objective_value": objective_value,
        "mean_lr_group1": float(lr_group1_values.mean()) if not lr_group1_values.empty else None,
        "mean_lr_group2": float(lr_group2_values.mean()) if not lr_group2_values.empty else None,
        "mean_share_group1": float(share_group1_values.mean()) if not share_group1_values.empty else None,
        "mean_severity_rmse": float(severity_rmse_values.mean()) if not severity_rmse_values.empty else None,
        "mean_severity_mae": float(severity_mae_values.mean()) if not severity_mae_values.empty else None,
        "mean_severity_r2": float(severity_r2_values.mean()) if not severity_r2_values.empty else None,
        "results_csv_path": str(results_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "results_csv": str(results_csv),
                "summary_json": str(summary_json),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
