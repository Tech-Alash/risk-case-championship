from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.data.io import read_csv
from risk_case.data.validation import validate_dataset
from risk_case.features.feature_store import build_train_feature_store
from risk_case.models.benchmark import BenchmarkConfig, CandidateResult, run_model_benchmark
from risk_case.orchestration.run_pipeline import RunConfig
from risk_case.settings import TARGET_CLAIM_COL, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick XGBoost tuning sweep on risk-case data")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.json",
        help="Path to pipeline config",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=4,
        help="Maximum number of XGBoost trials from built-in grid",
    )
    return parser.parse_args()


def _resolve_path(base: Path, maybe_relative: Path | None) -> Path | None:
    if maybe_relative is None:
        return None
    if maybe_relative.is_absolute():
        return maybe_relative
    return (base / maybe_relative).resolve()


def _default_xgb_grid() -> list[dict[str, Any]]:
    return [
        {
            "n_estimators": 220,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "gamma": 0.0,
        },
        {
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.04,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2.0,
            "reg_lambda": 1.2,
            "reg_alpha": 0.0,
            "gamma": 0.0,
        },
        {
            "n_estimators": 360,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "min_child_weight": 2.0,
            "reg_lambda": 1.5,
            "reg_alpha": 0.0,
            "gamma": 0.0,
        },
        {
            "n_estimators": 180,
            "max_depth": 5,
            "learning_rate": 0.07,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "gamma": 0.0,
        },
    ]


def _build_benchmark_cfg_from_run_config(run_cfg: RunConfig, candidate: str, params: dict[str, Any] | None = None) -> BenchmarkConfig:
    return BenchmarkConfig(
        enabled=True,
        candidates=[candidate],
        selection_metric=run_cfg.benchmark.selection_metric,
        constraints=run_cfg.benchmark.constraints,
        fallback_strategy=run_cfg.benchmark.fallback_strategy,
        fallback_candidate=candidate,
        random_state=run_cfg.benchmark.random_state,
        candidate_params={candidate: dict(params or {})} if params else {},
    )


def _result_to_row(result: CandidateResult, trial_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {"trial_id": trial_id, **result.to_record()}
    if extra:
        row.update(extra)
    return row


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parents[1] if config_path.parent.name == "configs" else config_path.parent
    config = RunConfig.from_json(config_path)

    train_path = _resolve_path(project_root, config.train_csv)
    artifacts_root = _resolve_path(project_root, config.artifacts_dir)
    assert train_path is not None
    assert artifacts_root is not None

    raw_train_df = read_csv(train_path)
    validation = validate_dataset(raw_train_df)
    if not validation.ok:
        raise ValueError(f"Validation failed: {validation.errors}")

    processed_train_df, _, preprocess_artifacts = build_train_feature_store(
        raw_df=raw_train_df,
        config=config.preprocessing,
        output_dir=ensure_dir(artifacts_root / "tuning" / "xgboost" / "preprocessed_cache"),
    )

    y = processed_train_df[TARGET_CLAIM_COL].fillna(0).astype(int)
    train_split, valid_split = train_test_split(
        processed_train_df,
        test_size=config.split_test_size,
        random_state=config.split_random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    alpha_grid = np.linspace(config.pricing_alpha_start, config.pricing_alpha_stop, config.pricing_alpha_num)
    beta_grid = np.linspace(config.pricing_beta_start, config.pricing_beta_stop, max(1, config.pricing_beta_num))
    if config.pricing_target_band_min is not None and config.pricing_target_band_max is not None:
        target_band = (float(config.pricing_target_band_min), float(config.pricing_target_band_max))
    else:
        target_band = (float(config.benchmark.constraints.lr_total_min), float(config.benchmark.constraints.lr_total_max))

    baseline_cfg = _build_benchmark_cfg_from_run_config(config, "baseline_freq_sev")
    baseline_result = run_model_benchmark(
        train_df=train_split,
        valid_df=valid_split,
        benchmark_config=baseline_cfg,
        pricing_target_lr=config.pricing_target_lr,
        pricing_alpha_grid=alpha_grid,
        pricing_beta_grid=beta_grid,
        pricing_target_band=target_band,
        model_max_iter=config.model_max_iter,
        model_ridge_alpha=config.model_ridge_alpha,
    )
    baseline_candidate = baseline_result.results[0]

    trials = _default_xgb_grid()[: max(1, int(args.max_trials))]
    trial_rows: list[dict[str, Any]] = []

    for idx, params in enumerate(trials, start=1):
        trial_id = f"xgb_trial_{idx:02d}"
        cfg = _build_benchmark_cfg_from_run_config(config, "xgboost_freq_sev", params=params)
        run = run_model_benchmark(
            train_df=train_split,
            valid_df=valid_split,
            benchmark_config=cfg,
            pricing_target_lr=config.pricing_target_lr,
            pricing_alpha_grid=alpha_grid,
            pricing_beta_grid=beta_grid,
            pricing_target_band=target_band,
            model_max_iter=config.model_max_iter,
            model_ridge_alpha=config.model_ridge_alpha,
        )
        trial_rows.append(_result_to_row(run.results[0], trial_id=trial_id, extra=params))

    trials_df = pd.DataFrame(trial_rows)
    if trials_df.empty:
        raise ValueError("No XGBoost trials were executed")

    trials_df["policy_score_sort"] = pd.to_numeric(trials_df["policy_score"], errors="coerce").fillna(-np.inf)
    trials_df["gini_sort"] = pd.to_numeric(trials_df["gini"], errors="coerce").fillna(-np.inf)
    trials_df["passes_constraints_sort"] = trials_df["passes_constraints"].astype(bool).astype(int)
    best_row = (
        trials_df.sort_values(
            by=["passes_constraints_sort", "policy_score_sort", "gini_sort"],
            ascending=[False, False, False],
        )
        .head(1)
        .iloc[0]
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tuning_dir = ensure_dir(artifacts_root / "tuning" / "xgboost" / ts)
    results_csv = tuning_dir / "results.csv"
    summary_json = tuning_dir / "summary.json"

    trials_df.drop(columns=["policy_score_sort", "gini_sort", "passes_constraints_sort"]).to_csv(results_csv, index=False)

    summary = {
        "timestamp_utc": ts,
        "config_path": str(config_path),
        "preprocessing": {
            "raw_rows": int(len(raw_train_df)),
            "policy_rows": int(preprocess_artifacts.row_count),
            "feature_count": int(len(preprocess_artifacts.feature_columns)),
        },
        "baseline": baseline_candidate.to_dict(),
        "best_xgb_trial": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_row.to_dict().items()},
        "results_csv_path": str(results_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "tuning_dir": str(tuning_dir),
                "results_csv": str(results_csv),
                "summary_json": str(summary_json),
                "baseline_policy_score": baseline_candidate.pricing.get("policy_score") if baseline_candidate.pricing else None,
                "best_xgb_trial": summary["best_xgb_trial"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
