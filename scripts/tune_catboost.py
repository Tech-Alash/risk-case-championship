from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import optuna
except Exception as exc:  # pragma: no cover - import guard for CLI clarity
    raise ImportError("optuna package is required for tune_catboost.py") from exc

from risk_case.data.io import read_csv
from risk_case.data.policy_aggregation import aggregate_to_policy_level
from risk_case.data.validation import validate_dataset
from risk_case.features.preprocessing import (
    build_oof_target_encoding_features,
    fit_preprocessor,
    transform_with_preprocessor,
)
from risk_case.models.benchmark import BenchmarkConfig, CandidateResult, run_model_benchmark
from risk_case.orchestration.run_pipeline import RunConfig, _resolve_target_band, _split_policy_train_valid
from risk_case.settings import TARGET_AMOUNT_COL, TARGET_CLAIM_COL, ensure_dir


LOGGER = logging.getLogger("risk_case.tuning.catboost")


@dataclass
class PreparedData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    split_meta: dict[str, Any]
    policy_rows: int
    feature_count: int
    raw_rows: int
    cache_dir: Path


@dataclass
class PreparedSplitCache:
    split_value: str
    split_meta: dict[str, Any]
    train_path: Path
    valid_path: Path
    train_rows: int
    valid_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning for CatBoost benchmark candidates")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.json",
        help="Path to pipeline config",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=240,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--timeout_sec",
        type=int,
        default=None,
        help="Optional timeout in seconds",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="catboost_tuning",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="",
        help="SQLite file path for study storage (optional, relative paths are resolved from artifacts/tuning/catboost)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (defaults to benchmark.random_state from config)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel Optuna workers",
    )
    parser.add_argument(
        "--catboost_threads",
        type=int,
        default=None,
        help="Optional CPU thread limit per CatBoost model fit (e.g. 4)",
    )
    parser.add_argument(
        "--candidate",
        choices=["catboost_freq_sev", "catboost_dep_freq_sev"],
        default="catboost_freq_sev",
        help="CatBoost benchmark candidate to tune",
    )
    parser.add_argument(
        "--phase",
        choices=["coarse", "fine", "focused"],
        default="coarse",
        help="Tuning phase: coarse (wider search), fine (narrow search), focused (tight region around best-known params)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "severity_only"],
        default="full",
        help="full: tune frequency+severity params; severity_only: tune only severity params with fixed frequency baseline",
    )
    parser.add_argument(
        "--time_holdout_start_override",
        type=str,
        default="",
        help="Optional override for validation.time_holdout_start in format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument(
        "--objective_mode",
        choices=["single_split", "stability_adjusted"],
        default="single_split",
        help="single_split: optimize on configured validation split; stability_adjusted: optimize mean(policy_score)-lambda*std over multiple holdout splits",
    )
    parser.add_argument(
        "--splits_config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "stability_splits.json",
        help="Path to split config with time_holdout_starts (used when objective_mode=stability_adjusted)",
    )
    parser.add_argument(
        "--stability_lambda",
        type=float,
        default=0.35,
        help="Penalty coefficient for std(policy_score) in stability_adjusted objective",
    )
    parser.add_argument(
        "--stability_min_constraints_rate",
        type=float,
        default=1.0,
        help="Minimum required constraints pass-rate for stability_adjusted objective",
    )
    parser.add_argument(
        "--stability_min_in_target_rate",
        type=float,
        default=0.8,
        help="Minimum required in_target rate for stability_adjusted objective",
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
        raise ValueError("splits config must contain list field: time_holdout_starts")
    cleaned = [str(item).strip() for item in values if str(item).strip()]
    if not cleaned:
        raise ValueError("time_holdout_starts must contain at least one value")
    return cleaned


def _build_benchmark_cfg_from_run_config(
    run_cfg: RunConfig,
    candidate: str,
    params: dict[str, Any] | None = None,
) -> BenchmarkConfig:
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


def _prepare_data(
    config: RunConfig,
    train_path: Path,
    artifacts_root: Path,
) -> PreparedData:
    raw_train_df = read_csv(train_path)
    validation = validate_dataset(raw_train_df)
    if not validation.ok:
        raise ValueError(f"Validation failed: {validation.errors}")

    policy_df = aggregate_to_policy_level(raw_train_df, contract_col=config.preprocessing.grain)
    train_policy_split, valid_policy_split, split_meta = _split_policy_train_valid(
        policy_df=policy_df,
        config=config,
        logger=LOGGER,
    )

    preprocessor = fit_preprocessor(train_policy_split, config.preprocessing)
    train_split = transform_with_preprocessor(train_policy_split, preprocessor)
    valid_split = transform_with_preprocessor(valid_policy_split, preprocessor)

    if config.preprocessing.target_encoding_enabled and preprocessor.target_encoding_maps:
        oof_target_encoding = build_oof_target_encoding_features(
            df=train_policy_split,
            state=preprocessor,
            target_column=TARGET_CLAIM_COL,
            n_splits=config.validation_group_kfold_n_splits,
            random_state=config.split_random_state,
            group_column=config.validation_group_column if "group" in split_meta.get("scheme", "") else None,
        )
        for column in oof_target_encoding.columns:
            train_split[column] = oof_target_encoding[column].values

    if TARGET_CLAIM_COL not in train_split.columns:
        raise ValueError(f"{TARGET_CLAIM_COL} is missing after preprocessing")
    if TARGET_AMOUNT_COL not in train_split.columns:
        raise ValueError(f"{TARGET_AMOUNT_COL} is missing after preprocessing")

    cache_dir = ensure_dir(artifacts_root / "tuning" / "catboost" / "preprocessed_cache")
    train_split.to_csv(cache_dir / "train_split_preprocessed.csv", index=False)
    valid_split.to_csv(cache_dir / "valid_split_preprocessed.csv", index=False)
    (cache_dir / "split_meta.json").write_text(json.dumps(split_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return PreparedData(
        train_df=train_split,
        valid_df=valid_split,
        split_meta=split_meta,
        policy_rows=int(len(policy_df)),
        feature_count=int(len(preprocessor.feature_columns)),
        raw_rows=int(len(raw_train_df)),
        cache_dir=cache_dir,
    )


def _prepare_stability_cache(
    config: RunConfig,
    train_path: Path,
    artifacts_root: Path,
    splits_config: Path,
) -> tuple[list[PreparedSplitCache], dict[str, Any]]:
    splits = _read_splits(splits_config)
    raw_train_df = read_csv(train_path)
    validation = validate_dataset(raw_train_df)
    if not validation.ok:
        raise ValueError(f"Validation failed: {validation.errors}")

    policy_df = aggregate_to_policy_level(raw_train_df, contract_col=config.preprocessing.grain)
    cache_root = ensure_dir(artifacts_root / "tuning" / "catboost" / "preprocessed_cache" / "stability_objective")

    split_refs: list[PreparedSplitCache] = []
    feature_count = 0
    original_holdout = config.validation_time_holdout_start
    try:
        for index, split_value in enumerate(splits):
            config.validation_time_holdout_start = split_value
            train_policy_split, valid_policy_split, split_meta = _split_policy_train_valid(
                policy_df=policy_df,
                config=config,
                logger=LOGGER,
            )

            preprocessor = fit_preprocessor(train_policy_split, config.preprocessing)
            train_split = transform_with_preprocessor(train_policy_split, preprocessor)
            valid_split = transform_with_preprocessor(valid_policy_split, preprocessor)

            if config.preprocessing.target_encoding_enabled and preprocessor.target_encoding_maps:
                oof_target_encoding = build_oof_target_encoding_features(
                    df=train_policy_split,
                    state=preprocessor,
                    target_column=TARGET_CLAIM_COL,
                    n_splits=config.validation_group_kfold_n_splits,
                    random_state=config.split_random_state,
                    group_column=config.validation_group_column if "group" in split_meta.get("scheme", "") else None,
                )
                for column in oof_target_encoding.columns:
                    train_split[column] = oof_target_encoding[column].values

            if TARGET_CLAIM_COL not in train_split.columns:
                raise ValueError(f"{TARGET_CLAIM_COL} is missing after preprocessing on split={split_value}")
            if TARGET_AMOUNT_COL not in train_split.columns:
                raise ValueError(f"{TARGET_AMOUNT_COL} is missing after preprocessing on split={split_value}")

            safe_split = (
                split_value.replace(" ", "_")
                .replace(":", "")
                .replace("-", "")
                .replace("/", "")
                .replace("\\", "")
            )
            split_dir = ensure_dir(cache_root / f"split_{index:02d}_{safe_split}")
            train_path_out = split_dir / "train_split_preprocessed.csv"
            valid_path_out = split_dir / "valid_split_preprocessed.csv"
            train_split.to_csv(train_path_out, index=False)
            valid_split.to_csv(valid_path_out, index=False)
            (split_dir / "split_meta.json").write_text(json.dumps(split_meta, ensure_ascii=False, indent=2), encoding="utf-8")

            feature_count = max(feature_count, int(len(preprocessor.feature_columns)))
            split_refs.append(
                PreparedSplitCache(
                    split_value=split_value,
                    split_meta=split_meta,
                    train_path=train_path_out,
                    valid_path=valid_path_out,
                    train_rows=int(len(train_split)),
                    valid_rows=int(len(valid_split)),
                )
            )
    finally:
        config.validation_time_holdout_start = original_holdout

    metadata = {
        "raw_rows": int(len(raw_train_df)),
        "policy_rows": int(len(policy_df)),
        "feature_count": int(feature_count),
        "cache_dir": str(cache_root),
        "splits_config_path": str(splits_config),
        "n_splits": int(len(split_refs)),
        "split_values": [item.split_value for item in split_refs],
    }
    return split_refs, metadata


def _load_cached_split(ref: PreparedSplitCache) -> tuple[pd.DataFrame, pd.DataFrame]:
    return read_csv(ref.train_path), read_csv(ref.valid_path)


def _suggest_catboost_params(trial: optuna.trial.Trial, phase: str, mode: str = "full") -> dict[str, Any]:
    mode = str(mode).strip().lower()
    if mode == "severity_only":
        if phase == "focused":
            return {
                "reg_iterations": trial.suggest_int("reg_iterations", 360, 700),
                "reg_learning_rate": trial.suggest_float("reg_learning_rate", 0.01, 0.04, log=True),
                "reg_depth": trial.suggest_int("reg_depth", 6, 8),
                "reg_l2_leaf_reg": trial.suggest_float("reg_l2_leaf_reg", 2.0, 12.0, log=True),
                "reg_random_strength": trial.suggest_float("reg_random_strength", 0.4, 1.8),
                "reg_bagging_temperature": trial.suggest_float("reg_bagging_temperature", 0.0, 2.0),
                "reg_border_count": trial.suggest_int("reg_border_count", 96, 200),
                "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.20, 1.75),
                "severity_loss_function": "TWEEDIE",
            }
        if phase == "fine":
            return {
                "reg_iterations": trial.suggest_int("reg_iterations", 240, 720),
                "reg_learning_rate": trial.suggest_float("reg_learning_rate", 0.015, 0.08, log=True),
                "reg_depth": trial.suggest_int("reg_depth", 5, 9),
                "reg_l2_leaf_reg": trial.suggest_float("reg_l2_leaf_reg", 1.5, 18.0, log=True),
                "reg_random_strength": trial.suggest_float("reg_random_strength", 0.0, 2.0),
                "reg_bagging_temperature": trial.suggest_float("reg_bagging_temperature", 0.0, 3.0),
                "reg_border_count": trial.suggest_int("reg_border_count", 96, 255),
                "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.15, 1.85),
                "severity_loss_function": "TWEEDIE",
            }
        return {
            "reg_iterations": trial.suggest_int("reg_iterations", 200, 900),
            "reg_learning_rate": trial.suggest_float("reg_learning_rate", 0.01, 0.2, log=True),
            "reg_depth": trial.suggest_int("reg_depth", 4, 10),
            "reg_l2_leaf_reg": trial.suggest_float("reg_l2_leaf_reg", 1.0, 30.0, log=True),
            "reg_random_strength": trial.suggest_float("reg_random_strength", 0.0, 2.5),
            "reg_bagging_temperature": trial.suggest_float("reg_bagging_temperature", 0.0, 5.0),
            "reg_border_count": trial.suggest_int("reg_border_count", 64, 255),
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.10, 1.90),
            "severity_loss_function": "TWEEDIE",
        }

    if phase == "focused":
        return {
            "iterations": trial.suggest_int("iterations", 620, 780),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, log=True),
            "depth": trial.suggest_int("depth", 7, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.5, 8.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.8, 1.5),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 1.0, 3.5),
            "border_count": trial.suggest_int("border_count", 96, 180),
            "reg_iterations": trial.suggest_int("reg_iterations", 360, 620),
            "reg_learning_rate": trial.suggest_float("reg_learning_rate", 0.01, 0.03, log=True),
            "reg_depth": trial.suggest_int("reg_depth", 6, 7),
            "reg_l2_leaf_reg": trial.suggest_float("reg_l2_leaf_reg", 3.0, 10.0, log=True),
            "reg_random_strength": trial.suggest_float("reg_random_strength", 0.8, 1.4),
            "reg_bagging_temperature": trial.suggest_float("reg_bagging_temperature", 0.0, 1.2),
            "reg_border_count": trial.suggest_int("reg_border_count", 96, 180),
        }

    if phase == "fine":
        return {
            "iterations": trial.suggest_int("iterations", 260, 520),
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.10, log=True),
            "depth": trial.suggest_int("depth", 5, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.5, 12.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.2),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.5),
            "border_count": trial.suggest_int("border_count", 96, 255),
            "reg_iterations": trial.suggest_int("reg_iterations", 260, 520),
            "reg_learning_rate": trial.suggest_float("reg_learning_rate", 0.025, 0.10, log=True),
            "reg_depth": trial.suggest_int("reg_depth", 5, 8),
            "reg_l2_leaf_reg": trial.suggest_float("reg_l2_leaf_reg", 1.5, 12.0, log=True),
            "reg_random_strength": trial.suggest_float("reg_random_strength", 0.0, 1.2),
            "reg_bagging_temperature": trial.suggest_float("reg_bagging_temperature", 0.0, 2.5),
            "reg_border_count": trial.suggest_int("reg_border_count", 96, 255),
        }

    return {
        "iterations": trial.suggest_int("iterations", 200, 900),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "border_count": trial.suggest_int("border_count", 64, 255),
        "reg_iterations": trial.suggest_int("reg_iterations", 200, 900),
        "reg_learning_rate": trial.suggest_float("reg_learning_rate", 0.01, 0.2, log=True),
        "reg_depth": trial.suggest_int("reg_depth", 4, 10),
        "reg_l2_leaf_reg": trial.suggest_float("reg_l2_leaf_reg", 1.0, 30.0, log=True),
        "reg_random_strength": trial.suggest_float("reg_random_strength", 0.0, 2.0),
        "reg_bagging_temperature": trial.suggest_float("reg_bagging_temperature", 0.0, 5.0),
        "reg_border_count": trial.suggest_int("reg_border_count", 64, 255),
    }


def _resolve_catboost_base_params(config: RunConfig, mode: str, candidate: str) -> dict[str, Any]:
    mode = str(mode).strip().lower()
    base_params = dict(config.benchmark.candidate_params.get(candidate) or {})
    if not base_params and candidate == "catboost_dep_freq_sev":
        base_params = dict(config.benchmark.candidate_params.get("catboost_freq_sev") or {})
    if mode != "severity_only":
        return base_params
    if "severity_loss_function" not in base_params:
        base_params["severity_loss_function"] = str(config.model_severity_loss or "TWEEDIE")
    base_params["severity_loss_function"] = "TWEEDIE"
    if "tweedie_variance_power" not in base_params:
        base_params["tweedie_variance_power"] = float(config.model_tweedie_variance_power)
    if candidate == "catboost_dep_freq_sev":
        base_params.setdefault("dep_oof_folds", 5)
        base_params.setdefault("dep_frequency_signal_name", "freq_risk_signal")
        base_params.setdefault("dep_use_frequency_signal", True)
    return base_params


def _candidate_to_attrs(candidate: CandidateResult) -> dict[str, Any]:
    pricing = candidate.pricing or {}
    ml = candidate.ml or {}
    severity = candidate.severity or {}
    return {
        "status": candidate.status,
        "error": candidate.error,
        "passes_constraints": bool(candidate.passes_constraints),
        "policy_score": pricing.get("policy_score"),
        "lr_total": pricing.get("lr_total"),
        "lr_group1": pricing.get("lr_group1"),
        "lr_group2": pricing.get("lr_group2"),
        "share_group1": pricing.get("share_group1"),
        "violations": pricing.get("violations"),
        "in_target": pricing.get("in_target"),
        "distance_to_target": pricing.get("distance_to_target"),
        "alpha": candidate.alpha,
        "beta": candidate.beta,
        "auc": ml.get("auc"),
        "gini": ml.get("gini"),
        "brier": ml.get("brier"),
        "severity_rmse": severity.get("rmse"),
        "severity_mae": severity.get("mae"),
        "elapsed_seconds": candidate.elapsed_seconds,
    }


def _objective_value_from_candidate(candidate: CandidateResult) -> float:
    pricing = candidate.pricing or {}
    policy_score = float(pricing.get("policy_score", float("-inf")))
    in_target = bool(pricing.get("in_target", False))
    distance = float(pricing.get("distance_to_target", 1e6))
    violations = int(pricing.get("violations", 0))

    if candidate.status == "ok" and candidate.passes_constraints and in_target:
        return policy_score

    return -1_000_000.0 - distance - (100.0 * max(0, violations))


def _objective_value_from_stability_candidates(
    split_values: list[str],
    candidates: list[CandidateResult],
    stability_lambda: float,
    min_constraints_rate: float,
    min_in_target_rate: float,
) -> tuple[float, dict[str, Any]]:
    n_splits = max(1, len(candidates))
    constraints_passes = 0
    in_target_passes = 0
    policy_scores: list[float] = []
    lr_totals: list[float] = []
    auc_values: list[float] = []
    gini_values: list[float] = []
    distance_values: list[float] = []
    violations_values: list[int] = []
    split_rows: list[dict[str, Any]] = []

    for split_value, candidate in zip(split_values, candidates):
        pricing = candidate.pricing or {}
        ml = candidate.ml or {}
        status_ok = candidate.status == "ok"
        passes_constraints = bool(candidate.passes_constraints) and status_ok
        in_target = bool(pricing.get("in_target", False)) and status_ok
        policy_score_raw = pricing.get("policy_score")
        policy_score = float(policy_score_raw) if policy_score_raw is not None else None
        lr_total_raw = pricing.get("lr_total")
        lr_total = float(lr_total_raw) if lr_total_raw is not None else None
        distance = float(pricing.get("distance_to_target", 1e6))
        violations = int(pricing.get("violations", 0))
        auc = ml.get("auc")
        gini = ml.get("gini")

        if passes_constraints:
            constraints_passes += 1
        if in_target:
            in_target_passes += 1

        if policy_score is not None and status_ok and passes_constraints and in_target:
            policy_scores.append(policy_score)
        if lr_total is not None and status_ok:
            lr_totals.append(lr_total)
        if auc is not None and status_ok:
            auc_values.append(float(auc))
        if gini is not None and status_ok:
            gini_values.append(float(gini))
        distance_values.append(distance)
        violations_values.append(violations)

        split_rows.append(
            {
                "split_value": split_value,
                "status": candidate.status,
                "passes_constraints": passes_constraints,
                "in_target": in_target,
                "policy_score": policy_score,
                "lr_total": lr_total,
                "distance_to_target": distance,
                "violations": violations,
                "auc": float(auc) if auc is not None else None,
                "gini": float(gini) if gini is not None else None,
            }
        )

    constraints_pass_rate = float(constraints_passes / n_splits)
    in_target_rate = float(in_target_passes / n_splits)
    policy_score_mean = float(np.mean(policy_scores)) if policy_scores else None
    policy_score_std = float(np.std(policy_scores, ddof=0)) if policy_scores else None
    lr_total_mean = float(np.mean(lr_totals)) if lr_totals else None
    auc_mean = float(np.mean(auc_values)) if auc_values else None
    gini_mean = float(np.mean(gini_values)) if gini_values else None
    distance_mean = float(np.mean(distance_values)) if distance_values else 1e6
    max_violations = int(max(violations_values) if violations_values else 0)

    gates_passed = (
        constraints_pass_rate >= float(min_constraints_rate)
        and in_target_rate >= float(min_in_target_rate)
        and policy_score_mean is not None
        and policy_score_std is not None
    )
    if gates_passed:
        objective_value = float(policy_score_mean - max(stability_lambda, 0.0) * policy_score_std)
        status = "ok"
    else:
        constraints_shortfall = max(0.0, float(min_constraints_rate) - constraints_pass_rate)
        in_target_shortfall = max(0.0, float(min_in_target_rate) - in_target_rate)
        objective_value = (
            -1_000_000.0
            - (10_000.0 * constraints_shortfall)
            - (10_000.0 * in_target_shortfall)
            - distance_mean
            - (100.0 * max_violations)
        )
        status = "gates_failed"

    attrs: dict[str, Any] = {
        "status": status,
        "error": None,
        "passes_constraints": bool(constraints_pass_rate >= float(min_constraints_rate)),
        "policy_score": policy_score_mean,
        "policy_score_mean": policy_score_mean,
        "policy_score_std": policy_score_std,
        "policy_score_count": int(len(policy_scores)),
        "lr_total": lr_total_mean,
        "lr_total_mean": lr_total_mean,
        "violations": int(max_violations),
        "max_violations": int(max_violations),
        "in_target": bool(in_target_rate >= float(min_in_target_rate)),
        "in_target_rate": in_target_rate,
        "constraints_pass_rate": constraints_pass_rate,
        "distance_to_target": distance_mean,
        "distance_to_target_mean": distance_mean,
        "auc": auc_mean,
        "gini": gini_mean,
        "stability_lambda": float(stability_lambda),
        "objective_mode": "stability_adjusted",
        "split_count": int(n_splits),
        "split_metrics_json": json.dumps(split_rows, ensure_ascii=False),
    }
    return objective_value, attrs


def _study_to_rows(study: optuna.study.Study) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_number": trial.number,
            "state": str(trial.state.name),
            "objective_value": float(trial.value) if trial.value is not None else None,
        }
        for key, value in trial.params.items():
            row[f"param_{key}"] = value
        for key, value in trial.user_attrs.items():
            row[key] = value
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    config_path = args.config.resolve()
    project_root = config_path.parents[1] if config_path.parent.name == "configs" else config_path.parent
    config = RunConfig.from_json(config_path)
    if args.time_holdout_start_override:
        config.validation_time_holdout_start = str(args.time_holdout_start_override)

    if config.preprocessing.feature_whitelist_path:
        config.preprocessing.feature_whitelist_path = str(
            _resolve_path(project_root, Path(config.preprocessing.feature_whitelist_path))
        )
    if config.preprocessing.feature_droplist_path:
        config.preprocessing.feature_droplist_path = str(
            _resolve_path(project_root, Path(config.preprocessing.feature_droplist_path))
        )

    train_path = _resolve_path(project_root, config.train_csv)
    artifacts_root = _resolve_path(project_root, config.artifacts_dir)
    assert train_path is not None
    assert artifacts_root is not None

    alpha_grid = np.linspace(config.pricing_alpha_start, config.pricing_alpha_stop, config.pricing_alpha_num)
    beta_grid = np.linspace(config.pricing_beta_start, config.pricing_beta_stop, max(1, config.pricing_beta_num))
    target_band = _resolve_target_band(config)
    prepared: PreparedData | None = None
    prepared_splits: list[PreparedSplitCache] = []
    stability_metadata: dict[str, Any] | None = None
    baseline_payload: dict[str, Any] = {}
    baseline_policy_score: float | None = None
    fixed_catboost_params: dict[str, Any] = _resolve_catboost_base_params(config, mode=args.mode, candidate=args.candidate)
    if args.catboost_threads is not None:
        if int(args.catboost_threads) <= 0:
            raise ValueError("--catboost_threads must be > 0")
        fixed_catboost_params["thread_count"] = int(args.catboost_threads)

    if args.objective_mode == "single_split":
        prepared = _prepare_data(config=config, train_path=train_path, artifacts_root=artifacts_root)
        baseline_cfg = _build_benchmark_cfg_from_run_config(config, "baseline_freq_sev")
        baseline_result = run_model_benchmark(
            train_df=prepared.train_df,
            valid_df=prepared.valid_df,
            benchmark_config=baseline_cfg,
            pricing_target_lr=config.pricing_target_lr,
            pricing_alpha_grid=alpha_grid,
            pricing_beta_grid=beta_grid,
            pricing_target_band=target_band,
            model_max_iter=config.model_max_iter,
            model_ridge_alpha=config.model_ridge_alpha,
        )
        baseline_candidate = baseline_result.results[0]
        baseline_payload = baseline_candidate.to_dict()
        baseline_policy_score_raw = (baseline_candidate.pricing or {}).get("policy_score")
        baseline_policy_score = float(baseline_policy_score_raw) if baseline_policy_score_raw is not None else None
    else:
        splits_config_path = args.splits_config.resolve()
        prepared_splits, stability_metadata = _prepare_stability_cache(
            config=config,
            train_path=train_path,
            artifacts_root=artifacts_root,
            splits_config=splits_config_path,
        )
        baseline_cfg = _build_benchmark_cfg_from_run_config(config, "baseline_freq_sev")
        baseline_candidates: list[CandidateResult] = []
        for split_ref in prepared_splits:
            split_train_df, split_valid_df = _load_cached_split(split_ref)
            split_result = run_model_benchmark(
                train_df=split_train_df,
                valid_df=split_valid_df,
                benchmark_config=baseline_cfg,
                pricing_target_lr=config.pricing_target_lr,
                pricing_alpha_grid=alpha_grid,
                pricing_beta_grid=beta_grid,
                pricing_target_band=target_band,
                model_max_iter=config.model_max_iter,
                model_ridge_alpha=config.model_ridge_alpha,
            )
            baseline_candidates.append(split_result.results[0])
        baseline_objective, baseline_attrs = _objective_value_from_stability_candidates(
            split_values=[item.split_value for item in prepared_splits],
            candidates=baseline_candidates,
            stability_lambda=float(args.stability_lambda),
            min_constraints_rate=float(args.stability_min_constraints_rate),
            min_in_target_rate=float(args.stability_min_in_target_rate),
        )
        baseline_payload = {
            "objective_mode": "stability_adjusted",
            "objective_value": baseline_objective,
            **baseline_attrs,
        }
        baseline_policy_score_raw = baseline_attrs.get("policy_score")
        baseline_policy_score = float(baseline_policy_score_raw) if baseline_policy_score_raw is not None else None

    seed = int(args.seed if args.seed is not None else config.benchmark.random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)

    storage_url: str | None = None
    if args.storage:
        storage_path = Path(args.storage)
        if not storage_path.is_absolute():
            storage_path = (artifacts_root / "tuning" / "catboost" / storage_path).resolve()
        ensure_dir(storage_path.parent)
        storage_url = f"sqlite:///{storage_path.as_posix()}"

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = dict(fixed_catboost_params)
        params.update(_suggest_catboost_params(trial, phase=args.phase, mode=args.mode))
        cfg = _build_benchmark_cfg_from_run_config(config, args.candidate, params=params)
        started = time.perf_counter()

        if args.objective_mode == "single_split":
            assert prepared is not None
            try:
                run = run_model_benchmark(
                    train_df=prepared.train_df,
                    valid_df=prepared.valid_df,
                    benchmark_config=cfg,
                    pricing_target_lr=config.pricing_target_lr,
                    pricing_alpha_grid=alpha_grid,
                    pricing_beta_grid=beta_grid,
                    pricing_target_band=target_band,
                    model_max_iter=config.model_max_iter,
                    model_ridge_alpha=config.model_ridge_alpha,
                    pricing_optimization_method=config.pricing_optimization_method,
                    pricing_retention=config.pricing_retention,
                    pricing_slsqp_options={
                        "maxiter": int(config.pricing_slsqp_maxiter),
                        "ftol": float(config.pricing_slsqp_ftol),
                        "eps": float(config.pricing_slsqp_eps),
                    },
                    pricing_stratified_config=config.pricing_stratified,
                )
                candidate = run.results[0]
            except Exception as exc:  # pragma: no cover
                trial.set_user_attr("status", "failed")
                trial.set_user_attr("error", str(exc))
                trial.set_user_attr("elapsed_seconds", float(time.perf_counter() - started))
                return -2_000_000.0

            for key, value in _candidate_to_attrs(candidate).items():
                trial.set_user_attr(key, value)

            objective_value = _objective_value_from_candidate(candidate)
        else:
            try:
                split_candidates: list[CandidateResult] = []
                for split_ref in prepared_splits:
                    split_train_df, split_valid_df = _load_cached_split(split_ref)
                    split_run = run_model_benchmark(
                        train_df=split_train_df,
                        valid_df=split_valid_df,
                        benchmark_config=cfg,
                        pricing_target_lr=config.pricing_target_lr,
                        pricing_alpha_grid=alpha_grid,
                        pricing_beta_grid=beta_grid,
                        pricing_target_band=target_band,
                        model_max_iter=config.model_max_iter,
                        model_ridge_alpha=config.model_ridge_alpha,
                        pricing_optimization_method=config.pricing_optimization_method,
                        pricing_retention=config.pricing_retention,
                        pricing_slsqp_options={
                            "maxiter": int(config.pricing_slsqp_maxiter),
                            "ftol": float(config.pricing_slsqp_ftol),
                            "eps": float(config.pricing_slsqp_eps),
                        },
                        pricing_stratified_config=config.pricing_stratified,
                    )
                    split_candidates.append(split_run.results[0])
            except Exception as exc:  # pragma: no cover
                trial.set_user_attr("status", "failed")
                trial.set_user_attr("error", str(exc))
                trial.set_user_attr("elapsed_seconds", float(time.perf_counter() - started))
                return -2_000_000.0

            objective_value, stability_attrs = _objective_value_from_stability_candidates(
                split_values=[item.split_value for item in prepared_splits],
                candidates=split_candidates,
                stability_lambda=float(args.stability_lambda),
                min_constraints_rate=float(args.stability_min_constraints_rate),
                min_in_target_rate=float(args.stability_min_in_target_rate),
            )
            for key, value in stability_attrs.items():
                trial.set_user_attr(key, value)

        trial.set_user_attr("objective_value", objective_value)
        trial.set_user_attr("elapsed_seconds", float(time.perf_counter() - started))

        trial.report(objective_value, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return objective_value

    study.optimize(
        objective,
        n_trials=max(1, int(args.n_trials)),
        timeout=args.timeout_sec,
        n_jobs=max(1, int(args.n_jobs)),
        show_progress_bar=False,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tuning_dir = ensure_dir(artifacts_root / "tuning" / "catboost" / ts)
    results_csv = tuning_dir / "results.csv"
    best_params_json = tuning_dir / "best_params.json"
    summary_json = tuning_dir / "summary.json"
    study_trials_json = tuning_dir / "study_trials.json"
    best_5_trials_summary_csv = tuning_dir / "best_5_trials_summary.csv"

    rows = _study_to_rows(study)
    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(by=["objective_value", "trial_number"], ascending=[False, True])
        top5_cols = [
            "trial_number",
            "objective_value",
            "passes_constraints",
            "policy_score",
            "policy_score_mean",
            "policy_score_std",
            "in_target",
            "in_target_rate",
            "constraints_pass_rate",
            "distance_to_target",
            "violations",
            "lr_total",
            "gini",
            "elapsed_seconds",
        ]
        top5_existing = [col for col in top5_cols if col in results_df.columns]
        results_df.head(5)[top5_existing].to_csv(best_5_trials_summary_csv, index=False)
    else:
        pd.DataFrame(columns=["trial_number", "objective_value"]).to_csv(best_5_trials_summary_csv, index=False)
    results_df.to_csv(results_csv, index=False)
    study_trials_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    best_params: dict[str, Any] = {}
    best_trial_payload: dict[str, Any] | None = None
    try:
        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        best_trial_payload = {
            "number": int(best_trial.number),
            "value": float(best_trial.value) if best_trial.value is not None else None,
            "params": best_params,
            "user_attrs": dict(best_trial.user_attrs),
        }
    except ValueError:
        best_trial_payload = None

    best_params_json.write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    best_policy_score = best_trial_payload["user_attrs"].get("policy_score") if best_trial_payload else None
    delta_policy_score = None
    if baseline_policy_score is not None and best_policy_score is not None:
        delta_policy_score = float(best_policy_score) - float(baseline_policy_score)

    constraints_pass_rate = None
    if not results_df.empty and "passes_constraints" in results_df.columns:
        constraints_pass_rate = float(pd.to_numeric(results_df["passes_constraints"], errors="coerce").fillna(0.0).mean())

    summary = {
        "timestamp_utc": ts,
        "config_path": str(config_path),
        "study_name": study.study_name,
        "seed": seed,
        "phase": args.phase,
        "mode": args.mode,
        "candidate": args.candidate,
        "catboost_threads": int(args.catboost_threads) if args.catboost_threads is not None else None,
        "fixed_catboost_params": fixed_catboost_params,
        "objective_mode": args.objective_mode,
        "stability_objective": {
            "lambda": float(args.stability_lambda),
            "min_constraints_rate": float(args.stability_min_constraints_rate),
            "min_in_target_rate": float(args.stability_min_in_target_rate),
            "splits_config_path": str(args.splits_config.resolve()) if args.objective_mode == "stability_adjusted" else None,
        },
        "n_trials_requested": int(args.n_trials),
        "n_trials_finished": int(len(study.trials)),
        "timeout_sec": args.timeout_sec,
        "storage_url": storage_url,
        "split_meta": (
            prepared.split_meta
            if prepared is not None
            else {
                "objective_mode": "stability_adjusted",
                "split_values": [item.split_value for item in prepared_splits],
            }
        ),
        "preprocessing": {
            "raw_rows": int(prepared.raw_rows) if prepared is not None else int((stability_metadata or {}).get("raw_rows", 0)),
            "policy_rows": int(prepared.policy_rows) if prepared is not None else int((stability_metadata or {}).get("policy_rows", 0)),
            "feature_count": int(prepared.feature_count)
            if prepared is not None
            else int((stability_metadata or {}).get("feature_count", 0)),
            "cache_dir": str(prepared.cache_dir)
            if prepared is not None
            else str((stability_metadata or {}).get("cache_dir", "")),
            "n_splits": int((stability_metadata or {}).get("n_splits", 1)) if prepared is None else 1,
        },
        "baseline": baseline_payload,
        "best_trial": best_trial_payload,
        "delta_vs_baseline_policy_score": delta_policy_score,
        "constraints_pass_rate": constraints_pass_rate,
        "results_csv_path": str(results_csv),
        "study_trials_json_path": str(study_trials_json),
        "best_params_json_path": str(best_params_json),
        "best_5_trials_summary_csv_path": str(best_5_trials_summary_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "tuning_dir": str(tuning_dir),
                "results_csv": str(results_csv),
                "summary_json": str(summary_json),
                "best_params_json": str(best_params_json),
                "best_5_trials_summary_csv": str(best_5_trials_summary_csv),
                "baseline_policy_score": baseline_policy_score,
                "best_trial_policy_score": best_policy_score,
                "delta_vs_baseline_policy_score": delta_policy_score,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
