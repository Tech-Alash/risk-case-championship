from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fnmatch
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from risk_case.settings import (
    CONTRACT_COL,
    DEFAULT_FORBIDDEN_FEATURE_COLUMNS,
    DEFAULT_TARGET_COLUMNS,
    PREMIUM_COL,
    PREMIUM_NET_COL,
    TARGET_CLAIM_COL,
)

LOGGER = logging.getLogger("risk_case.pipeline.preprocessing")

INTERACTION_MVP_BUSINESS_WHITELIST: tuple[str, ...] = (
    "score_x_premium_ratio",
    "score_x_bonus_malus",
    "score_x_car_age",
    "score_group_mean_diff",
    "premium_x_power",
    "premium_wo_term_x_power",
    "car_age_x_region_risk",
    "bm_x_region_risk",
    "score_missing_cnt_x_score_mean",
    "score_dispersion_x_premium",
)

DEFAULT_INTERACTION_MVP_DEFINITIONS: tuple[str, ...] = INTERACTION_MVP_BUSINESS_WHITELIST

INTERACTION_REQUIRED_SOURCES: dict[str, set[str]] = {
    "premium_per_driver": {"premium", "driver_count"},
    "premium_wo_term_per_driver": {"premium_wo_term", "driver_count"},
    "premium_per_power": {"premium", "engine_power"},
    "premium_wo_term_per_power": {"premium_wo_term", "engine_power"},
    "car_age_x_bonus_malus": {"car_age", "bonus_malus"},
    "region_x_vehicle_type": {"region_name", "vehicle_type_name"},
    "score_x_premium_ratio": {"score_stats", "premium"},
    "score_x_bonus_malus": {"score_stats", "bonus_malus"},
    "score_x_car_age": {"score_stats", "car_age"},
    "score_group_mean_diff": {"score_stats"},
    "premium_x_power": {"premium", "engine_power"},
    "premium_wo_term_x_power": {"premium_wo_term", "engine_power"},
    "car_age_x_region_risk": {"car_age", "region_risk"},
    "bm_x_region_risk": {"bonus_malus", "region_risk"},
    "score_missing_cnt_x_score_mean": {"score_stats", "score_missing_cnt_total"},
    "score_dispersion_x_premium": {"score_stats", "premium"},
}


@dataclass
class PreprocessingConfig:
    grain: str = CONTRACT_COL
    add_missing_flags: bool = True
    add_missing_aggregates: bool = True
    missing_aggregate_prefixes: list[str] = field(default_factory=lambda: ["SCORE_4_", "SCORE_11_", "SCORE_"])
    missing_flag_threshold: float = 0.05
    numeric_default_strategy: str = "median"
    financial_fill_value: float = 0.0
    winsorize_low: float = 0.01
    winsorize_high: float = 0.99
    rare_category_threshold: float = 0.005
    rare_category_min_count: int = 200
    date_columns: list[str] = field(default_factory=lambda: ["operation_date"])
    date_features: list[str] = field(
        default_factory=lambda: ["month", "quarter", "dayofweek", "is_month_end", "sin_month", "cos_month"]
    )
    target_encoding_enabled: bool = True
    target_encoding_columns: list[str] = field(
        default_factory=lambda: ["model", "mark", "ownerkato", "region_name", "car_year", "bonus_malus"]
    )
    target_encoding_smoothing: float = 20.0
    target_encoding_min_samples_leaf: int = 100
    target_encoding_noise_std: float = 0.0
    frequency_encoding_enabled: bool = True
    frequency_encoding_columns: list[str] = field(
        default_factory=lambda: ["model", "mark", "ownerkato", "region_name", "car_year", "bonus_malus"]
    )
    interaction_features_enabled: bool = True
    interaction_features: list[str] = field(
        default_factory=lambda: [
            "premium_per_driver",
            "premium_wo_term_per_driver",
            "premium_per_power",
            "premium_wo_term_per_power",
            "car_age_x_bonus_malus",
            "region_x_vehicle_type",
        ]
    )
    interaction_features_mvp_enabled: bool = False
    interaction_features_mvp_definitions: list[str] = field(default_factory=lambda: list(DEFAULT_INTERACTION_MVP_DEFINITIONS))
    interaction_features_mvp_max_features: int = 12
    interaction_features_mvp_corr_filter_threshold: float = 0.995
    interaction_features_mvp_psi_filter_threshold: float = 0.6
    interaction_features_mvp_require_business_whitelist: bool = True
    feature_generation_version: str = "v2_deep_features"
    log1p_columns: list[str] = field(default_factory=lambda: [PREMIUM_COL, PREMIUM_NET_COL])
    drop_columns: list[str] = field(default_factory=lambda: ["unique_id", "driver_iin", "insurer_iin", "car_number"])
    forbidden_feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_FORBIDDEN_FEATURE_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_COLUMNS))
    progress_log_every_n: int = 20
    feature_whitelist_path: str | None = None
    feature_droplist_path: str | None = None
    force_keep_features: list[str] = field(default_factory=lambda: [PREMIUM_COL, PREMIUM_NET_COL])
    force_drop_features: list[str] = field(default_factory=list)
    feature_pruning_enabled: bool = False
    feature_pruning_drop_exact_duplicates: bool = True
    feature_pruning_drop_missing_share: bool = True
    feature_pruning_corr_threshold: float = 0.995
    drift_pruning_enabled: bool = False
    drift_pruning_time_column: str = "operation_date"
    drift_pruning_reference_share: float = 0.7
    drift_pruning_psi_threshold: float = 0.25
    drift_pruning_bins: int = 10
    drift_pruning_min_rows: int = 500
    drift_pruning_exclude_columns: list[str] = field(default_factory=list)
    drift_pruning_exclude_patterns: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(raw: dict[str, Any] | None) -> "PreprocessingConfig":
        if not raw:
            return PreprocessingConfig()

        missing_cfg = raw.get("missing", {})
        outliers_cfg = raw.get("outliers", {}).get("winsorize", {})
        cat_cfg = raw.get("categorical", {})
        transforms_cfg = raw.get("transforms", {})
        selection_cfg = raw.get("selection_rules", {})
        date_cfg = raw.get("date_features", {})
        target_encoding_cfg = raw.get("target_encoding", {})
        freq_encoding_cfg = raw.get("freq_encoding", {})
        interactions_cfg = raw.get("interaction_features", {})
        interaction_mvp_cfg = raw.get("interaction_features_mvp", {})
        feature_pruning_cfg = raw.get("feature_pruning", {})
        drift_pruning_cfg = raw.get("drift_pruning", {})

        return PreprocessingConfig(
            grain=str(raw.get("grain", CONTRACT_COL)),
            add_missing_flags=bool(missing_cfg.get("add_missing_flags", True)),
            add_missing_aggregates=bool(missing_cfg.get("add_missing_aggregates", True)),
            missing_aggregate_prefixes=list(
                missing_cfg.get("missing_aggregate_prefixes", ["SCORE_4_", "SCORE_11_", "SCORE_"])
            ),
            missing_flag_threshold=float(missing_cfg.get("missing_flag_threshold", 0.05)),
            numeric_default_strategy=str(missing_cfg.get("numeric_default", "median")),
            financial_fill_value=float(missing_cfg.get("financial_fill", 0.0)),
            winsorize_low=float(outliers_cfg.get("low", 0.01)),
            winsorize_high=float(outliers_cfg.get("high", 0.99)),
            rare_category_threshold=float(cat_cfg.get("rare_threshold", 0.005)),
            rare_category_min_count=int(cat_cfg.get("rare_min_count", 200)),
            date_columns=list(date_cfg.get("columns", ["operation_date"])),
            date_features=list(
                date_cfg.get("features", ["month", "quarter", "dayofweek", "is_month_end", "sin_month", "cos_month"])
            ),
            target_encoding_enabled=bool(target_encoding_cfg.get("enabled", True)),
            target_encoding_columns=list(
                target_encoding_cfg.get(
                    "columns",
                    ["model", "mark", "ownerkato", "region_name", "car_year", "bonus_malus"],
                )
            ),
            target_encoding_smoothing=float(target_encoding_cfg.get("smoothing", 20.0)),
            target_encoding_min_samples_leaf=int(target_encoding_cfg.get("min_samples_leaf", 100)),
            target_encoding_noise_std=float(target_encoding_cfg.get("noise_std", 0.0)),
            frequency_encoding_enabled=bool(freq_encoding_cfg.get("enabled", True)),
            frequency_encoding_columns=list(
                freq_encoding_cfg.get(
                    "columns",
                    ["model", "mark", "ownerkato", "region_name", "car_year", "bonus_malus"],
                )
            ),
            interaction_features_enabled=bool(interactions_cfg.get("enabled", True)),
            interaction_features=list(
                interactions_cfg.get(
                    "definitions",
                    [
                        "premium_per_driver",
                        "premium_wo_term_per_driver",
                        "premium_per_power",
                        "premium_wo_term_per_power",
                        "car_age_x_bonus_malus",
                        "region_x_vehicle_type",
                    ],
                )
            ),
            interaction_features_mvp_enabled=bool(interaction_mvp_cfg.get("enabled", False)),
            interaction_features_mvp_definitions=[
                str(item)
                for item in interaction_mvp_cfg.get("definitions", list(DEFAULT_INTERACTION_MVP_DEFINITIONS))
                if str(item).strip()
            ],
            interaction_features_mvp_max_features=int(interaction_mvp_cfg.get("max_features", 12)),
            interaction_features_mvp_corr_filter_threshold=float(
                interaction_mvp_cfg.get("corr_filter_threshold", 0.995)
            ),
            interaction_features_mvp_psi_filter_threshold=float(
                interaction_mvp_cfg.get("psi_filter_threshold", 0.6)
            ),
            interaction_features_mvp_require_business_whitelist=bool(
                interaction_mvp_cfg.get("require_business_whitelist", True)
            ),
            feature_generation_version=str(raw.get("feature_generation_version", "v2_deep_features")),
            log1p_columns=list(transforms_cfg.get("log1p_columns", [PREMIUM_COL, PREMIUM_NET_COL])),
            drop_columns=list(raw.get("drop_columns", ["unique_id", "driver_iin", "insurer_iin", "car_number"])),
            forbidden_feature_columns=list(
                raw.get("forbidden_feature_columns", list(DEFAULT_FORBIDDEN_FEATURE_COLUMNS))
            ),
            target_columns=list(raw.get("target_columns", list(DEFAULT_TARGET_COLUMNS))),
            progress_log_every_n=int(raw.get("progress_log_every_n", 20)),
            feature_whitelist_path=raw.get("feature_whitelist_path"),
            feature_droplist_path=raw.get("feature_droplist_path"),
            force_keep_features=list(selection_cfg.get("force_keep", [PREMIUM_COL, PREMIUM_NET_COL])),
            force_drop_features=list(selection_cfg.get("force_drop", [])),
            feature_pruning_enabled=bool(feature_pruning_cfg.get("enabled", False)),
            feature_pruning_drop_exact_duplicates=bool(feature_pruning_cfg.get("drop_exact_duplicates", True)),
            feature_pruning_drop_missing_share=bool(feature_pruning_cfg.get("drop_missing_share", True)),
            feature_pruning_corr_threshold=float(feature_pruning_cfg.get("corr_threshold", 0.995)),
            drift_pruning_enabled=bool(drift_pruning_cfg.get("enabled", False)),
            drift_pruning_time_column=str(drift_pruning_cfg.get("time_column", "operation_date")),
            drift_pruning_reference_share=float(drift_pruning_cfg.get("reference_share", 0.7)),
            drift_pruning_psi_threshold=float(drift_pruning_cfg.get("psi_threshold", 0.25)),
            drift_pruning_bins=int(drift_pruning_cfg.get("bins", 10)),
            drift_pruning_min_rows=int(drift_pruning_cfg.get("min_rows", 500)),
            drift_pruning_exclude_columns=[str(col) for col in drift_pruning_cfg.get("exclude_columns", [])],
            drift_pruning_exclude_patterns=[str(pattern) for pattern in drift_pruning_cfg.get("exclude_patterns", [])],
        )


@dataclass
class FittedPreprocessor:
    config: PreprocessingConfig
    numeric_columns: list[str]
    categorical_columns: list[str]
    date_input_columns: list[str]
    date_feature_columns: list[str]
    categorical_output_columns: list[str]
    numeric_fill_values: dict[str, float]
    winsor_bounds: dict[str, tuple[float, float]]
    missing_flag_columns: list[str]
    missing_aggregate_definitions: dict[str, list[str]]
    rare_category_kept: dict[str, list[str]]
    target_encoding_maps: dict[str, dict[str, float]]
    target_encoding_global_mean: float
    frequency_encoding_maps: dict[str, dict[str, float]]
    interaction_feature_columns: list[str]
    feature_generation_version: str
    feature_columns: list[str]
    feature_pruning_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["winsor_bounds"] = {
            key: [float(value[0]), float(value[1])] for key, value in self.winsor_bounds.items()
        }
        return payload


def _protect_columns(config: PreprocessingConfig) -> set[str]:
    return {config.grain, *config.target_columns}


def _choose_fill_value(series: pd.Series, config: PreprocessingConfig, column: str) -> float:
    if column in {PREMIUM_COL, PREMIUM_NET_COL, "claim_amount", "claim_cnt"}:
        return float(config.financial_fill_value)

    numeric = pd.to_numeric(series, errors="coerce")
    if config.numeric_default_strategy == "zero":
        return 0.0

    median = numeric.median()
    if pd.isna(median):
        return 0.0
    return float(median)


def _clip_quantiles(values: pd.Series, low: float, high: float) -> tuple[float, float]:
    if values.dropna().empty:
        return 0.0, 0.0
    q_low = float(values.quantile(low))
    q_high = float(values.quantile(high))
    if q_low > q_high:
        q_low, q_high = q_high, q_low
    return q_low, q_high


def _load_feature_list(path: str | None) -> set[str]:
    if not path:
        return set()
    feature_path = Path(path)
    if not feature_path.exists():
        LOGGER.warning("Feature list path does not exist: %s", feature_path)
        return set()
    try:
        df = pd.read_csv(feature_path)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not read feature list %s: %s", feature_path, exc)
        return set()
    if "feature" not in df.columns:
        LOGGER.warning("Feature list %s missing 'feature' column", feature_path)
        return set()
    return set(df["feature"].dropna().astype(str).tolist())


def _normalize_series_to_category(series: pd.Series, kept: list[str] | None) -> pd.Series:
    normalized = series.astype("string").fillna("missing").astype(str)
    kept_set = set(kept or [])
    if kept_set:
        normalized = normalized.where(normalized.isin(kept_set), "other")
    return normalized


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _safe_numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for candidate in candidates:
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce")
    return None


def _safe_divide_series(numerator: pd.Series, denominator: pd.Series, fallback: float = 1.0) -> pd.Series:
    denominator_values = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    denominator_safe = np.where(np.abs(denominator_values.to_numpy(dtype=float)) <= 1e-9, fallback, denominator_values)
    result = pd.to_numeric(numerator, errors="coerce").fillna(0.0).to_numpy(dtype=float) / denominator_safe
    return pd.Series(result, index=numerator.index, dtype=float)


def _compute_score_statistics(df: pd.DataFrame) -> dict[str, pd.Series | None]:
    score_columns = [
        col
        for col in df.columns
        if col.startswith("SCORE_")
        and col not in {TARGET_CLAIM_COL, "claim_amount", "claim_cnt"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not score_columns:
        return {"score_mean": None, "score_std": None, "score_median": None}
    score_frame = df[score_columns].apply(pd.to_numeric, errors="coerce")
    return {
        "score_mean": score_frame.mean(axis=1, skipna=True).fillna(0.0),
        "score_std": score_frame.std(axis=1, ddof=0, skipna=True).fillna(0.0),
        "score_median": score_frame.median(axis=1, skipna=True).fillna(0.0),
    }


def _build_interaction_sources(df: pd.DataFrame) -> dict[str, pd.Series | None]:
    score_stats = _compute_score_statistics(df)
    return {
        "premium": _safe_numeric_series(df, [PREMIUM_COL]),
        "premium_wo_term": _safe_numeric_series(df, [PREMIUM_NET_COL]),
        "driver_count": _safe_numeric_series(df, ["driver_count"]),
        "engine_power": _safe_numeric_series(df, ["engine_power_mean", "engine_power"]),
        "car_age": _safe_numeric_series(df, ["car_age_mean", "car_age"]),
        "bonus_malus": _safe_numeric_series(df, ["bonus_malus_mean", "bonus_malus"]),
        "region_risk": _safe_numeric_series(df, ["region_name_te", "region_name_freq"]),
        "score_mean": score_stats["score_mean"],
        "score_std": score_stats["score_std"],
        "score_median": score_stats["score_median"],
        "score_missing_cnt_total": _safe_numeric_series(df, ["score_missing_cnt_total"]),
    }


def _available_interaction_sources(
    *,
    working: pd.DataFrame,
    categorical_columns: list[str],
    target_encoding_maps: dict[str, dict[str, float]],
    frequency_encoding_maps: dict[str, dict[str, float]],
    missing_aggregate_definitions: dict[str, list[str]],
) -> set[str]:
    score_stats = _compute_score_statistics(working)
    available: set[str] = set()
    if _safe_numeric_series(working, [PREMIUM_COL]) is not None:
        available.add("premium")
    if _safe_numeric_series(working, [PREMIUM_NET_COL]) is not None:
        available.add("premium_wo_term")
    if _safe_numeric_series(working, ["driver_count"]) is not None:
        available.add("driver_count")
    if _safe_numeric_series(working, ["engine_power_mean", "engine_power"]) is not None:
        available.add("engine_power")
    if _safe_numeric_series(working, ["car_age_mean", "car_age"]) is not None:
        available.add("car_age")
    if _safe_numeric_series(working, ["bonus_malus_mean", "bonus_malus"]) is not None:
        available.add("bonus_malus")
    if (
        "region_name" in target_encoding_maps
        or "region_name" in frequency_encoding_maps
        or _safe_numeric_series(working, ["region_name_te", "region_name_freq"]) is not None
    ):
        available.add("region_risk")
    if score_stats["score_mean"] is not None:
        available.add("score_stats")
    if "score_missing_cnt_total" in missing_aggregate_definitions:
        available.add("score_missing_cnt_total")
    if "region_name" in categorical_columns:
        available.add("region_name")
    if "vehicle_type_name" in categorical_columns:
        available.add("vehicle_type_name")
    return available


def _compute_interaction_feature_series(
    *,
    feature_name: str,
    df: pd.DataFrame,
    sources: dict[str, pd.Series | None],
) -> pd.Series | None:
    premium = sources.get("premium")
    premium_net = sources.get("premium_wo_term")
    driver_count = sources.get("driver_count")
    engine_power = sources.get("engine_power")
    car_age = sources.get("car_age")
    bonus_malus = sources.get("bonus_malus")
    region_risk = sources.get("region_risk")
    score_mean = sources.get("score_mean")
    score_std = sources.get("score_std")
    score_median = sources.get("score_median")
    score_missing_cnt_total = sources.get("score_missing_cnt_total")

    if feature_name == "premium_per_driver":
        if premium is None or driver_count is None:
            return None
        return _safe_divide_series(premium, driver_count, fallback=1.0)
    if feature_name == "premium_wo_term_per_driver":
        if premium_net is None or driver_count is None:
            return None
        return _safe_divide_series(premium_net, driver_count, fallback=1.0)
    if feature_name == "premium_per_power":
        if premium is None or engine_power is None:
            return None
        return _safe_divide_series(premium, engine_power, fallback=1.0)
    if feature_name == "premium_wo_term_per_power":
        if premium_net is None or engine_power is None:
            return None
        return _safe_divide_series(premium_net, engine_power, fallback=1.0)
    if feature_name == "car_age_x_bonus_malus":
        if car_age is None or bonus_malus is None:
            return None
        return car_age.fillna(0.0) * bonus_malus.fillna(0.0)
    if feature_name == "region_x_vehicle_type":
        if "region_name" not in df.columns or "vehicle_type_name" not in df.columns:
            return None
        return (
            df["region_name"].astype("string").fillna("missing").astype(str)
            + "__"
            + df["vehicle_type_name"].astype("string").fillna("missing").astype(str)
        )
    if feature_name == "score_x_premium_ratio":
        if score_mean is None or premium is None:
            return None
        return _safe_divide_series(score_mean, premium, fallback=1.0)
    if feature_name == "score_x_bonus_malus":
        if score_mean is None or bonus_malus is None:
            return None
        return score_mean.fillna(0.0) * bonus_malus.fillna(0.0)
    if feature_name == "score_x_car_age":
        if score_mean is None or car_age is None:
            return None
        return score_mean.fillna(0.0) * car_age.fillna(0.0)
    if feature_name == "score_group_mean_diff":
        if score_mean is None or score_median is None:
            return None
        return score_mean.fillna(0.0) - score_median.fillna(0.0)
    if feature_name == "premium_x_power":
        if premium is None or engine_power is None:
            return None
        return premium.fillna(0.0) * engine_power.fillna(0.0)
    if feature_name == "premium_wo_term_x_power":
        if premium_net is None or engine_power is None:
            return None
        return premium_net.fillna(0.0) * engine_power.fillna(0.0)
    if feature_name == "car_age_x_region_risk":
        if car_age is None or region_risk is None:
            return None
        return car_age.fillna(0.0) * region_risk.fillna(0.0)
    if feature_name == "bm_x_region_risk":
        if bonus_malus is None or region_risk is None:
            return None
        return bonus_malus.fillna(0.0) * region_risk.fillna(0.0)
    if feature_name == "score_missing_cnt_x_score_mean":
        if score_missing_cnt_total is None or score_mean is None:
            return None
        return score_missing_cnt_total.fillna(0.0) * score_mean.fillna(0.0)
    if feature_name == "score_dispersion_x_premium":
        if score_std is None or premium is None:
            return None
        return score_std.fillna(0.0) * premium.fillna(0.0)
    return None


def _filter_mvp_interaction_features(
    *,
    transformed_df: pd.DataFrame,
    feature_columns: list[str],
    mvp_candidates: list[str],
    config: PreprocessingConfig,
    source_df: pd.DataFrame | None = None,
) -> tuple[list[str], dict[str, Any]]:
    report: dict[str, Any] = {
        "enabled": bool(config.interaction_features_mvp_enabled),
        "definitions": list(mvp_candidates),
        "generated": [],
        "dropped_by_corr": [],
        "dropped_by_psi": [],
        "dropped_by_rank": [],
        "retained": [],
        "max_features": int(max(config.interaction_features_mvp_max_features, 0)),
        "corr_filter_threshold": float(max(config.interaction_features_mvp_corr_filter_threshold, 0.0)),
        "psi_filter_threshold": float(max(config.interaction_features_mvp_psi_filter_threshold, 0.0)),
    }
    if not mvp_candidates:
        return [], report

    generated = [name for name in mvp_candidates if name in transformed_df.columns]
    report["generated"] = list(generated)
    if not generated:
        return [], report

    corr_threshold = float(max(config.interaction_features_mvp_corr_filter_threshold, 0.0))
    psi_threshold = float(max(config.interaction_features_mvp_psi_filter_threshold, 0.0))
    max_features = int(max(config.interaction_features_mvp_max_features, 0))
    base_numeric_columns = [
        col
        for col in feature_columns
        if col not in set(generated)
        and col in transformed_df.columns
        and pd.api.types.is_numeric_dtype(transformed_df[col])
    ]
    target_series = (
        pd.to_numeric(transformed_df[TARGET_CLAIM_COL], errors="coerce").fillna(0.0)
        if TARGET_CLAIM_COL in transformed_df.columns
        else None
    )

    ordered_index, used_time_column = _resolve_time_order_index(source_df if source_df is not None else transformed_df, config)
    min_rows = max(int(config.drift_pruning_min_rows), 20)
    split_share = min(max(float(config.drift_pruning_reference_share), 0.5), 0.95)

    candidates_after_filters: list[dict[str, Any]] = []
    for name in generated:
        series = pd.to_numeric(transformed_df[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        best_corr = 0.0
        best_corr_feature: str | None = None
        for base_col in base_numeric_columns:
            base_series = pd.to_numeric(transformed_df[base_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            corr_value = series.corr(base_series)
            if pd.isna(corr_value):
                continue
            corr_abs = abs(float(corr_value))
            if corr_abs > best_corr:
                best_corr = corr_abs
                best_corr_feature = base_col
        if best_corr >= corr_threshold:
            report["dropped_by_corr"].append(
                {"feature": name, "max_abs_corr": best_corr, "with_feature": best_corr_feature, "threshold": corr_threshold}
            )
            continue

        psi_value: float | None = None
        if ordered_index is not None and len(ordered_index) >= (2 * min_rows):
            split_idx = int(round(len(ordered_index) * split_share))
            split_idx = min(max(split_idx, min_rows), max(len(ordered_index) - min_rows, 1))
            if split_idx > 0 and split_idx < len(ordered_index):
                ref_index = ordered_index[:split_idx]
                cur_index = ordered_index[split_idx:]
                psi_value = _population_stability_index(
                    reference=series.loc[ref_index],
                    current=series.loc[cur_index],
                    bins=max(int(config.drift_pruning_bins), 3),
                )
        if psi_value is not None and psi_value > psi_threshold:
            report["dropped_by_psi"].append(
                {
                    "feature": name,
                    "psi": float(psi_value),
                    "threshold": psi_threshold,
                    "time_column": used_time_column,
                }
            )
            continue

        target_corr_abs = 0.0
        if target_series is not None:
            target_corr = series.corr(target_series)
            if pd.notna(target_corr):
                target_corr_abs = abs(float(target_corr))
        candidates_after_filters.append(
            {
                "feature": name,
                "psi": float(psi_value) if psi_value is not None else float("inf"),
                "target_corr_abs": target_corr_abs,
            }
        )

    candidates_after_filters.sort(key=lambda item: (item["psi"], -item["target_corr_abs"], item["feature"]))
    retained = [item["feature"] for item in candidates_after_filters[:max_features]] if max_features > 0 else []
    dropped_by_rank = [item["feature"] for item in candidates_after_filters[max_features:]] if max_features > 0 else [
        item["feature"] for item in candidates_after_filters
    ]
    if dropped_by_rank:
        report["dropped_by_rank"] = dropped_by_rank
    report["retained"] = retained
    report["candidates_ranked"] = candidates_after_filters
    return retained, report


def _canonicalize_numeric_like_string(series: pd.Series, column: str | None = None) -> pd.Series:
    cleaned = series.astype("string")
    has_separator = cleaned.str.contains(r"[\u00a0\s]", regex=True, na=False)
    cleaned = cleaned.str.replace("\u00a0", "", regex=False)
    cleaned = cleaned.str.replace(" ", "", regex=False)
    cleaned = cleaned.str.strip()

    if column == "car_year":
        compact_year_mask = has_separator & cleaned.str.fullmatch(r"[12]\d{2}", na=False)
        if bool(compact_year_mask.any()):
            first = cleaned.str.slice(0, 1)
            suffix = cleaned.str.slice(1, 3)
            century_digit = first.map({"1": "9", "2": "0"}).fillna("")
            expanded = first + century_digit + suffix
            cleaned = cleaned.where(~compact_year_mask, expanded)

    lowered = cleaned.str.lower()
    cleaned = cleaned.mask(lowered.isin({"", "nan", "none", "null", "na", "n/a"}), pd.NA)
    return cleaned


def _clean_numeric_like_column(
    series: pd.Series,
    column: str,
    keep_numeric_dtype: bool,
) -> tuple[pd.Series, dict[str, int]]:
    non_null_mask = series.notna()
    raw_numeric = pd.to_numeric(series, errors="coerce")
    non_numeric_before = int((non_null_mask & raw_numeric.isna()).sum())

    canonical = _canonicalize_numeric_like_string(series, column=column)
    cleaned_numeric = pd.to_numeric(canonical, errors="coerce")
    rescued = int((non_null_mask & raw_numeric.isna() & cleaned_numeric.notna()).sum())
    coerced_to_missing = int((non_null_mask & cleaned_numeric.isna()).sum())

    if keep_numeric_dtype:
        cleaned = cleaned_numeric.astype(float)
    else:
        cleaned = pd.Series(pd.NA, index=series.index, dtype="string")
        valid_mask = cleaned_numeric.notna()
        if bool(valid_mask.any()):
            valid_numeric = cleaned_numeric.loc[valid_mask]
            rounded = np.round(valid_numeric.to_numpy())
            int_like = pd.Series(
                np.isclose(valid_numeric.to_numpy(), rounded, rtol=0.0, atol=1e-9),
                index=valid_numeric.index,
            )
            if bool(int_like.any()):
                int_values = valid_numeric.loc[int_like].round().astype("Int64").astype("string")
                cleaned.loc[int_values.index] = int_values
            float_like = ~int_like
            if bool(float_like.any()):
                float_values = valid_numeric.loc[float_like].map(lambda value: format(float(value), "g"))
                cleaned.loc[float_values.index] = float_values.astype("string")

    return cleaned, {
        "non_null": int(non_null_mask.sum()),
        "non_numeric_before": non_numeric_before,
        "rescued": rescued,
        "coerced_to_missing": coerced_to_missing,
    }


def _normalize_numeric_like_columns(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    for column in ("bonus_malus", "car_year"):
        if column not in df.columns:
            continue
        keep_numeric_dtype = bool(pd.api.types.is_numeric_dtype(df[column]))
        cleaned, stats = _clean_numeric_like_column(
            df[column],
            column=column,
            keep_numeric_dtype=keep_numeric_dtype,
        )
        df[column] = cleaned
        if stats["non_numeric_before"] > 0 or stats["rescued"] > 0:
            LOGGER.info(
                (
                    "Numeric-like cleanup[%s]: column=%s non_null=%d non_numeric_before=%d "
                    "rescued=%d coerced_to_missing=%d mode=%s"
                ),
                stage,
                column,
                stats["non_null"],
                stats["non_numeric_before"],
                stats["rescued"],
                stats["coerced_to_missing"],
                "numeric" if keep_numeric_dtype else "categorical",
            )
    return df


def _prefix_token(prefix: str) -> str:
    cleaned = prefix.lower().replace("score_", "score_")
    cleaned = cleaned.strip("_")
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in cleaned)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "missing"


def _fit_target_encoding_map(
    category: pd.Series,
    target: pd.Series,
    global_mean: float,
    smoothing: float,
    min_samples_leaf: int,
) -> dict[str, float]:
    frame = pd.DataFrame({"category": category.astype(str), "target": pd.to_numeric(target, errors="coerce").fillna(0.0)})
    grouped = frame.groupby("category", dropna=False)["target"].agg(["mean", "count"])
    counts = grouped["count"].astype(float)
    means = grouped["mean"].astype(float)
    smoothing_term = np.full(len(counts), max(smoothing, 0.0), dtype=float)
    if min_samples_leaf > 0:
        smoothing_term = np.where(counts < min_samples_leaf, smoothing_term * 2.0, smoothing_term)
    encoded = (means * counts + global_mean * smoothing_term) / np.where(counts + smoothing_term == 0, 1.0, counts + smoothing_term)
    return {str(k): float(v) for k, v in encoded.to_dict().items()}


def _fit_frequency_encoding_map(category: pd.Series) -> dict[str, float]:
    freq = category.astype(str).value_counts(dropna=False, normalize=True)
    return {str(k): float(v) for k, v in freq.to_dict().items()}


def _build_date_features(series: pd.Series, prefix: str, features: list[str]) -> tuple[dict[str, pd.Series], list[str]]:
    dt = _parse_datetime(series)
    generated: dict[str, pd.Series] = {}
    names: list[str] = []

    for feature_name in features:
        col_name = f"{prefix}_{feature_name}"
        if feature_name == "month":
            generated[col_name] = dt.dt.month.fillna(0).astype(float)
        elif feature_name == "quarter":
            generated[col_name] = dt.dt.quarter.fillna(0).astype(float)
        elif feature_name == "dayofweek":
            generated[col_name] = dt.dt.dayofweek.fillna(0).astype(float)
        elif feature_name == "is_month_end":
            generated[col_name] = dt.dt.is_month_end.fillna(False).astype(int)
        elif feature_name == "sin_month":
            month = dt.dt.month.fillna(0).astype(float)
            generated[col_name] = np.sin(2.0 * np.pi * month / 12.0)
        elif feature_name == "cos_month":
            month = dt.dt.month.fillna(0).astype(float)
            generated[col_name] = np.cos(2.0 * np.pi * month / 12.0)
        else:
            continue
        names.append(col_name)
    return generated, names


def _column_signature(series: pd.Series) -> str:
    fingerprint = pd.util.hash_pandas_object(series, index=False).values.tobytes()
    return hashlib.sha1(fingerprint).hexdigest()


def _population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10, eps: float = 1e-6) -> float | None:
    ref = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan)
    cur = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ref_valid = ref.dropna()
    cur_valid = cur.dropna()
    if len(ref_valid) < 2 or len(cur_valid) < 2:
        return None

    bins = max(int(bins), 3)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(ref_valid.to_numpy(dtype=float), quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        split_point = float(np.nanmedian(ref_valid.to_numpy(dtype=float)))
        edges = np.array([-np.inf, split_point, np.inf], dtype=float)
    else:
        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_valid.to_numpy(dtype=float), bins=edges)
    cur_counts, _ = np.histogram(cur_valid.to_numpy(dtype=float), bins=edges)

    ref_pct = ref_counts.astype(float) / max(float(ref_counts.sum()), 1.0)
    cur_pct = cur_counts.astype(float) / max(float(cur_counts.sum()), 1.0)

    # Include missingness as a dedicated bucket to capture null drift.
    ref_pct = np.append(ref_pct, float(ref.isna().mean()))
    cur_pct = np.append(cur_pct, float(cur.isna().mean()))

    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _resolve_time_order_index(df: pd.DataFrame, config: PreprocessingConfig) -> tuple[pd.Index | None, str | None]:
    candidate_columns: list[str] = []
    if config.drift_pruning_time_column:
        candidate_columns.append(config.drift_pruning_time_column)
    candidate_columns.extend([col for col in config.date_columns if col not in candidate_columns])

    for column in candidate_columns:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        valid = parsed.notna()
        if int(valid.sum()) < 2:
            continue
        ordered = parsed[valid].sort_values(kind="mergesort").index
        return ordered, column
    return None, None


def _prune_feature_columns_by_statistics(
    transformed_df: pd.DataFrame,
    feature_columns: list[str],
    config: PreprocessingConfig,
    source_df: pd.DataFrame | None = None,
) -> tuple[list[str], dict[str, Any]]:
    retained = [col for col in feature_columns if col in transformed_df.columns]
    dropped_exact_duplicates: list[dict[str, Any]] = []
    dropped_corr_pairs: list[dict[str, Any]] = []
    dropped_manual: list[str] = []
    dropped_drift_psi: list[dict[str, Any]] = []
    protected_from_pruning = {
        col
        for col in retained
        if col in set(config.force_keep_features) or col.endswith("_missing_cnt") or col.endswith("_missing_cnt_total")
    }

    numeric_columns = [col for col in retained if pd.api.types.is_numeric_dtype(transformed_df[col])]
    if config.feature_pruning_drop_exact_duplicates and numeric_columns:
        signature_to_columns: dict[str, list[str]] = {}
        for column in numeric_columns:
            signature = _column_signature(transformed_df[column])
            signature_to_columns.setdefault(signature, []).append(column)
        for columns in signature_to_columns.values():
            if len(columns) <= 1:
                continue
            ordered = sorted(columns, key=lambda col: (0 if col in protected_from_pruning else 1, col))
            keeper = ordered[0]
            for duplicate in ordered[1:]:
                if duplicate in protected_from_pruning:
                    continue
                if duplicate in retained:
                    retained.remove(duplicate)
                    dropped_exact_duplicates.append({"dropped": duplicate, "kept": keeper, "reason": "exact_duplicate"})

    if config.feature_pruning_drop_missing_share and "score_missing_share" in retained:
        retained.remove("score_missing_share")
        dropped_manual.append("score_missing_share")

    corr_threshold = float(config.feature_pruning_corr_threshold)
    if 0.0 < corr_threshold < 1.0:
        corr_numeric_columns = [col for col in retained if pd.api.types.is_numeric_dtype(transformed_df[col])]
        corr_numeric_columns = sorted(corr_numeric_columns, key=lambda col: (0 if col in protected_from_pruning else 1, col))
        if len(corr_numeric_columns) >= 2:
            corr_df = transformed_df[corr_numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            corr_matrix = corr_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop: set[str] = set()
            for keeper in corr_numeric_columns:
                if keeper in to_drop:
                    continue
                high_corr = upper.index[(upper[keeper] > corr_threshold).fillna(False)].tolist()
                for duplicate in high_corr:
                    if duplicate in to_drop:
                        continue
                    if duplicate in protected_from_pruning:
                        continue
                    value = upper.at[duplicate, keeper]
                    to_drop.add(duplicate)
                    dropped_corr_pairs.append(
                        {
                            "dropped": duplicate,
                            "kept": keeper,
                            "corr": float(value) if pd.notna(value) else None,
                            "reason": f"corr>{corr_threshold}",
                        }
                    )
            if to_drop:
                retained = [col for col in retained if col not in to_drop]

    drift_details: dict[str, Any] = {
        "enabled": bool(config.drift_pruning_enabled),
        "time_column": None,
        "reference_size": 0,
        "current_size": 0,
        "psi_threshold": float(config.drift_pruning_psi_threshold),
        "bins": int(config.drift_pruning_bins),
        "exclude_columns": sorted(set(config.drift_pruning_exclude_columns)),
        "exclude_patterns": [str(pattern) for pattern in config.drift_pruning_exclude_patterns if str(pattern).strip()],
        "excluded_columns": [],
    }
    if config.drift_pruning_enabled:
        drift_source = source_df if source_df is not None else transformed_df
        ordered_index, used_time_column = _resolve_time_order_index(drift_source, config)
        drift_details["time_column"] = used_time_column
        if ordered_index is None:
            drift_details["reason"] = "time_column_missing_or_unparseable"
        else:
            ordered_count = len(ordered_index)
            min_rows = max(int(config.drift_pruning_min_rows), 20)
            split_share = float(config.drift_pruning_reference_share)
            split_share = min(max(split_share, 0.5), 0.95)
            split_idx = int(round(ordered_count * split_share))
            split_idx = min(max(split_idx, min_rows), max(ordered_count - min_rows, 1))

            if split_idx <= 0 or split_idx >= ordered_count:
                drift_details["reason"] = "insufficient_rows_for_split"
            else:
                reference_index = ordered_index[:split_idx]
                current_index = ordered_index[split_idx:]
                drift_details["reference_size"] = int(len(reference_index))
                drift_details["current_size"] = int(len(current_index))

                drift_numeric_columns = [col for col in retained if pd.api.types.is_numeric_dtype(transformed_df[col])]
                drift_numeric_columns = sorted(
                    drift_numeric_columns,
                    key=lambda col: (0 if col in protected_from_pruning else 1, col),
                )
                psi_threshold = max(float(config.drift_pruning_psi_threshold), 0.0)
                bins = max(int(config.drift_pruning_bins), 3)
                exclude_columns = set(drift_details["exclude_columns"])
                exclude_patterns = drift_details["exclude_patterns"]
                excluded_columns: set[str] = set()

                for column in drift_numeric_columns:
                    if column in protected_from_pruning:
                        continue
                    if column in exclude_columns or any(fnmatch.fnmatch(column, pattern) for pattern in exclude_patterns):
                        excluded_columns.add(column)
                        continue
                    psi_value = _population_stability_index(
                        reference=transformed_df.loc[reference_index, column],
                        current=transformed_df.loc[current_index, column],
                        bins=bins,
                    )
                    if psi_value is None or psi_value <= psi_threshold:
                        continue
                    if column in retained:
                        retained.remove(column)
                    dropped_drift_psi.append(
                        {
                            "dropped": column,
                            "psi": float(psi_value),
                            "threshold": psi_threshold,
                            "reference_size": int(len(reference_index)),
                            "current_size": int(len(current_index)),
                            "time_column": used_time_column,
                            "reason": f"psi>{psi_threshold}",
                        }
                    )
                if excluded_columns:
                    drift_details["excluded_columns"] = sorted(excluded_columns)

    report = {
        "enabled": bool(config.feature_pruning_enabled or config.drift_pruning_enabled),
        "applied": bool(dropped_exact_duplicates or dropped_corr_pairs or dropped_manual or dropped_drift_psi),
        "before_count": len(feature_columns),
        "after_count": len(retained),
        "dropped_total": len(feature_columns) - len(retained),
        "settings": {
            "drop_exact_duplicates": bool(config.feature_pruning_drop_exact_duplicates),
            "drop_missing_share": bool(config.feature_pruning_drop_missing_share),
            "corr_threshold": corr_threshold,
            "drift_enabled": bool(config.drift_pruning_enabled),
            "drift_time_column": config.drift_pruning_time_column,
            "drift_reference_share": float(config.drift_pruning_reference_share),
            "drift_psi_threshold": float(config.drift_pruning_psi_threshold),
            "drift_bins": int(config.drift_pruning_bins),
            "drift_min_rows": int(config.drift_pruning_min_rows),
            "drift_exclude_columns": sorted(set(config.drift_pruning_exclude_columns)),
            "drift_exclude_patterns": [str(pattern) for pattern in config.drift_pruning_exclude_patterns if str(pattern).strip()],
        },
        "dropped_exact_duplicates": dropped_exact_duplicates,
        "dropped_high_corr": dropped_corr_pairs,
        "dropped_manual": dropped_manual,
        "dropped_drift_psi": dropped_drift_psi,
        "drift": drift_details,
    }
    return retained, report


def build_oof_target_encoding_features(
    df: pd.DataFrame,
    state: FittedPreprocessor,
    target_column: str = TARGET_CLAIM_COL,
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str | None = None,
) -> pd.DataFrame:
    target_encoded_columns = [col for col in state.target_encoding_maps.keys() if col in state.categorical_columns]
    if not target_encoded_columns or target_column not in df.columns:
        return pd.DataFrame(index=df.index)

    y = pd.to_numeric(df[target_column], errors="coerce").fillna(0.0)
    if len(df) < 2:
        payload = {
            f"{col}_te": pd.Series(
                np.full(len(df), state.target_encoding_global_mean, dtype=float),
                index=df.index,
            )
            for col in target_encoded_columns
        }
        return pd.DataFrame(payload, index=df.index)

    prepared_categories: dict[str, pd.Series] = {}
    for col in target_encoded_columns:
        raw_series = df[col] if col in df.columns else pd.Series("missing", index=df.index)
        prepared_categories[col] = _normalize_series_to_category(raw_series, state.rare_category_kept.get(col))

    effective_splits = max(2, int(n_splits))
    split_iterator: Any
    if group_column and group_column in df.columns:
        groups = df[group_column].astype("string").fillna("__missing_group__").astype(str)
        unique_groups = int(groups.nunique(dropna=False))
        if unique_groups >= 2:
            effective_splits = min(effective_splits, unique_groups)
            split_iterator = GroupKFold(n_splits=effective_splits).split(df, y, groups=groups)
        else:
            effective_splits = min(effective_splits, len(df))
            split_iterator = KFold(
                n_splits=effective_splits,
                shuffle=True,
                random_state=random_state,
            ).split(df)
    else:
        effective_splits = min(effective_splits, len(df))
        split_iterator = KFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=random_state,
        ).split(df)

    global_mean = state.target_encoding_global_mean
    oof_payload = {
        f"{col}_te": np.full(len(df), global_mean, dtype=float)
        for col in target_encoded_columns
    }
    index_array = np.arange(len(df))
    rng = np.random.default_rng(random_state)

    for train_idx, valid_idx in split_iterator:
        y_train = y.iloc[train_idx]
        fold_mean = float(y_train.mean()) if len(y_train) else global_mean

        for col in target_encoded_columns:
            mapping = _fit_target_encoding_map(
                category=prepared_categories[col].iloc[train_idx],
                target=y_train,
                global_mean=fold_mean,
                smoothing=state.config.target_encoding_smoothing,
                min_samples_leaf=state.config.target_encoding_min_samples_leaf,
            )
            encoded = prepared_categories[col].iloc[valid_idx].map(mapping).fillna(fold_mean).astype(float).values
            if state.config.target_encoding_noise_std > 0:
                encoded = encoded + rng.normal(
                    0.0,
                    state.config.target_encoding_noise_std,
                    size=len(encoded),
                )
            oof_payload[f"{col}_te"][index_array[valid_idx]] = encoded

    return pd.DataFrame(oof_payload, index=df.index)


def fit_preprocessor(df: pd.DataFrame, config: PreprocessingConfig) -> FittedPreprocessor:
    LOGGER.info("Preprocessor fit started: rows=%d cols=%d", len(df), len(df.columns))
    working = df.copy()
    protected_columns = _protect_columns(config)
    drop_candidates = [col for col in config.drop_columns if col in working.columns and col not in protected_columns]
    if drop_candidates:
        working = working.drop(columns=drop_candidates)
    working = _normalize_numeric_like_columns(working, stage="fit")

    date_input_columns = [col for col in config.date_columns if col in working.columns and col not in protected_columns]

    numeric_columns = [
        col
        for col in working.columns
        if col not in protected_columns and col not in date_input_columns and pd.api.types.is_numeric_dtype(working[col])
    ]
    categorical_columns = [
        col
        for col in working.columns
        if col not in protected_columns and col not in numeric_columns and col not in date_input_columns
    ]

    base_features = numeric_columns + categorical_columns + date_input_columns
    available_set = set(base_features)
    whitelist = _load_feature_list(config.feature_whitelist_path)
    droplist = _load_feature_list(config.feature_droplist_path)
    encoding_force_keep = set(config.target_encoding_columns + config.frequency_encoding_columns) & available_set
    force_keep = (set(config.force_keep_features) | set(date_input_columns) | encoding_force_keep) & available_set
    force_drop = set(config.force_drop_features)

    if whitelist:
        selected = (available_set & whitelist) | force_keep
    else:
        selected = set(available_set)
    selected -= droplist
    selected -= force_drop
    selected |= force_keep

    numeric_columns = [col for col in numeric_columns if col in selected]
    categorical_columns = [col for col in categorical_columns if col in selected]
    date_input_columns = [col for col in date_input_columns if col in selected]
    LOGGER.info(
        "Feature selection applied in preprocessing: base=%d selected=%d (num=%d cat=%d date=%d)",
        len(base_features),
        len(selected),
        len(numeric_columns),
        len(categorical_columns),
        len(date_input_columns),
    )

    if not numeric_columns and not categorical_columns:
        fallback = [PREMIUM_COL] if PREMIUM_COL in working.columns else []
        if fallback:
            numeric_columns = fallback
            LOGGER.warning("No features selected after filters. Fallback to %s", fallback)

    missing_flag_columns: list[str] = []
    numeric_fill_values: dict[str, float] = {}
    winsor_bounds: dict[str, tuple[float, float]] = {}

    for index, col in enumerate(numeric_columns, start=1):
        series = pd.to_numeric(working[col], errors="coerce")
        null_rate = float(series.isna().mean())
        if config.add_missing_flags and null_rate > config.missing_flag_threshold:
            missing_flag_columns.append(col)

        fill_value = _choose_fill_value(series, config, col)
        numeric_fill_values[col] = fill_value
        filled = series.fillna(fill_value)
        winsor_bounds[col] = _clip_quantiles(filled, config.winsorize_low, config.winsorize_high)
        if index % max(config.progress_log_every_n, 1) == 0 or index == len(numeric_columns):
            LOGGER.info("Fit numeric progress: %d/%d columns", index, len(numeric_columns))

    rare_category_kept: dict[str, list[str]] = {}
    for index, col in enumerate(categorical_columns, start=1):
        series = working[col].astype("string").fillna("missing")
        freq = series.value_counts(dropna=False, normalize=True)
        count = series.value_counts(dropna=False)
        kept = (
            set(freq[freq >= config.rare_category_threshold].index.astype(str).tolist())
            | set(count[count >= config.rare_category_min_count].index.astype(str).tolist())
        )
        kept = sorted(kept)
        if not kept:
            kept = series.value_counts(dropna=False).head(20).index.astype(str).tolist()
        rare_category_kept[col] = kept
        if index % max(config.progress_log_every_n, 1) == 0 or index == len(categorical_columns):
            LOGGER.info("Fit categorical progress: %d/%d columns", index, len(categorical_columns))

    date_feature_columns: list[str] = []
    for date_column in date_input_columns:
        _, names = _build_date_features(working[date_column], date_column, config.date_features)
        date_feature_columns.extend(names)

    target_series = pd.to_numeric(
        working.get(TARGET_CLAIM_COL, pd.Series(0, index=working.index)),
        errors="coerce",
    ).fillna(0.0)
    target_encoding_global_mean = float(target_series.mean()) if len(target_series) else 0.0
    target_encoding_maps: dict[str, dict[str, float]] = {}
    frequency_encoding_maps: dict[str, dict[str, float]] = {}

    if config.target_encoding_enabled:
        for column in config.target_encoding_columns:
            if column not in categorical_columns:
                continue
            prepared = _normalize_series_to_category(working[column], rare_category_kept.get(column))
            target_encoding_maps[column] = _fit_target_encoding_map(
                category=prepared,
                target=target_series,
                global_mean=target_encoding_global_mean,
                smoothing=config.target_encoding_smoothing,
                min_samples_leaf=config.target_encoding_min_samples_leaf,
            )

    if config.frequency_encoding_enabled:
        for column in config.frequency_encoding_columns:
            if column not in categorical_columns:
                continue
            prepared = _normalize_series_to_category(working[column], rare_category_kept.get(column))
            frequency_encoding_maps[column] = _fit_frequency_encoding_map(prepared)

    missing_aggregate_definitions: dict[str, list[str]] = {}
    if config.add_missing_aggregates and missing_flag_columns:
        score_flags = [f"{col}_is_missing" for col in missing_flag_columns if col.startswith("SCORE_")]
        if score_flags:
            missing_aggregate_definitions["score_missing_cnt_total"] = score_flags
            missing_aggregate_definitions["score_missing_share"] = score_flags

        for prefix in config.missing_aggregate_prefixes:
            if prefix == "SCORE_":
                continue
            prefixed_flags = [f"{col}_is_missing" for col in missing_flag_columns if col.startswith(prefix)]
            if prefixed_flags:
                missing_aggregate_definitions[f"{_prefix_token(prefix)}_missing_cnt"] = prefixed_flags

    interaction_feature_columns: list[str] = []
    interaction_mvp_report: dict[str, Any] = {
        "enabled": bool(config.interaction_features_mvp_enabled),
        "definitions": [],
        "generated": [],
        "dropped_by_whitelist": [],
        "dropped_missing_sources": [],
        "dropped_by_corr": [],
        "dropped_by_psi": [],
        "dropped_by_rank": [],
        "retained": [],
    }
    available_sources = _available_interaction_sources(
        working=working,
        categorical_columns=categorical_columns,
        target_encoding_maps=target_encoding_maps,
        frequency_encoding_maps=frequency_encoding_maps,
        missing_aggregate_definitions=missing_aggregate_definitions,
    )
    existing_interactions: set[str] = set()
    if config.interaction_features_enabled:
        for name in config.interaction_features:
            feature_name = str(name).strip()
            if not feature_name or feature_name in existing_interactions:
                continue
            required_sources = INTERACTION_REQUIRED_SOURCES.get(feature_name, set())
            if required_sources and not required_sources.issubset(available_sources):
                continue
            interaction_feature_columns.append(feature_name)
            existing_interactions.add(feature_name)

    mvp_candidates: list[str] = []
    if config.interaction_features_mvp_enabled:
        requested = [
            str(name).strip()
            for name in (
                config.interaction_features_mvp_definitions
                if config.interaction_features_mvp_definitions
                else list(DEFAULT_INTERACTION_MVP_DEFINITIONS)
            )
            if str(name).strip()
        ]
        interaction_mvp_report["definitions"] = list(requested)
        whitelist = set(INTERACTION_MVP_BUSINESS_WHITELIST)
        for feature_name in requested:
            if (
                config.interaction_features_mvp_require_business_whitelist
                and feature_name not in whitelist
            ):
                interaction_mvp_report["dropped_by_whitelist"].append(feature_name)
                continue
            required_sources = INTERACTION_REQUIRED_SOURCES.get(feature_name)
            if not required_sources:
                interaction_mvp_report["dropped_missing_sources"].append(
                    {"feature": feature_name, "missing_sources": ["unsupported_definition"]}
                )
                continue
            missing_sources = sorted([source for source in required_sources if source not in available_sources])
            if missing_sources:
                interaction_mvp_report["dropped_missing_sources"].append(
                    {"feature": feature_name, "missing_sources": missing_sources}
                )
                continue
            if feature_name in existing_interactions:
                continue
            mvp_candidates.append(feature_name)
            interaction_feature_columns.append(feature_name)
            existing_interactions.add(feature_name)

    generated_categorical_columns = list(categorical_columns)
    if "region_x_vehicle_type" in interaction_feature_columns:
        generated_categorical_columns.append("region_x_vehicle_type")

    feature_columns = list(numeric_columns + categorical_columns + date_feature_columns)
    for col in config.log1p_columns:
        if col in feature_columns:
            feature_columns.append(f"{col}_log1p")
    for col in missing_flag_columns:
        feature_columns.append(f"{col}_is_missing")
    for column in target_encoding_maps:
        feature_columns.append(f"{column}_te")
    for column in frequency_encoding_maps:
        feature_columns.append(f"{column}_freq")
    feature_columns.extend(missing_aggregate_definitions.keys())
    feature_columns.extend(interaction_feature_columns)
    feature_columns = sorted(set(feature_columns))

    fitted = FittedPreprocessor(
        config=config,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        date_input_columns=date_input_columns,
        date_feature_columns=sorted(set(date_feature_columns)),
        categorical_output_columns=sorted(set(generated_categorical_columns)),
        numeric_fill_values=numeric_fill_values,
        winsor_bounds=winsor_bounds,
        missing_flag_columns=missing_flag_columns,
        missing_aggregate_definitions=missing_aggregate_definitions,
        rare_category_kept=rare_category_kept,
        target_encoding_maps=target_encoding_maps,
        target_encoding_global_mean=target_encoding_global_mean,
        frequency_encoding_maps=frequency_encoding_maps,
        interaction_feature_columns=interaction_feature_columns,
        feature_generation_version=config.feature_generation_version,
        feature_columns=feature_columns,
        feature_pruning_report={
            "enabled": False,
            "applied": False,
            "before_count": len(feature_columns),
            "after_count": len(feature_columns),
            "dropped_total": 0,
            "interaction_mvp_report": interaction_mvp_report,
        },
    )

    needs_transformed_fit = bool(
        config.feature_pruning_enabled
        or config.drift_pruning_enabled
        or config.interaction_features_mvp_enabled
    )
    if needs_transformed_fit:
        try:
            transformed_fit = transform_with_preprocessor(df, fitted)
            if config.interaction_features_mvp_enabled:
                retained_mvp, filtered_mvp_report = _filter_mvp_interaction_features(
                    transformed_df=transformed_fit,
                    feature_columns=fitted.feature_columns,
                    mvp_candidates=mvp_candidates,
                    config=config,
                    source_df=df,
                )
                interaction_mvp_report = {
                    "enabled": bool(config.interaction_features_mvp_enabled),
                    "definitions": list(interaction_mvp_report.get("definitions", [])),
                    "generated": list(filtered_mvp_report.get("generated", [])),
                    "dropped_by_whitelist": list(interaction_mvp_report.get("dropped_by_whitelist", [])),
                    "dropped_missing_sources": list(interaction_mvp_report.get("dropped_missing_sources", [])),
                    "dropped_by_corr": list(filtered_mvp_report.get("dropped_by_corr", [])),
                    "dropped_by_psi": list(filtered_mvp_report.get("dropped_by_psi", [])),
                    "dropped_by_rank": list(filtered_mvp_report.get("dropped_by_rank", [])),
                    "retained": list(filtered_mvp_report.get("retained", [])),
                    "max_features": filtered_mvp_report.get(
                        "max_features",
                        int(max(config.interaction_features_mvp_max_features, 0)),
                    ),
                    "corr_filter_threshold": filtered_mvp_report.get(
                        "corr_filter_threshold",
                        float(max(config.interaction_features_mvp_corr_filter_threshold, 0.0)),
                    ),
                    "psi_filter_threshold": filtered_mvp_report.get(
                        "psi_filter_threshold",
                        float(max(config.interaction_features_mvp_psi_filter_threshold, 0.0)),
                    ),
                    "candidates_ranked": list(filtered_mvp_report.get("candidates_ranked", [])),
                }
                retained_mvp_set = set(retained_mvp)
                fitted.interaction_feature_columns = [
                    name
                    for name in fitted.interaction_feature_columns
                    if name not in set(mvp_candidates) or name in retained_mvp_set
                ]
                fitted.feature_columns = [
                    name
                    for name in fitted.feature_columns
                    if name not in set(mvp_candidates) or name in retained_mvp_set
                ]

            if config.feature_pruning_enabled or config.drift_pruning_enabled:
                pruned_feature_columns, pruning_report = _prune_feature_columns_by_statistics(
                    transformed_df=transformed_fit,
                    feature_columns=fitted.feature_columns,
                    config=config,
                    source_df=df,
                )
                pruning_report["interaction_mvp_report"] = interaction_mvp_report
                fitted.feature_columns = pruned_feature_columns
                fitted.feature_pruning_report = pruning_report
                if pruning_report.get("applied"):
                    LOGGER.info(
                        "Feature pruning applied: before=%d after=%d dropped=%d",
                        pruning_report.get("before_count", 0),
                        pruning_report.get("after_count", 0),
                        pruning_report.get("dropped_total", 0),
                    )
                else:
                    LOGGER.info(
                        "Feature pruning enabled: no drops (features=%d)",
                        len(fitted.feature_columns),
                    )
            else:
                fitted.feature_pruning_report = {
                    "enabled": False,
                    "applied": False,
                    "before_count": len(feature_columns),
                    "after_count": len(fitted.feature_columns),
                    "dropped_total": len(feature_columns) - len(fitted.feature_columns),
                    "interaction_mvp_report": interaction_mvp_report,
                }
        except Exception as exc:  # pragma: no cover - keep preprocessing robust
            LOGGER.warning("Feature pruning skipped due to error: %s", exc)
            fitted.feature_pruning_report = {
                "enabled": bool(config.feature_pruning_enabled or config.drift_pruning_enabled),
                "applied": False,
                "before_count": len(feature_columns),
                "after_count": len(feature_columns),
                "dropped_total": 0,
                "error": str(exc),
                "interaction_mvp_report": interaction_mvp_report,
            }

    LOGGER.info(
        "Preprocessor fit finished: numeric=%d categorical=%d date=%d features=%d",
        len(numeric_columns),
        len(categorical_columns),
        len(date_feature_columns),
        len(fitted.feature_columns),
    )
    return fitted


def transform_with_preprocessor(df: pd.DataFrame, state: FittedPreprocessor) -> pd.DataFrame:
    cfg = state.config
    working = df.copy()
    protected_columns = _protect_columns(cfg)

    drop_candidates = [col for col in cfg.drop_columns if col in working.columns and col not in protected_columns]
    if drop_candidates:
        working = working.drop(columns=drop_candidates)
    working = _normalize_numeric_like_columns(working, stage="transform")

    for date_column in state.date_input_columns:
        if date_column not in working.columns:
            working[date_column] = pd.NaT

    missing_flag_data: dict[str, pd.Series] = {}
    for index, col in enumerate(state.numeric_columns, start=1):
        if col not in working.columns:
            working[col] = np.nan
        series = pd.to_numeric(working[col], errors="coerce")
        missing_mask = series.isna()

        if col in state.missing_flag_columns:
            missing_flag_data[f"{col}_is_missing"] = missing_mask.astype(int)

        fill_value = state.numeric_fill_values.get(col, 0.0)
        filled = series.fillna(fill_value)
        if col not in {PREMIUM_COL, PREMIUM_NET_COL}:
            lower, upper = state.winsor_bounds.get(col, (None, None))
            if lower is not None and upper is not None:
                filled = filled.clip(lower=lower, upper=upper)
        working[col] = filled

        if col in cfg.log1p_columns:
            working[f"{col}_log1p"] = np.log1p(filled.clip(lower=0.0))
        if index % max(cfg.progress_log_every_n, 1) == 0 or index == len(state.numeric_columns):
            LOGGER.info("Transform numeric progress: %d/%d columns", index, len(state.numeric_columns))

    if missing_flag_data:
        missing_flags_df = pd.DataFrame(missing_flag_data, index=working.index)
        working = pd.concat([working, missing_flags_df], axis=1)
    else:
        missing_flags_df = pd.DataFrame(index=working.index)

    for index, col in enumerate(state.categorical_columns, start=1):
        if col not in working.columns:
            working[col] = "missing"
        working[col] = _normalize_series_to_category(working[col], state.rare_category_kept.get(col))
        if index % max(cfg.progress_log_every_n, 1) == 0 or index == len(state.categorical_columns):
            LOGGER.info("Transform categorical progress: %d/%d columns", index, len(state.categorical_columns))

    for date_column in state.date_input_columns:
        generated, _ = _build_date_features(working[date_column], date_column, cfg.date_features)
        for new_col, values in generated.items():
            working[new_col] = pd.to_numeric(values, errors="coerce").fillna(0.0)

    if state.target_encoding_maps:
        for column, mapping in state.target_encoding_maps.items():
            if column not in working.columns:
                working[column] = "missing"
            encoded = working[column].astype("string").fillna("missing").astype(str).map(mapping)
            encoded = encoded.fillna(state.target_encoding_global_mean).astype(float)
            working[f"{column}_te"] = encoded

    if state.frequency_encoding_maps:
        for column, mapping in state.frequency_encoding_maps.items():
            if column not in working.columns:
                working[column] = "missing"
            encoded = working[column].astype("string").fillna("missing").astype(str).map(mapping)
            working[f"{column}_freq"] = encoded.fillna(0.0).astype(float)

    if state.missing_aggregate_definitions:
        for output_column, source_flags in state.missing_aggregate_definitions.items():
            if not source_flags:
                continue
            values = missing_flags_df.reindex(columns=source_flags, fill_value=0).sum(axis=1).astype(float)
            if output_column.endswith("_share"):
                values = values / max(len(source_flags), 1)
            working[output_column] = values

    if state.interaction_feature_columns:
        interaction_sources = _build_interaction_sources(working)
        for feature_name in state.interaction_feature_columns:
            generated = _compute_interaction_feature_series(
                feature_name=feature_name,
                df=working,
                sources=interaction_sources,
            )
            if generated is not None:
                working[feature_name] = generated

    for col in state.feature_columns:
        if col not in working.columns:
            if col.endswith("_is_missing"):
                working[col] = 0
            elif col in state.categorical_output_columns:
                working[col] = "missing"
            else:
                working[col] = 0.0

    required_keep = [cfg.grain, *cfg.target_columns]
    keep_columns = [col for col in required_keep if col in working.columns]
    final_columns = keep_columns + [col for col in state.feature_columns if col in working.columns]

    transformed = working[final_columns].copy()
    transformed = transformed.replace([np.inf, -np.inf], np.nan)
    LOGGER.info("Transform finished: rows=%d cols=%d", len(transformed), len(transformed.columns))
    return transformed
