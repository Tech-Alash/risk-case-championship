from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import GroupKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from risk_case.features.builder import FeatureSchema, infer_feature_schema, prepare_features
from risk_case.models.frequency_severity import FrequencySeverityModel
from risk_case.models.metrics import classification_metrics, severity_metrics
from risk_case.models.tweedie_model import TweedieAggregateLossModel
from risk_case.models.woe_baseline import WoEFrequencySeverityModel
from risk_case.pricing.evaluator import (
    PricingEvaluation,
    RetentionConfig,
    StratifiedPricingConfig,
    select_best_pricing,
)
from risk_case.settings import TARGET_AMOUNT_COL, TARGET_CLAIM_COL, ensure_dir

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBM(Classifier|Regressor) was fitted with feature names",
    category=UserWarning,
)


def _safe_float(value: Any, default: float = float("-inf")) -> float:
    try:
        if value is None:
            return default
        value_f = float(value)
        if np.isnan(value_f):
            return default
        return value_f
    except (TypeError, ValueError):
        return default


def _make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _build_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if schema.numeric_cols:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("num", numeric_transformer, schema.numeric_cols))

    if schema.categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_onehot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_transformer, schema.categorical_cols))

    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required")

    return ColumnTransformer(transformers=transformers)


@dataclass
class BenchmarkConstraints:
    max_violations: int = 0
    lr_total_min: float = 0.69
    lr_total_max: float = 0.71
    share_group1_min: float = 0.0
    share_group1_max: float = 1.0

    @staticmethod
    def from_dict(raw: dict[str, Any] | None) -> "BenchmarkConstraints":
        data = raw or {}
        return BenchmarkConstraints(
            max_violations=int(data.get("max_violations", 0)),
            lr_total_min=float(data.get("lr_total_min", 0.69)),
            lr_total_max=float(data.get("lr_total_max", 0.71)),
            share_group1_min=float(data.get("share_group1_min", 0.0)),
            share_group1_max=float(data.get("share_group1_max", 1.0)),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "max_violations": self.max_violations,
            "lr_total_min": self.lr_total_min,
            "lr_total_max": self.lr_total_max,
            "share_group1_min": self.share_group1_min,
            "share_group1_max": self.share_group1_max,
        }


@dataclass
class BenchmarkCalibrationConfig:
    enabled: bool = False
    method: str = "none"
    oof_folds: int = 5
    group_column: str = "contract_number"
    min_samples: int = 500
    clip_eps: float = 1e-6

    @staticmethod
    def from_dict(raw: dict[str, Any] | None) -> "BenchmarkCalibrationConfig":
        data = raw or {}
        method = str(data.get("method", "none")).strip().lower()
        if method == "sigmoid":
            method = "platt"
        if method not in {"none", "platt", "isotonic"}:
            method = "none"
        return BenchmarkCalibrationConfig(
            enabled=bool(data.get("enabled", False)),
            method=method,
            oof_folds=max(2, int(data.get("oof_folds", 5))),
            group_column=str(data.get("group_column", "contract_number")),
            min_samples=max(50, int(data.get("min_samples", 500))),
            clip_eps=float(np.clip(float(data.get("clip_eps", 1e-6)), 1e-9, 0.1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "oof_folds": self.oof_folds,
            "group_column": self.group_column,
            "min_samples": self.min_samples,
            "clip_eps": self.clip_eps,
        }


@dataclass
class BenchmarkConfig:
    enabled: bool = False
    candidates: list[str] = field(default_factory=lambda: ["baseline_freq_sev"])
    selection_metric: str = "policy_score"
    stability_penalty: float = 0.0
    must_pass_constraints: bool = True
    constraints: BenchmarkConstraints = field(default_factory=BenchmarkConstraints)
    fallback_strategy: str = "best_metric"
    fallback_candidate: str = "baseline_freq_sev"
    random_state: int = 42
    candidate_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    calibration: BenchmarkCalibrationConfig = field(default_factory=BenchmarkCalibrationConfig)

    @staticmethod
    def from_dict(raw: dict[str, Any] | None) -> "BenchmarkConfig":
        data = raw or {}
        candidates = data.get("candidates", ["baseline_freq_sev"])
        if not candidates:
            candidates = ["baseline_freq_sev"]
        fallback_strategy = str(data.get("fallback_strategy", "best_metric"))
        if fallback_strategy not in {"best_metric", "configured_candidate"}:
            fallback_strategy = "best_metric"
        return BenchmarkConfig(
            enabled=bool(data.get("enabled", False)),
            candidates=[str(x) for x in candidates],
            selection_metric=str(data.get("selection_metric", "policy_score")),
            stability_penalty=float(data.get("stability_penalty", 0.0)),
            must_pass_constraints=bool(data.get("must_pass_constraints", True)),
            constraints=BenchmarkConstraints.from_dict(data.get("constraints")),
            fallback_strategy=fallback_strategy,
            fallback_candidate=str(data.get("fallback_candidate", "baseline_freq_sev")),
            random_state=int(data.get("random_state", 42)),
            candidate_params={
                str(name): dict(params or {})
                for name, params in (data.get("candidate_params") or {}).items()
            },
            calibration=BenchmarkCalibrationConfig.from_dict(data.get("calibration")),
        )


@dataclass
class CandidateResult:
    candidate_name: str
    status: str
    ml: dict[str, Any] | None = None
    severity: dict[str, Any] | None = None
    pricing: dict[str, Any] | None = None
    passes_constraints: bool = False
    constraint_reasons: list[str] = field(default_factory=list)
    alpha: float | None = None
    beta: float | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "status": self.status,
            "ml": self.ml,
            "severity": self.severity,
            "pricing": self.pricing,
            "passes_constraints": self.passes_constraints,
            "constraint_reasons": list(self.constraint_reasons),
            "alpha": self.alpha,
            "beta": self.beta,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "metadata": self.metadata,
        }

    def to_record(self) -> dict[str, Any]:
        ml = self.ml or {}
        severity = self.severity or {}
        pricing = self.pricing or {}
        metadata = self.metadata or {}
        return {
            "candidate_name": self.candidate_name,
            "status": self.status,
            "passes_constraints": self.passes_constraints,
            "constraint_reasons": "; ".join(self.constraint_reasons),
            "alpha": self.alpha,
            "beta": self.beta,
            "pricing_policy_kind": pricing.get("pricing_policy_kind"),
            "pricing_bucket_count": (pricing.get("pricing_policy") or {}).get("bucket_count")
            if isinstance(pricing.get("pricing_policy"), dict)
            else None,
            "policy_score": pricing.get("policy_score"),
            "violations": pricing.get("violations"),
            "lr_total": pricing.get("lr_total"),
            "lr_group1": pricing.get("lr_group1"),
            "lr_group2": pricing.get("lr_group2"),
            "share_group1": pricing.get("share_group1"),
            "auc": ml.get("auc"),
            "gini": ml.get("gini"),
            "brier": ml.get("brier"),
            "severity_rmse": severity.get("rmse"),
            "severity_mae": severity.get("mae"),
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "blend_base_candidates": ",".join(metadata.get("base_candidates", [])) if metadata.get("base_candidates") else None,
            "blend_weights": (
                json.dumps(metadata.get("weights"), ensure_ascii=False)
                if isinstance(metadata.get("weights"), dict)
                else None
            ),
            "blend_oof_policy_score": metadata.get("oof_policy_score"),
            "blend_oof_constraints_pass": metadata.get("oof_constraints_pass"),
            "calibration_status": metadata.get("calibration_status"),
            "calibration_method": metadata.get("calibration_method"),
            "calibration_reason": metadata.get("calibration_reason"),
            "calibration_oof_rows": metadata.get("calibration_oof_rows"),
            "calibration_oof_brier_raw": metadata.get("calibration_oof_brier_raw"),
            "calibration_oof_brier_calibrated": metadata.get("calibration_oof_brier_calibrated"),
            "calibration_oof_auc_raw": metadata.get("calibration_oof_auc_raw"),
            "calibration_oof_auc_calibrated": metadata.get("calibration_oof_auc_calibrated"),
        }


@dataclass
class BenchmarkResult:
    winner_name: str
    selection_reason: str
    winner_model: Any
    winner_valid_pred: pd.DataFrame
    winner_alpha: float
    winner_beta: float
    winner_premium: pd.Series
    winner_pricing_eval: PricingEvaluation
    winner_ml: dict[str, Any]
    winner_severity: dict[str, Any]
    results: list[CandidateResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner_name": self.winner_name,
            "selection_reason": self.selection_reason,
            "winner_alpha": self.winner_alpha,
            "winner_beta": self.winner_beta,
            "winner_pricing": self.winner_pricing_eval.to_dict(),
            "winner_ml": self.winner_ml,
            "winner_severity": self.winner_severity,
            "results": [result.to_dict() for result in self.results],
        }


class PipelineFrequencySeverityModel:
    def __init__(self, classifier: Any, regressor: Any) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.schema: FeatureSchema | None = None
        self.frequency_model: Pipeline | None = None
        self.severity_model: Pipeline | None = None
        self.severity_constant: float = 0.0
        self.severity_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "PipelineFrequencySeverityModel":
        self.schema = infer_feature_schema(df)
        X = prepare_features(df, self.schema)
        y_freq = df[TARGET_CLAIM_COL].fillna(0).astype(int).values

        self.frequency_model = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(self.schema)),
                ("classifier", self.classifier),
            ]
        )
        self.frequency_model.fit(X, y_freq)

        positive_mask = df[TARGET_CLAIM_COL].fillna(0).astype(int) > 0
        y_sev = pd.to_numeric(df[TARGET_AMOUNT_COL], errors="coerce").fillna(0.0).clip(lower=0.0)

        if positive_mask.sum() >= 20:
            self.severity_model = Pipeline(
                steps=[
                    ("preprocessor", _build_preprocessor(self.schema)),
                    ("regressor", self.regressor),
                ]
            )
            self.severity_model.fit(X.loc[positive_mask], np.log1p(y_sev.loc[positive_mask]))
            self.severity_fitted = True
        else:
            positive_amounts = y_sev.loc[positive_mask]
            self.severity_constant = float(positive_amounts.mean()) if len(positive_amounts) else 0.0
            self.severity_fitted = False

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.schema is None or self.frequency_model is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = prepare_features(df, self.schema)
        probabilities = self.frequency_model.predict_proba(X)
        p_claim = probabilities if np.ndim(probabilities) == 1 else probabilities[:, -1]

        if self.severity_fitted and self.severity_model is not None:
            sev_log = self.severity_model.predict(X)
            expected_severity = np.expm1(np.clip(sev_log, 0, 20))
        else:
            expected_severity = np.full(len(df), self.severity_constant)

        expected_loss = p_claim * expected_severity
        return pd.DataFrame(
            {
                "p_claim": p_claim,
                "expected_severity": expected_severity,
                "expected_loss": expected_loss,
            },
            index=df.index,
        )

    def save(self, path: Path | str) -> None:
        dump(self, path)

    @staticmethod
    def load(path: Path | str) -> "PipelineFrequencySeverityModel":
        return load(path)


class CatBoostFrequencySeverityModel:
    def __init__(
        self,
        random_state: int = 42,
        iterations: int = 250,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 0.0,
        border_count: int = 254,
        reg_iterations: int | None = None,
        reg_learning_rate: float | None = None,
        reg_depth: int | None = None,
        reg_l2_leaf_reg: float | None = None,
        reg_random_strength: float | None = None,
        reg_bagging_temperature: float | None = None,
        reg_border_count: int | None = None,
        thread_count: int = -1,
        severity_loss_function: str = "RMSE",
        tweedie_variance_power: float = 1.5,
        task_type: str = "CPU",
        devices: str | None = None,
    ) -> None:
        self.random_state = random_state
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.border_count = border_count
        self.reg_iterations = reg_iterations
        self.reg_learning_rate = reg_learning_rate
        self.reg_depth = reg_depth
        self.reg_l2_leaf_reg = reg_l2_leaf_reg
        self.reg_random_strength = reg_random_strength
        self.reg_bagging_temperature = reg_bagging_temperature
        self.reg_border_count = reg_border_count
        self.thread_count = thread_count
        self.severity_loss_function = severity_loss_function
        self.tweedie_variance_power = tweedie_variance_power
        self.task_type = str(task_type or "CPU").strip().upper()
        self.devices = str(devices).strip() if devices is not None else None
        self.schema: FeatureSchema | None = None
        self.severity_schema: FeatureSchema | None = None
        self.cat_feature_indices: list[int] = []
        self.severity_cat_feature_indices: list[int] = []
        self.frequency_model: Any = None
        self.severity_model: Any = None
        self.severity_constant: float = 0.0
        self.severity_fitted: bool = False

    def _prepare_catboost_input(self, df: pd.DataFrame, schema: FeatureSchema | None = None) -> pd.DataFrame:
        active_schema = schema or self.schema
        if active_schema is None:
            raise RuntimeError("Model must be fitted before prediction")
        X = prepare_features(df, active_schema).copy()
        for column in active_schema.numeric_cols:
            X[column] = pd.to_numeric(X[column], errors="coerce")
        for column in active_schema.categorical_cols:
            X[column] = X[column].astype("string").fillna("missing").astype(str)
        return X

    def _build_frequency_classifier(self) -> Any:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            border_count=self.border_count,
            loss_function="Logloss",
            random_seed=self.random_state,
            thread_count=self.thread_count,
            task_type=self.task_type,
            devices=self.devices,
            verbose=False,
        )

    def _build_severity_regressor(self) -> Any:
        from catboost import CatBoostRegressor

        reg_iterations = self.reg_iterations if self.reg_iterations is not None else self.iterations
        reg_learning_rate = self.reg_learning_rate if self.reg_learning_rate is not None else self.learning_rate
        reg_depth = self.reg_depth if self.reg_depth is not None else self.depth
        reg_l2_leaf_reg = self.reg_l2_leaf_reg if self.reg_l2_leaf_reg is not None else self.l2_leaf_reg
        reg_random_strength = self.reg_random_strength if self.reg_random_strength is not None else self.random_strength
        reg_bagging_temperature = (
            self.reg_bagging_temperature if self.reg_bagging_temperature is not None else self.bagging_temperature
        )
        reg_border_count = self.reg_border_count if self.reg_border_count is not None else self.border_count
        severity_loss_name = str(self.severity_loss_function or "RMSE").strip().upper()
        if severity_loss_name == "TWEEDIE":
            variance_power = float(np.clip(self.tweedie_variance_power, 1.01, 1.99))
            loss_function = f"Tweedie:variance_power={variance_power}"
        else:
            loss_function = "RMSE"
        return CatBoostRegressor(
            iterations=reg_iterations,
            learning_rate=reg_learning_rate,
            depth=reg_depth,
            l2_leaf_reg=reg_l2_leaf_reg,
            random_strength=reg_random_strength,
            bagging_temperature=reg_bagging_temperature,
            border_count=reg_border_count,
            loss_function=loss_function,
            random_seed=self.random_state,
            thread_count=self.thread_count,
            task_type=self.task_type,
            devices=self.devices,
            verbose=False,
        )

    def _build_severity_schema(self) -> None:
        if self.schema is None:
            raise RuntimeError("Model must be fitted before building severity schema")
        self.severity_schema = FeatureSchema(
            numeric_cols=list(self.schema.numeric_cols),
            categorical_cols=list(self.schema.categorical_cols),
        )
        self.severity_cat_feature_indices = [
            self.severity_schema.all_cols.index(col)
            for col in self.severity_schema.categorical_cols
            if col in self.severity_schema.all_cols
        ]

    def _prepare_severity_input(
        self,
        df: pd.DataFrame,
        frequency_signal: np.ndarray | pd.Series | None = None,
    ) -> pd.DataFrame:
        if self.severity_schema is None:
            raise RuntimeError("Severity schema is not initialized")
        return self._prepare_catboost_input(df, schema=self.severity_schema)

    def fit(self, df: pd.DataFrame) -> "CatBoostFrequencySeverityModel":
        try:
            from catboost import Pool
        except Exception as exc:  # pragma: no cover
            raise ImportError("catboost package is not available") from exc

        self.schema = infer_feature_schema(df)
        X = self._prepare_catboost_input(df)
        self.cat_feature_indices = [X.columns.get_loc(col) for col in self.schema.categorical_cols if col in X.columns]
        y_freq = df[TARGET_CLAIM_COL].fillna(0).astype(int).values

        train_pool = Pool(X, y_freq, cat_features=self.cat_feature_indices)
        self.frequency_model = self._build_frequency_classifier()
        self.frequency_model.fit(train_pool)

        positive_mask = df[TARGET_CLAIM_COL].fillna(0).astype(int) > 0
        y_sev = pd.to_numeric(df[TARGET_AMOUNT_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
        self._build_severity_schema()

        if positive_mask.sum() >= 20:
            X_pos = self._prepare_severity_input(df.loc[positive_mask])
            y_pos = np.log1p(y_sev.loc[positive_mask].values)
            sev_pool = Pool(X_pos, y_pos, cat_features=self.severity_cat_feature_indices)
            self.severity_model = self._build_severity_regressor()
            self.severity_model.fit(sev_pool)
            self.severity_fitted = True
        else:
            positive_amounts = y_sev.loc[positive_mask]
            self.severity_constant = float(positive_amounts.mean()) if len(positive_amounts) else 0.0
            self.severity_fitted = False

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.schema is None or self.frequency_model is None:
            raise RuntimeError("Model must be fitted before prediction")
        from catboost import Pool

        X = self._prepare_catboost_input(df)
        pool = Pool(X, cat_features=self.cat_feature_indices)
        p_claim = self.frequency_model.predict_proba(pool)[:, -1]

        if self.severity_fitted and self.severity_model is not None:
            X_sev = self._prepare_severity_input(df, frequency_signal=p_claim)
            sev_pool = Pool(X_sev, cat_features=self.severity_cat_feature_indices)
            sev_log = self.severity_model.predict(sev_pool)
            expected_severity = np.expm1(np.clip(sev_log, 0, 20))
        else:
            expected_severity = np.full(len(df), self.severity_constant)

        expected_loss = p_claim * expected_severity
        return pd.DataFrame(
            {
                "p_claim": p_claim,
                "expected_severity": expected_severity,
                "expected_loss": expected_loss,
            },
            index=df.index,
        )

    def save(self, path: Path | str) -> None:
        dump(self, path)

    @staticmethod
    def load(path: Path | str) -> "CatBoostFrequencySeverityModel":
        return load(path)


class DependentCatBoostFrequencySeverityModel(CatBoostFrequencySeverityModel):
    def __init__(
        self,
        random_state: int = 42,
        iterations: int = 250,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 0.0,
        border_count: int = 254,
        reg_iterations: int | None = None,
        reg_learning_rate: float | None = None,
        reg_depth: int | None = None,
        reg_l2_leaf_reg: float | None = None,
        reg_random_strength: float | None = None,
        reg_bagging_temperature: float | None = None,
        reg_border_count: int | None = None,
        thread_count: int = -1,
        severity_loss_function: str = "RMSE",
        tweedie_variance_power: float = 1.5,
        task_type: str = "CPU",
        devices: str | None = None,
        dep_oof_folds: int = 5,
        dep_frequency_signal_name: str = "freq_risk_signal",
        dep_use_frequency_signal: bool = True,
    ) -> None:
        super().__init__(
            random_state=random_state,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            border_count=border_count,
            reg_iterations=reg_iterations,
            reg_learning_rate=reg_learning_rate,
            reg_depth=reg_depth,
            reg_l2_leaf_reg=reg_l2_leaf_reg,
            reg_random_strength=reg_random_strength,
            reg_bagging_temperature=reg_bagging_temperature,
            reg_border_count=reg_border_count,
            thread_count=thread_count,
            severity_loss_function=severity_loss_function,
            tweedie_variance_power=tweedie_variance_power,
            task_type=task_type,
            devices=devices,
        )
        self.dep_oof_folds = max(2, int(dep_oof_folds))
        self.dep_frequency_signal_name = str(dep_frequency_signal_name)
        self.dep_use_frequency_signal = bool(dep_use_frequency_signal)
        self.train_oof_p_claim_: np.ndarray | None = None

    def _build_severity_schema(self) -> None:
        if self.schema is None:
            raise RuntimeError("Model must be fitted before building severity schema")
        numeric_cols = list(self.schema.numeric_cols)
        if self.dep_use_frequency_signal and self.dep_frequency_signal_name not in numeric_cols:
            numeric_cols.append(self.dep_frequency_signal_name)
        self.severity_schema = FeatureSchema(
            numeric_cols=numeric_cols,
            categorical_cols=list(self.schema.categorical_cols),
        )
        self.severity_cat_feature_indices = [
            self.severity_schema.all_cols.index(col)
            for col in self.severity_schema.categorical_cols
            if col in self.severity_schema.all_cols
        ]

    def _prepare_severity_input(
        self,
        df: pd.DataFrame,
        frequency_signal: np.ndarray | pd.Series | None = None,
    ) -> pd.DataFrame:
        if self.severity_schema is None:
            raise RuntimeError("Severity schema is not initialized")
        base_df = df.copy()
        if self.dep_use_frequency_signal:
            signal_series = pd.Series(frequency_signal, index=df.index, dtype=float) if frequency_signal is not None else pd.Series(
                0.0, index=df.index, dtype=float
            )
            base_df[self.dep_frequency_signal_name] = pd.to_numeric(signal_series, errors="coerce").fillna(0.0)
        return self._prepare_catboost_input(base_df, schema=self.severity_schema)

    def fit(self, df: pd.DataFrame) -> "DependentCatBoostFrequencySeverityModel":
        if not self.dep_use_frequency_signal:
            return super().fit(df)

        try:
            from catboost import Pool
        except Exception as exc:  # pragma: no cover
            raise ImportError("catboost package is not available") from exc

        self.schema = infer_feature_schema(df)
        X = self._prepare_catboost_input(df)
        self.cat_feature_indices = [X.columns.get_loc(col) for col in self.schema.categorical_cols if col in X.columns]
        y_freq = df[TARGET_CLAIM_COL].fillna(0).astype(int).values

        train_pool = Pool(X, y_freq, cat_features=self.cat_feature_indices)
        self.frequency_model = self._build_frequency_classifier()
        self.frequency_model.fit(train_pool)
        full_p_claim = self.frequency_model.predict_proba(train_pool)[:, -1]

        oof_signal = np.asarray(full_p_claim, dtype=float).copy()
        try:
            splits = _iter_cv_splits(
                df=df,
                n_splits=self.dep_oof_folds,
                random_state=self.random_state,
                group_column="contract_number",
            )
            oof_signal = np.full(len(df), np.nan, dtype=float)
            for fold_train_idx, fold_valid_idx in splits:
                y_fold = y_freq[fold_train_idx]
                if len(np.unique(y_fold)) < 2:
                    oof_signal[fold_valid_idx] = full_p_claim[fold_valid_idx]
                    continue
                fold_model = self._build_frequency_classifier()
                fold_pool_train = Pool(X.iloc[fold_train_idx], y_fold, cat_features=self.cat_feature_indices)
                fold_model.fit(fold_pool_train)
                fold_pool_valid = Pool(X.iloc[fold_valid_idx], cat_features=self.cat_feature_indices)
                oof_signal[fold_valid_idx] = fold_model.predict_proba(fold_pool_valid)[:, -1]
            if np.isnan(oof_signal).any():
                nan_mask = np.isnan(oof_signal)
                oof_signal[nan_mask] = full_p_claim[nan_mask]
        except Exception:
            oof_signal = np.asarray(full_p_claim, dtype=float).copy()

        self.train_oof_p_claim_ = oof_signal
        self._build_severity_schema()

        positive_mask = df[TARGET_CLAIM_COL].fillna(0).astype(int) > 0
        y_sev = pd.to_numeric(df[TARGET_AMOUNT_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
        if positive_mask.sum() >= 20:
            signal_series = pd.Series(oof_signal, index=df.index, dtype=float)
            X_pos = self._prepare_severity_input(
                df.loc[positive_mask],
                frequency_signal=signal_series.loc[positive_mask].values,
            )
            y_pos = np.log1p(y_sev.loc[positive_mask].values)
            sev_pool = Pool(X_pos, y_pos, cat_features=self.severity_cat_feature_indices)
            self.severity_model = self._build_severity_regressor()
            self.severity_model.fit(sev_pool)
            self.severity_fitted = True
        else:
            positive_amounts = y_sev.loc[positive_mask]
            self.severity_constant = float(positive_amounts.mean()) if len(positive_amounts) else 0.0
            self.severity_fitted = False

        return self


@dataclass
class ProbabilityCalibrator:
    method: str
    model: Any | None = None
    clip_eps: float = 1e-6

    def transform(self, probabilities: np.ndarray | pd.Series) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), self.clip_eps, 1.0 - self.clip_eps)
        if self.method == "none" or self.model is None:
            return clipped
        if self.method == "platt":
            calibrated = self.model.predict_proba(clipped.reshape(-1, 1))[:, -1]
            return np.clip(calibrated, self.clip_eps, 1.0 - self.clip_eps)
        if self.method == "isotonic":
            calibrated = self.model.predict(clipped)
            return np.clip(np.asarray(calibrated, dtype=float), self.clip_eps, 1.0 - self.clip_eps)
        raise ValueError(f"Unsupported calibrator method: {self.method}")

    @staticmethod
    def fit(y_true: np.ndarray, raw_probabilities: np.ndarray, method: str, clip_eps: float) -> "ProbabilityCalibrator":
        method = str(method).strip().lower()
        if method == "sigmoid":
            method = "platt"
        clip_eps = float(np.clip(clip_eps, 1e-9, 0.1))
        clipped = np.clip(np.asarray(raw_probabilities, dtype=float), clip_eps, 1.0 - clip_eps)
        y_true = np.asarray(y_true, dtype=int)

        if method == "none":
            return ProbabilityCalibrator(method="none", model=None, clip_eps=clip_eps)
        if method == "platt":
            model = LogisticRegression(max_iter=2000, solver="lbfgs")
            model.fit(clipped.reshape(-1, 1), y_true)
            return ProbabilityCalibrator(method="platt", model=model, clip_eps=clip_eps)
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(clipped, y_true)
            return ProbabilityCalibrator(method="isotonic", model=model, clip_eps=clip_eps)
        raise ValueError(f"Unsupported calibration method: {method}")


class CalibratedFrequencySeverityModel:
    def __init__(self, base_model: Any, calibrator: ProbabilityCalibrator) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def fit(self, df: pd.DataFrame) -> "CalibratedFrequencySeverityModel":
        self.base_model.fit(df)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        prediction = self.base_model.predict(df).copy()
        required_columns = {"p_claim", "expected_severity", "expected_loss"}
        missing = required_columns.difference(prediction.columns)
        if missing:
            raise ValueError(f"Missing prediction columns for calibrated model: {sorted(missing)}")
        calibrated_p_claim = self.calibrator.transform(prediction["p_claim"].values)
        prediction["p_claim"] = calibrated_p_claim
        prediction["expected_loss"] = prediction["expected_severity"].values * calibrated_p_claim
        return prediction

    def save(self, path: Path | str) -> None:
        dump(self, path)

    @staticmethod
    def load(path: Path | str) -> "CalibratedFrequencySeverityModel":
        return load(path)


class OOFWeightedBlendModel:
    def __init__(self, base_models: dict[str, Any], weights: dict[str, float]) -> None:
        self.base_models = dict(base_models)
        self.weights = dict(weights)

    def fit(self, df: pd.DataFrame) -> "OOFWeightedBlendModel":
        # Blend model is initialized from already fitted base models.
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.base_models or not self.weights:
            raise RuntimeError("Blend model has no base models or weights")
        expected_columns = {"p_claim", "expected_severity", "expected_loss"}
        blended: pd.DataFrame | None = None

        for candidate_name, model in self.base_models.items():
            weight = float(self.weights.get(candidate_name, 0.0))
            if weight <= 0:
                continue
            pred = model.predict(df)
            missing = expected_columns.difference(pred.columns)
            if missing:
                raise ValueError(f"Missing prediction columns for blend base {candidate_name}: {sorted(missing)}")
            weighted = pred.loc[:, ["p_claim", "expected_severity", "expected_loss"]] * weight
            blended = weighted if blended is None else blended.add(weighted, fill_value=0.0)

        if blended is None:
            raise RuntimeError("Blend model produced no predictions (all weights are zero)")
        return blended

    def save(self, path: Path | str) -> None:
        dump(self, path)

    @staticmethod
    def load(path: Path | str) -> "OOFWeightedBlendModel":
        return load(path)


def _blend_prediction_frames(
    predictions_by_candidate: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    required_columns = {"p_claim", "expected_severity", "expected_loss"}
    blended: pd.DataFrame | None = None
    for candidate_name, prediction in predictions_by_candidate.items():
        missing = required_columns.difference(prediction.columns)
        if missing:
            raise ValueError(f"Missing prediction columns for blend base {candidate_name}: {sorted(missing)}")
        weight = float(weights.get(candidate_name, 0.0))
        if weight <= 0:
            continue
        weighted = prediction.loc[:, ["p_claim", "expected_severity", "expected_loss"]] * weight
        blended = weighted if blended is None else blended.add(weighted, fill_value=0.0)

    if blended is None:
        raise ValueError("Blend weights produced empty prediction payload")
    return blended


def _build_candidate_model(
    candidate_name: str,
    model_max_iter: int,
    model_ridge_alpha: float,
    random_state: int,
    candidate_params: dict[str, dict[str, Any]] | None = None,
) -> Any:
    params_map = candidate_params or {}
    params = dict(params_map.get(candidate_name) or {})

    if candidate_name == "baseline_freq_sev":
        return FrequencySeverityModel(
            max_iter=int(params.get("max_iter", model_max_iter)),
            ridge_alpha=float(params.get("ridge_alpha", model_ridge_alpha)),
        )

    if candidate_name == "xgboost_freq_sev":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except Exception as exc:  # pragma: no cover
            raise ImportError("xgboost package is not available") from exc
        common = {
            "n_estimators": int(params.get("n_estimators", 220)),
            "max_depth": int(params.get("max_depth", 6)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "random_state": int(params.get("random_state", random_state)),
            "n_jobs": int(params.get("n_jobs", -1)),
        }
        xgb_classifier_extra: dict[str, Any] = {}
        xgb_regressor_extra: dict[str, Any] = {}
        for source_key, target_key in [
            ("tree_method", "tree_method"),
            ("device", "device"),
            ("predictor", "predictor"),
            ("max_bin", "max_bin"),
        ]:
            if source_key in params:
                xgb_classifier_extra[target_key] = params[source_key]
            reg_key = f"reg_{source_key}"
            if reg_key in params:
                xgb_regressor_extra[target_key] = params[reg_key]
        return PipelineFrequencySeverityModel(
            classifier=XGBClassifier(
                n_estimators=common["n_estimators"],
                max_depth=common["max_depth"],
                learning_rate=common["learning_rate"],
                subsample=common["subsample"],
                colsample_bytree=common["colsample_bytree"],
                min_child_weight=float(params.get("min_child_weight", 1.0)),
                reg_lambda=float(params.get("reg_lambda", 1.0)),
                reg_alpha=float(params.get("reg_alpha", 0.0)),
                gamma=float(params.get("gamma", 0.0)),
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=common["random_state"],
                n_jobs=common["n_jobs"],
                **xgb_classifier_extra,
            ),
            regressor=XGBRegressor(
                n_estimators=int(params.get("reg_n_estimators", common["n_estimators"] + 40)),
                max_depth=int(params.get("reg_max_depth", common["max_depth"])),
                learning_rate=float(params.get("reg_learning_rate", common["learning_rate"])),
                subsample=float(params.get("reg_subsample", common["subsample"])),
                colsample_bytree=float(params.get("reg_colsample_bytree", common["colsample_bytree"])),
                min_child_weight=float(params.get("reg_min_child_weight", params.get("min_child_weight", 1.0))),
                reg_lambda=float(params.get("reg_reg_lambda", params.get("reg_lambda", 1.0))),
                reg_alpha=float(params.get("reg_reg_alpha", params.get("reg_alpha", 0.0))),
                gamma=float(params.get("reg_gamma", params.get("gamma", 0.0))),
                objective="reg:squarederror",
                random_state=common["random_state"],
                n_jobs=common["n_jobs"],
                **xgb_regressor_extra,
            ),
        )

    if candidate_name == "lightgbm_freq_sev":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except Exception as exc:  # pragma: no cover
            raise ImportError("lightgbm package is not available") from exc
        common = {
            "n_estimators": int(params.get("n_estimators", 280)),
            "max_depth": int(params.get("max_depth", -1)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "num_leaves": int(params.get("num_leaves", 63)),
            "subsample": float(params.get("subsample", 0.85)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.85)),
            "random_state": int(params.get("random_state", random_state)),
            "n_jobs": int(params.get("n_jobs", -1)),
            "verbose": int(params.get("verbose", -1)),
        }
        lgbm_classifier_extra: dict[str, Any] = {}
        lgbm_regressor_extra: dict[str, Any] = {}
        for source_key, target_key in [
            ("device_type", "device_type"),
            ("gpu_platform_id", "gpu_platform_id"),
            ("gpu_device_id", "gpu_device_id"),
            ("max_bin", "max_bin"),
        ]:
            if source_key in params:
                lgbm_classifier_extra[target_key] = params[source_key]
            reg_key = f"reg_{source_key}"
            if reg_key in params:
                lgbm_regressor_extra[target_key] = params[reg_key]
        return PipelineFrequencySeverityModel(
            classifier=LGBMClassifier(
                n_estimators=common["n_estimators"],
                max_depth=common["max_depth"],
                learning_rate=common["learning_rate"],
                num_leaves=common["num_leaves"],
                subsample=common["subsample"],
                colsample_bytree=common["colsample_bytree"],
                verbose=common["verbose"],
                random_state=common["random_state"],
                n_jobs=common["n_jobs"],
                **lgbm_classifier_extra,
            ),
            regressor=LGBMRegressor(
                n_estimators=int(params.get("reg_n_estimators", common["n_estimators"] + 40)),
                max_depth=int(params.get("reg_max_depth", common["max_depth"])),
                learning_rate=float(params.get("reg_learning_rate", common["learning_rate"])),
                num_leaves=int(params.get("reg_num_leaves", common["num_leaves"])),
                subsample=float(params.get("reg_subsample", common["subsample"])),
                colsample_bytree=float(params.get("reg_colsample_bytree", common["colsample_bytree"])),
                verbose=common["verbose"],
                random_state=common["random_state"],
                n_jobs=common["n_jobs"],
                **lgbm_regressor_extra,
            ),
        )

    if candidate_name == "catboost_freq_sev":
        reg_iterations = params.get("reg_iterations")
        reg_learning_rate = params.get("reg_learning_rate")
        reg_depth = params.get("reg_depth")
        reg_l2_leaf_reg = params.get("reg_l2_leaf_reg")
        reg_random_strength = params.get("reg_random_strength")
        reg_bagging_temperature = params.get("reg_bagging_temperature")
        reg_border_count = params.get("reg_border_count")
        severity_loss_function = str(params.get("severity_loss_function", params.get("severity_loss", "RMSE")))
        tweedie_variance_power = float(params.get("tweedie_variance_power", 1.5))
        return CatBoostFrequencySeverityModel(
            random_state=int(params.get("random_state", random_state)),
            iterations=int(params.get("iterations", 250)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            depth=int(params.get("depth", 6)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
            random_strength=float(params.get("random_strength", 1.0)),
            bagging_temperature=float(params.get("bagging_temperature", 0.0)),
            border_count=int(params.get("border_count", 254)),
            reg_iterations=int(reg_iterations) if reg_iterations is not None else None,
            reg_learning_rate=float(reg_learning_rate) if reg_learning_rate is not None else None,
            reg_depth=int(reg_depth) if reg_depth is not None else None,
            reg_l2_leaf_reg=float(reg_l2_leaf_reg) if reg_l2_leaf_reg is not None else None,
            reg_random_strength=float(reg_random_strength) if reg_random_strength is not None else None,
            reg_bagging_temperature=float(reg_bagging_temperature) if reg_bagging_temperature is not None else None,
            reg_border_count=int(reg_border_count) if reg_border_count is not None else None,
            thread_count=int(params.get("thread_count", -1)),
            severity_loss_function=severity_loss_function,
            tweedie_variance_power=tweedie_variance_power,
            task_type=str(params.get("task_type", "CPU")),
            devices=(str(params.get("devices")) if params.get("devices") is not None else None),
        )

    if candidate_name == "catboost_dep_freq_sev":
        reg_iterations = params.get("reg_iterations")
        reg_learning_rate = params.get("reg_learning_rate")
        reg_depth = params.get("reg_depth")
        reg_l2_leaf_reg = params.get("reg_l2_leaf_reg")
        reg_random_strength = params.get("reg_random_strength")
        reg_bagging_temperature = params.get("reg_bagging_temperature")
        reg_border_count = params.get("reg_border_count")
        severity_loss_function = str(params.get("severity_loss_function", params.get("severity_loss", "RMSE")))
        tweedie_variance_power = float(params.get("tweedie_variance_power", 1.5))
        return DependentCatBoostFrequencySeverityModel(
            random_state=int(params.get("random_state", random_state)),
            iterations=int(params.get("iterations", 250)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            depth=int(params.get("depth", 6)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
            random_strength=float(params.get("random_strength", 1.0)),
            bagging_temperature=float(params.get("bagging_temperature", 0.0)),
            border_count=int(params.get("border_count", 254)),
            reg_iterations=int(reg_iterations) if reg_iterations is not None else None,
            reg_learning_rate=float(reg_learning_rate) if reg_learning_rate is not None else None,
            reg_depth=int(reg_depth) if reg_depth is not None else None,
            reg_l2_leaf_reg=float(reg_l2_leaf_reg) if reg_l2_leaf_reg is not None else None,
            reg_random_strength=float(reg_random_strength) if reg_random_strength is not None else None,
            reg_bagging_temperature=float(reg_bagging_temperature) if reg_bagging_temperature is not None else None,
            reg_border_count=int(reg_border_count) if reg_border_count is not None else None,
            thread_count=int(params.get("thread_count", -1)),
            severity_loss_function=severity_loss_function,
            tweedie_variance_power=tweedie_variance_power,
            task_type=str(params.get("task_type", "CPU")),
            devices=(str(params.get("devices")) if params.get("devices") is not None else None),
            dep_oof_folds=int(params.get("dep_oof_folds", 5)),
            dep_frequency_signal_name=str(params.get("dep_frequency_signal_name", "freq_risk_signal")),
            dep_use_frequency_signal=bool(params.get("dep_use_frequency_signal", True)),
        )

    if candidate_name == "woe_freq_sev":
        return WoEFrequencySeverityModel(
            n_bins=int(params.get("n_bins", 10)),
            min_bin_size=int(params.get("min_bin_size", 50)),
            max_iter=int(params.get("max_iter", model_max_iter)),
            ridge_alpha=float(params.get("ridge_alpha", model_ridge_alpha)),
            iv_threshold=float(params.get("iv_threshold", 0.02)),
            random_state=int(params.get("random_state", random_state)),
        )

    if candidate_name == "tweedie_aggregate":
        return TweedieAggregateLossModel(
            tweedie_variance_power=float(params.get("tweedie_variance_power", 1.5)),
            iterations=int(params.get("iterations", 500)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            depth=int(params.get("depth", 6)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
            random_state=int(params.get("random_state", random_state)),
            thread_count=int(params.get("thread_count", -1)),
            task_type=str(params.get("task_type", "CPU")),
            devices=(str(params.get("devices")) if params.get("devices") is not None else None),
            use_catboost=bool(params.get("use_catboost", True)),
        )

    raise ValueError(f"Unknown benchmark candidate: {candidate_name}")


def _iter_cv_splits(
    df: pd.DataFrame,
    n_splits: int,
    random_state: int,
    group_column: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = max(2, int(n_splits))
    if group_column in df.columns:
        groups = df[group_column].astype("string").fillna("__missing_group__").astype(str)
        unique_groups = int(groups.nunique(dropna=False))
        if unique_groups >= 2:
            n_splits = min(n_splits, unique_groups)
            splitter = GroupKFold(n_splits=n_splits)
            return list(splitter.split(df, groups=groups))
    n_splits = min(n_splits, len(df))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(df))


def _generate_simplex_weights(candidate_names: list[str], step: float) -> list[dict[str, float]]:
    if not candidate_names:
        return []
    if len(candidate_names) == 1:
        return [{candidate_names[0]: 1.0}]

    step = float(step)
    if step <= 0 or step > 1:
        step = 0.05
    units = max(1, int(round(1.0 / step)))

    vectors: list[dict[str, float]] = []

    def build(level: int, remaining: int, acc: list[int]) -> None:
        if level == len(candidate_names) - 1:
            values = acc + [remaining]
            vectors.append({name: value / units for name, value in zip(candidate_names, values)})
            return
        for value in range(remaining + 1):
            build(level + 1, remaining - value, acc + [value])

    build(0, units, [])
    return vectors


def _fit_predict_candidate(
    candidate_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    benchmark_config: BenchmarkConfig,
    model_max_iter: int,
    model_ridge_alpha: float,
) -> tuple[Any, pd.DataFrame]:
    model = _build_candidate_model(
        candidate_name=candidate_name,
        model_max_iter=model_max_iter,
        model_ridge_alpha=model_ridge_alpha,
        random_state=benchmark_config.random_state,
        candidate_params=benchmark_config.candidate_params,
    )
    model.fit(train_df)
    valid_pred = model.predict(valid_df)
    required_columns = {"p_claim", "expected_severity", "expected_loss"}
    missing = required_columns.difference(valid_pred.columns)
    if missing:
        raise ValueError(f"Missing prediction columns for {candidate_name}: {sorted(missing)}")
    return model, valid_pred


def _apply_calibrator_to_predictions(valid_pred: pd.DataFrame, calibrator: ProbabilityCalibrator) -> pd.DataFrame:
    calibrated_pred = valid_pred.copy()
    calibrated_p_claim = calibrator.transform(calibrated_pred["p_claim"].values)
    calibrated_pred["p_claim"] = calibrated_p_claim
    calibrated_pred["expected_loss"] = calibrated_pred["expected_severity"].values * calibrated_p_claim
    return calibrated_pred


def _fit_candidate_probability_calibrator(
    candidate_name: str,
    train_df: pd.DataFrame,
    benchmark_config: BenchmarkConfig,
    model_max_iter: int,
    model_ridge_alpha: float,
    logger: logging.Logger | None = None,
) -> tuple[ProbabilityCalibrator | None, dict[str, Any]]:
    calibration_cfg = benchmark_config.calibration
    metadata: dict[str, Any] = {
        "calibration_status": "disabled",
        "calibration_method": calibration_cfg.method,
        "calibration_reason": "disabled_in_config",
        "calibration_oof_rows": 0,
        "calibration_oof_brier_raw": None,
        "calibration_oof_brier_calibrated": None,
        "calibration_oof_auc_raw": None,
        "calibration_oof_auc_calibrated": None,
    }

    if not calibration_cfg.enabled or calibration_cfg.method == "none":
        return None, metadata

    metadata["calibration_status"] = "skipped"
    metadata["calibration_reason"] = "not_enough_samples"
    if len(train_df) < calibration_cfg.min_samples:
        return None, metadata

    if candidate_name == "oof_blend_freq_sev":
        metadata["calibration_reason"] = "blend_candidate_not_supported"
        return None, metadata

    y_train = train_df[TARGET_CLAIM_COL].fillna(0).astype(int).values
    if len(np.unique(y_train)) < 2:
        metadata["calibration_reason"] = "single_class_target"
        return None, metadata

    try:
        splits = _iter_cv_splits(
            df=train_df,
            n_splits=calibration_cfg.oof_folds,
            random_state=benchmark_config.random_state,
            group_column=calibration_cfg.group_column,
        )
        oof_p_claim = np.full(len(train_df), np.nan, dtype=float)
        for fold_train_idx, fold_valid_idx in splits:
            fold_train = train_df.iloc[fold_train_idx]
            fold_valid = train_df.iloc[fold_valid_idx]
            _, fold_valid_pred = _fit_predict_candidate(
                candidate_name=candidate_name,
                train_df=fold_train,
                valid_df=fold_valid,
                benchmark_config=benchmark_config,
                model_max_iter=model_max_iter,
                model_ridge_alpha=model_ridge_alpha,
            )
            oof_p_claim[fold_valid_idx] = fold_valid_pred["p_claim"].values
        if np.isnan(oof_p_claim).any():
            raise ValueError("OOF predictions contain NaNs")
    except Exception as exc:
        metadata["calibration_status"] = "failed"
        metadata["calibration_reason"] = f"oof_error:{exc}"
        if logger:
            logger.warning("Calibration OOF failed for %s: %s", candidate_name, exc)
        return None, metadata

    try:
        calibrator = ProbabilityCalibrator.fit(
            y_true=y_train,
            raw_probabilities=oof_p_claim,
            method=calibration_cfg.method,
            clip_eps=calibration_cfg.clip_eps,
        )
        calibrated_oof = calibrator.transform(oof_p_claim)
        raw_metrics = classification_metrics(y_train, oof_p_claim)
        calibrated_metrics = classification_metrics(y_train, calibrated_oof)
    except Exception as exc:
        metadata["calibration_status"] = "failed"
        metadata["calibration_reason"] = f"fit_error:{exc}"
        if logger:
            logger.warning("Calibration fit failed for %s: %s", candidate_name, exc)
        return None, metadata

    metadata.update(
        {
            "calibration_status": "applied",
            "calibration_reason": "ok",
            "calibration_oof_rows": int(len(train_df)),
            "calibration_oof_brier_raw": raw_metrics.get("brier"),
            "calibration_oof_brier_calibrated": calibrated_metrics.get("brier"),
            "calibration_oof_auc_raw": raw_metrics.get("auc"),
            "calibration_oof_auc_calibrated": calibrated_metrics.get("auc"),
        }
    )
    return calibrator, metadata


def _evaluate_candidate_predictions(
    valid_df: pd.DataFrame,
    valid_pred: pd.DataFrame,
    pricing_target_lr: float,
    pricing_alpha_grid: np.ndarray,
    pricing_beta_grid: np.ndarray,
    pricing_target_band: tuple[float, float] | None,
    pricing_optimization_method: str,
    pricing_retention: RetentionConfig | None,
    pricing_slsqp_options: dict[str, Any] | None,
    pricing_stratified_config: StratifiedPricingConfig | None,
) -> tuple[dict[str, Any], dict[str, Any], float, float, pd.Series, PricingEvaluation]:
    y_valid = valid_df[TARGET_CLAIM_COL].fillna(0).astype(int).values
    ml = classification_metrics(y_valid, valid_pred["p_claim"].values)
    pos_mask = valid_df[TARGET_CLAIM_COL].fillna(0).astype(int) > 0
    severity = severity_metrics(
        y_true=np.log1p(pd.to_numeric(valid_df.loc[pos_mask, TARGET_AMOUNT_COL], errors="coerce").fillna(0.0).values),
        y_pred=np.log1p(valid_pred.loc[pos_mask, "expected_severity"].clip(lower=0.0).values),
    )
    alpha, beta, premium, pricing_eval = select_best_pricing(
        df=valid_df,
        expected_loss=valid_pred["expected_loss"],
        target_lr=pricing_target_lr,
        alpha_grid=pricing_alpha_grid,
        beta_grid=pricing_beta_grid,
        target_band=pricing_target_band,
        method=pricing_optimization_method,
        retention_config=pricing_retention,
        slsqp_options=pricing_slsqp_options,
        stratified_config=pricing_stratified_config,
    )
    return ml, severity, alpha, beta, premium, pricing_eval


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _serialize_oof_checkpoint_candidate(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "weights": {str(key): float(value) for key, value in dict(payload.get("weights") or {}).items()},
        "alpha": payload.get("alpha"),
        "beta": payload.get("beta"),
        "pricing_dict": dict(payload.get("pricing_dict") or {}),
        "passes_constraints": bool(payload.get("passes_constraints", False)),
        "metric_value": payload.get("metric_value"),
    }


def _deserialize_oof_checkpoint_candidate(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    return {
        "weights": {str(key): float(value) for key, value in dict(payload.get("weights") or {}).items()},
        "alpha": payload.get("alpha"),
        "beta": payload.get("beta"),
        "pricing_dict": dict(payload.get("pricing_dict") or {}),
        "passes_constraints": bool(payload.get("passes_constraints", False)),
        "metric_value": payload.get("metric_value"),
    }


def _oof_blend_checkpoint_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "state.json"


def _oof_blend_checkpoint_predictions_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "oof_predictions.joblib"


def _load_oof_blend_checkpoint(checkpoint_dir: Path) -> tuple[dict[str, Any] | None, dict[str, pd.DataFrame] | None]:
    state_path = _oof_blend_checkpoint_state_path(checkpoint_dir)
    predictions_path = _oof_blend_checkpoint_predictions_path(checkpoint_dir)
    state: dict[str, Any] | None = None
    predictions: dict[str, pd.DataFrame] | None = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    if predictions_path.exists():
        loaded = load(predictions_path)
        if isinstance(loaded, dict):
            predictions = loaded
    return state, predictions


def _save_oof_blend_checkpoint(
    checkpoint_dir: Path,
    state: dict[str, Any],
    oof_predictions: dict[str, pd.DataFrame] | None = None,
) -> None:
    ensure_dir(checkpoint_dir)
    if oof_predictions is not None:
        dump(oof_predictions, _oof_blend_checkpoint_predictions_path(checkpoint_dir))
    _write_json_atomic(_oof_blend_checkpoint_state_path(checkpoint_dir), state)


def _evaluate_oof_blend_candidate(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    benchmark_config: BenchmarkConfig,
    pricing_target_lr: float,
    pricing_alpha_grid: np.ndarray,
    pricing_beta_grid: np.ndarray,
    pricing_target_band: tuple[float, float] | None,
    model_max_iter: int,
    model_ridge_alpha: float,
    pricing_optimization_method: str,
    pricing_retention: RetentionConfig | None,
    pricing_slsqp_options: dict[str, Any] | None,
    pricing_stratified_config: StratifiedPricingConfig | None,
    logger: logging.Logger | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, Any], dict[str, Any], float, float, pd.Series, PricingEvaluation, dict[str, Any]]:
    blend_params = dict(benchmark_config.candidate_params.get("oof_blend_freq_sev") or {})
    base_candidates = [str(item) for item in blend_params.get("base_candidates", ["catboost_freq_sev", "xgboost_freq_sev"])]
    base_candidates = [item for item in base_candidates if item != "oof_blend_freq_sev"]
    if not base_candidates:
        raise ValueError("oof_blend_freq_sev requires at least one base candidate")

    oof_folds = int(blend_params.get("oof_folds", 5))
    oof_group_column = str(blend_params.get("oof_group_column", "contract_number"))
    weight_grid_step = float(blend_params.get("weight_grid_step", 0.05))
    checkpoint_dir_raw = blend_params.get("resume_from_checkpoint") or blend_params.get("checkpoint_dir")
    checkpoint_dir = Path(str(checkpoint_dir_raw)).resolve() if checkpoint_dir_raw else None
    checkpoint_state: dict[str, Any] | None = None
    checkpoint_predictions: dict[str, pd.DataFrame] | None = None
    if checkpoint_dir is not None:
        checkpoint_state, checkpoint_predictions = _load_oof_blend_checkpoint(checkpoint_dir)
        if checkpoint_state:
            expected_signature = {
                "base_candidates": base_candidates,
                "oof_folds": oof_folds,
                "oof_group_column": oof_group_column,
                "weight_grid_step": weight_grid_step,
            }
            actual_signature = {
                "base_candidates": checkpoint_state.get("base_candidates"),
                "oof_folds": checkpoint_state.get("oof_folds"),
                "oof_group_column": checkpoint_state.get("oof_group_column"),
                "weight_grid_step": checkpoint_state.get("weight_grid_step"),
            }
            if actual_signature != expected_signature:
                raise ValueError(
                    "OOF blend checkpoint parameters do not match current run configuration"
                )
            if logger:
                logger.info(
                    "OOF blend checkpoint loaded: dir=%s stage=%s completed_folds=%s completed_combos=%s updated_at_utc=%s",
                    checkpoint_dir,
                    checkpoint_state.get("stage"),
                    len(checkpoint_state.get("completed_folds") or []),
                    ((checkpoint_state.get("weight_search") or {}).get("completed_combos")),
                    checkpoint_state.get("updated_at_utc"),
                )

    if logger:
        logger.info(
            "OOF blend start: base_candidates=%s oof_folds=%s group_column=%s weight_grid_step=%s train_rows=%s valid_rows=%s checkpoint_dir=%s",
            base_candidates,
            oof_folds,
            oof_group_column,
            weight_grid_step,
            len(train_df),
            len(valid_df),
            str(checkpoint_dir) if checkpoint_dir else None,
        )

    splits = _iter_cv_splits(
        df=train_df,
        n_splits=oof_folds,
        random_state=benchmark_config.random_state,
        group_column=oof_group_column,
    )
    if logger:
        logger.info("OOF blend prepared %s CV folds", len(splits))
    oof_predictions = checkpoint_predictions or {
        candidate_name: pd.DataFrame(index=train_df.index, columns=["p_claim", "expected_severity", "expected_loss"], dtype=float)
        for candidate_name in base_candidates
    }
    for candidate_name in base_candidates:
        if candidate_name not in oof_predictions:
            oof_predictions[candidate_name] = pd.DataFrame(
                index=train_df.index,
                columns=["p_claim", "expected_severity", "expected_loss"],
                dtype=float,
            )

    completed_folds = {int(value) for value in (checkpoint_state or {}).get("completed_folds", [])}
    checkpoint_base = checkpoint_state or {
        "version": 1,
        "stage": "oof_folds",
        "base_candidates": base_candidates,
        "oof_folds": oof_folds,
        "oof_group_column": oof_group_column,
        "weight_grid_step": weight_grid_step,
        "completed_folds": [],
        "weight_search": {
            "completed_combos": 0,
            "best_any": None,
            "best_compliant": None,
        },
    }

    total_fit_steps = max(1, len(splits) * len(base_candidates))
    fit_step = 0
    for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
        if fold_number in completed_folds:
            fit_step += len(base_candidates)
            if logger:
                logger.info("OOF blend fold resume skip: %s/%s", fold_number, len(splits))
            continue
        fold_train = train_df.iloc[train_idx]
        fold_valid = train_df.iloc[valid_idx]
        if logger:
            logger.info(
                "OOF blend fold start: %s/%s train_rows=%s valid_rows=%s",
                fold_number,
                len(splits),
                len(fold_train),
                len(fold_valid),
            )
        for candidate_name in base_candidates:
            fit_step += 1
            base_start = time.perf_counter()
            if logger:
                logger.info(
                    "OOF blend base fit start: fold=%s/%s base=%s step=%s/%s",
                    fold_number,
                    len(splits),
                    candidate_name,
                    fit_step,
                    total_fit_steps,
                )
            _, fold_pred = _fit_predict_candidate(
                candidate_name=candidate_name,
                train_df=fold_train,
                valid_df=fold_valid,
                benchmark_config=benchmark_config,
                model_max_iter=model_max_iter,
                model_ridge_alpha=model_ridge_alpha,
            )
            oof_predictions[candidate_name].iloc[valid_idx] = fold_pred.loc[:, ["p_claim", "expected_severity", "expected_loss"]].values
            if logger:
                logger.info(
                    "OOF blend base fit done: fold=%s/%s base=%s elapsed=%.2fs",
                    fold_number,
                    len(splits),
                    candidate_name,
                    float(time.perf_counter() - base_start),
                )
        if logger:
            logger.info("OOF blend fold done: %s/%s", fold_number, len(splits))
        completed_folds.add(fold_number)
        if checkpoint_dir is not None:
            checkpoint_base["stage"] = "oof_folds"
            checkpoint_base["completed_folds"] = sorted(completed_folds)
            checkpoint_base["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _save_oof_blend_checkpoint(checkpoint_dir, checkpoint_base, oof_predictions=oof_predictions)
            if logger:
                logger.info(
                    "OOF blend checkpoint saved: stage=oof_folds fold=%s/%s dir=%s",
                    fold_number,
                    len(splits),
                    checkpoint_dir,
                )

    for candidate_name, prediction in oof_predictions.items():
        if prediction.isna().any().any():
            missing_count = int(prediction.isna().sum().sum())
            raise ValueError(f"OOF predictions contain NaNs for {candidate_name}: {missing_count}")

    weight_grid = _generate_simplex_weights(base_candidates, step=weight_grid_step)
    if not weight_grid:
        raise ValueError("Could not generate blend weight grid")
    if logger:
        logger.info("OOF blend weight search start: combinations=%s", len(weight_grid))

    checkpoint_weight_search = dict(checkpoint_base.get("weight_search") or {})
    best_any = _deserialize_oof_checkpoint_candidate(checkpoint_weight_search.get("best_any"))
    best_compliant = _deserialize_oof_checkpoint_candidate(checkpoint_weight_search.get("best_compliant"))
    completed_combos = int(checkpoint_weight_search.get("completed_combos") or 0)
    if checkpoint_dir is not None:
        checkpoint_base["stage"] = "weight_search"
        checkpoint_base["completed_folds"] = sorted(completed_folds)
        checkpoint_base["weight_search"] = {
            "completed_combos": completed_combos,
            "best_any": _serialize_oof_checkpoint_candidate(best_any),
            "best_compliant": _serialize_oof_checkpoint_candidate(best_compliant),
            "total_combos": len(weight_grid),
        }
        checkpoint_base["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _save_oof_blend_checkpoint(checkpoint_dir, checkpoint_base, oof_predictions=oof_predictions)
    log_every = max(1, len(weight_grid) // 10)

    for combo_idx, weight_map in enumerate(weight_grid, start=1):
        if combo_idx <= completed_combos:
            if logger and (combo_idx == completed_combos or combo_idx == 1):
                logger.info("OOF blend weight resume skip: combo=%s/%s", combo_idx, len(weight_grid))
            continue
        combo_start = time.perf_counter()
        if logger:
            logger.info("OOF blend weight combo start: combo=%s/%s weights=%s", combo_idx, len(weight_grid), weight_map)
        blended_oof_pred = _blend_prediction_frames(oof_predictions, weights=weight_map)
        alpha, beta, premium, pricing_eval = select_best_pricing(
            df=train_df,
            expected_loss=blended_oof_pred["expected_loss"],
            target_lr=pricing_target_lr,
            alpha_grid=pricing_alpha_grid,
            beta_grid=pricing_beta_grid,
            target_band=pricing_target_band,
            method=pricing_optimization_method,
            retention_config=pricing_retention,
            slsqp_options=pricing_slsqp_options,
            stratified_config=pricing_stratified_config,
        )
        pricing_dict = {"alpha": alpha, "beta": beta, **pricing_eval.to_dict()}
        passes_constraints, _ = _evaluate_constraints(pricing_dict, benchmark_config.constraints)
        metric_value = _safe_float(pricing_dict.get(benchmark_config.selection_metric))
        candidate_payload = {
            "weights": weight_map,
            "alpha": alpha,
            "beta": beta,
            "premium": premium,
            "pricing_eval": pricing_eval,
            "pricing_dict": pricing_dict,
            "passes_constraints": passes_constraints,
            "metric_value": metric_value,
        }

        improved_any = False
        improved_compliant = False
        if best_any is None or metric_value > best_any["metric_value"]:
            best_any = candidate_payload
            improved_any = True
        if passes_constraints and (best_compliant is None or metric_value > best_compliant["metric_value"]):
            best_compliant = candidate_payload
            improved_compliant = True

        completed_combos = combo_idx
        if checkpoint_dir is not None:
            checkpoint_base["stage"] = "weight_search"
            checkpoint_base["weight_search"] = {
                "completed_combos": completed_combos,
                "best_any": _serialize_oof_checkpoint_candidate(best_any),
                "best_compliant": _serialize_oof_checkpoint_candidate(best_compliant),
                "total_combos": len(weight_grid),
            }
            checkpoint_base["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _save_oof_blend_checkpoint(checkpoint_dir, checkpoint_base)

        if logger and (improved_any or improved_compliant or combo_idx == 1 or combo_idx == len(weight_grid) or combo_idx % log_every == 0):
            logger.info(
                "OOF blend weight progress: combo=%s/%s policy_score=%s passes=%s best_any=%s best_compliant=%s elapsed=%.2fs weights=%s",
                combo_idx,
                len(weight_grid),
                pricing_dict.get("policy_score"),
                passes_constraints,
                best_any["pricing_dict"].get("policy_score") if best_any else None,
                best_compliant["pricing_dict"].get("policy_score") if best_compliant else None,
                float(time.perf_counter() - combo_start),
                weight_map if (improved_any or improved_compliant) else None,
            )

    selected = best_compliant or best_any
    if selected is None:
        raise ValueError("Failed to select blend weights")
    if checkpoint_dir is not None:
        checkpoint_base["stage"] = "weight_search_done"
        checkpoint_base["selected_weights"] = dict(selected["weights"])
        checkpoint_base["selected_oof_policy_score"] = selected["pricing_dict"].get("policy_score")
        checkpoint_base["selected_constraints_pass"] = bool(selected["passes_constraints"])
        checkpoint_base["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _save_oof_blend_checkpoint(checkpoint_dir, checkpoint_base)
    if logger:
        logger.info(
            "OOF blend weight search done: selected_weights=%s selected_oof_policy_score=%s selected_constraints_pass=%s",
            selected["weights"],
            selected["pricing_dict"].get("policy_score"),
            selected["passes_constraints"],
        )

    fitted_base_models: dict[str, Any] = {}
    valid_predictions_by_candidate: dict[str, pd.DataFrame] = {}
    for candidate_name in base_candidates:
        base_start = time.perf_counter()
        if logger:
            logger.info("OOF blend full fit start: base=%s", candidate_name)
        base_model, base_valid_pred = _fit_predict_candidate(
            candidate_name=candidate_name,
            train_df=train_df,
            valid_df=valid_df,
            benchmark_config=benchmark_config,
            model_max_iter=model_max_iter,
            model_ridge_alpha=model_ridge_alpha,
        )
        fitted_base_models[candidate_name] = base_model
        valid_predictions_by_candidate[candidate_name] = base_valid_pred
        if logger:
            logger.info("OOF blend full fit done: base=%s elapsed=%.2fs", candidate_name, float(time.perf_counter() - base_start))

    blended_valid_pred = _blend_prediction_frames(valid_predictions_by_candidate, weights=selected["weights"])
    blend_model = OOFWeightedBlendModel(base_models=fitted_base_models, weights=selected["weights"])
    ml, severity, alpha, beta, premium, pricing_eval = _evaluate_candidate_predictions(
        valid_df=valid_df,
        valid_pred=blended_valid_pred,
        pricing_target_lr=pricing_target_lr,
        pricing_alpha_grid=pricing_alpha_grid,
        pricing_beta_grid=pricing_beta_grid,
        pricing_target_band=pricing_target_band,
        pricing_optimization_method=pricing_optimization_method,
        pricing_retention=pricing_retention,
        pricing_slsqp_options=pricing_slsqp_options,
        pricing_stratified_config=pricing_stratified_config,
    )
    if logger:
        pricing_dict = pricing_eval.to_dict()
        logger.info(
            "OOF blend final validation: policy_score=%s lr_total=%s alpha=%s beta=%s gini=%s",
            pricing_dict.get("policy_score"),
            pricing_dict.get("lr_total"),
            alpha,
            beta,
            ml.get("gini"),
        )

    blend_metadata = {
        "base_candidates": base_candidates,
        "weights": selected["weights"],
        "oof_folds": oof_folds,
        "oof_group_column": oof_group_column,
        "weight_grid_step": weight_grid_step,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "oof_policy_score": selected["pricing_dict"].get("policy_score"),
        "oof_constraints_pass": bool(selected["passes_constraints"]),
        "oof_alpha": selected["alpha"],
        "oof_beta": selected["beta"],
    }
    return blend_model, blended_valid_pred, ml, severity, alpha, beta, premium, pricing_eval, blend_metadata


def _evaluate_constraints(pricing: dict[str, Any], constraints: BenchmarkConstraints) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    violations = int(pricing.get("violations") or 0)
    lr_total = _safe_float(pricing.get("lr_total"))
    share_group1 = _safe_float(pricing.get("share_group1"))

    if violations > constraints.max_violations:
        reasons.append(f"violations>{constraints.max_violations}")
    if lr_total < constraints.lr_total_min:
        reasons.append(f"lr_total<{constraints.lr_total_min}")
    if lr_total > constraints.lr_total_max:
        reasons.append(f"lr_total>{constraints.lr_total_max}")
    if share_group1 < constraints.share_group1_min:
        reasons.append(f"share_group1<{constraints.share_group1_min}")
    if share_group1 > constraints.share_group1_max:
        reasons.append(f"share_group1>{constraints.share_group1_max}")

    return len(reasons) == 0, reasons


def _candidate_sort_key(result: CandidateResult, selection_metric: str, stability_penalty: float) -> tuple[float, float]:
    metric_value = _safe_float((result.pricing or {}).get(selection_metric))
    metric_std = _safe_float((result.pricing or {}).get(f"{selection_metric}_std"), default=0.0)
    adjusted_metric = metric_value - max(stability_penalty, 0.0) * metric_std
    gini_value = _safe_float((result.ml or {}).get("gini"))
    return adjusted_metric, gini_value


def select_benchmark_winner(results: list[CandidateResult], config: BenchmarkConfig) -> tuple[str, str]:
    usable = [result for result in results if result.status == "ok"]
    if not usable:
        raise ValueError("No usable benchmark candidates")

    compliant = [result for result in usable if result.passes_constraints]
    if compliant:
        winner = max(
            compliant,
            key=lambda item: _candidate_sort_key(item, config.selection_metric, config.stability_penalty),
        )
        return winner.candidate_name, "best_compliant_candidate"

    if config.must_pass_constraints:
        raise ValueError("No compliant benchmark candidates under must_pass_constraints=true")

    if config.fallback_strategy == "configured_candidate":
        fallback = next((result for result in usable if result.candidate_name == config.fallback_candidate), None)
        if fallback is not None:
            return fallback.candidate_name, "fallback_no_compliant_candidates"

    winner = max(
        usable,
        key=lambda item: _candidate_sort_key(item, config.selection_metric, config.stability_penalty),
    )
    return winner.candidate_name, "best_available_no_compliant_candidates"


def run_model_benchmark(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    benchmark_config: BenchmarkConfig,
    pricing_target_lr: float,
    pricing_alpha_grid: np.ndarray,
    pricing_beta_grid: np.ndarray,
    pricing_target_band: tuple[float, float] | None,
    model_max_iter: int,
    model_ridge_alpha: float,
    pricing_optimization_method: str = "grid",
    pricing_retention: RetentionConfig | None = None,
    pricing_slsqp_options: dict[str, Any] | None = None,
    pricing_stratified_config: StratifiedPricingConfig | None = None,
    logger: logging.Logger | None = None,
) -> BenchmarkResult:
    results: list[CandidateResult] = []
    ok_payload: dict[str, dict[str, Any]] = {}

    for candidate_name in benchmark_config.candidates:
        start = time.perf_counter()
        if logger:
            logger.info("Benchmark candidate start: %s", candidate_name)

        try:
            blend_metadata: dict[str, Any] | None = None
            calibration_metadata: dict[str, Any] | None = None
            if candidate_name == "oof_blend_freq_sev":
                (
                    model,
                    valid_pred,
                    ml,
                    severity,
                    alpha,
                    beta,
                    premium,
                    pricing_eval,
                    blend_metadata,
                ) = _evaluate_oof_blend_candidate(
                    train_df=train_df,
                    valid_df=valid_df,
                    benchmark_config=benchmark_config,
                    pricing_target_lr=pricing_target_lr,
                    pricing_alpha_grid=pricing_alpha_grid,
                    pricing_beta_grid=pricing_beta_grid,
                    pricing_target_band=pricing_target_band,
                    model_max_iter=model_max_iter,
                    model_ridge_alpha=model_ridge_alpha,
                    pricing_optimization_method=pricing_optimization_method,
                    pricing_retention=pricing_retention,
                    pricing_slsqp_options=pricing_slsqp_options,
                    pricing_stratified_config=pricing_stratified_config,
                    logger=logger,
                )
            else:
                model, valid_pred = _fit_predict_candidate(
                    candidate_name=candidate_name,
                    train_df=train_df,
                    valid_df=valid_df,
                    benchmark_config=benchmark_config,
                    model_max_iter=model_max_iter,
                    model_ridge_alpha=model_ridge_alpha,
                )
                ml, severity, alpha, beta, premium, pricing_eval = _evaluate_candidate_predictions(
                    valid_df=valid_df,
                    valid_pred=valid_pred,
                    pricing_target_lr=pricing_target_lr,
                    pricing_alpha_grid=pricing_alpha_grid,
                    pricing_beta_grid=pricing_beta_grid,
                    pricing_target_band=pricing_target_band,
                    pricing_optimization_method=pricing_optimization_method,
                    pricing_retention=pricing_retention,
                    pricing_slsqp_options=pricing_slsqp_options,
                    pricing_stratified_config=pricing_stratified_config,
                )

            if benchmark_config.calibration.enabled:
                calibrator, calibration_metadata = _fit_candidate_probability_calibrator(
                    candidate_name=candidate_name,
                    train_df=train_df,
                    benchmark_config=benchmark_config,
                    model_max_iter=model_max_iter,
                    model_ridge_alpha=model_ridge_alpha,
                    logger=logger,
                )
                if calibrator is not None:
                    valid_pred = _apply_calibrator_to_predictions(valid_pred, calibrator)
                    model = CalibratedFrequencySeverityModel(base_model=model, calibrator=calibrator)
                    ml, severity, alpha, beta, premium, pricing_eval = _evaluate_candidate_predictions(
                        valid_df=valid_df,
                        valid_pred=valid_pred,
                        pricing_target_lr=pricing_target_lr,
                        pricing_alpha_grid=pricing_alpha_grid,
                        pricing_beta_grid=pricing_beta_grid,
                        pricing_target_band=pricing_target_band,
                        pricing_optimization_method=pricing_optimization_method,
                        pricing_retention=pricing_retention,
                        pricing_slsqp_options=pricing_slsqp_options,
                        pricing_stratified_config=pricing_stratified_config,
                    )

            pricing_dict = {"alpha": alpha, "beta": beta, **pricing_eval.to_dict()}
            passes_constraints, constraint_reasons = _evaluate_constraints(pricing_dict, benchmark_config.constraints)
            elapsed_seconds = float(time.perf_counter() - start)
            metadata_payload: dict[str, Any] | None = None
            if blend_metadata or calibration_metadata:
                metadata_payload = {}
                if blend_metadata:
                    metadata_payload.update(blend_metadata)
                if calibration_metadata:
                    metadata_payload.update(calibration_metadata)

            result = CandidateResult(
                candidate_name=candidate_name,
                status="ok",
                ml=ml,
                severity=severity,
                pricing=pricing_dict,
                passes_constraints=passes_constraints,
                constraint_reasons=constraint_reasons,
                alpha=alpha,
                beta=beta,
                elapsed_seconds=elapsed_seconds,
                metadata=metadata_payload,
            )
            results.append(result)
            ok_payload[candidate_name] = {
                "model": model,
                "valid_pred": valid_pred,
                "alpha": alpha,
                "beta": beta,
                "premium": premium,
                "pricing_eval": pricing_eval,
                "ml": ml,
                "severity": severity,
            }
            if metadata_payload:
                ok_payload[candidate_name]["metadata"] = metadata_payload

            if logger:
                logger.info(
                    "Benchmark candidate done: %s policy_score=%s alpha=%s beta=%s gini=%s constraints_pass=%s elapsed=%.2fs",
                    candidate_name,
                    pricing_dict.get("policy_score"),
                    alpha,
                    beta,
                    ml.get("gini"),
                    passes_constraints,
                    elapsed_seconds,
                )
                if blend_metadata:
                    logger.info(
                        "Benchmark blend details: %s base=%s weights=%s oof_policy_score=%s",
                        candidate_name,
                        blend_metadata.get("base_candidates"),
                        blend_metadata.get("weights"),
                        blend_metadata.get("oof_policy_score"),
                    )
                if calibration_metadata:
                    logger.info(
                        "Benchmark calibration details: %s status=%s method=%s brier_raw=%s brier_calibrated=%s",
                        candidate_name,
                        calibration_metadata.get("calibration_status"),
                        calibration_metadata.get("calibration_method"),
                        calibration_metadata.get("calibration_oof_brier_raw"),
                        calibration_metadata.get("calibration_oof_brier_calibrated"),
                    )
        except Exception as exc:  # pragma: no cover - kept broad for robust pipeline fallback
            elapsed_seconds = float(time.perf_counter() - start)
            result = CandidateResult(
                candidate_name=candidate_name,
                status="failed",
                error=str(exc),
                elapsed_seconds=elapsed_seconds,
            )
            results.append(result)
            if logger:
                logger.warning(
                    "Benchmark candidate failed: %s error=%s elapsed=%.2fs",
                    candidate_name,
                    exc,
                    elapsed_seconds,
                )

    winner_name, selection_reason = select_benchmark_winner(results, benchmark_config)
    winner_payload = ok_payload.get(winner_name)
    if winner_payload is None:
        raise ValueError(f"Winner candidate payload not found: {winner_name}")

    if logger:
        logger.info("Benchmark winner: %s (%s)", winner_name, selection_reason)

    return BenchmarkResult(
        winner_name=winner_name,
        selection_reason=selection_reason,
        winner_model=winner_payload["model"],
        winner_valid_pred=winner_payload["valid_pred"],
        winner_alpha=float(winner_payload["alpha"]),
        winner_beta=float(winner_payload["beta"]),
        winner_premium=winner_payload["premium"],
        winner_pricing_eval=winner_payload["pricing_eval"],
        winner_ml=winner_payload["ml"],
        winner_severity=winner_payload["severity"],
        results=results,
    )
