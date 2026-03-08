from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from risk_case.features.builder import FeatureSchema, infer_feature_schema, prepare_features
from risk_case.settings import TARGET_AMOUNT_COL, TARGET_CLAIM_COL


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
class FrequencySeverityModel:
    max_iter: int = 300
    ridge_alpha: float = 1.0

    schema: FeatureSchema | None = None
    frequency_model: Pipeline | None = None
    severity_model: Pipeline | None = None
    severity_constant: float = 0.0
    severity_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FrequencySeverityModel":
        self.schema = infer_feature_schema(df)
        X = prepare_features(df, self.schema)
        y_freq = df[TARGET_CLAIM_COL].fillna(0).astype(int).values

        preprocessor = _build_preprocessor(self.schema)
        self.frequency_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=self.max_iter, class_weight="balanced")),
            ]
        )
        self.frequency_model.fit(X, y_freq)

        positive_mask = df[TARGET_CLAIM_COL].fillna(0).astype(int) > 0
        y_sev = pd.to_numeric(df[TARGET_AMOUNT_COL], errors="coerce").fillna(0.0)
        y_sev = y_sev.clip(lower=0)

        if positive_mask.sum() >= 20:
            sev_preprocessor = _build_preprocessor(self.schema)
            self.severity_model = Pipeline(
                steps=[
                    ("preprocessor", sev_preprocessor),
                    ("regressor", Ridge(alpha=self.ridge_alpha)),
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
        p_claim = self.frequency_model.predict_proba(X)[:, 1]

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
    def load(path: Path | str) -> "FrequencySeverityModel":
        return load(path)
