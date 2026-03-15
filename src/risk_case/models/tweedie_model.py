"""Tweedie aggregate loss model.

Instead of separate frequency + severity models, this model predicts total
expected loss E(claim_amount) directly using a single Tweedie regression.

Tweedie regression is the standard actuarial approach for modelling aggregate
losses with a mass at zero (no-claim policies) and a continuous positive tail.

The model follows the same .fit() / .predict() / .save() / .load() contract
as existing benchmark candidates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load

from risk_case.features.builder import FeatureSchema, infer_feature_schema, prepare_features
from risk_case.settings import TARGET_AMOUNT_COL, TARGET_CLAIM_COL

LOGGER = logging.getLogger("risk_case.models.tweedie")


class TweedieAggregateLossModel:
    """Single-model Tweedie regression for aggregate loss prediction.

    Uses CatBoost with Tweedie loss function when available, otherwise falls
    back to sklearn's TweedieRegressor.
    """

    def __init__(
        self,
        tweedie_variance_power: float = 1.5,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_state: int = 42,
        thread_count: int = -1,
        task_type: str = "CPU",
        devices: str | None = None,
        use_catboost: bool = True,
    ) -> None:
        self.tweedie_variance_power = float(np.clip(tweedie_variance_power, 1.01, 1.99))
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.thread_count = thread_count
        self.task_type = str(task_type or "CPU").strip().upper()
        self.devices = str(devices).strip() if devices is not None else None
        self.use_catboost = use_catboost

        self.schema: FeatureSchema | None = None
        self.cat_feature_indices: list[int] = []
        self.model: Any = None
        self.mean_loss: float = 0.0
        self.mean_p_claim: float = 0.05  # rough prior for decomposition

    def _prepare_catboost_input(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.schema is None:
            raise RuntimeError("Model must be fitted before prediction")
        X = prepare_features(df, self.schema).copy()
        for col in self.schema.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        for col in self.schema.categorical_cols:
            X[col] = X[col].astype("string").fillna("missing").astype(str)
        return X

    def fit(self, df: pd.DataFrame) -> "TweedieAggregateLossModel":
        self.schema = infer_feature_schema(df)

        # Target: total claim amount (0 for no-claim)
        y = pd.to_numeric(df[TARGET_AMOUNT_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
        self.mean_loss = float(y.mean()) if len(y) > 0 else 0.0

        # Compute empirical claim rate for decomposition
        y_claim = df[TARGET_CLAIM_COL].fillna(0).astype(int)
        self.mean_p_claim = float(y_claim.mean()) if len(y_claim) > 0 else 0.05

        if self.use_catboost:
            try:
                from catboost import CatBoostRegressor, Pool

                X = self._prepare_catboost_input(df)
                self.cat_feature_indices = [
                    X.columns.get_loc(col)
                    for col in self.schema.categorical_cols
                    if col in X.columns
                ]

                variance_power = float(np.clip(self.tweedie_variance_power, 1.01, 1.99))
                loss_function = f"Tweedie:variance_power={variance_power}"

                self.model = CatBoostRegressor(
                    iterations=self.iterations,
                    learning_rate=self.learning_rate,
                    depth=self.depth,
                    l2_leaf_reg=self.l2_leaf_reg,
                    loss_function=loss_function,
                    random_seed=self.random_state,
                    thread_count=self.thread_count,
                    task_type=self.task_type,
                    devices=self.devices,
                    verbose=False,
                )
                pool = Pool(X, y.values, cat_features=self.cat_feature_indices)
                self.model.fit(pool)

                LOGGER.info(
                    "Tweedie CatBoost fitted: variance_power=%.2f iterations=%d depth=%d",
                    variance_power,
                    self.iterations,
                    self.depth,
                )
                return self
            except ImportError:
                LOGGER.warning("CatBoost not available, falling back to sklearn TweedieRegressor")
                self.use_catboost = False

        # Fallback: sklearn GeneralizedLinearRegressor with Tweedie
        from sklearn.linear_model import TweedieRegressor

        X = prepare_features(df, self.schema)
        numeric_only = X.select_dtypes(include=[np.number]).fillna(0.0)
        if numeric_only.empty:
            numeric_only = pd.DataFrame({"_dummy": np.zeros(len(df))})

        self.model = TweedieRegressor(
            power=self.tweedie_variance_power,
            alpha=self.l2_leaf_reg,
            max_iter=self.iterations,
        )
        self.model.fit(numeric_only.values, y.values)
        self._sklearn_cols = list(numeric_only.columns)

        LOGGER.info(
            "Tweedie sklearn fitted: variance_power=%.2f",
            self.tweedie_variance_power,
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.schema is None or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        if self.use_catboost:
            from catboost import Pool

            X = self._prepare_catboost_input(df)
            pool = Pool(X, cat_features=self.cat_feature_indices)
            expected_loss = np.clip(self.model.predict(pool), 0, None)
        else:
            X = prepare_features(df, self.schema)
            numeric_only = X.select_dtypes(include=[np.number]).fillna(0.0)
            for col in getattr(self, "_sklearn_cols", []):
                if col not in numeric_only.columns:
                    numeric_only[col] = 0.0
            numeric_only = numeric_only[self._sklearn_cols]
            expected_loss = np.clip(self.model.predict(numeric_only.values), 0, None)

        # Decompose expected_loss into p_claim and expected_severity
        # Using: E[loss] = P(claim) * E[severity|claim]
        # We use empirical claim rate as a rough p_claim and back-calculate severity
        mean_severity = self.mean_loss / max(self.mean_p_claim, 1e-9)
        p_claim = np.clip(expected_loss / max(mean_severity, 1e-9), 0, 1)
        expected_severity = np.where(
            p_claim > 1e-9,
            expected_loss / np.clip(p_claim, 1e-9, 1.0),
            mean_severity,
        )

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
    def load(path: Path | str) -> "TweedieAggregateLossModel":
        return load(path)
