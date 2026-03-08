from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_case.settings import (
    DEFAULT_FORBIDDEN_FEATURE_COLUMNS,
    DEFAULT_TARGET_COLUMNS,
    PREMIUM_COL,
)


BASE_PREFERRED_NUMERIC = [PREMIUM_COL, "driver_count"]


@dataclass
class FeatureSchema:
    numeric_cols: list[str]
    categorical_cols: list[str]

    @property
    def all_cols(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols


def infer_feature_schema(df: pd.DataFrame) -> FeatureSchema:
    forbidden = set(DEFAULT_FORBIDDEN_FEATURE_COLUMNS) | set(DEFAULT_TARGET_COLUMNS)
    numeric_cols = [
        col
        for col in df.columns
        if col not in forbidden and pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical_cols = [
        col
        for col in df.columns
        if col not in forbidden and col not in numeric_cols
    ]

    # Keep preferred columns first for stability across runs.
    prioritized = [col for col in BASE_PREFERRED_NUMERIC if col in numeric_cols]
    numeric_cols = prioritized + [col for col in numeric_cols if col not in prioritized]

    if not numeric_cols and not categorical_cols:
        # Safety fallback so preprocessing pipelines always receive at least one column.
        numeric_cols = [PREMIUM_COL] if PREMIUM_COL in df.columns else []
        if not numeric_cols:
            df["_dummy_feature"] = 0.0
            numeric_cols = ["_dummy_feature"]

    return FeatureSchema(numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def prepare_features(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    missing_numeric = [col for col in schema.numeric_cols if col not in df.columns]
    missing_cat = [col for col in schema.categorical_cols if col not in df.columns]

    base = df.copy()

    for col in missing_numeric:
        base[col] = np.nan
    for col in missing_cat:
        base[col] = "missing"

    for col in schema.numeric_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    for col in schema.categorical_cols:
        base[col] = base[col].astype("string").fillna("missing")

    return base[schema.all_cols]
