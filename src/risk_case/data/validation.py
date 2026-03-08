from __future__ import annotations

from typing import Iterable

import pandas as pd

from risk_case.data.contracts import RawValidationResult
from risk_case.settings import (
    PREMIUM_COL,
    PREMIUM_NET_COL,
    TARGET_AMOUNT_COL,
    TARGET_CLAIM_COL,
    TARGET_COUNT_COL,
    UNIQUE_ID_COL,
)


REQUIRED_COLUMNS = (
    PREMIUM_COL,
    PREMIUM_NET_COL,
    TARGET_CLAIM_COL,
    TARGET_AMOUNT_COL,
    TARGET_COUNT_COL,
)


def _missing_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return [column for column in required if column not in df.columns]


def validate_dataset(df: pd.DataFrame) -> RawValidationResult:
    result = RawValidationResult()

    missing = _missing_columns(df, REQUIRED_COLUMNS)
    if missing:
        result.errors.append(f"Missing required columns: {missing}")
        return result

    strict_null_columns = {PREMIUM_COL, PREMIUM_NET_COL, TARGET_CLAIM_COL}
    tolerant_claim_columns = {TARGET_AMOUNT_COL, TARGET_COUNT_COL}

    null_rate = df[list(REQUIRED_COLUMNS)].isna().mean().to_dict()
    for col, rate in null_rate.items():
        if col in strict_null_columns:
            if rate > 0.4:
                result.errors.append(f"High null-rate for required column {col}: {rate:.2%}")
            elif rate > 0.1:
                result.warnings.append(f"Moderate null-rate for required column {col}: {rate:.2%}")
        elif col in tolerant_claim_columns:
            if rate > 0.99:
                result.warnings.append(
                    f"Very high null-rate for claim column {col}: {rate:.2%}. "
                    "Expected for sparse claims but verify ingestion."
                )

    if TARGET_CLAIM_COL in df.columns:
        claim_values = set(df[TARGET_CLAIM_COL].dropna().unique().tolist())
        bad_values = claim_values - {0, 1}
        if bad_values:
            result.errors.append(f"{TARGET_CLAIM_COL} contains non-binary values: {sorted(bad_values)}")

    for col in [PREMIUM_COL, PREMIUM_NET_COL]:
        if (df[col].fillna(0) < 0).any():
            result.errors.append(f"Negative values detected in {col}")

    if TARGET_COUNT_COL in df.columns and (df[TARGET_COUNT_COL].fillna(0) < 0).any():
        result.errors.append(f"Negative values detected in {TARGET_COUNT_COL}")

    if TARGET_AMOUNT_COL in df.columns and (df[TARGET_AMOUNT_COL].fillna(0) < 0).any():
        result.errors.append(f"Negative values detected in {TARGET_AMOUNT_COL}")

    if UNIQUE_ID_COL in df.columns:
        dup_count = int(df[UNIQUE_ID_COL].duplicated().sum())
        if dup_count > 0:
            result.warnings.append(f"Duplicated {UNIQUE_ID_COL}: {dup_count}")

    return result
