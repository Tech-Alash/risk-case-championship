from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from risk_case.settings import (
    CONTRACT_COL,
    PREMIUM_COL,
    PREMIUM_NET_COL,
    TARGET_AMOUNT_COL,
    TARGET_CLAIM_COL,
    TARGET_COUNT_COL,
)

LOGGER = logging.getLogger("risk_case.pipeline.preprocessing")

IDENTIFIER_DROP_COLUMNS = {
    "unique_id",
    "driver_iin",
    "insurer_iin",
    "car_number",
}

TARGET_AND_FINANCIAL_COLUMNS = {
    PREMIUM_COL,
    PREMIUM_NET_COL,
    TARGET_CLAIM_COL,
    TARGET_COUNT_COL,
    TARGET_AMOUNT_COL,
}
def aggregate_to_policy_level(
    df: pd.DataFrame,
    contract_col: str = CONTRACT_COL,
) -> pd.DataFrame:
    LOGGER.info("Policy aggregation started: rows=%d cols=%d", len(df), len(df.columns))
    if contract_col not in df.columns:
        LOGGER.warning("Contract column %s is missing; skipping policy aggregation", contract_col)
        return df.copy()

    grouped = df.groupby(contract_col, dropna=False)
    aggregation_map: dict[str, Any] = {}
    rename_map: dict[str, str] = {}

    for column in df.columns:
        if column == contract_col:
            continue
        if column in IDENTIFIER_DROP_COLUMNS:
            continue

        series = df[column]
        if column in TARGET_AND_FINANCIAL_COLUMNS:
            aggregation_map[column] = "max"
            continue

        if np.issubdtype(series.dtype, np.number):
            if column.startswith("SCORE_"):
                aggregation_map[column] = "mean"
            else:
                aggregation_map[column] = "mean"
                rename_map[column] = f"{column}_mean"
        else:
            aggregation_map[column] = "first"

    agg_df = grouped.agg(aggregation_map).reset_index()
    if rename_map:
        agg_df = agg_df.rename(columns=rename_map)

    driver_count = grouped.size().reset_index(name="driver_count")
    policy_df = agg_df.merge(driver_count, on=contract_col, how="left")

    if TARGET_CLAIM_COL in policy_df.columns:
        policy_df[TARGET_CLAIM_COL] = pd.to_numeric(policy_df[TARGET_CLAIM_COL], errors="coerce").fillna(0).astype(int)
    if TARGET_COUNT_COL in policy_df.columns:
        policy_df[TARGET_COUNT_COL] = pd.to_numeric(policy_df[TARGET_COUNT_COL], errors="coerce").fillna(0.0)
    if TARGET_AMOUNT_COL in policy_df.columns:
        policy_df[TARGET_AMOUNT_COL] = pd.to_numeric(policy_df[TARGET_AMOUNT_COL], errors="coerce").fillna(0.0)

    LOGGER.info("Policy aggregation finished: policy_rows=%d cols=%d", len(policy_df), len(policy_df.columns))
    return policy_df
