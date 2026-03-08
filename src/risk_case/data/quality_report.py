from __future__ import annotations

from typing import Any

import pandas as pd

from risk_case.settings import CONTRACT_COL


def build_quality_report(
    raw_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    contract_col: str = CONTRACT_COL,
) -> dict[str, Any]:
    raw_null_rates = raw_df.isna().mean().sort_values(ascending=False).head(20).to_dict()
    policy_null_rates = policy_df.isna().mean().sort_values(ascending=False).head(20).to_dict()
    processed_null_rates = processed_df.isna().mean().sort_values(ascending=False).head(20).to_dict()

    duplicate_contracts = 0
    if contract_col in processed_df.columns:
        duplicate_contracts = int(processed_df[contract_col].duplicated().sum())

    return {
        "raw_shape": [int(raw_df.shape[0]), int(raw_df.shape[1])],
        "policy_shape": [int(policy_df.shape[0]), int(policy_df.shape[1])],
        "processed_shape": [int(processed_df.shape[0]), int(processed_df.shape[1])],
        "duplicate_contracts_processed": duplicate_contracts,
        "feature_count": len(feature_columns),
        "target_columns": target_columns,
        "raw_top_null_rates": raw_null_rates,
        "policy_top_null_rates": policy_null_rates,
        "processed_top_null_rates": processed_null_rates,
    }

