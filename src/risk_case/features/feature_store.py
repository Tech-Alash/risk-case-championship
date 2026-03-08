from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from risk_case.data.contracts import PreprocessArtifacts
from risk_case.data.policy_aggregation import aggregate_to_policy_level
from risk_case.data.quality_report import build_quality_report
from risk_case.features.preprocessing import (
    FittedPreprocessor,
    PreprocessingConfig,
    fit_preprocessor,
    transform_with_preprocessor,
)
from risk_case.settings import CONTRACT_COL, DEFAULT_TARGET_COLUMNS, ensure_dir

LOGGER = logging.getLogger("risk_case.pipeline.preprocessing")


def build_train_feature_store(
    raw_df: pd.DataFrame,
    config: PreprocessingConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, FittedPreprocessor, PreprocessArtifacts]:
    LOGGER.info("Build train feature store: output_dir=%s", output_dir)
    output_dir = ensure_dir(output_dir)
    policy_df = aggregate_to_policy_level(raw_df, contract_col=config.grain)

    preprocessor = fit_preprocessor(policy_df, config)
    processed_df = transform_with_preprocessor(policy_df, preprocessor)

    dataset_path = output_dir / "train_policy_preprocessed.csv"
    metadata_path = output_dir / "preprocess_metadata.json"
    quality_report_path = output_dir / "quality_report.json"

    processed_df.to_csv(dataset_path, index=False)
    quality_report = build_quality_report(
        raw_df=raw_df,
        policy_df=policy_df,
        processed_df=processed_df,
        feature_columns=preprocessor.feature_columns,
        target_columns=config.target_columns,
        contract_col=config.grain,
    )
    quality_report_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata_payload: dict[str, Any] = {
        "config": {
            "grain": config.grain,
            "target_columns": config.target_columns,
            "drop_columns": config.drop_columns,
            "forbidden_feature_columns": config.forbidden_feature_columns,
            "winsorize": {
                "low": config.winsorize_low,
                "high": config.winsorize_high,
            },
            "missing": {
                "numeric_default": config.numeric_default_strategy,
                "financial_fill": config.financial_fill_value,
                "add_missing_flags": config.add_missing_flags,
                "missing_flag_threshold": config.missing_flag_threshold,
            },
            "categorical": {
                "rare_threshold": config.rare_category_threshold,
            },
            "transforms": {
                "log1p_columns": config.log1p_columns,
            },
        },
        "preprocessor_state": preprocessor.to_dict(),
        "feature_columns": preprocessor.feature_columns,
        "target_columns": config.target_columns,
        "row_count": int(len(processed_df)),
    }
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts = PreprocessArtifacts(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        quality_report_path=quality_report_path,
        feature_columns=preprocessor.feature_columns,
        target_columns=config.target_columns,
        row_count=int(len(processed_df)),
    )
    LOGGER.info(
        "Train feature store ready: rows=%d features=%d",
        len(processed_df),
        len(preprocessor.feature_columns),
    )
    return processed_df, preprocessor, artifacts


def transform_inference_feature_store(
    raw_df: pd.DataFrame,
    preprocessor: FittedPreprocessor,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Transform inference feature store started")
    policy_df = aggregate_to_policy_level(raw_df, contract_col=preprocessor.config.grain)
    processed_df = transform_with_preprocessor(policy_df, preprocessor)

    if output_dir is not None:
        output_dir = ensure_dir(output_dir)
        processed_df.to_csv(output_dir / "inference_policy_preprocessed.csv", index=False)

    LOGGER.info("Transform inference feature store finished: rows=%d", len(processed_df))
    return processed_df, policy_df


def get_target_columns_from_config(config: PreprocessingConfig) -> list[str]:
    if config.target_columns:
        return config.target_columns
    return list(DEFAULT_TARGET_COLUMNS)


def ensure_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing after preprocessing: {missing}")


def policy_to_raw_join(
    raw_df: pd.DataFrame,
    policy_predictions: pd.DataFrame,
    contract_col: str = CONTRACT_COL,
) -> pd.DataFrame:
    if contract_col not in raw_df.columns or contract_col not in policy_predictions.columns:
        return raw_df.copy()
    return raw_df.merge(policy_predictions, on=contract_col, how="left")
