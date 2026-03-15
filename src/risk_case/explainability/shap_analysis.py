"""SHAP explainability module.

Generates SHAP-based feature importance analysis for fitted models.
Saves summary plots, bar charts, and top-feature tables as pipeline artifacts.

Usage in pipeline:
    from risk_case.explainability.shap_analysis import generate_shap_report
    generate_shap_report(model, valid_df, output_dir=run_dir / "shap")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from risk_case.settings import ensure_dir

LOGGER = logging.getLogger("risk_case.explainability.shap")


def _get_model_and_data_for_shap(
    model: Any,
    df: pd.DataFrame,
    max_samples: int = 2000,
) -> tuple[Any, pd.DataFrame, str]:
    """Extract the underlying model object and prepare data for SHAP.

    Returns (model_for_shap, X_prepared, model_type).
    """
    # Handle CalibratedFrequencySeverityModel
    if hasattr(model, "base_model"):
        model = model.base_model

    # Handle OOFWeightedBlendModel — pick first base model
    if hasattr(model, "base_models") and isinstance(model.base_models, dict):
        first_key = next(iter(model.base_models))
        model = model.base_models[first_key]

    # Sample for speed
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    # CatBoost models
    if hasattr(model, "frequency_model") and hasattr(model, "_prepare_catboost_input"):
        X = model._prepare_catboost_input(df)
        return model.frequency_model, X, "catboost"

    # PipelineFrequencySeverityModel / FrequencySeverityModel with sklearn Pipeline
    if hasattr(model, "frequency_model") and hasattr(model.frequency_model, "predict_proba"):
        from risk_case.features.builder import prepare_features

        if model.schema is not None:
            X = prepare_features(df, model.schema)
        else:
            X = df
        return model.frequency_model, X, "sklearn_pipeline"

    # WoE model
    if hasattr(model, "_transform_woe"):
        X = model._transform_woe(df).fillna(0.0)
        return model.frequency_model, X, "woe_logreg"

    # Generic fallback
    return model, df, "unknown"


def compute_shap_values(
    model: Any,
    df: pd.DataFrame,
    max_samples: int = 2000,
) -> tuple[np.ndarray | None, list[str], str]:
    """Compute SHAP values for the frequency model.

    Returns (shap_values, feature_names, model_type).
    Returns (None, [], model_type) if SHAP is not available or fails.
    """
    try:
        import shap
    except ImportError:
        LOGGER.warning("shap package is not installed. Install with: pip install shap")
        return None, [], "shap_not_available"

    inner_model, X, model_type = _get_model_and_data_for_shap(model, df, max_samples)

    try:
        if model_type == "catboost":
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(X)
            feature_names = list(X.columns)
        elif model_type == "sklearn_pipeline":
            # For sklearn pipelines, use KernelExplainer with background sample
            background = shap.sample(X, min(100, len(X)))
            predict_fn = lambda x: inner_model.predict_proba(pd.DataFrame(x, columns=X.columns))[:, -1]
            explainer = shap.KernelExplainer(predict_fn, background)
            sample_X = X.iloc[:min(200, len(X))]
            shap_values = explainer.shap_values(sample_X)
            feature_names = list(X.columns)
        elif model_type == "woe_logreg":
            background = shap.sample(X, min(100, len(X)))
            predict_fn = lambda x: inner_model.predict_proba(x)[:, -1]
            explainer = shap.KernelExplainer(predict_fn, background.values)
            sample_X = X.iloc[:min(200, len(X))]
            shap_values = explainer.shap_values(sample_X.values)
            feature_names = list(X.columns)
        else:
            LOGGER.warning("Unsupported model type for SHAP: %s", model_type)
            return None, [], model_type

        return shap_values, feature_names, model_type

    except Exception as exc:
        LOGGER.warning("SHAP computation failed: %s", exc)
        return None, [], model_type


def compute_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Compute mean absolute SHAP values as feature importance."""
    abs_shap = np.abs(shap_values)
    if abs_shap.ndim > 2:
        # Multi-class: take last class (positive)
        abs_shap = abs_shap[-1] if abs_shap.shape[0] == 2 else abs_shap[1]

    mean_abs = np.mean(abs_shap, axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names[:len(mean_abs)],
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df["cumulative_share"] = (
        importance_df["mean_abs_shap"].cumsum() /
        max(importance_df["mean_abs_shap"].sum(), 1e-9)
    )

    return importance_df.head(top_n)


def save_shap_plots(
    shap_values: np.ndarray,
    feature_names: list[str],
    X_sample: pd.DataFrame | np.ndarray,
    output_dir: Path,
) -> dict[str, str]:
    """Save SHAP summary plots to disk. Returns dict of artifact paths."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib or shap not available for plotting")
        return {}

    ensure_dir(output_dir)
    paths: dict[str, str] = {}

    sv = shap_values
    if sv.ndim > 2:
        sv = sv[-1] if sv.shape[0] == 2 else sv[1]

    # Summary dot plot
    try:
        plt.figure(figsize=(12, 8))
        if isinstance(X_sample, pd.DataFrame):
            shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=20)
        else:
            shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=20)
        summary_path = output_dir / "shap_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        paths["summary_plot"] = str(summary_path)
    except Exception as exc:
        LOGGER.warning("SHAP summary plot failed: %s", exc)

    # Bar plot (mean |SHAP|)
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
        bar_path = output_dir / "shap_importance_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        paths["bar_plot"] = str(bar_path)
    except Exception as exc:
        LOGGER.warning("SHAP bar plot failed: %s", exc)

    return paths


def generate_shap_report(
    model: Any,
    df: pd.DataFrame,
    output_dir: Path,
    max_samples: int = 2000,
    top_n: int = 20,
) -> dict[str, Any]:
    """Generate a complete SHAP report: importance table + plots.

    Args:
        model: A fitted model (any candidate type).
        df: Validation DataFrame.
        output_dir: Directory to save SHAP artifacts.
        max_samples: Max samples for SHAP computation.
        top_n: Number of top features to include.

    Returns:
        Dict with report metadata and artifact paths.
    """
    ensure_dir(output_dir)
    report: dict[str, Any] = {
        "status": "not_computed",
        "model_type": "unknown",
        "n_features": 0,
        "artifacts": {},
    }

    shap_values, feature_names, model_type = compute_shap_values(
        model, df, max_samples=max_samples
    )
    report["model_type"] = model_type

    if shap_values is None:
        report["status"] = "failed"
        LOGGER.warning("SHAP report: unable to compute SHAP values")
        return report

    # Feature importance table
    importance_df = compute_feature_importance(shap_values, feature_names, top_n=top_n)
    importance_path = output_dir / "shap_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    report["artifacts"]["importance_csv"] = str(importance_path)
    report["n_features"] = len(importance_df)

    # Prepare X_sample for plots
    _, X_prepared, _ = _get_model_and_data_for_shap(model, df, max_samples=max_samples)
    if isinstance(X_prepared, pd.DataFrame) and len(X_prepared) > max_samples:
        X_prepared = X_prepared.sample(n=max_samples, random_state=42)

    # Save plots
    plot_paths = save_shap_plots(
        shap_values, feature_names, X_prepared, output_dir
    )
    report["artifacts"].update(plot_paths)

    # Top features summary
    if not importance_df.empty:
        top_features = importance_df.head(5)
        report["top_features"] = [
            {"feature": row["feature"], "importance": float(row["mean_abs_shap"])}
            for _, row in top_features.iterrows()
        ]

    report["status"] = "ok"
    LOGGER.info(
        "SHAP report generated: model_type=%s features=%d artifacts=%d",
        model_type,
        report["n_features"],
        len(report["artifacts"]),
    )
    return report
