"""Bootstrap Confidence Intervals for key metrics.

Provides bootstrap-based 95% CIs for AUC, Gini, Brier score, LR, and
policy_score.  This gives the jury statistical evidence of model stability.

Usage:
    from risk_case.models.bootstrap_ci import compute_bootstrap_ci
    ci = compute_bootstrap_ci(y_true, p_pred, premiums, claims, n_bootstrap=1000)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger("risk_case.models.bootstrap_ci")


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_pred))


def _gini(auc: float | None) -> float | None:
    return (2.0 * auc - 1.0) if auc is not None else None


def _brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _loss_ratio(claims: np.ndarray, premiums: np.ndarray) -> float:
    total_premiums = float(np.sum(premiums))
    if total_premiums <= 0:
        return float("inf")
    return float(np.sum(claims)) / total_premiums


def compute_bootstrap_ci(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    premiums: np.ndarray | None = None,
    claims: np.ndarray | None = None,
    new_premiums: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute bootstrap 95% confidence intervals for key metrics.

    Args:
        y_true: Binary claim indicator (0/1).
        p_pred: Predicted probabilities of claim.
        premiums: Original premiums (for LR computation).
        claims: Claim amounts (for LR computation).
        new_premiums: New premiums after pricing (for new LR computation).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95).
        random_state: Random seed for reproducibility.

    Returns:
        Dict with point estimates and CI bounds for each metric.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    auc_samples: list[float] = []
    gini_samples: list[float] = []
    brier_samples: list[float] = []
    lr_original_samples: list[float] = []
    lr_new_samples: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = y_true[idx]
        p_boot = p_pred[idx]

        auc = _safe_auc(y_boot, p_boot)
        if auc is not None:
            auc_samples.append(auc)
            gini_val = _gini(auc)
            if gini_val is not None:
                gini_samples.append(gini_val)

        brier_samples.append(_brier(y_boot, p_boot))

        if premiums is not None and claims is not None:
            lr_original_samples.append(_loss_ratio(claims[idx], premiums[idx]))

        if new_premiums is not None and claims is not None:
            lr_new_samples.append(_loss_ratio(claims[idx], new_premiums[idx]))

    alpha = (1.0 - confidence) / 2.0

    def _ci(samples: list[float]) -> dict[str, float | None]:
        if not samples:
            return {"point": None, "lower": None, "upper": None, "std": None}
        arr = np.array(samples)
        return {
            "point": float(np.mean(arr)),
            "lower": float(np.percentile(arr, 100 * alpha)),
            "upper": float(np.percentile(arr, 100 * (1 - alpha))),
            "std": float(np.std(arr)),
        }

    result: dict[str, Any] = {
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "n_samples": n,
        "auc": _ci(auc_samples),
        "gini": _ci(gini_samples),
        "brier": _ci(brier_samples),
    }

    if lr_original_samples:
        result["lr_original"] = _ci(lr_original_samples)
    if lr_new_samples:
        result["lr_new_premium"] = _ci(lr_new_samples)

    # Compute point estimates
    point_auc = _safe_auc(y_true, p_pred)
    result["point_estimates"] = {
        "auc": point_auc,
        "gini": _gini(point_auc),
        "brier": _brier(y_true, p_pred),
    }

    LOGGER.info(
        "Bootstrap CI (n=%d): AUC=[%.4f, %.4f] Gini=[%.4f, %.4f]",
        n_bootstrap,
        result["auc"].get("lower", 0) or 0,
        result["auc"].get("upper", 0) or 0,
        result["gini"].get("lower", 0) or 0,
        result["gini"].get("upper", 0) or 0,
    )

    return result


def bootstrap_ci_dataframe(ci_result: dict[str, Any]) -> pd.DataFrame:
    """Convert CI results to a clean DataFrame for reporting."""
    rows = []
    for metric_name in ["auc", "gini", "brier", "lr_original", "lr_new_premium"]:
        if metric_name in ci_result and isinstance(ci_result[metric_name], dict):
            ci = ci_result[metric_name]
            rows.append({
                "metric": metric_name,
                "point_estimate": ci.get("point"),
                "ci_lower": ci.get("lower"),
                "ci_upper": ci.get("upper"),
                "std": ci.get("std"),
            })
    return pd.DataFrame(rows)
