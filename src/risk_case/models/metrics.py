from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def gini_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0


def classification_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    if len(np.unique(y_true)) < 2:
        metrics["auc"] = None
        metrics["gini"] = None
    else:
        auc = float(roc_auc_score(y_true, p_pred))
        metrics["auc"] = auc
        metrics["gini"] = float(gini_from_auc(auc))

    metrics["brier"] = float(np.mean((y_true - p_pred) ** 2))
    return metrics


def severity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"rmse": None, "mae": None, "r2": None}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # R2 is undefined for fewer than two observations.
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    return {"rmse": rmse, "mae": mae, "r2": r2}
