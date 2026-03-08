from __future__ import annotations

import numpy as np
import pandas as pd

from risk_case.pricing.artifacts import PricingPolicyArtifact
from risk_case.settings import PREMIUM_COL


def apply_pricing_policy(
    df: pd.DataFrame,
    expected_loss: pd.Series,
    alpha: float,
    beta: float = 1.0,
    mean_loss: float | None = None,
) -> pd.Series:
    base_premium = pd.to_numeric(df[PREMIUM_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    expected_loss = pd.to_numeric(expected_loss, errors="coerce").fillna(0.0).clip(lower=0.0)

    resolved_mean_loss = float(expected_loss.mean()) if mean_loss is None and len(expected_loss) else float(mean_loss or 0.0)
    if resolved_mean_loss <= 0:
        risk_index = np.ones(len(expected_loss))
    else:
        risk_index = expected_loss / resolved_mean_loss

    multiplier = 1.0 + alpha * (risk_index - 1.0)
    raw_premium = base_premium * float(beta) * multiplier

    lower_bound = np.zeros(len(base_premium))
    upper_bound = 3.0 * base_premium
    clipped = np.clip(raw_premium, lower_bound, upper_bound)

    return pd.Series(clipped, index=df.index, name="new_premium")


def _assign_bucket_ids(expected_loss: pd.Series, bucket_edges: list[float]) -> np.ndarray:
    values = pd.to_numeric(expected_loss, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    if not bucket_edges:
        return np.zeros(len(values), dtype=int)
    return np.digitize(values, np.asarray(bucket_edges, dtype=float), right=False)


def apply_pricing_policy_artifact(
    df: pd.DataFrame,
    expected_loss: pd.Series,
    pricing_policy: PricingPolicyArtifact,
) -> pd.Series:
    if pricing_policy.kind == "scalar":
        return apply_pricing_policy(
            df=df,
            expected_loss=expected_loss,
            alpha=float(pricing_policy.alpha if pricing_policy.alpha is not None else 0.0),
            beta=float(pricing_policy.beta if pricing_policy.beta is not None else 1.0),
        )

    if pricing_policy.kind != "stratified":
        raise ValueError(f"Unsupported pricing policy kind: {pricing_policy.kind}")

    base_expected_loss = pd.to_numeric(expected_loss, errors="coerce").fillna(0.0).clip(lower=0.0)
    bucket_ids = _assign_bucket_ids(base_expected_loss, pricing_policy.bucket_edges)
    new_premium = pd.Series(np.zeros(len(df), dtype=float), index=df.index, name="new_premium")
    bucket_params_by_id = {
        int(item.get("bucket_id", index)): dict(item)
        for index, item in enumerate(pricing_policy.bucket_params)
    }
    fallback_alpha = float(pricing_policy.global_init_alpha if pricing_policy.global_init_alpha is not None else 0.0)
    fallback_beta = float(pricing_policy.global_init_beta if pricing_policy.global_init_beta is not None else 1.0)

    for bucket_id in np.unique(bucket_ids):
        mask = bucket_ids == int(bucket_id)
        bucket_cfg = bucket_params_by_id.get(int(bucket_id), {})
        alpha = float(bucket_cfg.get("alpha", fallback_alpha))
        beta = float(bucket_cfg.get("beta", fallback_beta))
        mean_loss = bucket_cfg.get("mean_expected_loss")
        bucket_mean_loss = float(mean_loss) if mean_loss is not None else float(base_expected_loss.loc[mask].mean())
        bucket_premium = apply_pricing_policy(
            df=df.loc[mask],
            expected_loss=base_expected_loss.loc[mask],
            alpha=alpha,
            beta=beta,
            mean_loss=bucket_mean_loss,
        )
        new_premium.loc[mask] = bucket_premium.values

    return new_premium
