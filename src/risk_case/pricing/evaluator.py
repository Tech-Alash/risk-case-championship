from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from risk_case.pricing.artifacts import PricingPolicyArtifact
from risk_case.pricing.policy import apply_pricing_policy, apply_pricing_policy_artifact
from risk_case.settings import PREMIUM_COL, PREMIUM_NET_COL, TARGET_AMOUNT_COL


@dataclass
class RetentionConfig:
    enabled: bool = False
    base_retention: float = 0.90
    elasticity: float = 4.0
    center: float = 0.0
    floor: float = 0.05
    cap: float = 0.995

    @staticmethod
    def from_dict(raw: dict[str, Any] | None) -> "RetentionConfig":
        data = raw or {}
        floor = float(data.get("floor", 0.05))
        cap = float(data.get("cap", 0.995))
        if floor > cap:
            floor, cap = cap, floor
        return RetentionConfig(
            enabled=bool(data.get("enabled", False)),
            base_retention=float(data.get("base_retention", 0.90)),
            elasticity=float(data.get("elasticity", 4.0)),
            center=float(data.get("center", 0.0)),
            floor=float(max(0.0, min(1.0, floor))),
            cap=float(max(0.0, min(1.0, cap))),
        )

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "base_retention": self.base_retention,
            "elasticity": self.elasticity,
            "center": self.center,
            "floor": self.floor,
            "cap": self.cap,
        }


@dataclass
class StratifiedPricingConfig:
    enabled: bool = False
    n_buckets: int = 5
    bucket_on: str = "expected_loss"
    coordinate_passes: int = 2
    min_bucket_size: int = 50
    enforce_monotonic: bool = True

    @staticmethod
    def from_dict(raw: dict[str, Any] | None) -> "StratifiedPricingConfig":
        data = raw or {}
        return StratifiedPricingConfig(
            enabled=bool(data.get("enabled", False)),
            n_buckets=max(1, int(data.get("n_buckets", 5))),
            bucket_on=str(data.get("bucket_on", "expected_loss")),
            coordinate_passes=max(1, int(data.get("coordinate_passes", 2))),
            min_bucket_size=max(1, int(data.get("min_bucket_size", 50))),
            enforce_monotonic=bool(data.get("enforce_monotonic", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "n_buckets": self.n_buckets,
            "bucket_on": self.bucket_on,
            "coordinate_passes": self.coordinate_passes,
            "min_bucket_size": self.min_bucket_size,
            "enforce_monotonic": self.enforce_monotonic,
        }


@dataclass
class PricingEvaluation:
    lr_total: float
    lr_group1: float
    lr_group2: float
    share_group1: float
    violations: int
    score: float
    in_target: bool
    distance_to_target: float
    target_band: tuple[float, float]
    retention_rate: float | None = None
    retention_enabled: bool = False
    pricing_policy: PricingPolicyArtifact | None = None

    def to_dict(self) -> dict[str, float | int | bool | dict[str, float] | None]:
        payload: dict[str, float | int | bool | dict[str, float] | None] = {
            "lr_total": self.lr_total,
            "lr_group1": self.lr_group1,
            "lr_group2": self.lr_group2,
            "share_group1": self.share_group1,
            "violations": self.violations,
            "policy_score": self.score,
            "in_target": self.in_target,
            "distance_to_target": self.distance_to_target,
            "target_band": {
                "min": self.target_band[0],
                "max": self.target_band[1],
            },
            "retention_enabled": self.retention_enabled,
            "retention_rate": self.retention_rate,
        }
        if self.pricing_policy is not None:
            payload["pricing_policy_kind"] = self.pricing_policy.kind
            payload["pricing_policy"] = self.pricing_policy.to_summary()
        return payload


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _resolve_target_band(target_lr: float, target_band: tuple[float, float] | None) -> tuple[float, float]:
    if target_band is None:
        return float(target_lr - 0.01), float(target_lr + 0.01)
    band_min, band_max = float(target_band[0]), float(target_band[1])
    if band_min > band_max:
        band_min, band_max = band_max, band_min
    return band_min, band_max


def estimate_retention_probabilities(
    base_premium: pd.Series,
    new_premium: pd.Series,
    retention_config: RetentionConfig | None = None,
) -> pd.Series:
    config = retention_config or RetentionConfig()
    if not config.enabled:
        return pd.Series(np.ones(len(base_premium), dtype=float), index=base_premium.index)

    base = pd.to_numeric(base_premium, errors="coerce").fillna(0.0).clip(lower=0.0)
    repriced = pd.to_numeric(new_premium, errors="coerce").fillna(0.0).clip(lower=0.0)
    price_delta = np.where(base > 0, repriced / base - 1.0, 0.0)

    base_retention = float(np.clip(config.base_retention, 1e-6, 1.0 - 1e-6))
    logit_base = float(np.log(base_retention / (1.0 - base_retention)))
    logits = logit_base - float(config.elasticity) * (price_delta - float(config.center))
    logits = np.clip(logits, -35.0, 35.0)
    probs = 1.0 / (1.0 + np.exp(-logits))

    floor = float(np.clip(config.floor, 0.0, 1.0))
    cap = float(np.clip(config.cap, 0.0, 1.0))
    if floor > cap:
        floor, cap = cap, floor
    probs = np.clip(probs, floor, cap)
    return pd.Series(probs, index=base_premium.index)


def evaluate_pricing(
    df: pd.DataFrame,
    new_premium: pd.Series,
    target_lr: float,
    target_band: tuple[float, float] | None = None,
    retention_config: RetentionConfig | None = None,
) -> PricingEvaluation:
    base_premium = pd.to_numeric(df[PREMIUM_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    premium_wo_term = pd.to_numeric(df[PREMIUM_NET_COL], errors="coerce").fillna(base_premium)
    payout = pd.to_numeric(df[TARGET_AMOUNT_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    band_min, band_max = _resolve_target_band(target_lr=target_lr, target_band=target_band)
    retention = estimate_retention_probabilities(base_premium, new_premium, retention_config=retention_config)
    retention_enabled = bool(retention_config.enabled) if retention_config is not None else False

    term_ratio = np.where(base_premium > 0, premium_wo_term / base_premium, 1.0)
    term_ratio = np.nan_to_num(term_ratio, nan=1.0, posinf=1.0, neginf=1.0).clip(min=0.0, max=1.0)

    new_premium_net = pd.Series(new_premium.values * term_ratio, index=df.index)
    premium_weight = retention if retention_enabled else pd.Series(np.ones(len(df), dtype=float), index=df.index)
    payout_weight = retention if retention_enabled else pd.Series(np.ones(len(df), dtype=float), index=df.index)

    group1_mask = new_premium <= base_premium
    group2_mask = ~group1_mask

    payout_total = float((payout * payout_weight).sum())
    den_total = float((new_premium_net * premium_weight).sum())
    lr_total = _safe_ratio(payout_total, den_total)

    payout_g1 = float((payout[group1_mask] * payout_weight[group1_mask]).sum())
    den_g1 = float((new_premium_net[group1_mask] * premium_weight[group1_mask]).sum())
    lr_group1 = _safe_ratio(payout_g1, den_g1)

    payout_g2 = float((payout[group2_mask] * payout_weight[group2_mask]).sum())
    den_g2 = float((new_premium_net[group2_mask] * premium_weight[group2_mask]).sum())
    lr_group2 = _safe_ratio(payout_g2, den_g2)

    if retention_enabled:
        share_group1 = _safe_ratio(float(premium_weight[group1_mask].sum()), float(premium_weight.sum()))
    else:
        share_group1 = float(group1_mask.mean())

    lower_violations = (new_premium < 0).sum()
    upper_violations = (new_premium > 3.0 * base_premium).sum()
    violations = int(lower_violations + upper_violations)

    distance_to_target = abs(lr_total - target_lr)
    in_target = band_min <= lr_total <= band_max
    score = (
        -distance_to_target
        -0.7 * abs(lr_group1 - target_lr)
        -0.7 * abs(lr_group2 - target_lr)
        +0.15 * share_group1
        -2.0 * violations
    )
    if in_target:
        score += 0.5
    retention_rate = float(retention.mean()) if retention_enabled else None
    if retention_rate is not None:
        score += 0.10 * retention_rate

    return PricingEvaluation(
        lr_total=lr_total,
        lr_group1=lr_group1,
        lr_group2=lr_group2,
        share_group1=share_group1,
        violations=violations,
        score=score,
        in_target=in_target,
        distance_to_target=float(distance_to_target),
        target_band=(float(band_min), float(band_max)),
        retention_rate=retention_rate,
        retention_enabled=retention_enabled,
    )


def _candidate_key(pricing_eval: PricingEvaluation) -> tuple[int, float, float]:
    return (
        1 if pricing_eval.in_target else 0,
        float(pricing_eval.score),
        float(pricing_eval.share_group1),
    )


def _attach_policy(
    pricing_eval: PricingEvaluation,
    pricing_policy: PricingPolicyArtifact,
) -> PricingEvaluation:
    pricing_eval.pricing_policy = pricing_policy
    return pricing_eval


def _select_best_pricing_grid(
    df: pd.DataFrame,
    expected_loss: pd.Series,
    target_lr: float,
    alpha_grid: np.ndarray,
    beta_values: np.ndarray,
    target_band: tuple[float, float] | None,
    retention_config: RetentionConfig | None,
) -> tuple[float, float, pd.Series, PricingEvaluation]:
    best_alpha = float(alpha_grid[0])
    best_beta = float(beta_values[0])
    best_premium = None
    best_eval = None

    for beta in beta_values:
        for alpha in alpha_grid:
            premium = apply_pricing_policy(df=df, expected_loss=expected_loss, alpha=float(alpha), beta=float(beta))
            current_eval = _attach_policy(
                evaluate_pricing(
                    df=df,
                    new_premium=premium,
                    target_lr=target_lr,
                    target_band=target_band,
                    retention_config=retention_config,
                ),
                PricingPolicyArtifact.scalar(alpha=float(alpha), beta=float(beta), method="grid"),
            )

            if best_eval is None or _candidate_key(current_eval) > _candidate_key(best_eval):
                best_alpha = float(alpha)
                best_beta = float(beta)
                best_premium = premium
                best_eval = current_eval

    assert best_premium is not None
    assert best_eval is not None
    return best_alpha, best_beta, best_premium, best_eval


def _evaluate_candidate(
    df: pd.DataFrame,
    expected_loss: pd.Series,
    target_lr: float,
    alpha: float,
    beta: float,
    target_band: tuple[float, float] | None,
    retention_config: RetentionConfig | None,
) -> tuple[pd.Series, PricingEvaluation]:
    premium = apply_pricing_policy(df=df, expected_loss=expected_loss, alpha=float(alpha), beta=float(beta))
    eval_result = _attach_policy(
        evaluate_pricing(
            df=df,
            new_premium=premium,
            target_lr=target_lr,
            target_band=target_band,
            retention_config=retention_config,
        ),
        PricingPolicyArtifact.scalar(alpha=float(alpha), beta=float(beta), method="grid"),
    )
    return premium, eval_result


def _select_best_pricing_slsqp(
    df: pd.DataFrame,
    expected_loss: pd.Series,
    target_lr: float,
    alpha_grid: np.ndarray,
    beta_values: np.ndarray,
    target_band: tuple[float, float] | None,
    retention_config: RetentionConfig | None,
    slsqp_options: dict[str, Any] | None,
) -> tuple[float, float, pd.Series, PricingEvaluation]:
    grid_alpha, grid_beta, grid_premium, grid_eval = _select_best_pricing_grid(
        df=df,
        expected_loss=expected_loss,
        target_lr=target_lr,
        alpha_grid=alpha_grid,
        beta_values=beta_values,
        target_band=target_band,
        retention_config=retention_config,
    )

    try:
        from scipy.optimize import minimize
    except Exception:
        return grid_alpha, grid_beta, grid_premium, grid_eval

    alpha_bounds = (float(np.min(alpha_grid)), float(np.max(alpha_grid)))
    beta_bounds = (float(np.min(beta_values)), float(np.max(beta_values)))
    if alpha_bounds[0] == alpha_bounds[1] and beta_bounds[0] == beta_bounds[1]:
        return grid_alpha, grid_beta, grid_premium, grid_eval

    x0 = np.asarray([grid_alpha, grid_beta], dtype=float)

    def objective(x: np.ndarray) -> float:
        alpha = float(np.clip(x[0], alpha_bounds[0], alpha_bounds[1]))
        beta = float(np.clip(x[1], beta_bounds[0], beta_bounds[1]))
        _, eval_result = _evaluate_candidate(
            df=df,
            expected_loss=expected_loss,
            target_lr=target_lr,
            alpha=alpha,
            beta=beta,
            target_band=target_band,
            retention_config=retention_config,
        )
        return -float(eval_result.score)

    cfg = slsqp_options or {}
    options: dict[str, Any] = {
        "maxiter": int(cfg.get("maxiter", 200)),
        "ftol": float(cfg.get("ftol", 1e-6)),
        "eps": float(cfg.get("eps", 1e-3)),
        "disp": False,
    }

    try:
        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=[alpha_bounds, beta_bounds],
            options=options,
        )
    except Exception:
        return grid_alpha, grid_beta, grid_premium, grid_eval

    if not result.success or result.x is None or np.any(~np.isfinite(result.x)):
        return grid_alpha, grid_beta, grid_premium, grid_eval

    alpha_opt = float(np.clip(result.x[0], alpha_bounds[0], alpha_bounds[1]))
    beta_opt = float(np.clip(result.x[1], beta_bounds[0], beta_bounds[1]))
    premium_opt, eval_opt = _evaluate_candidate(
        df=df,
        expected_loss=expected_loss,
        target_lr=target_lr,
        alpha=alpha_opt,
        beta=beta_opt,
        target_band=target_band,
        retention_config=retention_config,
    )

    if _candidate_key(eval_opt) > _candidate_key(grid_eval):
        return alpha_opt, beta_opt, premium_opt, _attach_policy(
            eval_opt,
            PricingPolicyArtifact.scalar(alpha=alpha_opt, beta=beta_opt, method="slsqp"),
        )
    return grid_alpha, grid_beta, grid_premium, grid_eval


def _derive_stratified_edges(
    expected_loss: pd.Series,
    n_buckets: int,
    min_bucket_size: int,
) -> list[float]:
    values = pd.to_numeric(expected_loss, errors="coerce").fillna(0.0).clip(lower=0.0)
    if len(values) < max(2, min_bucket_size * 2):
        return []

    for bucket_count in range(max(1, int(n_buckets)), 0, -1):
        quantiles = np.linspace(0.0, 1.0, bucket_count + 1)
        raw_edges = values.quantile(quantiles).to_numpy(dtype=float)
        if len(raw_edges) < 2:
            continue
        internal_edges = np.unique(raw_edges[1:-1])
        bucket_ids = np.digitize(values.to_numpy(dtype=float), internal_edges, right=False)
        counts = pd.Series(bucket_ids).value_counts().sort_index()
        if counts.empty:
            continue
        if counts.min() >= int(min_bucket_size):
            return [float(item) for item in internal_edges.tolist()]
    return []


def _build_stratified_policy(
    expected_loss: pd.Series,
    bucket_edges: list[float],
    bucket_alpha: dict[int, float],
    bucket_beta: dict[int, float],
    method: str,
    global_init_alpha: float,
    global_init_beta: float,
) -> PricingPolicyArtifact:
    values = pd.to_numeric(expected_loss, errors="coerce").fillna(0.0).clip(lower=0.0)
    bucket_ids = np.digitize(values.to_numpy(dtype=float), np.asarray(bucket_edges, dtype=float), right=False)
    global_mean_loss = float(values.mean()) if len(values) else 0.0
    bucket_params: list[dict[str, Any]] = []
    for bucket_id in sorted(np.unique(bucket_ids).tolist()):
        mask = bucket_ids == int(bucket_id)
        bucket_mean_loss = float(values.loc[mask].mean()) if mask.any() else 0.0
        bucket_params.append(
            {
                "bucket_id": int(bucket_id),
                "alpha": float(bucket_alpha.get(int(bucket_id), global_init_alpha)),
                "beta": float(bucket_beta.get(int(bucket_id), global_init_beta)),
                "mean_expected_loss": global_mean_loss,
                "bucket_expected_loss_mean": bucket_mean_loss,
                "count": int(mask.sum()),
            }
        )
    return PricingPolicyArtifact(
        kind="stratified",
        method=method,
        alpha=float(global_init_alpha),
        beta=float(global_init_beta),
        bucket_edges=[float(item) for item in bucket_edges],
        bucket_params=bucket_params,
        global_init_alpha=float(global_init_alpha),
        global_init_beta=float(global_init_beta),
        bucket_feature="expected_loss",
    )


def _policy_monotonic_ok(
    df: pd.DataFrame,
    expected_loss: pd.Series,
    pricing_policy: PricingPolicyArtifact,
) -> bool:
    if pricing_policy.kind != "stratified":
        return True
    if len(pricing_policy.bucket_params) <= 1:
        return True

    base = pd.to_numeric(df[PREMIUM_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    premium = apply_pricing_policy_artifact(df=df, expected_loss=expected_loss, pricing_policy=pricing_policy)
    multiplier = np.where(base > 0, premium / base, 0.0)
    bucket_ids = np.digitize(
        pd.to_numeric(expected_loss, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float),
        np.asarray(pricing_policy.bucket_edges, dtype=float),
        right=False,
    )
    bucket_means = pd.Series(multiplier).groupby(bucket_ids).mean()
    if bucket_means.empty:
        return True
    diffs = np.diff(bucket_means.to_numpy(dtype=float))
    return bool(np.all(diffs >= -1e-9))


def _evaluate_policy_candidate(
    df: pd.DataFrame,
    expected_loss: pd.Series,
    target_lr: float,
    target_band: tuple[float, float] | None,
    retention_config: RetentionConfig | None,
    pricing_policy: PricingPolicyArtifact,
) -> tuple[pd.Series, PricingEvaluation]:
    premium = apply_pricing_policy_artifact(df=df, expected_loss=expected_loss, pricing_policy=pricing_policy)
    pricing_eval = _attach_policy(
        evaluate_pricing(
            df=df,
            new_premium=premium,
            target_lr=target_lr,
            target_band=target_band,
            retention_config=retention_config,
        ),
        pricing_policy,
    )
    return premium, pricing_eval


def _select_best_pricing_stratified_grid(
    df: pd.DataFrame,
    expected_loss: pd.Series,
    target_lr: float,
    alpha_grid: np.ndarray,
    beta_values: np.ndarray,
    target_band: tuple[float, float] | None,
    retention_config: RetentionConfig | None,
    stratified_config: StratifiedPricingConfig | None,
) -> tuple[float, float, pd.Series, PricingEvaluation]:
    cfg = stratified_config or StratifiedPricingConfig(enabled=True)
    global_alpha, global_beta, global_premium, global_eval = _select_best_pricing_grid(
        df=df,
        expected_loss=expected_loss,
        target_lr=target_lr,
        alpha_grid=alpha_grid,
        beta_values=beta_values,
        target_band=target_band,
        retention_config=retention_config,
    )

    bucket_edges = _derive_stratified_edges(
        expected_loss=expected_loss,
        n_buckets=cfg.n_buckets,
        min_bucket_size=cfg.min_bucket_size,
    )
    if not bucket_edges:
        return global_alpha, global_beta, global_premium, _attach_policy(
            global_eval,
            PricingPolicyArtifact.scalar(alpha=global_alpha, beta=global_beta, method="stratified_grid_fallback"),
        )

    bucket_ids = np.digitize(
        pd.to_numeric(expected_loss, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float),
        np.asarray(bucket_edges, dtype=float),
        right=False,
    )
    unique_bucket_ids = sorted(pd.Series(bucket_ids).unique().tolist())
    bucket_alpha = {int(bucket_id): float(global_alpha) for bucket_id in unique_bucket_ids}
    bucket_beta = {int(bucket_id): float(global_beta) for bucket_id in unique_bucket_ids}
    current_policy = _build_stratified_policy(
        expected_loss=expected_loss,
        bucket_edges=bucket_edges,
        bucket_alpha=bucket_alpha,
        bucket_beta=bucket_beta,
        method="stratified_grid",
        global_init_alpha=float(global_alpha),
        global_init_beta=float(global_beta),
    )
    current_premium, current_eval = _evaluate_policy_candidate(
        df=df,
        expected_loss=expected_loss,
        target_lr=target_lr,
        target_band=target_band,
        retention_config=retention_config,
        pricing_policy=current_policy,
    )
    if cfg.enforce_monotonic and not _policy_monotonic_ok(df=df, expected_loss=expected_loss, pricing_policy=current_policy):
        return global_alpha, global_beta, global_premium, global_eval

    for _ in range(int(cfg.coordinate_passes)):
        for bucket_id in unique_bucket_ids:
            best_local_key = _candidate_key(current_eval)
            best_local_state: tuple[dict[int, float], dict[int, float], pd.Series, PricingEvaluation] | None = None
            for beta in beta_values:
                for alpha in alpha_grid:
                    cand_alpha = dict(bucket_alpha)
                    cand_beta = dict(bucket_beta)
                    cand_alpha[int(bucket_id)] = float(alpha)
                    cand_beta[int(bucket_id)] = float(beta)
                    candidate_policy = _build_stratified_policy(
                        expected_loss=expected_loss,
                        bucket_edges=bucket_edges,
                        bucket_alpha=cand_alpha,
                        bucket_beta=cand_beta,
                        method="stratified_grid",
                        global_init_alpha=float(global_alpha),
                        global_init_beta=float(global_beta),
                    )
                    if cfg.enforce_monotonic and not _policy_monotonic_ok(
                        df=df,
                        expected_loss=expected_loss,
                        pricing_policy=candidate_policy,
                    ):
                        continue
                    premium, eval_result = _evaluate_policy_candidate(
                        df=df,
                        expected_loss=expected_loss,
                        target_lr=target_lr,
                        target_band=target_band,
                        retention_config=retention_config,
                        pricing_policy=candidate_policy,
                    )
                    candidate_key = _candidate_key(eval_result)
                    if candidate_key > best_local_key:
                        best_local_key = candidate_key
                        best_local_state = (cand_alpha, cand_beta, premium, eval_result)

            if best_local_state is not None:
                bucket_alpha, bucket_beta, current_premium, current_eval = best_local_state

    final_policy = _build_stratified_policy(
        expected_loss=expected_loss,
        bucket_edges=bucket_edges,
        bucket_alpha=bucket_alpha,
        bucket_beta=bucket_beta,
        method="stratified_grid",
        global_init_alpha=float(global_alpha),
        global_init_beta=float(global_beta),
    )
    final_premium, final_eval = _evaluate_policy_candidate(
        df=df,
        expected_loss=expected_loss,
        target_lr=target_lr,
        target_band=target_band,
        retention_config=retention_config,
        pricing_policy=final_policy,
    )
    if cfg.enforce_monotonic and not _policy_monotonic_ok(df=df, expected_loss=expected_loss, pricing_policy=final_policy):
        return global_alpha, global_beta, global_premium, global_eval
    if _candidate_key(final_eval) < _candidate_key(global_eval):
        return global_alpha, global_beta, global_premium, global_eval
    return float(global_alpha), float(global_beta), final_premium, final_eval


def select_best_pricing(
    df: pd.DataFrame,
    expected_loss: pd.Series,
    target_lr: float,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray | None = None,
    target_band: tuple[float, float] | None = None,
    method: str = "grid",
    retention_config: RetentionConfig | None = None,
    slsqp_options: dict[str, Any] | None = None,
    stratified_config: StratifiedPricingConfig | None = None,
) -> tuple[float, float, pd.Series, PricingEvaluation]:
    if len(alpha_grid) == 0:
        raise ValueError("alpha_grid must contain at least one value")
    beta_values = np.asarray(beta_grid, dtype=float) if beta_grid is not None else np.asarray([1.0], dtype=float)
    if len(beta_values) == 0:
        raise ValueError("beta_grid must contain at least one value")

    method_norm = (method or "grid").strip().lower()
    if method_norm == "stratified_grid":
        return _select_best_pricing_stratified_grid(
            df=df,
            expected_loss=expected_loss,
            target_lr=target_lr,
            alpha_grid=np.asarray(alpha_grid, dtype=float),
            beta_values=beta_values,
            target_band=target_band,
            retention_config=retention_config,
            stratified_config=stratified_config,
        )
    if method_norm == "slsqp":
        return _select_best_pricing_slsqp(
            df=df,
            expected_loss=expected_loss,
            target_lr=target_lr,
            alpha_grid=np.asarray(alpha_grid, dtype=float),
            beta_values=beta_values,
            target_band=target_band,
            retention_config=retention_config,
            slsqp_options=slsqp_options,
        )
    if method_norm != "grid":
        raise ValueError(f"Unknown pricing optimization method: {method}")
    return _select_best_pricing_grid(
        df=df,
        expected_loss=expected_loss,
        target_lr=target_lr,
        alpha_grid=np.asarray(alpha_grid, dtype=float),
        beta_values=beta_values,
        target_band=target_band,
        retention_config=retention_config,
    )


def select_best_alpha(
    df: pd.DataFrame,
    expected_loss: pd.Series,
    target_lr: float,
    alpha_grid: np.ndarray,
    target_band: tuple[float, float] | None = None,
) -> tuple[float, pd.Series, PricingEvaluation]:
    best_alpha, _, best_premium, best_eval = select_best_pricing(
        df=df,
        expected_loss=expected_loss,
        target_lr=target_lr,
        alpha_grid=alpha_grid,
        beta_grid=np.asarray([1.0], dtype=float),
        target_band=target_band,
        method="grid",
        stratified_config=None,
    )
    return best_alpha, best_premium, best_eval
