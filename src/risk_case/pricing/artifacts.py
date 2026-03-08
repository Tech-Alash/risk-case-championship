from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PricingPolicyArtifact:
    kind: str
    method: str
    alpha: float | None = None
    beta: float | None = None
    bucket_edges: list[float] = field(default_factory=list)
    bucket_params: list[dict[str, Any]] = field(default_factory=list)
    global_init_alpha: float | None = None
    global_init_beta: float | None = None
    bucket_feature: str = "expected_loss"

    @staticmethod
    def scalar(alpha: float, beta: float, method: str) -> "PricingPolicyArtifact":
        return PricingPolicyArtifact(
            kind="scalar",
            method=str(method),
            alpha=float(alpha),
            beta=float(beta),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "method": self.method,
            "alpha": self.alpha,
            "beta": self.beta,
            "bucket_edges": [float(item) for item in self.bucket_edges],
            "bucket_params": [dict(item) for item in self.bucket_params],
            "global_init_alpha": self.global_init_alpha,
            "global_init_beta": self.global_init_beta,
            "bucket_feature": self.bucket_feature,
        }

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "kind": self.kind,
            "method": self.method,
            "alpha": self.alpha,
            "beta": self.beta,
            "bucket_feature": self.bucket_feature,
            "bucket_count": int(len(self.bucket_params)),
        }
        if self.kind == "stratified":
            summary["global_init_alpha"] = self.global_init_alpha
            summary["global_init_beta"] = self.global_init_beta
            summary["bucket_edges"] = [float(item) for item in self.bucket_edges]
            summary["bucket_params_preview"] = [dict(item) for item in self.bucket_params[: min(3, len(self.bucket_params))]]
        return summary

    def save(self, path: Path | str) -> None:
        path_obj = Path(path)
        path_obj.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def from_dict(raw: dict[str, Any] | None) -> "PricingPolicyArtifact":
        data = raw or {}
        return PricingPolicyArtifact(
            kind=str(data.get("kind", "scalar")),
            method=str(data.get("method", "grid")),
            alpha=float(data["alpha"]) if data.get("alpha") is not None else None,
            beta=float(data["beta"]) if data.get("beta") is not None else None,
            bucket_edges=[float(item) for item in (data.get("bucket_edges") or [])],
            bucket_params=[dict(item) for item in (data.get("bucket_params") or [])],
            global_init_alpha=(
                float(data["global_init_alpha"]) if data.get("global_init_alpha") is not None else None
            ),
            global_init_beta=(
                float(data["global_init_beta"]) if data.get("global_init_beta") is not None else None
            ),
            bucket_feature=str(data.get("bucket_feature", "expected_loss")),
        )

    @staticmethod
    def load(path: Path | str) -> "PricingPolicyArtifact":
        path_obj = Path(path)
        return PricingPolicyArtifact.from_dict(json.loads(path_obj.read_text(encoding="utf-8")))
