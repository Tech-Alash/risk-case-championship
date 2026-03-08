from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RawValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


@dataclass
class PreprocessArtifacts:
    dataset_path: Path
    metadata_path: Path
    quality_report_path: Path
    feature_columns: list[str]
    target_columns: list[str]
    row_count: int

