from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TARGET_CLAIM_COL = "is_claim"
TARGET_AMOUNT_COL = "claim_amount"
TARGET_COUNT_COL = "claim_cnt"
PREMIUM_COL = "premium"
PREMIUM_NET_COL = "premium_wo_term"
CONTRACT_COL = "contract_number"
UNIQUE_ID_COL = "unique_id"
DRIVER_ID_COL = "driver_iin"
INSURER_ID_COL = "insurer_iin"
CAR_NUMBER_COL = "car_number"

DEFAULT_TARGET_COLUMNS = [TARGET_CLAIM_COL, TARGET_AMOUNT_COL, TARGET_COUNT_COL]
DEFAULT_FORBIDDEN_FEATURE_COLUMNS = [
    UNIQUE_ID_COL,
    CONTRACT_COL,
    DRIVER_ID_COL,
    INSURER_ID_COL,
    CAR_NUMBER_COL,
]


@dataclass(frozen=True)
class Targets:
    lr: float = 0.7


DEFAULT_TARGETS = Targets()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
