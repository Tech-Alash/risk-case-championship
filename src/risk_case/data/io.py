from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

