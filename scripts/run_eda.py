from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.eda import EDAConfig, run_eda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full EDA analysis for risk-case dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "eda.json",
        help="Path to EDA JSON config",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = EDAConfig.from_json(args.config)
    result = run_eda(config=config, project_root=ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

