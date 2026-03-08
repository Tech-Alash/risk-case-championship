from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.orchestration.run_pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run risk-case experiment pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.json",
        help="Path to JSON configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

