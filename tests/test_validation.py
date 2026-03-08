from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_case.data.validation import validate_dataset


class TestValidation(unittest.TestCase):
    def test_validation_passes_for_minimum_valid_frame(self) -> None:
        df = pd.DataFrame(
            {
                "premium": [100.0, 200.0],
                "premium_wo_term": [90.0, 180.0],
                "is_claim": [0, 1],
                "claim_amount": [0.0, 1500.0],
                "claim_cnt": [0, 1],
            }
        )
        result = validate_dataset(df)
        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_validation_fails_when_required_columns_missing(self) -> None:
        df = pd.DataFrame({"premium": [100.0]})
        result = validate_dataset(df)
        self.assertFalse(result.ok)
        self.assertTrue(any("Missing required columns" in message for message in result.errors))


if __name__ == "__main__":
    unittest.main()

