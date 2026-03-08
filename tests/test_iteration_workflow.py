from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stability_checks import _read_splits


class TestIterationWorkflow(unittest.TestCase):
    def test_default_config_has_mandatory_constraints_and_fixed_holdout(self) -> None:
        cfg = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
        self.assertTrue(cfg["benchmark"]["must_pass_constraints"])
        self.assertEqual(cfg["benchmark"]["selection_metric"], "policy_score")
        self.assertEqual(cfg["validation"]["time_holdout_start"], "2022-09-22 00:00:00")

    def test_stability_splits_config_has_multiple_dates(self) -> None:
        cfg = json.loads((ROOT / "configs" / "experiments" / "stability_splits.json").read_text(encoding="utf-8"))
        dates = cfg.get("time_holdout_starts")
        self.assertIsInstance(dates, list)
        self.assertGreaterEqual(len(dates), 3)
        for value in dates:
            self.assertIsInstance(value, str)
            self.assertTrue(value.strip())

    def test_read_splits_rejects_empty_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad_splits.json"
            path.write_text(json.dumps({"time_holdout_starts": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_splits(path)


if __name__ == "__main__":
    unittest.main()
