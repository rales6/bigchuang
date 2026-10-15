from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from pursuit.run_logging import RunLogBundle


class RunLogBundleTests(unittest.TestCase):
    def test_creates_unique_run_and_tees_console(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = RunLogBundle.create(
                root, "追踪 蓝色瓶子", "dry_run", ["runner"], {"dry_run": True}
            )
            original_stdout = sys.stdout
            try:
                bundle.install_console_tee()
                print("camera diagnostic line")
                bundle.finalize("interrupted", duration_s=1.5)
            finally:
                bundle.close()
            self.assertIs(sys.stdout, original_stdout)
            self.assertIn("camera diagnostic line", bundle.console_path.read_text("utf-8"))
            metadata = json.loads(bundle.metadata_path.read_text("utf-8"))
            self.assertEqual(metadata["status"], "interrupted")
            self.assertEqual(metadata["instruction"], "追踪 蓝色瓶子")
            self.assertEqual(Path((root / "LATEST.txt").read_text(encoding="utf-8").strip()),
                             bundle.run_dir.resolve())

    def test_telemetry_override_parent_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = RunLogBundle.create(root, "target", "esp32", [], {})
            override = root / "exports" / "run.csv"
            try:
                chosen = bundle.choose_telemetry_path(str(override))
            finally:
                bundle.close()
            self.assertEqual(chosen, override)
            self.assertTrue(override.parent.is_dir())


if __name__ == "__main__":
    unittest.main()
