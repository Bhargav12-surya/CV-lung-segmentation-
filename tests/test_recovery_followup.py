"""The recovery follow-up launches only a read-only audit, never training."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import finish_recovery_audit as followup


class RecoveryFollowupTests(unittest.TestCase):
    def test_completed_recovery_runs_only_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress, output = root / "progress.json", root / "audit.json"
            progress.write_text(json.dumps({"status": "complete_original_cache_restoration"}))
            with patch.object(followup.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                self.assertEqual(followup.main(["--progress", str(progress), "--output", str(output)]), 0)
                command = run.call_args.args[0]
                self.assertTrue(command[1].endswith("audit_dataset.py"))
                self.assertNotIn("train", command)
                self.assertIn("--full-hu", command)
            self.assertEqual(json.loads(output.with_suffix(".job.json").read_text())["status"], "passed")

    def test_failed_recovery_does_not_launch_any_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress, output = root / "progress.json", root / "audit.json"
            progress.write_text(json.dumps({"status": "failed", "reason": "missing source"}))
            with patch.object(followup.subprocess, "run") as run:
                self.assertEqual(followup.main(["--progress", str(progress), "--output", str(output)]), 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
