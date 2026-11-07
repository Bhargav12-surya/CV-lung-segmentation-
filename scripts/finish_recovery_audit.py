"""Wait for the current cache recovery, then run a read-only audit. Never train."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write_status(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--output", required=True, help="NEW audit JSON destination")
    args = parser.parse_args(argv)
    progress, output = Path(args.progress).resolve(), Path(args.output).resolve()
    status_file = output.with_suffix(".job.json")
    if output.exists() or status_file.exists():
        parser.error("Choose a new output; existing verification records are preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_status(status_file, {"status": "waiting_for_cache_recovery", "progress": str(progress)})
    while True:
        state = json.loads(progress.read_text(encoding="utf-8"))
        if state["status"] == "complete_original_cache_restoration":
            break
        if state["status"] in ("failed", "interrupted"):
            write_status(status_file, {"status": "recovery_not_complete", "recovery": state["status"],
                                       "reason": state.get("reason"), "progress": str(progress)})
            return 2
        time.sleep(10)
    write_status(status_file, {"status": "running_read_only_audit", "output": str(output)})
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts" / "audit_dataset.py"),
               "--run", str(root / "runs" / "hybrid_compact_v2_seed42"),
               "--raw-limit", "12", "--full-hu", "--output", str(output)]
    with output.with_suffix(".log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
    write_status(status_file, {"status": "passed" if result.returncode == 0 else "audit_failed",
                               "returncode": result.returncode, "output": str(output),
                               "scope": "all restored cache arrays; 12 raw geometry rechecks; no model training"})
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
