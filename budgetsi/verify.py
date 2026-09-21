"""Verify the project adapter without models, GPUs, APIs, or training data."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "ac26e38d6f1572eb027597b48a9f4e01f6915ef8"


def main():
    changes = subprocess.check_output(["git", "diff", "--name-only", PIN], cwd=ROOT, text=True).splitlines()
    if any(not p.startswith("budgetsi/") for p in changes):
        raise RuntimeError("This delivery must not modify upstream framework files")
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "budgetsi/tests", "-v"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    sources = sorted((ROOT / "budgetsi").rglob("*.py"))
    report = {
        "status": "passed" if result.returncode == 0 else "failed",
        "upstream_pin": PIN,
        "upstream_framework_modified": False,
        "scope": (
            "CPU project adapter and exact pinned function bodies with toy model/container; no GPU/Ray integration"
        ),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "test_log": result.stdout + result.stderr,
        "training_started": False,
        "full_social_pipeline_integrated": False,
    }
    (ROOT / "budgetsi/VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["test_log"])
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
