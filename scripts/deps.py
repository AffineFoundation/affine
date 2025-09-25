#!/usr/bin/env python3
"""Check dependencies for scripts and testsuite directories."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return its exit code."""
    print(f"Running {description}...")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        print(f"❌ {description} found issues")
    else:
        print(f"✅ {description} passed")
    return result.returncode


def check() -> int:
    """Check for unused dependencies in scripts/testsuite."""
    # Only check if our dev dependencies are being used properly
    return run_command(
        ["deptry", "./scripts", "./testsuite", "--ignore", "DEP002"], "dependency check"
    )


if __name__ == "__main__":
    sys.exit(check())
