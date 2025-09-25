#!/usr/bin/env python3
"""Linting and formatting scripts for the affine project."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return its exit code."""
    print(f"Running {description}...")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        print(f"❌ {description} failed")
    else:
        print(f"✅ {description} passed")
    return result.returncode


def check() -> int:
    """Check linting without fixing."""
    return run_command(["ruff", "check", "./scripts", "./testsuite"], "lint check")


def fix() -> int:
    """Fix linting issues and format code."""
    lint_result = run_command(
        ["ruff", "check", "--fix", "./scripts", "./testsuite"], "lint fix"
    )
    format_result = run_command(
        ["ruff", "format", "./scripts", "./testsuite"], "format"
    )
    return max(lint_result, format_result)


def format() -> int:
    """Format code."""
    return run_command(["ruff", "format", "./scripts", "./testsuite"], "format")


def format_check() -> int:
    """Check formatting without fixing."""
    return run_command(
        ["ruff", "format", "--check", "./scripts", "./testsuite"], "format check"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        func_name = sys.argv[1]
        if func_name in globals() and callable(globals()[func_name]):
            sys.exit(globals()[func_name]())
    print("Usage: python scripts/lint.py [check|fix|format|format_check]")
    sys.exit(1)
