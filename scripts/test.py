#!/usr/bin/env python3
"""
Simple test runner script for uv run compatibility.

This provides a simple interface to run tests with common options.
"""

import sys
import os
import subprocess
from pathlib import Path


def main():
    """Main entry point."""
    # Change to the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Default to running all tests with verbose output
    args = [sys.executable, "-m", "pytest", "testsuite/", "-v"]

    # Simple argument parsing
    if len(sys.argv) > 1:
        if "--unit" in sys.argv or "-u" in sys.argv:
            args.extend(["-m", "unit"])
        elif "--integration" in sys.argv or "-i" in sys.argv:
            args.extend(["-m", "integration"])

        if "--parallel" in sys.argv or "-p" in sys.argv:
            args.extend(["-n", "auto"])

    print(f"Running: {' '.join(args)}")
    print("=" * 50)

    # Execute pytest
    result = subprocess.run(args)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
