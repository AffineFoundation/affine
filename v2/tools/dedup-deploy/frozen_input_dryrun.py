#!/usr/bin/env python3
"""Isolated launcher for the real sn120_v2 CLI in an operator-selected worktree.

Usage: python -I frozen_input_dryrun.py --v2-root /selected/worktree/v2 ARGS...
Only the explicit v2 root is added to sys.path; ambient PYTHONPATH is ignored.
The frozen_inputs.py module in this script's directory is not imported.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v2-root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = args.v2_root.resolve()
    if not root.is_dir() or not (root / "sn120_v2" / "cli.py").is_file():
        raise SystemExit("selected --v2-root does not contain sn120_v2/cli.py")
    sys.path.insert(0, str(root))
    cli = importlib.import_module("sn120_v2.cli")
    return cli.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
