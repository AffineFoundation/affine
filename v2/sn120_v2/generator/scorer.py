"""``python -m sn120_v2.generator.scorer`` pod command."""

from __future__ import annotations

import sys

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
