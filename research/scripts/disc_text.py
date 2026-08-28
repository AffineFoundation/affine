#!/usr/bin/env python3
"""Text helpers shared by the discriminator trainer and the offline checks.

Kept free of torch so the leak/baseline checks can run on a laptop without the
training stack, and so both paths provably use the same normalisation.
"""
from __future__ import annotations

import json
import re

# Reasoning markers that would otherwise identify the source model. Measured on
# the raw data, "<think>" appears in 33.5% of teacher thoughts vs 7.0% of miner
# thoughts; a rule matching that tag alone scores AUC 0.632, so it must go.
MARKER_RE = re.compile(r"</?\s*(think|thinking|thought|reasoning|answer)\s*>", re.I)
WS_RE = re.compile(r"\n{3,}")


def normalize(s: str) -> str:
    """Strip reasoning markers so the source model is not identifiable by format."""
    s = MARKER_RE.sub("", s or "")
    s = WS_RE.sub("\n\n", s)
    return s.strip()


def as_list(v):
    """teacher_y / teacher_z are sometimes a list, sometimes a JSON-encoded list."""
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        try:
            p = json.loads(v)
            return [str(x) for x in p] if isinstance(p, list) else [v]
        except Exception:
            return [v]
    return []
