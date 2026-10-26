"""Frozen task pool + deterministic seeded sampling.

The pool manifest (data/pool.json) is a snapshot of every usable instance in
the SWE-rebench leaderboard set. It is built once by scripts/build_pool.py and
committed, so `sample(n, seed)` is reproducible forever regardless of what the
upstream HF dataset does later.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

POOL_PATH = Path(__file__).resolve().parent / "data" / "pool.json"


def image_for(iid: str) -> str:
    """Docker image naming rule shared with the SWE-rebench harness."""
    return ("swerebench/sweb.eval.x86_64."
            f"{iid.replace('__', '_1776_').lower()}:latest")


def load_pool(path: Path = POOL_PATH) -> dict:
    pool = json.loads(path.read_text())
    for key in ("dataset", "split", "namespace", "instance_ids"):
        if key not in pool:
            raise ValueError(f"pool manifest missing {key!r}")
    return pool


def sample(pool: dict, n: int, seed: int) -> list[str]:
    """Deterministic panel: shuffle the sorted pool with `seed`, take n.

    Same (pool, n, seed) -> same task set, always. Result is sorted for
    stable presentation; the selection itself is order-independent.
    """
    ids = sorted(set(pool["instance_ids"]))
    if n > len(ids):
        raise ValueError(f"n={n} exceeds pool size {len(ids)}")
    rng = random.Random(seed)
    rng.shuffle(ids)
    return sorted(ids[:n])


def panel_key(instance_ids: list[str]) -> str:
    """Stable short key for a task set (used to cache dataset subsets)."""
    blob = "\n".join(sorted(instance_ids)).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
