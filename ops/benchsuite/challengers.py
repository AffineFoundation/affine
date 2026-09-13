#!/usr/bin/env python
"""Pick the losing challengers worth benchmarking: the near-misses.

Reads the validator's `affine/state/history.jsonl` (one row per duel), keeps
verdict rows of the last --window-hours whose challenger LOST with a positive
paired margin (it out-scored the king on the slice but not by 2·SE / δ), sorts
by margin, prints the top-k as JSON lines {challenge_id, hotkey, repo,
revision, margin, z, at}. `run_pass.sh <repo> <challenge_id> <run_id> challenger`
then benches one (CHALLENGER_REVISION=<revision>). The question this serves is
RT-7: does duel margin predict benchmark score?

  python challengers.py --window-hours 24 --top 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HISTORY = REPO / "affine" / "state" / "history.jsonl"


def iso_ts(s: str) -> float:
    return time.mktime(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))


def near_misses(window_hours: float, top: int, history: Path = HISTORY) -> list[dict]:
    cutoff = time.time() - window_hours * 3600
    rows = []
    for line in history.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event") != "verdict" or r.get("accepted"):
            continue
        v = r.get("verdict") or {}
        margin = v.get("margin")
        if margin is None or margin <= 0 or v.get("rejection_reason"):
            continue
        if iso_ts(r.get("at", "1970-01-01T00:00:00")) < cutoff:
            continue
        rows.append({"challenge_id": r.get("challenge_id"), "hotkey": r.get("hotkey"),
                     "repo": r.get("repo"), "revision": r.get("revision"),
                     "margin": margin, "z": v.get("z"), "at": r.get("at")})
    rows.sort(key=lambda x: -x["margin"])
    return rows[:top]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-hours", type=float, default=24)
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    for r in near_misses(a.window_hours, a.top):
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
