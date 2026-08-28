#!/usr/bin/env python3
"""Unified overnight status: one screen for all experiment tracks.

Each track appends pipe-delimited lines to research/logs/track{A,B,C}_status.log.
This renders the latest state and the round-metric trajectory so the coordinator
(and the user) can read the night at a glance without touching the pods.
"""
from __future__ import annotations

import os
import re
import time

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

TRACKS = {
    "M": "two-box KOTH mock: crown-triggered from-scratch D, frozen per reign",
    "A": "realistic scale: G=Qwen3.6-35B-A3B vs T=Qwen3.8-27B, online D, 150-instance panel",
    "B": "fast dynamics: G=Qwen3-4B vs T=Qwen3-32B, ONLINE D (the thesis arm)",
    "C": "control: same as B but D trained once then FROZEN (mechanism ablation)",
}

ROUND_RE = re.compile(r"round[ =]+(\d+)", re.I)


def read(track):
    path = os.path.join(LOGS, f"track{track}_status.log")
    if not os.path.exists(path):
        return None, []
    lines = [l.strip() for l in open(path) if l.strip()]
    return path, lines


def main():
    now = time.time()
    for t, desc in TRACKS.items():
        path, lines = read(t)
        print("=" * 78)
        print(f"TRACK {t} - {desc}")
        if not lines:
            print("  (no status yet)")
            continue
        age = int(now - os.path.getmtime(path))
        print(f"  lines={len(lines)}  last update {age}s ago")
        rounds = [l for l in lines if ROUND_RE.search(l)]
        stages = [l for l in lines if not ROUND_RE.search(l)]
        for l in stages[-4:]:
            print("   ", l[:160])
        if rounds:
            print(f"  -- {len(rounds)} loop rounds logged; latest:")
            for l in rounds[-6:]:
                print("   ", l[:160])
    print("=" * 78)


if __name__ == "__main__":
    main()
