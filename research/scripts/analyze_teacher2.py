"""Compare S under GLM-Air (T1) vs Qwen3-32B (T2) on overlapping kings.

Usage: source .venv/bin/activate && python scripts/analyze_teacher2.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.config import KING_BENCH  # noqa: E402
from harness.score import rank_term  # noqa: E402

RESULTS = ROOT / "results"
T1_FILES = [
    RESULTS / "ekings_v2_all.jsonl",
    RESULTS / "ekings_w2_v2_fullz.jsonl",
    RESULTS / "ekings_w3_v2.jsonl",
    RESULTS / "ekings_w3_VII.jsonl",
    RESULTS / "ekings_w3_all.jsonl",
    RESULTS / "ekings_w4.jsonl",
]
T2_FILE = RESULTS / "ekings_teacher2_qwen32.jsonl"


def load(files) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for pf in files:
        if not Path(pf).exists():
            continue
        for line in open(pf):
            r = json.loads(line)
            if not (r.get("valid") and "pairs" in r):
                continue
            key = (r["turn_id"], r["miner"])
            if key in seen:
                continue
            seen.add(key)
            by[r["miner"]].extend(r["pairs"])
    return by


def main() -> None:
    if not T2_FILE.exists():
        print("missing", T2_FILE); return
    t1, t2 = load(T1_FILES), load([T2_FILE])
    rows = []
    for m, ps2 in t2.items():
        suf = m.removeprefix("king-")
        if suf not in KING_BENCH or m not in t1:
            continue
        s1 = st.mean(rank_term(p) for p in t1[m])
        s2 = st.mean(rank_term(p) for p in ps2)
        rows.append((suf, s1, s2, KING_BENCH[suf]))
    if len(rows) < 4:
        print("need ≥4 overlapping kings; have", len(rows)); return
    print(f"{'king':8s} {'S_T1':>9s} {'S_T2':>9s} {'swe':>6s}")
    for r in sorted(rows, key=lambda x: -x[3]):
        print(f"{r[0]:8s} {r[1]:+9.4f} {r[2]:+9.4f} {r[3]:6.1f}")
    s1 = [r[1] for r in rows]; s2 = [r[2] for r in rows]; swe = [r[3] for r in rows]
    print(f"\nT1 vs swe ρ={stats.spearmanr(s1,swe)[0]:+.3f} n={len(rows)}")
    print(f"T2 vs swe ρ={stats.spearmanr(s2,swe)[0]:+.3f} n={len(rows)}")
    print(f"T1 vs T2  ρ={stats.spearmanr(s1,s2)[0]:+.3f}")
    out = RESULTS / "teacher2_compare.txt"
    with open(out, "w") as f:
        f.write("Second-teacher replication: GLM-4.5-Air (T1) vs Qwen3-32B (T2)\n")
        for r in sorted(rows, key=lambda x: -x[3]):
            f.write(f"{r[0]:8s} T1={r[1]:+.4f} T2={r[2]:+.4f} swe={r[3]}\n")
        f.write(f"T1vsSWE={stats.spearmanr(s1,swe)[0]:+.3f} "
                f"T2vsSWE={stats.spearmanr(s2,swe)[0]:+.3f} "
                f"T1vsT2={stats.spearmanr(s1,s2)[0]:+.3f} n={len(rows)}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
