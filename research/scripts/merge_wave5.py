"""Merge wave-5 E-KINGS into n≈30 Spearman table under S* v2.

Usage: source .venv/bin/activate && python scripts/merge_wave5.py
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
from harness.score import (  # noqa: E402
    DEFAULT_GAMMA,
    DEFAULT_GAMMA_BANK,
    DEFAULT_R_HI,
    DEFAULT_R_LO,
    calibration_ratio,
    gate_pass,
    rank_term,
)

RESULTS = ROOT / "results"
PAIR_FILES = [
    RESULTS / "ekings_v2_all.jsonl",
    RESULTS / "ekings_w2_v2_fullz.jsonl",
    RESULTS / "ekings_w3_v2.jsonl",
    RESULTS / "ekings_w3_VII.jsonl",
    RESULTS / "ekings_w3_all.jsonl",
    RESULTS / "ekings_w4.jsonl",
    RESULTS / "ekings_w5a.jsonl",
    RESULTS / "ekings_w5b.jsonl",
    RESULTS / "ekings_w5c.jsonl",
]


def load_bank() -> dict[str, float]:
    bf_lists: dict[str, list[float]] = defaultdict(list)
    for name in ["bank_w2_fullz.jsonl", "bank_w1.jsonl", "bank_w3.jsonl",
                 "bank_w4.jsonl", "bank_w5a.jsonl", "bank_w5b.jsonl",
                 "bank_w5c.jsonl"]:
        p = RESULTS / name
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            bf_lists[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
    return {m: st.mean(v) for m, v in bf_lists.items()}


def main() -> None:
    by: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for pf in PAIR_FILES:
        if not pf.exists():
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

    bf = load_bank()
    rows, su, tu, sh, th = [], [], [], [], []
    rejected = []
    for suf, swe in sorted(KING_BENCH.items(), key=lambda x: -x[1]):
        m = f"king-{suf}"
        ps = by.get(m)
        if not ps:
            continue
        mix = st.mean(rank_term(p) for p in ps)
        gate = st.mean(1.0 if gate_pass(p) else 0.0 for p in ps)
        r = calibration_ratio(ps)
        bank = bf.get(m)
        calib_ok = r is not None and DEFAULT_R_LO <= r <= DEFAULT_R_HI
        bank_ok = bank is None or bank >= DEFAULT_GAMMA_BANK
        valid = gate >= DEFAULT_GAMMA and bank_ok and calib_ok
        rows.append((suf, mix, gate, bank, r, valid, swe))
        su.append(mix); tu.append(swe)
        if valid:
            sh.append(mix); th.append(swe)
        else:
            rejected.append(suf)

    ru, pu = stats.spearmanr(su, tu)
    rh = ph = float("nan")
    if len(sh) >= 4:
        rh, ph = stats.spearmanr(sh, th)
    out = RESULTS / "hybrid_w5_table.txt"
    with open(out, "w") as f:
        f.write("S* v2 after wave-5\n")
        f.write("king        S_mix   gate   bank      r valid   swe\n")
        for suf, mix, gate, bank, r, valid, swe in rows:
            bs = f"{bank:.3f}" if bank is not None else "—"
            rs = f"{r:.3f}" if r is not None else "—"
            f.write(f"{suf:8s} {mix:8.4f} {gate*100:5.0f}% {bs:>6} {rs:>6} "
                    f"{str(valid):5} {swe:5.1f}\n")
        f.write(f"\nungated Spearman={ru:+.3f} (n={len(su)}, p={pu:.4g})\n")
        f.write(f"hybrid  Spearman={rh:+.3f} (n={len(sh)}, p={ph:.4g})\n")
        f.write(f"rejected: {rejected}\n")
    print(f"ungated ρ={ru:+.3f} n={len(su)} p={pu:.4g}")
    print(f"hybrid  ρ={rh:+.3f} n={len(sh)} p={ph:.4g} rej={rejected}")
    print("wrote", out)


if __name__ == "__main__":
    main()
