"""Analyze D_tau2 programmability: S_tau2 vs tau2 vs swe.

Usage: source .venv/bin/activate && python scripts/analyze_tau2_prog.py
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
from harness.score import calibration_ratio, gate_pass, rank_term  # noqa: E402

RESULTS = ROOT / "results"
PAIR_FILES = [
    RESULTS / "ekings_tau2n_a.jsonl",
    RESULTS / "ekings_tau2n_b.jsonl",
    RESULTS / "ekings_tau2a.jsonl",
    RESULTS / "ekings_tau2b.jsonl",
]
TAU2 = json.loads((RESULTS / "tau2_mean_all.json").read_text())


def main() -> None:
    by: dict[str, list[dict]] = defaultdict(list)
    for pf in PAIR_FILES:
        if not pf.exists():
            continue
        for line in open(pf):
            r = json.loads(line)
            if r.get("valid") and "pairs" in r:
                by[r["miner"]].extend(r["pairs"])

    rows = []
    for suf, swe in KING_BENCH.items():
        m = f"king-{suf}"
        ps = by.get(m)
        if not ps or suf not in TAU2:
            continue
        mix = st.mean(rank_term(p) for p in ps)
        gate = st.mean(1.0 if gate_pass(p) else 0.0 for p in ps)
        r = calibration_ratio(ps)
        rows.append((suf, mix, gate, r, swe, TAU2[suf]))

    rows.sort(key=lambda x: -x[5])  # by tau2
    print(f"{'king':8} {'S_tau2':>8} {'gate':>5} {'r':>6} {'swe':>6} {'tau2':>6}")
    for suf, mix, gate, r, swe, t2 in rows:
        rr = f"{r:.3f}" if r is not None else "—"
        print(f"{suf:8} {mix:+8.4f} {gate:4.0%} {rr:>6} {swe:6.1f} {t2:6.3f}")

    if len(rows) >= 4:
        s = [x[1] for x in rows]
        swe = [x[4] for x in rows]
        t2 = [x[5] for x in rows]
        rs, ps = stats.spearmanr(s, swe)
        rt, pt = stats.spearmanr(s, t2)
        rb, pb = stats.spearmanr(swe, t2)
        print(f"\nSpearman S_tau2 vs swe  = {rs:+.3f} (p={ps:.3g}, n={len(rows)})")
        print(f"Spearman S_tau2 vs tau2 = {rt:+.3f} (p={pt:.3g}, n={len(rows)})")
        print(f"Spearman swe vs tau2    = {rb:+.3f} (p={pb:.3g}, n={len(rows)})")
        # verdict heuristic
        if rt > 0.5 and abs(rs) < 0.3:
            print("VERDICT: programmability SUPPORTED (S tracks tau2, orthogonal to swe)")
        elif rt < -0.5:
            print("VERDICT: ANTI-isomorphism — likely action-contract/format mismatch "
                  "(short-style wins; try native tool-call injection)")
        elif rt > rs and rt > 0.3:
            print("VERDICT: partial support (S tilts toward tau2 vs coding-D baseline)")
        else:
            print("VERDICT: not yet supported / need more n or format check")

    out = RESULTS / "tau2_prog_table.txt"
    lines = ["D_tau2 programmability probe (S* v2 ungated mix on tool-use turns)\n"]
    lines.append(f"{'king':8} {'S_tau2':>8} {'gate':>5} {'r':>6} {'swe':>6} {'tau2':>6}\n")
    for suf, mix, gate, r, swe, t2 in rows:
        rr = f"{r:.3f}" if r is not None else "—"
        lines.append(f"{suf:8} {mix:+8.4f} {gate:4.0%} {rr:>6} {swe:6.1f} {t2:6.3f}\n")
    if len(rows) >= 4:
        lines.append(
            f"\nS vs swe={rs:+.3f}  S vs tau2={rt:+.3f}  "
            f"swe vs tau2={rb:+.3f}  n={len(rows)}\n"
        )
    out.write_text("".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
