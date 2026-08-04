"""Freeze hybrid S* Spearman table from local E-KINGS + bank jsonls.

Usage: python scripts/freeze_hybrid.py
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
from harness.score import (  # noqa: E402
    DEFAULT_GAMMA,
    DEFAULT_GAMMA_BANK,
    gate_pass,
    rank_term,
)

RESULTS = ROOT / "results"
TRUTH = {
    "king-genesis": 58.2, "king-XCIX": 39.8, "king-I": 38.4, "king-II": 37.2,
    "king-VIII": 36.2, "king-XCIV": 36.2, "king-XLI": 34.2, "king-XI": 33.6,
    "king-VII": 33.2, "king-III": 32.8, "king-V": 32.0, "king-XLV": 26.0,
    "king-XLVI": 13.2, "king-CI": 12.4, "king-LI": 11.6,
}
PAIR_FILES = [
    RESULTS / "ekings_v2_all.jsonl",
    RESULTS / "ekings_w2_v2_fullz.jsonl",
    RESULTS / "ekings_w3_v2.jsonl",
    RESULTS / "ekings_w3_VII.jsonl",
    RESULTS / "ekings_w3_all.jsonl",
]


def load_bank_fracs() -> tuple[dict[str, float], dict[str, int], set[str]]:
    """Prefer bank_w2_fullz over truncated bank_w2 for late kings."""
    bf_lists: dict[str, list[float]] = defaultdict(list)
    miners_from_fullz: set[str] = set()
    fullz = RESULTS / "bank_w2_fullz.jsonl"
    if fullz.exists():
        for line in open(fullz):
            r = json.loads(line)
            bf_lists[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
            miners_from_fullz.add(r["miner"])
    for p in [RESULTS / "bank_w1.jsonl", RESULTS / "bank_w3.jsonl",
              RESULTS / "bank_w2.jsonl"]:
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            if p.name == "bank_w2.jsonl" and r["miner"] in miners_from_fullz:
                continue
            bf_lists[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
    bf = {m: st.mean(v) for m, v in bf_lists.items()}
    bn = {m: len(v) for m, v in bf_lists.items()}
    return bf, bn, miners_from_fullz


def main() -> None:
    by_pairs: dict[str, list[dict]] = defaultdict(list)
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
            by_pairs[r["miner"]].extend(r["pairs"])

    bf, bn, miners_from_fullz = load_bank_fracs()
    scores_u, truth_u, scores_h, truth_h, rejected, rows = [], [], [], [], [], []
    for m, swe in sorted(TRUTH.items(), key=lambda x: -x[1]):
        ps = by_pairs.get(m)
        if not ps:
            print("MISSING", m)
            continue
        mix = st.mean(rank_term(p) for p in ps)
        gate = st.mean(1.0 if gate_pass(p) else 0.0 for p in ps)
        bank = bf.get(m)
        valid = gate >= DEFAULT_GAMMA and (bank is None or bank >= DEFAULT_GAMMA_BANK)
        rows.append((m.removeprefix("king-"), mix, gate, bank, bn.get(m, 0), valid, swe))
        scores_u.append(mix)
        truth_u.append(swe)
        if gate < DEFAULT_GAMMA or (bank is not None and bank < DEFAULT_GAMMA_BANK):
            rejected.append(m.removeprefix("king-"))
        else:
            scores_h.append(mix)
            truth_h.append(swe)

    ru, pu = stats.spearmanr(scores_u, truth_u)
    rh, ph = stats.spearmanr(scores_h, truth_h)
    note = "fullz" if miners_from_fullz else "truncated-w2"
    out = RESULTS / "hybrid_w1_table.txt"
    with open(out, "w") as f:
        f.write(f"S* = mix w=1.0 + gates γ=0.3 γ_bank=0.08  (banks: w1+w3+{note})\n")
        f.write("king        S_mix   gate   bank  nbank valid   swe\n")
        for r in rows:
            bs = f"{r[3]:.3f}" if r[3] is not None else "—"
            f.write(
                f"{r[0]:8s} {r[1]:8.4f} {r[2]*100:5.0f}% {bs:>6} "
                f"{r[4]:5d} {str(r[5]):5} {r[6]:5.1f}\n"
            )
        f.write(f"\nungated Spearman={ru:+.3f} (n={len(scores_u)}, p={pu:.4g})\n")
        f.write(f"hybrid  Spearman={rh:+.3f} (n={len(scores_h)}, p={ph:.4g})\n")
        f.write(f"rejected: {rejected}\n")
    meta = {
        "ungated_spearman": ru, "ungated_p": pu, "ungated_n": len(scores_u),
        "hybrid_spearman": rh, "hybrid_p": ph, "hybrid_n": len(scores_h),
        "rejected": rejected,
        "bank_fracs": {m: {"frac": bf[m], "n": bn[m]} for m in sorted(bf)},
        "fullz_miners": sorted(miners_from_fullz),
    }
    (RESULTS / "hybrid_freeze_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
