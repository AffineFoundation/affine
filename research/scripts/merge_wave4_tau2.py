"""Merge ekings_w4 + bank_w4 into hybrid/tau2 tables when wave-4 finishes.

Usage: python scripts/merge_wave4_tau2.py
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
import urllib.request
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
    "king-XLII": 33.4, "king-VII": 33.2, "king-III": 32.8, "king-XL": 32.2,
    "king-C": 32.0, "king-V": 32.0, "king-XCVI": 32.0, "king-XLV": 26.0,
    "king-XLVI": 13.2, "king-CI": 12.4, "king-LI": 11.6,
}
PAIR_FILES = [
    RESULTS / "ekings_v2_all.jsonl",
    RESULTS / "ekings_w2_v2_fullz.jsonl",
    RESULTS / "ekings_w3_v2.jsonl",
    RESULTS / "ekings_w3_VII.jsonl",
    RESULTS / "ekings_w3_all.jsonl",
    RESULTS / "ekings_w4.jsonl",
]


def load_bank() -> tuple[dict[str, float], dict[str, int]]:
    bf_lists: dict[str, list[float]] = defaultdict(list)
    fullz: set[str] = set()
    if (RESULTS / "bank_w2_fullz.jsonl").exists():
        for line in open(RESULTS / "bank_w2_fullz.jsonl"):
            r = json.loads(line)
            bf_lists[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
            fullz.add(r["miner"])
    for p in [RESULTS / "bank_w1.jsonl", RESULTS / "bank_w3.jsonl",
              RESULTS / "bank_w4.jsonl", RESULTS / "bank_w2.jsonl"]:
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            if p.name == "bank_w2.jsonl" and r["miner"] in fullz:
                continue
            bf_lists[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
    # Prefer bank_w4 means for those miners (recompute last)
    if (RESULTS / "bank_w4.jsonl").exists():
        w4: dict[str, list[float]] = defaultdict(list)
        for line in open(RESULTS / "bank_w4.jsonl"):
            r = json.loads(line)
            w4[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
        for m, vs in w4.items():
            bf_lists[m] = vs
    bf = {m: st.mean(v) for m, v in bf_lists.items()}
    bn = {m: len(v) for m, v in bf_lists.items()}
    return bf, bn


def load_tau2() -> dict[str, float]:
    req = urllib.request.Request(
        "https://albedo.tech/data/benchmarks.json",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    b = json.loads(urllib.request.urlopen(req, timeout=60).read())
    tau: dict[str, float] = {}
    for m in b["models"]:
        mid = m.get("id", "")
        lab = m.get("label", "")
        suf = None
        mm = re.search(r"king-([A-Z]+)$", mid)
        if mm:
            suf = mm.group(1)
        elif lab == "genesis":
            suf = "genesis"
        else:
            mm = re.search(r"King\s+([A-Z]+)$", lab)
            if mm:
                suf = mm.group(1)
        if not suf:
            continue
        ls = m.get("latest_scores") or {}
        vals = []
        for k in ["tau2_airline", "tau2_retail", "tau2_telecom"]:
            if k in ls and ls[k].get("score") is not None \
                    and ls[k].get("state") == "SUCCEEDED":
                vals.append(ls[k]["score"])
        if vals:
            tau[suf] = st.mean(vals)
    return tau


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

    bf, bn = load_bank()
    scores_u, truth_u, scores_h, truth_h, rejected, rows = [], [], [], [], [], []
    for m, swe in sorted(TRUTH.items(), key=lambda x: -x[1]):
        ps = by.get(m)
        if not ps:
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
    print(f"ungated swe ρ={ru:+.3f} n={len(scores_u)} p={pu:.4g}")
    print(f"hybrid  swe ρ={rh:+.3f} n={len(scores_h)} p={ph:.4g} rej={rejected}")

    tau = load_tau2()
    pairs_tau = []
    pairs_tau_h = []
    for lab, mix, _gate, _bank, _nbank, valid, _swe in rows:
        if lab not in tau:
            continue
        pairs_tau.append((mix, tau[lab], lab))
        if valid:
            pairs_tau_h.append((mix, tau[lab], lab))
    if len(pairs_tau) >= 5:
        r, p = stats.spearmanr([a for a, _, _ in pairs_tau],
                               [b for _, b, _ in pairs_tau])
        print(f"ungated tau2 ρ={r:+.3f} n={len(pairs_tau)} p={p:.4g} "
              f"kings={[c for _, _, c in pairs_tau]}")
    if len(pairs_tau_h) >= 5:
        r, p = stats.spearmanr([a for a, _, _ in pairs_tau_h],
                               [b for _, b, _ in pairs_tau_h])
        print(f"hybrid  tau2 ρ={r:+.3f} n={len(pairs_tau_h)} p={p:.4g}")

    out = RESULTS / "hybrid_w4_table.txt"
    with open(out, "w") as f:
        f.write("S* mix w=1.0 + gates (incl. wave-4)\n")
        f.write("king        S_mix   gate   bank  nbank valid   swe\n")
        for rrow in rows:
            bs = f"{rrow[3]:.3f}" if rrow[3] is not None else "—"
            f.write(f"{rrow[0]:8s} {rrow[1]:8.4f} {rrow[2]*100:5.0f}% {bs:>6} "
                    f"{rrow[4]:5d} {str(rrow[5]):5} {rrow[6]:5.1f}\n")
        f.write(f"\nungated Spearman={ru:+.3f} (n={len(scores_u)}, p={pu:.4g})\n")
        f.write(f"hybrid  Spearman={rh:+.3f} (n={len(scores_h)}, p={ph:.4g})\n")
        f.write(f"rejected: {rejected}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
