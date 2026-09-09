"""Offline band-width sweep for min(R, G) (no GPU).

Joins the saved suffix-attack echoes (per turn x variant: r_c, m) with each
turn's grounding band (mu, sd) from train_refs.jsonl, then recomputes the
grounding leg G and the rule min(R, G) for several band widths:

    w = max(BAND_C * sd, w_min)
    G = min(m - (mu - w), (mu + w) - m)

Reports, per BAND_C, the mean min(R,G) for the honest thought vs each attack,
and the paired separation (fraction of turns where honest scores strictly
above the attack). This calibrates how tight the band can be before it hurts
honest thoughts and how wide before the filler suffix slips through.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results" / "minrg_round2"
REFS = RES / "train_refs.jsonl"
ATTACK = RES / "suffix_attack.jsonl"
W_MIN = 0.002
BAND_CS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
KINDS = ["honest", "h_suffix", "stub_suffix", "suffix_only", "boiler"]


def load() -> tuple[dict, dict]:
    band = {}
    for line in open(REFS):
        r = json.loads(line)
        if r.get("mu") is not None:
            band[r["turn_id"]] = (r["mu"], r["sd"])
    rows: dict = {}
    for line in open(ATTACK):
        r = json.loads(line)
        if r.get("m") is None or r.get("r_c") is None:
            continue
        rows.setdefault(r["turn_id"], {})[r["kind"]] = r
    return band, rows


def g_leg(m: float, mu: float, sd: float, band_c: float) -> float:
    w = max(band_c * sd, W_MIN)
    return min(m - (mu - w), (mu + w) - m)


def main() -> None:
    band, rows = load()
    print(f"{len(rows)} turns joined with band\n")
    for bc in BAND_CS:
        stats = {k: [] for k in KINDS}
        pair_win = {k: [0, 0] for k in KINDS}  # honest>attack, total
        for tid, d in rows.items():
            if tid not in band or "honest" not in d:
                continue
            mu, sd = band[tid]
            minrg = {}
            for k in KINDS:
                if k not in d:
                    continue
                g = g_leg(d[k]["m"], mu, sd, bc)
                minrg[k] = min(d[k]["r_c"], g)
                stats[k].append(minrg[k])
            h = minrg.get("honest")
            for k in KINDS:
                if k == "honest" or k not in minrg or h is None:
                    continue
                pair_win[k][1] += 1
                pair_win[k][0] += h > minrg[k]
        hon = st.mean(stats["honest"]) if stats["honest"] else float("nan")
        print(f"=== BAND_C={bc}  (honest mean min(R,G)={hon:+.4f}) ===")
        for k in KINDS:
            if k == "honest":
                continue
            m = st.mean(stats[k]) if stats[k] else float("nan")
            w, n = pair_win[k]
            frac = 100 * w / n if n else float("nan")
            print(f"  {k:12s} mean={m:+.4f}  honest_beats={w}/{n} ({frac:.0f}%)")
        print()


if __name__ == "__main__":
    main()
