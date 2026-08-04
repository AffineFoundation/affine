"""RT-3b defense study: miner-channel calibration / bounding of L1lift.

Candidate terms (per pair):
  l1lift     = lpA(y_C|z_A) - lpA(y_C|e)                     [production]
  l1cal_zc   = lpA(y_C|z_A) - max(lpA(y_C|e), lpA(y_C|z_C))  [foreign-thought cal]
  cap{c}     = min(l1lift, c)                                [bounded gain]
  clip{c}    = clip(l1lift, -c, c)                           [bounded influence]
  tanh{c}    = c * tanh(l1lift / c)                          [smooth bound]

Attacker models for RT-3b (rescale lp' = lp*(1-s), pushing logprobs toward 0):
  self    : rescale only lpA(y_C|z_A)  — attacker fingerprints its own thoughts
  blind   : rescale lpA(y_C|z_A) AND lpA(y_C|z_C)
  uniform : rescale all lpA components incl. empty (serving temperature)

Two evaluations per (variant, attacker):
  1. mean test: min strength s where attacked king's mean mix > genesis mean
  2. duel test: min s where paired duel z > 3 on common turns (production rule)

Usage: source .venv/bin/activate && python scripts/rt3_calibration.py
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.config import KING_BENCH  # noqa: E402
from harness.score import DEFAULT_L1_WEIGHT, lambda2  # noqa: E402

RESULTS = ROOT / "results"
PAIR_FILES = [
    RESULTS / "ekings_v2_all.jsonl",
    RESULTS / "ekings_w2_v2_fullz.jsonl",
    RESULTS / "ekings_w3_v2.jsonl",
    RESULTS / "ekings_w3_VII.jsonl",
    RESULTS / "ekings_w3_all.jsonl",
    RESULTS / "ekings_w4.jsonl",
]
NEEDED = ("lpA_yc_za", "lpA_yc_e", "lpA_yc_zc", "lpC_yc_za", "lpC_yc_e")
ATTACK_KINGS = ["king-I", "king-II", "king-V", "king-VII"]
STRENGTHS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
VARIANTS = ["l1lift", "l1cal_zc",
            "cap0.05", "cap0.1", "cap0.2",
            "clip0.05", "clip0.1", "clip0.2",
            "tanh0.05", "tanh0.1", "tanh0.2"]


def term_value(p: dict, variant: str) -> float:
    za, e, zc = p["lpA_yc_za"], p["lpA_yc_e"], p["lpA_yc_zc"]
    lift = za - e
    if variant == "l1lift":
        return lift
    if variant == "l1cal_zc":
        return za - max(e, zc)
    if variant.startswith("cap"):
        return min(lift, float(variant[3:]))
    if variant.startswith("clip"):
        c = float(variant[4:])
        return max(-c, min(lift, c))
    if variant.startswith("tanh"):
        c = float(variant[4:])
        return c * math.tanh(lift / c)
    raise ValueError(variant)


def attacked_pair(p: dict, attacker: str, s: float) -> dict:
    q = dict(p)
    if attacker == "self":
        q["lpA_yc_za"] = p["lpA_yc_za"] * (1 - s)
    elif attacker == "blind":
        q["lpA_yc_za"] = p["lpA_yc_za"] * (1 - s)
        q["lpA_yc_zc"] = p["lpA_yc_zc"] * (1 - s)
    elif attacker == "uniform":
        for k in ("lpA_yc_za", "lpA_yc_zc", "lpA_yc_e"):
            q[k] = p[k] * (1 - s)
    else:
        raise ValueError(attacker)
    return q


def mix(p: dict, variant: str, w: float = DEFAULT_L1_WEIGHT) -> float:
    return lambda2(p) + w * term_value(p, variant)


def load_pairs() -> dict[str, dict[str, list[dict]]]:
    """by[miner][turn_id] -> list of pairs."""
    by: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    dropped = 0
    for pf in PAIR_FILES:
        if not pf.exists():
            continue
        for line in open(pf):
            r = json.loads(line)
            if not (r.get("valid") and "pairs" in r):
                continue
            if r["turn_id"] in by[r["miner"]]:
                continue
            good = [p for p in r["pairs"] if all(k in p for k in NEEDED)]
            dropped += len(r["pairs"]) - len(good)
            if good:
                by[r["miner"]][r["turn_id"]] = good
    total = sum(len(ps) for m in by.values() for ps in m.values())
    print(f"loaded {total} pairs / {len(by)} miners "
          f"({dropped} dropped for missing components)")
    return by


def flat(by_turn: dict[str, list[dict]]) -> list[dict]:
    return [p for ps in by_turn.values() for p in ps]


def ranking_table(by: dict[str, dict[str, list[dict]]]) -> None:
    print("\n== unattacked ranking per variant (mix = L2 + 1.0*term) ==")
    print(f"{'variant':10s} {'rho':>7s} {'p':>8s}  genesis  top3")
    for variant in VARIANTS:
        rows = []
        for m, turns in by.items():
            suf = m.removeprefix("king-")
            if suf not in KING_BENCH:
                continue
            rows.append((suf, st.mean(mix(p, variant) for p in flat(turns)),
                         KING_BENCH[suf]))
        rows.sort(key=lambda x: -x[1])
        rho, pval = stats.spearmanr([r[1] for r in rows], [r[2] for r in rows])
        g_rank = next(i + 1 for i, r in enumerate(rows) if r[0] == "genesis")
        top3 = ", ".join(r[0] for r in rows[:3])
        print(f"{variant:10s} {rho:+7.3f} {pval:8.2g}  #{g_rank:<6d} {top3}")


def duel_z(king_turns: dict[str, list[dict]],
           gen_turns: dict[str, list[dict]],
           variant: str, attacker: str, s: float) -> float:
    diffs = []
    for tid in sorted(set(king_turns) & set(gen_turns)):
        mk = st.mean(mix(attacked_pair(p, attacker, s), variant)
                     for p in king_turns[tid])
        mg = st.mean(mix(p, variant) for p in gen_turns[tid])
        diffs.append(mk - mg)
    if len(diffs) < 2:
        return float("nan")
    se = st.stdev(diffs) / math.sqrt(len(diffs))
    return st.mean(diffs) / se if se > 0 else 0.0


def attack_tables(by: dict[str, dict[str, list[dict]]]) -> None:
    gen = by["king-genesis"]
    g_mean = {v: st.mean(mix(p, v) for p in flat(gen)) for v in VARIANTS}
    print("\n== RT-3b: min strength to (a) beat genesis mean / (b) duel z>3 ==")
    print(f"{'variant':10s} {'attacker':8s} " +
          " ".join(f"{k.removeprefix('king-'):>11s}" for k in ATTACK_KINGS))
    for variant in VARIANTS:
        for attacker in ("self", "blind", "uniform"):
            cells = []
            for king in ATTACK_KINGS:
                turns = by.get(king, {})
                ps = flat(turns)
                beat_mean = next(
                    (s for s in STRENGTHS
                     if st.mean(mix(attacked_pair(p, attacker, s), variant)
                                for p in ps) > g_mean[variant]), None)
                beat_duel = next(
                    (s for s in STRENGTHS
                     if duel_z(turns, gen, variant, attacker, s) > 3.0), None)
                fm = f"{beat_mean:.2f}" if beat_mean is not None else "neva"
                fd = f"{beat_duel:.2f}" if beat_duel is not None else "neva"
                cells.append(f"{fm:>5s}/{fd:<5s}")
            print(f"{variant:10s} {attacker:8s} " + " ".join(cells))
    print("\nbaseline duel z at s=0 (honest), per variant:")
    for variant in VARIANTS:
        zs = ", ".join(
            f"{k.removeprefix('king-')}={duel_z(by.get(k, {}), gen, variant, 'self', 0.0):+.2f}"
            for k in ATTACK_KINGS)
        print(f"  {variant:10s} {zs}")


def main() -> None:
    by = load_pairs()
    ranking_table(by)
    attack_tables(by)


if __name__ == "__main__":
    main()
