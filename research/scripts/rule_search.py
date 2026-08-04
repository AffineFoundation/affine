"""Offline scoring-rule search over v2 E-KINGS data (raw lp components per pair).

For each candidate rule, reports:
  - Spearman/Pearson of per-king aggregate vs swe-rebench truth (10 kings)
  - silent-miner attack value: the rule recomputed with the thought channel
    replaced by the recorded empty-baseline components (z_A := ""), compared
    against the best honest king (attack succeeds if silent > best honest)

Usage: python scripts/rule_search.py results/ekings_v2_all.jsonl
"""

import argparse
import json
import statistics as st
from collections import defaultdict

from scipy import stats

TRUTH = {
    "king-genesis": 58.2, "king-I": 38.4, "king-XCIX": 39.8, "king-VIII": 36.2,
    "king-XI": 33.6, "king-V": 32.0, "king-XLV": 26.0, "king-XLVI": 13.2,
    "king-LI": 11.6, "king-CI": 12.4,
}


def leakage(z: str, y: str) -> float:
    """Fraction of the action's command text that appears verbatim in z."""
    cmd = y.removeprefix("```bash\n").removesuffix("\n```").strip()
    return 1.0 if cmd and cmd in z else 0.0


# Each rule maps one (pair, turn_causality) to a per-pair score (higher = better miner).
# `p` keys: lpA_yc_za lpC_yc_za lpA_yc_zc lpA_yc_e lpA_ya_za lpC_ya_za lpA_ya_zc
#           lpA_ya_e lpC_ya_e lpC_ya_zc lpC_yc_zc lpC_yc_e z_a y_a
def rule_v1(p, c):
    d1 = p["lpC_yc_zc"] - p["lpA_yc_za"]
    d2 = p["lpC_yc_zc"] - p["lpC_yc_za"]
    d3 = p["lpC_yc_zc"] - p["lpA_yc_zc"]
    d4 = abs(p["lpA_ya_za"] - p["lpC_ya_za"])
    d5 = abs(p["lpA_ya_za"] - p["lpA_ya_zc"])
    return -(d1 + d2 + d3 + 0.5 * d4 + 0.5 * d5)


def rule_d2(p, c):
    return -(p["lpC_yc_zc"] - p["lpC_yc_za"])


def rule_lift2(p, c):
    return p["lpC_yc_za"] - p["lpC_yc_e"]


def rule_action_quality(p, c):
    # teacher-anchored quality of the miner's action under the teacher's plan
    return p["lpC_ya_zc"]


def rule_action_quality_uncond(p, c):
    return p["lpC_ya_e"]


def rule_lift2_plus_aq(p, c):
    return (p["lpC_yc_za"] - p["lpC_yc_e"]) + 0.5 * p["lpC_ya_zc"]


def rule_lift1(p, c):
    # miner-side lift of own thoughts on teacher action
    return p["lpA_yc_za"] - p["lpA_yc_e"]


def rule_l1_plus_l2(p, c):
    return (p["lpA_yc_za"] - p["lpA_yc_e"]) + (p["lpC_yc_za"] - p["lpC_yc_e"])


RULES = {
    "v1 composite": rule_v1,
    "D2 raw": rule_d2,
    "L2 lift": rule_lift2,
    "L1 lift (miner-side)": rule_lift1,
    "L1 + L2": rule_l1_plus_l2,
    "action quality lpC(yA|zC)": rule_action_quality,
    "action quality lpC(yA|e)": rule_action_quality_uncond,
    "L2 + 0.5*AQ": rule_lift2_plus_aq,
}


def silent_variant(p):
    """Component table for the same pair if the miner had submitted z_A = ''."""
    q = dict(p)
    q["lpA_yc_za"] = p["lpA_yc_e"]
    q["lpC_yc_za"] = p["lpC_yc_e"]
    q["lpA_ya_za"] = p["lpA_ya_e"]
    q["lpC_ya_za"] = p["lpC_ya_e"]
    q["z_a"] = ""
    return q


def gate_pass(p, tau=0.02):
    """Miner-side causality gate with leakage mask."""
    if leakage(p.get("z_a", ""), p.get("y_a", "")):
        return False
    return (p["lpA_ya_za"] - p["lpA_ya_e"]) >= tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--gate-tau", type=float, default=0.02)
    args = ap.parse_args()

    per_pairs = defaultdict(list)  # miner -> [(pair, causality)]
    for line in open(args.results):
        r = json.loads(line)
        if not r.get("valid") or "pairs" not in r:
            continue
        for p in r["pairs"]:
            per_pairs[r["miner"]].append((p, r["causality"]))

    miners = sorted((m for m in per_pairs if m in TRUTH), key=lambda m: -TRUTH[m])
    t = [TRUTH[m] for m in miners]
    n_pairs = {m: len(per_pairs[m]) for m in miners}
    print(f"miners: {len(miners)}, pairs per miner: "
          f"{min(n_pairs.values())}–{max(n_pairs.values())}")

    gate_rates = {m: st.mean(1.0 if gate_pass(p, args.gate_tau) else 0.0
                             for p, _ in per_pairs[m]) for m in miners}
    print("\ngate pass-rate (honest kings, tau={:.2f}): ".format(args.gate_tau)
          + " ".join(f"{m.removeprefix('king-')}={gate_rates[m]:.0%}" for m in miners))

    # Miner-level gate: rank on ALL turns (no per-turn selection bias); the
    # gate statistic only *rejects* miners whose thought channel is degenerate
    # (silent/stuffed), it never reshapes an honest miner's scored set.
    silent_rate = st.mean(
        1.0 if gate_pass(silent_variant(p), args.gate_tau) else 0.0
        for m in miners for p, _ in per_pairs[m])
    print(f"\nminer-level gate: honest pass-rates ≥ "
          f"{min(gate_rates.values()):.0%}, synthetic silent miner "
          f"{silent_rate:.0%} → rejected (threshold 30%)")

    print(f"\n{'rule (ranked on all turns)':<34}{'Spearman':>10}{'p':>8}{'Pearson':>9}"
          f"{'silent':>10}")
    for name, fn in RULES.items():
        scores = [st.mean(fn(p, c) for p, c in per_pairs[m]) for m in miners]
        rho, pv = stats.spearmanr(scores, t)
        pr, _ = stats.pearsonr(scores, t)
        verdict = "rejected-by-gate" if silent_rate < 0.3 else "CHECK"
        print(f"{name:<34}{rho:>+10.3f}{pv:>8.4f}{pr:>+9.3f}{'':>6}{verdict}")
        order = sorted(zip(miners, scores), key=lambda x: -x[1])
        print("    " + " > ".join(m.removeprefix("king-") for m, _ in order))


if __name__ == "__main__":
    main()
