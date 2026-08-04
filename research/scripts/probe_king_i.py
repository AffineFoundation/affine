"""Probe the king-I anomaly: why does L2 rank it near/above genesis?

Reads ekings_v2_all.jsonl and reports:
  - per-turn L2 distributions (genesis vs I)
  - whether I's edge is concentrated on low-causality turns
  - qualitative: longest/shortest thoughts, leakage rates, action overlap with teacher
  - per-term breakdown of where I beats genesis

Usage: python scripts/probe_king_i.py results/ekings_v2_all.jsonl
"""

import argparse
import json
import statistics as st
from collections import defaultdict


def l2(p):
    return p["lpC_yc_za"] - p["lpC_yc_e"]


def leakage(z, y):
    cmd = y.removeprefix("```bash\n").removesuffix("\n```").strip()
    return 1.0 if cmd and cmd in z else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()

    by_turn = defaultdict(dict)  # turn_id -> miner -> row
    for line in open(args.results):
        r = json.loads(line)
        if not r.get("valid") or "pairs" not in r:
            continue
        if r["miner"] in ("king-genesis", "king-I"):
            by_turn[r["turn_id"]][r["miner"]] = r

    paired = [(tid, d) for tid, d in by_turn.items()
              if "king-genesis" in d and "king-I" in d]
    print(f"paired turns: {len(paired)}")

    diffs_all, diffs_hi, diffs_lo = [], [], []
    leak_g, leak_i = [], []
    zlen_g, zlen_i = [], []
    for tid, d in paired:
        g, i = d["king-genesis"], d["king-I"]
        c = g["causality"]
        lg = st.mean(l2(p) for p in g["pairs"])
        li = st.mean(l2(p) for p in i["pairs"])
        diffs_all.append(li - lg)
        (diffs_hi if c >= 0.02 else diffs_lo).append(li - lg)
        for p in g["pairs"]:
            leak_g.append(leakage(p.get("z_a", ""), p.get("y_a", "")))
            zlen_g.append(len(p.get("z_a", "")))
        for p in i["pairs"]:
            leak_i.append(leakage(p.get("z_a", ""), p.get("y_a", "")))
            zlen_i.append(len(p.get("z_a", "")))

    def summarize(name, xs):
        if not xs:
            print(f"{name}: n=0")
            return
        m = st.mean(xs)
        se = st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0
        wins = sum(1 for x in xs if x > 0) / len(xs)
        print(f"{name}: n={len(xs)} mean(I-genesis)={m:+.5f} se={se:.5f} "
              f"z={m / se if se else 0:+.1f} I_wins={wins:.0%}")

    summarize("all turns", diffs_all)
    summarize("causality>=0.02", diffs_hi)
    summarize("causality<0.02", diffs_lo)
    print(f"leakage rate: genesis={st.mean(leak_g):.1%}  I={st.mean(leak_i):.1%}")
    print(f"mean |z|: genesis={st.mean(zlen_g):.0f}  I={st.mean(zlen_i):.0f}")

    # Where does I win hardest? show top-5 turns by L2 gap with sample thoughts.
    ranked = sorted(
        ((st.mean(l2(p) for p in d["king-I"]["pairs"])
          - st.mean(l2(p) for p in d["king-genesis"]["pairs"]),
          tid, d)
         for tid, d in paired),
        reverse=True)
    print("\n=== top 5 turns where I beats genesis on L2 ===")
    for gap, tid, d in ranked[:5]:
        zi = d["king-I"]["pairs"][0].get("z_a", "")[:180].replace("\n", " ")
        zg = d["king-genesis"]["pairs"][0].get("z_a", "")[:180].replace("\n", " ")
        print(f"\n{tid} gap={gap:+.4f} causality={d['king-genesis']['causality']:+.4f}")
        print(f"  I z: {zi}")
        print(f"  G z: {zg}")

    print("\n=== top 5 turns where genesis beats I ===")
    for gap, tid, d in ranked[-5:][::-1]:
        zi = d["king-I"]["pairs"][0].get("z_a", "")[:180].replace("\n", " ")
        zg = d["king-genesis"]["pairs"][0].get("z_a", "")[:180].replace("\n", " ")
        print(f"\n{tid} gap={gap:+.4f} causality={d['king-genesis']['causality']:+.4f}")
        print(f"  I z: {zi}")
        print(f"  G z: {zg}")


if __name__ == "__main__":
    main()
