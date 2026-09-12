#!/usr/bin/env python
"""E4 — usefulness-gate calibration from the continuation results.

Reads the run_states.py results (results.jsonl: one row per continuation,
state_id = <rollout>:<turn>:<arm>:r<rep>) and the E4 states index, and for
every probe turn computes
  solved_nohint / n, solved_hinted / n, lift = rate(hinted) − rate(nohint),
  survive = lift > 0                     (the usefulness gate)
then the survival rate per harness / group / grounding, and the correlation
of survival with the E1 signal on the same turns (turn_conditions.jsonl of
analyze.py: hinted-condition a_sd, R of the teacher held-out, paired d).

  python e4_analyze.py --e4-dir RUN/e4 --results RUN/e4/out/results.jsonl \
      --turn-conditions RUN/analysis/turn_conditions.jsonl --out RUN/analysis
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path


def rate(xs: list[bool]) -> float:
    return sum(1 for x in xs if x) / len(xs) if xs else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    if len(x) < 4:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4-dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--turn-conditions")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    states = {}
    for line in open(Path(args.e4_dir) / "states.jsonl"):
        s = json.loads(line)
        states[s["state_id"]] = s
    results = [json.loads(l) for l in open(args.results)] if Path(args.results).exists() else []
    by_turn: dict[str, dict] = collections.defaultdict(lambda: {"nohint": [], "hinted": [], "errored": 0})
    for r in results:
        s = states.get(r["state_id"])
        if s is None:
            continue
        t = by_turn[s["probe_turn_id"]]
        t["harness"] = s["harness"]
        t["group"] = s["probe_group"]
        t["grounded"] = s.get("hint_grounded")
        t["depth"] = s["turn_idx"]
        if r.get("status") != "ok":
            t["errored"] += 1
            continue
        solved = (r.get("outcome") == "solved")
        (t["nohint"] if s["arm"] == "nohint" else t["hinted"]).append(solved)
    rows = []
    for tid, t in by_turn.items():
        if not t["nohint"] or not t["hinted"]:
            continue
        r0, r1 = rate(t["nohint"]), rate(t["hinted"])
        rows.append({"turn_id": tid, "harness": t["harness"], "group": t["group"],
                     "grounded": t["grounded"], "depth": t["depth"],
                     "n_nohint": len(t["nohint"]), "n_hinted": len(t["hinted"]),
                     "solved_nohint": r0, "solved_hinted": r1, "lift": r1 - r0,
                     "survive": r1 > r0, "errored": t["errored"]})
    with open(out / "e4_turns.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lines = ["# E4 — usefulness gate\n",
             f"turns with both arms: {len(rows)}; continuations: {len(results)}\n",
             "| slice | turns | solved nohint | solved hinted | mean lift | survive (lift>0) | hurt (lift<0) |",
             "|---|---|---|---|---|---|---|"]

    def slice_row(name, sub):
        if not sub:
            return
        lines.append(f"| {name} | {len(sub)} | {st.mean(r['solved_nohint'] for r in sub):.2f} | "
                     f"{st.mean(r['solved_hinted'] for r in sub):.2f} | {st.mean(r['lift'] for r in sub):+.2f} | "
                     f"{rate([r['survive'] for r in sub]):.2f} | {rate([r['lift'] < 0 for r in sub]):.2f} |")
    slice_row("all", rows)
    for h in sorted({r["harness"] for r in rows}):
        slice_row(f"harness {h}", [r for r in rows if r["harness"] == h])
    for g in sorted({r["group"] for r in rows}):
        slice_row(f"group {g}", [r for r in rows if r["group"] == g])
    for gr in (True, False):
        slice_row(f"grounded={gr}", [r for r in rows if r["grounded"] is gr])
    # correlation with the E1 signal
    if args.turn_conditions and Path(args.turn_conditions).exists():
        tc = collections.defaultdict(dict)
        for line in open(args.turn_conditions):
            t = json.loads(line)
            tc[t["turn_id"]][t["cond"]] = t
        lines.append("\n## Survival vs E1 signal (same turns, condition ds_fact)\n")
        lines.append("| E1 metric | n | Spearman(metric, lift) | mean metric survive | mean metric not |")
        lines.append("|---|---|---|---|---|")
        for label, getter in (
            ("a_sd teacher_heldout (hinted refs)", lambda t: t["miners"].get("teacher_heldout", {}).get("a_sd")),
            ("R teacher_heldout (hinted refs)", lambda t: t["miners"].get("teacher_heldout", {}).get("R")),
            ("R king_live (hinted refs)", lambda t: t["miners"].get("king_live", {}).get("R")),
            ("d teacher_heldout − king_live", lambda t: (t["miners"].get("teacher_heldout", {}).get("score") or 0)
                                                    - (t["miners"].get("king_live", {}).get("score") or 0)
                                                    if t["miners"].get("teacher_heldout") and t["miners"].get("king_live") else None),
            ("identical refs", lambda t: 1.0 if t["identical"] else 0.0),
            ("ref yield", lambda t: t["n_valid"] / 3),
        ):
            xs, ys, s1, s0 = [], [], [], []
            for r in rows:
                t = tc.get(r["turn_id"], {}).get("ds_fact")
                if not t:
                    continue
                v = getter(t)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                xs.append(v); ys.append(r["lift"])
                (s1 if r["survive"] else s0).append(v)
            if xs:
                lines.append(f"| {label} | {len(xs)} | {spearman(xs, ys):+.2f} | "
                             f"{(st.mean(s1) if s1 else float('nan')):.4f} | {(st.mean(s0) if s0 else float('nan')):.4f} |")
    (out / "e4_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
