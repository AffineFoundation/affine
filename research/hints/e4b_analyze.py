#!/usr/bin/env python
"""E4b — recovery lift from hints at the states the unhinted teacher fails.

Inputs: RUN/e4b/states.jsonl (+ state files), run_states.py results
(results.jsonl, one row per continuation; state_id = <rollout>:<turn>:<arm>:r<k>),
optional probe results on the same turns (results.jsonl of run_probe.py) for
the ref-diversity join.

Per state and arm: solved continuations / n. Recovery = any continuation
solved (and, separately, the per-continuation rate). Baseline on these states
is 0 % by construction (one unhinted failure each); the `nohint` control arm
measures the run-to-run noise of that label.
First action differs: the continuation's first action (dialect parser on the
first reply) != the king's action at the state.
Keeps spread: identical-ref fraction of the k = 3 HINTED refs at that turn
(from the probe rows) — the "recovers AND keeps spread" set.

  python e4b_analyze.py --e4b-dir RUN/e4b --results RUN/e4b/out/results.jsonl \
      [--probe RUN/e4b_probe/results.jsonl] --out RUN/analysis
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))
sys.path.insert(0, str(HERE))

from affine import dialects  # noqa: E402

import analyze as A  # noqa: E402

WS = re.compile(r"\s+")
FENCE_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
ARM_ORDER = ["nohint", "ds_fact", "self_fact", "ds_plan", "pivot_action"]


def norm(s: str) -> str:
    return WS.sub(" ", s or "").strip()


def first_action(reply: dict | None, action_kind: str) -> str:
    if not reply:
        return ""
    if reply.get("tool_calls"):
        calls = []
        for tc in reply["tool_calls"]:
            fn = tc.get("function") or tc
            calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
        return json.dumps(calls, sort_keys=True)
    text = FENCE_RE.sub("```bash\n", reply.get("content") or "")
    try:
        return dialects.last_action(text, action_kind)
    except dialects.UnknownDialect:
        return ""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4b-dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--probe")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for line in open(Path(args.e4b_dir) / "states.jsonl"):
        s = json.loads(line)
        jobs[s["state_id"]] = s
    results = [json.loads(l) for l in open(args.results)] if Path(args.results).exists() else []
    per: dict[tuple[str, str], dict] = {}
    for r in results:
        s = jobs.get(r["state_id"])
        if s is None:
            continue
        key = (s["probe_turn_id"], s["arm"])
        d = per.setdefault(key, {"solved": [], "differs": [], "errored": 0, "turns": [], "meta": s})
        if r.get("status") != "ok":
            d["errored"] += 1
            continue
        d["solved"].append(r.get("outcome") == "solved")
        fa = first_action(r.get("first_reply"), s.get("action_kind") or "bash")
        d["differs"].append(norm(fa) != norm(s.get("king_action") or ""))
        d["turns"].append(r.get("n_turns") or 0)
    # probe join: identical-ref fraction of the hinted refs at the turn
    probe: dict[tuple[str, str], dict] = {}
    if args.probe and Path(args.probe).exists():
        for line in open(args.probe):
            r = json.loads(line)
            if r.get("failed"):
                continue
            for cond in r.get("conditions", {}):
                tc = A.turn_condition(r, cond)
                if tc:
                    probe[(r["turn_id"], cond)] = tc
    rows = []
    for (tid, arm), d in per.items():
        if not d["solved"]:
            continue
        m = d["meta"]
        pc = probe.get((tid, "H0" if arm == "nohint" else arm))
        rows.append({
            "turn_id": tid, "arm": arm, "harness": m["harness"], "depth": int(m["turn_idx"]),
            "state_kind": m.get("state_kind"), "grounded": m.get("hint_grounded"),
            "leaks_future": m.get("hint_leaks_future"),
            "n": len(d["solved"]), "n_solved": sum(d["solved"]),
            "recovered_any": any(d["solved"]), "rate": st.mean(1.0 if x else 0.0 for x in d["solved"]),
            "first_differs": st.mean(1.0 if x else 0.0 for x in d["differs"]),
            "mean_turns": st.mean(d["turns"]) if d["turns"] else None,
            "errored": d["errored"],
            "identical_refs": (pc["identical"] if pc else None),
            "ref_valid": (pc["n_valid"] if pc else None),
            "a_sd_heldout": (pc["miners"].get("teacher_heldout", {}).get("a_sd") if pc else None),
        })
    with open(out / "e4b_turns.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def depth_bucket(d: int) -> str:
        return "0-19" if d < 20 else "20-39" if d < 40 else "40-59" if d < 60 else "60+"

    lines = ["# E4b — recovery lift from hints at unhinted-failure states\n",
             f"continuations: {len(results)}; (state, arm) cells with ≥ 1 graded continuation: {len(rows)}\n",
             "Baseline: every state here is a recorded unhinted-teacher failure (0 %). `nohint` = the control re-run.\n",
             "| arm | states | continuations | recovered ≥1 of n | 95 % CI | per-continuation rate | first action ≠ king | mean turns | errored |",
             "|---|---|---|---|---|---|---|---|---|"]

    def arm_row(label, sub):
        if not sub:
            return
        k = sum(1 for r in sub if r["recovered_any"])
        lo, hi = wilson(k, len(sub))
        lines.append(f"| {label} | {len(sub)} | {sum(r['n'] for r in sub)} | {k / len(sub):.2f} | [{lo:.2f}, {hi:.2f}] | "
                     f"{st.mean(r['rate'] for r in sub):.2f} | {st.mean(r['first_differs'] for r in sub):.2f} | "
                     f"{st.mean(r['mean_turns'] for r in sub if r['mean_turns'] is not None):.1f} | {sum(r['errored'] for r in sub)} |")
    for arm in ARM_ORDER:
        arm_row(arm, [r for r in rows if r["arm"] == arm])
    for split_name, key in (("harness", "harness"), ("depth", None), ("state kind", "state_kind"), ("grounded", "grounded")):
        lines.append(f"\n## By {split_name}\n")
        lines.append("| arm | " + split_name + " | states | recovered ≥1 | 95 % CI | per-continuation | first ≠ king |")
        lines.append("|---|---|---|---|---|---|---|")
        vals = sorted({(depth_bucket(r["depth"]) if key is None else r[key]) for r in rows}, key=str)
        for arm in ARM_ORDER:
            for v in vals:
                sub = [r for r in rows if r["arm"] == arm and (depth_bucket(r["depth"]) if key is None else r[key]) == v]
                if not sub:
                    continue
                k = sum(1 for r in sub if r["recovered_any"])
                lo, hi = wilson(k, len(sub))
                lines.append(f"| {arm} | {v} | {len(sub)} | {k / len(sub):.2f} | [{lo:.2f}, {hi:.2f}] | "
                             f"{st.mean(r['rate'] for r in sub):.2f} | {st.mean(r['first_differs'] for r in sub):.2f} |")
    if probe:
        lines.append("\n## Recovers AND keeps spread (hinted k = 3 refs at the same turn)\n")
        lines.append("| arm | states with probe | recovered | recovered & refs not identical | recovered & identical | not recovered & identical | identical frac (all) |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm in ARM_ORDER:
            sub = [r for r in rows if r["arm"] == arm and r["identical_refs"] is not None]
            if not sub:
                continue
            rec = [r for r in sub if r["recovered_any"]]
            lines.append(f"| {arm} | {len(sub)} | {len(rec)} | {sum(1 for r in rec if not r['identical_refs'])} | "
                         f"{sum(1 for r in rec if r['identical_refs'])} | {sum(1 for r in sub if not r['recovered_any'] and r['identical_refs'])} | "
                         f"{st.mean(1.0 if r['identical_refs'] else 0.0 for r in sub):.2f} |")
    (out / "e4b_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
