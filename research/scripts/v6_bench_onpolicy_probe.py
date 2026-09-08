"""On-policy probe: do the duel legs (R, G, A) track bench outcomes? (2026-09-04)

The duel scores a miner on TEACHER-generated prefixes. The advisory bench
runs the miner on its OWN trajectory with a known pass/fail. This probe
scores the legs on the bench trajectories themselves:

  for each king's bench run (25 SWE tasks x 6 reigns), for up to S agent
  steps: prefix x = the messages the agent saw; (z_A, y_A) = its reply split
  by the duel's parser; k=3 teacher refs sampled on x (swarm); then the
  exact evalsrv echoes (miner_terms with action_echo) -> R, G, A, B per step.

Then, WITHIN the same task (the only fair comparison — hard tasks make
everyone look bad): on tasks some reigns solved and others failed, does the
solving run carry higher R / G / A? Reported as per-task mean differences,
a sign count, and a pooled within-task AUC per leg.

    python research/scripts/v6_bench_onpolicy_probe.py --steps 12 \
        --out research/results/v6_bench_onpolicy
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import itertools
import json
import math
import re
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.config import load_config  # noqa: E402
from affine.score import (action_leg, centered_reason, grounding,  # noqa: E402
                          teacher_causality)
from evalsrv.chat import split_rollout  # noqa: E402
from evalsrv.terms import miner_terms, teacher_reference  # noqa: E402
from evalsrv.vllm_client import Served, VllmModel  # noqa: E402

BENCHES = REPO / "affine/state/benches"
ROLES = {"system", "user", "assistant"}
FENCE_FIX = ("```mswea_bash_command", "```bash")
TEMPERATURE, MAX_THOUGHT, MAX_ACTION, K = 0.8, 1024, 768, 3


def norm(text: str) -> str:
    return str(text).replace(*FENCE_FIX)


def load_runs() -> list[dict]:
    idx = [json.loads(l) for l in (BENCHES / "index.jsonl").read_text().splitlines()]
    runs = []
    for row in idx:
        if not row["ok"]:
            continue
        d = json.load(gzip.open(BENCHES / Path(row["key"]).name))
        for iid, inst in d["instances"].items():
            runs.append({"label": row["label"], "repo": row["repo"], "task": iid,
                         "resolved": bool(inst["resolved"]),
                         "exit_status": inst["exit_status"],
                         "messages": inst["messages"]})
    return runs


def steps_of(messages: list[dict], max_prefix_chars: int) -> list[tuple[list[dict], str, str, int]]:
    """(prefix, z_A, y_A, step_index) for every parseable assistant reply."""
    out = []
    hist: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role not in ROLES:
            continue
        content = norm(m.get("content", ""))
        if role == "assistant":
            body = content
            if body.lstrip().startswith("<think>"):
                body = body.lstrip()[len("<think>"):]
            z, y = split_rollout(body, "bash")
            if y and sum(len(x["content"]) for x in hist) <= max_prefix_chars:
                out.append(([dict(x) for x in hist], z, y, len(out)))
        hist.append({"role": role, "content": content})
    return out


def subsample(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    idxs = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idxs]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12, help="max steps per run")
    ap.add_argument("--runs", type=int, default=0, help="limit runs (smoke)")
    ap.add_argument("--max-prefix-chars", type=int, default=120_000)
    ap.add_argument("--swarm", default="http://127.0.0.1:9100/v1")
    ap.add_argument("--turn-concurrency", type=int, default=24)
    ap.add_argument("--out", default="research/results/v6_bench_onpolicy")
    args = ap.parse_args()

    cfg = load_config()
    tau, band_c, band_floor = cfg.duel.tau, cfg.duel.band_c, cfg.duel.band_floor
    runs = load_runs()
    if args.runs:
        runs = runs[: args.runs]
    work = []
    for r in runs:
        for prefix, z, y, si in subsample(steps_of(r["messages"], args.max_prefix_chars), args.steps):
            work.append((r, prefix, z, y, si))
    print(f"{len(runs)} runs, {len(work)} steps to score", flush=True)

    served = Served(name="teacher", repo=cfg.teacher.repo, revision=None,
                    port=0, base_url=args.swarm)
    results: list[dict] = []
    t0 = time.time()
    async with httpx.AsyncClient() as http:
        teacher = VllmModel(served, http, asyncio.Semaphore(args.turn_concurrency * 4))
        sem = asyncio.Semaphore(args.turn_concurrency)

        async def one(r: dict, prefix: list[dict], z: str, y: str, si: int) -> None:
            async with sem:
                try:
                    ref = await teacher_reference(
                        teacher, prefix, K, TEMPERATURE, MAX_THOUGHT, MAX_ACTION,
                        thought_echo=True, action_kind="bash")
                    if len(ref) < 2:
                        return
                    t = await miner_terms(
                        teacher, None, prefix, ref, 1, TEMPERATURE, MAX_THOUGHT,
                        MAX_ACTION, reason_only=True, causality_gate=True,
                        thought_echo=True, action_echo=True, action_kind="bash",
                        rollouts=[(z, y)])
                except Exception as e:  # one bad step never kills the probe
                    print(f"  step failed ({r['label']} {r['task']} #{si}): "
                          f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                    return
                if not t.get("valid"):
                    return
                pairs = t["pairs"]
                R = centered_reason(pairs, tau)
                G = grounding(pairs, band_c, band_floor)
                A = action_leg(pairs, tau)
                results.append({
                    "label": r["label"], "task": r["task"], "resolved": r["resolved"],
                    "exit_status": r["exit_status"], "step": si,
                    "n_refs": len(ref), "len_z": len(z), "len_y": len(y),
                    "R": R, "G": G, "A": A, "B": teacher_causality(pairs[0]),
                    "b": [p["lpC_ya_zc"] - p["lpC_ya_e"] for p in pairs],
                })
                if len(results) % 100 == 0:
                    print(f"  {len(results)} steps scored ({time.time() - t0:.0f}s)", flush=True)

        await asyncio.gather(*[one(*w) for w in work])

    text = report(results, runs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"n_runs": len(runs), "n_steps": len(results), "tau": tau,
         "steps_per_run": args.steps, "rows": results}, indent=1))
    out.with_suffix(".txt").write_text(text)
    print(text)
    return 0


LEGS = ("R", "G", "A", "B", "minRG", "minRGA")


def legs_of(row: dict) -> dict:
    d = {k: row[k] for k in ("R", "G", "A", "B")}
    d["minRG"] = min(row["R"], row["G"])
    d["minRGA"] = min(row["R"], row["G"], row["A"])
    return d


def report(rows: list[dict], runs: list[dict]) -> str:
    lines = [f"on-policy bench probe — {len(rows)} steps from {len(runs)} runs", ""]
    # per-run means
    by_run: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_run[(r["label"], r["task"])].append(legs_of(r))
    run_mean = {k: {leg: st.mean(x[leg] for x in v) for leg in LEGS} for k, v in by_run.items()}
    meta = {(r["label"], r["task"]): r for r in runs}

    lines.append("== per reign: mean over steps ==")
    lines.append(f"{'reign':9} {'steps':>5} " + " ".join(f"{l:>8}" for l in LEGS) + f" {'A<0':>6}")
    for label in sorted({r["label"] for r in rows}):
        grp = [legs_of(r) for r in rows if r["label"] == label]
        lines.append(f"{label:9} {len(grp):5d} " + " ".join(f"{st.mean(g[l] for g in grp):8.4f}" for l in LEGS)
                     + f" {sum(1 for g in grp if g['A'] < 0) / len(grp):6.1%}")
    lines.append("")

    lines.append("== resolved vs unresolved runs, pooled (confounded by task difficulty) ==")
    for leg in LEGS:
        ok = [m[leg] for k, m in run_mean.items() if meta[k]["resolved"]]
        bad = [m[leg] for k, m in run_mean.items() if not meta[k]["resolved"]]
        lines.append(f"{leg:7} resolved mean {st.mean(ok):8.4f} (n={len(ok)})  unresolved {st.mean(bad):8.4f} (n={len(bad)})  diff {st.mean(ok) - st.mean(bad):+.4f}")
    lines.append("")

    lines.append("== WITHIN TASK: tasks some reigns solved and others failed ==")
    tasks = sorted({t for _, t in run_mean})
    mixed = [t for t in tasks
             if any(meta[(l, t)]["resolved"] for l, tt in run_mean if tt == t)
             and any(not meta[(l, t)]["resolved"] for l, tt in run_mean if tt == t)]
    lines.append(f"mixed-outcome tasks with scored steps: {len(mixed)}")
    per_task_diff: dict[str, list[float]] = defaultdict(list)
    auc_wins: dict[str, list[float]] = defaultdict(list)
    for t in mixed:
        solved = [run_mean[(l, tt)] for (l, tt) in run_mean if tt == t and meta[(l, tt)]["resolved"]]
        failed = [run_mean[(l, tt)] for (l, tt) in run_mean if tt == t and not meta[(l, tt)]["resolved"]]
        for leg in LEGS:
            per_task_diff[leg].append(st.mean(s[leg] for s in solved) - st.mean(f[leg] for f in failed))
            for s, f in itertools.product(solved, failed):
                auc_wins[leg].append(1.0 if s[leg] > f[leg] else 0.5 if s[leg] == f[leg] else 0.0)
    lines.append(f"{'leg':7} {'mean(solved−failed)':>20} {'tasks with diff>0':>18} {'within-task AUC':>16} {'pairs':>6}")
    for leg in LEGS:
        d = per_task_diff[leg]
        if not d:
            lines.append(f"{leg:7} (no mixed-outcome tasks scored)")
            continue
        se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        lines.append(f"{leg:7} {st.mean(d):+12.4f} ±{se:.4f} {sum(1 for x in d if x > 0):9d}/{len(d):<8d} "
                     f"{st.mean(auc_wins[leg]):16.3f} {len(auc_wins[leg]):6d}")
    lines.append("")
    lines.append("AUC = P(solving run's leg > failing run's leg) over all solved/failed run pairs of the")
    lines.append("same task; 0.5 = the leg does not see the outcome, 1.0 = it always does.")
    lines.append("")

    lines.append("== step-level: submitted-and-resolved vs each failure mode (pooled, coarse) ==")
    by_status: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = "Resolved" if r["resolved"] else str(r["exit_status"])
        by_status[key].append(legs_of(r))
    for key, grp in sorted(by_status.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"{key:28} n={len(grp):5d} " + " ".join(f"{l}={st.mean(g[l] for g in grp):+.4f}" for l in ("R", "G", "A")))
    lines.append("")
    lines.append("== does A pay short actions here too? ==")
    ly = [r["len_y"] for r in rows]
    med = st.median(ly)
    short = [r["A"] for r in rows if r["len_y"] <= med]
    long_ = [r["A"] for r in rows if r["len_y"] > med]
    lines.append(f"A median: actions ≤{med:.0f} chars {st.median(short):.4f}  vs longer {st.median(long_):.4f}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
