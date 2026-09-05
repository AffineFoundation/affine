"""Observation-dependence legs on the king bench trajectories (2026-09-05).

Adds to the on-policy probe (v6_bench_onpolicy_probe.py):
  O_z = lpC(z_A|x) − lpC(z_A|x⁻)        does the thought READ the last observation?
  O_y = lpC(y_A|x,z_A) − lpC(y_A|x⁻,z_A) does the action?
  O_C^i = lpC(z_C^i|x) − lpC(z_C^i|x⁻)   the teacher's own values (band for O_z)
  N_echo = lpC(y_A|x,∅) − lpC(y_A|x_noassist,∅)  how much of the command's
          predictability is copying earlier assistant turns
  repeat = command body exactly matches an earlier command in the run (label)
where x⁻ drops the last observation (last user message; step 0 has none) and
x_noassist replaces every earlier assistant message with "(elided)".

Step labels for the well-powered test: `repeat` (loop step), `dead_tail`
(one of the last 3 steps of a run that hit a limit), `submitted_run`.
Every leg (R, G, A, B, O_z, O_y, N_echo) is reported on those labels, plus
per-reign drift and the within-task AUC from before.

    python research/scripts/v6_bench_obsdep_probe.py --steps 10 --extra-repeats 5 \
        --out research/results/v6_bench_obsdep
"""

from __future__ import annotations

import argparse
import asyncio
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
sys.path.insert(0, str(REPO / "research" / "scripts"))

from affine.config import load_config  # noqa: E402
from affine.score import (action_leg, centered_reason, grounding,  # noqa: E402
                          teacher_causality)
from evalsrv.terms import EMPTY_THOUGHTS, miner_terms, teacher_reference  # noqa: E402
from evalsrv.vllm_client import Served, VllmModel  # noqa: E402
from v6_bench_onpolicy_probe import load_runs, steps_of  # noqa: E402

TEMPERATURE, MAX_THOUGHT, MAX_ACTION, K = 0.8, 1024, 768, 3
DEAD = {"LimitsExceeded", "ContextWindowExceededError", "RepeatedFormatError"}
LEGS = ("R", "G", "A", "B", "O_z", "O_y", "N_echo", "minRG")


def body(y: str) -> str:
    """Command text inside the fence, whitespace-normalized (for repeat detection)."""
    inner = y.strip().removeprefix("```bash").removesuffix("```").strip()
    return re.sub(r"\s+", " ", inner)


def without_last_obs(prefix: list[dict]) -> list[dict] | None:
    users = [i for i, m in enumerate(prefix) if m["role"] == "user"]
    if len(users) < 2:
        return None  # step 0: the only user message is the task itself
    return prefix[: users[-1]] + prefix[users[-1] + 1:]


def without_assistants(prefix: list[dict]) -> list[dict]:
    return [dict(m, content="(elided)") if m["role"] == "assistant" else m for m in prefix]


def pick_steps(steps, n_even: int, n_rep: int):
    if len(steps) <= n_even:
        chosen = list(range(len(steps)))
    else:
        chosen = sorted({round(i * (len(steps) - 1) / (n_even - 1)) for i in range(n_even)})
    seen_bodies: set[str] = set()
    repeats = []
    for i, (_, _, y, _) in enumerate(steps):
        b = body(y)
        if b in seen_bodies:
            repeats.append(i)
        seen_bodies.add(b)
    extra = [i for i in repeats if i not in chosen][:n_rep]
    return sorted(set(chosen) | set(extra)), set(repeats)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--extra-repeats", type=int, default=5)
    ap.add_argument("--runs", type=int, default=0)
    ap.add_argument("--max-prefix-chars", type=int, default=120_000)
    ap.add_argument("--swarm", default="http://127.0.0.1:9100/v1")
    ap.add_argument("--turn-concurrency", type=int, default=40)
    ap.add_argument("--out", default="research/results/v6_bench_obsdep")
    args = ap.parse_args()

    cfg = load_config()
    tau, band_c, band_floor = cfg.duel.tau, cfg.duel.band_c, cfg.duel.band_floor
    runs = load_runs()
    if args.runs:
        runs = runs[: args.runs]
    work = []
    for r in runs:
        steps = steps_of(r["messages"], args.max_prefix_chars)
        idxs, repeats = pick_steps(steps, args.steps, args.extra_repeats)
        n_steps = len(steps)
        for i in idxs:
            prefix, z, y, si = steps[i]
            work.append((r, prefix, z, y, si, i in repeats, i >= n_steps - 3))
    print(f"{len(runs)} runs, {len(work)} steps ({sum(1 for w in work if w[5])} repeat steps)", flush=True)

    served = Served(name="teacher", repo=cfg.teacher.repo, revision=None, port=0, base_url=args.swarm)
    results: list[dict] = []
    t0 = time.time()
    async with httpx.AsyncClient() as http:
        teacher = VllmModel(served, http, asyncio.Semaphore(args.turn_concurrency * 4))
        sem = asyncio.Semaphore(args.turn_concurrency)

        async def one(r, prefix, z, y, si, is_repeat, is_tail) -> None:
            async with sem:
                try:
                    ref = await teacher_reference(teacher, prefix, K, TEMPERATURE, MAX_THOUGHT,
                                                  MAX_ACTION, thought_echo=True, action_kind="bash")
                    if len(ref) < 2:
                        return
                    t = await miner_terms(teacher, None, prefix, ref, 1, TEMPERATURE, MAX_THOUGHT,
                                          MAX_ACTION, reason_only=True, causality_gate=True,
                                          thought_echo=True, action_echo=True, action_kind="bash",
                                          rollouts=[(z, y)])
                    if not t.get("valid"):
                        return
                    pairs = t["pairs"]
                    xm = without_last_obs(prefix)
                    xna = without_assistants(prefix)
                    tasks = [teacher.score_action(xna, EMPTY_THOUGHTS, y)]
                    if xm is not None:
                        tasks += [teacher.score_thought(xm, z), teacher.score_action(xm, z, y)]
                        tasks += [teacher.score_thought(xm, rf["z"]) for rf in ref]
                    res = await asyncio.gather(*tasks)
                except Exception as e:
                    print(f"  step failed ({r['label']} {r['task']} #{si}): {type(e).__name__}: {str(e)[:100]}", flush=True)
                    return
                p0 = pairs[0]
                n_echo = p0["lpC_ya_e"] - res[0]["lp_per_byte"]
                o_z = o_y = None
                o_c = []
                if xm is not None:
                    o_z = p0["lpC_za_x"] - res[1]["lp_per_byte"]
                    o_y = p0["lpC_ya_za"] - res[2]["lp_per_byte"]
                    o_c = [rf["lp_thought"] - rr["lp_per_byte"] for rf, rr in zip(ref, res[3:])]
                R = centered_reason(pairs, tau)
                G = grounding(pairs, band_c, band_floor)
                results.append({
                    "label": r["label"], "task": r["task"], "resolved": r["resolved"],
                    "exit_status": r["exit_status"], "step": si, "repeat": is_repeat,
                    "dead_tail": bool(is_tail and r["exit_status"] in DEAD),
                    "submitted_run": r["exit_status"] == "Submitted",
                    "len_z": len(z), "len_y": len(y),
                    "R": R, "G": G, "A": action_leg(pairs, tau), "B": teacher_causality(p0),
                    "minRG": min(R, G),
                    "O_z": o_z, "O_y": o_y, "O_C": o_c, "N_echo": n_echo,
                })
                if len(results) % 100 == 0:
                    print(f"  {len(results)} steps ({time.time() - t0:.0f}s)", flush=True)

        await asyncio.gather(*[one(*w) for w in work])

    text = report(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"n_steps": len(results), "tau": tau, "rows": results}, indent=1))
    out.with_suffix(".txt").write_text(text)
    print(text)
    return 0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def _z(a, b):
    a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    return (st.mean(a) - st.mean(b)) / se if se else float("nan")


def _auc(a, b):
    a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
    if not a or not b:
        return float("nan")
    wins = sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in a for y in b)
    return wins / (len(a) * len(b))


def report(rows: list[dict]) -> str:
    L = [f"observation-dependence probe — {len(rows)} steps, {sum(1 for r in rows if r['repeat'])} repeat steps, "
         f"{sum(1 for r in rows if r['dead_tail'])} dead-tail steps", ""]
    L.append("== step-level failure-mode discrimination (higher = leg prefers the healthy step) ==")
    L.append(f"{'leg':7} {'repeat':>9} {'non-repeat':>11} {'z':>6} {'AUC':>6} | {'dead tail':>9} {'submitted':>10} {'z':>6} {'AUC':>6}")
    for leg in LEGS:
        rep = [r[leg] for r in rows if r["repeat"]]; non = [r[leg] for r in rows if not r["repeat"]]
        dead = [r[leg] for r in rows if r["dead_tail"]]; sub = [r[leg] for r in rows if r["submitted_run"] and not r["dead_tail"]]
        L.append(f"{leg:7} {_mean(rep):+9.4f} {_mean(non):+11.4f} {_z(non, rep):+6.2f} {_auc(non, rep):6.3f} | "
                 f"{_mean(dead):+9.4f} {_mean(sub):+10.4f} {_z(sub, dead):+6.2f} {_auc(sub, dead):6.3f}")
    L.append("z = (healthy − failing) / SE; AUC = P(healthy step scores higher than failing step).")
    L.append("")
    L.append("== teacher's own observation-dependence (band reference for O_z) ==")
    oc = [x for r in rows for x in r["O_C"]]
    oz = [r["O_z"] for r in rows if r["O_z"] is not None]
    L.append(f"teacher O_C: med {st.median(oc):+.4f}  p10 {sorted(oc)[len(oc)//10]:+.4f}  p90 {sorted(oc)[9*len(oc)//10]:+.4f}  <0: {sum(1 for x in oc if x<0)/len(oc):.1%}")
    L.append(f"miner   O_z: med {st.median(oz):+.4f}  p10 {sorted(oz)[len(oz)//10]:+.4f}  p90 {sorted(oz)[9*len(oz)//10]:+.4f}  <0: {sum(1 for x in oz if x<0)/len(oz):.1%}")
    L.append("")
    L.append("== per reign (mean over steps): drift test ==")
    labels = sorted({r["label"] for r in rows})
    L.append(f"{'reign':9} {'n':>5} " + " ".join(f"{l:>8}" for l in LEGS) + f" {'repeat%':>8}")
    for lab in labels:
        g = [r for r in rows if r["label"] == lab]
        L.append(f"{lab:9} {len(g):5d} " + " ".join(f"{_mean([r[l] for r in g]):8.4f}" for l in LEGS)
                 + f" {sum(1 for r in g if r['repeat'])/len(g):8.1%}")
    L.append("")
    L.append("== within task (mixed-outcome tasks), run-level means ==")
    by_run = defaultdict(list)
    meta = {}
    for r in rows:
        by_run[(r["label"], r["task"])].append(r); meta[(r["label"], r["task"])] = r["resolved"]
    rm = {k: {l: _mean([x[l] for x in v]) for l in LEGS} for k, v in by_run.items()}
    tasks = sorted({t for _, t in rm})
    mixed = [t for t in tasks if any(meta[k] for k in rm if k[1] == t) and any(not meta[k] for k in rm if k[1] == t)]
    L.append(f"mixed tasks: {len(mixed)}")
    L.append(f"{'leg':7} {'AUC':>6} {'pairs':>6}")
    for l in LEGS:
        wins = []
        for t in mixed:
            s = [rm[k][l] for k in rm if k[1] == t and meta[k]]; f = [rm[k][l] for k in rm if k[1] == t and not meta[k]]
            for a, b in itertools.product(s, f):
                if not (math.isnan(a) or math.isnan(b)):
                    wins.append(1.0 if a > b else 0.5 if a == b else 0.0)
        L.append(f"{l:7} {st.mean(wins) if wins else float('nan'):6.3f} {len(wins):6d}")
    L.append("")
    L.append("== short vs long actions (per-byte bias check) ==")
    med = st.median(r["len_y"] for r in rows)
    for l in ("A", "O_y", "N_echo", "B"):
        s = [r[l] for r in rows if r["len_y"] <= med]; lg = [r[l] for r in rows if r["len_y"] > med]
        L.append(f"{l:7} ≤{med:.0f} chars {_mean(s):+.4f}   longer {_mean(lg):+.4f}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
