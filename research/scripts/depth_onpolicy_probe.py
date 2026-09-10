"""Depth probe: does min(R,G) still see the king at prefix depths D never
covers? (2026-09-09)

Corpus D caps a turn's prefix at 120k chars (~32k tokens). Real agent runs
spend most of their steps far beyond that (Claude Code: 84-92% of steps; the
bench's mini-swe runs: ~30% of steps). This probe scores the advisory bench
trajectories of the teacher and the kings at EVERY depth, with the exact duel
legs (k=3 teacher refs on the prefix, thought/action echoes on the teacher),
and asks three things per depth bin:

  1. Does the machinery work there? (refs sampled, finite lp*, G in band for
     the teacher's own steps — the positive control)
  2. Does the king diverge from the teacher more at depth than at the
     depths D covers? (mean R, G, min(R,G) per bin, king - teacher)
  3. Is a repeated step (an action the run already issued verbatim — the
     king's signature failure at depth) scored lower than a fresh step?

    python research/scripts/depth_onpolicy_probe.py --labels teacher,reign-9 \
        --steps 24 --max-prefix-chars 400000 --out research/results/depth_onpolicy
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.config import load_config  # noqa: E402
from affine.score import centered_reason, grounding, teacher_causality  # noqa: E402
from evalsrv.chat import split_rollout  # noqa: E402
from evalsrv.terms import miner_terms, teacher_reference  # noqa: E402
from evalsrv.vllm_client import Served, VllmModel  # noqa: E402

BENCHES = REPO / "affine/state/benches"
ROLES = {"system", "user", "assistant"}
FENCE_FIX = ("```mswea_bash_command", "```bash")
TEMPERATURE, MAX_THOUGHT, MAX_ACTION, K = 0.8, 1024, 768, 3
BINS = ((0, 40_000), (40_000, 120_000), (120_000, 250_000), (250_000, 10**9))


def norm(text: str) -> str:
    return str(text).replace(*FENCE_FIX)


def bin_of(chars: int) -> str:
    for lo, hi in BINS:
        if lo <= chars < hi:
            return f"{lo // 1000}-{hi // 1000 if hi < 10**9 else 'inf'}k"
    return "?"


def load_runs(labels: set[str], suite: str) -> list[dict]:
    idx = [json.loads(l) for l in (BENCHES / "index.jsonl").read_text().splitlines()]
    runs = []
    for row in idx:
        if not row["ok"] or row["label"] not in labels or row["suite"] != suite:
            continue
        d = json.load(gzip.open(BENCHES / Path(row["key"]).name))
        for iid, inst in d["instances"].items():
            runs.append({"label": row["label"], "task": iid,
                         "resolved": bool(inst["resolved"]),
                         "exit_status": inst["exit_status"],
                         "messages": inst["messages"]})
    return runs


def steps_of(messages: list[dict], max_prefix_chars: int) -> list[dict]:
    """Every parseable assistant reply with its prefix, split (z, y), depth
    in chars and whether y repeats an earlier action of the same run."""
    out = []
    hist: list[dict] = []
    seen_actions: set[str] = set()
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
            depth = sum(len(x["content"]) for x in hist)
            if y and depth <= max_prefix_chars:
                key = " ".join(y.split()).lower()
                out.append({"prefix": [dict(x) for x in hist], "z": z, "y": y,
                            "step": len(out), "depth": depth,
                            "repeat": key in seen_actions})
                seen_actions.add(key)
        hist.append({"role": role, "content": content})
    return out


def subsample(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    if n <= 1:
        return items[-1:]
    idxs = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idxs]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="teacher,reign-9")
    ap.add_argument("--suite", default="swe_rebench_lite_300")
    ap.add_argument("--steps", type=int, default=24, help="max steps per run")
    ap.add_argument("--runs", type=int, default=0, help="limit runs (smoke)")
    ap.add_argument("--max-prefix-chars", type=int, default=400_000)
    ap.add_argument("--swarm", default="http://127.0.0.1:9100/v1")
    ap.add_argument("--turn-concurrency", type=int, default=8)
    ap.add_argument("--out", default="research/results/depth_onpolicy")
    args = ap.parse_args()

    cfg = load_config()
    tau, band_c, band_floor = cfg.duel.tau, cfg.duel.band_c, cfg.duel.band_floor
    runs = load_runs(set(args.labels.split(",")), args.suite)
    if args.runs:
        runs = runs[: args.runs]
    work = []
    for r in runs:
        # keep every repeated step (rare, the object of question 3) plus an
        # even spread of the rest over the run's depth
        steps = steps_of(r["messages"], args.max_prefix_chars)
        keep = [s for s in steps if s["repeat"]]
        rest = [s for s in steps if not s["repeat"]]
        for s in keep + subsample(rest, max(1, args.steps - len(keep))):
            work.append((r, s))
    by_bin: dict[str, int] = defaultdict(int)
    for _, s in work:
        by_bin[bin_of(s["depth"])] += 1
    print(f"{len(runs)} runs, {len(work)} steps to score "
          f"({sum(1 for _, s in work if s['repeat'])} repeats; depth bins "
          f"{dict(by_bin)})", flush=True)

    served = Served(name="teacher", repo=cfg.teacher.repo, revision=None,
                    port=0, base_url=args.swarm)
    results: list[dict] = []
    failures: list[str] = []
    t0 = time.time()
    async with httpx.AsyncClient() as http:
        teacher = VllmModel(served, http, asyncio.Semaphore(args.turn_concurrency * 4))
        sem = asyncio.Semaphore(args.turn_concurrency)

        async def one(r: dict, s: dict) -> None:
            async with sem:
                try:
                    ref = await teacher_reference(
                        teacher, s["prefix"], K, TEMPERATURE, MAX_THOUGHT, MAX_ACTION,
                        thought_echo=True, action_kind="bash")
                    if len(ref) < 2:
                        failures.append(f"refs<2 depth={s['depth']}")
                        return
                    t = await miner_terms(
                        teacher, None, s["prefix"], ref, 1, TEMPERATURE, MAX_THOUGHT,
                        MAX_ACTION, reason_only=True, causality_gate=True,
                        thought_echo=True, action_echo=False, action_kind="bash",
                        rollouts=[(s["z"], s["y"])])
                except Exception as e:  # one bad step never kills the probe
                    failures.append(f"{type(e).__name__} depth={s['depth']}: {str(e)[:100]}")
                    return
                if not t.get("valid"):
                    failures.append(f"invalid depth={s['depth']}")
                    return
                pairs = t["pairs"]
                results.append({
                    "label": r["label"], "task": r["task"], "resolved": r["resolved"],
                    "exit_status": r["exit_status"], "step": s["step"],
                    "depth": s["depth"], "bin": bin_of(s["depth"]), "repeat": s["repeat"],
                    "n_refs": len(ref), "len_z": len(s["z"]), "len_y": len(s["y"]),
                    "ref_len_z": st.mean(len(x["z"]) for x in ref),
                    "R": centered_reason(pairs, tau), "G": grounding(pairs, band_c, band_floor),
                    "B": teacher_causality(pairs[0]),
                    "m": pairs[0]["lpC_za_x"], "t": [p["lpC_zc_x"] for p in pairs],
                })
                if len(results) % 50 == 0:
                    print(f"  {len(results)} steps scored, {len(failures)} failed "
                          f"({time.time() - t0:.0f}s)", flush=True)

        await asyncio.gather(*[one(*w) for w in work])

    text = report(results, failures)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"n_runs": len(runs), "n_steps": len(results), "n_failed": len(failures),
         "tau": tau, "band_c": band_c, "band_floor": band_floor,
         "steps_per_run": args.steps, "max_prefix_chars": args.max_prefix_chars,
         "failures": failures[:200], "rows": results}, indent=1))
    out.with_suffix(".txt").write_text(text)
    print(text)
    return 0


def mean_se(v: list[float]) -> str:
    if not v:
        return "     -            "
    m = st.mean(v)
    se = st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return f"{m:+.4f} ±{se:.4f}"


def report(rows: list[dict], failures: list[str]) -> str:
    lines = [f"depth probe — {len(rows)} steps scored, {len(failures)} failed"]
    fb = defaultdict(int)
    for f in failures:
        fb[f.split(":")[0].split(" depth=")[0]] += 1
    lines.append(f"failures by kind: {dict(fb)}")
    labels = sorted({r["label"] for r in rows})
    bins = [f"{lo // 1000}-{hi // 1000 if hi < 10**9 else 'inf'}k" for lo, hi in BINS]
    lines.append("\n== per label x depth bin (prefix chars): mean ± se ==")
    lines.append(f"{'label':<9} {'bin':<10} {'n':>4} {'G>0':>5}  {'R':>17} {'G':>17} {'minRG':>17} {'B':>17}  {'len_z':>6} {'ref_z':>6}")
    for lab in labels:
        for b in bins:
            rs = [r for r in rows if r["label"] == lab and r["bin"] == b]
            if not rs:
                continue
            lines.append(
                f"{lab:<9} {b:<10} {len(rs):>4} {sum(1 for r in rs if r['G'] > 0) / len(rs):>5.0%}  "
                f"{mean_se([r['R'] for r in rs])} {mean_se([r['G'] for r in rs])} "
                f"{mean_se([min(r['R'], r['G']) for r in rs])} {mean_se([r['B'] for r in rs])}  "
                f"{st.median(r['len_z'] for r in rs):>6.0f} {st.median(r['ref_len_z'] for r in rs):>6.0f}")
    if "teacher" in labels:
        lines.append("\n== king − teacher per bin (unpaired means) ==")
        for lab in labels:
            if lab == "teacher":
                continue
            for b in bins:
                a = [r for r in rows if r["label"] == lab and r["bin"] == b]
                t = [r for r in rows if r["label"] == "teacher" and r["bin"] == b]
                if len(a) < 5 or len(t) < 5:
                    continue
                d = lambda k: st.mean(k(r) for r in a) - st.mean(k(r) for r in t)  # noqa: E731
                lines.append(f"{lab:<9} {b:<10} n={len(a):>3}/{len(t):<3} ΔR {d(lambda r: r['R']):+.4f}  "
                             f"ΔG {d(lambda r: r['G']):+.4f}  ΔminRG {d(lambda r: min(r['R'], r['G'])):+.4f}  "
                             f"Δlen_z {d(lambda r: r['len_z']):+.0f}")
    lines.append("\n== repeated step (action already issued verbatim in this run) vs fresh step ==")
    for lab in labels:
        rep = [r for r in rows if r["label"] == lab and r["repeat"]]
        fresh = [r for r in rows if r["label"] == lab and not r["repeat"]]
        if not rep:
            lines.append(f"{lab:<9} no repeated steps scored")
            continue
        lines.append(f"{lab:<9} repeats n={len(rep):>3}: R {mean_se([r['R'] for r in rep])}  G {mean_se([r['G'] for r in rep])}  "
                     f"minRG {mean_se([min(r['R'], r['G']) for r in rep])}")
        lines.append(f"{'':<9} fresh   n={len(fresh):>3}: R {mean_se([r['R'] for r in fresh])}  G {mean_se([r['G'] for r in fresh])}  "
                     f"minRG {mean_se([min(r['R'], r['G']) for r in fresh])}")
    lines.append("\n== within run: does the king think shorter than the teacher's refs where the refs are long? ==")
    for lab in labels:
        for lo, hi in ((0, 300), (300, 800), (800, 1500), (1500, 10**9)):
            rs = [r for r in rows if r["label"] == lab and lo <= r["ref_len_z"] < hi]
            if len(rs) >= 5:
                lines.append(f"{lab:<9} ref thought {lo:>4}-{hi if hi < 10**9 else 'inf':>4} chars: n={len(rs):>3}  "
                             f"own thought median {st.median(r['len_z'] for r in rs):>5.0f}  "
                             f"ratio {st.median(r['len_z'] / max(1, r['ref_len_z']) for r in rs):.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
