#!/usr/bin/env python3
"""PE v5 sampling speed-ups that never change the draw (Jacob t1365u: optimize every piece within 5 eval nodes).
- lpt_order: longest-first dispatch. Same indices, same k, same plan seed; the public worker derives each sample's seed from
  sha256(seed:env:index:j), so ONLY the send order changes. Expected length per task = the round's head BASELINE mean output
  tokens for that task (12 head rollouts on exactly these indices; head is public, so the order leaks nothing private).
- split2: two shards by task (all k samples of a task stay on one node), greedy LPT balance on the same expected tokens.
- merge: union of shard per_task (disjoint by construction, asserted) + aggregates recomputed exactly as probe_worker does;
  any missing/partial shard -> raise (the attempt fails; never a partial verdict). Scoring is untouched (pe_score reads per_task)."""
import json, math, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_common import *


def expected_tokens(job):
    """{index: expected output tokens} from the round head baseline; {} if unavailable (falls back to the original order)."""
    rid = job.get("round_id")
    if not rid:
        return {}
    b = rj(HELDOUT / rid / "BASELINE.private.json") or {}
    r = rj(HELDOUT / rid / "ROUND.private.json") or {}
    bl = b.get("per_task") or {}
    if not bl or not r.get("cfg"):
        return {}
    sys.path.insert(0, str(GATE_CODE))
    import baseline_v1 as BV
    out = {}
    for i in job["indices"]:
        c = bl.get(BV.task_of(int(i), r["cfg"])) or {}
        if c.get("n"):
            out[int(i)] = float(c.get("tok") or sum(c.get("L") or [])) / float(c["n"])
    return out


def lpt_order(job, exp=None):
    """Same index multiset, longest expected first; ties and unknown lengths keep the original relative order."""
    idx = [int(i) for i in job["indices"]]
    exp = expected_tokens(job) if exp is None else exp
    if not exp:
        return idx, dict(lpt=False, why="no baseline lengths")
    pos = {i: n for n, i in enumerate(idx)}
    known = sum(1 for i in idx if i in exp)
    out = sorted(idx, key=lambda i: (-exp.get(i, -1.0), pos[i]))
    assert sorted(out) == sorted(idx) and len(out) == len(idx)
    return out, dict(lpt=True, known=known, n=len(idx))


def split2(order, exp):
    """Greedy LPT into 2 shards over the (already longest-first) order. -> [list, list], each keeping longest-first order."""
    sh = [[], []]; load = [0.0, 0.0]
    mean = (sum(exp.values()) / len(exp)) if exp else 1.0
    for i in order:
        k = 0 if load[0] <= load[1] else 1
        sh[k].append(i); load[k] += exp.get(i, mean)
    assert sorted(sh[0] + sh[1]) == sorted(order) and not (set(sh[0]) & set(sh[1]))
    return sh, dict(est_tokens=[round(x) for x in load])


AGG = ("samples", "correct", "wrong", "unresolved", "budget_hits", "errors", "out_tokens")


def merge(results, shards, env_id, k):
    """results[i] = worker result for shards[i]. Raises on any missing/partial/foreign/duplicate task."""
    if len(results) != len(shards) or any(r is None for r in results):
        raise RuntimeError("shard missing")
    per, prompt = {}, 0
    for r, sh in zip(results, shards):
        if r.get("partial") or int(r.get("samples") or 0) < len(sh) * k:
            raise RuntimeError("shard partial: samples=%s planned=%d" % (r.get("samples"), len(sh) * k))
        st = ((r.get("envs") or {}).get(env_id)) or {}
        pt = st.get("per_task") or {}
        want = {str(i) for i in sh}
        if set(pt) != want:
            raise RuntimeError("shard per_task keys != shard indices (%d vs %d)" % (len(pt), len(want)))
        if set(pt) & set(per):
            raise RuntimeError("duplicate task across shards")
        per.update(pt)
        prompt += int(st.get("prompt_tokens") or 0)
    st = dict(tasks=len(per), prompt_tokens=prompt)
    for a in AGG:
        st[a] = 0
    for v in per.values():
        st["samples"] += v.get("n", 0); st["correct"] += v.get("c", 0); st["wrong"] += v.get("w", 0)
        st["unresolved"] += v.get("u", 0); st["budget_hits"] += v.get("b", 0); st["errors"] += v.get("x", 0); st["out_tokens"] += v.get("tok", 0)
    n = st["samples"]; graded = st["correct"] + st["wrong"] + st["unresolved"]
    p = st["correct"] / graded if graded else None
    pts = [v for v in per.values() if isinstance(v, dict) and v.get("n")]
    se = None
    if p is not None and len(pts) > 1:
        m = len(pts); rs = [v["c"] / v["n"] for v in pts]; mean = sum(rs) / m
        se = math.sqrt(sum((x - mean) ** 2 for x in rs) / (m - 1) / m)
    env = dict(st, pass_rate=p, se_task_clustered=se, mean_out_tokens=(st["out_tokens"] / n) if n else None,
               mean_prompt_tokens=(prompt / st["tasks"]) if st["tasks"] else None, per_task=per)
    wall = max(float(r.get("wall_seconds") or 0) for r in results)
    gen = sum(int(r.get("gen_tokens") or 0) for r in results)
    planned = sum(len(sh) for sh in shards) * k
    r0 = results[0]
    return dict(version=r0.get("version"), checkpoint_id=r0.get("checkpoint_id"), seed=r0.get("seed"),
                started_at=min(r.get("started_at") or 0 for r in results), finished_at=max(r.get("finished_at") or 0 for r in results),
                wall_seconds=wall, gen_tokens=gen, samples=n, tok_per_s=gen / max(wall, 1e-9), rollouts_per_min=n / max(wall, 1e-9) * 60,
                concurrency=sum(int(r.get("concurrency") or 0) for r in results), replicas=sum(int(r.get("replicas") or 0) for r in results),
                requests_per_replica=sum((r.get("requests_per_replica") or [] for r in results), []),
                transient_retries=sum(int(r.get("transient_retries") or 0) for r in results),
                preempted_requests=sum(int(r.get("preempted_requests") or 0) for r in results),
                partial=n < planned, stop_reason=None, planned_samples=planned, envs={env_id: env},
                shards=[dict(tasks=len(sh), samples=r.get("samples"), wall_seconds=r.get("wall_seconds"), gen_tokens=r.get("gen_tokens"),
                             tok_per_s=r.get("tok_per_s")) for r, sh in zip(results, shards)])
