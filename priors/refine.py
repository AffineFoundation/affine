"""Refine step of the prior-table design (NOT LIVE; ready to plug into the held-out gate, not wired into gate_service.py).
1. draw S with selection.draw (secret seed + commitment), from a prior-table snapshot (digest recorded privately).
2. baseline: the head runs S `blocks` x `k_block` times (default 4 x 4 = 16 rollouts per task; 12 = 3 x 4 also fine). Block b uses
   plan seed sha256("<seed>:prior-baseline:<b>"); per-request seed = sha256("<plan seed>:<env>:<task>:<j>") like the probe.
3. write-back: the merged baseline cells go into the prior table (source 'baseline') -> they also sharpen the priors.
4. gate bar for miners: candidate C runs k_cand per task on S minus the training batch with block-1 request seeds (CRN, same as
   today's v2), and is scored with heldout.score_ivw unchanged: g_t = rate_C - p_t (p_t = 16-rollout head rate), w_t = 1/(q(1-q)),
   q = (c+1)/(n+2) from the PRIOR cell of task t at draw time (independent of the baseline runs, so no selection bias; plays the
   role of today's selection block) and no 0<c<n filter (every drawn task is kept). sigma_run / bar_z / cap rule unchanged.
   Hajek (1/pi) weighted gain is reported next to it as an analysis number (population-level gain).
score_ivw wants int task keys, so env-prefixed ids are mapped to their position in sorted(S) before the call."""
from __future__ import annotations
import hashlib, importlib.util, os
from . import selection as SEL

HELDOUT = os.environ.get("PRIORS_HELDOUT_PY", "/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code/heldout.py")
_H = {}


def heldout():
    if "m" not in _H:
        spec = importlib.util.spec_from_file_location("heldout_live", HELDOUT); m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m); _H["m"] = m
    return _H["m"]


def baseline_plan_seed(seed, b):
    return hashlib.sha256(("%s:prior-baseline:%d" % (seed, b)).encode()).hexdigest()


def request_seed(plan_seed, env, task_id, j):
    return int(hashlib.sha256(("%s:%s:%s:%d" % (plan_seed, env, task_id, j)).encode()).hexdigest()[:12], 16)


def plan(draw, seed, blocks=4, k_block=4):
    """Pod-runnable baseline plan (task ids + per-block plan seeds); candidates reuse block 1's seeds (CRN)."""
    return dict(version=SEL.VERSION, ids=draw["ids"], blocks=[dict(block=b, k=k_block, plan_seed=baseline_plan_seed(seed, b))
                                                           for b in range(blocks)], cand_plan_seed=baseline_plan_seed(seed, 1))


def merge_blocks(blocks: list[dict]) -> dict:
    """blocks: [{task_id: {n, c, b}}] -> merged {task_id: {n, c, b}}"""
    out = {}
    for blk in blocks:
        for t, v in blk.items():
            o = out.setdefault(t, dict(n=0, c=0, b=0))
            for k in ("n", "c", "b"): o[k] += int(v.get(k) or 0)
    return out


def writeback(table, base: dict, head: str, env_of: dict):
    return table.ingest([dict(task_id=t, env=env_of[t], head=head, n=v["n"], c=v["c"], b=v.get("b", 0)) for t, v in base.items()],
                        source="baseline")


def prior_cells(draw):
    """The prior (draw-time) cell per task as an int cell for heldout.weight: q = (c+1)/(n+2)."""
    return {t: dict(n=int(round(v["n"])), c=int(round(v["s"])), b=int(round(v.get("b") or 0))) for t, v in draw["cells"].items()}


def score(draw, base, cand, batch_ids=(), *, bar_z=1.0, cap_max=0.20, min_kept=100, sigma_run=0.022, n_p_blocks=4, z_inclusive=False):
    H = heldout(); ids = sorted(draw["ids"]); ix = {t: i for i, t in enumerate(ids)}
    sel = {ix[t]: v for t, v in prior_cells(draw).items()}
    b_ = {ix[t]: v for t, v in base.items() if t in ix}; c_ = {ix[t]: v for t, v in cand.items() if t in ix}
    batch = [ix[t] for t in batch_ids if t in ix]
    res = H.score_ivw(sel, b_, c_, list(range(len(ids))), batch, env="prior-set", bar_z=bar_z, z_inclusive=z_inclusive, cap_max=cap_max,
                      min_kept=min_kept, set_size=len(ids), sigma_run=sigma_run, n_p_blocks=n_p_blocks)
    bset = set(batch_ids)
    g = {t: (cand[t]["c"] / cand[t]["n"] - base[t]["c"] / base[t]["n"]) for t in ids
         if t not in bset and cand.get(t, {}).get("n") and base.get(t, {}).get("n")}
    res["hajek_gain"] = SEL.hajek(g, draw["cells"]); res["scoring"] = "ivw-v2 on prior-selected set (prior q weights, no filter)"
    return res
