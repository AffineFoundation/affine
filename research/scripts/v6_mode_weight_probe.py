"""Mode-concentration M as a per-turn weight — does it raise duel power? (2026-09-05)

M_t = mean over ordered ref pairs i≠j of  lpC(y_C^j | x, z_C^i) − lpC(y_C^j | x, ∅):
how much the teacher's thoughts agree with each other's actions on turn t.
Low M = the teacher is undecided on this turn = the noise floor every leg
inherits (yesterday: even the teacher's own action scores A<0 on ~20% of
turns because no other ref shares its mode).

Purely analytic once the k(k−1) cross echoes exist: recompute the paired
duel statistics with per-turn weights w(M) and compare margin / SE / z with
the unweighted verdict. Also applied to the teacher-own-vs-miner A contrast
from v6_action_leg_probe.json (joined on turn_id).

    python research/scripts/v6_mode_weight_probe.py --record affine/state/evals/chal-00248.json.gz \
        --out research/results/v6_mode_weight
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import shutil
import statistics as st
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.config import load_config  # noqa: E402
from affine.score import turn_score  # noqa: E402
from evalsrv.corpus import CorpusSync  # noqa: E402
from evalsrv.vllm_client import Served, VllmModel  # noqa: E402


def wstats(diffs: list[float], w: list[float]) -> tuple[float, float, float, float]:
    """Weighted mean, its SE (Σw²(d−m)²)^½/Σw, z, and effective n."""
    sw = sum(w)
    m = sum(wi * di for wi, di in zip(w, diffs)) / sw
    se = math.sqrt(sum((wi * (di - m)) ** 2 for wi, di in zip(w, diffs))) / sw
    n_eff = sw ** 2 / sum(wi * wi for wi in w)
    return m, se, (m / se if se else float("nan")), n_eff


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--aprobe", default="research/results/v6_action_leg_probe.json")
    ap.add_argument("--swarm", default="http://127.0.0.1:9100/v1")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--out", default="research/results/v6_mode_weight")
    args = ap.parse_args()

    cfg = load_config()
    d = json.load(gzip.open(args.record))
    dp = d["verdict"]["duel_params"]
    tau, band_c, band_floor, mode = dp["tau"], dp["band_c"], dp["band_floor"], dp["score_mode"]
    refs = d["teacher_refs"]
    k_by = {r["turn_id"]: r for r in d["king_rows"] if r.get("valid")}
    c_by = {r["turn_id"]: r for r in d["challenger_rows"] if r.get("valid")}
    tids = [t for t in d["turn_ids"] if t in k_by and t in c_by and len(refs.get(t) or []) >= 2]

    scratch = Path("/tmp/v6_probe_corpus")
    if not scratch.exists():
        shutil.copytree(REPO / "affine/state/corpus_cache", scratch)
    corpus = CorpusSync(cfg.dataset.corpus_base_url, cfg.dataset.manifest_key, scratch, lazy_chunks=True)
    rows = {r["turn_id"]: r for r in corpus.load_index_rows()}
    turns = corpus.materialize_turns([rows[t] for t in tids])
    prefix_by = {t: rec["prefix"] for t, rec in zip(tids, turns)}

    served = Served(name="teacher", repo=cfg.teacher.repo, revision=None, port=0, base_url=args.swarm)
    M: dict[str, float] = {}
    t0 = time.time()
    async with httpx.AsyncClient() as http:
        model = VllmModel(served, http, asyncio.Semaphore(args.concurrency))
        sem = asyncio.Semaphore(args.concurrency // 3)

        async def one(tid: str) -> None:
            async with sem:
                ref = refs[tid]
                prefix = prefix_by[tid]
                pairs = [(i, j) for i in range(len(ref)) for j in range(len(ref)) if i != j]
                try:
                    res = await asyncio.gather(*[
                        model.score_action(prefix, ref[i]["z"], ref[j]["y"]) for i, j in pairs])
                except Exception as e:
                    print(f"  {tid} failed: {type(e).__name__}", flush=True)
                    return
                M[tid] = st.mean(r["lp_per_byte"] - ref[j]["lp_empty"] for r, (_, j) in zip(res, pairs))
                if len(M) % 100 == 0:
                    print(f"  {len(M)}/{len(tids)} ({time.time() - t0:.0f}s)", flush=True)

        await asyncio.gather(*[one(t) for t in tids])

    # Paired duel diffs on the same turns.
    diffs, ms = [], []
    for t in tids:
        if t not in M:
            continue
        rc = turn_score(c_by[t]["pairs"], tau, mode, band_c, band_floor)
        rk = turn_score(k_by[t]["pairs"], tau, mode, band_c, band_floor)
        diffs.append(rc - rk); ms.append(M[t])

    L = [f"mode-concentration weighting — {len(diffs)} paired turns of {args.record.split('/')[-1]}", ""]
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    L.append(f"M: med {st.median(ms):+.4f}  p10 {q(ms,.1):+.4f}  p90 {q(ms,.9):+.4f}  M<0 on {sum(1 for m in ms if m<0)/len(ms):.1%} of turns")
    L.append("")
    # Does |diff| shrink with M? (noise vs ambiguity)
    order = sorted(range(len(ms)), key=lambda i: ms[i])
    for name, sl in (("lowest-M quartile", order[: len(order) // 4]), ("highest-M quartile", order[-(len(order) // 4):])):
        dd = [diffs[i] for i in sl]
        L.append(f"{name:20} n={len(dd):4d}  mean diff {st.mean(dd):+.5f}  sd {st.pstdev(dd):.4f}")
    L.append("")
    L.append("== crown statistic under per-turn weights w(M) ==")
    L.append(f"{'weighting':34} {'margin':>9} {'SE':>8} {'z':>6} {'n_eff':>7}")
    schemes = {
        "unweighted (live rule)": [1.0] * len(ms),
        "drop M<0": [1.0 if m >= 0 else 0.0 for m in ms],
        "drop lowest 20% M": [1.0 if m >= q(ms, .2) else 0.0 for m in ms],
        "w = clip(M,0,∞)": [max(m, 0.0) for m in ms],
        "w = rank(M)": [0.0] * len(ms),
        "w = sigmoid((M−med)/sd)": [1 / (1 + math.exp(-(m - st.median(ms)) / st.pstdev(ms))) for m in ms],
    }
    for rank, i in enumerate(order):
        schemes["w = rank(M)"][i] = (rank + 1) / len(ms)
    for name, w in schemes.items():
        if sum(w) <= 0:
            continue
        m, se, z, ne = wstats(diffs, w)
        L.append(f"{name:34} {m:+9.5f} {se:8.5f} {z:+6.2f} {ne:7.0f}")
    L.append("")
    # Same weights on the teacher-own-vs-miner A contrast (like-for-like, refs 2 and 3).
    try:
        ap_rows = {r["turn_id"]: r for r in json.load(open(args.aprobe))["rows"]}
        def lme(b):
            mx = max(b); return mx + tau * math.log(st.mean(math.exp((x - mx) / tau) for x in b))
        common = [t for t in tids if t in M and t in ap_rows]
        tdiff = [lme(ap_rows[t]["teacher_own"]["b"]) - lme(ap_rows[t]["king"]["b"][1:]) for t in common]
        mm = [M[t] for t in common]
        L.append(f"== teacher-own − king on the A leg (n={len(common)}), same weights ==")
        for name, wf in (("unweighted", lambda m: 1.0), ("drop M<0", lambda m: 1.0 if m >= 0 else 0.0),
                         ("w = clip(M,0,∞)", lambda m: max(m, 0.0)),
                         ("w = sigmoid((M−med)/sd)", lambda m: 1 / (1 + math.exp(-(m - st.median(mm)) / st.pstdev(mm))))):
            w = [wf(m) for m in mm]
            m_, se, z, ne = wstats(tdiff, w)
            L.append(f"{name:34} {m_:+9.5f} {se:8.5f} {z:+6.2f} {ne:7.0f}")
    except FileNotFoundError:
        L.append("(action-leg probe json not found; skipped)")
    text = "\n".join(L) + "\n"
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps({"record": args.record, "M": M, "diffs": dict(zip([t for t in tids if t in M], diffs))}, indent=0))
    out.with_suffix(".txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
