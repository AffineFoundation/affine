"""v6 action-leg calibration probe (2026-09-04).

Question: what scale does the A leg live on, relative to R and G, and does
it order things the right way? A = tau·LME_i(b_i), b_i = lpC(y_A|z_C^i) −
lpC(y_A|∅), teacher-side, per byte. Not centered (see affine/score.py).

Uses a stored min(R,G) duel record (king/challenger thoughts + actions and
the k teacher refs on every turn), re-materializes each turn prefix from the
corpus, and echoes on the teacher swarm:
  king / challenger    b_i for the side's real action        (k + 1 echoes)
  teacher-own          y_1 scored under z_C^2, z_C^3          (2 + 1)  positive control
  generic              a task-blind action under each z_C^i  (k + 1)  negative control

Reports A distributions, A vs R/G bind fractions, and the paired margin of
the duel under min(R,G) vs min(R,G,A) on the probed turns.

    python research/scripts/v6_action_leg_probe.py --record affine/state/evals/chal-00248.json.gz \
        --n 400 --out research/results/v6_action_leg_probe
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import random
import shutil
import statistics as st
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.config import load_config  # noqa: E402
from affine.score import centered_reason, grounding  # noqa: E402
from evalsrv.corpus import CorpusSync  # noqa: E402
from evalsrv.terms import EMPTY_THOUGHTS  # noqa: E402
from evalsrv.vllm_client import Served, VllmModel  # noqa: E402

GENERIC_ACTION = "```bash\nls -la\n```"


def lme(vals: list[float], tau: float) -> float:
    if len(vals) == 1 or tau <= 0:
        return st.mean(vals)
    m = max(vals)
    return m + tau * math.log(st.mean(math.exp((v - m) / tau) for v in vals))


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


async def side_action_leg(model: VllmModel, prefix: list[dict], ref_thoughts: list[str],
                          action: str, tau: float) -> tuple[float, list[float]]:
    res = await asyncio.gather(
        *[model.score_action(prefix, z, action) for z in ref_thoughts],
        model.score_action(prefix, EMPTY_THOUGHTS, action))
    base = res[-1]["lp_per_byte"]
    b = [r["lp_per_byte"] - base for r in res[:-1]]
    return lme(b, tau), b


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--swarm", default="http://127.0.0.1:9100/v1")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--out", default="research/results/v6_action_leg_probe")
    args = ap.parse_args()

    cfg = load_config()
    d = json.load(gzip.open(args.record))
    dp = d["verdict"]["duel_params"]
    tau, band_c, band_floor = dp["tau"], dp["band_c"], dp["band_floor"]
    refs = d["teacher_refs"]
    k_by = {r["turn_id"]: r for r in d["king_rows"]}
    c_by = {r["turn_id"]: r for r in d["challenger_rows"]}
    tids = [t for t in d["turn_ids"]
            if len(refs.get(t) or []) >= 2
            and not_forfeit(k_by.get(t)) and not_forfeit(c_by.get(t))]
    random.Random(args.seed).shuffle(tids)
    tids = tids[: args.n]

    # Materialize prefixes from a scratch copy of the dashboard's corpus cache.
    scratch = Path("/tmp/v6_probe_corpus")
    if not scratch.exists():
        shutil.copytree(REPO / "affine/state/corpus_cache", scratch)
    corpus = CorpusSync(cfg.dataset.corpus_base_url, cfg.dataset.manifest_key,
                        scratch, lazy_chunks=True)
    rows = {r["turn_id"]: r for r in corpus.load_index_rows()}
    turns = corpus.materialize_turns([rows[t] for t in tids])
    prefix_by = {t: rec["prefix"] for t, rec in zip(tids, turns)}

    served = Served(name="teacher", repo=cfg.teacher.repo, revision=None,
                    port=0, base_url=args.swarm)
    out_rows = []
    t0 = time.time()
    async with httpx.AsyncClient() as http:
        model = VllmModel(served, http, asyncio.Semaphore(args.concurrency))
        sem = asyncio.Semaphore(args.concurrency // 4)

        async def one(tid: str) -> None:
            async with sem:
                prefix = prefix_by[tid]
                ref = refs[tid]
                zs = [r["z"] for r in ref]
                kp, cp = k_by[tid]["pairs"], c_by[tid]["pairs"]
                a_k, b_k = await side_action_leg(model, prefix, zs, kp[0]["y_a"], tau)
                a_c, b_c = await side_action_leg(model, prefix, zs, cp[0]["y_a"], tau)
                a_t, b_t = await side_action_leg(model, prefix, zs[1:], ref[0]["y"], tau)
                a_g, b_g = await side_action_leg(model, prefix, zs, GENERIC_ACTION, tau)
                out_rows.append({
                    "turn_id": tid,
                    "king": {"A": a_k, "b": b_k, "R": centered_reason(kp, tau),
                             "G": grounding(kp, band_c, band_floor),
                             "B": kp[0]["lpC_ya_za"] - kp[0]["lpC_ya_e"],
                             "len_y": len(kp[0]["y_a"])},
                    "challenger": {"A": a_c, "b": b_c, "R": centered_reason(cp, tau),
                                   "G": grounding(cp, band_c, band_floor),
                                   "B": cp[0]["lpC_ya_za"] - cp[0]["lpC_ya_e"],
                                   "len_y": len(cp[0]["y_a"])},
                    "teacher_own": {"A": a_t, "b": b_t, "len_y": len(ref[0]["y"])},
                    "generic": {"A": a_g, "b": b_g},
                })
                if len(out_rows) % 50 == 0:
                    print(f"  {len(out_rows)}/{len(tids)} turns "
                          f"({time.time() - t0:.0f}s)", flush=True)

        await asyncio.gather(*[one(t) for t in tids])

    out_rows.sort(key=lambda r: r["turn_id"])
    report = summarize(out_rows, tau, d["verdict"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"record": args.record, "n": len(out_rows), "tau": tau,
         "swarm": args.swarm, "summary": report["summary"], "rows": out_rows},
        indent=1))
    out.with_suffix(".txt").write_text(report["text"])
    print(report["text"])
    return 0


def not_forfeit(row: dict | None) -> bool:
    return bool(row and row.get("valid") and "pairs" in row)


def summarize(rows: list[dict], tau: float, verdict: dict) -> dict:
    def col(side: str, key: str) -> list[float]:
        return [r[side][key] for r in rows if r[side].get(key) is not None]

    lines = [f"v6 action-leg probe — {len(rows)} turns of {verdict.get('slice', {}).get('n')} "
             f"(duel margin {verdict['margin']:+.5f}, z={verdict['z']:.2f})", ""]
    lines.append(f"{'side':12} {'A med':>8} {'A p10':>8} {'A p90':>8} {'A<0':>6} {'|y| med':>8}")
    for side in ("king", "challenger", "teacher_own", "generic"):
        a = col(side, "A")
        ly = col(side, "len_y") if side != "generic" else [len(GENERIC_ACTION)] * len(a)
        lines.append(f"{side:12} {st.median(a):8.4f} {pct(a, .1):8.4f} {pct(a, .9):8.4f} "
                     f"{sum(1 for x in a if x < 0) / len(a):6.1%} {st.median(ly):8.0f}")
    lines.append("")
    # Ordering checks (paired per turn)
    def frac(f) -> float:
        return sum(1 for r in rows if f(r)) / len(rows)
    lines.append(f"teacher-own A > generic A on {frac(lambda r: r['teacher_own']['A'] > r['generic']['A']):.1%} of turns")
    lines.append(f"king A > generic A        on {frac(lambda r: r['king']['A'] > r['generic']['A']):.1%} of turns")
    lines.append(f"challenger A > generic A  on {frac(lambda r: r['challenger']['A'] > r['generic']['A']):.1%} of turns")
    lines.append("")
    # Scale vs R and G, and bind fractions
    for side in ("king", "challenger"):
        R, G, A = col(side, "R"), col(side, "G"), col(side, "A")
        lines.append(f"{side:12} R med={st.median(R):.4f}  G med={st.median(G):.4f}  A med={st.median(A):.4f}  "
                     f"B med={st.median(col(side, 'B')):.4f}")
        binds = {"R": 0, "G": 0, "A": 0}
        for r in rows:
            s = r[side]
            legs = {"R": s["R"], "G": s["G"], "A": s["A"]}
            binds[min(legs, key=legs.get)] += 1
        n = len(rows)
        lines.append(f"{'':12} binding leg under min(R,G,A): "
                     f"R {binds['R'] / n:.1%}  G {binds['G'] / n:.1%}  A {binds['A'] / n:.1%}")
    lines.append("")
    # Paired margin on probed turns: min(R,G) vs min(R,G,A)
    d_rg = [min(r["challenger"]["R"], r["challenger"]["G"]) - min(r["king"]["R"], r["king"]["G"]) for r in rows]
    d_rga = [min(r["challenger"]["R"], r["challenger"]["G"], r["challenger"]["A"])
             - min(r["king"]["R"], r["king"]["G"], r["king"]["A"]) for r in rows]
    for name, dd in (("min(R,G)", d_rg), ("min(R,G,A)", d_rga)):
        m = st.mean(dd); se = st.stdev(dd) / math.sqrt(len(dd))
        lines.append(f"paired margin (challenger − king) under {name:11}: {m:+.5f}  SE {se:.5f}  z {m / se:+.2f}")
    # Does A correlate with action length? (tiny-command concern)
    for side in ("king", "challenger"):
        pairs_ = [(r[side]["len_y"], r[side]["A"]) for r in rows]
        short = [a for l, a in pairs_ if l <= st.median(x for x, _ in pairs_)]
        long_ = [a for l, a in pairs_ if l > st.median(x for x, _ in pairs_)]
        lines.append(f"{side:12} A med for short actions {st.median(short):.4f} vs long actions {st.median(long_):.4f}")
    text = "\n".join(lines) + "\n"
    return {"text": text, "summary": {"lines": lines}}


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
