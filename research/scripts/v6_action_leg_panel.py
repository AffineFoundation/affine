"""Isomorphism test for the staged A leg on the crown chain (2026-09-10).

Question before any min(R,G,A) flip: does the duel margin under the new
rule track real coding performance better than min(R,G) does?

Panel: every stored duel whose challenger has a `swe_rebench_lite_300`
bench score (affine/state/benches/index.jsonl). Those are exactly the
crowned reigns, so each duel's king is the previous chain member and has a
bench score too → Δbench = bench(challenger) − bench(king) per duel.

Per duel (N turns, seeded, both sides valid, k=3 refs): re-materialize the
prefix from the duel's PINNED manifest, echo lpC(y_A | z_C^i) (3 per side)
and lpC(y_A | ∅) (1 per side) on the teacher swarm, and record the summed
lift S_i per ref. Then, on the same sampled turns:
  margin under min(R,G)                       (stored pairs, no new echoes)
  margin under min(R,G,A_fixed)  A_fixed = LME_tau(S_i / L0), L0 = 128
  margin under min(R,G,max(A_fixed, −floor))  floors 0.01 / 0.005
Summary: Spearman(margin, Δbench) per rule across duels; Spearman(mean
A_fixed of a side, that side's bench) across models; which duels each rule
would have crowned (margin > δ) against the sign of Δbench.

    python research/scripts/v6_action_leg_panel.py --n 400 --concurrency 32 \
        --out research/results/v6_action_leg_panel
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

import httpx
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.config import load_config  # noqa: E402
from affine.score import centered_reason, grounding  # noqa: E402
from evalsrv.corpus import CorpusSync  # noqa: E402
from evalsrv.terms import EMPTY_THOUGHTS  # noqa: E402
from evalsrv.vllm_client import Served, VllmModel  # noqa: E402

EVALS = REPO / "affine/state/evals"
BENCH_INDEX = REPO / "affine/state/benches/index.jsonl"
SUITE = "swe_rebench_lite_300"
L0 = 128.0
FLOORS = (None, 0.02, 0.01, 0.005)


def lme(vals: list[float], tau: float) -> float:
    if len(vals) == 1 or tau <= 0:
        return st.mean(vals)
    m = max(vals)
    return m + tau * math.log(st.mean(math.exp((v - m) / tau) for v in vals))


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def not_forfeit(row: dict | None) -> bool:
    return bool(row and row.get("valid") and "pairs" in row)


def load_panel(records: list[str] | None = None) -> list[dict]:
    """Default: every duel whose challenger has a bench score (the crown chain).
    With `records`: exactly those duels, benched or not (loser panel; the
    challenger's bench is None until bench_run.py lands it)."""
    bench = {}
    for line in open(BENCH_INDEX):
        b = json.loads(line)
        if b["suite"] == SUITE and b.get("score") is not None:
            bench[b["revision"]] = b["score"]
    want = set(records) if records else None
    panel = []
    for line in open(EVALS / "index.jsonl"):
        e = json.loads(line)
        if want is not None:
            if e["challenge_id"] not in want:
                continue
        elif e.get("revision") not in bench:
            continue
        d = json.load(gzip.open(EVALS / f"{e['challenge_id']}.json.gz"))
        krev = d["request"]["king_revision"]
        panel.append({"record": e["challenge_id"], "bench_c": bench.get(e["revision"]),
                      "bench_k": bench.get(krev), "king_revision": krev,
                      "challenger_revision": e["revision"], "data": d,
                      "stored_margin": e.get("margin"), "stored_wins": e.get("challenger_wins")})
    panel.sort(key=lambda p: p["record"])
    return panel


def _f(x: float | None, w: int = 7, plus: bool = False) -> str:
    if x is None:
        return f"{'n/a':>{w}}"
    return f"{x:+{w}.2f}" if plus else f"{x:{w}.2f}"


def pinned_corpus(cfg, sha: str) -> CorpusSync:
    scratch = Path(f"/tmp/frontier_probe_corpus/{sha[:12]}")
    for key in (f"corpus/manifests/{sha}.json", f"turns/manifests/{sha}.json"):
        c = CorpusSync(cfg.dataset.corpus_base_url, key, scratch, lazy_chunks=True)
        try:
            if not c.ready:
                c.refresh()
        except Exception as exc:  # noqa: BLE001 — try the other manifest tree
            print(f"    manifest {key} not syncable ({exc}); trying next", flush=True)
            continue
        if c.ready:
            return c
    raise SystemExit(f"pinned manifest {sha[:12]} not syncable")


async def side_echo(model: VllmModel, prefix: list[dict], zs: list[str], action: str) -> dict:
    res = await asyncio.gather(
        *[model.score_action(prefix, z, action) for z in zs],
        model.score_action(prefix, EMPTY_THOUGHTS, action))
    base = res[-1]
    return {"S": [r["sum_lp"] - base["sum_lp"] for r in res[:-1]],
            "b": [r["lp_per_byte"] - base["lp_per_byte"] for r in res[:-1]],
            "n_bytes": base["n_bytes"]}


async def run_record(model: VllmModel, cfg, p: dict, n: int, seed: int,
                     concurrency: int) -> list[dict]:
    d = p["data"]
    v = d["verdict"]; dp = v["duel_params"]
    tau, band_c, band_floor = dp["tau"], dp["band_c"], dp["band_floor"]
    refs = d["teacher_refs"]
    k_by = {r["turn_id"]: r for r in d["king_rows"]}
    c_by = {r["turn_id"]: r for r in d["challenger_rows"]}
    tids = [t for t in d["turn_ids"]
            if len(refs.get(t) or []) == dp["n_teacher_samples"]
            and not_forfeit(k_by.get(t)) and not_forfeit(c_by.get(t))]
    random.Random(seed).shuffle(tids)
    tids = tids[:n]
    corpus = pinned_corpus(cfg, v["slice"]["manifest_sha256"])
    rows = {r["turn_id"]: r for r in corpus.load_index_rows()}
    turns = corpus.materialize_turns([rows[t] for t in tids])
    prefix_by = {t: rec["prefix"] for t, rec in zip(tids, turns)}
    sem = asyncio.Semaphore(max(1, concurrency // 8))
    out: list[dict] = []

    async def one(tid: str) -> None:
        async with sem:
            zs = [r["z"] for r in refs[tid]]
            kp, cp = k_by[tid]["pairs"], c_by[tid]["pairs"]
            ek, ec = await asyncio.gather(
                side_echo(model, prefix_by[tid], zs, kp[0]["y_a"]),
                side_echo(model, prefix_by[tid], zs, cp[0]["y_a"]))
            out.append({
                "record": p["record"], "turn_id": tid,
                "king": {**ek, "R": centered_reason(kp, tau), "G": grounding(kp, band_c, band_floor),
                         "len_y": len(kp[0]["y_a"])},
                "challenger": {**ec, "R": centered_reason(cp, tau), "G": grounding(cp, band_c, band_floor),
                               "len_y": len(cp[0]["y_a"])},
            })

    await asyncio.gather(*[one(t) for t in tids])
    out.sort(key=lambda r: r["turn_id"])
    return out


def a_fixed(side: dict, tau: float, floor: float | None) -> float:
    a = lme([s / L0 for s in side["S"]], tau)
    return a if floor is None else max(a, -floor)


def summarize(panel: list[dict], rows_by: dict[str, list[dict]], delta: float) -> str:
    L: list[str] = []
    L.append(f"A-leg isomorphism panel — {len(panel)} crown-chain duels, {SUITE}, L0={L0:g}, δ={delta}")
    L.append("")
    hdr = (f"{'record':11} {'bench_k':>7} {'bench_c':>7} {'Δbench':>7} {'n':>4} | {'minRG':>8} {'z':>6}"
           + "".join(f" | {'A' + (f'f{f}' if f else ''):>9} {'z':>6}" for f in FLOORS)
           + f" | {'A_k':>7} {'A_c':>7}")
    L.append(hdr)
    per = []
    for p in panel:
        rows = rows_by[p["record"]]
        tau = p["data"]["verdict"]["duel_params"]["tau"]
        n = len(rows)
        d_rg = [min(r["challenger"]["R"], r["challenger"]["G"]) - min(r["king"]["R"], r["king"]["G"]) for r in rows]
        m_rg = st.mean(d_rg); z_rg = m_rg / (st.stdev(d_rg) / math.sqrt(n))
        both = p["bench_k"] is not None and p["bench_c"] is not None
        entry = {"record": p["record"], "bench_k": p["bench_k"], "bench_c": p["bench_c"],
                 "dbench": (p["bench_c"] - p["bench_k"]) if both else None,
                 "n": n, "minRG": m_rg, "z_minRG": z_rg, "rules": {},
                 "stored_margin": p.get("stored_margin"), "stored_wins": p.get("stored_wins")}
        line = (f"{p['record']:11} {_f(p['bench_k'])} {_f(p['bench_c'])} {_f(entry['dbench'], plus=True)} "
                f"{n:4d} | {m_rg:+8.5f} {z_rg:+6.2f}")
        for f in FLOORS:
            dd = [min(r["challenger"]["R"], r["challenger"]["G"], a_fixed(r["challenger"], tau, f))
                  - min(r["king"]["R"], r["king"]["G"], a_fixed(r["king"], tau, f)) for r in rows]
            m = st.mean(dd); z = m / (st.stdev(dd) / math.sqrt(n))
            entry["rules"][str(f)] = {"margin": m, "z": z}
            line += f" | {m:+9.5f} {z:+6.2f}"
        ak = st.mean(a_fixed(r["king"], tau, None) for r in rows)
        ac = st.mean(a_fixed(r["challenger"], tau, None) for r in rows)
        entry["A_k"], entry["A_c"] = ak, ac
        line += f" | {ak:+7.4f} {ac:+7.4f}"
        L.append(line)
        per.append(entry)
    L.append("")
    have = [e for e in per if e["dbench"] is not None]
    L.append(f"Spearman(margin, Δbench) over {len(have)} duels with both sides benched:")
    L.append(f"  min(R,G)                : {spearman([e['minRG'] for e in have], [e['dbench'] for e in have]):+.3f}")
    for f in FLOORS:
        L.append(f"  min(R,G,A{'' if f is None else f' floor {f}'}) : "
                 f"{spearman([e['rules'][str(f)]['margin'] for e in have], [e['dbench'] for e in have]):+.3f}")
    L.append("")
    L.append("Spearman(margin z, Δbench):")
    L.append(f"  min(R,G)                : {spearman([e['z_minRG'] for e in have], [e['dbench'] for e in have]):+.3f}")
    for f in FLOORS:
        L.append(f"  min(R,G,A{'' if f is None else f' floor {f}'}) : "
                 f"{spearman([e['rules'][str(f)]['z'] for e in have], [e['dbench'] for e in have]):+.3f}")
    L.append("")
    # model-level: mean A_fixed of a side vs that side's bench (challenger side of each duel + king of first)
    xs, ys = [], []
    for e in per:
        if e["bench_c"] is not None:
            xs.append(e["A_c"]); ys.append(e["bench_c"])
    if per and per[0]["bench_k"] is not None:
        xs.append(per[0]["A_k"]); ys.append(per[0]["bench_k"])
    L.append(f"Spearman(mean A_fixed of a model on its own duel slice, its bench) over {len(xs)} models: "
             f"{spearman(xs, ys):+.3f}  (slices differ per duel — coarse)")
    L.append("")
    L.append("crown decisions on the sampled turns (margin > δ) vs sign of Δbench:")
    L.append(f"{'record':11} {'Δbench':>7} {'minRG':>6}" + "".join(f" {'A' + (f'f{f}' if f else ''):>8}" for f in FLOORS))
    for e in per:
        L.append(f"{e['record']:11} {_f(e['dbench'], plus=True)} "
                 f"{'crown' if e['minRG'] > delta else '  -  ':>6}"
                 + "".join(f" {'crown' if e['rules'][str(f)]['margin'] > delta else '  -  ':>8}" for f in FLOORS))
    L.append("")
    L.append("NOTE: swe_rebench_lite_300 is the pinned 25-task panel at a 300-step budget (not 300 tasks).")
    L.append("bench SE ≈ sqrt(p(1−p)/25) ≈ 0.10 per model → Δbench SE ≈ 0.14; only |Δbench| ≥ 0.2 is a clear move.")
    return "\n".join(L) + "\n"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--swarm", default="http://127.0.0.1:9100/v1")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default="research/results/v6_action_leg_panel")
    ap.add_argument("--records", nargs="*", help="restrict to these chal ids")
    ap.add_argument("--resume", action="store_true", help="reuse rows already in --out.json")
    args = ap.parse_args()

    cfg = load_config()
    panel = load_panel(args.records)
    if not panel:
        raise SystemExit("empty panel")
    print(f"panel: {len(panel)} duels", flush=True)
    for p in panel:
        print(f"  {p['record']}  king {p['king_revision'][:12]} bench {p['bench_k']}  "
              f"chal {p['challenger_revision'][:12]} bench {p['bench_c']}", flush=True)

    out = Path(args.out)
    rows_by: dict[str, list[dict]] = {}
    if args.resume and out.with_suffix(".json").exists():
        prev = json.load(open(out.with_suffix(".json")))
        for r in prev["rows"]:
            rows_by.setdefault(r["record"], []).append(r)
        print(f"resumed {sum(len(v) for v in rows_by.values())} rows for {len(rows_by)} records", flush=True)

    served = Served(name="teacher", repo=cfg.teacher.repo, revision=None, port=0, base_url=args.swarm)
    t0 = time.time()
    async with httpx.AsyncClient() as http:
        model = VllmModel(served, http, asyncio.Semaphore(args.concurrency))
        for p in panel:
            if p["record"] in rows_by and len(rows_by[p["record"]]) >= args.n:
                continue
            print(f"[{p['record']}] echoing {args.n} turns × 2 sides × 4 …", flush=True)
            rows_by[p["record"]] = await run_record(model, cfg, p, args.n, args.seed, args.concurrency)
            print(f"[{p['record']}] done, {len(rows_by[p['record']])} turns ({time.time() - t0:.0f}s)", flush=True)
            delta = float(cfg.duel.min_margin)
            report = summarize([q for q in panel if q["record"] in rows_by], rows_by, delta)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(".json").write_text(json.dumps({
                "suite": SUITE, "L0": L0, "n_per_record": args.n, "seed": args.seed,
                "panel": [{k: v for k, v in q.items() if k != "data"} for q in panel],
                "rows": [r for rs in rows_by.values() for r in rs]}, indent=1))
            out.with_suffix(".txt").write_text(report)
    # Re-summarize from whatever is on disk (also the --resume, nothing-to-echo path).
    done = [q for q in panel if q["record"] in rows_by]
    if done:
        out.with_suffix(".txt").write_text(summarize(done, rows_by, float(cfg.duel.min_margin)))
    print(open(out.with_suffix(".txt")).read())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
