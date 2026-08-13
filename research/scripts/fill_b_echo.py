"""Fill teacher-side B on stored Reason-only n80 artifacts.

For each pair, rebuild the turn prefix from the public corpus and ask the
teacher for lpC(y_A|z_A) and lpC(y_A|∅). No miner sample. No king sample.

Uses the dedicated GLM-Air teacher replicas. Does not touch live eval or
mine-* n80 teachers.

Run from repo root:
    source .venv/bin/activate && source .env
    export HF_TOKEN="${HUGGINGFACE:-$HF_TOKEN}"
    python research/scripts/fill_b_echo.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics as st
from collections import Counter
from pathlib import Path

import httpx

from affine.score import b_gate_pass, leakage, teacher_causality
from evalsrv.corpus import CorpusSync
from evalsrv.vllm_client import ModelPool, Served, VllmModel

ROOT = Path(__file__).resolve().parents[2]
TAU = 0.02
GAMMA = 0.30
TEACHER_REPO = "zai-org/GLM-4.5-Air-FP8"
TEACHER_URLS = [
    "http://152.236.142.232:40000/v1",
    "http://152.236.142.232:40001/v1",
]
JOBS = [
    {
        "label": "thermo",
        "path": ROOT / "mining/experiments/r160-thermopylae-grpo/artifacts/r160_mid50_sim_result_artifact.json",
        "side": "challenger_rows",
        "expect": "fail",
    },
    {
        "label": "guass",
        "path": ROOT / "mining/experiments/r17-coder-rl/artifacts/h135_sim_result_artifact_guass_p2210.json",
        "side": "king_rows",
        "expect": "pass",
    },
]


def _pairs(rows: list[dict]) -> list[tuple[str, dict]]:
    out = []
    for r in rows:
        if not r.get("valid") or "pairs" not in r:
            continue
        tid = r.get("turn_id")
        for p in r["pairs"]:
            out.append((tid, p))
    return out


def _index_by_turn_id(corpus: CorpusSync) -> dict[str, dict]:
    rows = corpus.load_index_rows()
    by = {}
    for row in rows:
        tid = row.get("turn_id") or f"{row.get('traj_id')}:{row.get('turn_idx')}"
        by[str(tid)] = row
    return by


async def _echo_one(teacher: ModelPool, prefix: list[dict], z: str, y: str,
                    tid: str) -> tuple[float, float]:
    za, empty = await asyncio.gather(
        teacher.score_action(prefix, z, y, sticky_key=tid),
        teacher.score_action(prefix, "", y, sticky_key=tid),
    )
    return za["lp_per_byte"], empty["lp_per_byte"]


def _summarize(label: str, filled: list[dict]) -> dict:
    b_vals = [teacher_causality(p) for p in filled]
    b_have = [v for v in b_vals if v is not None]
    flags = [b_gate_pass(p, tau=TAU) for p in filled]
    flags_have = [g for g in flags if g is not None]
    lens = [len((p.get("z_a") or "").strip()) for p in filled]
    leak = st.mean(
        1.0 if leakage(p.get("z_a") or "", p.get("y_a") or "") else 0.0
        for p in filled)
    zs = Counter((p.get("z_a") or "").strip() for p in filled)
    rate = st.mean(1.0 if g else 0.0 for g in flags_have) if flags_have else None
    return {
        "label": label,
        "n": len(filled),
        "mean_b": float(st.mean(b_have)) if b_have else None,
        "median_b": float(st.median(b_have)) if b_have else None,
        "b_pass_rate": rate,
        "would_pass_gamma": None if rate is None else rate >= GAMMA,
        "leak_frac": float(leak),
        "median_len_z": float(st.median(lens)) if lens else None,
        "top_z": [(z[:50] if z else "(empty)", n) for z, n in zs.most_common(5)],
    }


async def fill_job(teacher: ModelPool, corpus: CorpusSync, index: dict[str, dict],
                   job: dict, limit: int | None) -> dict:
    obj = json.loads(job["path"].read_text())
    pairs = _pairs(obj[job["side"]])
    if limit is not None:
        pairs = pairs[:limit]
    missing = 0
    errors = 0
    first_err = ""
    sem = asyncio.Semaphore(32)

    async def one(tid: str, pair: dict) -> dict | None:
        nonlocal missing, errors, first_err
        row = index.get(tid)
        if row is None:
            missing += 1
            return None
        try:
            turns = corpus.materialize_turns([row])
        except Exception as e:
            missing += 1
            if not first_err:
                first_err = f"materialize {tid}: {type(e).__name__}: {e}"
            return None
        prefix = turns[0]["prefix"]
        z = pair.get("z_a") or ""
        y = pair.get("y_a") or ""
        if not y:
            missing += 1
            return None
        async with sem:
            try:
                lp_za, lp_e = await _echo_one(teacher, prefix, z, y, tid)
            except Exception as e:
                errors += 1
                if not first_err:
                    first_err = f"echo {tid}: {type(e).__name__}: {e}"
                return None
        out = dict(pair)
        out["lpC_ya_za"] = lp_za
        out["lpC_ya_e"] = lp_e
        return out

    results = await asyncio.gather(*[one(tid, p) for tid, p in pairs])
    filled = [r for r in results if r is not None]
    if not filled:
        return {
            "label": job["label"],
            "expect": job["expect"],
            "n": 0,
            "n_requested": len(pairs),
            "n_missing_prefix": missing,
            "n_errors": errors,
            "first_err": first_err,
            "matched_expect": False,
            "would_pass_gamma": None,
            "b_pass_rate": None,
        }
    summary = _summarize(job["label"], filled)
    summary["expect"] = job["expect"]
    summary["n_requested"] = len(pairs)
    summary["n_missing_prefix"] = missing
    summary["n_errors"] = errors
    if first_err:
        summary["first_err"] = first_err
    summary["matched_expect"] = (
        summary["would_pass_gamma"] is False if job["expect"] == "fail"
        else summary["would_pass_gamma"] is True)
    return summary


async def main_async(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus = CorpusSync(
        "https://s3.hippius.com/affine-sn120",
        "turns/manifest.json",
        data_dir,
        lazy_chunks=True,
    )
    if not corpus.refresh():
        raise SystemExit("corpus refresh failed")
    print(f"corpus {json.dumps(corpus.info())}", flush=True)
    index = _index_by_turn_id(corpus)
    print(f"index turns={len(index)}", flush=True)

    served = [
        Served(name="teacher", repo=TEACHER_REPO, revision=None,
               port=40000, base_url=url)
        for url in TEACHER_URLS
    ]
    timeout = httpx.Timeout(180.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as http:
        pool = ModelPool([
            VllmModel(s, http, asyncio.Semaphore(24)) for s in served
        ])
        summaries = []
        for job in JOBS:
            print(f"filling {job['label']} from {job['path'].name} ...",
                  flush=True)
            s = await fill_job(pool, corpus, index, job, args.limit)
            summaries.append(s)
            print(json.dumps(s, indent=2), flush=True)

    out = ROOT / "research/results/b_gate_fillin.json"
    out.write_text(json.dumps({
        "tau": TAU, "gamma": GAMMA, "teacher": TEACHER_URLS,
        "jobs": summaries,
    }, indent=2))
    print(f"wrote {out}", flush=True)
    all_ok = all(s.get("matched_expect") for s in summaries)
    print("EXPECTATION: PASS" if all_ok else "EXPECTATION: FAIL", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="max pairs per job (probe)")
    ap.add_argument("--data-dir", type=Path,
                    default=ROOT / "research/data/corpus_b_fill")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
