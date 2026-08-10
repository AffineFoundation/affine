"""Subnet duel loop skeleton: challenger vs king under production Reason (v3).

Assumes teacher + both miners are already serving. Scores the same N turns
(paired), ranks by mean Reason = lpC(y_C|z_A) − lpC(y_C|∅), dethrones at
k_sigma·SE. Gates/bank/L1lift are recorded as telemetry only.

Usage (on pod):
  python -m harness.duel \
      --king king-genesis:8001 \
      --challenger king-I:8002 \
      --turns /root/data/turns_minicoder.jsonl --n-turns 80 \
      --ref-cache /root/results/ref_minicoder.jsonl \
      --out /root/results/duel_genesis_I.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import pathlib
import statistics as st

import httpx

from .client import VllmModel
from .config import TEACHER, ModelCfg, RunCfg, king_cfg
from .runner import load_turns, turn_id
from .score import (
    duel as score_duel,
    l1_lift,
    reason,
    score_miner,
)
from .terms import miner_terms, teacher_reference


def _parse_miner(spec: str) -> ModelCfg:
    """name:port or name=repo:port."""
    if "=" in spec:
        name, rest = spec.split("=", 1)
        repo, port = rest.rsplit(":", 1)
        return ModelCfg(name=name, repo=repo, port=int(port),
                        family="generic", gpus="?")
    name, port = spec.rsplit(":", 1)
    return king_cfg(name.removeprefix("king-"), int(port), "?")


def _load_rows(path: pathlib.Path) -> tuple[set[tuple[str, str]], dict[str, list[dict]]]:
    """Return done keys and rows grouped by miner from an existing jsonl."""
    done: set[tuple[str, str]] = set()
    by_miner: dict[str, list[dict]] = {}
    if not path.exists():
        return done, by_miner
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["turn_id"], r["miner"])
            done.add(key)
            by_miner.setdefault(r["miner"], []).append(r)
    return done, by_miner


RANKING_FORMULA = "Reason = lpC(y_C|z_A) − lpC(y_C|∅)"


def _miner_audit(rows: list[dict]) -> dict[str, float | None]:
    pairs = [p for r in rows if r.get("valid") and "pairs" in r
             for p in r["pairs"]]
    if not pairs:
        return {"reason": None, "mean_l1lift": None}
    return {
        "reason": st.mean(reason(p) for p in pairs),
        "mean_l1lift": st.mean(l1_lift(p) for p in pairs),
    }


def _mean_bank(rows: list[dict]) -> float | None:
    vals = [r["bank_frac"] for r in rows if r.get("valid") and "bank_frac" in r]
    return sum(vals) / len(vals) if vals else None


def _json_safe(obj):
    """Replace non-finite floats so json.dumps emits strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def build_summary(king_rows: list[dict], chal_rows: list[dict]) -> dict:
    """Aggregate miner scores + duel result for production reporting."""
    ks = score_miner(king_rows, bank_frac=_mean_bank(king_rows))
    cs = score_miner(chal_rows, bank_frac=_mean_bank(chal_rows))
    result = score_duel(
        chal_rows, king_rows,
        challenger_bank_frac=cs.bank_frac,
        king_bank_frac=ks.bank_frac,
    )
    king_audit = _miner_audit(king_rows)
    chal_audit = _miner_audit(chal_rows)
    return {
        "ranking_term": "reason",
        "ranking_formula": RANKING_FORMULA,
        "king": {
            "name": ks.miner, "reason": ks.reason,
            "gate_pass_rate": ks.gate_pass_rate, "bank_frac": ks.bank_frac,
            "n_turns": ks.n_turns,
            "mean_l1lift": king_audit["mean_l1lift"],
        },
        "challenger": {
            "name": cs.miner, "reason": cs.reason,
            "gate_pass_rate": cs.gate_pass_rate, "bank_frac": cs.bank_frac,
            "n_turns": cs.n_turns,
            "mean_l1lift": chal_audit["mean_l1lift"],
        },
        "duel": {
            "margin": result.margin, "se": result.se, "z": result.z,
            "k_sigma": result.k_sigma,
            "challenger_wins": result.challenger_wins,
            "n_paired_turns": result.n_paired_turns,
        },
    }


def verdict_from_summary(summary: dict) -> dict:
    """Compact JSON verdict for subnet drop-in."""
    return {
        "challenger_wins": summary["duel"]["challenger_wins"],
        "margin": summary["duel"]["margin"],
        "z": summary["duel"]["z"],
        "se": summary["duel"]["se"],
        "n_paired_turns": summary["duel"]["n_paired_turns"],
        "ranking_term": summary["ranking_term"],
        "ranking_formula": summary["ranking_formula"],
        "telemetry": {
            "king": {
                "name": summary["king"]["name"],
                "reason": summary["king"]["reason"],
                "gate_pass_rate": summary["king"]["gate_pass_rate"],
                "bank_frac": summary["king"]["bank_frac"],
            },
            "challenger": {
                "name": summary["challenger"]["name"],
                "reason": summary["challenger"]["reason"],
                "gate_pass_rate": summary["challenger"]["gate_pass_rate"],
                "bank_frac": summary["challenger"]["bank_frac"],
            },
        },
    }


async def score_side(teacher: VllmModel, miner: VllmModel, turns: list[dict],
                     ref_cache: dict, ref_f, rc: RunCfg, done: set[tuple[str, str]],
                     out_f, existing: list[dict], score_bank: bool,
                     turn_sem: asyncio.Semaphore | None = None) -> list[dict]:
    """Score all turns for one miner. Parallel across turns (bounded by turn_sem)."""
    lock = asyncio.Lock()
    rows = list(existing)
    pending = [rec for rec in turns
               if (turn_id(rec), miner.cfg.name) not in done]
    done_count = len(turns) - len(pending)
    total = len(turns)

    async def one(rec: dict) -> dict | None:
        nonlocal done_count
        tid = turn_id(rec)
        prefix = rec["prefix"]
        ref = ref_cache.get(tid)
        if ref is None:
            async with (turn_sem or asyncio.Semaphore(1)):
                ref = ref_cache.get(tid)
                if ref is None:
                    ref = await teacher_reference(
                        teacher, prefix, rc.n_teacher_samples, rc.temperature,
                        rc.max_thought_tokens, rc.max_action_tokens)
                    ref_cache[tid] = ref
                    if ref_f is not None:
                        ref_f.write(json.dumps({"turn_id": tid, "ref": ref}) + "\n")
                        ref_f.flush()
        if not ref:
            return None
        causality = sum(r["lp_own"] - r["lp_empty"] for r in ref) / len(ref)
        async with (turn_sem or asyncio.Semaphore(1)):
            t = await miner_terms(
                teacher, miner, prefix, ref, rc.n_miner_samples, rc.temperature,
                rc.max_thought_tokens, rc.max_action_tokens, score_bank=score_bank)
        t.update({"turn_id": tid, "miner": miner.cfg.name,
                  "causality": causality})
        async with lock:
            out_f.write(json.dumps(t) + "\n")
            out_f.flush()
            rows.append(t)
            done_count += 1
            print(f"  [{done_count}/{total}] {miner.cfg.name} {tid} "
                  f"L2_bank={t.get('L2_bank', float('nan')):+.4f} "
                  f"bank_frac={t.get('bank_frac', float('nan')):.0%}",
                  flush=True)
        return t

    if pending:
        await asyncio.gather(*[one(rec) for rec in pending])
    return rows


async def run_duel(args) -> tuple[dict, pathlib.Path, pathlib.Path]:
    """Score king + challenger, return (summary, summary_path, rows_path)."""
    rc = RunCfg()
    rc.max_concurrency = args.concurrency
    sem = asyncio.Semaphore(rc.max_concurrency)
    king_cfg_ = _parse_miner(args.king)
    chal_cfg = _parse_miner(args.challenger)
    turns = load_turns(args.turns, args.n_turns)

    ref_cache = {}
    ref_path = pathlib.Path(args.ref_cache) if args.ref_cache else None
    if ref_path:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
    if ref_path and ref_path.exists():
        with open(ref_path) as f:
            for line in f:
                r = json.loads(line)
                ref_cache[r["turn_id"]] = r["ref"]
    ref_f = open(ref_path, "a") if ref_path else None

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows_path = outp.with_suffix(".jsonl")
    done, by_miner = _load_rows(rows_path)
    out_f = open(rows_path, "a")

    turn_sem = asyncio.Semaphore(max(4, min(args.concurrency, 16)))
    try:
        async with httpx.AsyncClient() as http:
            teacher = VllmModel(TEACHER, http, sem)
            king = VllmModel(king_cfg_, http, sem)
            chal = VllmModel(chal_cfg, http, sem)
            print(f"scoring king={king.cfg.name} "
                  f"({len(turns) - sum(1 for t in turns if (turn_id(t), king.cfg.name) in done)} "
                  f"turns remaining)...", flush=True)
            king_rows = await score_side(
                teacher, king, turns, ref_cache, ref_f, rc, done, out_f,
                by_miner.get(king.cfg.name, []), args.score_bank, turn_sem)
            print(f"scoring challenger={chal.cfg.name} "
                  f"({len(turns) - sum(1 for t in turns if (turn_id(t), chal.cfg.name) in done)} "
                  f"turns remaining)...", flush=True)
            chal_rows = await score_side(
                teacher, chal, turns, ref_cache, ref_f, rc, done, out_f,
                by_miner.get(chal.cfg.name, []), args.score_bank, turn_sem)
    finally:
        out_f.close()
        if ref_f is not None:
            ref_f.close()

    return build_summary(king_rows, chal_rows), outp, rows_path


async def run(args):
    summary, outp, rows_path = await run_duel(args)
    safe = _json_safe(summary)
    outp.write_text(json.dumps(safe, indent=2) + "\n")
    print(json.dumps(safe, indent=2), flush=True)
    print(f"wrote {outp} and {rows_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--king", required=True)
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--turns", default="/root/data/turns_minicoder.jsonl")
    ap.add_argument("--n-turns", type=int, default=80)
    ap.add_argument("--ref-cache", default="/root/results/ref_minicoder.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--score-bank", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="compute per-pair L2_bank + miner bank_frac (default: on)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
