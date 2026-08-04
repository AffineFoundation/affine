"""RT-1b: prior-bank baseline defense against paraphrase/generic-prior attacks.

Instead of Λ2 = lpC(y_C|z) − lpC(y_C|∅), score
  Λ2_bank = lpC(y_C|z) − max_k lpC(y_C|prior_k)
where prior_k is a fixed published bank of generic SWE thoughts (empty included).
A miner that submits a bank prior (or paraphrase thereof) scores ≤ 0; only thoughts
that beat every prior on the teacher's action earn positive score.

Also records a specificity feature: count of unique code-like identifiers in z.

Usage:
  python -m harness.rt1b_priorbank --n-turns 40 --out /root/results/rt1b_priorbank.jsonl
"""

import argparse
import asyncio
import json
import pathlib
import re

import httpx

from .client import VllmModel
from .config import TEACHER, RunCfg
from .runner import load_turns, turn_id
from .priors import PRIOR_BANK
from .terms import EMPTY_THOUGHTS, teacher_reference


def n_idents(z: str) -> int:
    return len(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", z)))


async def score_bank(teacher: VllmModel, prefix, ref, z: str) -> dict:
    if not ref:
        return {"valid": False}
    # score z and every prior on each y_C
    keys = ["z"] + list(PRIOR_BANK)
    zs = [z] + list(PRIOR_BANK.values())
    tasks = [teacher.score_action(prefix, zz, r["y"])
             for r in ref for zz in zs]
    res = await asyncio.gather(*tasks)
    n_ref, n_z = len(ref), len(zs)
    # res layout: for each ref i, n_z scores
    per_key = {k: [] for k in keys}
    lifts_bank = []
    for i in range(n_ref):
        lps = [res[i * n_z + j]["lp_per_byte"] for j in range(n_z)]
        for k, lp in zip(keys, lps):
            per_key[k].append(lp)
        lifts_bank.append(lps[0] - max(lps[1:]))  # z minus best prior
    out = {
        "valid": True,
        "L2_bank": sum(lifts_bank) / len(lifts_bank),
        "L2_empty": (sum(per_key["z"]) - sum(per_key["empty"])) / n_ref,
        "best_prior": max(
            ((sum(v) / n_ref, k) for k, v in per_key.items() if k != "z"),
        )[1],
        "n_idents": n_idents(z),
        "z_len": len(z),
    }
    return out


async def run(args):
    rc = RunCfg()
    sem = asyncio.Semaphore(rc.max_concurrency)
    async with httpx.AsyncClient() as http:
        teacher = VllmModel(TEACHER, http, sem)
        # Load honest z from a previous v2 run if provided; else use ref z_C as
        # a strong-thought baseline and PRIOR_BANK['para_ls'] as the attack.
        honest_z = {}
        if args.honest_from:
            for line in open(args.honest_from):
                r = json.loads(line)
                if r.get("valid") and "pairs" in r and r.get("miner") == args.honest_miner:
                    # mean isn't needed; keep first z per turn
                    honest_z[r["turn_id"]] = r["pairs"][0].get("z_a", "")

        turns = load_turns(args.turns, args.n_turns)
        outp = pathlib.Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(outp, "a")
        ref_cache = {}
        if args.ref_cache and pathlib.Path(args.ref_cache).exists():
            with open(args.ref_cache) as f:
                for line in f:
                    r = json.loads(line)
                    ref_cache[r["turn_id"]] = r["ref"]

        for ti, rec in enumerate(turns):
            tid = turn_id(rec)
            prefix = rec["prefix"]
            ref = ref_cache.get(tid)
            if ref is None:
                ref = await teacher_reference(
                    teacher, prefix, rc.n_teacher_samples, rc.temperature,
                    rc.max_thought_tokens, rc.max_action_tokens)
            if not ref:
                continue
            channels = {
                "empty": EMPTY_THOUGHTS,
                "para_ls": PRIOR_BANK["para_ls"],
                "prior_grep": PRIOR_BANK["grep"],
                "teacher_z": ref[0]["z"],
            }
            if tid in honest_z and honest_z[tid]:
                channels["honest"] = honest_z[tid]
            row = {"turn_id": tid,
                   "causality": sum(r["lp_own"] - r["lp_empty"] for r in ref) / len(ref)}
            for cname, z in channels.items():
                row[cname] = await score_bank(teacher, prefix, ref, z)
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            h = row.get("honest", {})
            p = row["para_ls"]
            print(f"[{ti + 1}/{len(turns)}] {tid} "
                  f"bank_honest={h.get('L2_bank', float('nan')):+.4f} "
                  f"bank_para={p.get('L2_bank', float('nan')):+.4f} "
                  f"idents_h={h.get('n_idents', '—')} "
                  f"idents_p={p.get('n_idents', 0)}",
                  flush=True)
        out_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/data/turns_minicoder.jsonl")
    ap.add_argument("--n-turns", type=int, default=40)
    ap.add_argument("--ref-cache", default="/root/results/ref_minicoder.jsonl")
    ap.add_argument("--honest-from", default="/root/results/ekings_w1_v2.jsonl",
                    help="v2 jsonl to pull honest king thoughts from")
    ap.add_argument("--honest-miner", default="king-genesis")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
