#!/usr/bin/env python3
"""Deterministic teacher delta bank via direct HF forward passes (batch 1).

vLLM logprobs are non-deterministic across calls for this MoE (batch
composition perturbs expert routing), with a noise floor at the scale of the
deltas themselves. A fixed batch-1 HF forward is bit-reproducible, so
delta = lp(a|h1) - lp(a|h2) carries no cross-call noise.

Shardable: --shard i --num-shards N; resumable per shard output file.
Output format matches build_delta_bank.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, "/root/work")
from gad_common import build_prompt, load_rollout_cache, load_turns  # noqa: E402


@torch.no_grad()
def lp_completion(model, tok, prompt, completion, max_len):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"]
    if not c or len(p) + len(c) > max_len:
        return None
    ids = torch.tensor([p + c], device="cuda")
    logits = model(input_ids=ids).logits[0, len(p) - 1:-1].float()
    tgt = ids[0, len(p):]
    lp = torch.log_softmax(logits, dim=-1).gather(
        1, tgt.unsqueeze(1)).sum().item()
    return lp, len(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--max-len", type=int, default=24576)
    ap.add_argument("--null-cap", type=int, default=0,
                    help="if >0, keep only this many NULL pairs (gate only)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    turns = load_turns(args.turns)
    cache = load_rollout_cache(args.rollouts)

    pairs = [json.loads(x) for x in open(args.pairs)]
    if args.null_cap:
        nulls = [p for p in pairs if p["family"] == "NULL"]
        keep = set(id(p) for p in nulls[:args.null_cap])
        pairs = [p for p in pairs
                 if p["family"] != "NULL" or id(p) in keep]
    pairs = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(json.loads(line)["pair_id"])
            except Exception:
                pass
    pairs = [p for p in pairs if p["pair_id"] not in done
             and p["turn_id"] in turns and p["turn_id"] in cache]
    print(f"shard {args.shard}/{args.num_shards}: {len(pairs)} pairs "
          f"(resumed past {len(done)})", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda").eval()

    out = open(args.out, "a")
    ok = skip = 0
    for p in pairs:
        msgs = turns[p["turn_id"]]
        a = cache[p["turn_id"]][0]["raw"]
        prompt1 = build_prompt(tok, msgs)
        msgs2 = copy.deepcopy(msgs)
        msgs2[p["msg_idx"]] = dict(msgs2[p["msg_idx"]], content=p["edited"])
        prompt2 = build_prompt(tok, msgs2)
        r1 = lp_completion(model, tok, prompt1, a, args.max_len)
        r2 = lp_completion(model, tok, prompt2, a, args.max_len)
        if r1 is None or r2 is None:
            skip += 1
            continue
        n = max(min(r1[1], r2[1]), 1)
        out.write(json.dumps({
            "pair_id": p["pair_id"], "turn_id": p["turn_id"],
            "family": p["family"], "msg_idx": p["msg_idx"],
            "lp1": round(r1[0], 3), "lp2": round(r2[0], 3), "n_tok": n,
            "delta_per_tok": round((r1[0] - r2[0]) / n, 5)}) + "\n")
        out.flush()
        ok += 1
        if ok % 50 == 0:
            print(f"  shard{args.shard} scored={ok} skip={skip}", flush=True)
    out.close()
    print(f"HF_BANK_DONE shard={args.shard} ok={ok} skip={skip}", flush=True)


if __name__ == "__main__":
    main()
