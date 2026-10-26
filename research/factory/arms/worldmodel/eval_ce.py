#!/usr/bin/env python3
"""Held-out teacher-forced cross-entropy: mean NLL/token of the student
(base or base+adapter) on teacher rollout completions it never trained on.
The cheap leading indicator that distillation is moving the student toward
the teacher before any SWE panel lands. Appends one json line to --report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gad_common import build_prompt, load_turns  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--lora", default="", help="adapter dir; empty = base only")
    ap.add_argument("--rollouts", required=True, help="held-out teacher cache")
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=400, help="max examples")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--report", required=True, help="jsonl to append to")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    turns = load_turns(args.turns)

    exs = []
    with open(args.rollouts) as fh:
        for line in fh:
            r = json.loads(line)
            tid = r["turn_id"]
            if tid not in turns:
                continue
            prompt = build_prompt(tok, turns[tid])
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            for roll in r.get("rollouts", []):
                c_ids = tok(roll["raw"], add_special_tokens=False)["input_ids"]
                if len(p_ids) + len(c_ids) > args.max_len:
                    continue
                exs.append((p_ids, c_ids))
            if len(exs) >= args.limit:
                break
    print(f"eval examples={len(exs)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.to("cuda").eval()

    tot_nll = 0.0
    tot_tok = 0
    for p, c in exs:
        ids = torch.tensor([p + c], device="cuda")
        lg = model(input_ids=ids).logits[0, :-1].float()
        tgt = ids[0, 1:]
        nll = F.cross_entropy(lg[len(p) - 1:], tgt[len(p) - 1:],
                              reduction="sum")
        tot_nll += nll.item()
        tot_tok += len(c)

    rec = {"tag": args.tag, "lora": args.lora or None,
           "nll_per_tok": round(tot_nll / max(tot_tok, 1), 4),
           "n_examples": len(exs), "n_tokens": tot_tok}
    with open(args.report, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("CE_EVAL " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
