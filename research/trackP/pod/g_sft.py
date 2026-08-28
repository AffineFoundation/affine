#!/usr/bin/env python3
"""Best-of-n rejection SFT for the Track P generator (Qwen3.6-35B-A3B MoE).

Same method as Track A's g-step: LoRA confined to the ATTENTION projections
(adapters on expert weights are not reliably loadable by the vLLM server),
lr 1e-5, loss masked to completion tokens only. Runs as a per-round
subprocess on its own GPUs (device_map=auto shards the 35B across every
visible device).

Input: winners.jsonl, one line per prefix: {"turn_id", "prompt", "completion"}
where prompt is the fully rendered chat-template text ending inside <think>,
and completion is the RAW sampled text of the top-reward candidate.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--winners", required=True)
    ap.add_argument("--lora-in", default="", help="adapter dir to continue from")
    ap.add_argument("--lora-out", required=True)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=40, help="optimizer steps cap")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(x) for x in open(args.winners)]
    exs = []
    n_skip = 0
    for r in rows:
        p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
        c_ids = c_ids + [tok.eos_token_id]
        if len(p_ids) + len(c_ids) > args.max_len:
            keep = args.max_len - len(c_ids)
            if keep < 256:
                n_skip += 1
                continue
            p_ids = p_ids[-keep:]
        exs.append((p_ids, c_ids))
    print(f"examples={len(exs)} skipped_long={n_skip}", flush=True)
    if not exs:
        raise SystemExit("no usable winners")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    if args.lora_in and os.path.isdir(args.lora_in):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_in, is_trainable=True)
        print(f"continuing adapter from {args.lora_in}", flush=True)
    else:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=ATTN_TARGETS))
        print("fresh attention-only LoRA adapter", flush=True)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    dev = next(model.parameters()).device

    order = list(range(len(exs)))
    rng = random.Random(0)
    step = mb = 0
    run = []
    done = False
    n_epochs = 0
    while not done and n_epochs < args.epochs - 1e-9:
        rng.shuffle(order)
        n_epochs += 1
        for i0 in range(0, len(order), args.batch):
            batch = [exs[j] for j in order[i0:i0 + args.batch]]
            n = max(len(p) + len(c) for p, c in batch)
            ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
            lbl = torch.full((len(batch), n), -100, dtype=torch.long)
            msk = torch.zeros((len(batch), n), dtype=torch.long)
            for bi, (p, c) in enumerate(batch):
                seq = p + c
                ids[bi, :len(seq)] = torch.tensor(seq)
                msk[bi, :len(seq)] = 1
                lbl[bi, len(p):len(seq)] = torch.tensor(c)
            out = model(input_ids=ids.to(dev), attention_mask=msk.to(dev))
            lg = out.logits[:, :-1].float()
            tgt = lbl[:, 1:].to(lg.device)
            loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                                   ignore_index=-100) / args.accum
            loss.backward()
            run.append(loss.item() * args.accum)
            mb += 1
            if mb % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0:
                    print(f"step {step} loss={sum(run)/len(run):.4f}", flush=True)
                    run = []
                if step >= args.max_steps:
                    done = True
                    break

    model.save_pretrained(args.lora_out)
    print(f"G_SFT_DONE steps={step} out={args.lora_out}", flush=True)


if __name__ == "__main__":
    main()
