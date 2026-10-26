#!/usr/bin/env python3
"""Track D milestone 1: plain SFT distillation of the student on raw teacher
rollouts (the CTRL arm every later Sobolev arm is compared against).

Input is the sample_rollouts.py cache: one line per turn,
{"turn_id", "rollouts": [{"z","y","raw"}, ...]}. Each rollout becomes one
training example: prompt = chat-template prefix ending inside <think>,
completion = the RAW teacher bytes + EOS. Loss is masked to completion
tokens, so the student learns exactly what the teacher emitted in exactly
the serving format.

Saves adapter checkpoints every --save-every optimizer steps to
<out>/step<N> plus <out>/final, so the bench watcher can score progress.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gad_common import build_prompt, load_turns  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--rollouts", required=True, help="teacher rollout cache jsonl")
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--out", required=True, help="checkpoint root dir")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--save-every", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-think", action="store_true",
                    help="for non-thinking teachers: do not inject <think> "
                         "into the prompt and drop the teacher's leading "
                         "</think> from completions, so the student learns "
                         "a coherent think-free format")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    turns = load_turns(args.turns)
    exs = []
    n_skip_long = n_skip_turn = 0
    with open(args.rollouts) as fh:
        for line in fh:
            r = json.loads(line)
            tid = r["turn_id"]
            if tid not in turns:
                n_skip_turn += 1
                continue
            if args.no_think:
                prompt = tok.apply_chat_template(
                    turns[tid], tokenize=False, add_generation_prompt=True)
            else:
                prompt = build_prompt(tok, turns[tid])
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            for roll in r.get("rollouts", []):
                raw = roll["raw"]
                if args.no_think:
                    s = raw.lstrip()
                    if s.startswith("</think>"):
                        raw = s[len("</think>"):].lstrip("\n")
                c_ids = tok(raw, add_special_tokens=False)["input_ids"]
                c_ids = c_ids + [tok.eos_token_id]
                if len(p_ids) + len(c_ids) > args.max_len:
                    n_skip_long += 1
                    continue
                exs.append((p_ids, c_ids))
    print(f"examples={len(exs)} skipped_long={n_skip_long} "
          f"skipped_no_turn={n_skip_turn}", flush=True)
    if len(exs) < 100:
        raise SystemExit("too few usable examples")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.to("cuda")
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = min(args.max_steps,
                      int(args.epochs * len(exs) / (args.batch * args.accum)))
    warmup = max(10, total_steps // 20)

    def lr_at(s):
        if s < warmup:
            return args.lr * s / warmup
        t = (s - warmup) / max(total_steps - warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    def save(tag):
        d = os.path.join(args.out, tag)
        model.save_pretrained(d)
        print(f"CKPT_SAVED {d}", flush=True)

    order = list(range(len(exs)))
    rng = random.Random(args.seed)
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
            out = model(input_ids=ids.cuda(), attention_mask=msk.cuda())
            lg = out.logits[:, :-1].float()
            tgt = lbl[:, 1:].cuda()
            loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                                   ignore_index=-100) / args.accum
            loss.backward()
            run.append(loss.item() * args.accum)
            mb += 1
            if mb % args.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"step {step}/{total_steps} "
                          f"loss={sum(run)/len(run):.4f} lr={lr_at(step):.2e}",
                          flush=True)
                    run = []
                if step % args.save_every == 0:
                    save(f"step{step}")
                if step >= total_steps:
                    done = True
                    break

    save("final")
    print(f"DISTILL_SFT_DONE steps={step}", flush=True)


if __name__ == "__main__":
    main()
