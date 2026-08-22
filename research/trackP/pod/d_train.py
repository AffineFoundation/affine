#!/usr/bin/env python3
"""Track P discriminator LoRA trainer (disc_ab_train.py adapted for 27B+LoRA).

D = the TEACHER base (Qwen/Qwen3.8-27B) with a LoRA adapter, trained as an
A/B chooser: CE on the logits of tokens "A" vs "B" at the last prompt
position, teacher slot randomised per example. Runs as a per-D-round
subprocess on dedicated GPUs (device_map=auto), saves the adapter, exits;
the driver then hot-swaps the adapter into the vLLM D server.

Pairs file: jsonl of {"prefix_text", "ref", "mine"} -- texts already
normalised (thought markers stripped) by the driver.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

from judge_common import ab_token_ids, fit_ids

WANT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def existing_targets(model):
    names = set()
    for n, _ in model.named_modules():
        names.add(n.rsplit(".", 1)[-1])
    got = [t for t in WANT_TARGETS if t in names]
    if not got:
        raise SystemExit(f"no LoRA targets found; module leaf names: {sorted(names)[:50]}")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--lora-in", default="", help="adapter dir to continue from")
    ap.add_argument("--lora-out", required=True)
    ap.add_argument("--max-len", type=int, default=3584)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ab = ab_token_ids(tok)

    pairs = [json.loads(x) for x in open(args.pairs)]
    print(f"pairs={len(pairs)}", flush=True)
    if not pairs:
        raise SystemExit("no pairs")

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
        targets = existing_targets(model)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules=targets))
        print(f"fresh LoRA adapter r={args.lora_r} targets={targets}", flush=True)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    dev = next(model.parameters()).device

    def batch_ids(examples):
        n = max(len(e) for e in examples)
        ids = torch.full((len(examples), n), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(examples), n), dtype=torch.long)
        for i, e in enumerate(examples):
            ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            mask[i, :len(e)] = 1
        return ids.to(dev), mask.to(dev)

    rng = random.Random(args.seed)
    step = mb = 0
    losses, accs = [], []
    n_ep = 0
    done = False
    while not done and n_ep < args.epochs - 1e-9:
        order = list(range(len(pairs)))
        rng.shuffle(order)
        n_ep += 1
        for i0 in range(0, len(order), args.batch):
            chunk = [pairs[j] for j in order[i0:i0 + args.batch]]
            exs, lbls = [], []
            for p in chunk:
                t_in_a = rng.random() < 0.5
                a, b = (p["ref"], p["mine"]) if t_in_a else (p["mine"], p["ref"])
                exs.append(fit_ids(tok, p["prefix_text"], a, b, args.max_len))
                lbls.append(0 if t_in_a else 1)
            ids, mask = batch_ids(exs)
            out = model(input_ids=ids, attention_mask=mask).logits
            idx = mask.sum(1) - 1
            last = out[torch.arange(out.size(0), device=out.device), idx]
            lg = last[:, ab].float()
            tgt = torch.tensor(lbls, device=lg.device)
            loss = F.cross_entropy(lg, tgt) / args.accum
            loss.backward()
            losses.append(loss.item() * args.accum)
            accs.append((lg.argmax(-1) == tgt).float().mean().item())
            mb += 1
            if mb % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"step {step} loss={sum(losses[-80:])/len(losses[-80:]):.4f} "
                          f"train_acc={sum(accs[-80:])/len(accs[-80:]):.3f}", flush=True)
                if step >= args.max_steps:
                    done = True
                    break

    model.save_pretrained(args.lora_out)
    meta = {"steps": step, "examples_seen": mb * args.batch,
            "loss": sum(losses) / max(len(losses), 1),
            "train_acc_tail": sum(accs[-80:]) / max(len(accs[-80:]), 1)}
    with open(os.path.join(args.lora_out, "train_meta.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"D_TRAIN_DONE {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
