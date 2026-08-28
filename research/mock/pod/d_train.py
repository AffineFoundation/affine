#!/usr/bin/env python3
"""Track M discriminator LoRA trainer -- DDP, one full replica per GPU.

D = the teacher base (Qwen/Qwen3.8-27B) + LoRA, trained as an A/B chooser:
CE on the logits of tokens "A"/"B" at the last prompt position, teacher slot
randomised per example. From scratch at every crown, on the full archive.

Launch:  torchrun --nproc-per-node=N d_train.py ...   (or plain python, N=1)
The 27B in bf16 + LoRA fits a single H200 WITHOUT gradient checkpointing;
per-GPU replicas + DDP beat the old device_map=auto sharding ~10x (audited:
90 min -> minutes).

Saves TWO formats:
  <out>/hf/   PEFT/HF keys (model.layers.*)  -- for HF reload/continuation
  <out>/      vLLM keys (language_model.model.layers.*) -- for serving.
The serving copy is remapped because the checkpoint's architecture is
*ForConditionalGeneration: vLLM nests the text tower under language_model and
SILENTLY no-ops adapters whose keys don't match (bug caught 2026-08-22).
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F

from judge_common import ab_token_ids, fit_ids
from remap_lora import remap

WANT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def existing_targets(model):
    names = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    got = [t for t in WANT_TARGETS if t in names]
    if not got:
        raise SystemExit("no LoRA targets found")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--lora-out", required=True)
    ap.add_argument("--max-len", type=int, default=3584)
    ap.add_argument("--batch", type=int, default=2, help="per-GPU microbatch")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="memory fallback; ~3x slower")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)

    def say(msg):
        if rank == 0:
            print(msg, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ab = ab_token_ids(tok)

    pairs = [json.loads(x) for x in open(args.pairs)]
    say(f"pairs={len(pairs)} world={world}")
    if not pairs:
        raise SystemExit("no pairs")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": local})
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=existing_targets(model)))
    say(f"fresh LoRA adapter r={args.lora_r} (from scratch)")
    model.train()
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local])
    core = model.module if world > 1 else model

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    dev = torch.device(f"cuda:{local}")

    def batch_ids(examples):
        n = max(len(e) for e in examples)
        ids = torch.full((len(examples), n), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(examples), n), dtype=torch.long)
        for i, e in enumerate(examples):
            ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            mask[i, :len(e)] = 1
        return ids.to(dev), mask.to(dev)

    rng = random.Random(args.seed)          # same shuffle on every rank
    exrng = random.Random(args.seed + 1)    # same slot flips on every rank
    step = mb = 0
    losses, accs = [], []
    n_ep = 0
    done = False
    gstride = args.batch * world
    while not done and n_ep < args.epochs - 1e-9:
        order = list(range(len(pairs)))
        rng.shuffle(order)
        n_ep += 1
        for i0 in range(0, len(order) - gstride + 1, gstride):
            chunk_all = order[i0:i0 + gstride]
            flips = [exrng.random() < 0.5 for _ in chunk_all]
            mine_idx = chunk_all[rank * args.batch:(rank + 1) * args.batch]
            mine_flip = flips[rank * args.batch:(rank + 1) * args.batch]
            exs, lbls = [], []
            for j, t_in_a in zip(mine_idx, mine_flip):
                p = pairs[j]
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
                    say(f"step {step} loss={sum(losses[-40:])/len(losses[-40:]):.4f} "
                        f"train_acc={sum(accs[-40:])/len(accs[-40:]):.3f}")
                if step >= args.max_steps:
                    done = True
                    break

    if world > 1:
        dist.barrier()
    if rank == 0:
        hf_dir = os.path.join(args.lora_out, "hf")
        core.save_pretrained(hf_dir)
        hit, tot = remap(hf_dir, args.lora_out,
                         prefix="language_model.model.")
        meta = {"steps": step, "examples_seen": mb * args.batch * world,
                "world": world,
                "loss": sum(losses) / max(len(losses), 1),
                "train_acc_tail": sum(accs[-40:]) / max(len(accs[-40:]), 1),
                "remapped_keys": f"{hit}/{tot}"}
        with open(os.path.join(args.lora_out, "train_meta.json"), "w") as fh:
            json.dump(meta, fh)
        print(f"D_TRAIN_DONE {json.dumps(meta)}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
