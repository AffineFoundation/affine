#!/usr/bin/env python3
"""Phase 1 of Track C: train the discriminator ONCE, then freeze it.

Pairs are (fresh Qwen3-32B teacher rollout) vs (raw Qwen3-4B = G0 rollout) on
the same turn, in the 'both' channel (normalized thought + blank line +
action). All prompt construction, order randomization, evaluation, and the
length-bar control are reused verbatim from disc_ab_train.py so the frozen
judge is built exactly like the online track's judge.

This is the ONLY time D trains in Track C.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trackc_common import candidate_text, load_rollouts, load_turns  # noqa: F401
from disc_ab_train import (ABData, ab_logits, ab_token_ids, build_model,
                           collate, evaluate)


def build_pairs(teacher_map, g_map, per_turn):
    out = []
    for tid in sorted(set(teacher_map) & set(g_map)):
        ts = teacher_map[tid]
        gs = g_map[tid]
        for i in range(min(len(ts), len(gs), per_turn)):
            ref = candidate_text(ts[i]["z"], ts[i]["y"])
            mine = candidate_text(gs[i]["z"], gs[i]["y"])
            if not ref or not mine or ref == mine:
                continue
            out.append({"turn_id": tid, "repo": tid.split(":")[0],
                        "mine": mine, "ref": ref})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/work/research/data/disc_pairs")
    ap.add_argument("--teacher-train", required=True)
    ap.add_argument("--g-train", required=True)
    ap.add_argument("--teacher-eval", required=True)
    ap.add_argument("--g-eval", required=True)
    ap.add_argument("--pairs-per-turn", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--out", default="/root/work/trackC/discD")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)

    from trackc_common import parse_prefix
    turns_full = load_turns(os.path.join(args.data, "turns.jsonl.gz"))
    turns = {k: (parse_prefix(v.get("prefix")) or [])
             for k, v in turns_full.items()}

    tr = build_pairs(load_rollouts(args.teacher_train),
                     load_rollouts(args.g_train), args.pairs_per_turn)
    te = build_pairs(load_rollouts(args.teacher_eval),
                     load_rollouts(args.g_eval), 1)
    print(f"train pairs={len(tr)}  heldout pairs={len(te)}", flush=True)

    tok, model = build_model(args.model, args.lora_r, train=True,
                             grad_ckpt=not args.no_grad_ckpt)
    model.to(dev)
    tok_ids = ab_token_ids(tok)

    print("=== ZERO-SHOT ===", flush=True)
    zs = evaluate(model, te, turns, tok, tok_ids, dev, args, "zeroshot")

    dl_tr = DataLoader(ABData(tr, turns, tok, args.max_len),
                       batch_size=args.batch, shuffle=True, num_workers=4,
                       collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    total = max(len(dl_tr) * args.epochs // args.accum, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=total, pct_start=0.1)

    best = {"acc": 0.0, "step": 0}
    history = []

    def eval_and_maybe_save(tag, step):
        acc, lacc, macc, bias = evaluate(model, te, turns, tok, tok_ids,
                                         dev, args, tag)
        history.append({"step": step, "acc": acc, "len_acc": lacc,
                        "matched_acc": macc, "pos_bias": bias})
        if acc > best["acc"]:
            best.update(acc=acc, step=step)
            model.save_pretrained(os.path.join(args.out, "lora"))
            print(f"  saved best adapter (acc={acc:.4f})", flush=True)
        return acc

    step = 0
    t0 = time.time()
    run = []
    stop = False
    for ep in range(args.epochs):
        if stop:
            break
        for i, (ids, mask, labels, meta) in enumerate(dl_tr):
            lg = ab_logits(model, ids.to(dev), mask.to(dev), tok_ids)
            loss = F.cross_entropy(lg, labels.to(dev)) / args.accum
            loss.backward()
            run.append(loss.item() * args.accum)
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"ep{ep} step {step}/{total} loss={sum(run)/len(run):.4f} "
                          f"{(time.time()-t0):.0f}s", flush=True)
                    run = []
                if step % args.eval_every == 0:
                    eval_and_maybe_save(f"step{step}", step)
                    # plateau: two consecutive evals with no improvement after
                    # at least 3 evals total
                    if (len(history) >= 3
                            and history[-1]["acc"] <= best["acc"] - 1e-9
                            and history[-2]["acc"] <= best["acc"] - 1e-9
                            and best["step"] <= history[-3]["step"]):
                        print("plateau reached; stopping", flush=True)
                        stop = True
                        break
        if stop:
            break

    print("=== final ===", flush=True)
    eval_and_maybe_save("final", step)
    json.dump({"model": args.model, "train_pairs": len(tr),
               "heldout_pairs": len(te), "zero_shot":
               {"acc": zs[0], "len_acc": zs[1], "matched_acc": zs[2],
                "pos_bias": zs[3]},
               "best": best, "history": history},
              open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(f"FROZEN: best acc={best['acc']:.4f} at step {best['step']}; "
          f"adapter at {os.path.join(args.out, 'lora')}", flush=True)


if __name__ == "__main__":
    main()
