#!/usr/bin/env python3
"""Stage 2: train a GAD-style discriminator on (prefix, action) pairs.

Bradley-Terry objective: D(prefix, teacher_action) > D(prefix, miner_action).

The discriminator is a small causal backbone (LoRA-adapted) plus a scalar head
read off the last non-pad token, i.e. a standard reward-model head.

Controls that decide whether a result means anything:
  * exact-match pairs are dropped (miner action == a teacher ref => no signal)
  * the length-only pairwise AUC is reported alongside the model's, because
    teacher actions are systematically longer and a model can score ~0.60 by
    learning "longer == teacher" and nothing else
  * a length-matched slice is scored separately to isolate non-length signal
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROMPT_SEP = "\n### Candidate next action:\n"


def as_list(v):
    """teacher_y is sometimes a list, sometimes a JSON-encoded list."""
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        try:
            p = json.loads(v)
            return [str(x) for x in p] if isinstance(p, list) else [v]
        except Exception:
            return [v]
    return []


def render_prefix(messages, max_chars):
    """Flatten the turn prefix, keeping the most recent text."""
    parts = []
    for m in messages:
        role = m.get("role", "?")
        parts.append(f"<{role}>\n{m.get('content','')}")
    text = "\n".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


def load_turns(path):
    turns = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            t = json.loads(line)
            turns[t["turn_id"]] = t.get("prefix") or []
    return turns


def load_pairs(data_dir, split, limit=None, seed=0):
    """Usable pairs only: exact-match pairs carry no Bradley-Terry signal."""
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, "pairs_*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("split") != split:
                    continue
                if int(r.get("n_exact_match") or 0) > 0:
                    continue
                ya = (r.get("y_a") or "").strip()
                pos = next(
                    (c.strip() for c in as_list(r.get("teacher_y")) if c.strip() and c.strip() != ya),
                    None,
                )
                if not pos or not ya:
                    continue
                out.append(
                    {
                        "turn_id": r["turn_id"],
                        "repo": r.get("repo"),
                        "neg": ya,
                        "pos": pos,
                        "corpus_epoch": r.get("corpus_epoch"),
                    }
                )
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


class PairData(Dataset):
    def __init__(self, pairs, turns, tok, max_len, max_action_tok):
        self.pairs, self.turns, self.tok = pairs, turns, tok
        self.max_len, self.max_action_tok = max_len, max_action_tok
        # A generous char budget; exact truncation happens in token space.
        self.prefix_chars = max_len * 6

    def __len__(self):
        return len(self.pairs)

    def encode(self, prefix_text, action):
        a_ids = self.tok(PROMPT_SEP + action, add_special_tokens=False)["input_ids"][
            : self.max_action_tok
        ]
        budget = self.max_len - len(a_ids)
        p_ids = self.tok(prefix_text, add_special_tokens=False)["input_ids"]
        # Left-truncate: the decision depends on the most recent context.
        p_ids = p_ids[-budget:] if budget > 0 else []
        return p_ids + a_ids

    def __getitem__(self, i):
        p = self.pairs[i]
        prefix_text = render_prefix(self.turns.get(p["turn_id"], []), self.prefix_chars)
        return {
            "pos": self.encode(prefix_text, p["pos"]),
            "neg": self.encode(prefix_text, p["neg"]),
            "len_pos": len(p["pos"]),
            "len_neg": len(p["neg"]),
            "repo": p["repo"],
            "turn_id": p["turn_id"],
        }


def collate(batch, pad_id):
    seqs = [b["pos"] for b in batch] + [b["neg"] for b in batch]
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask, batch


class Disc(nn.Module):
    """Causal backbone + scalar head on the last real token."""

    def __init__(self, backbone, hidden):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden, 1, dtype=torch.float32)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        idx = mask.sum(1) - 1
        last = h[torch.arange(h.size(0), device=h.device), idx]
        return self.head(last.float()).squeeze(-1)


def build_model(name, lora_r):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(name)
    # sdpa keeps attention memory linear; without it the 1536-token activations
    # for a 16-sequence batch materialise full attention matrices and OOM.
    kw = dict(attn_implementation="sdpa")
    try:
        backbone = AutoModel.from_pretrained(name, dtype=torch.bfloat16, **kw)
    except TypeError:  # older transformers
        backbone = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16, **kw)
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.enable_input_require_grads()
    lcfg = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    backbone = get_peft_model(backbone, lcfg)
    return tok, Disc(backbone, cfg.hidden_size)


@torch.no_grad()
def evaluate(model, loader, dev, tag, dump=None):
    model.eval()
    wins = ties = tot = 0
    lw = lt = 0
    matched_w = matched_t = matched_n = 0
    recs = []
    for ids, mask, meta in loader:
        r = model(ids.to(dev), mask.to(dev)).float().cpu()
        b = len(meta)
        rp, rn = r[:b], r[b:]
        for i, m in enumerate(meta):
            d = (rp[i] - rn[i]).item()
            tot += 1
            wins += d > 0
            ties += d == 0
            # length-only baseline on the same slice
            dl = m["len_pos"] - m["len_neg"]
            lw += dl > 0
            lt += dl == 0
            # length-matched control: actions within 10% length of each other
            mx = max(m["len_pos"], m["len_neg"]) or 1
            if abs(dl) / mx <= 0.10:
                matched_n += 1
                matched_w += d > 0
                matched_t += d == 0
            if dump is not None:
                recs.append({"repo": m["repo"], "turn_id": m["turn_id"], "delta": d})
    auc = (wins + 0.5 * ties) / max(tot, 1)
    lauc = (lw + 0.5 * lt) / max(tot, 1)
    mauc = (matched_w + 0.5 * matched_t) / max(matched_n, 1)
    print(
        f"[{tag}] n={tot}  model_AUC={auc:.4f}  length_only_AUC={lauc:.4f}  "
        f"length_matched_AUC={mauc:.4f} (n={matched_n})",
        flush=True,
    )
    if dump:
        with open(dump, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    model.train()
    return auc, lauc, mauc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/disc_pairs")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--max-action-tok", type=int, default=384)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--train-limit", type=int, default=20000)
    ap.add_argument("--eval-limit", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="/root/results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)

    print("loading turns/prefixes ...", flush=True)
    turns = load_turns(os.path.join(args.data, "turns.jsonl.gz"))
    tr = load_pairs(args.data, "train", args.train_limit)
    te = load_pairs(args.data, "test", args.eval_limit)
    print(f"turns={len(turns)}  train_pairs={len(tr)}  test_pairs={len(te)}", flush=True)

    tok, model = build_model(args.model, args.lora_r)
    model.to(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.2f}M", flush=True)

    mk = lambda d, sh: DataLoader(
        PairData(d, turns, tok, args.max_len, args.max_action_tok),
        batch_size=args.batch,
        shuffle=sh,
        num_workers=4,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )
    dl_tr, dl_te = mk(tr, True), mk(te, False)

    opt = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n], "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=0.0,
    )
    total = len(dl_tr) // args.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.head_lr], total_steps=max(total, 1), pct_start=0.1
    )

    print("=== pre-training eval (head is random; expect ~0.50) ===", flush=True)
    evaluate(model, dl_te, dev, "step0")

    step = 0
    t0 = time.time()
    run_loss = []
    for i, (ids, mask, meta) in enumerate(dl_tr):
        b = len(meta)
        r = model(ids.to(dev), mask.to(dev))
        loss = -F.logsigmoid(r[:b] - r[b:]).mean() / args.accum
        loss.backward()
        run_loss.append(loss.item() * args.accum)
        if (i + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                print(
                    f"step {step}/{total}  loss={sum(run_loss)/len(run_loss):.4f}  "
                    f"{(i+1)*2*args.batch/(time.time()-t0):.1f} seq/s",
                    flush=True,
                )
                run_loss = []
            if step % args.eval_every == 0:
                evaluate(model, dl_te, dev, f"step{step}")

    print("=== final eval ===", flush=True)
    auc, lauc, mauc = evaluate(model, dl_te, dev, "final", dump=os.path.join(args.out, "test_scores.jsonl"))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(
            {
                "model": args.model,
                "train_pairs": len(tr),
                "test_pairs": len(te),
                "max_len": args.max_len,
                "model_auc": auc,
                "length_only_auc": lauc,
                "length_matched_auc": mauc,
            },
            fh,
            indent=2,
        )
    torch.save(model.head.state_dict(), os.path.join(args.out, "head.pt"))
    model.backbone.save_pretrained(os.path.join(args.out, "lora"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
