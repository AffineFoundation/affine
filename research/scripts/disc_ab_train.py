#!/usr/bin/env python3
"""A/B-choice discriminator: which candidate came from the reference model?

Structure (differs from the scalar-head reward model in disc_train.py):

  * the backbone keeps its language-modelling head, so the "classifier" is the
    model's own probability of emitting the token "A" versus "B". No randomly
    initialised head, which means a ZERO-SHOT baseline is measurable before any
    training -- an untrained scalar head is stuck at chance by construction.
  * both candidates sit in ONE context, so attention can compare them directly
    instead of scoring each in isolation. One forward pass per pair, not two.
  * which slot holds the teacher is randomised per example, and every eval
    scores BOTH orderings and averages. Position bias in A/B judging is large,
    so it is cancelled in the metric and reported separately as a diagnostic.

Leak control. Reasoning markers are stripped from both sides before the prompt
is built. Measured on the raw data, the literal string "<think>" appears in
33.5% of teacher thoughts but only 7.0% of miner thoughts -- a rule matching
that tag alone scores AUC 0.632, so leaving it in would hand the model a free
shortcut that says nothing about reasoning quality.

Reported every eval, because a single accuracy number is not interpretable:
  acc         order-averaged accuracy (the headline)
  pos_bias    P(picks slot A) averaged over both orderings; 0.5 is unbiased
  len_acc     accuracy of "the longer candidate is the teacher" on the same slice
  matched_acc accuracy restricted to pairs whose candidates are within 10% length
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from disc_text import as_list, normalize  # noqa: F401  (shared with offline checks)

QUESTION = (
    "Above is the transcript of a software-engineering task in progress.\n"
    "Two assistants each proposed the next step. One of them is the reference "
    "assistant.\n\nWhich one is the reference assistant? Reply with a single "
    "letter, A or B."
)


def render_prefix(messages, max_chars):
    parts = [f"<{m.get('role','?')}>\n{m.get('content','')}" for m in messages]
    text = "\n".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


def load_turns(path):
    turns = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            t = json.loads(line)
            turns[t["turn_id"]] = t.get("prefix") or []
    return turns


def load_pairs(data_dir, split, channel, limit=None, seed=0):
    """Pairs where miner and teacher genuinely differ on the chosen channel."""
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, "pairs_*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("split") != split:
                    continue
                if channel == "thought":
                    mine = normalize(r.get("z_a"))
                    refs = [normalize(x) for x in as_list(r.get("teacher_z"))]
                elif channel == "action":
                    mine = normalize(r.get("y_a"))
                    refs = [normalize(x) for x in as_list(r.get("teacher_y"))]
                else:  # both: thought then action, as one response
                    mine = (normalize(r.get("z_a")) + "\n\n" + normalize(r.get("y_a"))).strip()
                    tz = [normalize(x) for x in as_list(r.get("teacher_z"))]
                    ty = [normalize(x) for x in as_list(r.get("teacher_y"))]
                    refs = [(a + "\n\n" + b).strip() for a, b in zip(tz, ty)] or ty
                ref = next((c for c in refs if c and c != mine), None)
                if not ref or not mine:
                    continue
                out.append({
                    "turn_id": r["turn_id"],
                    "repo": r.get("repo"),
                    "mine": mine,
                    "ref": ref,
                    "corpus_epoch": r.get("corpus_epoch"),
                })
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


class ABData(Dataset):
    """Builds the A/B prompt. teacher_slot is fixed when swap is not None."""

    def __init__(self, pairs, turns, tok, max_len, swap=None, seed=0):
        self.pairs, self.turns, self.tok = pairs, turns, tok
        self.max_len, self.swap, self.seed = max_len, swap, seed

    def __len__(self):
        return len(self.pairs)

    def _template(self, prefix_text, a_text, b_text):
        body = (
            f"{prefix_text}\n\n=== Candidate A ===\n{a_text}\n\n"
            f"=== Candidate B ===\n{b_text}\n\n{QUESTION}"
        )
        msgs = [{"role": "user", "content": body}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # tokenizers without the Qwen3 thinking switch
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    def _fit(self, prefix, a_text, b_text):
        """Shrink the prefix until the templated prompt fits max_len."""
        chars = min(len(prefix), self.max_len * 6)
        ids = None
        for _ in range(5):
            text = self._template(prefix[-chars:] if chars else "", a_text, b_text)
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) <= self.max_len:
                return ids
            over = len(ids) - self.max_len
            nxt = chars - int(over * 4.0) - 256
            if nxt <= 0:
                break
            chars = nxt
        return ids[-self.max_len:]

    def __getitem__(self, i):
        p = self.pairs[i]
        if self.swap is None:
            teacher_in_a = random.Random(self.seed * 1000003 + i).random() < 0.5
        else:
            teacher_in_a = not self.swap
        a, b = (p["ref"], p["mine"]) if teacher_in_a else (p["mine"], p["ref"])
        prefix = render_prefix(self.turns.get(p["turn_id"], []), self.max_len * 6)
        return {
            "ids": self._fit(prefix, a, b),
            "label": 0 if teacher_in_a else 1,  # 0 -> "A", 1 -> "B"
            "len_ref": len(p["ref"]),
            "len_mine": len(p["mine"]),
            "repo": p["repo"],
            "turn_id": p["turn_id"],
        }


def collate(batch, pad_id):
    n = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = torch.tensor(b["ids"], dtype=torch.long)
        mask[i, : len(b["ids"])] = 1
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return ids, mask, labels, batch


def ab_token_ids(tok):
    """Single-token ids for the two answer letters."""
    out = []
    for letter in ("A", "B"):
        cand = None
        for form in (letter, " " + letter):
            enc = tok(form, add_special_tokens=False)["input_ids"]
            if len(enc) == 1:
                cand = enc[0]
                break
        if cand is None:
            raise SystemExit(f"cannot map {letter!r} to a single token")
        out.append(cand)
    return out


def build_model(name, lora_r, train: bool, device_map=None, grad_ckpt=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(attn_implementation="sdpa")
    if device_map:
        # Shard the model across all visible GPUs; caller must NOT call .to().
        kw["device_map"] = device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, **kw)
    model.config.use_cache = False
    if train:
        from peft import LoraConfig, get_peft_model

        # Checkpointing trades ~30-40% throughput for memory. With sdpa and a
        # 14B LoRA on a 183GB card there is memory to spare, so it is optional.
        if grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
    return tok, model


def ab_logits(model, ids, mask, tok_ids):
    """Logits for A and B at the last real token of each row."""
    out = model(input_ids=ids, attention_mask=mask).logits
    idx = mask.sum(1) - 1
    last = out[torch.arange(out.size(0), device=out.device), idx]
    return last[:, tok_ids].float()


@torch.no_grad()
def evaluate(model, pairs, turns, tok, tok_ids, dev, args, tag, dump=None):
    """Score every pair in BOTH orderings and average, cancelling position bias."""
    model.eval()
    per = {}
    picked_a = 0
    seen = 0
    for swap in (False, True):
        ds = ABData(pairs, turns, tok, args.max_len, swap=swap)
        dl = DataLoader(ds, batch_size=args.eval_batch, shuffle=False,
                        num_workers=4, collate_fn=lambda b: collate(b, tok.pad_token_id))
        for ids, mask, labels, meta in dl:
            lg = ab_logits(model, ids.to(dev), mask.to(dev), tok_ids)
            p = torch.softmax(lg, -1).cpu()
            for i, m in enumerate(meta):
                # probability mass on the slot that actually holds the teacher
                p_ref = p[i, labels[i]].item()
                k = (m["turn_id"], m["repo"], m["len_ref"], m["len_mine"])
                per.setdefault(k, []).append(p_ref)
                picked_a += p[i, 0].item()
                seen += 1
    correct = tot = 0
    lcorrect = lties = 0
    mcorrect = mtot = 0
    recs = []
    for (turn_id, repo, lref, lmine), ps in per.items():
        p_ref = sum(ps) / len(ps)  # order-averaged
        tot += 1
        correct += p_ref > 0.5
        dl_ = lref - lmine
        lcorrect += dl_ > 0
        lties += dl_ == 0
        mx = max(lref, lmine) or 1
        if abs(dl_) / mx <= 0.10:
            mtot += 1
            mcorrect += p_ref > 0.5
        if dump is not None:
            # log-odds that the judge picks the teacher; Stage 3 uses -mean(delta).
            # Lengths ride along so merged shards can recompute every control.
            q = min(max(p_ref, 1e-6), 1 - 1e-6)
            recs.append({"repo": repo, "turn_id": turn_id,
                         "delta": math.log(q / (1 - q)), "p_ref": p_ref,
                         "len_ref": lref, "len_mine": lmine})
    acc = correct / max(tot, 1)
    lacc = (lcorrect + 0.5 * lties) / max(tot, 1)
    macc = mcorrect / max(mtot, 1)
    bias = picked_a / max(seen, 1)
    # A length rule may point either way, so the bar it sets is the better
    # of "longer is teacher" and "shorter is teacher".
    lbar = max(lacc, 1 - lacc)
    print(f"[{tag}] n={tot}  acc={acc:.4f}  |  length_bar={lbar:.4f} "
          f"(raw len_acc={lacc:.4f})  matched_acc={macc:.4f} (n={mtot})  "
          f"pos_bias(P_A)={bias:.3f}", flush=True)
    if dump:
        with open(dump, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    model.train()
    return acc, lacc, macc, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/work/disc_pairs")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--channel", default="thought", choices=["thought", "action", "both"])
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--eval-limit", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--zero-shot-only", action="store_true")
    ap.add_argument("--device-map", default=None,
                    help="e.g. 'auto' to shard a large judge across all GPUs")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (faster; needs more VRAM)")
    ap.add_argument("--shard", default=None,
                    help="'i/n' -- score only slice i of n, for data-parallel "
                         "eval across GPUs (one process per GPU). Merge with "
                         "disc_ab_merge.py")
    ap.add_argument("--out", default="/root/work/ab_run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)

    turns = load_turns(os.path.join(args.data, "turns.jsonl.gz"))
    te = load_pairs(args.data, "test", args.channel, args.eval_limit)
    tag = ""
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        te = te[i::n]
        tag = f".shard{i}of{n}"
        print(f"shard {i}/{n}: {len(te)} test pairs", flush=True)
    tr = [] if args.zero_shot_only else load_pairs(args.data, "train", args.channel, args.train_limit)
    print(f"channel={args.channel}  turns={len(turns)}  train={len(tr)}  test={len(te)}", flush=True)

    tok, model = build_model(args.model, args.lora_r, train=not args.zero_shot_only,
                             device_map=args.device_map,
                             grad_ckpt=not args.no_grad_ckpt)
    if not args.device_map:
        model.to(dev)
    else:
        # inputs must land on the shard holding the embedding layer
        dev = str(getattr(model, "device", None) or "cuda:0")
        print(f"sharded across {torch.cuda.device_count()} GPUs; inputs -> {dev}", flush=True)
    tok_ids = ab_token_ids(tok)
    print(f"A/B token ids = {tok_ids}", flush=True)

    print("=== ZERO-SHOT (no training; the pretrained judge on its own) ===", flush=True)
    zs = evaluate(model, te, turns, tok, tok_ids, dev, args, "zeroshot" + tag,
                  dump=os.path.join(args.out, f"zeroshot_scores{tag}.jsonl"))
    json.dump({"model": args.model, "channel": args.channel, "shard": args.shard,
               "max_len": args.max_len, "n": len(te), "zero_shot":
               {"acc": zs[0], "len_acc": zs[1], "matched_acc": zs[2], "pos_bias": zs[3]}},
              open(os.path.join(args.out, f"zeroshot{tag}.json"), "w"), indent=2)
    if args.zero_shot_only:
        print("zero-shot only; stopping", flush=True)
        return

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.2f}M", flush=True)

    dl_tr = DataLoader(ABData(tr, turns, tok, args.max_len), batch_size=args.batch,
                       shuffle=True, num_workers=4,
                       collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    total = max(len(dl_tr) // args.accum, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=total, pct_start=0.1)

    step = 0
    t0 = time.time()
    run = []
    for i, (ids, mask, labels, meta) in enumerate(dl_tr):
        lg = ab_logits(model, ids.to(dev), mask.to(dev), tok_ids)
        loss = F.cross_entropy(lg, labels.to(dev)) / args.accum
        loss.backward()
        run.append(loss.item() * args.accum)
        if (i + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(f"step {step}/{total}  loss={sum(run)/len(run):.4f}  "
                      f"{(i+1)*args.batch/(time.time()-t0):.1f} ex/s", flush=True)
                run = []
            if step % args.eval_every == 0:
                evaluate(model, te, turns, tok, tok_ids, dev, args, f"step{step}")

    print("=== final ===", flush=True)
    fin = evaluate(model, te, turns, tok, tok_ids, dev, args, "final",
                   dump=os.path.join(args.out, "test_scores.jsonl"))
    json.dump({"model": args.model, "channel": args.channel,
               "train_pairs": len(tr), "test_pairs": len(te),
               "zero_shot": {"acc": zs[0], "len_acc": zs[1], "matched_acc": zs[2], "pos_bias": zs[3]},
               "trained": {"acc": fin[0], "len_acc": fin[1], "matched_acc": fin[2], "pos_bias": fin[3]}},
              open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    model.save_pretrained(os.path.join(args.out, "lora"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
