#!/usr/bin/env python3
"""Pre-train the loop's discriminator on real teacher-vs-generator pairs.

Inside the adversarial loop the discriminator only sees a couple of dozen pairs
per round, which is far too little to learn a subtle distinction: it sat at
chance, so the generator's reward carried almost no information about which of
its own samples was the most teacher-like. This script trains the same judge, in
the same "both" format, on a few thousand cached pairs first, and reports how
well it separates the two models on held-out repositories.

The judge class and the prompt format are imported from the loop itself, so the
adapter trained here is exactly the adapter the loop will use.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gad_loop import Judge, both_channel, load_turns, parse_prefix, valid_action  # noqa: E402


def load_rollouts(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for c in r.get("candidates") or []:
                if c.get("z") and valid_action(c.get("y") or ""):
                    out[r["turn_id"]] = (c["z"], c["y"])
                    break
    return out


def auc(scores, labels):
    """Rank-based AUC. scores: higher means "predicted teacher"."""
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    # average ranks over ties so identical scores do not fake separation
    vals = [s for s, _ in pairs]
    r = [0.0] * len(pairs)
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[k] = avg
        i = j + 1
    pos = sum(1 for _, l in pairs if l == 1)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float("nan")
    s = sum(r[k] for k, (_, l) in enumerate(pairs) if l == 1)
    return (s - pos * (pos + 1) / 2) / (pos * neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--teacher-rollouts", required=True)
    ap.add_argument("--gen-rollouts", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--status-log", default=None)
    args = ap.parse_args()

    def say(msg):
        line = (f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} | "
                f"DISC_PRETRAIN {msg}")
        print(line, flush=True)
        if args.status_log:
            with open(args.status_log, "a") as fh:
                fh.write(line + "\n")

    turns = load_turns(args.turns)
    tea = load_rollouts(args.teacher_rollouts)
    gen = load_rollouts(args.gen_rollouts)
    shared = [t for t in tea if t in gen and t in turns]
    say(f"data | teacher={len(tea)} gen={len(gen)} shared={len(shared)}")

    # Split by repository so the test set measures generalisation to unseen
    # code, not memorisation of a repo's house style.
    def repo_of(tid):
        return (turns[tid].get("repo") or str(tid).rsplit("-", 1)[0])

    repos = sorted({repo_of(t) for t in shared})
    random.Random(0).shuffle(repos)
    n_test = max(1, int(len(repos) * args.test_frac))
    test_repos = set(repos[:n_test])
    say(f"split | repos={len(repos)} test_repos={len(test_repos)}")

    def build(tids):
        rows = []
        for tid in tids:
            msgs = parse_prefix(turns[tid].get("prefix")) or []
            ptext = "\n".join(str(m.get("content", "")) for m in msgs)
            rows.append((ptext, both_channel(*tea[tid]), both_channel(*gen[tid])))
        return rows

    tr_ids = [t for t in shared if repo_of(t) not in test_repos]
    te_ids = [t for t in shared if repo_of(t) in test_repos]
    random.Random(1).shuffle(tr_ids)
    random.Random(2).shuffle(te_ids)
    tr = build(tr_ids[: args.limit])
    te = build(te_ids[: args.eval_n])
    say(f"pairs | train={len(tr)} test={len(te)}")

    judge = Judge(args.model, args.device, lora_r=args.lora_r,
                  max_len=args.max_len, lr=args.lr)
    say("model loaded")

    def evaluate(rows, tag):
        scores, labels = [], []
        for ptext, tref, gtext in rows:
            # score both members of the pair against each other
            scores.append(judge.p_teacher(ptext, tref, gtext))
            labels.append(1)
            scores.append(judge.p_teacher(ptext, gtext, tref))
            labels.append(0)
        acc = sum(1 for s, l in zip(scores, labels)
                  if (s > 0.5) == (l == 1)) / max(1, len(scores))
        a = auc(scores, labels)
        mean_t = sum(s for s, l in zip(scores, labels) if l == 1) / max(1, sum(labels))
        mean_g = (sum(s for s, l in zip(scores, labels) if l == 0)
                  / max(1, len(labels) - sum(labels)))
        say(f"eval {tag} | acc={acc:.3f} auc={a:.3f} "
            f"mean_p_teacher={mean_t:.3f} mean_p_gen={mean_g:.3f} n={len(rows)}")
        return a

    evaluate(te, "zero_shot")
    for ep in range(1, args.epochs + 1):
        random.Random(10 + ep).shuffle(tr)
        step = 200
        for i in range(0, len(tr), step):
            loss, acc = judge.train_step(tr[i:i + step], accum=args.accum)
            say(f"train ep{ep} {i + step}/{len(tr)} | loss={loss:.4f} "
                f"train_acc={acc:.3f}")
        evaluate(te, f"after_ep{ep}")
        os.makedirs(args.out, exist_ok=True)
        judge.model.save_pretrained(args.out)
        say(f"saved | {args.out}")

    say("done")


if __name__ == "__main__":
    main()
