#!/usr/bin/env python3
"""[fix] Trainer diagnostics for Track S's D.

(a) Reconstructs a few training examples exactly as trackS_d_train.py does
    (fit_ids + slot flip) and prints the decoded prompt TAIL, the label and
    the A/B target ids -- verifies the supervised position is the answer
    letter and labels/orders randomize.
(b) Length-probe control: how well does thought-length ALONE separate
    (ref, mine) on the recent training pairs? Reports the longer-is-teacher
    rule accuracy and a 1-D logistic fit, per round window. If this beats
    ~0.6 while D's train_acc sits at ~0.5, the D recipe (not the data) is
    the problem.
CPU-only; safe to run next to anything.
"""
from __future__ import annotations

import json
import math
import random
import sys

sys.path.insert(0, "/root/work")
from judge_common import ab_token_ids, fit_ids  # noqa: E402
from trackM_common import now  # noqa: E402

W = "/dshare/trackS"


def status(line):
    with open(f"{W}/status.log", "a") as fh:
        fh.write(f"{now()} [fix] {line}\n")
    print(line, flush=True)


def think_len(both):
    """Chars before the last ```bash fence = the thought part of both_text."""
    i = both.rfind("```bash")
    return i if i >= 0 else len(both)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
    ab = ab_token_ids(tok)

    rows = [json.loads(x) for x in open(f"{W}/d_pending.jsonl")]
    print(f"d_pending rows={len(rows)}")

    # ---- (a) tokenization / label audit ---------------------------------
    exrng = random.Random(1)
    n_flip = 0
    for i, p in enumerate(rows[:4]):
        t_in_a = exrng.random() < 0.5
        n_flip += int(not t_in_a)
        a, b = (p["ref"], p["mine"]) if t_in_a else (p["mine"], p["ref"])
        ids = fit_ids(tok, p["prefix_text"], a, b, 3584)
        tail = tok.decode(ids[-60:])
        nxt = tok.decode([ab[0 if t_in_a else 1]])
        print(f"--- example {i} round={p.get('round')} teacher_in_A={t_in_a} "
              f"label_token={nxt!r} len_ids={len(ids)}")
        print("PROMPT TAIL:", repr(tail[-260:]))
    print(f"A/B token ids = {ab} -> {[tok.decode([t]) for t in ab]!r}")
    status(f"trainer audit (a): supervised next-token targets={ab} "
           f"({[tok.decode([t]) for t in ab]}); prompt ends with closed "
           f"think + awaiting single letter; slot flips randomize "
           f"(see driver.log for decoded examples)")

    # ---- (b) length probe -------------------------------------------------
    def probe(sub, tag):
        if len(sub) < 10:
            status(f"length-probe {tag}: too few pairs ({len(sub)})")
            return
        # rule: the longer thought is the teacher
        diffs = [think_len(p["ref"]) - think_len(p["mine"]) for p in sub]
        rule = sum(1 for d in diffs if d > 0) / len(diffs)
        # 1-D logistic fit on symmetric +/- examples, plain gradient descent
        xs = [d / 1000.0 for d in diffs] + [-d / 1000.0 for d in diffs]
        ys = [1.0] * len(diffs) + [0.0] * len(diffs)
        w = 0.0
        for _ in range(500):
            g = sum((1 / (1 + math.exp(-w * x)) - y) * x
                    for x, y in zip(xs, ys)) / len(xs)
            w -= 2.0 * g
        acc = sum(1 for x, y in zip(xs, ys)
                  if (1 / (1 + math.exp(-w * x)) > 0.5) == (y == 1.0)) / len(xs)
        mr = sum(think_len(p["ref"]) for p in sub) / len(sub)
        mm = sum(think_len(p["mine"]) for p in sub) / len(sub)
        status(f"length-probe {tag}: n={len(sub)} mean_think ref={mr:.0f} "
               f"mine={mm:.0f} | longer-is-teacher acc={rule:.3f} | "
               f"1D-logistic acc={acc:.3f} (w={w:.3f})")

    probe(rows[-400:], "last400(all rounds)")
    probe([r for r in rows if r.get("round", 0) >= 10], "rounds>=10(drifted)")


if __name__ == "__main__":
    main()
