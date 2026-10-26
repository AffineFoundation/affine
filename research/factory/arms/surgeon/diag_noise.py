#!/usr/bin/env python3
"""Diagnose the surgeon bank: is the NULL-family delta real or vLLM noise?

1. noise floor: score the SAME (prompt, completion) twice; |lp_a-lp_b|/n
   should be ~0 if the server is deterministic. Whatever it is, that is the
   measurement noise the bank deltas sit on.
2. family tails: fraction of banked pairs whose |delta| clears 2x/4x the
   NULL median (signal lives in the tail if the teacher only reacts to some
   edits).
"""
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, "/root/work")
from gad_common import build_prompt, load_rollout_cache, load_turns  # noqa: E402

URL = "http://127.0.0.1:8010"
MODEL = "Qwen/Qwen3.6-35B-A3B"


def lp(tok, prompt, completion):
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    body = {"model": MODEL, "prompt": prompt + completion, "max_tokens": 1,
            "temperature": 0.0, "echo": True, "logprobs": 0}
    r = requests.post(f"{URL}/v1/completions", json=body, timeout=600)
    r.raise_for_status()
    lps = r.json()["choices"][0]["logprobs"]["token_logprobs"]
    tail = [x for x in lps[n_prompt:len(lps) - 1] if x is not None]
    return sum(tail), len(tail)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    turns = load_turns("/root/work/turns.jsonl.gz")
    cache = load_rollout_cache("/root/work/teacher_train.jsonl")
    tids = [t for t in list(cache)[:40] if t in turns]

    def rep(tid):
        prompt = build_prompt(tok, turns[tid])
        if len(prompt) > 60000:
            return None
        a = cache[tid][0]["raw"]
        l1, n1 = lp(tok, prompt, a)
        l2, n2 = lp(tok, prompt, a)
        if n1 == 0:
            return None
        return abs(l1 - l2) / max(min(n1, n2), 1)

    with ThreadPoolExecutor(max_workers=8) as ex:
        diffs = [d for d in ex.map(rep, tids) if d is not None]
    diffs.sort()
    print(f"repeat-noise n={len(diffs)} median={statistics.median(diffs):.5f} "
          f"p90={diffs[int(0.9 * len(diffs))]:.5f} max={diffs[-1]:.5f}")

    rows = [json.loads(l) for l in open("/root/work/bank_train.jsonl")]
    null_med = statistics.median(
        [abs(r["delta_per_tok"]) for r in rows if r["family"] == "NULL"])
    print(f"NULL median = {null_med:.5f}")
    for fam in ("NULL", "SYM", "ERR", "NUM"):
        v = [abs(r["delta_per_tok"]) for r in rows if r["family"] == fam]
        if not v:
            continue
        f2 = sum(1 for x in v if x > 2 * null_med) / len(v)
        f4 = sum(1 for x in v if x > 4 * null_med) / len(v)
        p90 = sorted(v)[int(0.9 * len(v))]
        print(f"{fam:5s} n={len(v):4d} p90={p90:.5f} "
              f">2xNULLmed={f2:.2f} >4xNULLmed={f4:.2f}")


if __name__ == "__main__":
    main()
