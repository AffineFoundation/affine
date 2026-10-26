#!/usr/bin/env python3
"""Run the held-out collect path on a few tids, printing failure reasons."""
import sys
import traceback

import requests

sys.path.insert(0, "/root/work")
from gad_common import load_rollout_cache, load_turns  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TEACHER_URL = "http://127.0.0.1:8010"
MINER_URL = "http://127.0.0.1:8021"
TEACHER = "Qwen/Qwen3.6-35B-A3B"
MINER = "Qwen/Qwen3-8B"

ttok = AutoTokenizer.from_pretrained(TEACHER)
mtok = AutoTokenizer.from_pretrained(MINER)
turns = load_turns("/root/work/turns.jsonl.gz")
hcache = load_rollout_cache("/root/work/teacher_heldout.jsonl")


def plain_prompt(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


def lp_of(url, model, tok, prompt, completion):
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    body = {"model": model, "prompt": prompt + completion, "max_tokens": 1,
            "temperature": 0.0, "echo": True, "logprobs": 0}
    r = requests.post(f"{url}/v1/completions", json=body, timeout=600)
    r.raise_for_status()
    lps = r.json()["choices"][0]["logprobs"]["token_logprobs"]
    tail = [x for x in lps[n_prompt:len(lps) - 1] if x is not None]
    return sum(tail), max(len(tail), 1)


held = [t for t in hcache if t in turns][:8]
for tid in held:
    try:
        raw = hcache[tid][0]["raw"]
        m = raw.find("</think>")
        if m < 0:
            print(f"{tid}: SKIP no </think>")
            continue
        y = raw[m + len("</think>"):].lstrip("\n")
        p = plain_prompt(ttok, turns[tid])
        ep = plain_prompt(mtok, turns[tid] + [
            {"role": "user",
             "content": "Briefly explain which action should be taken next and why."}])
        if len(ep) > 60000:
            print(f"{tid}: SKIP ep len {len(ep)}")
            continue
        r = requests.post(f"{MINER_URL}/v1/completions",
                          json={"model": MINER, "prompt": ep, "n": 1,
                                "temperature": 0.9, "max_tokens": 256},
                          timeout=600)
        r.raise_for_status()
        z = r.json()["choices"][0]["text"].strip()
        if not z:
            print(f"{tid}: EMPTY z")
            continue
        zc = z.replace("<think>", " ").replace("</think>", " ").strip()
        lp0, n0 = lp_of(TEACHER_URL, TEACHER, ttok, p + "</think>\n\n", y)
        lp1, n1 = lp_of(TEACHER_URL, TEACHER, ttok,
                        p + zc + "\n</think>\n\n", y)
        print(f"{tid}: OK lift={(lp1 - lp0) / max(min(n0, n1), 1):+.4f}")
    except Exception as e:
        print(f"{tid}: FAIL {type(e).__name__}: {e}")
        traceback.print_exc()
