#!/usr/bin/env python3
"""Diagnose why reason-arm lift is systematically negative.

For a few train turns:
  1. show the teacher action target y (first 400 chars)
  2. generate one miner thought z, show it
  3. compute lift as the loop does (z + "\n\n" prepended to y)
  4. compute lift for an "oracle" thought = first 300 chars of y itself
     (if even a verbatim prefix-hint scores negative, the concatenation
     format is the problem, not thought quality)
"""
import json
import sys

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
cache = load_rollout_cache("/root/work/teacher_train.jsonl")


def lp_of(url, model, tok, prompt, completion):
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    body = {"model": model, "prompt": prompt + completion, "max_tokens": 1,
            "temperature": 0.0, "echo": True, "logprobs": 0}
    r = requests.post(f"{url}/v1/completions", json=body, timeout=600)
    r.raise_for_status()
    lps = r.json()["choices"][0]["logprobs"]["token_logprobs"]
    tail = [x for x in lps[n_prompt:len(lps) - 1] if x is not None]
    return sum(tail), max(len(tail), 1)


def plain_prompt(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


tids = [t for t in cache if t in turns][:4]
for tid in tids:
    y = cache[tid][0]["raw"]
    p = plain_prompt(ttok, turns[tid])
    lp0, n0 = lp_of(TEACHER_URL, TEACHER, ttok, p, y)

    ep = plain_prompt(mtok, turns[tid] + [
        {"role": "user",
         "content": "Briefly explain which action should be taken next and why."}])
    if len(ep) > 60000:
        print(f"== {tid}: elicit prompt too long, skip")
        continue
    r = requests.post(f"{MINER_URL}/v1/completions",
                      json={"model": MINER, "prompt": ep, "n": 1,
                            "temperature": 0.9, "max_tokens": 256},
                      timeout=600)
    z = r.json()["choices"][0]["text"].strip()

    # --- proposed think-slot construction ---
    m = y.find("</think>")
    if m < 0:
        print(f"== {tid}: no </think> in y, skip")
        continue
    y_think = y[:m].strip()
    y_act = y[m + len("</think>"):].lstrip("\n")
    zc = z.replace("<think>", " ").replace("</think>", " ").strip()

    blp, bn = lp_of(TEACHER_URL, TEACHER, ttok, p + "</think>\n\n", y_act)
    zlp, zn = lp_of(TEACHER_URL, TEACHER, ttok, p + zc + "\n</think>\n\n", y_act)
    olp, on = lp_of(TEACHER_URL, TEACHER, ttok,
                    p + y_think + "\n</think>\n\n", y_act)
    lift_z2 = (zlp - blp) / max(min(bn, zn), 1)
    lift_o2 = (olp - blp) / max(min(bn, on), 1)

    print("=" * 78)
    print(f"tid={tid}")
    print(f"  Y_ACT HEAD: {json.dumps(y_act[:200])}")
    print(f"  MINER Z: {json.dumps(zc[:300])}")
    print(f"  NEW lift(miner z)       = {lift_z2:+.4f}  (base={blp:.1f}/{bn}, z={zlp:.1f})")
    print(f"  NEW lift(oracle=y think) = {lift_o2:+.4f}  (o={olp:.1f})")
