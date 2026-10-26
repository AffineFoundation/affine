#!/usr/bin/env python3
"""Check the hint-channel lift construction has positive headroom.

oracle 1: the teacher's own think content for this turn
oracle 2: a one-line description of the exact action (bash body)
"""
import json
import sys

import requests

sys.path.insert(0, "/root/work")
from gad_common import load_rollout_cache, load_turns  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TEACHER_URL = "http://127.0.0.1:8010"
TEACHER = "Qwen/Qwen3.6-35B-A3B"
ttok = AutoTokenizer.from_pretrained(TEACHER)
turns = load_turns("/root/work/turns.jsonl.gz")
cache = load_rollout_cache("/root/work/teacher_train.jsonl")


def plain_prompt(msgs):
    return ttok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)


def lp_of(prompt, completion):
    n_prompt = len(ttok(prompt, add_special_tokens=False)["input_ids"])
    body = {"model": TEACHER, "prompt": prompt + completion, "max_tokens": 1,
            "temperature": 0.0, "echo": True, "logprobs": 0}
    r = requests.post(f"{TEACHER_URL}/v1/completions", json=body, timeout=600)
    r.raise_for_status()
    lps = r.json()["choices"][0]["logprobs"]["token_logprobs"]
    tail = [x for x in lps[n_prompt:len(lps) - 1] if x is not None]
    return sum(tail), max(len(tail), 1)


def hinted(tid, z):
    return plain_prompt(turns[tid] + [{
        "role": "user",
        "content": "Hint from a collaborator about the next action:\n"
                   + z.strip()}])


tids = [t for t in cache if t in turns][5:11]
for tid in tids:
    y = cache[tid][0]["raw"]
    p = plain_prompt(turns[tid])
    lp0, n0 = lp_of(p, y)

    m = y.find("</think>")
    o1 = y[:m].strip() if m > 0 else ""
    body = y[m:] if m > 0 else y
    i = body.find("```bash")
    j = body.find("```", i + 7) if i >= 0 else -1
    o2 = ("The next action is to run:\n" + body[i:j + 3]) if j > 0 else ""

    out = [f"tid={tid} lp0={lp0:.1f}/{n0}"]
    for name, z in (("teacher-think", o1), ("action-desc", o2)):
        if not z:
            out.append(f"  {name}: n/a")
            continue
        lp1, n1 = lp_of(hinted(tid, z), y)
        out.append(f"  {name}: lift={(lp1 - lp0) / max(min(n0, n1), 1):+.4f}")
    print("\n".join(out), flush=True)
