#!/usr/bin/env python3
"""Sample thoughts from the current adapter, score with hint lift, show them."""
import sys

import requests

sys.path.insert(0, "/root/work")
from gad_common import load_rollout_cache, load_turns  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TEACHER_URL = "http://127.0.0.1:8010"
MINER_URL = "http://127.0.0.1:8021"
TEACHER = "Qwen/Qwen3.6-35B-A3B"
TAG = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"

ttok = AutoTokenizer.from_pretrained(TEACHER)
mtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
turns = load_turns("/root/work/turns.jsonl.gz")
cache = load_rollout_cache("/root/work/teacher_train.jsonl")


def pp(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False,
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


tids = [t for t in cache if t in turns][20:32]
results = []
for tid in tids:
    y = cache[tid][0]["raw"]
    p = pp(ttok, turns[tid])
    ep = pp(mtok, turns[tid] + [
        {"role": "user",
         "content": "Briefly explain which action should be taken next and why."}])
    if len(ep) > 60000:
        continue
    r = requests.post(f"{MINER_URL}/v1/completions",
                      json={"model": TAG, "prompt": ep, "n": 4,
                            "temperature": 0.9, "max_tokens": 256},
                      timeout=600)
    r.raise_for_status()
    lp0, n0 = lp_of(p, y)
    for c in r.json()["choices"]:
        z = (c.get("text") or "").strip()
        if not z:
            continue
        hp = pp(ttok, turns[tid] + [{
            "role": "user",
            "content": "Hint from a collaborator about the next action:\n" + z}])
        lp1, n1 = lp_of(hp, y)
        results.append(((lp1 - lp0) / max(min(n0, n1), 1), tid, z))

results.sort(key=lambda x: -x[0])
lifts = [v for v, _, _ in results]
n_pos = sum(1 for v in lifts if v > 0)
print(f"tag={TAG} n={len(lifts)} pos={n_pos} "
      f"max={lifts[0]:+.4f} med={lifts[len(lifts)//2]:+.4f} min={lifts[-1]:+.4f}")
print("\n--- BEST ---")
for v, tid, z in results[:2]:
    print(f"[{v:+.4f}] {tid}\n{z[:500]}\n")
print("--- WORST ---")
for v, tid, z in results[-1:]:
    print(f"[{v:+.4f}] {tid}\n{z[:400]}\n")
