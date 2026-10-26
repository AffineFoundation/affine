#!/usr/bin/env python3
"""Sample abduction thoughts from the miner and measure copy-paste overlap.

For each sampled turn: build the same elicit prompt the RAFT loop uses,
generate one thought z from the given miner model tag, and compare z with
the observation e (the last user message). Copy metric: fraction of z's
word 8-grams that appear verbatim in e.

Usage (on pod):
  /root/trainenv/bin/python sample_thoughts.py --tag <model-or-adapter-tag> \
      --n 12 --out /root/work/thought_samples_<tag>.jsonl
"""
import argparse
import json
import random
import sys

import requests

sys.path.insert(0, "/root/work")
from gad_common import load_rollout_cache, load_turns  # noqa: E402

ELICIT = ("Briefly state the key facts of the current environment "
          "state that determine what happens next.")


def ngram_containment(z: str, e: str, n: int = 8) -> float:
    zw, ew = z.split(), e.split()
    if len(zw) < n:
        return 1.0 if " ".join(zw) in e else 0.0
    egrams = {" ".join(ew[i:i + n]) for i in range(len(ew) - n + 1)}
    zgrams = [" ".join(zw[i:i + n]) for i in range(len(zw) - n + 1)]
    return sum(g in egrams for g in zgrams) / len(zgrams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--miner-url", default="http://127.0.0.1:8021")
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--heldout", default="/root/work/teacher_heldout.jsonl")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-z-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    turns = load_turns(args.turns)
    hcache = load_rollout_cache(args.heldout)
    tids = [t for t in hcache if t in turns
            and turns[t] and turns[t][-1].get("role") == "user"]
    random.Random(args.seed).shuffle(tids)

    out = open(args.out, "w")
    done = 0
    for tid in tids:
        if done >= args.n:
            break
        msgs = turns[tid] + [{"role": "user", "content": ELICIT}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        if len(prompt) > 60000:
            continue
        r = requests.post(f"{args.miner_url}/v1/completions",
                          json={"model": args.tag, "prompt": prompt, "n": 1,
                                "temperature": 0.9,
                                "max_tokens": args.max_z_tokens},
                          timeout=600)
        r.raise_for_status()
        z = (r.json()["choices"][0].get("text") or "").strip()
        if not z:
            continue
        e = (turns[tid][-1].get("content") or "").strip()[:8000]
        rec = {"tid": tid, "tag": args.tag,
               "copy8": round(ngram_containment(z, e), 4),
               "z_words": len(z.split()), "e_words": len(e.split()),
               "z": z, "e_head": e[:1200]}
        out.write(json.dumps(rec) + "\n")
        done += 1
        print(f"{tid} copy8={rec['copy8']:.3f} z_words={rec['z_words']}",
              flush=True)
    out.close()
    print(f"wrote {done} samples to {args.out}")


if __name__ == "__main__":
    main()
