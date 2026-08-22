#!/usr/bin/env python3
"""Sample rollouts from a served model (teacher or generator) for Track C.

Like gad_gen.py but: turn selection comes from an explicit turn-list JSON, all
choices are kept (with a valid flag) so valid-action rate is measurable, the
raw completion text is preserved for SFT, and requests can fan out over
several interchangeable server replicas (--url may repeat).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from trackc_common import (build_gen_prompt, cand_record, load_turns,
                           parse_prefix, sample_completions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/work/research/data/disc_pairs")
    ap.add_argument("--tokenizer", required=True,
                    help="HF repo whose chat template renders the prompt")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--url", action="append", required=True)
    ap.add_argument("--turns", required=True, help="JSON list of turn_ids")
    ap.add_argument("--shard", default=None, help="'i/n' slice of the turn list")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--max-prompt-chars", type=int, default=60000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    turns = load_turns(os.path.join(args.data, "turns.jsonl.gz"))
    tids = json.load(open(args.turns))
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        tids = tids[i::n]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["turn_id"])
                except Exception:
                    pass
        tids = [t for t in tids if t not in done]
    print(f"model={args.model} turns to sample={len(tids)} "
          f"(resumed past {len(done)}) n={args.n}", flush=True)

    jobs = []
    for tid in tids:
        msgs = parse_prefix(turns[tid].get("prefix"))
        if not msgs:
            continue
        prompt = build_gen_prompt(tok, msgs)
        if len(prompt) > args.max_prompt_chars:
            continue
        jobs.append((tid, prompt))

    lock = threading.Lock()
    fh_out = open(args.out, "a")
    stats = {"turns": 0, "valid": 0, "total": 0, "fail": 0}

    def work(job):
        tid, prompt = job
        res = sample_completions(args.url, args.model, [(tid, prompt)],
                                 args.n, args.temp, args.max_tokens,
                                 workers=1)
        choices = res.get(tid)
        with lock:
            if choices is None:
                stats["fail"] += 1
                return
            cands = [cand_record(c["text"], c["finish"]) for c in choices]
            fh_out.write(json.dumps({"turn_id": tid, "candidates": cands}) + "\n")
            fh_out.flush()
            stats["turns"] += 1
            stats["total"] += len(cands)
            stats["valid"] += sum(c["valid"] for c in cands)
            if stats["turns"] % 100 == 0:
                print(f"  turns={stats['turns']} valid={stats['valid']}/"
                      f"{stats['total']} fail={stats['fail']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))
    fh_out.close()
    print(f"done: {stats}", flush=True)


if __name__ == "__main__":
    main()
