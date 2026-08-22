#!/usr/bin/env python3
"""Sample k rollouts per turn from one or more vLLM servers (teacher phase 0,
or ad-hoc G sampling). Keeps the RAW completion text so SFT can train on the
exact bytes the policy emitted. Resumable: skips turn_ids already in --out.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gad_common import build_prompt, load_turn_meta, load_turns, split_rollout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--meta", default="/root/work/turn_meta.jsonl.gz")
    ap.add_argument("--urls", required=True, help="comma-separated vLLM base urls")
    ap.add_argument("--model", required=True, help="served model name to request")
    ap.add_argument("--tokenizer", required=True, help="HF repo for chat template")
    ap.add_argument("--split", default="train")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--max-prompt-chars", type=int, default=60000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="turn subsample seed")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--rescue", action="store_true",
                    help="second-pass force-close <think> + 768-token action "
                         "budget for rollouts with no valid bash block "
                         "(teacher gen only; G must not be rescued)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    urls = args.urls.split(",")
    turns = load_turns(args.turns)
    meta = load_turn_meta(args.meta)
    tids = sorted(t for t, m in meta.items()
                  if m.get("split") == args.split and t in turns)
    if args.limit and args.limit < len(tids):
        tids = random.Random(args.seed).sample(tids, args.limit)
        tids.sort()

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["turn_id"])
                except Exception:
                    pass
        tids = [t for t in tids if t not in done]
    print(f"model={args.model} split={args.split} turns={len(tids)} "
          f"(resumed past {len(done)}) k={args.k}", flush=True)

    lock = threading.Lock()
    fh_out = open(args.out, "a")
    stats = {"ok": 0, "invalid": 0, "err": 0, "skip": 0, "n": 0, "rescued": 0}
    counter = {"i": 0}

    def force_finish(url, prompt, raw):
        # Qwen3-32B frequently spends the whole budget inside <think>. Close
        # the block for it and grant a fresh action budget so the rollout
        # still yields a usable (thought, action) pair.
        cont = raw if "</think>" in raw else raw + "\n</think>\n\n"
        body = {"model": args.model, "prompt": prompt + cont, "n": 1,
                "temperature": args.temp, "max_tokens": 768}
        r = requests.post(f"{url}/v1/completions", json=body, timeout=600)
        r.raise_for_status()
        return cont + (r.json()["choices"][0].get("text") or "")

    def work(tid):
        msgs = turns[tid]
        prompt = build_prompt(tok, msgs)
        if len(prompt) > args.max_prompt_chars:
            with lock:
                stats["skip"] += 1
            return
        with lock:
            counter["i"] += 1
            url = urls[counter["i"] % len(urls)]
        body = {"model": args.model, "prompt": prompt, "n": args.k,
                "temperature": args.temp, "max_tokens": args.max_tokens}
        try:
            r = requests.post(f"{url}/v1/completions", json=body, timeout=1800)
            r.raise_for_status()
            choices = r.json()["choices"]
        except Exception as e:
            with lock:
                stats["err"] += 1
                if stats["err"] <= 5:
                    print(f"  http err {tid}: {type(e).__name__}: {e}", flush=True)
            return
        rolls = []
        for c in choices:
            raw = c.get("text") or ""
            z, y = split_rollout(raw)
            if (not z or not y) and args.rescue:
                try:
                    raw = force_finish(url, prompt, raw)
                    z, y = split_rollout(raw)
                    if z and y:
                        with lock:
                            stats["rescued"] += 1
                except Exception:
                    pass
            if not z or not y:
                with lock:
                    stats["invalid"] += 1
                continue
            rolls.append({"z": z, "y": y, "raw": raw})
        if not rolls:
            return
        rec = {"turn_id": tid, "repo": meta[tid].get("repo"), "rollouts": rolls}
        with lock:
            fh_out.write(json.dumps(rec) + "\n")
            fh_out.flush()
            stats["ok"] += 1
            stats["n"] += len(rolls)
            if stats["ok"] % 50 == 0:
                print(f"  turns={stats['ok']} rollouts={stats['n']} "
                      f"invalid={stats['invalid']} err={stats['err']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, tids))
    fh_out.close()
    tot = stats["n"] + stats["invalid"]
    print(f"DONE turns={stats['ok']} rollouts={stats['n']} invalid={stats['invalid']} "
          f"rescued={stats['rescued']} err={stats['err']} skip={stats['skip']} "
          f"valid_rate={stats['n']/max(tot,1):.3f}", flush=True)


if __name__ == "__main__":
    main()
