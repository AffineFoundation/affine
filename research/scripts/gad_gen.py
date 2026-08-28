#!/usr/bin/env python3
"""GAD generator, stage 1: sample candidate rollouts from the policy.

In Generative Adversarial Distillation the generator is rewarded for fooling
the discriminator, so we need several candidates per turn to rank against each
other. Candidates are drawn exactly the way the validator draws a duel rollout:
render the turn prefix through the policy's own chat template, leave the text
ending inside an open <think> block, and drive /v1/completions. That keeps a
candidate byte-comparable to a real rollout, so a reward measured here means
the same thing it would mean in a live duel.

Each rollout is split with the validator's own contract (evalsrv/chat.py):
  z = latent <think> text plus the visible THOUGHT section
  y = the last closed ```bash block
Rollouts with no closed bash block are unusable and dropped, matching the
validator, which filters on empty y.

Output is one JSON line per turn holding the teacher thought and every usable
candidate, ready for stage 2 to score.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from disc_text import as_list, normalize  # noqa: E402

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
BASH_RE = re.compile(r"```bash\n.*?\n```", re.DOTALL)
THOUGHT_LABEL_RE = re.compile(r"^\s*THOUGHT:\s*")


def split_rollout(text: str) -> tuple[str, str]:
    """Validator contract (evalsrv/chat.py): completion -> (thought, action)."""
    if THINK_CLOSE in text:
        latent, _, rest = text.partition(THINK_CLOSE)
    else:
        latent, rest = "", text
    matches = list(BASH_RE.finditer(rest))
    if not matches:
        return "", ""
    m = matches[-1]
    y = m.group(0)
    visible = THOUGHT_LABEL_RE.sub("", rest[: m.start()].strip())
    z = "\n".join(s for s in (latent.strip(), visible.strip()) if s)
    return z, y


def parse_prefix(raw):
    """Prefixes were stored with str(list_of_dicts), so JSON parsing fails."""
    if isinstance(raw, list):
        return raw
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(raw)
            if isinstance(v, list):
                return v
        except Exception:
            continue
    return None


def load_turns(path):
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["turn_id"]] = r
    return out


def load_teacher(data_dir, split):
    """turn_id -> (repo, teacher thought). One row per turn; refs repeat.

    Rows carry their own chronological split label, same as the trainer reads.
    """
    out = {}
    import glob

    for f in sorted(glob.glob(os.path.join(data_dir, "pairs_*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("split") != split:
                    continue
                repo = (r.get("repo") or "").lower()
                tid = r["turn_id"]
                if tid in out:
                    continue
                refs = [normalize(x) for x in as_list(r.get("teacher_z"))]
                ref = next((c for c in refs if c), None)
                if ref:
                    out[tid] = (repo, ref)
    return out


def build_prompt(tok, prefix_messages):
    p = tok.apply_chat_template(prefix_messages, tokenize=False,
                               add_generation_prompt=True)
    if not p.rstrip().endswith(THINK_OPEN):
        p = p + THINK_OPEN
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "disc_pairs"))
    ap.add_argument("--policy", required=True, help="HF repo of the policy being trained")
    ap.add_argument("--url", default="http://127.0.0.1:8002")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=8, help="candidates per turn")
    # validator duel defaults: temperature 0.8, 1024 thought + 768 action tokens
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--max-prompt-chars", type=int, default=60000,
                    help="skip pathological prefixes that would stall the server")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.policy)

    turns = load_turns(os.path.join(args.data, "turns.jsonl.gz"))
    teacher = load_teacher(args.data, args.split)
    tids = [t for t in teacher if t in turns]
    tids.sort()
    if args.limit:
        tids = tids[: args.limit]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["turn_id"])
                except Exception:
                    pass
        tids = [t for t in tids if t not in done]
        print(f"resuming: {len(done)} turns already sampled", flush=True)

    print(f"policy={args.policy}  split={args.split}  turns to sample={len(tids)}  "
          f"n={args.n}  temp={args.temp}", flush=True)

    lock = threading.Lock()
    fh_out = open(args.out, "a")
    stats = {"ok": 0, "empty_y": 0, "http_err": 0, "skipped": 0, "cands": 0}

    def work(tid):
        rec = turns[tid]
        msgs = parse_prefix(rec.get("prefix"))
        if not msgs:
            with lock:
                stats["skipped"] += 1
            return
        prompt = build_prompt(tok, msgs)
        if len(prompt) > args.max_prompt_chars:
            with lock:
                stats["skipped"] += 1
            return
        body = {"model": args.policy, "prompt": prompt, "n": args.n,
                "temperature": args.temp, "max_tokens": args.max_tokens}
        try:
            r = requests.post(f"{args.url}/v1/completions", json=body, timeout=1200)
            r.raise_for_status()
            choices = r.json()["choices"]
        except Exception as e:
            with lock:
                stats["http_err"] += 1
                if stats["http_err"] <= 3:
                    print(f"  http error on {tid}: {type(e).__name__}: {e}", flush=True)
            return

        cands = []
        for c in choices:
            z, y = split_rollout(c.get("text") or "")
            if not y or not z:
                with lock:
                    stats["empty_y"] += 1
                continue
            cands.append({"z": z, "y": y})
        if not cands:
            return
        repo, tz = teacher[tid]
        line = json.dumps({"turn_id": tid, "repo": repo, "teacher_z": tz,
                           "candidates": cands})
        with lock:
            fh_out.write(line + "\n")
            fh_out.flush()
            stats["ok"] += 1
            stats["cands"] += len(cands)
            if stats["ok"] % 25 == 0:
                print(f"  turns={stats['ok']}  candidates={stats['cands']}  "
                      f"unusable_rollouts={stats['empty_y']}  "
                      f"http_err={stats['http_err']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, tids))
    fh_out.close()

    print(f"\ndone: turns={stats['ok']}  candidates={stats['cands']}  "
          f"unusable={stats['empty_y']}  http_err={stats['http_err']}  "
          f"skipped_prefix={stats['skipped']}", flush=True)
    if stats["ok"]:
        print(f"mean usable candidates per turn: {stats['cands']/stats['ok']:.2f} "
              f"(asked for {args.n})", flush=True)


if __name__ == "__main__":
    main()
