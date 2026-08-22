#!/usr/bin/env python3
"""Sample rollouts from a served model for turns in the SWE corpus.

Used for both sides of the adversarial setup:
  * the teacher, once, to cache the reference rollouts the discriminator learns
    to recognise (the teacher is frozen, so these are reusable forever);
  * the generator, every round, to produce the candidates it gets scored on.

Sampling copies the validator's rollout contract exactly, so a candidate here is
byte-comparable to one produced in a live duel: render the turn prefix through
the model's own chat template, leave the text ending inside an open <think>
block, and drive /v1/completions rather than the chat endpoint (server-side chat
templating would mangle the injection).

Rollouts are split into (z, y) with the validator's own rule: z is the reasoning
text, y is the last closed ```bash block. A rollout with no closed bash block is
unusable and dropped, exactly as the validator drops it.
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


def load_turns(path, limit=0):
    out = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("prefix"):
                out.append(r)
            if limit and len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True, help="turns.jsonl.gz")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--tokenizer", default=None,
                    help="chat template source; defaults to --model")
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--n", type=int, default=1, help="rollouts per turn")
    # validator duel defaults: temperature 0.8, 1024 thought + 768 action tokens
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--max-prompt-chars", type=int, default=60000,
                    help="skip pathological prefixes that would stall the server")
    ap.add_argument("--limit", type=int, default=0, help="cap turns read")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    turns = load_turns(args.turns, args.limit)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["turn_id"])
                except Exception:
                    pass
        print(f"resuming: {len(done)} turns already done", flush=True)
    todo = [t for t in turns if t["turn_id"] not in done]

    print(f"model={args.model}  turns={len(todo)}  n={args.n}  temp={args.temp}",
          flush=True)

    lock = threading.Lock()
    fh_out = open(args.out, "a")
    st = {"ok": 0, "rollouts": 0, "unusable": 0, "http_err": 0, "skipped": 0}

    def work(rec):
        msgs = parse_prefix(rec.get("prefix"))
        if not msgs:
            with lock:
                st["skipped"] += 1
            return
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        if not prompt.rstrip().endswith(THINK_OPEN):
            prompt += THINK_OPEN
        if len(prompt) > args.max_prompt_chars:
            with lock:
                st["skipped"] += 1
            return
        body = {"model": args.model, "prompt": prompt, "n": args.n,
                "temperature": args.temp, "max_tokens": args.max_tokens}
        try:
            r = requests.post(f"{args.url}/v1/completions", json=body, timeout=1800)
            r.raise_for_status()
            choices = r.json()["choices"]
        except Exception as e:
            with lock:
                st["http_err"] += 1
                if st["http_err"] <= 3:
                    print(f"  http error: {type(e).__name__}: {e}", flush=True)
            return

        cands = []
        for c in choices:
            z, y = split_rollout(c.get("text") or "")
            if z and y:
                cands.append({"z": z, "y": y})
            else:
                with lock:
                    st["unusable"] += 1
        if not cands:
            return
        line = json.dumps({"turn_id": rec["turn_id"],
                           "instance_id": rec.get("instance_id"),
                           "candidates": cands})
        with lock:
            fh_out.write(line + "\n")
            fh_out.flush()
            st["ok"] += 1
            st["rollouts"] += len(cands)
            if st["ok"] % 100 == 0:
                print(f"  turns={st['ok']}  rollouts={st['rollouts']}  "
                      f"unusable={st['unusable']}  http_err={st['http_err']}",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh_out.close()

    print(f"\ndone: turns={st['ok']}  rollouts={st['rollouts']}  "
          f"unusable={st['unusable']}  http_err={st['http_err']}  "
          f"skipped={st['skipped']}", flush=True)


if __name__ == "__main__":
    main()
