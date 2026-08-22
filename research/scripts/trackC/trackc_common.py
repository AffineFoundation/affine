#!/usr/bin/env python3
"""Shared helpers for Track C (frozen-discriminator GAD control).

Track C runs the same generator loop as Track B but trains the discriminator
exactly once (phase 1) and freezes it. Everything here is shared between the
rollout sampler, the one-shot D trainer, and the phase-2 loop so the three
stages provably use the same parsing, normalisation, and prompt formats.
"""
from __future__ import annotations

import ast
import glob
import gzip
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from disc_text import normalize  # noqa: E402

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
BASH_RE = re.compile(r"```bash\n.*?\n```", re.DOTALL)
THOUGHT_LABEL_RE = re.compile(r"^\s*THOUGHT:\s*")
WORD_RE = re.compile(r"[a-z0-9_]+")


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
    """Prefixes were stored with str(list_of_dicts), so JSON parsing may fail."""
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


def load_splits(data_dir):
    """turn_id -> 'train'/'test' from the pairs shards (repo-level split)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "pairs_*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                tid, sp = r.get("turn_id"), r.get("split")
                if tid and sp and tid not in out:
                    out[tid] = sp
    return out


def build_gen_prompt(tok, prefix_messages):
    """Render the prefix ending inside an open <think> block (validator style)."""
    p = tok.apply_chat_template(prefix_messages, tokenize=False,
                                add_generation_prompt=True)
    if not p.rstrip().endswith(THINK_OPEN):
        p = p + THINK_OPEN
    return p


def candidate_text(z: str, y: str) -> str:
    """The 'both' channel block D sees: normalized thought, blank line, action."""
    return (normalize(z) + "\n\n" + normalize(y)).strip()


def word_set(s: str) -> set:
    return set(WORD_RE.findall((s or "").lower()))


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def action_inner(y: str) -> str:
    """Strip the ```bash fence so agreement compares commands, not markup."""
    s = (y or "").strip()
    if s.startswith("```bash"):
        s = s[len("```bash"):]
    if s.endswith("```"):
        s = s[: -3]
    return s.strip()


def sample_completions(urls, model, jobs, n, temp, max_tokens,
                       workers=32, timeout=1800, retries=3):
    """jobs: list of (key, prompt). Returns key -> list of {text, finish} or None.

    urls is a list of interchangeable /v1 endpoints; jobs round-robin over them.
    """
    out = {}
    lock = threading.Lock()

    def work(item):
        i, (key, prompt) = item
        body = {"model": model, "prompt": prompt, "n": n,
                "temperature": temp, "max_tokens": max_tokens}
        for attempt in range(retries):
            url = urls[(i + attempt) % len(urls)]
            try:
                r = requests.post(f"{url}/v1/completions", json=body,
                                  timeout=timeout)
                r.raise_for_status()
                ch = r.json()["choices"]
                with lock:
                    out[key] = [{"text": c.get("text") or "",
                                 "finish": c.get("finish_reason")} for c in ch]
                return
            except Exception:
                time.sleep(4 * (attempt + 1))
        with lock:
            out[key] = None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, enumerate(jobs)))
    return out


def cand_record(text: str, finish: str) -> dict:
    z, y = split_rollout(text)
    return {"raw": text, "z": z, "y": y, "finish": finish,
            "valid": bool(z and y)}


def load_rollouts(path, valid_only=True):
    """trackc_gen output -> turn_id -> [candidate dicts]."""
    out = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            cands = r.get("candidates") or []
            if valid_only:
                cands = [c for c in cands if c.get("valid")]
            if cands:
                out[r["turn_id"]] = cands
    return out


def wait_healthy(url, timeout=1800, model=None):
    """Block until a vLLM server answers /v1/models (optionally lists model)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{url}/v1/models", timeout=10)
            if r.status_code == 200:
                if model is None:
                    return True
                ids = [m["id"] for m in r.json().get("data", [])]
                if model in ids:
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False
