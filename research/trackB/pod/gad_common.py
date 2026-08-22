#!/usr/bin/env python3
"""Shared helpers for the Track B GAN-style distillation loop.

Rollout contract matches the validator (and gad_gen.py):
  prompt = prefix rendered through the policy's chat template, ending inside
  an open <think> block; completion is split into
    z = latent <think> text + visible THOUGHT text
    y = LAST closed ```bash fenced block
  rollouts with no closed bash block are invalid.
"""
from __future__ import annotations

import ast
import gzip
import json
import re

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
BASH_RE = re.compile(r"```bash\n.*?\n```", re.DOTALL)
THOUGHT_LABEL_RE = re.compile(r"^\s*THOUGHT:\s*")

# same normalisation as disc_text.normalize (kept inline so the pod needs no
# extra file): strip reasoning markers that leak the source model
MARKER_RE = re.compile(r"</?\s*(think|thinking|thought|reasoning|answer)\s*>", re.I)
WS_RE = re.compile(r"\n{3,}")


def normalize(s: str) -> str:
    s = MARKER_RE.sub("", s or "")
    s = WS_RE.sub("\n\n", s)
    return s.strip()


def split_rollout(text: str) -> tuple[str, str]:
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
            msgs = parse_prefix(r.get("prefix"))
            if msgs:
                out[r["turn_id"]] = msgs
    return out


def load_turn_meta(path):
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["turn_id"]] = r
    return out


def build_prompt(tok, prefix_messages):
    p = tok.apply_chat_template(prefix_messages, tokenize=False,
                                add_generation_prompt=True)
    if not p.rstrip().endswith(THINK_OPEN):
        p = p + THINK_OPEN
    return p


def load_rollout_cache(path):
    """teacher/G rollout cache: turn_id -> list of {z, y, raw}."""
    out = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("rollouts"):
                out[r["turn_id"]] = r["rollouts"]
    return out


def both_text(z: str, y: str) -> str:
    """One candidate block: thinking + action, normalised (leak control)."""
    return (normalize(z) + "\n\n" + normalize(y)).strip()


def bash_body(y: str) -> str:
    """Strip the ```bash fence and collapse whitespace for agreement checks."""
    s = y.strip()
    if s.startswith("```bash"):
        s = s[len("```bash"):]
    if s.endswith("```"):
        s = s[: -3]
    return " ".join(s.split())


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)
