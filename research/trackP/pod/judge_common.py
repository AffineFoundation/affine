#!/usr/bin/env python3
"""Shared judge-prompt construction for Track P's discriminator.

The SAME token sequence must be produced by the vLLM scoring path (driver)
and the HF training path (d_train.py), or train/serve would silently skew.
Prompt = chat-templated user message [prefix + candidate A + candidate B +
QUESTION], ending right before the judge's one-token A/B answer. Thinking is
forced closed (empty <think></think>) so the first generated/supervised
token IS the letter.
"""
from __future__ import annotations

QUESTION = (
    "Above is the transcript of a software-engineering task in progress.\n"
    "Two assistants each proposed the next step. One of them is the reference "
    "assistant.\n\nWhich one is the reference assistant? Reply with a single "
    "letter, A or B."
)

EMPTY_THINK = "<think>\n\n</think>\n\n"


def ab_token_ids(tok):
    out = []
    for letter in ("A", "B"):
        for form in (letter, " " + letter):
            enc = tok(form, add_special_tokens=False)["input_ids"]
            if len(enc) == 1:
                out.append(enc[0])
                break
        else:
            raise SystemExit(f"cannot map {letter!r} to a single token")
    return out


def render(tok, body):
    msgs = [{"role": "user", "content": body}]
    try:
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    # some templates ignore enable_thinking; force the closed think block so
    # the next token is the answer letter
    if "</think>" not in text[-40:]:
        text = text + EMPTY_THINK
    return text


def fit_ids(tok, prefix, a_text, b_text, max_len):
    """Token ids for one judge example, prefix left-truncated to fit max_len."""
    chars = min(len(prefix), max_len * 6)
    ids = None
    for _ in range(5):
        body = (f"{prefix[-chars:] if chars else ''}\n\n"
                f"=== Candidate A ===\n{a_text}\n\n"
                f"=== Candidate B ===\n{b_text}\n\n{QUESTION}")
        ids = tok(render(tok, body), add_special_tokens=False)["input_ids"]
        if len(ids) <= max_len:
            return ids
        over = len(ids) - max_len
        nxt = chars - int(over * 4.0) - 256
        if nxt <= 0:
            break
        chars = nxt
    return ids[-max_len:]
