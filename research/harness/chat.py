"""Prompt construction: natural generation, thought injection, teacher forcing.

Turn contracts:
  production dialects (bash, tool_call, boxed — affine.dialects, the same
  registry evalsrv scores against):
    z = latent <think> + visible THOUGHT: section
    y = final complete action span in the turn's dialect
        (```bash block / <tool_call>…</tool_call> / \\boxed{…})
    inject body = </think>\\nTHOUGHT: {z}\\n\\n{y}   (identical across dialects)

  tool (D_tau2-era research-only contract, kept for archived replay):
    z = latent <think> + <thinking>…</thinking> content
    y = compact tool-call JSON {"name":…,"arguments":…}
    inject body = </think>\\n<thinking>{z}</thinking>\\n{y}

We always render through the model's own chat template to a string and drive
/v1/completions directly, so injection and forcing are byte-exact.
"""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from functools import lru_cache

from transformers import AutoTokenizer

from affine import dialects

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
BASH_RE = dialects.BASH_PATTERN
THOUGHT_LABEL_RE = re.compile(r"^\s*THOUGHT:\s*")
THINKING_RE = re.compile(
    r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE
)
LEGACY_TOOL_KIND = "tool"

# Per-turn action contract. Set by runner from turn["action_kind"].
action_kind: ContextVar[str] = ContextVar("action_kind", default="bash")


@lru_cache(maxsize=32)
def get_tokenizer(repo: str):
    return AutoTokenizer.from_pretrained(repo)


def gen_prompt(repo: str, prefix_messages: list[dict]) -> str:
    """Prompt for a natural rollout; always ends inside an open <think> block."""
    tok = get_tokenizer(repo)
    p = tok.apply_chat_template(
        prefix_messages, tokenize=False, add_generation_prompt=True
    )
    if not p.rstrip().endswith(THINK_OPEN):
        p = p + THINK_OPEN
    return p


def inject_prompt(repo: str, prefix_messages: list[dict], thoughts: str) -> str:
    """Plant `thoughts` as the full reasoning channel; leave cursor at action."""
    kind = action_kind.get()
    base = gen_prompt(repo, prefix_messages) + THINK_CLOSE
    if kind == LEGACY_TOOL_KIND:
        return base + "\n<thinking>" + thoughts + "</thinking>\n"
    return base + "\nTHOUGHT: " + thoughts + "\n\n"


def force_text(repo: str, prefix_messages: list[dict], thoughts: str, action: str) -> str:
    """Full text whose action span we score via echo+logprobs."""
    return inject_prompt(repo, prefix_messages, thoughts) + action


def _extract_tool_json(text: str) -> str | None:
    """Return compact JSON string of the first tool-call object, or None."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return None


def think_closed(text: str) -> bool:
    """Research twin of affine/evalsrv/chat.py::think_closed."""
    return THINK_CLOSE in text


def split_rollout(text: str, require_think_close: bool = False
                  ) -> tuple[str, str]:
    """Split a completion (started inside <think>) into (z, y).

    Returns ("", "") when no valid action is found — or, with
    require_think_close (staged v7 knob, twin of affine/evalsrv/chat.py),
    when the completion never emitted </think>.
    """
    if THINK_CLOSE in text:
        latent, _, rest = text.partition(THINK_CLOSE)
    elif require_think_close:
        return "", ""
    else:
        latent, rest = "", text

    kind = action_kind.get()
    if kind == LEGACY_TOOL_KIND:
        y = _extract_tool_json(rest) or _extract_tool_json(text)
        if not y:
            return "", ""
        # Prefer explicit <thinking> content; else text before the JSON.
        m = THINKING_RE.search(rest)
        if m:
            visible = m.group(1).strip()
        else:
            # strip everything from first tool JSON onward
            idx = rest.find(y) if y in rest else -1
            visible = rest[:idx].strip() if idx >= 0 else rest.strip()
            visible = THINKING_RE.sub("", visible).strip()
        z = "\n".join(s for s in (latent.strip(), visible) if s)
        return z, y

    # Production dialects: same split rule as evalsrv/chat.py.
    before, y = dialects.split_action(rest, kind)
    if not y:
        return "", ""
    visible = THOUGHT_LABEL_RE.sub("", before.strip())
    z = "\n".join(s for s in (latent.strip(), visible.strip()) if s)
    return z, y


def extract_action(text: str) -> str:
    """Pull the action out of an injected rollout completion."""
    kind = action_kind.get()
    if kind == LEGACY_TOOL_KIND:
        return _extract_tool_json(text) or ""
    return dialects.last_action(text, kind)
