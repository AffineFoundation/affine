"""Prompt construction: natural generation, thought injection, teacher forcing.

Turn contract for the SWE turn domain:
  z (thoughts) = all reasoning text: latent <think> content plus the visible
                 THOUGHT section, normalized to plain text.
  y (action)   = the final closed ```bash fenced block — the discrete action
                 that drives the environment.

We always render through the model's own chat template to a string and drive
/v1/completions directly, so injection and forcing are byte-exact and cannot
be mangled by server-side chat templating. Injection uses a canonical
assistant body (`</think>\nTHOUGHT: {z}\n\n{y}`) identical across model
families, so forced logprob differences reflect the models, not the rendering.
"""

from __future__ import annotations

import re
from functools import lru_cache

from transformers import AutoTokenizer

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
BASH_RE = re.compile(r"```bash\n.*?\n```", re.DOTALL)
THOUGHT_LABEL_RE = re.compile(r"^\s*THOUGHT:\s*")


@lru_cache(maxsize=8)
def get_tokenizer(repo: str, revision: str | None = None):
    return AutoTokenizer.from_pretrained(repo, revision=revision)


def gen_prompt(repo: str, revision: str | None, prefix_messages: list[dict]) -> str:
    """Prompt for a natural rollout; always ends inside an open <think> block."""
    tok = get_tokenizer(repo, revision)
    p = tok.apply_chat_template(
        prefix_messages, tokenize=False, add_generation_prompt=True
    )
    if not p.rstrip().endswith(THINK_OPEN):
        p = p + THINK_OPEN
    return p


def inject_prompt(repo: str, revision: str | None,
                  prefix_messages: list[dict], thoughts: str) -> str:
    """Prompt where `thoughts` are planted as the full reasoning channel."""
    return (
        gen_prompt(repo, revision, prefix_messages)
        + THINK_CLOSE + "\nTHOUGHT: " + thoughts + "\n\n"
    )


def force_text(repo: str, revision: str | None, prefix_messages: list[dict],
               thoughts: str, action: str) -> str:
    """Full text whose action span we score via echo+logprobs."""
    return inject_prompt(repo, revision, prefix_messages, thoughts) + action


def thought_text(repo: str, revision: str | None,
                 prefix_messages: list[dict], thoughts: str) -> str:
    """Full text whose THOUGHT span we score via echo+logprobs.

    Same canonical rendering as inject_prompt but WITHOUT the trailing
    separator, so the scored span is exactly the thought bytes. Used for the
    grounding leg of min(R, G): m = lpC(z_A|x) and t_i = lpC(z_C^i|x).
    """
    return (
        gen_prompt(repo, revision, prefix_messages)
        + THINK_CLOSE + "\nTHOUGHT: " + thoughts
    )


def split_rollout(text: str) -> tuple[str, str]:
    """Split a completion (which started inside <think>) into (z, y).

    Returns ("", "") when the rollout contains no closed bash block; callers
    filter on empty y.
    """
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


def extract_action(text: str) -> str:
    """Pull the action (last closed bash block) out of an injected rollout."""
    matches = list(BASH_RE.finditer(text))
    return matches[-1].group(0) if matches else ""
