"""Prompt construction: natural generation, thought injection, teacher forcing.

Turn contract:
  z (thoughts) = all reasoning text: latent <think> content plus the visible
                 THOUGHT section, normalized to plain text.
  y (action)   = the final complete action span, located by the turn's
                 action dialect (affine.dialects) — the discrete action that
                 drives the environment. `bash` (one closed ```bash block)
                 is the default and the only kind live D admits today.

We always render through the model's own chat template to a string and drive
/v1/completions directly, so injection and forcing are byte-exact and cannot
be mangled by server-side chat templating. Injection uses a canonical
assistant body (`</think>\nTHOUGHT: {z}\n\n{y}`) identical across model
families AND across dialects, so forced logprob differences reflect the
models, not the rendering — and so the grounding band (m vs t_i, both scored
through thought_text) stays comparable whatever the action format is. Only
the action *parsing* varies per dialect; the thought channel never does.
"""

from __future__ import annotations

import re
from functools import lru_cache

from transformers import AutoTokenizer

from affine import dialects

from . import r2store

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THOUGHT_LABEL_RE = re.compile(r"^\s*THOUGHT:\s*")


@lru_cache(maxsize=8)
def get_tokenizer(repo: str, revision: str | None = None):
    # r2 refs: the verified local snapshot (no hub revision to pin).
    if r2store.is_r2(repo):
        return AutoTokenizer.from_pretrained(r2store.model_path(repo, revision))
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


def chat_prompt(repo: str, revision: str | None, messages: list[dict],
                tools: list[dict] | None = None) -> str:
    """Prompt exactly as an OpenAI-compatible client would render it: the
    model's own chat template with add_generation_prompt and, when given,
    the tool schemas the template folds into its system block. Ends inside
    the open <think> the template emits (thinking on). Used by the protocol
    probe; the duel path keeps gen_prompt (no tools — D carries them baked
    into the prefix text)."""
    tok = get_tokenizer(repo, revision)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    p = tok.apply_chat_template(messages, **kwargs)
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


def think_closed(text: str) -> bool:
    """Did the completion close its reasoning block?

    The prompt always ends inside an open <think>, so a well-formed reply
    emits </think> before its visible answer. Every OpenAI-compatible client
    that separates reasoning from content (vLLM's reasoning parsers, Cursor,
    the chat pod) depends on that tag; a reply without it is delivered as
    100% reasoning and 0% answer. Measured on every sample (telemetry), and
    required when [duel].require_think_close is on.
    """
    return THINK_CLOSE in text


def split_rollout(text: str, action_kind: str | None = dialects.DEFAULT_KIND,
                  require_think_close: bool = False) -> tuple[str, str]:
    """Split a completion (which started inside <think>) into (z, y).

    Returns ("", "") when the rollout contains no complete action in the
    turn's dialect; callers filter on empty y. An unparsable action is a
    forfeited turn, not an error — that is the incentive for a miner to
    honor the contract the prefix states.

    require_think_close (staged 2026-09-07, off by default): a rollout that
    never emits </think> is treated exactly like one with no parseable
    action — ("", ""), i.e. a forfeit. Without it the tag is optional here,
    which is why kings trained against this score dropped it (bench
    transcripts: genesis closes </think> on 95–98% of replies, kings of
    reigns 1–5 on 0–4%) and then render as empty replies in Cursor.
    Flipping the knob changes which turns score, so it is a
    weight_version_key event.
    """
    if THINK_CLOSE in text:
        latent, _, rest = text.partition(THINK_CLOSE)
    elif require_think_close:
        return "", ""
    else:
        latent, rest = "", text
    before, y = dialects.split_action(rest, action_kind)
    if not y:
        return "", ""
    visible = THOUGHT_LABEL_RE.sub("", before.strip())
    z = "\n".join(s for s in (latent.strip(), visible.strip()) if s)
    return z, y


def extract_action(text: str, action_kind: str | None = dialects.DEFAULT_KIND
                   ) -> str:
    """Pull the action (last complete span in the dialect) out of an
    injected rollout."""
    return dialects.last_action(text, action_kind)
