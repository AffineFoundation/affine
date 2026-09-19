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

# [duel].thought_rendering (wvk 22, 2026-09-18, explicit operator directive
# 19:40 UTC "fold it in"): how a thought z is rendered into the teacher's
# assistant body for EVERY echo (G / typicality, R's injection, B, the A leg,
# the content-mask unconditioned echo) and how split_rollout composes z.
#   "canonical"    (wvk <= 21)  body = "</think>\nTHOUGHT: " + z [+ "\n\n" + y]
#                               z = latent + "\n" + visible (one prose string)
#   "as_generated" (wvk >= 22)  body = latent + "\n</think>" [+ "\n\n" + visible] [+ "\n\n" + y]
#                               z = latent + "\n</think>\n" + visible  (marker kept
#                               inside z so the split survives as a string;
#                               visible verbatim — no label stripped or added)
# Under "as_generated" the latent is scored inside <think>…</think>, where the
# model produced it, and the visible thought after the real </think> with NO
# added label (decision 2026-09-18: the model's own visible text, as generated;
# split_rollout still strips a literal "THOUGHT:" label the prompt asked for).
# The scored spans are the latent bytes and the visible bytes; the
# "\n</think>\n\n" separator between them is rendered but not scored (parity
# with the 2026-09-18 rendering study, HANDOVER.md §1). Token rule unchanged: a
# token is scored iff its start offset lies inside a span (so the first latent
# token — after the "<think>\n" newline token — IS scored; under canonical the
# first z token, merged with "THOUGHT: ", is not, exactly as before).
# Process-wide setting: run_duel sets it from the toml before any echo;
# research replays default to canonical.
THOUGHT_RENDERINGS = ("canonical", "as_generated")
_THOUGHT_RENDERING = "canonical"
Z_SPLIT = "\n" + THINK_CLOSE + "\n"


def set_thought_rendering(mode: str) -> None:
    global _THOUGHT_RENDERING
    if mode not in THOUGHT_RENDERINGS:
        raise ValueError(f"thought_rendering must be one of {THOUGHT_RENDERINGS}, got {mode!r}")
    _THOUGHT_RENDERING = mode


def thought_rendering() -> str:
    return _THOUGHT_RENDERING


def split_z(z: str) -> tuple[str, str]:
    """(latent, visible) of a z composed under as_generated; a z without the
    marker is latent-only (the kings' shape, and every canonical z)."""
    if Z_SPLIT in z:
        latent, _, visible = z.partition(Z_SPLIT)
        return latent, visible
    if z.startswith(THINK_CLOSE + "\n"):        # latent empty, visible only
        return "", z[len(THINK_CLOSE) + 1:]
    return z, ""


def thought_chars(z: str) -> int:
    """Characters of thought text in z (marker excluded) — the length floor."""
    latent, visible = split_z(z)
    return len(latent.strip()) + len(visible.strip())


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


def thought_body(thoughts: str) -> tuple[str, list[tuple[int, int]]]:
    """Assistant body for a thought under the active rendering, plus the
    scored spans as (start, end) character offsets INTO THE BODY.

    canonical:    "</think>\nTHOUGHT: " + z            span = z
    as_generated: latent + "\n</think>"                span = latent
                  [+ "\n" + visible]                   [+ span = visible]
    An empty thought renders as "</think>" (as_generated) so the body still
    closes the think block the template opened.
    """
    if _THOUGHT_RENDERING == "canonical":
        head = THINK_CLOSE + "\nTHOUGHT: "
        return head + thoughts, [(len(head), len(head) + len(thoughts))]
    latent, visible = split_z(thoughts)
    # The model always generates "…latent\n</think>\n\n…"; an empty latent
    # renders as the template's own thinking-off form "<think>\n\n</think>".
    body = latent + "\n" + THINK_CLOSE
    spans = [(0, len(latent))] if latent else []
    if visible:
        body += "\n\n"
        spans.append((len(body), len(body) + len(visible)))
        body += visible
    return body, spans


def inject_prompt(repo: str, revision: str | None,
                  prefix_messages: list[dict], thoughts: str) -> str:
    """Prompt where `thoughts` are planted as the full reasoning channel."""
    body, _ = thought_body(thoughts)
    return gen_prompt(repo, revision, prefix_messages) + body + "\n\n"


def force_text(repo: str, revision: str | None, prefix_messages: list[dict],
               thoughts: str, action: str) -> str:
    """Full text whose action span we score via echo+logprobs."""
    return inject_prompt(repo, revision, prefix_messages, thoughts) + action


def thought_text(repo: str, revision: str | None,
                 prefix_messages: list[dict], thoughts: str
                 ) -> tuple[str, list[tuple[int, int]]]:
    """Full text whose THOUGHT span(s) we score via echo+logprobs, and the
    absolute (start, end) offsets of those spans in it.

    Same rendering as inject_prompt but WITHOUT the trailing separator, so
    the scored bytes are exactly the thought bytes. Used for the grounding /
    typicality echoes: m = lpC(z_A|x), t_i = lpC(z_C^i|x), and lpC(z|∅).
    """
    gp = gen_prompt(repo, revision, prefix_messages)
    body, spans = thought_body(thoughts)
    return gp + body, [(len(gp) + a, len(gp) + b) for a, b in spans]


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


TEXT_FALLBACK_KINDS = ("tool_call",)


def split_rollout(text: str, action_kind: str | None = dialects.DEFAULT_KIND,
                  require_think_close: bool = False,
                  text_fallback_at_tool_turns: bool = False) -> tuple[str, str]:
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
    closed = THINK_CLOSE in text
    if closed:
        latent, _, rest = text.partition(THINK_CLOSE)
    elif require_think_close:
        return "", ""
    else:
        latent, rest = "", text
    before, y = dialects.split_action(rest, action_kind)
    if not y and text_fallback_at_tool_turns and closed \
            and (action_kind or dialects.DEFAULT_KIND) in TEXT_FALLBACK_KINDS:
        # wvk 18 (2026-09-15): at a tool-call turn a reply that closed its
        # reasoning and says something visible but calls no tool is a
        # prose (`text`) action — the whole visible reply — the same rule
        # the fold's teacher probe applies. Only with </think> closed: an
        # unclosed block has no visible reply and stays a forfeit / drop.
        before, y = dialects.split_action(rest, "text")
    if not y:
        return "", ""
    visible = THOUGHT_LABEL_RE.sub("", before.strip())
    if _THOUGHT_RENDERING == "as_generated":
        # Visible part VERBATIM (outer whitespace only): no label stripped,
        # none added — harness-neutral (mini-swe replies carry "THOUGHT:",
        # tool / pi / terminus / text replies never do). Keep the
        # latent/visible split inside z (see thought_body); a latent-only
        # reply is z = latent, exactly as before.
        lat, vis = latent.strip(), before.strip()
        if vis:
            return (lat + Z_SPLIT + vis) if lat else (THINK_CLOSE + "\n" + vis), y
        return lat, y
    z = "\n".join(s for s in (latent.strip(), visible.strip()) if s)
    return z, y


def extract_action(text: str, action_kind: str | None = dialects.DEFAULT_KIND
                   ) -> str:
    """Pull the action (last complete span in the dialect) out of an
    injected rollout."""
    return dialects.last_action(text, action_kind)
