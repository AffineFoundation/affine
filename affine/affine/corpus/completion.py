"""Completion rule: is the final reply of a rollout the reply that ended it
on purpose? (fold group `completion`, data event 2026-09-11)

Vendored from the per-harness rule of the deterministic failure labeler
(`affine/corpus/labels.py`, PR #5, not merged when this shipped), applied
to the BAKED final conversation the slicer already derives, so the fold
needs no second reading of the trace:

  bash (mini-swe)        the single command is `submit` or carries
                         COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT   -> submit
                         else a prose reply with no action        -> text
  tool_call loops        one tool call named like finish / submit /
                         done / attempt_completion / final_answer -> finish_tool
                         else a reply with no tool call and visible
                         text                                     -> text
  terminus_json          the JSON batch carries task_complete: true
                                                                  -> task_complete
  boxed (math)           the reply holds a \\boxed{...}          -> boxed
  text                   the visible reply                        -> text

A reply with no visible text (reasoning-only finish) matches nothing: the
baked assistant content is the visible part only, so an empty reply is
exactly "said nothing". A prose reply right after a format nudge is the
harness's third strike (mini-swe RepeatedFormatError), not a report.
"""

from __future__ import annotations

import json
import re

from affine import dialects
from datagen.slicer import _normalize as normalize_fence

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
FINISH_TOOL_RE = re.compile(
    r"^(?:submit|finish|done|complete|task_complete|complete_task|"
    r"attempt_completion|final_answer|submit_answer|finish_task)$", re.IGNORECASE)
TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
NUDGE_RE = re.compile(
    r"exactly one|format error|please always provide|did not include|"
    r"no (?:bash )?code block|your response must|must contain|"
    r"could not parse|invalid json|malformed|wasted call|"
    r"refer to that earlier|unknown tool|tool .* not found", re.I)


def bash_body(block: str) -> str:
    inner = block.split("\n", 1)[1] if "\n" in block else ""
    return inner.rsplit("```", 1)[0].strip()


def is_submit_command(body: str) -> bool:
    return body.strip() == "submit" or SUBMIT_MARKER in body


def completion_kind(reply: str, action_kind: str) -> str | None:
    """Which rule the final reply satisfies, or None."""
    content = normalize_fence(reply)
    visible = content.strip()
    if not visible:
        return None
    if action_kind == dialects.TEXT_KIND:
        return "text"
    d = dialects.get(action_kind)
    acts = d.actions(content)
    if action_kind == dialects.DEFAULT_KIND:
        if len(acts) == 1 and is_submit_command(bash_body(acts[0])):
            return "submit"
        return "text" if not acts else None
    if action_kind == "tool_call":
        if len(acts) == 1:
            m = TOOL_NAME_RE.search(acts[0])
            return "finish_tool" if m and FINISH_TOOL_RE.match(m.group(1)) else None
        return "text" if not acts else None
    if action_kind == "terminus_json":
        if len(acts) != 1:
            return None
        try:
            return "task_complete" if json.loads(acts[0]).get("task_complete") else None
        except (ValueError, AttributeError):
            return None
    if action_kind == "boxed":
        return "boxed" if acts else None
    return None


def final_completion(convs: list[list[dict]], action_kind: str) -> str | None:
    """Completion kind of the rollout's FINAL reply (`convs[-1]`), or None.
    The caller vouches the rollout stopped as `agent_completed` and was
    graded solved."""
    if not convs or not convs[-1] or convs[-1][-1]["role"] != "assistant":
        return None
    conv = convs[-1]
    kind = completion_kind(conv[-1]["content"], action_kind)
    if kind == "text" and len(conv) >= 2 and conv[-2]["role"] == "user" \
            and NUDGE_RE.search(conv[-2]["content"][:1500]):
        return None
    return kind
