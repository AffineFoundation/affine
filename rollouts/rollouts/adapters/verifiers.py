"""Wrap verifiers trace-v1 episodes in envelopes (near-passthrough).

A verifiers eval writes traces.jsonl: one episode per line, each carrying
`traces: [trace, ...]`. The trace dicts are stored verbatim inside the
envelope — content hashes (e.g. the view's run_tag over the sorted-keys
dump) stay identical to what the trace file contained.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rollouts.schema import PolicyStamp, make_envelope, trace_task_name

log = logging.getLogger("rollouts.adapters.verifiers")

# Reply contract (2026-09-11): a rollout that "finishes" with a final reply
# that has no visible text is a failed rollout, not a completed one. The pi
# harness ends the agent when its last tool completes even if the model then
# says nothing (`allow_empty_tool_reply=True` in verifiers' pi harness — the
# reasoning-only finish: 11 of 19 king pi completions had zero visible text,
# teacher 1 of 105); a first reply without visible text already raises
# HarnessError "no visible reply" there. This makes the two paths agree on
# the datagen side, before the trace is stored: stop_condition becomes
# `no_visible_reply`, which affine.corpus.view.rollout_outcome classifies as
# FAILED (the king-failure fold groups keep it), and the empty final reply
# can no longer fold as a `text` completion.
NO_VISIBLE_REPLY_STOP = "no_visible_reply"
THINK_CLOSE = "</think>"


def visible_text(content) -> str:
    """The reply's visible text: what follows the last `</think>` (or the
    whole content when reasoning is carried separately), stripped."""
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text")
    text = content or ""
    if THINK_CLOSE in text:
        text = text.rsplit(THINK_CLOSE, 1)[1]
    return text.strip()


def mark_no_visible_reply(trace: dict) -> bool:
    """Re-stamp an `agent_completed` trace whose final sampled reply carries
    neither a tool call nor visible text. Returns True when it did."""
    if trace.get("stop_condition") != "agent_completed":
        return False
    sampled = [n for n in trace.get("nodes") or []
               if n.get("sampled") and (n.get("message") or {}).get("role") == "assistant"]
    if not sampled:
        return False
    last = sampled[-1]["message"]
    if last.get("tool_calls") or visible_text(last.get("content")):
        return False
    trace["stop_condition"] = NO_VISIBLE_REPLY_STOP
    trace.setdefault("info", {})["reply_contract"] = {
        "was": "agent_completed",
        "reason": "final reply has no visible text",
        "reasoning_chars": len(last.get("reasoning_content") or ""),
    }
    return True


def envelopes_from_traces(traces_path: Path, *, source: str, env_id: str,
                          meta_by_uid: dict[str, dict],
                          policy: PolicyStamp) -> tuple[list[dict], list[str]]:
    """(envelopes for traces whose task is in the batch, unknown task uids).

    Tasks missing from meta_by_uid (a taskset emitting surprise rows) are
    skipped but reported — an envelope without catalog identity would be
    unusable for scheduling and views."""
    envelopes: list[dict] = []
    unknown: list[str] = []
    if not traces_path.exists():
        return envelopes, unknown
    for line in open(traces_path, encoding="utf-8"):
        try:
            episode = json.loads(line)
        except json.JSONDecodeError:
            continue
        for trace in episode.get("traces", []):
            uid = trace_task_name(trace)
            meta = meta_by_uid.get(uid)
            if meta is None:
                unknown.append(uid)
                continue
            envelopes.append(make_envelope(
                source=source, env_id=env_id, task=meta,
                policy=policy, trace=trace))
    if unknown:
        log.warning("%d trace(s) with unknown task uid (first: %s)",
                    len(unknown), unknown[0])
    return envelopes, unknown
