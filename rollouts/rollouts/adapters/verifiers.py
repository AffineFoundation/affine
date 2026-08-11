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
