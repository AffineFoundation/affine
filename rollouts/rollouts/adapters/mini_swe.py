"""Convert mini-swe-agent *.traj.json files into trace-v1-shaped dicts.

The mapping is lossless where the trajectory has data and honest where it
does not:

    messages[]              -> nodes[] (assistant nodes sampled=True)
    message.extra.response  -> calls[] (usage per assistant message)
    swebench verdict        -> rewards.solved (telemetry; absent = unscored)
    info.exit_status        -> stop_condition
    file mtime / raw sha    -> info.generated_at / info.run_tag (the exact
                               values the legacy slicer stamped, so the
                               duel_turns view reproduces its output)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rollouts.schema import PolicyStamp, make_envelope

log = logging.getLogger("rollouts.adapters.mini_swe")

# Agent exit statuses that mean the rollout ran to a real terminal state
# (same set the legacy loop used for eval eligibility).
ATTEMPTED_EXITS = {"Submitted", "LimitsExceeded", "TimeExceeded",
                   "RepeatedFormatError"}


def _calls_from_messages(messages: list[dict], model: str,
                         endpoint: str) -> list[dict]:
    calls = []
    for i, m in enumerate(messages):
        usage = (m.get("extra", {}).get("response", {}) or {}).get("usage")
        if not usage:
            continue
        calls.append({
            "node": i,
            "model": model,
            "endpoint": endpoint,
            "finish_reason": None,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "cached_input_tokens": int(
                    (usage.get("prompt_tokens_details") or {})
                    .get("cached_tokens") or 0),
                "cost": 0.0,
            },
        })
    return calls


def traj_to_trace(raw: bytes, *, instance_id: str, model_label: str,
                  endpoint: str, resolved: float | None,
                  generated_at: str | None = None) -> dict:
    """One trajectory file's bytes -> a trace-v1-shaped dict.

    resolved: swebench verdict when the eval ran (1.0/0.0), None when the
    rollout was never scored (no patch, eval skipped). generated_at
    defaults to now; pass the file mtime to reproduce legacy records."""
    data = json.loads(raw)
    info = data.get("info", {})
    exit_status = str(info.get("exit_status") or "")
    messages = data.get("messages", [])
    attempted = exit_status in ATTEMPTED_EXITS

    nodes = [{
        "message": {"role": m.get("role"), "content": m.get("content")},
        "sampled": m.get("role") == "assistant",
        "timestamp": None,
    } for m in messages]

    rewards = {}
    if resolved is not None:
        rewards["solved"] = {"score": float(resolved), "weight": 1.0}

    return {
        "version": 1,
        "id": hashlib.sha256(raw).hexdigest()[:32],
        "run": {"type": "mini_swe", "id": ""},
        "task": {"type": "mini_swe", "data": {"name": instance_id}},
        "agent": {
            "config": {
                "model": model_label,
                "harness": {"id": "mini_swe_textbased"},
                "sampling": {},
            },
            "name": "mini-swe-agent",
        },
        "nodes": nodes,
        "calls": _calls_from_messages(messages, model_label, endpoint),
        "rewards": rewards,
        "metrics": {},
        "info": {
            "exit_status": exit_status,
            "mini_version": info.get("mini_version"),
            # The legacy slicer's identity inputs, preserved exactly:
            # run_tag = sha256(raw file bytes)[:8], generated_at = mtime.
            "run_tag": hashlib.sha256(raw).hexdigest()[:8],
            "generated_at": generated_at,
        },
        "is_completed": attempted,
        "ok": attempted,
        "stop_condition": exit_status or None,
        "errors": [] if attempted else [{"type": exit_status or "unknown"}],
        "timing": {},
    }


def envelope_from_traj(path: Path, *, source: str, env_id: str, task: dict,
                       policy: PolicyStamp,
                       resolved: float | None) -> dict | None:
    """Read one trajectory file into an envelope; None if unreadable."""
    try:
        raw = path.read_bytes()
        generated_at = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc,
        ).isoformat(timespec="seconds")
        trace = traj_to_trace(
            raw, instance_id=task["uid"], model_label=policy.model,
            endpoint=policy.endpoint, resolved=resolved,
            generated_at=generated_at)
    except Exception:
        log.warning("unreadable trajectory %s", path, exc_info=True)
        return None
    return make_envelope(source=source, env_id=env_id, task=task,
                         policy=policy, trace=trace)
