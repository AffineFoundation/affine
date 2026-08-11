"""Canonical rollout data model.

The unit of storage is one **envelope** per rollout: a thin identity wrapper
around a verifiers trace-v1 dict (the trace is stored verbatim — never
mutated, so content hashes over it stay stable):

    {
      "schema": 1,                     # envelope schema version
      "rollout_id": "...",             # trace id (unique per rollout)
      "source": "swesmith",            # registry source name
      "env_id": "swesmith-v1",         # taskset / environment identity
      "task":   {uid, sid, repo, language, image, ...},   # catalog row
      "policy": {id, model, harness, endpoint},           # who acted
      "stored_at": "2026-08-11T12:00:00+00:00",
      "trace":  {... trace v1 ...},
    }

Two trace producers exist:
  - verifiers v1 evals emit trace v1 natively (nodes / calls / rewards /
    timing / errors) — adapters.verifiers wraps each one in an envelope.
  - mini-swe-agent writes *.traj.json — adapters.mini_swe converts it into
    the same trace shape (kind tag "mini_swe" in trace["run"]["type"]).

Everything downstream (store, index, views) reads envelopes only.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

ENVELOPE_SCHEMA = 1

# Keys every catalog row (envelope["task"]) must carry. `uid` is unique
# within its env; `sid` is the stratification id fed to the slicer
# (make_traj_id) so corpus repo-strata group correctly.
REQUIRED_TASK_KEYS = ("uid", "sid", "repo", "language")


@dataclass(frozen=True)
class Endpoint:
    """One serving route for a model. Ordered lists of these form a
    policy's fallback chain; the position actually used is stamped on the
    envelope (and, for verifiers runs, on every call in the trace)."""

    name: str                 # short route tag, e.g. "engy"
    model: str                # model name as the endpoint knows it
    base_url: str
    key_env: str              # env var holding the API key (never the key)
    litellm_model: str = ""   # mini_swe route override (e.g. openrouter/...)

    @property
    def label(self) -> str:
        """Turn-record model tag, e.g. engy/glm-5.2."""
        return f"{self.name}/{self.model}"

    @property
    def litellm(self) -> str:
        """Model string for litellm-based harnesses; defaults to the
        OpenAI-compatible route against base_url."""
        return self.litellm_model or f"openai/{self.model}"


@dataclass(frozen=True)
class Policy:
    """Who plays the agent seat: model identity + harness + sampling.

    The endpoint chain is operational fallback, not identity — two policies
    differing only in endpoints are still different rows here because retry
    behavior and quantization can differ per route."""

    id: str                                   # registry key
    harness: str                              # verifiers harness id
    endpoints: tuple[Endpoint, ...]           # ordered fallback chain
    sampling: dict = field(default_factory=dict)   # temperature etc.
    share: float = 1.0                        # target share within a source

    def available_endpoints(self, env: dict) -> list[Endpoint]:
        return [e for e in self.endpoints if env.get(e.key_env)]


@dataclass(frozen=True)
class PolicyStamp:
    """The (policy, endpoint) that actually produced a rollout."""

    policy_id: str
    model: str        # provider-qualified label, e.g. engy/glm-5.2
    harness: str
    endpoint: str     # endpoint name, e.g. "engy"

    def to_dict(self) -> dict:
        return {"id": self.policy_id, "model": self.model,
                "harness": self.harness, "endpoint": self.endpoint}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_envelope(*, source: str, env_id: str, task: dict,
                  policy: PolicyStamp, trace: dict) -> dict:
    """Wrap one trace dict in its identity envelope (see module docstring)."""
    for key in REQUIRED_TASK_KEYS:
        if key not in task:
            raise ValueError(f"task meta missing {key!r} (uid={task.get('uid')})")
    return {
        "schema": ENVELOPE_SCHEMA,
        "rollout_id": trace.get("id") or uuid.uuid4().hex,
        "source": source,
        "env_id": env_id,
        "task": task,
        "policy": policy.to_dict(),
        "stored_at": utc_now_iso(),
        "trace": trace,
    }


# -- trace accessors -----------------------------------------------------------
# Shared by the index and the views; tolerate both native verifiers traces
# and mini_swe-adapted ones (same shape by construction).

def trace_messages(trace: dict) -> list[dict]:
    """Linear conversation: drop the non-sampled assistant echo the graph
    records alongside every sampled assistant node. Tolerates nodes without
    a content key (tool-call-only assistant messages) and multimodal part
    lists; non-chat roles are dropped like the slicer does."""
    msgs = []
    for nd in trace["nodes"]:
        m = nd["message"]
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            continue
        if role == "assistant" and not nd.get("sampled"):
            continue
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text")
        msgs.append({"role": role, "content": content or ""})
    return msgs


def trace_stats(trace: dict) -> dict:
    """Token/cost/wall telemetry summed over the trace's model calls."""
    p = c = cache = calls = 0
    cost = 0.0
    for call in trace.get("calls", []):
        u = call.get("usage") or {}
        calls += 1
        p += int(u.get("prompt_tokens") or 0)
        c += int(u.get("completion_tokens") or 0)
        cache += int(u.get("cached_input_tokens") or 0)
        cost += float(u.get("cost") or 0.0)
    tm = trace.get("timing") or {}
    ag = tm.get("agent") or {}
    wall = None
    if ag.get("start") and ag.get("end"):
        wall = round(ag["end"] - ag["start"])
    return {"prompt_tokens": p, "completion_tokens": c,
            "cached_tokens": cache, "n_calls": calls,
            "cost_usd": round(cost, 5), "agent_wall_s": wall}


def trace_reward_score(trace: dict) -> float | None:
    """Primary scalar outcome as telemetry: `solved` if the env publishes
    it, else `passed_fraction`. None when the env scored nothing."""
    rewards = trace.get("rewards") or {}
    score = (rewards.get("solved") or {}).get("score")
    if score is None:
        score = (rewards.get("passed_fraction") or {}).get("score")
    return score


def trace_error_type(trace: dict) -> str | None:
    errors = trace.get("errors") or []
    if not errors:
        return None
    return errors[0].get("type") or errors[0].get("error") or "unknown"


def trace_task_name(trace: dict) -> str:
    return (trace.get("task") or {}).get("data", {}).get("name") or "unknown"
