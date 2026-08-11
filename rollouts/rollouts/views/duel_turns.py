"""duel_turns@v3 — the production duel-corpus turn view.

Derives corpus turn records from stored envelopes through
datagen.slicer.slice_messages (the production slice path: mswea fence
normalization, single-closed-bash-block rule, prefix-ends-on-user rule,
120k prefix cap, verbatim-leakage filter), then replays the
ops/datagen_refresh.py prefilter contract (validate_records) so only
records the fold would accept are queued.

Identity contract (reproduces both legacy pipelines exactly):

    slice instance_id   task["sid"]
    repo                task["repo"]
    model               policy["model"] (provider-qualified label)
    run_tag             trace.info.run_tag when the adapter stamped one
                        (mini_swe: sha256 of the raw traj file), else
                        sha256 of the sorted-keys trace dump (verifiers)
    generated_at        trace.info.generated_at when stamped (mini_swe:
                        file mtime), else the caller-supplied batch stamp

Resolved score is telemetry, not a keep-gate (Reason v3 policy).
"""

from __future__ import annotations

import hashlib
import json
import re

from datagen.slicer import MAX_PREFIX_CHARS, slice_messages

from rollouts.panel import PanelKeys, panel_drop, panel_keys
from rollouts.schema import trace_error_type, trace_messages, utc_now_iso

VIEW_SPEC = "duel_turns@v3"

BASH_BLOCK = re.compile(r"```bash\n.*?\n```", re.DOTALL)
WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return WS_RE.sub(" ", s).strip().lower()


def _run_tag(trace: dict) -> str:
    stamped = (trace.get("info") or {}).get("run_tag")
    if stamped:
        return stamped
    return hashlib.sha256(
        json.dumps(trace, sort_keys=True).encode()).hexdigest()[:8]


def derive_turns(envelope: dict, *, panel: PanelKeys | None = None,
                 generated_at: str | None = None) -> list[dict]:
    """Turn records from one envelope. Errored rollouts and panel-excluded
    tasks derive nothing; every record is tagged source + language."""
    panel = panel or panel_keys()
    trace = envelope["trace"]
    task = envelope["task"]
    if trace_error_type(trace) is not None:
        return []
    if panel_drop(task["uid"], task.get("repo") or "", panel):
        return []
    stamped_at = (trace.get("info") or {}).get("generated_at")
    records = slice_messages(
        trace_messages(trace),
        instance_id=task["sid"],
        repo=task.get("repo") or "",
        model=envelope["policy"]["model"],
        run_tag=_run_tag(trace),
        generated_at=stamped_at or generated_at or utc_now_iso())
    for rec in records:
        rec["source"] = envelope["source"]
        rec["language"] = task.get("language") or ""
    return records


def validate_records(records: list[dict], panel: PanelKeys | None = None,
                     ) -> tuple[list[dict], dict[str, int]]:
    """Replay of the ops/datagen_refresh.py prefilter contract; returns the
    surviving records (fail-closed: only validated records are queued)."""
    panel_ids, panel_repos, panel_bare = panel or panel_keys()
    drops: dict[str, int] = {}
    kept: list[dict] = []
    seen: set[str] = set()

    def drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    for rec in records:
        tid, tix = rec.get("traj_id"), rec.get("turn_idx")
        if not (isinstance(tid, str) and tid and isinstance(tix, int)):
            drop("bad_ids")
            continue
        turn_id = f"{tid}:{tix}"
        if turn_id in seen:
            drop("dup_turn_id")
            continue
        repo = str(rec.get("repo", "")).lower()
        if (repo in panel_repos or repo in panel_bare
                or str(rec.get("instance_id", "")) in panel_ids):
            drop("bench_panel_overlap")
            continue
        if rec.get("action_kind") != "bash":
            drop("action_kind_not_bash")
            continue
        prefix = rec.get("prefix")
        if (not isinstance(prefix, list) or not prefix
                or not all(isinstance(m, dict)
                           and isinstance(m.get("role"), str)
                           and isinstance(m.get("content"), str)
                           for m in prefix)
                or prefix[-1]["role"] != "user"):
            drop("prefix_shape")
            continue
        sys_msgs = [m for m in prefix if m["role"] == "system"]
        if not sys_msgs or "bash" not in sys_msgs[0]["content"].lower():
            drop("system_msg_no_bash_mandate")
            continue
        if sum(len(m["content"]) for m in prefix) > MAX_PREFIX_CHARS:
            drop("prefix_too_long")
            continue
        blocks = BASH_BLOCK.findall(rec.get("reference_turn") or "")
        if len(blocks) != 1:
            drop(f"ref_bash_blocks={len(blocks)}")
            continue
        body = _norm(blocks[0])
        if len(body) > 40 and any(body in _norm(m["content"]) for m in prefix):
            drop("reference_leaked_into_prefix")
            continue
        seen.add(turn_id)
        kept.append(rec)
    return kept, drops


def derive_and_validate(envelopes: list[dict],
                        panel: PanelKeys | None = None,
                        generated_at: str | None = None,
                        ) -> tuple[list[dict], dict[str, int]]:
    """Derive + validate across envelopes; the shape process_batch queues."""
    panel = panel or panel_keys()
    records: list[dict] = []
    for env in envelopes:
        records.extend(
            derive_turns(env, panel=panel, generated_at=generated_at))
    return validate_records(records, panel)
