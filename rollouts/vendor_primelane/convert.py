"""Convert verifiers traces into affine corpus turn records (any lane taskset).

Keep turns from any finished rollout (resolved score is telemetry);
sliced through datagen.slicer.slice_messages — the production slice path,
including mswea fence normalization, single-closed-bash-block rule,
prefix-ends-on-user rule, 120k prefix cap and verbatim-leakage filter.
Every record gets `source: <taskset>`; a second independent pass replays the
ops/datagen_refresh.py prefilter contract before anything is queued.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from datagen.slicer import MAX_PREFIX_CHARS, slice_messages

BASH_BLOCK = re.compile(r"```bash\n.*?\n```", re.DOTALL)
WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return WS_RE.sub(" ", s).strip().lower()


def trace_messages(trace: dict) -> list[dict]:
    """Linear conversation: drop the non-sampled assistant echo the graph
    records alongside every sampled assistant node. Tolerates nodes without
    a content key (e.g. tool-call-only assistant messages) and multimodal
    part lists; non-chat roles are dropped like the slicer does."""
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


def convert_run(run_dir: Path, source: str, model_label: str,
                meta_by_uid: dict[str, dict], panel_ids: set[str],
                panel_repos: set[str], panel_bare: set[str],
                ) -> tuple[list[dict], list[dict]]:
    """(corpus records from finished rollouts, per-task outcome rows)."""
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records: list[dict] = []
    per_task: list[dict] = []
    traces_path = run_dir / "traces.jsonl"
    if not traces_path.exists():
        return records, per_task
    for line in open(traces_path, encoding="utf-8"):
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            continue
        for trace in ep.get("traces", []):
            uid = trace["task"]["data"].get("name") or "unknown"
            meta = meta_by_uid.get(uid)
            rewards = trace.get("rewards") or {}
            score = (rewards.get("solved") or {}).get("score")
            if score is None:
                score = (rewards.get("passed_fraction") or {}).get("score")
            row = {"uid": uid, "resolved": score,
                   "stop": trace.get("stop_condition"),
                   "error": (trace.get("errors") or [{}])[0].get("type")
                   if trace.get("errors") else None,
                   **trace_stats(trace), "n_records": 0}
            # Reason v3: keep prefixes from any finished rollout; resolved
            # score is telemetry only. Panel exclusion and slicer quality
            # filters (bash shape / leakage) still apply.
            if meta is not None and row["error"] is None:
                repo = meta["repo"]
                if (repo in panel_repos or repo in panel_bare
                        or uid in panel_ids):
                    row["error"] = "panel_excluded"
                else:
                    run_tag = hashlib.sha256(
                        json.dumps(trace, sort_keys=True).encode()
                    ).hexdigest()[:8]
                    recs = slice_messages(
                        trace_messages(trace), instance_id=meta["sid"],
                        repo=repo, model=model_label, run_tag=run_tag,
                        generated_at=generated_at)
                    for r in recs:
                        r["source"] = source
                        r["language"] = meta.get("language") or ""
                    row["n_records"] = len(recs)
                    records.extend(recs)
            per_task.append(row)
    return records, per_task


def validate_records(records: list[dict], panel_ids: set[str],
                     panel_repos: set[str], panel_bare: set[str],
                     ) -> tuple[list[dict], dict[str, int]]:
    """Replay of the ops/datagen_refresh.py prefilter contract; returns the
    surviving records (fail-closed: only validated records are queued)."""
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


def main() -> None:  # ad-hoc reconversion of a run dir
    from primelane.lanespec import panel_keys
    run_dir, source, label = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    ids, repos, bare = panel_keys()
    recs, tasks = convert_run(run_dir, source, label, {}, ids, repos, bare)
    print(json.dumps({"records": len(recs), "tasks": tasks}, indent=2))


if __name__ == "__main__":
    main()
