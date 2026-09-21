"""Build the teacher-continuation states the side-table does not have yet.

Runs on the validator box (the trace chunk cache and the fold's own loop
labeler live there). For the current king it walks every failed king
rollout on a resumable harness and emits one state per

  * loop onset  -- `affine.corpus.loops.label_loops` on the main root, the
                   exact labeler the fold routes `king_loop_onset` with;
  * pivot       -- admitted rows of the king-review side-table
                   (affine/state/king_pivots/<digest12>.jsonl), filtered
                   like the fold's `[king_pivot]` (admit, confidence);

that is not already a row of the recoverable side-table
(`(rollout_id, turn_idx, state_kind)`; errored rows count as done unless
--retry-errored), plus RE-RUN candidates: table rows with `pending_reruns > 0`
(fewer OK continuations than the 3 the majority rule wants; aggregate.py)
get a state again with `continuations_needed` = the missing count, so
run_states.py only adds what is missing. Order: new pivots, new onsets by
rank (earliest onset of a rollout first) and depth, then re-runs — states
the teacher has NOT solved so far first (they can flip to admitted), then
solved-once states, each pivots / first onsets / shallow first.
--max-states caps the batch; --reruns-only / --no-reruns select one half.

  python candidates.py --chunks ops/corpus_build/cache/traces/chunks \
      --pivots affine/state/king_pivots/0ce59769300c.jsonl \
      --side-table affine/state/recoverable/0ce59769300c.jsonl \
      --king-digest 0ce59769300c --out /path/run/states

Output is states.py's format (states.jsonl + states/<id>.json), so
run_states.py / aggregate.py consume it unchanged.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))
sys.path.insert(0, str(HERE))

from affine.corpus.loops import ONSET, label_loops  # noqa: E402
from affine.corpus.trace import ToolParityError, TraceShapeError, trace_conversations  # noqa: E402
from affine.corpus.view import main_root_indices, rollout_outcome  # noqa: E402
from affine.toolbake import ToolBaker  # noqa: E402
from states import SAME_TASK, build_state, resume_kind_of  # noqa: E402

DIGEST_RE = re.compile(r"king-([0-9a-f]{12})")
UID_RE = re.compile(r'"uid":\s*"([^"]+)"')
STATE_KEY = ("rollout_id", "turn_idx", "state_kind")
# The king groups exclude these sources (sources.toml [king_recoverable]
# exclude_sources); no continuation is worth running on them.
DEFAULT_EXCLUDE_SOURCES = "affine_wiki,affine_agent,affine_math"


def king_digest12(policy: dict) -> str:
    m = DIGEST_RE.search(str(policy.get("model") or ""))
    return m.group(1) if m else ""


def load_side_table(path: Path, retry_errored: bool,
                    target: int) -> tuple[set[tuple], dict[tuple, dict]]:
    """(done, reruns): `done` = states the table settles (no candidate);
    `reruns` = key -> row for states with OK continuations but fewer than
    `target` (pending_reruns > 0). Errored rows are done unless
    --retry-errored; not_run rows are never done (they never ran)."""
    done: set[tuple] = set()
    reruns: dict[tuple, dict] = {}
    if not path.is_file():
        return done, reruns
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["rollout_id"], int(row["turn_idx"]), row["state_kind"])
        status = row.get("teacher_status")
        if status == "not_run":
            continue
        if status == "errored" and retry_errored:
            continue
        pending = row.get("pending_reruns")
        if pending is None:   # pre-rule row: one continuation stored
            pending = target - 1 if status == "ok" else target
        if status == "ok" and pending > 0:
            reruns[key] = row
        else:
            done.add(key)
    return done, reruns


def load_pivots(path: Path, min_confidence: float,
                exclude_categories: set[str]) -> dict[str, dict[int, dict]]:
    """rollout_id -> turn_idx -> pivot row, admitted rows only (the fold's
    [king_pivot] filter)."""
    table: dict[str, dict[int, dict]] = {}
    if not path.is_file():
        return table
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("admit") or str(row.get("failure_category")) in exclude_categories:
            continue
        if float(row.get("confidence") or 0) < min_confidence:
            continue
        table.setdefault(str(row["rollout_id"]), {})[int(row["turn_idx"])] = row
    return table


def iter_king_envelopes(chunk_dir: Path, digest: str, kinds: set[str],
                        exclude_sources: set[str] = frozenset()):
    """Envelopes of the given king on a harness of the wanted resume kinds
    (plugin kinds or `same_task`), failed by their env. Chunk files are
    scanned newest-first-agnostic (sorted by name); a rollout appears in one
    chunk only."""
    for name in sorted(os.listdir(chunk_dir)):
        if not name.endswith(".jsonl.gz"):
            continue
        with gzip.open(chunk_dir / name, "rt", encoding="utf-8") as f:
            for line in f:
                if '"king_' not in line:
                    continue
                env = json.loads(line)
                policy = env.get("policy") or {}
                if not str(policy.get("id", "")).startswith("king_"):
                    continue
                if king_digest12(policy) != digest:
                    continue
                kind = resume_kind_of(policy.get("harness", ""))
                if kind not in kinds or env.get("source") in exclude_sources:
                    continue
                if rollout_outcome(env["trace"]) != "failed":
                    continue
                yield env


def teacher_prior(chunk_dir: Path, uids: set[str]) -> dict[str, dict]:
    """Second pass over the chunks: the teacher's OWN datagen rollouts on
    the given task uids (any harness), uid -> {n, solved, harnesses}. Used
    only to ORDER the same-task proxy work: a task the teacher already
    solved under another harness is the most likely admit, so it runs
    first. (Same-harness teacher rollouts are rare — the teacher seat plays
    each task once under one harness — so they do not count as
    continuations here.)"""
    out: dict[str, dict] = {}
    for name in sorted(os.listdir(chunk_dir)):
        if not name.endswith(".jsonl.gz"):
            continue
        with gzip.open(chunk_dir / name, "rt", encoding="utf-8") as f:
            for line in f:
                if '"king_' in line:
                    continue
                m = UID_RE.search(line)
                if not m or m.group(1) not in uids:
                    continue
                env = json.loads(line)
                uid = str(env.get("task", {}).get("uid"))
                if uid not in uids:
                    continue
                policy = env.get("policy") or {}
                rec = out.setdefault(uid, {"n": 0, "solved": 0, "harnesses": []})
                rec["n"] += 1
                rec["solved"] += rollout_outcome(env["trace"]) == "solved"
                rec["harnesses"].append(str(policy.get("harness") or ""))
    return out


def onset_turns(env: dict, baker: ToolBaker) -> dict[int, int]:
    """turn_idx -> the earlier turn it repeats, for every loop onset on the
    main root (the fold's rule)."""
    convs = trace_conversations(env["trace"], baker)
    main = main_root_indices(env["trace"])
    kind = (env.get("policy") or {}).get("action_kind") or "bash"
    out: dict[int, int] = {}
    for j, lab in enumerate(label_loops([convs[i] for i in main], kind)):
        if lab.label == ONSET:
            out[main[j]] = int(main[int(lab.repeats)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chunks", required=True, type=Path)
    ap.add_argument("--pivots", required=True, type=Path)
    ap.add_argument("--side-table", required=True, type=Path)
    ap.add_argument("--king-digest", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--kinds", default="textbased,bash,terminus",
                    help="resume kinds to build: plugin kinds textbased / bash / "
                         "terminus / null (null = wiki/math, excluded from the king "
                         "groups by the fold config) and same_task (the ACP "
                         "same-task proxy: claude_code / pi / kimi_code / hermes_agent)")
    ap.add_argument("--exclude-sources", default=DEFAULT_EXCLUDE_SOURCES,
                    help="comma-separated sources never built (default: the king "
                         "groups' exclude_sources)")
    ap.add_argument("--no-proxy-prior", action="store_true",
                    help="skip the second chunk pass that orders same_task work by "
                         "the teacher's own outcome on the task")
    ap.add_argument("--max-states", type=int, default=0)
    ap.add_argument("--min-confidence", type=float, default=0.7)
    ap.add_argument("--exclude-categories", default="")
    ap.add_argument("--retry-errored", action="store_true")
    ap.add_argument("--no-onsets", action="store_true")
    ap.add_argument("--no-pivots", action="store_true")
    ap.add_argument("--continuations", type=int, default=3,
                    help="OK continuations a state needs (aggregate.py majority rule)")
    ap.add_argument("--reruns-only", action="store_true",
                    help="only states already in the table with pending_reruns > 0")
    ap.add_argument("--no-reruns", action="store_true", help="new states only")
    args = ap.parse_args()

    kinds = set(args.kinds.split(","))
    done, reruns = load_side_table(args.side_table, args.retry_errored, args.continuations)
    if args.no_reruns:
        done |= set(reruns)
        reruns = {}
    pivots = {} if args.no_pivots else load_pivots(
        args.pivots, args.min_confidence,
        {c for c in args.exclude_categories.split(",") if c})
    baker = None if args.no_onsets else ToolBaker.from_pretrained()
    counts: collections.Counter = collections.Counter()
    exclude_sources = {s for s in args.exclude_sources.split(",") if s}
    # (sort key, env, turn, kind, label, continuations_needed). Sort key
    # tiers: 0 new pivots, 1 new onsets (by rank), 2 new same-task proxy
    # tasks (teacher-solved-elsewhere first; whole tasks, so every state of a
    # rollout stays adjacent), 3 re-runs of unsolved states, 4 re-runs of
    # solved-once states.
    cands: list[tuple[tuple, dict, int, str, dict, int]] = []
    proxy_envs: dict[str, dict] = {}     # rollout_id -> env (same_task tasks)

    for env in iter_king_envelopes(args.chunks, args.king_digest, kinds, exclude_sources):
        counts["failed_rollouts"] += 1
        rid = env["rollout_id"]
        is_proxy = resume_kind_of((env.get("policy") or {}).get("harness", "")) == SAME_TASK
        onsets: dict[int, int] = {}
        if not args.no_onsets:
            try:
                onsets = onset_turns(env, baker)
            except (ToolParityError, TraceShapeError) as e:
                counts[f"label_error:{type(e).__name__}"] += 1
        ranks = {turn: rank for rank, turn in enumerate(sorted(onsets), start=1)}
        tier_new = 2 if is_proxy else None
        if not args.reruns_only:
            for rank, (turn, repeats) in enumerate(sorted(onsets.items()), start=1):
                counts["onsets_seen"] += 1
                if (rid, turn, "loop_onset") in done or (rid, turn, "loop_onset") in reruns:
                    counts["onsets_in_table"] += 1
                    continue
                cands.append(((tier_new or 1, rank, turn), env, turn, "loop_onset",
                              {"loop_onset_of": repeats, "onset_rank": rank,
                               "labeler": "affine.corpus.loops"}, args.continuations))
                if is_proxy:
                    proxy_envs[rid] = env
            for turn, row in pivots.get(rid, {}).items():
                counts["pivots_seen"] += 1
                if (rid, turn, "pivot") in done or (rid, turn, "pivot") in reruns:
                    counts["pivots_in_table"] += 1
                    continue
                cands.append(((tier_new or 0, 0, turn), env, turn, "pivot", row,
                              args.continuations))
                if is_proxy:
                    proxy_envs[rid] = env
        for (r_rid, turn, kind), row in reruns.items():
            if r_rid != rid:
                continue
            counts["reruns_seen"] += 1
            pending = int(row.get("pending_reruns") or (args.continuations - 1))
            if kind == "pivot":
                label = pivots.get(rid, {}).get(turn) or {"node_id": row.get("node_id")}
                prio = 0
            else:
                rank = ranks.get(turn) or int(row.get("onset_rank") or 0)
                label = {"loop_onset_of": onsets.get(turn), "onset_rank": rank,
                         "labeler": "affine.corpus.loops"}
                prio = rank
            unsolved_first = 0 if not row.get("teacher_solved") else 1
            cands.append(((3 + unsolved_first, prio, turn), env, turn, kind, label, pending))

    # Same-task proxy: order the tasks by the teacher's own record on them
    # (solved under any harness first), then keep each task's states together.
    prior: dict[str, dict] = {}
    if proxy_envs and not args.no_proxy_prior:
        prior = teacher_prior(args.chunks, {e["task"]["uid"] for e in proxy_envs.values()})
        counts["proxy_tasks"] = len(proxy_envs)
        counts["proxy_tasks_teacher_solved_elsewhere"] = sum(
            1 for e in proxy_envs.values() if (prior.get(e["task"]["uid"]) or {}).get("solved"))

    def task_rank(env: dict) -> tuple:
        p = prior.get(env["task"]["uid"]) or {}
        # 0 = solved elsewhere, 1 = no teacher record, 2 = teacher failed elsewhere
        bucket = 0 if p.get("solved") else (1 if not p else 2)
        return (bucket, env["source"], env["rollout_id"])

    def sort_key(c: tuple) -> tuple:
        (tier, prio, turn), env = c[0], c[1]
        if tier == 2:
            return (2, task_rank(env), prio, turn)
        return (tier, (), prio, turn)

    cands.sort(key=sort_key)
    if args.max_states and len(cands) > args.max_states:
        counts["deferred_by_cap"] = len(cands) - args.max_states
        cands = cands[: args.max_states]

    (args.out / "states").mkdir(parents=True, exist_ok=True)
    rows = []
    for key, env, turn, kind, label, needed in cands:
        st = build_state(env, turn, kind, label)
        if st is None:
            counts["bad_turn_idx"] += 1
            continue
        path = args.out / "states" / (st["state_id"].replace(":", "_") + ".json")
        path.write_text(json.dumps(st, ensure_ascii=False))
        meta = {k: v for k, v in st.items()
                if k not in ("messages", "king_reply", "task_system_prompt",
                             "task_prompt", "label")}
        meta["path"] = str(path)
        meta["continuations_needed"] = needed
        meta["rerun"] = key[0] >= 3
        if st.get("proxy_key"):
            meta["proxy_prior"] = prior.get(env["task"]["uid"])
        # Continuations the table already counts: run_states.py does not
        # let a stored result with one of these trace ids fill the need.
        row = reruns.get((env["rollout_id"], turn, kind)) or {}
        meta["table_trace_ids"] = [c["trace_id"] for c in row.get("continuations") or []
                                   if c.get("trace_id")]
        rows.append(meta)
        tag = "rerun:" if key[0] >= 3 else ("proxy:" if key[0] == 2 else "")
        counts[f"state:{tag}{kind}:{st['harness']}"] += 1
    with open(args.out / "states.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in sorted(counts.items()):
        print(f"{v:6d}  {k}")
    print(f"{len(rows)} states -> {args.out}")


if __name__ == "__main__":
    main()
