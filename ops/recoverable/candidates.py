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
--retry-errored). Pivots come first, then onsets by rank (earliest onset of a
rollout first) and depth; --max-states caps the batch.

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
from states import RESUMABLE, build_state  # noqa: E402

DIGEST_RE = re.compile(r"king-([0-9a-f]{12})")
STATE_KEY = ("rollout_id", "turn_idx", "state_kind")


def king_digest12(policy: dict) -> str:
    m = DIGEST_RE.search(str(policy.get("model") or ""))
    return m.group(1) if m else ""


def load_side_table(path: Path, retry_errored: bool) -> set[tuple]:
    done: set[tuple] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if retry_errored and row.get("teacher_status") == "errored":
            continue
        done.add((row["rollout_id"], int(row["turn_idx"]), row["state_kind"]))
    return done


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


def iter_king_envelopes(chunk_dir: Path, digest: str, kinds: set[str]):
    """Envelopes of the given king on a resumable harness of the wanted
    kinds, failed by their env. Chunk files are scanned newest-first-agnostic
    (sorted by name); a rollout appears in one chunk only."""
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
                kind = RESUMABLE.get(policy.get("harness", ""))
                if kind not in kinds:
                    continue
                if rollout_outcome(env["trace"]) != "failed":
                    continue
                yield env


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
                    help="resume kinds to build (null = wiki/math is excluded "
                         "from the king groups by the fold config)")
    ap.add_argument("--max-states", type=int, default=0)
    ap.add_argument("--min-confidence", type=float, default=0.7)
    ap.add_argument("--exclude-categories", default="")
    ap.add_argument("--retry-errored", action="store_true")
    ap.add_argument("--no-onsets", action="store_true")
    ap.add_argument("--no-pivots", action="store_true")
    args = ap.parse_args()

    kinds = set(args.kinds.split(","))
    done = load_side_table(args.side_table, args.retry_errored)
    pivots = {} if args.no_pivots else load_pivots(
        args.pivots, args.min_confidence,
        {c for c in args.exclude_categories.split(",") if c})
    baker = None if args.no_onsets else ToolBaker.from_pretrained()
    counts: collections.Counter = collections.Counter()
    cands: list[tuple[tuple, dict, int, str, dict]] = []   # (sort key, env, turn, kind, label)

    for env in iter_king_envelopes(args.chunks, args.king_digest, kinds):
        counts["failed_rollouts"] += 1
        rid = env["rollout_id"]
        onsets: dict[int, int] = {}
        if not args.no_onsets:
            try:
                onsets = onset_turns(env, baker)
            except (ToolParityError, TraceShapeError) as e:
                counts[f"label_error:{type(e).__name__}"] += 1
        for rank, (turn, repeats) in enumerate(sorted(onsets.items()), start=1):
            counts["onsets_seen"] += 1
            if (rid, turn, "loop_onset") in done:
                counts["onsets_in_table"] += 1
                continue
            cands.append(((1, rank, turn), env, turn, "loop_onset",
                          {"loop_onset_of": repeats, "onset_rank": rank,
                           "labeler": "affine.corpus.loops"}))
        for turn, row in pivots.get(rid, {}).items():
            counts["pivots_seen"] += 1
            if (rid, turn, "pivot") in done:
                counts["pivots_in_table"] += 1
                continue
            cands.append(((0, 0, turn), env, turn, "pivot", row))

    cands.sort(key=lambda c: c[0])
    if args.max_states and len(cands) > args.max_states:
        counts["deferred_by_cap"] = len(cands) - args.max_states
        cands = cands[: args.max_states]

    (args.out / "states").mkdir(parents=True, exist_ok=True)
    rows = []
    for _, env, turn, kind, label in cands:
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
        rows.append(meta)
        counts[f"state:{kind}:{st['harness']}"] += 1
    with open(args.out / "states.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in sorted(counts.items()):
        print(f"{v:6d}  {k}")
    print(f"{len(rows)} states -> {args.out}")


if __name__ == "__main__":
    main()
