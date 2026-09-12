"""Step 1 -- pick the king rollouts the judge will read.

For one king digest: a stratified sample of its FAILED rollouts, up to
`--per-cell` per (env group, harness) cell and `--max-total` overall, with
a deterministic seed, plus the teacher's rollout on the same task when one
exists (same harness preferred, solved preferred). Errored rollouts
(harness / API failures) are not the king's doing and are skipped, as the
fold skips them.

  python select.py --king king-0ce59769300c --out /tmp/king-review/reign/sample.jsonl
  python select.py --king current ...          # king from affine.io/api/v1/snapshot

Output: one JSON line per sampled rollout with pointers into the chunk
cache (`chunk`, `line`) for the king rollout and, when found, `teacher`
(same pointers + outcome + harness match flag).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from krlib import (KING_PREFIX, TEACHER_PREFIX, TraceStore, digest12,
                   load_env_groups, load_king_pivot_config, resolve_current_king,
                   write_jsonl)


def pick_teacher(candidates: list[dict], harness: str) -> dict | None:
    """Best teacher rollout for one task: same harness + solved, then any
    harness + solved, then same harness any clean outcome, then anything
    that is not errored."""
    clean = [c for c in candidates if c["outcome"] != "errored" and c["n_replies"] > 0]
    if not clean:
        return None
    ranked = sorted(clean, key=lambda c: (
        c["harness"] != harness, c["outcome"] != "solved", c["stored_at"]))
    best = ranked[0]
    return {**best, "same_harness": best["harness"] == harness}


def interleave(cells: dict[tuple[str, str], list[dict]]) -> list[dict]:
    """Round-robin over the cells so a budget stop leaves every cell
    represented instead of exhausting the first one."""
    out: list[dict] = []
    queues = {k: list(v) for k, v in sorted(cells.items())}
    while queues:
        for k in list(queues):
            if queues[k]:
                out.append(queues[k].pop(0))
            if not queues[k]:
                del queues[k]
    return out


def select_sample(rows: list[dict], *, king: str, per_cell: int, max_total: int,
                  seed: int, env_groups: dict[str, str],
                  exclude_sources: frozenset[str] = frozenset(),
                  all_failed: bool = False,
                  include_single_reply: bool = False) -> list[dict]:
    """`all_failed`: every failed rollout of the king (the daily / bulk
    mode), single-reply harnesses last, cells interleaved; `per_cell` /
    `max_total` are ignored. Sources in `exclude_sources` (the fold's
    `[king_pivot].exclude_sources`) are skipped: the fold would not route
    their pivots, so judging them buys nothing for D."""
    king_id = f"king-{digest12(king)}"
    failed = [r for r in rows if r["king"] == king_id and r["outcome"] == "failed"
              and r["n_replies"] > 0 and r["policy_id"].startswith(KING_PREFIX)
              and r["source"] not in exclude_sources]
    teacher_by_sid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["policy_id"].startswith(TEACHER_PREFIX):
            teacher_by_sid[r["sid"]].append(r)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in failed:
        cells[(env_groups.get(r["source"], "?"), r["harness"])].append(r)
    rng = random.Random(f"{seed}:{king_id}")
    picked: list[dict] = []
    if all_failed:
        # Single-reply rollouts (math, wiki, the `general` one-shot envs)
        # have no decision point to find: their only turn IS the king_fail
        # turn already, so a pivot there would move a turn, not add one.
        # The bulk / daily run skips them; the stratified review keeps a
        # cell of them for the report.
        multi: dict[tuple[str, str], list[dict]] = {}
        single: dict[tuple[str, str], list[dict]] = {}
        for key, pool in cells.items():
            pool = sorted((r for r in pool if include_single_reply or r["n_replies"] > 1),
                          key=lambda r: r["rollout_id"])
            if not pool:
                continue
            rng.shuffle(pool)
            pool.sort(key=lambda r: r["sid"] not in teacher_by_sid)
            target = multi if key[1] != "null" else single
            target[key] = [{**r, "env_group": key[0]} for r in pool]
        picked = interleave(multi) + interleave(single)
        return finish_sample(picked, king_id, teacher_by_sid)
    for key in sorted(cells):
        pool = sorted(cells[key], key=lambda r: r["rollout_id"])
        rng.shuffle(pool)
        # tasks the teacher also played are worth more to the review
        pool.sort(key=lambda r: r["sid"] not in teacher_by_sid)
        for r in pool[:per_cell]:
            picked.append({**r, "env_group": key[0]})
    if len(picked) > max_total:
        # trim the biggest cells first, one rollout at a time, so every cell
        # keeps representation
        by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in picked:
            by_cell[(r["env_group"], r["harness"])].append(r)
        while sum(len(v) for v in by_cell.values()) > max_total:
            biggest = max(by_cell, key=lambda k: (len(by_cell[k]), k))
            by_cell[biggest].pop()
        picked = [r for k in sorted(by_cell) for r in by_cell[k]]
    return finish_sample(picked, king_id, teacher_by_sid)


def finish_sample(picked: list[dict], king_id: str,
                  teacher_by_sid: dict[str, list[dict]]) -> list[dict]:
    out = []
    for r in picked:
        t = pick_teacher(teacher_by_sid.get(r["sid"], []), r["harness"])
        out.append({
            "king": king_id, "rollout_id": r["rollout_id"], "chunk": r["chunk"],
            "line": r["line"], "source": r["source"], "env_group": r["env_group"],
            "harness": r["harness"], "policy_id": r["policy_id"],
            "action_kind": r["action_kind"], "sid": r["sid"],
            "stop_condition": r["stop_condition"], "n_replies": r["n_replies"],
            "teacher": ({"rollout_id": t["rollout_id"], "chunk": t["chunk"],
                         "line": t["line"], "policy_id": t["policy_id"],
                         "harness": t["harness"], "outcome": t["outcome"],
                         "same_harness": t["same_harness"]} if t else None),
        })
    return out


def cell_table(sample: list[dict], rows: list[dict], king: str,
               env_groups: dict[str, str]) -> list[dict]:
    king_id = f"king-{digest12(king)}"
    supply: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if r["king"] == king_id and r["outcome"] == "failed" and r["n_replies"] > 0:
            supply[(env_groups.get(r["source"], "?"), r["harness"])] += 1
    picked: dict[tuple[str, str], int] = defaultdict(int)
    with_teacher: dict[tuple[str, str], int] = defaultdict(int)
    for s in sample:
        k = (s["env_group"], s["harness"])
        picked[k] += 1
        with_teacher[k] += s["teacher"] is not None
    return [{"env_group": k[0], "harness": k[1], "failed_available": supply[k],
             "sampled": picked[k], "with_teacher": with_teacher[k]}
            for k in sorted(set(supply) | set(picked))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--king", required=True, help="king digest (12+ hex) or 'current'")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-cell", type=int, default=16)
    ap.add_argument("--max-total", type=int, default=250)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-sync", action="store_true", help="use the chunk cache as is")
    ap.add_argument("--all", action="store_true",
                    help="every failed rollout of the king (daily / bulk mode); "
                         "per-cell and max-total are ignored")
    ap.add_argument("--include-excluded-sources", action="store_true",
                    help="also select sources the fold's [king_pivot] excludes")
    ap.add_argument("--include-single-reply", action="store_true",
                    help="--all: also judge one-reply rollouts (no pivot to find)")
    ap.add_argument("--state-json", type=Path, default=None,
                    help="resolve --king current from this validator state.json")
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    king = (resolve_current_king(state_json=args.state_json)["digest12"]
            if args.king == "current" else digest12(args.king))
    ts = TraceStore()
    if not args.no_sync:
        ts.sync()
    rows = ts.index(procs=args.procs)
    env_groups = load_env_groups()
    fold = load_king_pivot_config()
    excluded = frozenset() if args.include_excluded_sources else fold["exclude_sources"]
    if excluded:
        print(f"skipping sources the fold's [king_pivot] excludes: {sorted(excluded)}")
    sample = select_sample(rows, king=king, per_cell=args.per_cell,
                           max_total=args.max_total, seed=args.seed, env_groups=env_groups,
                           exclude_sources=excluded, all_failed=args.all,
                           include_single_reply=args.include_single_reply)
    write_jsonl(args.out, sample)
    table = cell_table(sample, rows, king, env_groups)
    (args.out.parent / "sample_cells.json").write_text(json.dumps(table, indent=1))
    print(f"king-{king}: sampled {len(sample)} failed rollouts "
          f"({sum(s['teacher'] is not None for s in sample)} with a teacher rollout)")
    for c in table:
        print(f"  {c['env_group']:<9} {c['harness']:<20} available {c['failed_available']:>4}"
              f"  sampled {c['sampled']:>3}  teacher {c['with_teacher']:>3}")


if __name__ == "__main__":
    main()
