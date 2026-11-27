"""The operator side of the pool quality guard: ``python -m sn120_v2.guard``.

Three commands, none of which is ever called from the round loop:

  prejudge  judge candidates that the ledger has not decided for their current
            body, and append their rows. This is the only command that spends
            money and the only one that reaches the network.
  monitor   read-only: what the ledger decides, how many outside candidates are
            eligible, and today's estimated spend.
  apply     withdraw active tasks the guard has rejected, with reason
            ``quality:<category>``.

``apply`` is deliberately conservative. It runs between opens, never takes the
journal lock, takes the reserve's exclusive ingest lock for the one batched
pointer rewrite, caps a round at ``MAX_WITHDRAW`` ids, and refuses to withdraw
more active tasks than the eligible supply can replace. It writes a per-round
marker, so running it twice in one round is a no-op.

Running it in the middle of a round is not safe: a withdrawn id leaves the
catalog, and a learner that restarts before the round closes can no longer
resolve the active tasks it already issued. ``apply`` therefore requires the
run log to show that the current round's open has completed, and the operator
is expected to run it in the close/open gap.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from ..cli import read_state
from ..envs.reserve import ingest_lock, reserve_dir
from ..envs.reserve.__main__ import withdraw_many_locked
from ..errors import ConfigError, V2Error
from . import Budget, BudgetExhausted, Decision, NearIndex, Universe, decide, guard_version, record
from .hint import epoch_counts, hint_for, wrong_answers

MAX_WITHDRAW = 300  # the most active ids one round may withdraw
REPLACEMENT_BUFFER = 50  # eligible passes that must remain after a batch, so the next open can still fill
OPEN_LINE = re.compile(r"open: the public files of round (\S+)")
MARKER_DIR = "guard-applied"


def run_log_path(args: argparse.Namespace) -> Path:
    given = getattr(args, "run_log", None) or os.environ.get("GUARD_RUN_LOG")
    if given:
        return Path(given)
    work = os.environ.get("SN120_WORK_DIR")
    if not work:
        raise ConfigError("apply needs --run-log, GUARD_RUN_LOG or SN120_WORK_DIR to find the loop's run.log")
    return Path(work) / "logs" / "run.log"


def last_opened_round(path: Path) -> str:
    """The round whose open has completed, from the last matching run-log line."""
    try:
        text = path.read_text(errors="replace")
    except OSError as error:
        raise ConfigError(f"apply cannot read the run log at {path}: {error}") from None
    matches = OPEN_LINE.findall(text)
    if not matches:
        raise ConfigError(f"the run log at {path} shows no completed open; apply refuses to run mid-round")
    return matches[-1]


def marker_path(root: Path, round_id: str) -> Path:
    return Path(root) / MARKER_DIR / f"{round_id}.json"


def _pool(args: argparse.Namespace, command: str):
    """The live PoolRuntime from the operator's stack. ``read_state`` never takes the journal lock."""
    _state, params, stack = read_state(args, command)
    if stack is None or stack.pool is None:
        raise ConfigError(f"{command} needs --stack with a pool runtime")
    return stack.pool, params


def _outside(pool) -> list[str]:
    """Non-withdrawn catalog ids that are not in the active set, in catalog order."""
    active = set(pool.active_ids)
    withdrawn = set(pool.reserve.withdrawn_ids)
    return [task_id for task_id in pool.catalog.task_ids if task_id not in active and task_id not in withdrawn]


def _content_sha(pool, task_id: str) -> str | None:
    meta = pool.candidate_meta.get(task_id) or {}
    return meta.get("content_sha")


def _eligible(pool, ids: Sequence[str]) -> list[str]:
    return [task_id for task_id in ids if pool.guard.eligible(task_id, _content_sha(pool, task_id))]


def _hints(args: argparse.Namespace) -> tuple[dict, dict]:
    counts: dict = {}
    wrong: dict = {}
    if getattr(args, "epoch_results", None):
        counts = epoch_counts(json.loads(Path(args.epoch_results).read_bytes()))
    if getattr(args, "gate_rollouts", None):
        rows = [json.loads(line) for line in Path(args.gate_rollouts).read_bytes().splitlines() if line.strip()]
        wrong = wrong_answers(rows)
    return counts, wrong


def withdrawn_bodies(root: Path, withdrawn: set[str]) -> dict[str, str]:
    """The problems of withdrawn reserve rows, which the catalog no longer carries.

    A candidate that repeats a task we already threw out for quality is still a
    duplicate, so the duplicate check has to see those bodies too.
    """
    out: dict[str, str] = {}
    shards = Path(root) / "shards"
    if not withdrawn or not shards.is_dir():
        return out
    for shard in sorted(shards.glob("*.jsonl")):
        for line in shard.read_bytes().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") in withdrawn and "problem" in row:
                out[row["id"]] = row["problem"]
    return out


def _universe(pool) -> tuple[dict[str, str], Universe]:
    """Every body a candidate could repeat, in priority order: active, withdrawn, then the rest.

    The owner of a repeated problem is the first task in that order. Active
    tasks come first because they are in use, and withdrawn bodies come before
    the candidates so that a body we already threw out cannot return under a
    new id. Among two identical candidates the earlier one owns the body and
    the later one is the duplicate.
    """
    active = [task_id for task_id in pool.active_ids if task_id in set(pool.catalog.task_ids)]
    rest = [task_id for task_id in pool.catalog.task_ids if task_id not in set(active)]
    bodies = {task_id: pool.catalog.task(task_id).problem for task_id in (*active, *rest)}
    gone = withdrawn_bodies(reserve_dir(), set(pool.reserve.withdrawn_ids))
    ranked = ([(task_id, bodies[task_id], "active") for task_id in active]
              + [(task_id, problem, "withdrawn") for task_id, problem in sorted(gone.items())]
              + [(task_id, bodies[task_id], "pool") for task_id in rest])
    return bodies, Universe(ranked)


def cmd_prejudge(args: argparse.Namespace) -> dict[str, Any]:
    pool, _params = _pool(args, "prejudge")
    decided = {task_id: sha for mapping in (pool.guard.passes, pool.guard.rejects)
               for task_id, sha in mapping.items()}
    todo = [task_id for task_id in _outside(pool)
            if decided.get(task_id) != _content_sha(pool, task_id) and _content_sha(pool, task_id)]
    if args.ids:
        wanted = set(args.ids)
        todo = [task_id for task_id in todo if task_id in wanted]
    todo = todo[:args.limit]
    bodies, universe = _universe(pool)
    near = NearIndex(bodies)
    counts, wrong = _hints(args)
    budget = Budget(Path(args.budget_state or (reserve_dir() / "guard-budget.json")))
    decisions: list[Decision] = []
    stopped = ""
    for task_id in todo:
        task = pool.catalog.task(task_id)
        try:
            decisions.append(decide(task, universe, near=near, hint=hint_for(task_id, counts, wrong), budget=budget))
        except BudgetExhausted as error:
            stopped = str(error)
            break
    report = {"candidates": len(todo), "judged": len(decisions), "guard_version": guard_version(),
              "passed": sum(1 for row in decisions if row.passed),
              "spent_usd_today": round(budget.spent_usd, 4), "stopped": stopped}
    if args.dry_run:
        report["dry_run"] = True
        report["categories"] = sorted({row.category for row in decisions})
        return report
    appended, skipped = record(reserve_dir(), decisions)
    return report | {"ledger_appended": appended, "ledger_skipped": skipped}


def cmd_monitor(args: argparse.Namespace) -> dict[str, Any]:
    pool, params = _pool(args, "monitor")
    outside = _outside(pool)
    eligible = _eligible(pool, outside)
    active_rejected = [task_id for task_id in pool.active_ids
                       if pool.guard.rejects.get(task_id) == _content_sha(pool, task_id)]
    budget = Budget(Path(args.budget_state or (reserve_dir() / "guard-budget.json")))
    return {
        "rotation_rule": params.rotation_rule,
        "guard_version": guard_version(),
        "guard_ledger_sha": pool.guard.sha256,
        "guard_rows": pool.guard.count,
        "guard_passes": len(pool.guard.passes),
        "guard_rejects": len(pool.guard.rejects),
        "active_size": len(pool.active_ids),
        "active_rejected": len(active_rejected),
        "outside_candidates": len(outside),
        "outside_eligible": len(eligible),
        "undecided_outside": len(outside) - len(eligible) - sum(
            1 for task_id in outside if pool.guard.rejects.get(task_id) == _content_sha(pool, task_id)),
        "withdraw_headroom": max(0, min(MAX_WITHDRAW, len(eligible) - REPLACEMENT_BUFFER))
        if params.rotation_rule == "info-guard" else MAX_WITHDRAW,
        "spent_usd_today": round(budget.spent_usd, 4),
        "daily_limit_usd": budget.limit_usd,
    }


def withdrawn_on_disk(root: Path) -> set[str]:
    """The ids the live ``withdrawn.jsonl`` already names."""
    path = Path(root) / "withdrawn.jsonl"
    if not path.exists():
        return set()
    out = set()
    for line in path.read_bytes().splitlines():
        if line.strip():
            out.add(json.loads(line)["id"])
    return out


def cmd_apply(args: argparse.Namespace) -> dict[str, Any]:
    pool, params = _pool(args, "apply")
    root = Path(args.reserve or reserve_dir())
    round_id = last_opened_round(run_log_path(args))
    marker = marker_path(root, round_id)
    if marker.exists() and not args.force:
        return {"round_id": round_id, "withdrawn": 0, "skipped": "already applied in this round",
                "marker": str(marker)}
    # The reserve on disk, not the runtime's snapshot of it, decides what is
    # still to withdraw: the pool may have been loaded before an earlier batch.
    already = withdrawn_on_disk(root) | set(pool.reserve.withdrawn_ids)
    bad = [task_id for task_id in pool.active_ids
           if task_id not in already and pool.guard.rejects.get(task_id) == _content_sha(pool, task_id)]
    cap = MAX_WITHDRAW
    limit_reason = "round cap"
    if params.rotation_rule == "info-guard":
        supply = len(_eligible(pool, _outside(pool))) - REPLACEMENT_BUFFER
        if supply < cap:
            cap, limit_reason = max(0, supply), "eligible replacement supply"
    if args.max is not None:
        cap = min(cap, args.max)
    chosen = sorted(bad)[:cap]
    report = {"round_id": round_id, "rejected_active": len(bad), "cap": cap, "limited_by": limit_reason,
              "withdrawing": len(chosen), "ids": chosen}
    if not chosen:
        return report | {"withdrawn": 0}
    if args.dry_run:
        return report | {"dry_run": True, "withdrawn": 0}
    rows = [{"id": task_id, "reason": "quality:" + _category(pool, task_id)} for task_id in chosen]
    with ingest_lock(root, exclusive=True):
        result = withdraw_many_locked(root, rows)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"round_id": round_id, "ids": chosen,
                                      "guard_version": guard_version()}, sort_keys=True) + "\n")
    return report | {"withdrawn": result["withdrawn"], "marker": str(marker)}


def _category(pool, task_id: str) -> str:
    """The reject category recorded for this id's current body, from the private verdict file when present."""
    path = Path(reserve_dir()) / "verdicts.jsonl"
    sha = _content_sha(pool, task_id)
    if path.exists():
        for line in reversed(path.read_bytes().splitlines()):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("id") == task_id and row.get("content_sha") == sha and row.get("category"):
                return str(row["category"])
    return "guard"


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m sn120_v2.guard", description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    for name, help_text in (("prejudge", "judge undecided candidates and append their ledger rows"),
                            ("monitor", "read-only guard and eligibility report"),
                            ("apply", "withdraw active tasks the guard rejected")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--state-dir", required=True, help="directory of the journal; the journal lock is never taken")
        sub.add_argument("--stack", required=True, help="module:function that builds the stack, as for the run loop")
        sub.add_argument("--params", help="parameters file, when the journal has none pinned")
        sub.add_argument("--budget-state", help="where today's estimated judge spend is kept")
        sub.add_argument("--dry-run", action="store_true", help="decide and report, write nothing")
        if name == "prejudge":
            sub.add_argument("--limit", type=int, default=200, help="the most candidates to judge in one run")
            sub.add_argument("--ids", nargs="*", default=[], help="judge only these candidate ids")
            sub.add_argument("--epoch-results", help="epoch results JSON, for the king hint's pass counts")
            sub.add_argument("--gate-rollouts", help="public gate rollout JSONL, for the most common wrong answer")
        if name == "apply":
            sub.add_argument("--reserve", help="the live reserve directory")
            sub.add_argument("--run-log", help="the loop's run.log, which must show a completed open")
            sub.add_argument("--max", type=int, help=f"withdraw at most this many ids (never more than {MAX_WITHDRAW})")
            sub.add_argument("--force", action="store_true", help="ignore this round's marker")
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = {"prejudge": cmd_prejudge, "monitor": cmd_monitor, "apply": cmd_apply}[args.command](args)
    except (V2Error, ConfigError, OSError, ValueError) as error:
        print(json.dumps({"error": type(error).__name__, "reason": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
