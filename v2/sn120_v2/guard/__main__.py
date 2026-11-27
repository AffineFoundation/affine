"""The operator side of the pool quality guard: ``python -m sn120_v2.guard``.

Commands, none of which is ever called from the round loop:

  prejudge         judge candidates that the ledger has not decided for their
                   current body, and append their rows. Spends money and
                   reaches the network.
  monitor          read-only: what the ledger decides, how many outside
                   candidates are eligible, and today's estimated spend.
  apply            withdraw active tasks the guard has rejected, with reason
                   ``quality:<category>``.

The judge relay splits ``prejudge`` across two hosts, for when the judge key
lives on a host that does not hold the state:

  export-requests  (state host) the same selection and deterministic checks as
                   ``prejudge``; hard rejects go to the ledger as prejudge
                   writes them, the rest become LLM request rows. No network.
  judge-requests   (key host) judge the request rows. Needs only this repo and
                   the standard library. Public-safe verdict rows go to the
                   output file; the judge's own words go only to the
                   ``.private.jsonl`` file beside it, which never leaves that
                   host.
  ingest-verdicts  (state host) append the ledger rows of verdicts whose body
                   is still the candidate's current body. The rows are the
                   bytes prejudge would have written for the same verdict.

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
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from ..canonical import canonical_json
from ..envs.reserve import ingest_lock, reserve_dir
from ..errors import ConfigError, V2Error
from ..rotation import GUARD_PASS, GUARD_REJECT, parse_guard_ledger
from . import Budget, BudgetExhausted, Decision, decide, guard_version, precheck, record
from . import judge
from .hint import epoch_counts, hint_for, wrong_answers
from .store import guard_path, public_row

if TYPE_CHECKING:
    from .checks import Universe

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
    from ..cli import read_state  # the stack loader; not needed by judge-requests
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


def _universe(pool) -> tuple[dict[str, str], "Universe"]:
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
    from .checks import Universe  # needs the math grader
    return bodies, Universe(ranked)


def _candidates(pool, args: argparse.Namespace) -> list[str]:
    """Outside candidates the ledger has not decided for their current body, in catalog order, capped."""
    decided = {task_id: sha for mapping in (pool.guard.passes, pool.guard.rejects)
               for task_id, sha in mapping.items()}
    todo = [task_id for task_id in _outside(pool)
            if decided.get(task_id) != _content_sha(pool, task_id) and _content_sha(pool, task_id)]
    if args.ids:
        wanted = set(args.ids)
        todo = [task_id for task_id in todo if task_id in wanted]
    return todo[:args.limit]


def cmd_prejudge(args: argparse.Namespace) -> dict[str, Any]:
    pool, _params = _pool(args, "prejudge")
    todo = _candidates(pool, args)
    bodies, universe = _universe(pool)
    from .checks import NearIndex  # needs the math grader
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
    from ..envs.reserve.__main__ import withdraw_many_locked
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

# ---------------------------------------------------------------------------
# The judge relay: export-requests -> judge-requests -> ingest-verdicts
# ---------------------------------------------------------------------------

REQUEST_FIELDS = ("id", "content_sha", "problem", "answer", "flags", "hint")
VERDICT_FIELDS = frozenset({"id", "content_sha", "verdict", "category", "guard_version", "confidence"})
_GUARD_VERSION = re.compile(r"guard-v1:[0-9a-f]{16}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file; a torn last line (a crash mid-append) is ignored, any other bad line is an error."""
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines()
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            if number == len(lines):
                break
            raise ValueError(f"{path}:{number} is not JSON") from None
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        rows.append(row)
    return rows


def _write_atomic(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n"
                              for row in rows))
    temp.replace(path)


def _drop_torn_tail(path: Path) -> None:
    """Cut a partial last line left by a crash mid-append, so the next append starts on a fresh line."""
    if not path.exists():
        return
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        with path.open("r+b") as handle:
            handle.truncate(raw.rfind(b"\n") + 1)


def private_path(out: Path) -> Path:
    """``V.jsonl`` -> ``V.private.jsonl``: where the judge's own words are kept on the judge host."""
    out = Path(out)
    stem = out.name[:-len(".jsonl")] if out.name.endswith(".jsonl") else out.name
    return out.with_name(stem + ".private.jsonl")


def cmd_export_requests(args: argparse.Namespace) -> dict[str, Any]:
    """The deterministic half of ``prejudge``: hard rejects to the ledger, judge requests to a file."""
    pool, _params = _pool(args, "export-requests")
    todo = _candidates(pool, args)
    bodies, universe = _universe(pool)
    from .checks import NearIndex  # needs the math grader
    near = NearIndex(bodies)
    counts, wrong = _hints(args)
    finals: list[Decision] = []
    requests: list[dict] = []
    for task_id in todo:
        task = pool.catalog.task(task_id)
        final, result, hint_text = precheck(task, universe, near=near, hint=hint_for(task_id, counts, wrong))
        if final is not None:
            finals.append(final)
            continue
        requests.append({"id": task_id, "content_sha": result.content_sha, "problem": task.problem,
                         "answer": task.answer, "flags": list(result.flags), "hint": hint_text})
    _write_atomic(Path(args.out), requests)
    report = {"candidates": len(todo), "hard_rejected": len(finals), "requests": len(requests),
              "out": str(args.out), "guard_version": guard_version(),
              "categories": sorted({row.category for row in finals})}
    if args.dry_run:
        return report | {"dry_run": True}
    appended, skipped = record(reserve_dir(), finals)
    return report | {"ledger_appended": appended, "ledger_skipped": skipped}


class _SharedBudget:
    """A ``Budget`` that judge threads may share: the check and the charge are serialised."""

    def __init__(self, budget: Budget) -> None:
        self._budget = budget
        self._lock = threading.Lock()

    @property
    def limit_usd(self) -> float:
        return self._budget.limit_usd

    @property
    def spent_usd(self) -> float:
        return self._budget.spent_usd

    def exhausted(self) -> bool:
        with self._lock:
            return self._budget.exhausted()

    def charge(self, usage: dict) -> float:
        with self._lock:
            return self._budget.charge(usage)

    def cost(self, usage: dict) -> float:
        return self._budget.cost(usage)


def _verdict_rows(request: dict, verdict: "judge.Verdict", version: str) -> tuple[dict, dict]:
    """(public verdict row, private record) for one judged request."""
    from . import finish
    decision = finish(request["id"], request["content_sha"], verdict, version=version,
                      hint=str(request.get("hint") or ""))
    public = {key: value for key, value in decision.row().items() if key != "added_at"}
    public["confidence"] = verdict.confidence
    private = decision.private()
    private.pop("added_at", None)
    private["flags"] = list(request.get("flags") or [])
    private["judged_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return public, private


def _validate_request(row: dict) -> None:
    missing = [key for key in REQUEST_FIELDS if key not in row]
    if missing:
        raise ValueError(f"request row {row.get('id')!r} lacks {missing}")
    if not _SHA.fullmatch(str(row["content_sha"])):
        raise ValueError(f"request row {row['id']!r} has a malformed content_sha")


def cmd_judge_requests(args: argparse.Namespace) -> dict[str, Any]:
    """Judge request rows. Needs no state, no stack and no math grader: only the key in the environment.

    Resumable: a request whose (id, content_sha) already has a row in ``--out``
    is not judged again. Each verdict is appended as soon as it returns, private
    record first, so a crash loses at most the calls in flight.
    """
    requests = _read_jsonl(Path(args.requests))
    for row in requests:
        _validate_request(row)
    out = Path(args.out)
    private = private_path(out)
    done = {(row.get("id"), row.get("content_sha")) for row in _read_jsonl(out)}
    todo: list[dict] = []
    for row in requests:
        key = (row["id"], row["content_sha"])
        if key not in done:
            done.add(key)
            todo.append(row)
    budget_path = Path(args.budget_state) if args.budget_state else out.with_name("guard-budget.json")
    budget = _SharedBudget(Budget(budget_path))
    version = guard_version()
    report: dict[str, Any] = {"requests": len(requests), "already_judged": len(requests) - len(todo),
                              "judged": 0, "passed": 0, "undecided": 0, "guard_version": version,
                              "stopped": "", "out": str(out), "private": str(private)}
    if args.dry_run:
        return report | {"to_judge": len(todo), "dry_run": True}
    if not todo:
        return report | {"spent_usd_today": round(budget.spent_usd, 4)}
    out.parent.mkdir(parents=True, exist_ok=True)
    _drop_torn_tail(out)
    _drop_torn_tail(private)

    def one(request: dict):
        return request, judge.ask(request["problem"], request["answer"], flags=tuple(request.get("flags") or ()),
                                  hint=str(request.get("hint") or ""), budget=budget)

    failure: BaseException | None = None
    pending = iter(todo)
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor, \
            private.open("ab") as private_handle, out.open("ab") as public_handle:
        running = set()
        stop = False
        while True:
            while not stop and len(running) < max(1, args.parallel):
                request = next(pending, None)
                if request is None:
                    break
                running.add(executor.submit(one, request))
            if not running:
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                try:
                    request, verdict = future.result()
                except BudgetExhausted as error:
                    stop, report["stopped"] = True, str(error)
                    continue
                except BaseException as error:  # noqa: BLE001 - stop submitting, drain, then re-raise
                    stop, failure = True, failure or error
                    continue
                public, record_ = _verdict_rows(request, verdict, version)
                private_handle.write(json.dumps(record_, sort_keys=True, ensure_ascii=False).encode() + b"\n")
                private_handle.flush()
                public_handle.write(json.dumps(public, sort_keys=True, ensure_ascii=False).encode() + b"\n")
                public_handle.flush()
                report["judged"] += 1
                report["passed"] += public["verdict"] == GUARD_PASS
                report["undecided"] += verdict.undecided
    if failure is not None:
        raise failure
    return report | {"spent_usd_today": round(budget.spent_usd, 4)}


def _verdict_problem(row: dict) -> str | None:
    """Why a verdict row cannot become a ledger row, or None when it is well formed."""
    if set(row) != VERDICT_FIELDS:
        return "fields"
    if not isinstance(row["id"], str) or not _SHA.fullmatch(str(row["content_sha"])):
        return "id"
    if not _GUARD_VERSION.fullmatch(str(row["guard_version"])):
        return "guard_version"
    category = str(row["category"])
    if row["verdict"] == GUARD_PASS:
        if category != "ok" or row["confidence"] not in judge.PASS_CONFIDENCE:
            return "pass_inconsistent"
    elif row["verdict"] == GUARD_REJECT:
        if not category or category == "ok":
            return "reject_inconsistent"
    else:
        return "verdict"
    try:  # the same strict parse the rotation rule applies to the ledger
        row_ = public_row(row["id"], row["content_sha"], row["verdict"] == GUARD_PASS, category,
                          row["guard_version"])
        if row_["category"] != category:
            return "category"
        parse_guard_ledger(canonical_json(row_) + b"\n")
    except V2Error:
        return "ledger_row"
    return None


def cmd_ingest_verdicts(args: argparse.Namespace) -> dict[str, Any]:
    """Append the ledger rows of relayed verdicts whose body is still the candidate's current body."""
    pool, _params = _pool(args, "ingest-verdicts")
    rows = _read_jsonl(Path(args.verdicts))
    catalog = set(pool.catalog.task_ids)
    withdrawn = set(pool.reserve.withdrawn_ids)
    root = Path(reserve_dir())
    path = guard_path(root)
    on_disk = parse_guard_ledger(path.read_bytes() if path.exists() else b"")
    decided = {(task_id, sha) for mapping in (on_disk.passes, on_disk.rejects) for task_id, sha in mapping.items()}
    counts = {"invalid": 0, "unknown_or_withdrawn": 0, "stale_content_sha": 0, "already_in_ledger": 0,
              "duplicate_rows": 0}
    invalid_reasons: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    decisions: list[Decision] = []
    for row in rows:
        problem = _verdict_problem(row)
        if problem is not None:
            counts["invalid"] += 1
            invalid_reasons[problem] = invalid_reasons.get(problem, 0) + 1
            continue
        key = (row["id"], row["content_sha"])
        if key in seen:
            counts["duplicate_rows"] += 1
            continue
        seen.add(key)
        if row["id"] not in catalog or row["id"] in withdrawn:
            counts["unknown_or_withdrawn"] += 1
        elif _content_sha(pool, row["id"]) != row["content_sha"]:
            counts["stale_content_sha"] += 1
        elif key in decided:
            counts["already_in_ledger"] += 1
        else:
            decisions.append(Decision(task_id=row["id"], content_sha=row["content_sha"],
                                      passed=row["verdict"] == GUARD_PASS, category=str(row["category"]),
                                      guard_version=row["guard_version"]))
    report: dict[str, Any] = {"verdict_rows": len(rows), "eligible": len(decisions),
                              "passed": sum(1 for row in decisions if row.passed),
                              "skipped": counts, "invalid_reasons": invalid_reasons,
                              "guard_versions": sorted({row.guard_version for row in decisions})}
    if args.dry_run:
        return report | {"dry_run": True}
    appended, skipped = record(root, decisions)
    return report | {"ledger_appended": appended, "ledger_skipped": skipped}


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m sn120_v2.guard", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)
    for name, help_text in (("prejudge", "judge undecided candidates and append their ledger rows"),
                            ("monitor", "read-only guard and eligibility report"),
                            ("apply", "withdraw active tasks the guard rejected"),
                            ("export-requests", "deterministic checks; write judge request rows (no network)"),
                            ("ingest-verdicts", "append ledger rows for relayed verdicts that still match")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--state-dir", required=True, help="directory of the journal; the journal lock is never taken")
        sub.add_argument("--stack", required=True, help="module:function that builds the stack, as for the run loop")
        sub.add_argument("--params", help="parameters file, when the journal has none pinned")
        sub.add_argument("--budget-state", help="where today's estimated judge spend is kept")
        sub.add_argument("--dry-run", action="store_true", help="decide and report, write nothing")
        if name in ("prejudge", "export-requests"):
            sub.add_argument("--limit", type=int, default=200, help="the most candidates to take in one run")
            sub.add_argument("--ids", nargs="*", default=[], help="take only these candidate ids")
            sub.add_argument("--epoch-results", help="epoch results JSON, for the king hint's pass counts")
            sub.add_argument("--gate-rollouts", help="public gate rollout JSONL, for the most common wrong answer")
        if name == "export-requests":
            sub.add_argument("--out", required=True, help="request rows (JSONL) for judge-requests")
        if name == "ingest-verdicts":
            sub.add_argument("--verdicts", required=True, help="public-safe verdict rows from judge-requests")
        if name == "apply":
            sub.add_argument("--reserve", help="the live reserve directory")
            sub.add_argument("--run-log", help="the loop's run.log, which must show a completed open")
            sub.add_argument("--max", type=int, help=f"withdraw at most this many ids (never more than {MAX_WITHDRAW})")
            sub.add_argument("--force", action="store_true", help="ignore this round's marker")
    sub = commands.add_parser("judge-requests", help="judge request rows; needs only this repo, stdlib and the key")
    sub.add_argument("--requests", required=True, help="request rows from export-requests")
    sub.add_argument("--out", required=True, help="public-safe verdict rows (appended; resumable)")
    sub.add_argument("--parallel", type=int, default=24, help="judge calls in flight")
    sub.add_argument("--budget-state", help="today's estimated spend (default: guard-budget.json beside --out)")
    sub.add_argument("--dry-run", action="store_true", help="count what would be judged, call nothing")
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = {"prejudge": cmd_prejudge, "monitor": cmd_monitor, "apply": cmd_apply,
                  "export-requests": cmd_export_requests, "judge-requests": cmd_judge_requests,
                  "ingest-verdicts": cmd_ingest_verdicts}[args.command](args)
    except (V2Error, ConfigError, OSError, ValueError) as error:
        print(json.dumps({"error": type(error).__name__, "reason": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
