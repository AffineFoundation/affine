"""Pod-local Generator scorer orchestration. No chain, R2 or publication I/O."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..canonical import canonical_json
from ..errors import ConfigError, ContractError, InfraError, VerifyError
from .http import LocalHttp, default_urls
from .pool import PriorPool
from .scoring import (
    DONE_FILE,
    PROGRESS_FILE,
    RESULTS_FILE,
    ROLLOUTS_DIR,
    ScoringJob,
    TaskState,
    initial_states,
    load_rollouts,
    make_results,
    next_batch,
    pending_rollouts,
    settle,
)
from .worker import BatchRunner, atomic_write, repair_checkpoints


def _utc(timestamp: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def check_out(out: Path) -> None:
    if out.exists() and not out.is_dir():
        raise ContractError(f"Generator output {out} is not a directory")
    if out.is_symlink():
        raise ContractError("Generator output must not be a symlink")
    out.mkdir(parents=True, exist_ok=True)
    (out / ROLLOUTS_DIR).mkdir(exist_ok=True)


def _output_identity(job: ScoringJob) -> dict[str, Any]:
    return {key: value for key, value in job.to_dict().items() if key != "seed_secret"} | {
        "seed_commitment": hashlib.sha256(job.seed_secret.encode("utf-8")).hexdigest()
    }


def recover(job: ScoringJob, out: Path, *, shard: tuple[int, int] = (0, 1)) -> tuple[dict[str, TaskState], bool]:
    """Return checkpointed state and whether an identical completed run exists."""
    check_out(out)
    identity_path = out / "job-identity.json"
    identity = _output_identity(job)
    if identity_path.exists():
        try:
            if json.loads(identity_path.read_bytes()) != identity:
                raise VerifyError("Generator output belongs to a different private job")
        except (OSError, ValueError) as exc:
            if isinstance(exc, VerifyError):
                raise
            raise VerifyError("Generator output job identity is unreadable") from exc
    else:
        atomic_write(identity_path, canonical_json(identity))
    repair_checkpoints(job, out)
    states = initial_states(job, shard)
    load_rollouts(job, states, out)
    done_path = out / DONE_FILE
    if done_path.exists():
        try:
            if json.loads(done_path.read_bytes()) != identity:
                raise VerifyError("Generator DONE marker does not match the job")
            result = json.loads((out / RESULTS_FILE).read_bytes())
            expected = result["tasks"]
            actual = [
                {"id": task.task_id, "hotkey": task.hotkey, "s": states[task.task_id].s, "n": states[task.task_id].n}
                for task in job.tasks
            ]
            if [{key: row[key] for key in actual[0]} for row in expected] != actual:
                raise VerifyError("Generator DONE marker does not match rollout checkpoints")
        except (OSError, ValueError, KeyError, IndexError) as exc:
            if isinstance(exc, VerifyError):
                raise
            raise VerifyError("Generator completed output is unreadable") from exc
        return states, True
    return states, False


def progress(job: ScoringJob, states: Mapping[str, TaskState], stage: str, out: Path, now: float) -> None:
    """Write operational state only; never reveal the seed before completion."""
    rows = [
        {
            "id": task.task_id,
            "hotkey": task.hotkey,
            "s": states[task.task_id].s,
            "n": states[task.task_id].n,
            "decision": states[task.task_id].decision,
            "ungraded": len(states[task.task_id].responses),
        }
        for task in job.tasks if task.task_id in states
    ]
    atomic_write(out / PROGRESS_FILE, canonical_json({"epoch": job.epoch, "stage": stage, "updated_at": now, "tasks": rows}))


def prepare_file_descriptors(clients_per_server: int) -> int:
    """Raise the soft FD limit before opening inference sockets, or cap safely."""
    target = min(65536, resource.getrlimit(resource.RLIMIT_NOFILE)[1])
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 8 * clients_per_server + 256
    if soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
        except (OSError, ValueError):
            pass
    if soft < needed:
        return max(1, min(clients_per_server, (soft - 256) // 8))
    return clients_per_server


def run_job(
    job: ScoringJob,
    out: Path,
    *,
    pool: Any,
    grader: Any,
    urls: Sequence[str] | None = None,
    http: Any = None,
    clients_per_server: int = 32,
    clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
    shard: tuple[int, int] = (0, 1),
) -> dict[str, Any]:
    """Resume or finish a scoring job, returning its complete local results."""
    started_at, started = clock(), monotonic()
    clients_per_server = prepare_file_descriptors(clients_per_server)
    urls = tuple(urls or default_urls())
    states, completed = recover(job, out, shard=shard)
    if completed:
        return json.loads((out / RESULTS_FILE).read_bytes())
    progress(job, states, "recovered", out, clock())
    own_http = http is None
    transport = http or LocalHttp()
    health_started = monotonic()
    needs_inference = any(
        not state.final and (state.n < 32 or state.decision == "continue")
        for state in states.values()
    )
    if clock() < job.budget_deadline and needs_inference:
        for url in urls:
            transport.check(url, min(60.0, max(0.001, job.budget_deadline - clock())))
    ready_at = clock()
    runner = BatchRunner(
        job,
        states,
        out,
        pool,
        grader,
        urls=urls,
        http=transport,
        clients_per_server=clients_per_server,
        clock=clock,
        monotonic=monotonic,
    )
    capacity = len(urls) * clients_per_server
    stage_capacity = max(capacity, 32 * len(job.tasks))
    try:
        while clock() < job.budget_deadline:
            pending = pending_rollouts(states)
            if pending:
                runner.run(pending, [])
            for state in states.values():
                if not state.final and state.n == len(state.rollouts) and state.n not in state.responses:
                    settle(state)
            batch, stage = next_batch(job, states, stage_capacity)
            if not batch:
                if pending_rollouts(states):
                    continue
                break
            progress(job, states, stage, out, clock())
            runner.run([], batch)
            progress(job, states, "checkpointed", out, clock())
        exhausted = clock() >= job.budget_deadline
        # Responses that arrived just before cancellation can still be graded;
        # no fresh inference may be dispatched at or beyond the deadline.
        pending = pending_rollouts(states)
        if pending:
            runner.run(pending, [])
        for state in states.values():
            settle(state, exhausted=exhausted)
        finished_at = clock()
        wall_s = max(0.0, monotonic() - started)
        timings = {
            "started_at": _utc(started_at),
            "ready_at": _utc(ready_at),
            "finished_at": _utc(finished_at),
            "wall_s": wall_s,
            "startup_s": max(0.0, ready_at - started_at),
            "health_s": monotonic() - health_started,
            "generate_s": runner.usage.generate_seconds,
            "grade_s": runner.usage.grade_seconds,
            "budget_exhausted": exhausted,
        }
        throughput = {
            "rollouts": sum(state.n for state in states.values()),
            "new_rollouts": runner.usage.completed,
            "requests": runner.usage.requests,
            "prompt_tokens": runner.usage.prompt_tokens,
            "completion_tokens": runner.usage.completion_tokens,
            "rollouts_per_s": runner.usage.completed / wall_s if wall_s else 0.0,
            "completion_tokens_per_s": runner.usage.completion_tokens / wall_s if wall_s else 0.0,
            "clients_per_server": clients_per_server,
            "servers": len(urls),
        }
        results = make_results(job, states, timings=timings, throughput=throughput, servers=urls)
        if shard != (0, 1):
            results["shard"] = {"index": shard[0], "count": shard[1], "tasks": len(states)}
        for task in job.tasks:
            if task.task_id not in states:
                continue
            path = out / ROLLOUTS_DIR / f"{task.task_id}.jsonl"
            if not path.exists():
                atomic_write(path, b"")
        atomic_write(out / RESULTS_FILE, canonical_json(results) + b"\n")
        # DONE is last: a publisher seeing it can always read complete results.
        atomic_write(out / DONE_FILE, canonical_json(_output_identity(job)))
        return results
    finally:
        try:
            progress(job, states, "stopped", out, clock())
        finally:
            if own_http:
                transport.cancel()


def lock_job(job: ScoringJob, root: Path) -> Any:
    directory = root / "jobs" / job.epoch
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / ".scorer.lock").open("a+b")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise InfraError("another Generator scorer holds this epoch's job lock") from None
    return handle


def real_run(args: argparse.Namespace) -> dict[str, Any]:
    from ..envs.math_grader import GradePool

    job = ScoringJob.load(args.job)
    try:
        shard = tuple(int(value) for value in args.shard.split("/", 1))
        if len(shard) != 2 or shard[1] < 1 or not 0 <= shard[0] < shard[1]:
            raise ValueError
    except (ValueError, TypeError):
        raise ConfigError("scorer --shard must be I/K with 0 <= I < K") from None
    out = Path(args.out) if args.out else Path(args.root) / "jobs" / job.epoch / "out"
    lock = lock_job(job, Path(args.root))
    try:
        pool = PriorPool(Path(args.pool_cache))
        grader = GradePool(workers=args.grade_workers, timeout_s=args.grade_timeout)
        try:
            return run_job(
                job,
                out,
                pool=pool,
                grader=grader,
                clients_per_server=args.clients_per_server,
                shard=args.shard,
            )
        finally:
            grader.close()
    finally:
        lock.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m sn120_v2.generator.scorer")
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or resume one private job on the GPU pod")
    run.add_argument("job", help="private scoring job JSON")
    run.add_argument("--root", default="/root/gen", help="job root (default: /root/gen)")
    run.add_argument("--pool-cache", default="/root/gen/pool", help="verified prior-v1 shard cache")
    run.add_argument("--out", help="override output directory (local testing only)")
    run.add_argument("--clients-per-server", type=int, default=224)
    run.add_argument("--shard", default="0/1", help="owned task shard I/K (default 0/1)")
    run.add_argument("--grade-workers", type=int, default=16)
    run.add_argument("--grade-timeout", type=float, default=60.0)
    test = commands.add_parser("selftest", help="run two jobs against eight fake loopback vLLM servers")
    test.add_argument("--root", help="isolated test root (default: a fresh temporary directory)")
    test.add_argument("--verbose", action="store_true")
    test.add_argument("--first-port", type=int, default=0, help="first of eight loopback ports; 0 allocates free CI ports")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "selftest":
        from .selftest import selftest

        return selftest(root=args.root, verbose=args.verbose, first_port=args.first_port)
    results = real_run(args)
    # Only paths and aggregate counts: no job secret, prompts or rollout text.
    counts = results["throughput"]
    print(f"Generator {results['epoch']}: {counts['rollouts']} graded rollouts; results are in the job's out directory")
    return 0
