"""Pure job validation, fair scheduling and result assembly for Generator.

The private seed is used only to derive rollout seeds and to write final
results. It never appears in exceptions or non-secret progress documents.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..canonical import canonical_json
from ..errors import ContractError, VerifyError
from ..types import NULL_MODEL
from .constants import (
    ACCEPT,
    CONTINUE,
    MAX_SAMPLES,
    MIN_SAMPLES,
    POOL_ID,
    POOL_MANIFEST_SHA256,
    REJECT,
    SAMPLE_INCREMENT,
    SAMPLING,
)
from .decisions import decision
from .epoch import epoch_deadline, epoch_id, epoch_results, rollout_object
from .protocol import valid_task_id

JOB_VERSION = "affine-generator-scoring-job-v1"
RESULTS_VERSION = "affine-generator-results-v1"
ROLLOUT_VERSION = "affine-generator-rollout-v1"
UNSCORED = "unscored"
RESULTS_FILE = "results.json"
PROGRESS_FILE = "progress.json"
ROLLOUTS_DIR = "rollouts"
DONE_FILE = "DONE"
DEFAULT_ROOT = Path("/root/gen")
MAX_JOB_BYTES = 16 * 1024 * 1024
MAX_TASKS = 100_000
MAX_SECRET_BYTES = 4096
MAX_HEAD_VERSION = 64
SERVER_COUNT = 8
FIRST_PORT = 8100


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(f"Generator scoring job: {message}")


def _number(value: Any, name: str) -> float:
    _need(type(value) in (int, float) and math.isfinite(value) and value > 0, f"{name} must be a finite positive Unix timestamp")
    return float(value)


def _head_text(value: Any) -> str:
    if isinstance(value, Mapping):  # a complete HeadVersion object is also convenient
        value = value.get("version_id")
    _need(isinstance(value, str) and 0 < len(value) <= MAX_HEAD_VERSION, "head_version must be a nonempty string")
    _need("\x00" not in value, "head_version contains NUL")
    return value


def rollout_seed(secret: str, task_id: str, sample: int) -> int:
    """Exactly ``int(sha256(secret:task_id:j)[:12],16)`` (48 bits)."""
    _need(valid_task_id(task_id), "invalid rollout task ID")
    _need(type(sample) is int and 0 <= sample < MAX_SAMPLES, "rollout index must be 0..31")
    _need(isinstance(secret, str) and bool(secret), "seed secret must not be empty")
    return int(hashlib.sha256(f"{secret}:{task_id}:{sample}".encode("utf-8")).hexdigest()[:12], 16)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    hotkey: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.task_id, "hotkey": self.hotkey}


@dataclass(frozen=True, slots=True)
class ScoringJob:
    epoch: str
    tasks: tuple[TaskSpec, ...]
    head_version: str
    seed_secret: str
    budget_deadline: float
    pool_pin: str = POOL_ID
    version: str = JOB_VERSION

    def __post_init__(self) -> None:
        _need(self.version == JOB_VERSION, f"version must be {JOB_VERSION}")
        try:
            canonical = epoch_id(self.epoch)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Generator scoring job: {exc}") from exc
        _need(self.epoch == canonical, "epoch must be its canonical UTC deadline ID")
        _need(1 <= len(self.tasks) <= MAX_TASKS, f"tasks must contain 1..{MAX_TASKS} unique IDs")
        _need(len({task.task_id for task in self.tasks}) == len(self.tasks), "task IDs must be unique")
        for task in self.tasks:
            _need(valid_task_id(task.task_id), "every task ID must be an original lowercase 20-hex pool ID")
            _need(isinstance(task.hotkey, str) and bool(task.hotkey), "every task needs a nonempty owner hotkey")
        _head_text(self.head_version)
        _need(isinstance(self.seed_secret, str) and bool(self.seed_secret), "seed_secret must be a nonempty UTF-8 string")
        _need(len(self.seed_secret.encode("utf-8")) <= MAX_SECRET_BYTES, "seed_secret is too large")
        _need(type(self.budget_deadline) in (int, float) and math.isfinite(self.budget_deadline), "budget_deadline must be finite")
        _need(self.budget_deadline > epoch_deadline(self.epoch).timestamp(), "budget_deadline must be after the epoch deadline")
        _need(self.pool_pin in (POOL_ID, POOL_MANIFEST_SHA256), "pool_pin must be the exact published pool ID or manifest SHA-256")

    @property
    def pool(self) -> str:
        return self.pool_pin if ":" in self.pool_pin else POOL_ID

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    @property
    def owners(self) -> dict[str, str]:
        return {task.task_id: task.hotkey for task in self.tasks}

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "epoch": self.epoch,
            "tasks": [task.to_dict() for task in self.tasks],
            "head_version": self.head_version,
            "seed_secret": self.seed_secret,
            "budget_deadline": self.budget_deadline,
            "pool_pin": self.pool_pin,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScoringJob":
        _need(isinstance(data, Mapping), "expected a JSON object")
        fields = {"version", "epoch", "tasks", "head_version", "seed_secret", "budget_deadline", "pool_pin"}
        _need(set(data) == fields, f"fields must be exactly {', '.join(sorted(fields))}")
        raw_tasks = data["tasks"]
        _need(isinstance(raw_tasks, list), "tasks must be a list of {id, hotkey} objects")
        tasks = []
        for item in raw_tasks:
            _need(isinstance(item, Mapping), "each task must be an object")
            _need(set(item) == {"id", "hotkey"}, "each task must have exactly id and hotkey")
            tasks.append(TaskSpec(item["id"], item["hotkey"]))
        try:
            return cls(
                epoch=data["epoch"],
                tasks=tuple(tasks),
                head_version=_head_text(data["head_version"]),
                seed_secret=data["seed_secret"],
                budget_deadline=_number(data["budget_deadline"], "budget_deadline"),
                pool_pin=data["pool_pin"],
                version=data["version"],
            )
        except KeyError as exc:
            raise ContractError(f"Generator scoring job: missing field {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"Generator scoring job: invalid field type: {exc}") from exc

    @classmethod
    def from_json(cls, raw: bytes | str) -> "ScoringJob":
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        _need(isinstance(raw, bytes) and 0 < len(raw) <= MAX_JOB_BYTES, f"JSON must contain 1..{MAX_JOB_BYTES} bytes")
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ContractError(f"Generator scoring job: invalid UTF-8 JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> "ScoringJob":
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise ContractError(f"Generator scoring job cannot be read: {type(exc).__name__}: {exc}") from exc
        return cls.from_json(raw)


@dataclass(slots=True)
class TaskState:
    spec: TaskSpec
    s: int = 0
    n: int = 0
    decision: str = UNSCORED
    final: bool = False
    q: float = 0.5
    reason: str = "not reached"
    terminal: bool = False
    rollouts: list[dict[str, Any]] = field(default_factory=list)
    responses: dict[int, dict[str, Any]] = field(default_factory=dict)

    def counts(self) -> tuple[int, int]:
        if not 0 <= self.s <= self.n <= MAX_SAMPLES:
            raise VerifyError(f"Generator task {self.spec.task_id} has invalid sample counts")
        return self.s, self.n


def initial_states(job: ScoringJob) -> dict[str, TaskState]:
    return {task.task_id: TaskState(task) for task in job.tasks}


def stage_one_rounds(job: ScoringJob, states: Mapping[str, TaskState]) -> Iterator[list[tuple[str, int]]]:
    """Interleave owners in job order; each task receives eight samples."""
    queues: dict[str, deque[str]] = defaultdict(deque)
    owners: list[str] = []
    for task in job.tasks:
        if task.hotkey not in queues:
            owners.append(task.hotkey)
        queues[task.hotkey].append(task.task_id)
    while any(queues.values()):
        for owner in owners:
            queue = queues[owner]
            if not queue:
                continue
            task_id = queue.popleft()
            state = states[task_id]
            if not state.final and state.n < MIN_SAMPLES:
                yield [(task_id, sample) for sample in range(state.n, MIN_SAMPLES)]


def pending_rollouts(states: Mapping[str, TaskState]) -> list[tuple[str, int]]:
    """Durable responses whose grades were not committed before a crash."""
    return [
        (task_id, state.n)
        for task_id, state in states.items()
        if state.decision == UNSCORED and not state.terminal and state.n in state.responses
    ]


def undecided_order(job: ScoringJob, states: Mapping[str, TaskState]) -> list[list[tuple[str, int]]]:
    """Eight-rollout groups, closest q to 0.5 first; ties retain job order."""
    positions = {task.task_id: index for index, task in enumerate(job.tasks)}
    groups = []
    for task in job.tasks:
        state = states[task.task_id]
        if state.final or state.decision != UNSCORED or state.n < MIN_SAMPLES or state.n >= MAX_SAMPLES:
            continue
        current = decision(state.s, state.n)
        if current.action != CONTINUE or current.next_total is None:
            raise VerifyError(f"Generator task {task.task_id} has an inconsistent decision state")
        groups.append((abs(current.q - 0.5), positions[task.task_id], [(task.task_id, j) for j in range(state.n, current.next_total)]))
    groups.sort(key=lambda group: (group[0], group[1]))
    return [group[2] for group in groups]


def next_batch(job: ScoringJob, states: Mapping[str, TaskState], capacity: int) -> tuple[list[tuple[str, int]], str]:
    """Fill server capacity from fair stage 1, then closest-q escalation."""
    if capacity < 1:
        return [], "capacity"
    if pending_rollouts(states):
        return [], "grade"
    batch: list[tuple[str, int]] = []
    for stage, groups in (("stage1", stage_one_rounds(job, states)), ("escalation", iter(undecided_order(job, states)))):
        for group in groups:
            batch.extend(group)
            if len(batch) >= capacity:
                return batch[:capacity], stage
        if batch:
            return batch, stage
    return batch, "decided"


def settle(state: TaskState, *, exhausted: bool = False) -> None:
    """Delegate decisions to the protocol; never accept an unreached task."""
    state.counts()
    if state.final:
        return
    if state.n >= MAX_SAMPLES or (exhausted and state.n >= MIN_SAMPLES):
        result = decision(state.s, state.n, final=exhausted and state.n < MAX_SAMPLES)
    elif not exhausted and state.n >= MIN_SAMPLES and state.n % SAMPLE_INCREMENT == 0:
        result = decision(state.s, state.n)
    else:
        if exhausted:
            state.final = True
            state.reason = f"unscored: {state.n} completed rollouts; {MIN_SAMPLES} required"
        elif state.n:
            state.q = decision(state.s, state.n).q
        return
    if result.action == CONTINUE:
        state.q, state.reason = result.q, result.reason
    else:
        state.decision, state.q, state.final, state.terminal, state.reason = result.action, result.q, True, True, result.reason


def finish_grade(state: TaskState, record: dict[str, Any]) -> None:
    """Commit one already-durable grade; settle at the batch boundary."""
    if state.n >= len(state.rollouts) or state.rollouts[state.n] is not record:
        raise VerifyError(f"Generator task {state.spec.task_id} has a non-prefix grade")
    grade = record.get("grade")
    if not isinstance(grade, Mapping) or type(grade.get("passed")) is not bool:
        raise VerifyError(f"Generator task {state.spec.task_id} has no complete grade")
    if state.terminal or state.decision != UNSCORED:
        raise VerifyError(f"Generator task {state.spec.task_id} was already final")
    state.s += grade["passed"]
    state.n += 1


def append_rollout(state: TaskState, record: Mapping[str, Any], *, job: ScoringJob) -> None:
    """Journal one unique response; it may arrive before an earlier sample."""
    task_id = state.spec.task_id
    if record.get("version") != ROLLOUT_VERSION or record.get("task_id") != task_id:
        raise VerifyError(f"Generator rollout for {task_id} has the wrong schema or task ID")
    if type(record.get("j")) is not int or not 0 <= record["j"] < MAX_SAMPLES:
        raise VerifyError(f"Generator rollout for {task_id} has an invalid sample index")
    if record.get("hotkey") != state.spec.hotkey or record["j"] in state.responses:
        raise VerifyError(f"Generator rollout for {task_id} has the wrong owner or a duplicate index")
    if record.get("epoch") != job.epoch or record.get("head_version") != job.head_version:
        raise VerifyError(f"Generator rollout for {task_id} belongs to another scoring job")
    if record.get("seed") != rollout_seed(job.seed_secret, task_id, record["j"]):
        raise VerifyError(f"Generator rollout for {task_id} has the wrong seed")
    if not isinstance(record.get("messages"), list) or not isinstance(record.get("response"), str) or type(record.get("capped")) is not bool:
        raise VerifyError(f"Generator rollout for {task_id} lacks prompt, response or capped flag")
    if record.get("grade") != {}:
        raise VerifyError("Generator response journals must be ungraded; grades live in committed rollout files")
    state.responses[record["j"]] = json.loads(canonical_json(record))


def load_rollouts(job: ScoringJob, states: Mapping[str, TaskState], out: Path) -> int:
    """Recover committed grades first, then ungraded out-of-order responses."""
    count = 0
    for task in job.tasks:
        state = states[task.task_id]
        path = out / ROLLOUTS_DIR / f"{task.task_id}.jsonl"
        try:
            if path.exists():
                for raw in path.read_bytes().splitlines():
                    record = json.loads(raw)
                    if not isinstance(record, dict) or record.get("j") != state.n:
                        raise VerifyError(f"Generator committed rollout {task.task_id} is not an ordered prefix")
                    if record.get("seed") != rollout_seed(job.seed_secret, task.task_id, record["j"]):
                        raise VerifyError(f"Generator committed rollout {task.task_id} has the wrong seed")
                    if record.get("epoch") != job.epoch or record.get("head_version") != job.head_version or record.get("hotkey") != task.hotkey:
                        raise VerifyError(f"Generator committed rollout {task.task_id} belongs to another job")
                    grade = record.get("grade")
                    if not isinstance(grade, dict) or type(grade.get("passed")) is not bool:
                        raise VerifyError(f"Generator committed rollout {task.task_id} lacks a grade")
                    state.rollouts.append(record)
                    state.s += grade["passed"]
                    state.n += 1
                    if state.n % SAMPLE_INCREMENT == 0 or state.n == MAX_SAMPLES:
                        settle(state)
            responses_path = out / ROLLOUTS_DIR / "responses" / f"{task.task_id}.jsonl"
            if responses_path.exists():
                for raw in responses_path.read_bytes().splitlines():
                    record = json.loads(raw)
                    if not isinstance(record, dict):
                        raise VerifyError(f"Generator response {task.task_id} is not an object")
                    if type(record.get("j")) is int and (record["j"] < state.n or record["j"] in state.responses):
                        continue  # first durable response wins, including out-of-order duplicates
                    append_rollout(state, record, job=job)
                    count += 1
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            if isinstance(exc, VerifyError):
                raise
            raise VerifyError(f"cannot resume Generator rollouts for {task.task_id}: {exc}") from exc
    return count


def task_result(job: ScoringJob, state: TaskState) -> dict[str, Any]:
    if state.decision not in (ACCEPT, REJECT, UNSCORED):
        raise VerifyError(f"Generator task {state.spec.task_id} has an invalid final decision")
    if state.n < MIN_SAMPLES and state.decision != UNSCORED:
        raise VerifyError("Generator tasks below the minimum cannot be accepted or rejected")
    if state.n != len(state.rollouts) or not state.final:
        raise VerifyError(f"Generator task {state.spec.task_id} is unfinished")
    return {
        "id": state.spec.task_id,
        "hotkey": state.spec.hotkey,
        "s": state.s,
        "n": state.n,
        "q": state.q if state.decision != UNSCORED else None,
        "decision": state.decision,
        "unscored_reason": "time_budget" if state.decision == UNSCORED and "completed rollouts; 8 required" in state.reason else None,
        "reason": state.reason,
        "rollouts": rollout_object(job.epoch, f"{state.spec.task_id}.jsonl"),
    }


def hotkey_results(job: ScoringJob, states: Mapping[str, TaskState]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for task in job.tasks:
        counts.setdefault(task.hotkey, dict.fromkeys(("submitted", "eligible", "scored", "accepted", "rejected", "unscored", "rollouts", "passes"), 0))
    for state in states.values():
        row = counts[state.spec.hotkey]
        state.counts()
        row["submitted"] += 1
        row["eligible"] += 1
        row["scored"] += state.n >= MIN_SAMPLES
        row[{ACCEPT: "accepted", REJECT: "rejected"}.get(state.decision, UNSCORED)] += 1
        row["rollouts"] += state.n
        row["passes"] += state.s
    return counts


def make_results(
    job: ScoringJob,
    states: Mapping[str, TaskState],
    *,
    timings: Mapping[str, Any],
    throughput: Mapping[str, Any],
    servers: Sequence[str],
) -> dict[str, Any]:
    return {
        "version": RESULTS_VERSION,
        "epoch": job.epoch,
        "head_version": job.head_version,
        "pool": job.pool,
        "pool_pin": job.pool_pin,
        "seed_secret": job.seed_secret,
        "seed_commitment": hashlib.sha256(job.seed_secret.encode("utf-8")).hexdigest(),
        "budget_deadline": job.budget_deadline,
        "sampling": {
            "model": NULL_MODEL,
            "n": 1,
            "temperature": SAMPLING.temperature,
            "top_p": SAMPLING.top_p,
            "top_k": SAMPLING.top_k,
            "max_tokens": SAMPLING.max_tokens,
            "thinking": SAMPLING.thinking,
            "seed": "int(sha256(seed_secret:task_id:j).hexdigest()[:12],16)",
            "skip_special_tokens": False,
        },
        "servers": list(servers),
        "timings": dict(timings),
        "throughput": dict(throughput),
        "hotkeys": hotkey_results(job, states),
        "tasks": [task_result(job, states[task.task_id]) for task in job.tasks],
        "publication": {
            "results": epoch_results(job.epoch),
            "note": "Complete pod-local record. Validator publishes accepted rollouts only; rejected submissions remain hashes.",
        },
    }
