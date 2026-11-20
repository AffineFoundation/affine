"""Durable rollout journal and concurrent vLLM batch execution."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..canonical import canonical_json
from ..errors import InfraError, VerifyError
from ..gpu.evaluator import HttpError, Params, chat_body, read_reply
from .constants import SAMPLING
from .http import BudgetDeadline, LocalHttp, loopback_url
from .scoring import ROLLOUT_VERSION, ROLLOUTS_DIR, ScoringJob, TaskState, append_rollout, finish_grade, next_batch, rollout_seed, settle


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    """Atomically replace a private file, syncing its bytes and parent entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def append_durable(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(canonical_json(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def trim_torn_line(path: Path) -> None:
    """Discard only an unterminated tail; complete prior lines are untouched."""
    if not path.is_file():
        return
    with path.open("rb+") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        if size == 0:
            return
        stream.seek(size - 1)
        if stream.read(1) == b"\n":
            return
        position = size
        while position:
            start = max(0, position - 1024 * 1024)
            stream.seek(start)
            chunk = stream.read(position - start)
            index = chunk.rfind(b"\n")
            if index >= 0:
                position = start + index + 1
                break
            position = start
        stream.truncate(position)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def repair_checkpoints(job: ScoringJob, out: Path) -> None:
    for task in job.tasks:
        trim_torn_line(out / ROLLOUTS_DIR / f"{task.task_id}.jsonl")
        trim_torn_line(out / ROLLOUTS_DIR / "responses" / f"{task.task_id}.jsonl")


class TaskProvider(Protocol):
    def tasks(self, task_ids: Sequence[str]) -> Sequence[Any]: ...


class GradeProvider(Protocol):
    def grade_many(self, jobs: Sequence[tuple[Any, str, bool]]) -> Sequence[Any]: ...


def request_body(task: Any, seed: int) -> dict[str, Any]:
    """Production chat template, thinking and special-token handling, n=1."""
    params = Params(
        n_gpus=8,
        max_output_tokens=SAMPLING.max_tokens,
        max_context_tokens=69632,
        temperature=SAMPLING.temperature,
        top_p=SAMPLING.top_p,
        top_k=SAMPLING.top_k,
        dtype="bfloat16",
    )
    return chat_body("null", task.messages(), seed, params) | {"n": 1}


@dataclass(slots=True)
class Usage:
    completed: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    generate_seconds: float = 0.0
    grade_seconds: float = 0.0
    requests: int = 0

    def add(self, seconds: float, completion_tokens: int, prompt_tokens: int) -> None:
        self.completed += 1
        self.completion_tokens += completion_tokens
        self.prompt_tokens += prompt_tokens
        self.generate_seconds += seconds


class SlotUnavailable(Exception):
    def __init__(self, slot: int):
        super().__init__(f"slot {slot} temporarily unavailable")
        self.slot = slot


class BatchRunner:
    """Assign work to the least-loaded server; grade only durable responses."""

    def __init__(
        self,
        job: ScoringJob,
        states: dict[str, TaskState],
        out: Path,
        pool: TaskProvider,
        grader: GradeProvider,
        *,
        urls: Sequence[str],
        http: LocalHttp,
        clients_per_server: int = 32,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.job, self.states, self.out = job, states, out
        self.pool, self.grader = pool, grader
        self.urls = tuple(loopback_url(url) for url in urls)
        if len(self.urls) != 8:
            raise InfraError("Generator scoring needs exactly eight local vLLM endpoints")
        if type(clients_per_server) is not int or not 1 <= clients_per_server <= 256:
            raise InfraError("Generator clients_per_server must be 1..256")
        self.http, self.clock, self.monotonic = http, clock, monotonic
        self.clients_per_server = clients_per_server
        self.usage = Usage()
        self._tasks: dict[str, Any] = {}
        self._cutoff = False
        self._disk_lock = threading.Lock()
        self._futures: dict[Future[tuple[str, int, dict[str, Any]]], tuple[str, int, int]] = {}
        self._assigned: dict[tuple[str, int], int] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._first_error: BaseException | None = None
        self._slot_cooldown = [0.0] * len(self.urls)
        self._next_slot = 0
        self._slot_errors = [0] * len(self.urls)
        self._last_progress = 0.0
        self._stage = "stage1"

    @property
    def cutoff(self) -> bool:
        return self._cutoff

    def task(self, task_id: str) -> Any:
        if task_id not in self._tasks:
            self._tasks[task_id] = self.pool.tasks((task_id,))[0]
        return self._tasks[task_id]

    def _choose_slot(self) -> int:
        if self._cutoff or self.clock() >= self.job.budget_deadline:
            raise BudgetDeadline()
        now = self.monotonic()
        for _ in range(len(self.urls)):
            slot = self._next_slot
            self._next_slot = (slot + 1) % len(self.urls)
            if self._slot_cooldown[slot] <= now:
                return slot
        return None

    def _request(self, item: tuple[str, int], slot: int) -> tuple[str, int, dict[str, Any]]:
        task_id, sample = item
        seed = rollout_seed(self.job.seed_secret, task_id, sample)
        sent = self.monotonic()
        attempts = 0
        while attempts < 3:
            if self.clock() >= self.job.budget_deadline or self._cutoff:
                raise BudgetDeadline()
            timeout = max(0.001, self.job.budget_deadline - self.clock())
            try:
                self.usage.requests += 1
                reply = self.http(self.urls[slot] + "/v1/chat/completions", request_body(self.task(task_id), seed), timeout)
                text, capped = read_reply(reply)
                if len(reply.get("choices", [])) != 1:
                    raise ValueError("expected exactly one choice for n=1")
                usage = reply.get("usage")
                completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, Mapping) else 0
                prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, Mapping) else 0
                if type(completion_tokens) is not int or type(prompt_tokens) is not int:
                    raise ValueError("invalid token usage")
                seconds = self.monotonic() - sent
                self.usage.add(seconds, completion_tokens, prompt_tokens)
                return task_id, sample, {
                    "text": text,
                    "capped": capped,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "seconds": seconds,
                    "server": self.urls[slot],
                }
            except HttpError as exc:
                if 400 <= exc.status < 500:
                    raise InfraError(f"Generator vLLM rejected sampling parameters with HTTP {exc.status}") from exc
                self._mark_sick(slot, f"HTTP {exc.status}")
                raise SlotUnavailable(slot) from exc
            except (TimeoutError, OSError) as exc:
                if self.clock() >= self.job.budget_deadline or self._cutoff:
                    raise BudgetDeadline()
                self._mark_sick(slot, f"{type(exc).__name__} errno={exc.errno if isinstance(exc, OSError) else None}")
                raise SlotUnavailable(slot) from exc
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise InfraError(f"Generator vLLM returned an invalid chat response: {type(exc).__name__}: {exc}") from exc
        raise InfraError(f"Generator local server at {self.urls[slot]} failed three requests")

    def _mark_sick(self, slot: int, detail: str) -> None:
        """A transport fault affects one slot, never all in-flight requests."""
        self._slot_errors[slot] += 1
        delay = min(60.0, 0.25 * (2 ** min(self._slot_errors[slot] - 1, 8)))
        self._slot_cooldown[slot] = self.monotonic() + delay
        print(f"generator scorer slot={slot} transient {detail} backoff_s={delay:g}", file=sys.stderr, flush=True)

    def _checkpoint(self, task_id: str, sample: int, reply: Mapping[str, Any]) -> dict[str, Any]:
        state = self.states[task_id]
        record = {
            "version": ROLLOUT_VERSION,
            "epoch": self.job.epoch,
            "task_id": task_id,
            "hotkey": state.spec.hotkey,
            "j": sample,
            "seed": rollout_seed(self.job.seed_secret, task_id, sample),
            "head_version": self.job.head_version,
            "messages": self.task(task_id).messages(),
            "response": reply["text"],
            "finish_reason": "length" if reply["capped"] else "stop",
            "capped": reply["capped"],
            "server": reply["server"],
            "grade": {},
            "completed_at": self.clock(),
            "seconds": reply["seconds"],
            "prompt_tokens": reply["prompt_tokens"],
            "completion_tokens": reply["completion_tokens"],
        }
        with self._disk_lock:
            if sample < self.states[task_id].n or sample in self.states[task_id].responses:
                return self.states[task_id].responses.get(sample, {})
            append_durable(self.out / ROLLOUTS_DIR / "responses" / f"{task_id}.jsonl", record)
            append_rollout(state, record, job=self.job)
        return record

    @staticmethod
    def _grade_record(record: dict[str, Any], grade: Any) -> None:
        record["grade"] = {
            "code": grade.code,
            "passed": grade.reward == 1.0 and grade.code == "P",
            "answer": grade.answer,
            "rule": grade.rule,
            "machine_cause": grade.machine_cause,
        }
        record["passed"] = record["grade"]["passed"]

    def _commit_grade(self, record: dict[str, Any]) -> None:
        state = self.states[record["task_id"]]
        with self._disk_lock:
            state.rollouts.append(record)
            # Counts change only after this graded prefix is durable. A crash
            # replays from state.n, so no generated sample is lost or repeated.
            raw = b"".join(canonical_json(row) + b"\n" for row in state.rollouts)
            atomic_write(self.out / ROLLOUTS_DIR / f"{record['task_id']}.jsonl", raw)
            finish_grade(state, record)

    def _abort(self, error: BaseException) -> None:
        if self._first_error is None:
            self._first_error = error
        self._cutoff = True
        self.http.cancel()
        for future in self._futures:
            future.cancel()

    def _grade_responses(self, work: Sequence[tuple[str, int]]) -> int:
        responses = []
        for task_id in dict.fromkeys(task_id for task_id, _ in work):
            state = self.states[task_id]
            next_sample = state.n
            while not state.terminal and next_sample in state.responses:
                responses.append((task_id, state.responses.pop(next_sample)))
                next_sample += 1
        if not responses:
            return 0
        started = self.monotonic()
        grades = self.grader.grade_many([
            (self.task(task_id), record["response"], record["capped"])
            for task_id, record in responses
        ])
        if len(grades) != len(responses):
            raise InfraError("Generator grader returned the wrong number of grades")
        for (_, raw_record), grade in zip(responses, grades):
            record = dict(raw_record)
            self._grade_record(record, grade)
            self._commit_grade(record)
        self.usage.grade_seconds += self.monotonic() - started
        return len(responses)

    def run(self, pending: Sequence[tuple[str, int]], batch: Sequence[tuple[str, int]]) -> int:
        """Keep inference slots full; checkpoint and grade without a batch barrier."""
        self.http.reset()
        self._cutoff = False
        self._first_error = None
        self._futures = {}
        self._assigned = {}
        self._tasks = {}
        preload = list(dict.fromkeys([task_id for task_id, _ in pending] + [task_id for task_id, _ in batch]))
        if preload:
            self._tasks.update(zip(preload, self.pool.tasks(tuple(preload))))
        graded = self._grade_responses(pending) if pending else 0
        preload = list(dict.fromkeys([task_id for task_id, _ in pending] + [task_id for task_id, _ in batch]))
        if preload:
            self._tasks.update(zip(preload, self.pool.tasks(tuple(preload))))
        queue = deque(batch)
        self._assigned.clear()
        stage = self._stage
        capacity = len(self.urls) * self.clients_per_server
        if not queue:
            for state in self.states.values():
                if not state.final:
                    settle(state)
            refill, stage = next_batch(self.job, self.states, max(capacity, 32 * len(self.job.tasks)))
            queue.extend(refill)
            self._stage = stage
        self._executor = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix="generator-scorer")

        def submit(item):
            slot = self._choose_slot()
            if slot is None:
                return False
            self._assigned[item] = slot
            future = self._executor.submit(self._request, item, slot)
            self._futures[future] = (item[0], item[1], slot)
            return True

        try:
            while self._futures or (queue and self._first_error is None):
                if self._first_error is not None and not self._futures:
                    break
                while self._first_error is None and len(self._futures) < capacity and self.clock() < self.job.budget_deadline and not self._cutoff:
                    if not queue:
                        break
                    item = queue.popleft()
                    if sum(task == item[0] for task, _, _ in self._futures.values()) >= 8:
                        continue
                    if item[1] < self.states[item[0]].n or item[1] in self.states[item[0]].responses:
                        continue
                    if any(task == item[0] and sample == item[1] for task, sample, _ in self._futures.values()):
                        continue
                    if not submit(item):
                        queue.appendleft(item)
                        self._assigned.pop(item, None)
                        break
                if not self._futures:
                    if queue and self.clock() < self.job.budget_deadline:
                        time.sleep(0.01)
                        continue
                    break
                done, _ = wait(tuple(self._futures), timeout=0.1, return_when=FIRST_COMPLETED)
                if not done:
                    if self.clock() >= self.job.budget_deadline:
                        self._cutoff = True
                        self.http.cancel()
                    continue
                for future in done:
                    task_id, sample, slot = self._futures.pop(future)
                    self._assigned.pop((task_id, sample), None)
                    try:
                        returned_task, returned_sample, reply = future.result()
                        self._checkpoint(returned_task, returned_sample, reply)
                        graded += self._grade_responses([(returned_task, returned_sample)])
                        settle(self.states[returned_task])
                    except SlotUnavailable:
                        self._assigned.pop((task_id, sample), None)
                        queue.appendleft((task_id, sample))
                    except BudgetDeadline:
                        self._cutoff = True
                    except BaseException as exc:
                        self._abort(exc)
                        # Continue draining other completed futures into durable journals.
                        self._cutoff = True
            graded += self._grade_responses([(task.task_id, self.states[task.task_id].n) for task in self.job.tasks])
            if self._first_error is not None:
                raise InfraError(f"Generator inference failed; rerun resumes completed rollouts: {self._first_error}") from self._first_error
            return graded
        finally:
            self._cutoff = True
            self.http.cancel()
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            self._futures.clear()
            self._assigned.clear()
