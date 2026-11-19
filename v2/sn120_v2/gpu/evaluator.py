"""Paired null/LoRA scoring on one vLLM server per GPU.

Task index determines the server for every model. ``evaluate`` retains the
block-at-a-time seam used by the CPU simulation; ``stream`` drives the same
servers without a block barrier and reports graded rollouts in draw order.
"""

from __future__ import annotations

import json
import queue
import resource
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from ..errors import ContractError, InfraError
from ..params import Params
from ..sampling import rollout_seed
from ..seams import ObjectStore
from ..types import CAP, ERROR, FAIL, NULL_MODEL, PASS, EvalChunk, EvalRequest, HeadVersion
from .batch import END_OF_TURN
from .plan import BoxConfig, check_pins, log_phase
from .procs import Child, Process, Spawn, gpu_env, nvidia_smi, stop_all, wait_gpus_free
from .resident import Resident, resident_head


class TaskPool(Protocol):
    def task(self, task_id: str) -> Any: ...


class Grader(Protocol):
    def grade_many(self, jobs: Sequence[tuple[Any, str, bool]]) -> list[Any]: ...


class HttpError(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


class _Cancelled(Exception):
    pass


class _Deadline(Exception):
    pass


def http_json(url: str, body: dict | None, timeout_s: float) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as reply:  # noqa: S310 - a local http server
            raw = reply.read()
    except urllib.error.HTTPError as exc:
        raise HttpError(exc.code) from exc
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class CancellableHttp:
    """JSON HTTP whose in-flight model requests can be disconnected.

    vLLM aborts a generation when its client disconnects. Cancellation is
    per model: a dropped LoRA must not disconnect null or another LoRA.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()
        self._active: dict[str, set[Any]] = {}

    def reset(self) -> None:
        with self._lock:
            if any(self._active.values()):
                raise ContractError("an earlier evaluation still has requests in flight")
            self._cancelled.clear()

    def cancel(self, model: str) -> None:
        with self._lock:
            self._cancelled.add(model)
            replies = tuple(self._active.get(model, ()))
        for reply in replies:
            reply.close()

    def __call__(self, url: str, body: dict | None, timeout_s: float) -> dict:
        model = body.get("model", "") if body and url.endswith("/v1/chat/completions") else ""
        with self._lock:
            if model in self._cancelled:
                raise _Cancelled(model)
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        reply = None
        try:
            reply = urllib.request.urlopen(request, timeout=timeout_s)  # noqa: S310 - a local http server
            with self._lock:
                cancelled = model in self._cancelled
                if model and not cancelled:
                    self._active.setdefault(model, set()).add(reply)
            if cancelled:
                raise _Cancelled(model)
            try:
                raw = reply.read()
            finally:
                with self._lock:
                    if model:
                        self._active.get(model, set()).discard(reply)
            try:
                parsed = json.loads(raw)
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        except urllib.error.HTTPError as exc:
            raise HttpError(exc.code) from exc
        finally:
            if reply is not None:
                reply.close()


RolloutResult = tuple[str, int, int, tuple[str, bool] | None]


def socket_room(sockets: int, spare: int) -> int:
    """Raise the soft open-file limit and return room for rollout sockets."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return sockets
    wanted = sockets + spare
    reachable = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
    if reachable > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (reachable, hard))
            soft = reachable
        except (OSError, ValueError):
            pass
    return max(0, soft - spare)


@dataclass(frozen=True)
class ServeSettings:
    """Local serving knobs. None of them changes a score."""

    first_port: int = 8100
    start_timeout_s: float = 1800.0
    clients_per_server: int = 512
    admin_sockets_per_server: int = 8
    retries: int = 2
    spare_files: int = 512
    free_limit_mib: int = 1024
    free_timeout_s: float = 120.0
    grade_tries: int = 3


def lora_table(params: Params, batches: int = 0) -> int:
    """The ticket fixes --max-loras for the whole round, on every box."""
    if not 0 <= batches <= params.max_loras_per_box:
        raise ContractError("the batches of a look must be inside max_loras_per_box")
    return batches or params.max_loras_per_box


def server_argv(base_dir: str, port: int, params: Params, max_seqs: int | None = None, max_loras: int | None = None) -> list[str]:
    rank = params.recipe.bound("rank")
    max_rank = int(rank.high) if rank else params.recipe.rank
    max_seqs = params.max_num_seqs if max_seqs is None else max_seqs
    return [
        *(sys.executable, "-m", "vllm.entrypoints.openai.api_server"),
        *("--model", base_dir, "--served-model-name", NULL_MODEL),
        *("--host", "127.0.0.1", "--port", str(port)),
        *("--dtype", params.dtype, "--kv-cache-dtype", params.kv_cache_dtype, "--max-model-len", str(params.max_context_tokens)),
        *("--enable-prefix-caching", "--enable-lora", "--max-num-seqs", str(max_seqs)),
        *("--max-loras", str(max_loras or lora_table(params)), "--max-lora-rank", str(max_rank)),
        *("--limit-mm-per-prompt", '{"image":0,"video":0}'),
    ]


def server_env(gpu: int, work_dir: Path) -> dict[str, str]:
    cache = work_dir / "vllm-cache" / f"gpu{gpu}"
    names = {"VLLM_CACHE_ROOT": "vllm", "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor"}
    for name in names.values():
        (cache / name).mkdir(parents=True, exist_ok=True)
    env = gpu_env([gpu])
    env.update({key: str(cache / name) for key, name in names.items()})
    env.update(HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    return env


WORKER_DIED = "worker-died"


def box_fault(grade: Any) -> bool:
    return getattr(grade, "machine_cause", None) == WORKER_DIED


def chat_body(model: str, messages: list[dict[str, str]], seed: int, params: Params) -> dict:
    return {
        "model": model,
        "messages": messages,
        "seed": seed,
        "max_tokens": params.max_output_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def read_reply(reply: dict) -> tuple[str, bool]:
    choice = reply["choices"][0]
    text, reason = choice["message"]["content"], choice["finish_reason"]
    if not isinstance(text, str) or reason not in ("stop", "length"):
        raise ValueError(f"unexpected reply: finish_reason {reason!r}")
    if reason == "stop" and text.endswith(END_OF_TURN):
        text = text[: -len(END_OF_TURN)]
    return text, reason == "length"


def written_facts(written: Sequence[tuple[int, float]]) -> dict[str, Any]:
    longest = max(written, key=lambda pair: pair[1], default=(0, 0.0))
    tokens = [count for count, _ in written]
    return {"output_tokens": sum(tokens), "longest_rollout_s": longest[1], "longest_rollout_tokens": longest[0], "most_output_tokens": max(tokens, default=0)}


class VllmEvaluator:
    def __init__(
        self,
        store: ObjectStore,
        config: BoxConfig,
        pool: TaskPool,
        grader: Grader,
        *,
        spawn: Spawn = Child,
        http: Http | None = None,
        gpus_free: Callable[[int, float], None] = wait_gpus_free,
        settings: ServeSettings = ServeSettings(),
        sleep: Callable[[float], None] = time.sleep,
        room: Callable[[int, int], int] = socket_room,
    ):
        self.store, self.config, self.pool, self.grader = store, config, pool, grader
        self.spawn, self.gpus_free, self.settings, self.sleep, self.room = spawn, gpus_free, settings, sleep, room
        self.http = CancellableHttp() if http is None else http
        self.resident = Resident(store, config.work_dir)
        self._servers: list[Process] = []
        self._urls: list[str] = []
        self._head: HeadVersion | None = None
        self._table = 0
        self._loaded: set[str] = set()
        self._stream_lock = threading.Lock()
        self._solo_servers: dict[int, Process] = {}
        self._solo_server_clients: dict[int, int] = {}
        self._solo_urls: dict[int, str] = {}
        self._solo_loaded: dict[int, set[str]] = {}
        self._solo_start_locks: dict[int, threading.Lock] = {}
        self._solo_lock = threading.RLock()
        self._solo_clients: int | None = None
        self._grader_lock = threading.Lock()
        self._grader_failure = False

    def _reset_grader(self) -> None:
        """Discard every worker after a failure; the next round starts clean."""
        reset = getattr(self.grader, "reset", None)
        if reset is not None:
            reset()
        else:
            closer = getattr(self.grader, "close", None)
            if closer is not None:
                closer()
        self._grader_failure = False

    def solo_client_budget(self, n_gpus: int) -> int:
        """Raise NOFILE before solo starts, then share the reachable socket room.

        The reserve covers grader pipes, log/store files, and the process's
        existing descriptors. Each server also has an admin allowance for
        health and LoRA loads that must not compete with rollout sockets.
        """
        if n_gpus < 1:
            raise ContractError("solo evaluation needs at least one GPU")
        admin = n_gpus * self.settings.admin_sockets_per_server
        room = self.room(n_gpus * self.settings.clients_per_server + admin, admin + self.settings.spare_files)
        if room < admin + n_gpus:
            raise InfraError(f"the open file limit leaves {room} descriptors for solo evaluation; {admin + n_gpus} are needed for one client and admin calls per GPU")
        return max(1, min(self.settings.clients_per_server, (room - admin) // n_gpus))

    def _grade_batch(self, jobs: Sequence[tuple[Any, str, bool]]) -> list[Any]:
        """One pool caller at a time, even with independent solo GPU workers."""
        with self._grader_lock:
            if self._grader_failure:
                self._reset_grader()
            try:
                grades = self.grader.grade_many(jobs)
            except BaseException:
                self._grader_failure = True
                raise
            if len(grades) != len(jobs):
                self._grader_failure = True
                raise InfraError("the grader returned a wrong number of grades")
            return grades

    def _prepare(self, request: EvalRequest) -> tuple[list[str], int, int]:
        params = request.params
        check_pins(self.config.image_digest, params)
        if any(lora.head_version_id != request.head.version_id for lora in request.loras):
            raise ContractError("a LoRA of this look was trained on another head")
        models = [NULL_MODEL, *(lora.lora_hash for lora in request.loras)]
        table = lora_table(params, request.batches)
        if len(set(models)) != len(models) or len(request.loras) > table:
            raise ContractError("the LoRAs of a look must be distinct and inside the LoRA table of its round")
        clients = params.n_gpus * self.settings.clients_per_server
        room = self.room(clients, self.settings.spare_files)
        if room < clients:
            raise InfraError(f"the open file limit of this process has room for {room} sockets; a look holds {clients}. Raise the limit (ulimit -n) and start run again")
        self._serve(request.head, params, table)
        fresh = [(lora.lora_hash, str(self.resident.tree(lora.prefix, lora.lora_hash))) for lora in request.loras if lora.lora_hash not in self._loaded]

        def load(url: str) -> None:
            cards = self._call(url + "/v1/models", None, 60.0).get("data")
            held = {card.get("id") for card in cards if isinstance(card, dict)} if isinstance(cards, list) else set()
            for name, path in fresh:
                if name not in held:
                    self._call(url + "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}, 600.0)

        with ThreadPoolExecutor(len(self._urls)) as threads:
            list(threads.map(load, self._urls))
        self._loaded.update(name for name, _ in fresh)
        return models, table, clients

    def _rollout(self, request: EvalRequest, url: str, job: tuple[str, int, int], written: list[tuple[int, float]]) -> tuple[str, bool] | None:
        model, sample, index = job
        task = request.task_ids[index]
        body = chat_body(model, self.pool.task(task).messages(), rollout_seed(request.round_seed, task, sample), request.params)
        for _ in range(self.settings.retries + 1):
            budget = float(request.params.eval_time_budget_s)
            if request.deadline:
                budget = min(budget, request.deadline - time.time())
                if budget <= 0:
                    raise _Deadline()
            try:
                sent = time.monotonic()
                reply = self.http(url + "/v1/chat/completions", body, budget)
                result = read_reply(reply)
                usage = reply.get("usage")
                written.append((int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0, time.monotonic() - sent))
                return result
            except _Deadline:
                raise
            except HttpError as exc:
                if exc.status < 500:
                    return None
            except _Cancelled:
                return None
            except (OSError, KeyError, IndexError, TypeError, ValueError):
                pass
        return None

    def _cut(self, request: EvalRequest, started: float, served: float, models: list[str], table: int, written: list[tuple[int, float]], jobs: int, finished: int) -> InfraError:
        cut = {"serve_s": served - started, "generate_s": time.monotonic() - served, "wall_s": time.monotonic() - started}
        cut.update(models=len(models), tasks=len(request.task_ids), rollouts=jobs, finished=finished, reply_cap=request.params.max_output_tokens, max_loras=table, open_files_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0])
        try:
            log_phase(self.config.work_dir, "eval_cut", round_id=request.round_id, look=request.look, n_gpus=len(self._urls), **cut, **written_facts(written))
        except Exception:  # noqa: BLE001 - an operator log cannot change the outcome
            pass
        return InfraError("the round's time limit passed during this chunk; an unfinished chunk is not scored")

    def evaluate(self, request: EvalRequest) -> EvalChunk:
        """The original block seam: generation, then one grading call."""
        started, written = time.monotonic(), []
        models, table, clients = self._prepare(request)
        params, tasks = request.params, request.task_ids
        served = time.monotonic()
        jobs = [(model, sample, index) for model in models for sample in range(params.samples_per_task) for index in range(len(tasks))]
        deadline_hit = False

        def rollout(job: tuple[str, int, int]) -> tuple[str, bool] | None:
            nonlocal deadline_hit
            try:
                return self._rollout(request, self._urls[job[2] % len(self._urls)], job, written)
            except _Deadline:
                deadline_hit = True
                return None

        with ThreadPoolExecutor(clients) as threads:
            replies = list(threads.map(rollout, jobs))
        if not all(server.alive() for server in self._servers):
            self.close()
            raise InfraError("a vLLM server died during the look; nothing of it can be scored")
        if deadline_hit:
            raise self._cut(request, started, served, models, table, written, len(jobs), sum(reply is not None for reply in replies))
        generated = time.monotonic()
        codes = self._grade(tasks, jobs, replies)
        outcomes = {model: tuple("".join(codes[model, sample, index] for index in range(len(tasks))) for sample in range(params.samples_per_task)) for model in models}
        times = {"serve_s": served - started, "generate_s": generated - served, "grade_s": time.monotonic() - generated, "wall_s": time.monotonic() - started}
        counts = {"models": len(models), "tasks": len(tasks), "rollouts": len(jobs), "capped": sum(code == CAP for code in codes.values()), "n_gpus": len(self._urls), "max_loras": table, "open_files_limit": resource.getrlimit(resource.RLIMIT_NOFILE)[0]}
        log_phase(self.config.work_dir, "eval", round_id=request.round_id, look=request.look, **times, **counts, **written_facts(written))
        return EvalChunk(look=request.look, task_ids=tasks, outcomes=outcomes, image_digest=self.config.image_digest)

    def solo_worker(self, ticket, trained, deadline, gpu):
        from .solo import SoloWorker

        return SoloWorker(self, ticket, trained, deadline, gpu)

    def _load_one(self, gpu: int, lora) -> None:
        """Load one adapter on one solo server, once per server lifetime."""
        if lora.lora_hash in self._solo_loaded.get(gpu, set()):
            return
        with self._solo_lock:
            if lora.lora_hash in self._solo_loaded.setdefault(gpu, set()):
                return
            url = self._solo_urls[gpu]
            path = str(self.resident.tree(lora.prefix, lora.lora_hash))
            self._call(url + "/v1/load_lora_adapter", {"lora_name": lora.lora_hash, "lora_path": path}, 600.0)
            self._solo_loaded[gpu].add(lora.lora_hash)

    def stream(
        self,
        request: EvalRequest,
        results: Callable[[str, int, int, str], None],
        eliminated: Callable[[str], bool] | None = None,
        done: Callable[[], bool] | None = None,
    ) -> int:
        """Stream every task, grade in order, and return completed blocks.

        There is one continuously fed queue per server, not a limit on queued
        work. A result callback runs only after every earlier result of that
        model has been graded. The return is the common completed-block
        prefix before the deadline, or before all live models finished.
        """
        with self._stream_lock:
            return self._stream(request, results, eliminated or (lambda model: False), done or (lambda: False))

    def _stream(self, request: EvalRequest, results: Callable[[str, int, int, str], None], eliminated: Callable[[str], bool], done: Callable[[], bool]) -> int:
        started, written = time.monotonic(), []
        models, table, clients = self._prepare(request)
        if hasattr(self.http, "reset"):
            self.http.reset()  # type: ignore[attr-defined]
        params, tasks = request.params, request.task_ids
        samples = params.samples_per_task
        queues = [queue.Queue() for _ in self._urls]
        replies: queue.Queue[tuple[str, object]] = queue.Queue()
        stop = threading.Event()
        cut = threading.Event()
        dropped = {model for model in models if eliminated(model)}
        jobs = [(model, sample, index) for model in models if model not in dropped for sample in range(samples) for index in range(len(tasks))]
        unfinished = set(jobs)
        for gpu, target in enumerate(queues):
            for job in jobs:
                if job[2] % len(queues) == gpu:
                    target.put(job)

        def cancel(model: str) -> None:
            if model not in dropped:
                dropped.add(model)
                if hasattr(self.http, "cancel"):
                    self.http.cancel(model)  # type: ignore[attr-defined]

        for model in tuple(dropped):
            if hasattr(self.http, "cancel"):
                self.http.cancel(model)  # type: ignore[attr-defined]

        def worker(gpu: int) -> None:
            try:
                while not stop.is_set():
                    try:
                        job = queues[gpu].get_nowait()
                    except queue.Empty:
                        return
                    model, sample, index = job
                    if eliminated(model) or (done() and model == NULL_MODEL):
                        cancel(model)
                        unfinished.discard(job)
                        replies.put(("reply", (gpu, (model, sample, index, None))))
                        continue
                    try:
                        reply = self._rollout(request, self._urls[gpu], job, written)
                        replies.put(("reply", (gpu, (model, sample, index, reply))))
                    except _Deadline:
                        unfinished.discard(job)
                        replies.put(("reply", (gpu, (model, sample, index, None))))
                        stop.set()
                        cut.set()
                        return
                    except Exception as exc:  # noqa: BLE001 - a request thread may never silently lose a job
                        stop.set()
                        replies.put(("reply", (gpu, (model, sample, index, None))))
                        failures.append(exc)
                        return
                    if eliminated(model):
                        cancel(model)
            finally:
                replies.put(("exit", gpu))

        failures: list[Exception] = []
        pending = set(jobs)
        next_result = {(model, sample): 0 for model in models for sample in range(samples)}
        ordered: dict[tuple[str, int, int], tuple[str, bool] | None] = {}
        codes: dict[tuple[str, int, int], str] = {}
        cancelled: set[tuple[str, int, int]] = set()
        served = time.monotonic()
        # A server has its own clients_per_server threads. Its own queue is
        # shared by those threads; no task waits for an unrelated GPU's tail.
        threads = [
            threading.Thread(target=worker, args=(gpu,), name=f"eval-gpu-{gpu}-{client}", daemon=True)
            for gpu in range(len(queues))
            for client in range(self.settings.clients_per_server)
        ]
        for thread in threads:
            thread.start()
        completed_workers = 0
        common = 0
        # A cancelled job needs no grade. Each worker announces its own exit
        # through the reply queue, so cancellation never leaves a lost waiter.
        deadline_hit = False
        finished = 0
        while completed_workers < len(threads):
            if done():
                stop.set()
                cancel(NULL_MODEL)
            for model in models:
                if eliminated(model):
                    cancel(model)
            kind, item = replies.get()
            if kind == "exit":
                completed_workers += 1
                continue
            gpu, result = item
            model, sample, index, reply = result
            job = (model, sample, index)
            if model in dropped or eliminated(model) or (done() and model == NULL_MODEL):
                cancel(model)
                cancelled.add(job)
                pending.discard(job)
                unfinished.discard(job)
                continue
            pending.discard(job)
            unfinished.discard(job)
            if request.deadline and time.time() >= request.deadline and (unfinished or ordered):
                deadline_hit = True
                cut.set()
                stop.set()
            finished += reply is not None
            ordered[job] = reply
            key = (model, sample)
            while (model, sample, next_result[key]) in ordered:
                current = (model, sample, next_result[key])
                outcome = self._grade(tasks, [current], [ordered.pop(current)])[current]
                results(model, sample, current[2], outcome)
                if done():
                    stop.set()
                if eliminated(model):
                    cancel(model)
                    # The decision is made from the exact look prefix. Any
                    # later result for this model is now discarded.
                    for later in tuple(ordered):
                        if later[0] == model:
                            ordered.pop(later)
                            cancelled.add(later)
                    break
                codes[current] = outcome
                next_result[key] += 1
            common = self._common_prefix(codes, dropped, models, samples, len(tasks), params.eval_chunk_tasks)
            if common == len(tasks) or (not unfinished and not ordered):
                # Cancelled jobs no longer constrain the shared prefix.
                # Other live models continue to fill every server.
                stop.set()
        stop.set()
        for thread in threads:
            thread.join()
        if not all(server.alive() for server in self._servers):
            self.close()
            raise InfraError("a vLLM server died during the stream; nothing of it can be scored")
        if failures:
            raise InfraError(f"a request worker failed: {failures[0]}") from failures[0]
        stopped_deadline = cut.is_set() or bool(request.deadline and time.time() >= request.deadline and (unfinished or ordered or deadline_hit))
        if stopped_deadline:
            # Complete blocks remain valid; the unfinished tail is discarded.
            # Log the cutoff for the operator without changing those codes.
            self._cut(request, started, served, models, table, written, len(jobs), finished)
        for model in models:
            for sample in range(samples):
                if eliminated(model) or model in dropped:
                    continue
                for index in range(common):
                    if (model, sample, index) not in codes:
                        raise InfraError("the stream ended with a gap in a live model's completed prefix")
        counts = {"models": len(models), "tasks": len(tasks), "rollouts": len(jobs), "finished": finished, "completed_prefix": common, "cancelled": len(cancelled), "n_gpus": len(self._urls), "max_loras": table, "open_files_limit": resource.getrlimit(resource.RLIMIT_NOFILE)[0]}
        log_phase(self.config.work_dir, "eval_stream", round_id=request.round_id, wall_s=time.monotonic() - started, generate_s=time.monotonic() - served, serve_s=served - started, **counts, **written_facts(written))
        return common

    @staticmethod
    def _common_prefix(codes: dict[tuple[str, int, int], str], dropped: set[str], models: list[str], samples: int, tasks: int, block: int) -> int:
        live = [model for model in models if model not in dropped]
        prefix = 0
        for index in range(tasks):
            if all((model, sample, index) in codes for model in live for sample in range(samples)):
                prefix = index + 1
            else:
                break
        return prefix // block * block

    def _wait_one_gpu_free(self, gpu: int) -> None:
        """Wait for this GPU only, never for another GPU's active trainer."""
        def report() -> str:
            lines = [line for line in nvidia_smi().splitlines() if line.strip()]
            for line in lines:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == gpu:
                    return line
            return "not-readable, not-readable"  # fail closed when this GPU is absent

        wait_gpus_free(self.settings.free_limit_mib, self.settings.free_timeout_s, report=report, sleep=self.sleep)

    def _serve_one(self, head: HeadVersion, params: Params, table: int, gpu: int) -> str:
        """Start or reuse exactly one GPU's server, independent of its peers."""
        with self._solo_lock:
            start_lock = self._solo_start_locks.setdefault(gpu, threading.Lock())
        with start_lock:
            return self._start_one(head, params, table, gpu)

    def _start_one(self, head: HeadVersion, params: Params, table: int, gpu: int) -> str:
        """One GPU startup, without blocking startup or loading on its peers."""
        clients = self._solo_clients if self._solo_clients is not None else self.settings.clients_per_server
        with self._solo_lock:
            existing = self._solo_servers.get(gpu)
            if existing is not None and existing.alive() and self._head is not None and self._head.version_id == head.version_id and self._table == table and self._solo_server_clients.get(gpu) == clients:
                return self._solo_urls[gpu]
            self._solo_servers.pop(gpu, None)
            self._solo_server_clients.pop(gpu, None)
            self._solo_urls.pop(gpu, None)
            self._solo_loaded.pop(gpu, None)
        if existing is not None:
            existing.stop()
        base_dir = resident_head(self.resident, head)
        self._wait_one_gpu_free(gpu)
        logs = self.config.work_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        port = self.settings.first_port + gpu
        argv = server_argv(str(base_dir), port, params, clients, table)
        # Spawning and the health wait are slow: never hold the shared Solo
        # lock while another GPU can start, load an adapter, or claim work.
        server = self.spawn(argv, server_env(gpu, self.config.work_dir), logs / f"vllm-{gpu}.log")
        url = f"http://127.0.0.1:{port}"
        with self._solo_lock:
            # A concurrent caller for this GPU may have installed a server
            # while this one was spawning. Never let the stale process win.
            if gpu in self._solo_servers:
                server.stop()
                existing = self._solo_servers[gpu]
                if existing.alive() and self._head is not None and self._head.version_id == head.version_id and self._table == table and self._solo_server_clients.get(gpu) == clients:
                    return self._solo_urls[gpu]
                self._solo_servers.pop(gpu, None)
                self._solo_server_clients.pop(gpu, None)
                self._solo_urls.pop(gpu, None)
                self._solo_loaded.pop(gpu, None)
                existing.stop()
            self._head, self._table = head, table
            self._solo_servers[gpu] = server
            self._solo_server_clients[gpu] = clients
            self._solo_urls[gpu] = url
            self._solo_loaded[gpu] = set()
        waited = 0.0
        while True:
            try:
                self.http(url + "/health", None, 10.0)
                return url
            except (OSError, HttpError):
                if not server.alive() or waited >= self.settings.start_timeout_s:
                    server.stop()
                    with self._solo_lock:
                        if self._solo_servers.get(gpu) is server:
                            self._solo_servers.pop(gpu, None)
                            self._solo_server_clients.pop(gpu, None)
                            self._solo_urls.pop(gpu, None)
                            self._solo_loaded.pop(gpu, None)
                    raise InfraError(f"the vLLM server at {url} did not start") from None
                self.sleep(5.0)
                waited += 5.0

    def close_one(self, gpu: int) -> None:
        with self._solo_lock:
            server = self._solo_servers.pop(gpu, None)
            self._solo_server_clients.pop(gpu, None)
            self._solo_urls.pop(gpu, None)
            self._solo_loaded.pop(gpu, None)
        if server is not None:
            server.stop()

    def close(self) -> None:
        servers, self._servers, self._urls, self._loaded = self._servers, [], [], set()
        head, self._head = self._head, None
        errors = []
        for group in (servers, list(self._solo_servers.values())):
            try:
                stop_all(group)
            except Exception as exc:  # noqa: BLE001 - release the other resources too
                errors.append(exc)
        self._solo_servers, self._solo_server_clients, self._solo_urls, self._solo_loaded = {}, {}, {}, {}
        self._solo_start_locks = {}
        self._solo_clients = None
        self._solo_scheduler = None
        try:
            self._reset_grader()
        except Exception as exc:  # noqa: BLE001 - a cache cleanup failure cannot mask this one
            errors.append(exc)
        if head is not None:
            try:
                self.resident.prune({head.base_hash, head.base_fp32_master_digest})
            except Exception as exc:  # noqa: BLE001 - every close path is attempted
                errors.append(exc)
        if errors:
            raise InfraError(f"evaluator cleanup failed: {errors[0]}") from errors[0]

    def _grade(self, tasks: Sequence[str], jobs: list, replies: list) -> dict[tuple[str, int, int], str]:
        codes = {job: ERROR if reply is None else CAP for job, reply in zip(jobs, replies)}
        ended = [(job, reply[0]) for job, reply in zip(jobs, replies) if reply is not None and not reply[1]]
        for _ in range(self.settings.grade_tries):
            try:
                grades = self._grade_batch([(self.pool.task(tasks[job[2]]), text, False) for job, text in ended])
            except Exception as exc:  # noqa: BLE001
                raise InfraError(f"the grader failed: {exc}") from exc
            if len(grades) != len(ended):
                raise InfraError("the grader returned a wrong number of grades")
            for (job, _), grade in zip(ended, grades):
                codes[job] = grade.code if grade.code in (PASS, FAIL, ERROR) else ERROR
            ended = [item for item, grade in zip(ended, grades) if box_fault(grade)]
            if not ended:
                return codes
        raise InfraError(f"a grader worker died on {len(ended)} replies in every one of {self.settings.grade_tries} tries; the look is not scored")

    def _serve(self, head: HeadVersion, params: Params, table: int) -> None:
        if self._head is not None and self._head.version_id == head.version_id and self._table == table:
            if all(server.alive() for server in self._servers):
                return
        self.close()
        base_dir = resident_head(self.resident, head)
        self.gpus_free(self.settings.free_limit_mib, self.settings.free_timeout_s)
        logs = self.config.work_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self._head, self._table = head, table
        for gpu in range(params.n_gpus):
            port = self.settings.first_port + gpu
            argv = server_argv(str(base_dir), port, params, self.settings.clients_per_server, table)
            self._servers.append(self.spawn(argv, server_env(gpu, self.config.work_dir), logs / f"vllm-{gpu}.log"))
            self._urls.append(f"http://127.0.0.1:{port}")
        waited = 0.0
        for server, url in zip(self._servers, self._urls):
            while True:
                try:
                    self.http(url + "/health", None, 10.0)
                    break
                except (OSError, HttpError):
                    if not server.alive() or waited >= self.settings.start_timeout_s:
                        self.close()
                        raise InfraError(f"the vLLM server at {url} did not start") from None
                    self.sleep(5.0)
                    waited += 5.0

    def _call(self, url: str, body: dict | None, timeout_s: float) -> dict:
        try:
            return self.http(url, body, timeout_s)
        except (OSError, HttpError) as exc:
            raise InfraError(f"{url} failed: {exc}") from exc
