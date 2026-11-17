"""Streaming evaluation: null M and every LoRA on the same tasks, seeds and caps.

One vLLM server runs per GPU. Each serves the head as the model ``null``
and every LoRA of the round as an adapter on top of it. Task i goes to server
i modulo the number of GPUs for every model. Each server keeps its request
slots full as replies arrive, across grading blocks and looks; there is no
500-task barrier. Requests are task-major, null M first then each open LoRA.

Outcome codes: ``P`` pass and ``F`` fail come from the official grader.
``C`` is a reply that was cut at the token cap: it is never graded. ``E`` is
a failure of ours (request or grader): it is not a pass and it stays in the
count. A dead server is not an ``E``: it is restarted, its accepted replies
are kept, and its other jobs run again. A rollout cut by the round deadline
is discarded and never becomes an ``E``.
"""

from __future__ import annotations

import json
import resource
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Sequence

from ..errors import ContractError, InfraError
from ..params import Params
from ..sampling import rollout_seed
from ..seams import ObjectStore
from ..types import CAP, ERROR, FAIL, NULL_MODEL, PASS, EvalBlock, EvalChunk, EvalRequest, HeadVersion, eval_blocks
from .batch import END_OF_TURN
from .plan import BoxConfig, check_pins, log_phase
from .procs import Child, Process, Spawn, gpu_env, stop_all, wait_gpus_free
from .resident import Resident, resident_head


class TaskPool(Protocol):
    """The task pool of the env (``envs.math_pool.MathPool``). A task has ``messages()``: the scoring prompt."""

    def task(self, task_id: str) -> Any: ...


class Grader(Protocol):
    """The official grader (``envs.math_grader.GradePool``). Each grade has ``code``: P, F, C or E."""

    def grade_many(self, jobs: Sequence[tuple[Any, str, bool]]) -> list[Any]: ...


class HttpError(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


def http_json(url: str, body: dict | None, timeout_s: float) -> dict:
    """GET (no body) or POST JSON. Returns the JSON reply, or ``{}`` for a reply that is not JSON."""
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


Http = Callable[[str, dict | None, float], dict]


def socket_room(sockets: int, spare: int) -> int:
    """How many sockets this process may hold next to ``spare`` other files. It first raises its soft limit of open files.

    Every rollout in flight holds one socket, and a look sends up to 256
    rollouts per GPU at once. A shell or a container starts a process with a
    soft limit of 1,024 open files. Past the limit a socket cannot be
    opened: the rollouts of the first models in the order get the sockets,
    and every later rollout fails with "Too many open files" before any
    server sees it. So the soft limit is raised here to ``sockets + spare``,
    as far as the hard limit allows. The caller compares the answer with
    what the look needs. A limit that is already high enough is not changed.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return sockets
    wanted = sockets + spare
    reachable = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
    if reachable > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (reachable, hard))
            soft = reachable
        except (OSError, ValueError):  # the limit stays as it was: the answer says so
            pass
    return max(0, soft - spare)


@dataclass(frozen=True)
class ServeSettings:
    """Local serving knobs. None of them changes a score."""

    first_port: int = 8100
    start_timeout_s: float = 1800.0  # model load time of one server
    clients_per_server: int = 256  # requests in flight per server, and the server's --max-num-seqs
    retries: int = 2  # extra tries after a transport failure or a 5xx reply
    spare_files: int = 512  # open files of this process next to the sockets of a look: logs, pipes, the grader's workers
    free_limit_mib: int = 1024  # a GPU above this is still in use
    free_timeout_s: float = 120.0
    grade_tries: int = 3  # times a reply is graded when the grader's worker process dies on it
    server_deaths: int = 3  # server deaths allowed per evaluation; the next one is InfraError


def lora_table(params: Params, batches: int = 0) -> int:
    """The size of a server's LoRA table (``--max-loras``) for a round: the entries of the box's ticket.

    vLLM sizes its LoRA kernels by this table, not by the adapters in use: a
    table of 32 makes five adapters slower than a table of five. The count is
    that of the ticket, not of the LoRAs that trained, so it is the same on
    any box and on a rerun, and the same for every look of the round. A base
    request takes no place in the table: vLLM keeps one more for it itself.
    """
    if not 0 <= batches <= params.max_loras_per_box:
        raise ContractError("the batches of a look must be inside max_loras_per_box")
    return batches or params.max_loras_per_box


def server_argv(base_dir: str, port: int, params: Params, max_seqs: int = ServeSettings.clients_per_server, max_loras: int | None = None) -> list[str]:
    rank = params.recipe.bound("rank")
    max_rank = int(rank.high) if rank else params.recipe.rank
    return [
        *(sys.executable, "-m", "vllm.entrypoints.openai.api_server"),
        *("--model", base_dir, "--served-model-name", NULL_MODEL),
        *("--host", "127.0.0.1", "--port", str(port)),
        *("--dtype", params.dtype, "--max-model-len", str(params.max_context_tokens)),
        *("--enable-prefix-caching", "--enable-lora", "--max-num-seqs", str(max_seqs)),
        *("--max-loras", str(max_loras or lora_table(params)), "--max-lora-rank", str(max_rank)),
        # Text only. v1 lesson: without this limit vLLM does not load this checkpoint.
        *("--limit-mm-per-prompt", '{"image":0,"video":0}'),
    ]


def server_env(gpu: int, work_dir: Path) -> dict[str, str]:
    """One GPU and compile caches of its own. v1 lesson: servers that share a compile cache kill each other."""
    cache = work_dir / "vllm-cache" / f"gpu{gpu}"
    names = {"VLLM_CACHE_ROOT": "vllm", "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor"}
    for name in names.values():
        (cache / name).mkdir(parents=True, exist_ok=True)
    env = gpu_env([gpu])
    env.update({key: str(cache / name) for key, name in names.items()})
    env.update(HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
    # Without this variable, vLLM does not make the route /v1/load_lora_adapter. The server listens on 127.0.0.1 only.
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    return env


WORKER_DIED = "worker-died"  # Grade.machine_cause of the env's grader when its worker process died


def box_fault(grade: Any) -> bool:
    """True when this grade says nothing about the text: the worker process that graded it died."""
    return getattr(grade, "machine_cause", None) == WORKER_DIED


def chat_body(model: str, messages: list[dict[str, str]], seed: int, params: Params) -> dict:
    """One rollout request. Everything but ``model`` is the same for null M and every LoRA."""
    return {
        "model": model,
        "messages": messages,
        "seed": seed,
        "max_tokens": params.max_output_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "skip_special_tokens": False,  # the grader needs <think> and </think> in the text
        "chat_template_kwargs": {"enable_thinking": True},
    }


def read_reply(reply: dict) -> tuple[str, bool]:
    """``(text, capped)`` of a chat reply. A reply that stopped at the token cap is capped: it did not end."""
    choice = reply["choices"][0]
    text, reason = choice["message"]["content"], choice["finish_reason"]
    if not isinstance(text, str) or reason not in ("stop", "length"):
        raise ValueError(f"unexpected reply: finish_reason {reason!r}")
    if reason == "stop" and text.endswith(END_OF_TURN):
        text = text[: -len(END_OF_TURN)]
    return text, reason == "length"


def written_facts(written: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """What the log keeps of the rollouts that got a reply, from the ``(output tokens, seconds)`` of each one."""
    # The longest rollout of the replies logged here. Its time includes the wait for a free place on its server.
    longest = max(written, key=lambda pair: pair[1], default=(0, 0.0))
    tokens = [count for count, _ in written]
    return {"output_tokens": sum(tokens), "longest_rollout_s": longest[1], "longest_rollout_tokens": longest[0], "most_output_tokens": max(tokens, default=0)}


@dataclass(frozen=True)
class _Job:
    model: str
    sample: int
    index: int
    server: int
    body: dict


class _Jobs:
    """Task-major jobs, null M first then every open LoRA, with one ordered queue per server."""

    def __init__(self, task_ids: Sequence[str], models: Sequence[str], samples: int, n_servers: int, limit: Callable[[str], int], body: Callable[[str, int, int], dict]):
        self.task_ids, self.models, self.samples, self.n_servers = task_ids, models, samples, n_servers
        self.limit, self.body = limit, body
        self.index = 0
        self.pending = [deque() for _ in range(n_servers)]
        self.seen: set[tuple[str, int, int]] = set()
        self.accepted: dict[tuple[str, int, int], tuple[str, bool] | None] = {}
        self.deaths = 0
        self.dispatched = 0

    def next(self, server: int) -> _Job | None:
        while not self.pending[server] and self.index < len(self.task_ids):
            index = self.index
            for model in self.models:
                for sample in range(self.samples):
                    key = (model, sample, index)
                    if index < self.limit(model) and key not in self.seen and key not in self.accepted:
                        self.seen.add(key)
                        self.pending[index % self.n_servers].append(_Job(model, sample, index, index % self.n_servers, self.body(model, sample, index)))
            self.index += 1
        return self.pending[server].popleft() if self.pending[server] else None

    def again(self, job: _Job) -> None:
        self.pending[job.server].append(job)

    def result(self, job: _Job, result: tuple[str, bool] | None, alive: bool, final: bool) -> None:
        """Accept a reply once. A transport failure is E only after every try on a live server."""
        key = (job.model, job.sample, job.index)
        if result is not None:
            self.accepted.setdefault(key, result)
        elif final and alive and key not in self.accepted:
            self.accepted[key] = None

    def ready(self, model: str, start: int, end: int) -> bool:
        return all((model, sample, index) in self.accepted for sample in range(self.samples) for index in range(start, end))

    def done(self, models: Sequence[str]) -> bool:
        return all(limit <= index or (model, sample, index) in self.accepted for model in models for limit in [self.limit(model)] for sample in range(self.samples) for index in range(len(self.task_ids)))


class VllmEvaluator:
    def __init__(
        self,
        store: ObjectStore,
        config: BoxConfig,
        pool: TaskPool,
        grader: Grader,
        *,
        spawn: Spawn = Child,
        http: Http = http_json,
        gpus_free: Callable[[int, float], None] = wait_gpus_free,
        settings: ServeSettings = ServeSettings(),
        sleep: Callable[[float], None] = time.sleep,
        room: Callable[[int, int], int] = socket_room,
    ):
        self.store, self.config, self.pool, self.grader = store, config, pool, grader
        self.spawn, self.http, self.gpus_free, self.settings, self.sleep, self.room = spawn, http, gpus_free, settings, sleep, room
        self.resident = Resident(store, config.work_dir)
        self._servers: list[Process] = []
        self._urls: list[str] = []
        self._head: HeadVersion | None = None
        self._table = 0  # --max-loras of the servers that run
        self._generation: dict[int, int] = {}  # process generation per server, retained across close/restart
        self._loaded: set[str] = set()

    def evaluate(self, request: EvalRequest) -> EvalChunk:
        """Collect a direct stream for serving-seam tests and local tools. The runner uses ``stream`` instead."""
        outcomes: dict[str, list[str]] = {}
        for block in self.stream(request, lambda model: len(request.task_ids)):
            rows = outcomes.setdefault(block.model, [""] * request.params.samples_per_task)
            for sample, text in enumerate(block.outcomes):
                rows[sample] += text
        if request.deadline and time.time() >= request.deadline:
            raise InfraError("the round's time limit passed during the stream; a rollout cut by it is not scored")
        if any(not server.alive() for server in self._servers):
            raise InfraError("a vLLM server died while the evaluation closed")
        return EvalChunk(request.look, request.task_ids, {model: tuple(rows) for model, rows in outcomes.items()}, self.config.image_digest)

    def stream(self, request: EvalRequest, limit: Callable[[str], int]) -> Iterator[EvalBlock]:
        """Keep every server's slots full, and yield each graded block as soon as its replies are ready.

        ``limit(model)`` is the runner's live dispatch limit. Requests after it are not sent; already running requests
        may still finish. Only this thread grades or yields, and a generator close waits for those callbacks to return.
        """
        params, tasks = request.params, request.task_ids
        check_pins(self.config.image_digest, params)
        if any(lora.head_version_id != request.head.version_id for lora in request.loras):
            raise ContractError("a LoRA of this evaluation was trained on another head")
        models = [NULL_MODEL, *(lora.lora_hash for lora in request.loras)]
        table = lora_table(params, request.batches)
        if len(set(models)) != len(models) or len(request.loras) > table:
            raise ContractError("the LoRAs of an evaluation must be distinct and inside the LoRA table of its round")
        # Before anything is sent: a stream that cannot hold its sockets would turn the rollouts of the later models into E.
        clients = params.n_gpus * self.settings.clients_per_server
        room = self.room(clients, self.settings.spare_files)
        if room < clients:
            raise InfraError(f"the open file limit of this process has room for {room} sockets; a stream holds {clients}. Raise the limit (ulimit -n) and start run again")
        started = time.monotonic()
        self._serve(request.head, params, table)
        self._load(request.loras)
        served = time.monotonic()
        blocks = eval_blocks(params.eval_chunk_tasks, params.look_sizes(len(tasks)))
        starts = [start for start, _ in blocks]

        def block_start(index: int) -> int:
            return max(start for start in starts if start <= index)

        written: dict[str, list[tuple[int, float]]] = {model: [] for model in models}
        block_started: dict[tuple[str, int], float] = {}
        block_generate: dict[tuple[str, int], float] = {}
        completed: dict[Future, tuple[tuple[str, bool] | None, bool, bool, int, float]] = {}
        complete_lock = threading.Lock()

        def body(model: str, sample: int, index: int) -> dict:
            return chat_body(model, self.pool.task(tasks[index]).messages(), rollout_seed(request.round_seed, tasks[index], sample), params)

        state = _Jobs(tasks, models, params.samples_per_task, len(self._urls), limit, body)

        def timed_out() -> bool:
            return bool(request.deadline and time.time() >= request.deadline)

        def rollout(job: _Job, server: Process, url: str, generation: int) -> tuple[tuple[str, bool] | None, bool, bool, int, float]:
            if generation != self._generation.get(job.server):
                return None, False, False, 0, 0.0
            sent = time.monotonic()
            for _ in range(self.settings.retries + 1):
                budget = float(params.eval_time_budget_s)
                if request.deadline:
                    budget = min(budget, request.deadline - time.time())
                    if budget <= 0:
                        return None, False, False, 0, 0.0
                try:
                    reply = self.http(url, job.body, budget)
                    result = read_reply(reply)
                    usage = reply.get("usage")
                    tokens = int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0
                    return result, server.alive() and generation == self._generation.get(job.server), False, tokens, time.monotonic() - sent
                except HttpError as exc:
                    if exc.status < 500:
                        return None, server.alive() and generation == self._generation.get(job.server), True, 0, 0.0
                except (OSError, KeyError, IndexError, TypeError, ValueError):
                    pass
                if not server.alive() and generation == self._generation.get(job.server):
                    return None, False, False, 0, 0.0
            return None, True, True, 0, 0.0

        def finished(future: Future) -> None:
            with complete_lock:
                completed[future] = future.result()

        def restart(server: int) -> None:
            state.deaths += 1
            if state.deaths > self.settings.server_deaths:
                raise InfraError(f"a vLLM server died {state.deaths} times during this evaluation")
            # Only this server's unresolved jobs are retried. Other servers' queues, in-flight requests and E codes stay.
            for future, job in list(flying.items()):
                if job.server == server:
                    state.again(job)
                    flying.pop(future, None)
                    in_flight[job.server] -= 1
            for (model, sample, index), reply in list(state.accepted.items()):
                if reply is None and index % len(self._urls) == server:
                    del state.accepted[model, sample, index]
                    state.again(_Job(model, sample, index, server, body(model, sample, index)))
            self._generation[server] += 1
            # Restart only the dead GPU's server. Other servers keep their accepted and in-flight work.
            old = self._servers[server]
            old.stop()
            head = request.head
            base_dir = resident_head(self.resident, head)
            port = self.settings.first_port + server
            logs = self.config.work_dir / "logs"
            self._servers[server] = self.spawn(server_argv(str(base_dir), port, params, self.settings.clients_per_server, table), server_env(server, self.config.work_dir), logs / f"vllm-{server}.log")
            url = f"http://127.0.0.1:{port}"
            self._urls[server] = url
            waited = 0.0
            while True:
                try:
                    self.http(url + "/health", None, 10.0)
                    break
                except (OSError, HttpError):
                    if not self._servers[server].alive() or waited >= self.settings.start_timeout_s:
                        raise InfraError(f"the vLLM server at {url} did not restart") from None
                    self.sleep(5.0)
                    waited += 5.0
            self._load(request.loras, force=True)

        def grade(model: str, start: int, end: int) -> EvalBlock | None:
            dead = next((index for index, server in enumerate(self._servers) if not server.alive()), None)
            if dead is not None:
                restart(dead)
            if limit(model) <= start:
                return None
            jobs = [(model, sample, index) for sample in range(params.samples_per_task) for index in range(start, end)]
            replies = [state.accepted[model, sample, index] for _, sample, index in jobs]
            generated = time.monotonic()
            codes = self._grade(tasks, jobs, replies)
            outcomes = tuple("".join(codes[model, sample, index] for index in range(start, end)) for sample in range(params.samples_per_task))
            began = block_started.get((model, start), started)
            times = {"serve_s": served - started, "generate_s": block_generate.get((model, start), generated) - began, "grade_s": time.monotonic() - generated, "wall_s": time.monotonic() - began}
            counts = {"models": len(models), "tasks": end - start, "rollouts": len(jobs), "capped": sum(code == CAP for code in codes.values()), "n_gpus": len(self._urls), "max_loras": table, "open_files_limit": resource.getrlimit(resource.RLIMIT_NOFILE)[0], "model": model}
            look = next((index for index, stop in enumerate(params.look_sizes(len(tasks))) if end <= stop), len(params.look_sizes(len(tasks))) - 1)
            log_phase(self.config.work_dir, "eval", round_id=request.round_id, look=look, **times, **counts, **written_facts(written[model]))
            return EvalBlock(model, start, end, outcomes)

        flying: dict[Future, _Job] = {}
        in_flight = [0] * len(self._urls)
        graded: set[tuple[str, int]] = set()
        stopped = timed_out()
        with ThreadPoolExecutor(len(self._urls) * self.settings.clients_per_server) as pool:
            try:
                while True:
                    if any(not server.alive() for server in self._servers):
                        dead = next(index for index, server in enumerate(self._servers) if not server.alive())
                        restart(dead)
                    with complete_lock:
                        done = [(future, completed.pop(future)) for future in list(completed)]
                    for future, (result, alive, final, tokens, elapsed) in done:
                        job = flying.pop(future, None)
                        if job is None:
                            continue  # a late reply from a server that has been replaced
                        in_flight[job.server] -= 1
                        key = (job.model, job.sample, job.index)
                        if result is not None:
                            if key not in state.accepted:
                                written[job.model].append((tokens, elapsed))
                                block_generate[job.model, block_start(job.index)] = time.monotonic()
                        state.result(job, result, alive, final)
                        if result is None and not timed_out() and key not in state.accepted:
                            state.seen.discard(key)
                            state.again(job)
                    for server in range(len(self._urls)):
                        while in_flight[server] < self.settings.clients_per_server:
                            if timed_out():
                                stopped = True
                                break
                            job = state.next(server)
                            if job is None:
                                break
                            state.dispatched += 1
                            block_started.setdefault((job.model, block_start(job.index)), time.monotonic())
                            future = pool.submit(rollout, job, self._servers[job.server], self._urls[job.server] + "/v1/chat/completions", self._generation[job.server])
                            flying[future] = job
                            in_flight[server] += 1
                            future.add_done_callback(finished)
                    for start, end in blocks:
                        for model in models:
                            if (model, start) not in graded and state.ready(model, start, end):
                                block = grade(model, start, end)
                                graded.add((model, start))
                                if block is not None:
                                    yield block
                    if state.done(models) and not flying:
                        break
                    if stopped and not flying:
                        for start, end in blocks:
                            for model in models:
                                if (model, start) not in graded and state.ready(model, start, end):
                                    block = grade(model, start, end)
                                    graded.add((model, start))
                                    if block is not None:
                                        yield block
                        break
                    if not flying and not any(state.pending) and not state.done(models) and state.index >= len(tasks):
                        raise InfraError("the stream has unresolved rollouts but no dispatchable jobs")
                    wait(tuple(flying), timeout=0.05, return_when=FIRST_COMPLETED) if flying else time.sleep(0.01)
                if stopped or timed_out():
                    facts = {"serve_s": served - started, "generate_s": time.monotonic() - served, "wall_s": time.monotonic() - started}
                    facts.update(models=len(models), tasks=len(tasks), rollouts=state.dispatched, finished=len(state.accepted), reply_cap=params.max_output_tokens, max_loras=table, open_files_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0])
                    try:
                        log_phase(self.config.work_dir, "eval_cut", round_id=request.round_id, look=request.look, n_gpus=len(self._urls), **facts, **written_facts([pair for rows in written.values() for pair in rows]))
                    except Exception:  # noqa: BLE001 - a cut's log cannot become a miner error
                        pass
            finally:
                # Closing a generator stops dispatch. Only this thread ever grades or yields blocks.
                for future in flying:
                    future.result()

    def close(self) -> None:
        servers, self._servers, self._urls, self._loaded = self._servers, [], [], set()
        self._generation = {server: generation + 1 for server, generation in self._generation.items()}
        head, self._head = self._head, None
        stop_all(servers)
        if head is not None:  # LoRAs of the round leave the box; the head and its master stay
            self.resident.prune({head.base_hash, head.base_fp32_master_digest})

    def _load(self, loras: Sequence[Any], force: bool = False) -> None:
        """Load every new adapter on every server, in parallel across servers, once per adapter name.

        ``force`` reloads the whole round on a replacement server, whose own adapter table starts empty.
        """
        fresh = [(lora.lora_hash, str(self.resident.tree(lora.prefix, lora.lora_hash))) for lora in loras if force or lora.lora_hash not in self._loaded]

        def load(url: str) -> None:
            cards = self._call(url + "/v1/models", None, 60.0).get("data")
            held = {card.get("id") for card in cards if isinstance(card, dict)} if isinstance(cards, list) else set()
            for name, path in fresh:
                if name not in held:
                    self._call(url + "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}, 600.0)

        with ThreadPoolExecutor(len(self._urls)) as threads:
            list(threads.map(load, self._urls))
        self._loaded.update(name for name, _ in fresh)

    def _grade(self, tasks: Sequence[str], jobs: list, replies: list) -> dict[tuple[str, int, int], str]:
        """One code per job. The grader sees only replies that ended; it never learns which model wrote one.

        An ``E`` of the grader itself (it cannot decide, or its own time limit)
        is final and is never graded again. A grade lost to a dead worker
        process is a failure of this box: only those replies are graded
        again, and if a worker still dies the look is an ``InfraError``. So
        a dead worker never puts a code into an outcome string.
        """
        codes = {job: ERROR if reply is None else CAP for job, reply in zip(jobs, replies)}
        ended = [(job, reply[0]) for job, reply in zip(jobs, replies) if reply is not None and not reply[1]]
        for _ in range(self.settings.grade_tries):
            try:
                grades = self.grader.grade_many([(self.pool.task(tasks[job[2]]), text, False) for job, text in ended])
            except Exception as exc:  # noqa: BLE001 - a broken grader is a failure of this box
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
        """Make one server per GPU serve ``head`` with a LoRA table of ``table``. Servers of another head or table are stopped first."""
        if self._head is not None and self._head.version_id == head.version_id and self._table == table:
            if all(server.alive() for server in self._servers):
                return
        self.close()
        base_dir = resident_head(self.resident, head)
        self.gpus_free(self.settings.free_limit_mib, self.settings.free_timeout_s)  # training must be gone first
        logs = self.config.work_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self._head, self._table = head, table
        for gpu in range(params.n_gpus):
            self._generation[gpu] = self._generation.get(gpu, 0) + 1
            port = self.settings.first_port + gpu
            self._servers.append(self.spawn(server_argv(str(base_dir), port, params, self.settings.clients_per_server, table), server_env(gpu, self.config.work_dir), logs / f"vllm-{gpu}.log"))
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
