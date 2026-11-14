"""The ``Evaluator`` seam: null M and every LoRA on the same tasks, seeds and caps.

One vLLM server runs per GPU. Each serves the head as the model ``null``
and every LoRA of the round as an adapter on top of it. A task goes to the
same server for every model, with the same seed, so a LoRA and null M differ
only by the adapter.

Outcome codes: ``P`` pass and ``F`` fail come from the official grader.
``C`` is a reply that was cut at the token cap: it is never graded. ``E`` is
a failure of ours (request or grader): it is not a pass and it stays in the
count. A dead server is not an ``E``: the whole look is an ``InfraError``.
The same holds for a rollout that this process could not send because it
had no file left for the socket.
"""

from __future__ import annotations

import errno
import json
import resource
import sys
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


def no_file_left(exc: OSError) -> bool:
    """True when the request was never sent: this process, or the machine, had no file left for its socket."""
    cause = exc.reason if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, OSError) else exc
    return cause.errno in (errno.EMFILE, errno.ENFILE)


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
    retry_pause_s: float = 5.0  # wait before each extra try
    spare_files: int = 512  # open files of this process next to the sockets of a look: logs, pipes, the grader's workers
    free_limit_mib: int = 1024  # a GPU above this is still in use
    free_timeout_s: float = 120.0
    grade_tries: int = 3  # times a reply is graded when the grader's worker process dies on it


def server_argv(base_dir: str, port: int, params: Params, max_seqs: int = ServeSettings.clients_per_server) -> list[str]:
    rank = params.recipe.bound("rank")
    max_rank = int(rank.high) if rank else params.recipe.rank
    return [
        *(sys.executable, "-m", "vllm.entrypoints.openai.api_server"),
        *("--model", base_dir, "--served-model-name", NULL_MODEL),
        *("--host", "127.0.0.1", "--port", str(port)),
        *("--dtype", params.dtype, "--max-model-len", str(params.max_context_tokens)),
        *("--enable-prefix-caching", "--enable-lora", "--max-num-seqs", str(max_seqs)),
        *("--max-loras", str(params.max_loras_per_box), "--max-lora-rank", str(max_rank)),
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
    # The longest rollout sets the least time of a chunk. Its time includes the wait for a free place on its server.
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
        self._loaded: set[str] = set()

    def evaluate(self, request: EvalRequest) -> EvalChunk:
        params, tasks = request.params, request.task_ids
        check_pins(self.config.image_digest, params)
        started, written = time.monotonic(), []  # written: (output tokens, seconds) of each rollout that got a reply
        if any(lora.head_version_id != request.head.version_id for lora in request.loras):
            raise ContractError("a LoRA of this look was trained on another head")
        models = [NULL_MODEL, *(lora.lora_hash for lora in request.loras)]
        if len(set(models)) != len(models) or len(request.loras) > params.max_loras_per_box:
            raise ContractError("the LoRAs of a look must be distinct and inside max_loras_per_box")
        # Before anything is sent: a look that cannot hold its sockets would turn the rollouts of the later models into E.
        clients = params.n_gpus * self.settings.clients_per_server
        room = self.room(clients, self.settings.spare_files)
        if room < clients:
            raise InfraError(f"the open file limit of this process has room for {room} sockets; a look holds {clients}. Raise the limit (ulimit -n) and start run again")
        self._serve(request.head, params)
        fresh = [(lora.lora_hash, str(self.resident.tree(lora.prefix, lora.lora_hash))) for lora in request.loras if lora.lora_hash not in self._loaded]

        def load(url: str) -> None:  # one server loads its adapters one after another; the servers load side by side
            # vLLM refuses a name that it holds. A server holds one when an earlier try of this look failed on another server.
            cards = self._call(url + "/v1/models", None, 60.0).get("data")
            held = {card.get("id") for card in cards if isinstance(card, dict)} if isinstance(cards, list) else set()
            for name, path in fresh:
                if name not in held:
                    self._call(url + "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}, 600.0)

        with ThreadPoolExecutor(len(self._urls)) as threads:
            list(threads.map(load, self._urls))
        self._loaded.update(name for name, _ in fresh)
        jobs = [(model, sample, index) for model in models for sample in range(params.samples_per_task) for index in range(len(tasks))]
        served, unsent = time.monotonic(), []  # unsent: rollouts that this process could not send for lack of a file

        def rollout(job: tuple[str, int, int]) -> tuple[str, bool] | None:
            model, sample, index = job
            messages = self.pool.task(tasks[index]).messages()
            body = chat_body(model, messages, rollout_seed(request.round_seed, tasks[index], sample), params)
            url = self._urls[index % len(self._urls)] + "/v1/chat/completions"
            starved = False  # the last try failed because this process had no file left for its socket
            for attempt in range(self.settings.retries + 1):
                if unsent:  # the look is already lost: send nothing more
                    return None
                if attempt:
                    self.sleep(self.settings.retry_pause_s)
                budget = float(params.eval_time_budget_s)
                if request.deadline:  # the round's time limit: no rollout of this chunk may run past it
                    budget = min(budget, request.deadline - time.time())
                    if budget <= 0:
                        return None
                starved = False
                try:
                    sent = time.monotonic()
                    reply = self.http(url, body, budget)
                    result = read_reply(reply)
                    usage = reply.get("usage")
                    written.append((int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0, time.monotonic() - sent))
                    return result
                except HttpError as exc:
                    if exc.status < 500:
                        return None
                except OSError as exc:
                    starved = no_file_left(exc)
                except (KeyError, IndexError, TypeError, ValueError):
                    starved = False
            if starved:  # the last try never left this process: that says nothing about the model
                unsent.append(job)
            return None

        with ThreadPoolExecutor(clients) as threads:
            replies = list(threads.map(rollout, jobs))
        if not all(server.alive() for server in self._servers):
            self.close()
            raise InfraError("a vLLM server died during the look; nothing of it can be scored")
        if unsent:
            # Not an E: an E counts against its model, and these rollouts never reached a server. So the look is not scored at all.
            raise InfraError(f"this process had no file left for the socket of {len(unsent)} rollouts (too many open files); nothing of the look is scored")
        if request.deadline and time.time() >= request.deadline and any(reply is None for reply in replies):
            # A rollout that the limit cut is not an error of its model. Long outputs are cut first, so this chunk is not scored at all.
            # Its speeds are still logged, as their own phase: no reader may count it as a chunk that was scored.
            cut = {"serve_s": served - started, "generate_s": time.monotonic() - served, "wall_s": time.monotonic() - started}
            cut.update(models=len(models), tasks=len(tasks), rollouts=len(jobs), finished=sum(reply is not None for reply in replies), reply_cap=params.max_output_tokens)
            try:
                log_phase(self.config.work_dir, "eval_cut", round_id=request.round_id, look=request.look, n_gpus=len(self._urls), **cut, **written_facts(written))
            except Exception:  # noqa: BLE001 - the line is a help for the operator: a log that cannot be written must not change how the round ends
                pass
            raise InfraError("the round's time limit passed during this chunk; an unfinished chunk is not scored")
        generated = time.monotonic()
        codes = self._grade(tasks, jobs, replies)
        outcomes = {
            model: tuple("".join(codes[model, sample, index] for index in range(len(tasks))) for sample in range(params.samples_per_task))
            for model in models
        }
        times = {"serve_s": served - started, "generate_s": generated - served, "grade_s": time.monotonic() - generated, "wall_s": time.monotonic() - started}
        counts = {"models": len(models), "tasks": len(tasks), "rollouts": len(jobs), "capped": sum(code == CAP for code in codes.values()), "n_gpus": len(self._urls)}
        log_phase(self.config.work_dir, "eval", round_id=request.round_id, look=request.look, **times, **counts, **written_facts(written))
        return EvalChunk(look=request.look, task_ids=tasks, outcomes=outcomes, image_digest=self.config.image_digest)

    def close(self) -> None:
        servers, self._servers, self._urls, self._loaded = self._servers, [], [], set()
        head, self._head = self._head, None
        stop_all(servers)
        if head is not None:  # LoRAs of the round leave the box; the head and its master stay
            self.resident.prune({head.base_hash, head.base_fp32_master_digest})

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

    def _serve(self, head: HeadVersion, params: Params) -> None:
        """Make one server per GPU serve ``head``. Servers of another head are stopped first."""
        if self._head is not None and self._head.version_id == head.version_id:
            if all(server.alive() for server in self._servers):
                return
        self.close()
        base_dir = resident_head(self.resident, head)
        self.gpus_free(self.settings.free_limit_mib, self.settings.free_timeout_s)  # training must be gone first
        logs = self.config.work_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self._head = head
        for gpu in range(params.n_gpus):
            port = self.settings.first_port + gpu
            self._servers.append(self.spawn(server_argv(str(base_dir), port, params, self.settings.clients_per_server), server_env(gpu, self.config.work_dir), logs / f"vllm-{gpu}.log"))
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
