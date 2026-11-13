"""The ``Evaluator`` seam: null M and every LoRA on the same tasks, seeds and caps.

One vLLM server runs per GPU. Each serves the head as the model ``null``
and every LoRA of the round as an adapter on top of it. A task goes to the
same server for every model, with the same seed, so a LoRA and null M differ
only by the adapter.

Outcome codes: ``P`` pass and ``F`` fail come from the official grader.
``C`` is a reply that was cut at the token cap: it is never graded. ``E`` is
a failure of ours (request or grader): it is not a pass and it stays in the
count. A dead server is not an ``E``: the whole look is an ``InfraError``.
"""

from __future__ import annotations

import json
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
from .plan import BoxConfig, check_pins
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


@dataclass(frozen=True)
class ServeSettings:
    """Local serving knobs. None of them changes a score."""

    first_port: int = 8100
    start_timeout_s: float = 1800.0  # model load time of one server
    clients_per_server: int = 64  # requests in flight per server
    retries: int = 2  # extra tries after a transport failure or a 5xx reply
    free_limit_mib: int = 1024  # a GPU above this is still in use
    free_timeout_s: float = 120.0
    grade_tries: int = 3  # times a reply is graded when the grader's worker process dies on it


def server_argv(base_dir: str, port: int, params: Params) -> list[str]:
    rank = params.recipe.bound("rank")
    max_rank = int(rank.high) if rank else params.recipe.rank
    return [
        *(sys.executable, "-m", "vllm.entrypoints.openai.api_server"),
        *("--model", base_dir, "--served-model-name", NULL_MODEL),
        *("--host", "127.0.0.1", "--port", str(port)),
        *("--dtype", params.dtype, "--max-model-len", str(params.max_context_tokens)),
        *("--enable-prefix-caching", "--enable-lora"),
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
    return env


def box_fault(grade: Any) -> bool:
    """True when this grade says nothing about the text: the worker process that graded it died."""
    return getattr(grade, "rule", "") == "grade-worker-died"


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
    ):
        self.store, self.config, self.pool, self.grader = store, config, pool, grader
        self.spawn, self.http, self.gpus_free, self.settings, self.sleep = spawn, http, gpus_free, settings, sleep
        self.resident = Resident(store, config.work_dir)
        self._servers: list[Process] = []
        self._urls: list[str] = []
        self._head: HeadVersion | None = None
        self._loaded: set[str] = set()

    def evaluate(self, request: EvalRequest) -> EvalChunk:
        params, tasks = request.params, request.task_ids
        check_pins(self.config.image_digest, params)
        if any(lora.head_version_id != request.head.version_id for lora in request.loras):
            raise ContractError("a LoRA of this look was trained on another head")
        models = [NULL_MODEL, *(lora.lora_hash for lora in request.loras)]
        if len(set(models)) != len(models) or len(request.loras) > params.max_loras_per_box:
            raise ContractError("the LoRAs of a look must be distinct and inside max_loras_per_box")
        self._serve(request.head, params)
        for lora in request.loras:
            if lora.lora_hash not in self._loaded:
                path = self.resident.tree(lora.prefix, lora.lora_hash)
                for url in self._urls:
                    self._call(url + "/v1/load_lora_adapter", {"lora_name": lora.lora_hash, "lora_path": str(path)}, 600.0)
                self._loaded.add(lora.lora_hash)
        jobs = [(model, sample, index) for model in models for sample in range(params.samples_per_task) for index in range(len(tasks))]

        def rollout(job: tuple[str, int, int]) -> tuple[str, bool] | None:
            model, sample, index = job
            messages = self.pool.task(tasks[index]).messages()
            body = chat_body(model, messages, rollout_seed(request.round_seed, tasks[index], sample), params)
            url = self._urls[index % len(self._urls)] + "/v1/chat/completions"
            for _ in range(self.settings.retries + 1):
                try:
                    return read_reply(self.http(url, body, float(params.eval_time_budget_s)))
                except HttpError as exc:
                    if exc.status < 500:
                        return None
                except (OSError, KeyError, IndexError, TypeError, ValueError):
                    pass
            return None

        with ThreadPoolExecutor(len(self._urls) * self.settings.clients_per_server) as threads:
            replies = list(threads.map(rollout, jobs))
        if not all(server.alive() for server in self._servers):
            self.close()
            raise InfraError("a vLLM server died during the look; nothing of it can be scored")
        codes = self._grade(tasks, jobs, replies)
        outcomes = {
            model: tuple("".join(codes[model, sample, index] for index in range(len(tasks))) for sample in range(params.samples_per_task))
            for model in models
        }
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
            self._servers.append(self.spawn(server_argv(str(base_dir), port, params), server_env(gpu, self.config.work_dir), logs / f"vllm-{gpu}.log"))
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

    def _call(self, url: str, body: dict, timeout_s: float) -> dict:
        try:
            return self.http(url, body, timeout_s)
        except (OSError, HttpError) as exc:
            raise InfraError(f"{url} failed: {exc}") from exc
