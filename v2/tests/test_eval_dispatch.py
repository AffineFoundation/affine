"""Scoring with many LoRAs: every model gets every task on every server, from the ticket to the box result.

The numbers are those of mainnet r000003: null M and 7 LoRAs on 8 servers,
chunks of 500 tasks, so 4,000 rollouts in a chunk for a request pool of 2,048
threads. The servers are stand-ins that record the ``model`` of every chat
request, and no real socket is opened. Everything else is the code of the
box: ``LocalRunner`` and ``VllmEvaluator``.
"""

from __future__ import annotations

import dataclasses
import errno
import json
import sys
import threading
import time
import urllib.error
from collections import Counter
from types import SimpleNamespace

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import manifest_digest, sha256_hex
from sn120_v2.errors import InfraError
from sn120_v2.gpu import plan
from sn120_v2.gpu.evaluator import HttpError, ServeSettings, VllmEvaluator
from sn120_v2.judging import check_result, score_entry
from sn120_v2.sampling import rollout_seed
from sn120_v2.store import LocalStore
from sn120_v2.types import NULL_MODEL, SCORED, EvalRequest, HeadVersion, LoRAArtifact


def collect(seam, request):
    """One direct stream, for tests of the serving seam rather than the runner's look decisions."""
    blocks = list(seam.stream(request, lambda model: len(request.task_ids)))
    outcomes = {}
    for block in blocks:
        outcomes.setdefault(block.model, [""] * request.params.samples_per_task)
        for sample, text in enumerate(block.outcomes):
            outcomes[block.model][sample] += text
    return {model: tuple(strings) for model, strings in outcomes.items()}
from sn120_v2.worker import LocalRunner

N_GPUS, N_LORAS, N_TASKS, CHUNK, FIRST_PORT = 8, 7, 2000, 500, ServeSettings.first_port
PORTS = tuple(range(FIRST_PORT, FIRST_PORT + N_GPUS))
POOL = SimpleNamespace(task=lambda task_id: SimpleNamespace(messages=lambda: [{"role": "user", "content": task_id}]))
GRADER = SimpleNamespace(grade_many=lambda jobs: [SimpleNamespace(code=text, machine_cause=None) for _, text, _ in jobs])


class Servers:
    """Stand-ins for the vLLM servers. Each keeps the adapters it loaded and the ``(model, task)`` of every chat request."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.up: set[int] = set()
        self.starts = 0
        self.generation: dict[int, int] = {}
        self.processes: dict[int, list[SimpleNamespace]] = {port: [] for port in PORTS}
        self.loaded_generation: dict[int, list[tuple[int, str]]] = {port: [] for port in PORTS}
        self.argv: dict[int, list[str]] = {}
        self.loaded: dict[int, list[str]] = {port: [] for port in PORTS}
        self.chats: dict[int, list[tuple[str, str]]] = {port: [] for port in PORTS}

    def spawn(self, argv, env, log):
        port = int(argv[argv.index("--port") + 1])
        self.starts += 1
        self.up.add(port)
        self.generation[port] = self.generation.get(port, 0) + 1
        self.argv[port] = list(argv)
        process = SimpleNamespace(alive=lambda: port in self.up, stop=lambda: self.up.discard(port), adapters=set())
        self.processes[port].append(process)
        return process

    def http(self, url, body, timeout_s):
        port = int(url.split(":")[2].split("/")[0])
        if port not in self.up:
            raise ConnectionRefusedError(url)
        with self.lock:
            current = self.processes[port][-1]
            if url.endswith("/v1/models"):
                return {"data": [{"id": NULL_MODEL}, *({"id": name} for name in current.adapters)]}
            if url.endswith("/v1/load_lora_adapter"):
                current.adapters.add(body["lora_name"])
                self.loaded_generation[port].append((self.generation[port], body["lora_name"]))
            if not url.endswith("/v1/chat/completions"):
                return {}
            if body["model"] != NULL_MODEL and body["model"] not in current.adapters:
                raise HttpError(404)  # vLLM answers 404 for a model name that it does not hold
            self.chats[port].append((body["model"], body["messages"][-1]["content"]))
        return {"choices": [{"message": {"content": "P"}, "finish_reason": "stop"}]}

    def got(self, model: str) -> dict[int, list[str]]:
        """The tasks that each server got for ``model``, in the order of their task numbers."""
        return {port: sorted(task for name, task in self.chats[port] if name == model) for port in PORTS}


class Sockets:
    """An open file limit in front of the servers: at most ``limit`` chat requests hold a socket at once.

    A request past the limit fails as ``urllib`` fails when the process has
    no file left. A request that got its socket keeps it until ``failures``
    tries have failed, as a rollout that generates for minutes keeps its
    socket while every other rollout fails in under a millisecond.
    """

    def __init__(self, servers: Servers, limit: int, failures: int) -> None:
        self.servers, self.limit, self.failures = servers, limit, failures
        self.lock, self.go = threading.Lock(), threading.Event()
        self.held = self.most = self.failed = 0

    def http(self, url, body, timeout_s):
        if not url.endswith("/v1/chat/completions"):
            return self.servers.http(url, body, timeout_s)
        with self.lock:
            refused = self.held >= self.limit
            self.failed += refused
            self.held += not refused
            self.most = max(self.most, self.held)
            if self.failed >= self.failures:
                self.go.set()
        if refused:
            raise urllib.error.URLError(OSError(errno.EMFILE, "Too many open files"))
        assert self.go.wait(120.0)
        return self.servers.http(url, body, timeout_s)


@pytest.fixture
def round_of_seven(tmp_path, build):
    """A ticket of 7 batches on a box of 8 GPUs, the 7 LoRAs in the store, and the evaluator of the box on stand-in servers."""
    store = LocalStore(tmp_path / "store")

    def put(prefix_of, files: dict[str, bytes]) -> str:
        identity = manifest_digest({name: sha256_hex(data) for name, data in files.items()})
        for name, data in files.items():
            store.put(f"{prefix_of(identity)}/{name}", data)
        return identity

    base = put(layout.public_base_prefix, {"config.json": b"{}", "model.safetensors": b"bf16"})
    master = put(layout.public_master_prefix, {"model.part00.safetensors": b"fp32"})
    head = HeadVersion.make(parent_version_id=None, base_hash=base, base_fp32_master_digest=master, loras=(), round_id="genesis", applied_at=1)
    # The numbers of r000003: 500, 1,000 and 2,000 tasks at the three looks, in chunks of 500. A chunk has 8 x 500 = 4,000
    # rollouts for a pool of 8 x 256 = 2,048 threads.
    params = build.params(n_gpus=N_GPUS, n_tasks=N_TASKS, looks=(0.25, 0.5, 1.0), eval_chunk_tasks=CHUNK, batch_cap=N_LORAS)
    now = int(time.time())  # the evaluator compares the deadline of a chunk with the real time
    ticket = dataclasses.replace(build.ticket(n_entries=N_LORAS, n_tasks=N_TASKS, params=params), head=head, issued_at=now, expires_at=now + params.lease_timeout_s)
    loras: dict[int, LoRAArtifact] = {}
    for entry in ticket.entries:
        files = {"adapter_config.json": b"{}", "adapter_model.safetensors": f"lora-{entry.queue_pos}".encode()}
        identity = put(lambda digest: f"pending/loras/{digest}", files)
        made = build.artifact(head, entry, params)
        loras[entry.queue_pos] = dataclasses.replace(made, lora_hash=identity, prefix=f"pending/loras/{identity}", files={name: sha256_hex(data) for name, data in files.items()})
    servers = Servers()
    seam = VllmEvaluator(store, plan.BoxConfig(tmp_path / "work", build.image), POOL, GRADER, spawn=servers.spawn, http=servers.http, gpus_free=lambda limit, timeout: None, settings=ServeSettings(start_timeout_s=10.0), sleep=lambda s: None, room=lambda sockets, spare: sockets)
    trainer = SimpleNamespace(train=lambda request: loras[request.entry.queue_pos], close=lambda: None)
    runner = LocalRunner(trainer, seam, lambda: int(time.time()), build.image)
    names = [loras[pos].lora_hash for pos in sorted(loras)]
    lines = lambda: [json.loads(line) for line in (tmp_path / "work" / "logs" / "phases.jsonl").read_text(encoding="utf-8").splitlines()]
    return SimpleNamespace(ticket=ticket, params=params, head=head, loras=loras, names=names, servers=servers, seam=seam, runner=runner, lines=lines)


def test_null_m_and_seven_loras_each_get_every_task_on_every_server_and_full_cover(round_of_seven):
    world = round_of_seven
    ticket, servers, tasks = world.ticket, world.servers, world.ticket.draw.task_ids
    result = world.runner.run(ticket)
    # Every server loaded every LoRA under its hash, and that hash is the model name of the LoRA's requests.
    assert all(sorted(servers.processes[port][-1].adapters) == sorted(world.names) for port in PORTS)
    assert {servers.argv[port][servers.argv[port].index("--max-loras") + 1] for port in PORTS} == {str(N_LORAS)}
    for model in (NULL_MODEL, *world.names):
        got = servers.got(model)
        # Each task one time, on the server of its place in its chunk: 250 of the 2,000 tasks on each of the 8 servers.
        assert sorted(task for port in PORTS for task in got[port]) == sorted(tasks), f"{model}: not every task was sent one time"
        assert all(got[port] == sorted(tasks[index] for index in range(N_TASKS) if index % N_GPUS == port - FIRST_PORT) for port in PORTS), model
        # The first chunk, as the access log of a server counts it: 63 requests on each of the servers 0 to 3, 62 on each of 4 to 7.
        assert [len(set(got[port]) & set(tasks[:CHUNK])) for port in PORTS] == [63, 63, 63, 63, 62, 62, 62, 62], model
    assert Counter(name for port in PORTS for name, _ in servers.chats[port]) == {model: N_TASKS for model in (NULL_MODEL, *world.names)}
    # The box result: null M and all 7 LoRAs have an outcome for every task, and no rollout is an E.
    assert result.null_outcomes == ("P" * N_TASKS,) and not result.stopped_at_limit
    # The log of the box: 4 chunks of 500 tasks, each with 8 models and 4,000 rollouts, at the looks 0, 1, 2 and 2.
    chunks = [(line["look"], line["model"], line["models"], line["tasks"], line["rollouts"], line["n_gpus"], line["max_loras"]) for line in world.lines() if line["phase"] == "eval"]
    assert Counter(chunks) == Counter([(0, model, 1 + N_LORAS, CHUNK, CHUNK, N_GPUS, N_LORAS) for model in (NULL_MODEL, *world.names)] + [(1, model, 1 + N_LORAS, CHUNK, CHUNK, N_GPUS, N_LORAS) for model in (NULL_MODEL, *world.names)] + [(2, model, 1 + N_LORAS, CHUNK, CHUNK, N_GPUS, N_LORAS) for model in (NULL_MODEL, *world.names) for _ in (0, 1)])
    assert [(entry.queue_pos, entry.status, entry.outcomes, entry.eliminated_at_look) for entry in result.entries] == [(pos, SCORED, ("P" * N_TASKS,), None) for pos in range(N_LORAS)]
    for entry in result.entries:
        score = score_entry(ticket.draw.weights, result.null_outcomes, entry.outcomes, world.params, N_TASKS)
        assert (score.cover, score.n_tasks) == (1.0, N_TASKS) and entry.lora.lora_hash == world.loras[entry.queue_pos].lora_hash


def test_seed_pairing_uses_global_task_index_and_one_server_per_task(round_of_seven):
    world = round_of_seven
    request = EvalRequest("r000001", world.head, tuple(world.loras[pos] for pos in sorted(world.loras)), world.ticket.draw.task_ids[:CHUNK], "seed-0", 0, world.params, 0, N_LORAS)
    original = world.servers.http
    seen = []
    def http(url, body, timeout_s):
        if url.endswith("/v1/chat/completions"):
            seen.append((url, body))
        return original(url, body, timeout_s)
    world.servers.http = http
    world.seam.http = http
    collect(world.seam, request)
    by_task = {}
    for url, body in seen:
        task = body["messages"][-1]["content"]
        by_task.setdefault(task, []).append((url, body))
    assert len(by_task) == CHUNK
    for index, task in enumerate(request.task_ids):
        rows = by_task[task]
        assert len({url for url, _ in rows}) == 1
        assert int(rows[0][0].split(":")[2].split("/")[0]) == FIRST_PORT + index % N_GPUS
        for _, body in rows:
            assert body["seed"] == rollout_seed("seed-0", task, 0)
        assert len({json.dumps({**body, "model": ""}, sort_keys=True) for _, body in rows}) == 1


def test_one_blocked_rollout_does_not_stop_refill_or_later_blocks(round_of_seven):
    world = round_of_seven
    servers = world.servers
    request = EvalRequest("r000001", world.head, tuple(world.loras[pos] for pos in sorted(world.loras)), world.ticket.draw.task_ids[:CHUNK], "seed-0", 0, world.params, 0, N_LORAS)
    blocked = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    flying = Counter()
    maximum = Counter()
    calls = Counter()
    original = servers.http
    def http(url, body, timeout_s):
        if not url.endswith("/v1/chat/completions"):
            return original(url, body, timeout_s)
        port = int(url.split(":")[2].split("/")[0])
        task = body["messages"][-1]["content"]
        with lock:
            flying[port] += 1
            maximum[port] = max(maximum[port], flying[port])
            calls[task] += 1
        try:
            if task == "t000":
                blocked.set()
                assert release.wait(10)
            return original(url, body, timeout_s)
        finally:
            with lock:
                flying[port] -= 1
    servers.http = http
    world.seam.http = http
    done = threading.Event()
    answer = []
    def run():
        try:
            answer.append(collect(world.seam, request))
        finally:
            done.set()
    thread = threading.Thread(target=run)
    thread.start()
    assert blocked.wait(5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not done.is_set():
        with lock:
            if maximum and all(value <= ServeSettings.clients_per_server for value in maximum.values()) and max(calls.values()) >= 2:
                break
        time.sleep(0.01)
    release.set()
    thread.join(30)
    assert done.is_set() and answer
    assert all(value <= ServeSettings.clients_per_server for value in maximum.values())
    assert any(task != "t000" for task in calls)


def test_dead_server_restarts_and_reruns_lost_jobs_without_e(round_of_seven):
    world = round_of_seven
    servers = world.servers
    request = EvalRequest("r000001", world.head, tuple(world.loras[pos] for pos in sorted(world.loras)), world.ticket.draw.task_ids[:CHUNK], "seed-0", 0, world.params, 0, N_LORAS)
    original = servers.http
    killed = False
    dead_port = None
    def http(url, body, timeout_s):
        nonlocal killed, dead_port
        if url.endswith("/v1/chat/completions") and not killed:
            killed = True
            dead_port = int(url.split(":")[2].split("/")[0])
            servers.up.discard(dead_port)
            raise ConnectionResetError("fake server death")
        return original(url, body, timeout_s)
    servers.http = http
    world.seam.http = http
    outcomes = collect(world.seam, request)
    assert all("E" not in text for strings in outcomes.values() for text in strings)
    assert all(len(strings[0]) == CHUNK for strings in outcomes.values())
    assert all(sorted(servers.processes[port][-1].adapters) == sorted(world.names) for port in PORTS)
    # A replacement server starts with an empty adapter table and gets the whole round's adapters again.
    assert dead_port is not None and len(servers.processes[dead_port]) >= 2, "the dead process was actually replaced"
    replacement = next(port for port, names in servers.loaded_generation.items() if any(generation > 1 for generation, _ in names))
    assert sorted(name for generation, name in servers.loaded_generation[replacement] if generation > 1) == sorted(world.names)
    for model in (NULL_MODEL, *world.names):
        for index, task in enumerate(request.task_ids):
            assert sum((model, task) in servers.chats[port] for port in PORTS) == 1, (model, task)


def test_fourth_server_death_is_infra_error(round_of_seven):
    world = round_of_seven
    servers = world.servers
    request = EvalRequest("r000001", world.head, tuple(world.loras[pos] for pos in sorted(world.loras)), world.ticket.draw.task_ids[:CHUNK], "seed-0", 0, world.params, 0, N_LORAS)
    original = servers.http
    original_spawn = servers.spawn
    deaths = 0
    dead_now = False
    def spawn(argv, env, log):
        nonlocal dead_now
        proc = original_spawn(argv, env, log)
        dead_now = False
        original_alive = proc.alive
        def alive():
            return original_alive() and not dead_now
        proc.alive = alive
        return proc
    servers.spawn = spawn
    world.seam.spawn = spawn
    def http(url, body, timeout_s):
        nonlocal deaths, dead_now
        if url.endswith("/v1/chat/completions") and not dead_now and deaths < 4:
            dead_now = True
            deaths += 1
            return original(url, body, timeout_s)
        return original(url, body, timeout_s)
    servers.http = http
    world.seam.http = http
    with pytest.raises(InfraError, match="died [45] times"):
        collect(world.seam, request)


def test_socket_room_is_checked_before_dispatch_and_the_per_server_cap_is_never_exceeded(round_of_seven):
    world = round_of_seven
    servers = world.servers
    request = EvalRequest("r000001", world.head, tuple(world.loras[pos] for pos in sorted(world.loras)), world.ticket.draw.task_ids[:CHUNK], "seed-0", 0, world.params, 0, N_LORAS)
    # Too little room is an infra failure before a server starts or a rollout is sent.
    world.seam.room = lambda sockets, spare: 1024 - 65
    with pytest.raises(InfraError, match="room for 959 sockets; a stream holds 2048"):
        collect(world.seam, request)
    assert not any(servers.chats.values())
    # With room for the ceiling, a slow fake server still gets exactly the same paired requests.
    world.seam.room = lambda sockets, spare: sockets
    outcomes = collect(world.seam, request)
    assert all(strings == ("P" * CHUNK,) for strings in outcomes.values())
    assert Counter(name for port in PORTS for name, _ in servers.chats[port]) == {model: CHUNK for model in (NULL_MODEL, *world.names)}
