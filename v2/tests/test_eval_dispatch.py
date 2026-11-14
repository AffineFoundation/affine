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
from sn120_v2.judging import score_entry
from sn120_v2.store import LocalStore
from sn120_v2.types import NULL_MODEL, SCORED, EvalRequest, HeadVersion, LoRAArtifact
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
        self.argv: dict[int, list[str]] = {}
        self.loaded: dict[int, list[str]] = {port: [] for port in PORTS}
        self.chats: dict[int, list[tuple[str, str]]] = {port: [] for port in PORTS}

    def spawn(self, argv, env, log):
        port = int(argv[argv.index("--port") + 1])
        self.up.add(port)
        self.argv[port] = list(argv)
        return SimpleNamespace(alive=lambda: port in self.up, stop=lambda: self.up.discard(port))

    def http(self, url, body, timeout_s):
        port = int(url.split(":")[2].split("/")[0])
        if port not in self.up:
            raise ConnectionRefusedError(url)
        with self.lock:
            if url.endswith("/v1/models"):
                return {"data": [{"id": NULL_MODEL}, *({"id": name} for name in self.loaded[port])]}
            if url.endswith("/v1/load_lora_adapter"):
                self.loaded[port].append(body["lora_name"])
            if not url.endswith("/v1/chat/completions"):
                return {}
            if body["model"] != NULL_MODEL and body["model"] not in self.loaded[port]:
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
    assert all(sorted(servers.loaded[port]) == sorted(world.names) for port in PORTS)
    assert {servers.argv[port][servers.argv[port].index("--max-loras") + 1] for port in PORTS} == {str(N_LORAS)}
    for model in (NULL_MODEL, *world.names):
        got = servers.got(model)
        # Each task one time, on the server of its place in its chunk: 250 of the 2,000 tasks on each of the 8 servers.
        assert sorted(task for port in PORTS for task in got[port]) == sorted(tasks), f"{model}: not every task was sent one time"
        assert all(got[port] == sorted(tasks[index] for index in range(N_TASKS) if index % CHUNK % N_GPUS == port - FIRST_PORT) for port in PORTS), model
        # The first chunk, as the access log of a server counts it: 63 requests on each of the servers 0 to 3, 62 on each of 4 to 7.
        assert [len(set(got[port]) & set(tasks[:CHUNK])) for port in PORTS] == [63, 63, 63, 63, 62, 62, 62, 62], model
    assert Counter(name for port in PORTS for name, _ in servers.chats[port]) == {model: N_TASKS for model in (NULL_MODEL, *world.names)}
    # The box result: null M and all 7 LoRAs have an outcome for every task, and no rollout is an E.
    assert result.null_outcomes == ("P" * N_TASKS,) and not result.stopped_at_limit
    # The log of the box: 4 chunks of 500 tasks, each with 8 models and 4,000 rollouts, at the looks 0, 1, 2 and 2.
    chunks = [(line["look"], line["models"], line["tasks"], line["rollouts"], line["n_gpus"], line["max_loras"]) for line in world.lines() if line["phase"] == "eval"]
    assert chunks == [(look, 1 + N_LORAS, CHUNK, 4000, N_GPUS, N_LORAS) for look in (0, 1, 2, 2)]
    assert [(entry.queue_pos, entry.status, entry.outcomes, entry.eliminated_at_look) for entry in result.entries] == [(pos, SCORED, ("P" * N_TASKS,), None) for pos in range(N_LORAS)]
    for entry in result.entries:
        score = score_entry(ticket.draw.weights, result.null_outcomes, entry.outcomes, world.params, N_TASKS)
        assert (score.cover, score.n_tasks) == (1.0, N_TASKS) and entry.lora.lora_hash == world.loras[entry.queue_pos].lora_hash


def test_at_1024_open_files_null_m_is_full_the_first_lora_is_short_and_the_others_get_nothing(round_of_seven):
    """Mainnet r000003 in its own numbers: 8 x 500 = 4,000 rollouts, 2,048 threads, and 959 sockets left of a limit of 1,024."""
    world = round_of_seven
    servers, tasks = world.servers, world.ticket.draw.task_ids[:CHUNK]
    request = EvalRequest("r000001", world.head, tuple(world.loras[pos] for pos in sorted(world.loras)), tasks, "seed-0", 0, world.params, 0, N_LORAS)
    rollouts, limit = (1 + N_LORAS) * len(tasks), 1024 - 65  # the process held 65 files before the look
    files = Sockets(servers, limit, failures=(rollouts - limit) * (ServeSettings().retries + 1))
    world.seam.http = files.http
    # The request loop before the fix did not ask for room: it sent, and a rollout without a socket became an E.
    world.seam.room = lambda sockets, spare: sockets
    interval = sys.getswitchinterval()
    sys.setswitchinterval(60.0)  # a thread runs until it waits: the rollouts reach the limit in the order of the jobs, so the test has one result
    try:
        outcomes = world.seam.evaluate(request).outcomes
    finally:
        sys.setswitchinterval(interval)
    first, later = world.names[0], world.names[1:]
    # Null M has all 500. The first LoRA has its first 459 tasks, and its last 41 are E. Each of the 6 others is 500 E.
    assert outcomes[NULL_MODEL] == ("P" * 500,) and outcomes[first] == ("P" * 459 + "E" * 41,)
    assert all(outcomes[name] == ("E" * 500,) for name in later)
    cover = lambda name: score_entry(world.ticket.draw.weights, outcomes[NULL_MODEL], outcomes[name], world.params).cover
    assert round(cover(first), 3) == 0.918 and {cover(name) for name in later} == {0.0}  # r000002: 0.918 for the first LoRA, 0.000 for the others
    # What the servers saw: 959 requests, of null M and of one LoRA only, and none of the 6 other LoRAs that each of them holds.
    assert Counter(name for port in PORTS for name, _ in servers.chats[port]) == {NULL_MODEL: 500, first: 459}
    assert [len(servers.chats[port]) for port in PORTS] == [121, 121, 121, 120, 119, 119, 119, 119]
    assert all(sorted(servers.loaded[port]) == sorted(world.names) for port in PORTS)
    assert (files.most, files.failed) == (limit, (rollouts - limit) * 3)  # 3,041 rollouts failed each of their 3 tries without a socket
    # The log line of that chunk cannot tell: the job list held all 8 models, and a try without a socket wrote no line.
    assert [(line["phase"], line["models"], line["rollouts"]) for line in world.lines()] == [("eval", 1 + N_LORAS, rollouts)]
    # Now the look asks first. The same 959 sockets are a failure of the box before any rollout: no LoRA gets an E.
    world.seam.room = lambda sockets, spare: limit
    with pytest.raises(InfraError, match="room for 959 sockets; a look holds 2048"):
        world.seam.evaluate(request)
    assert sum(len(chats) for chats in servers.chats.values()) == limit
