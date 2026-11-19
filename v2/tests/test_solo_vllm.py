"""Solo vLLM lifecycle with a fake HTTP server and real request bodies."""

from __future__ import annotations

import dataclasses
import json
import os
import resource
import socket
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2.gpu.evaluator import HttpError, ServeSettings, VllmEvaluator, server_argv
from sn120_v2.gpu.solo import SoloRunner, SoloSettings, SoloWorker
from sn120_v2.gpu import plan
from sn120_v2.sampling import rollout_seed
from sn120_v2.store import LocalStore
from sn120_v2.types import NULL_MODEL, HeadVersion
from sn120_v2 import layout
from sn120_v2.canonical import manifest_digest, sha256_hex


class FakeServers:
    def __init__(self):
        self.lock = threading.Lock()
        self.up = set()
        self.loaded = {}
        self.bodies = []
        self.argv = {}

    def spawn(self, argv, env, log):
        port = int(argv[argv.index("--port") + 1])
        self.up.add(port)
        self.argv[port] = list(argv)
        self.loaded.setdefault(port, set())
        return SimpleNamespace(alive=lambda: port in self.up, stop=lambda: self.up.discard(port))

    def http(self, url, body, timeout_s):
        port = int(url.split(":")[2].split("/")[0])
        if port not in self.up:
            raise ConnectionRefusedError(url)
        with self.lock:
            if url.endswith("/health"):
                return {}
            if url.endswith("/v1/models"):
                return {"data": [{"id": NULL_MODEL}]}
            if url.endswith("/v1/load_lora_adapter"):
                self.loaded[port].add(body["lora_name"])
                return {}
            if url.endswith("/v1/chat/completions"):
                if body["model"] != NULL_MODEL and body["model"] not in self.loaded[port]:
                    raise HttpError(404)
                self.bodies.append((port, body))
                return {"choices": [{"message": {"content": "P"}, "finish_reason": "stop"}]}
        return {}


def test_solo_gpu_free_check_ignores_a_busy_peer(build, tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    base = manifest_digest({"config.json": sha256_hex(b"{}")})
    master = manifest_digest({"model.part00.safetensors": sha256_hex(b"master")})
    store.put(f"{layout.public_base_prefix(base)}/config.json", b"{}")
    store.put(f"{layout.public_master_prefix(master)}/model.part00.safetensors", b"master")
    head = HeadVersion.make(parent_version_id=None, base_hash=base, base_fp32_master_digest=master, loras=(), round_id="genesis", applied_at=1)
    params = build.params(n_gpus=2)
    farm = FakeServers()
    pool = SimpleNamespace(task=lambda task: SimpleNamespace(messages=lambda: [{"role": "user", "content": task}]))
    grader = SimpleNamespace(grade_many=lambda jobs: [SimpleNamespace(code="P", machine_cause=None) for _ in jobs])
    evaluator = VllmEvaluator(store, plan.BoxConfig(tmp_path / "work", build.image), pool, grader, spawn=farm.spawn, http=farm.http, settings=ServeSettings(start_timeout_s=10.0), sleep=lambda _: None)
    monkeypatch.setattr("sn120_v2.gpu.evaluator.nvidia_smi", lambda: "0, 0\n1, 50000\n")
    calls = []
    from sn120_v2.gpu.procs import wait_gpus_free

    def check(limit, timeout, report, sleep):
        calls.append(report())
        assert timeout == evaluator.settings.free_timeout_s
        return wait_gpus_free(limit, timeout, report, sleep)

    monkeypatch.setattr("sn120_v2.gpu.evaluator.wait_gpus_free", check)
    assert evaluator._serve_one(head, params, 0, 0).endswith(":8100")
    assert calls == ["0, 0"]
    evaluator.close()


def test_solo_servers_prewarm_and_keep_paired_requests(build, tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    base = manifest_digest({"config.json": sha256_hex(b"{}")})
    master = manifest_digest({"model.part00.safetensors": sha256_hex(b"master")})
    store.put(f"{layout.public_base_prefix(base)}/config.json", b"{}")
    store.put(f"{layout.public_master_prefix(master)}/model.part00.safetensors", b"master")
    head = HeadVersion.make(parent_version_id=None, base_hash=base, base_fp32_master_digest=master, loras=(), round_id="genesis", applied_at=1)
    params = build.params(n_tasks=8, n_gpus=2, looks=(1.0,), eval_chunk_tasks=4)
    ticket = dataclasses.replace(build.ticket(n_entries=2, n_tasks=8, params=params), head=head, issued_at=int(time.time()), expires_at=int(time.time()) + 5000)
    loras = {}
    for entry in ticket.entries:
        files = {"adapter_config.json": b"{}", "adapter_model.safetensors": str(entry.queue_pos).encode()}
        identity = manifest_digest({name: sha256_hex(data) for name, data in files.items()})
        for name, data in files.items():
            store.put(f"pending/loras/{identity}/{name}", data)
        artifact = build.artifact(head, entry, params)
        loras[entry.queue_pos] = dataclasses.replace(artifact, lora_hash=identity, prefix=f"pending/loras/{identity}", files={name: sha256_hex(data) for name, data in files.items()})
    farm = FakeServers()
    pool = SimpleNamespace(task=lambda task: SimpleNamespace(messages=lambda: [{"role": "user", "content": task}]))
    grader = SimpleNamespace(grade_many=lambda jobs: [SimpleNamespace(code="P", machine_cause=None) for _ in jobs])
    evaluator = VllmEvaluator(store, plan.BoxConfig(tmp_path / "work", build.image), pool, grader, spawn=farm.spawn, http=farm.http, gpus_free=lambda limit, timeout: None, settings=ServeSettings(start_timeout_s=10.0, clients_per_server=8), sleep=lambda _: None, room=lambda sockets, spare: sockets + spare)
    # Solo uses the real _serve_one path with the normal server argv.
    monkeypatch.setattr("sn120_v2.gpu.evaluator.nvidia_smi", lambda: "0, 0\n1, 0\n")
    trainer = SimpleNamespace(train=lambda request: loras[request.entry.queue_pos], close=lambda: None)
    runner = SoloRunner(trainer, evaluator, lambda: int(time.time()), build.image, settings=SoloSettings(prewarm=True, steal_block_tasks=2), poll_s=0.01)
    budget_calls = []
    budget = evaluator.solo_client_budget

    def check_budget(gpus, max_clients):
        budget_calls.append((gpus, max_clients, len(farm.argv)))
        return budget(gpus, max_clients)

    evaluator.solo_client_budget = check_budget
    result = runner.run(ticket)
    assert budget_calls == [(2, 8, 0)], "the solo budget is set before any prewarm server starts"
    assert result.null_outcomes == ("P" * 8,)
    assert len(farm.up) == 0  # close() tears down all solo servers
    assert all("--gpu-memory-utilization" not in argv for argv in farm.argv.values())
    assert all(argv == server_argv(argv[argv.index("--model") + 1], int(argv[argv.index("--port") + 1]), params, 8, len(ticket.entries)) for argv in farm.argv.values())
    by_task = {}
    for port, body in farm.bodies:
        task = body["messages"][-1]["content"]
        assert body["seed"] == rollout_seed(ticket.draw.round_seed, task, 0)
        by_task.setdefault(task, []).append((port, body))
    assert len(by_task) == 8
    assert all(len({body["model"] for _, body in pairs}) == 3 for pairs in by_task.values())
    phase = json.loads((tmp_path / "work" / "logs" / "phases.jsonl").read_text().splitlines()[-1])
    assert phase["phase"] == "solo_eval" and phase["clients_per_server"] == 8
    assert phase["configured_clients_per_server"] == 8 and phase["open_files_limit"] == resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def test_solo_server_spawns_and_health_waits_do_not_block_other_gpus(build, tmp_path):
    seam = VllmEvaluator.__new__(VllmEvaluator)
    seam.settings = ServeSettings(start_timeout_s=10.0, first_port=8100)
    seam._solo_lock = threading.RLock()
    seam._solo_start_locks = {}
    seam._solo_servers, seam._solo_urls, seam._solo_loaded = {}, {}, {}
    seam._solo_server_clients, seam._solo_clients = {}, None
    seam._head, seam._table = None, 0
    seam.config = SimpleNamespace(work_dir=tmp_path)
    seam.resident = SimpleNamespace(tree=lambda prefix, identity: tmp_path / "base")
    seam._wait_one_gpu_free = lambda gpu: None
    waiting, spawned = set(), []
    gate = threading.Event()
    spawn_gate = threading.Event()

    def spawn(argv, env, log):
        gpu = int(argv[argv.index("--port") + 1]) - 8100
        spawned.append(gpu)
        if len(spawned) == 2:
            spawn_gate.set()
        assert spawn_gate.wait(timeout=5), "a server spawn held the shared lock"
        return SimpleNamespace(alive=lambda: True, stop=lambda: None)

    def http(url, body, timeout):
        gpu = int(url.split(":")[2].split("/")[0]) - 8100
        waiting.add(gpu)
        if len(waiting) == 2:
            gate.set()
        assert gate.wait(timeout=5), "a health wait held the shared lock"
        return {}

    seam.spawn, seam.http = spawn, http
    seam.sleep = lambda _: None
    params = build.params(n_gpus=2)
    with ThreadPoolExecutor(2) as pool:
        urls = list(pool.map(lambda gpu: seam._serve_one(build.head(), params, 2, gpu), range(2)))
    assert urls == ["http://127.0.0.1:8100", "http://127.0.0.1:8101"]
    assert set(spawned) == set(waiting) == {0, 1}


def test_loaded_lora_short_circuit_does_not_take_the_solo_lock(build):
    seam = VllmEvaluator.__new__(VllmEvaluator)
    artifact = build.artifact(build.head(), build.entry(0), build.params())
    seam._solo_loaded = {0: {artifact.lora_hash}}

    class Locked:
        def __enter__(self):
            raise AssertionError("an already-loaded LoRA took the shared lock")

        def __exit__(self, *args):
            pass

    seam._solo_lock = Locked()
    seam._call = lambda *args: (_ for _ in ()).throw(AssertionError("an already-loaded LoRA was loaded again"))
    seam._load_one(0, artifact)


@pytest.mark.parametrize("limit", [256, 512])
def test_solo_params_drive_server_and_client_limits(build, limit):
    params = build.params(max_num_seqs=limit, kv_cache_dtype="fp8")
    seam = VllmEvaluator.__new__(VllmEvaluator)
    seam.settings = ServeSettings()
    argv = server_argv("/base", 8100, params)
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"
    assert argv[argv.index("--max-num-seqs") + 1] == str(limit)
    assert seam.clients_for(params) == limit
    ticket = build.ticket(n_entries=1, params=params)
    evaluator = SimpleNamespace(_solo_clients=None, _solo_scheduler=None, clients_for=seam.clients_for, settings=ServeSettings())
    worker = SoloWorker(evaluator, ticket, {}, ticket.issued_at, 0)
    assert worker.clients == limit


def test_default_server_command_omits_kv_cache_dtype_and_preserves_256(build):
    params = build.params()
    argv = server_argv("/base", 8100, params)
    assert "--kv-cache-dtype" not in argv
    assert argv[argv.index("--max-num-seqs") + 1] == "256"


def test_solo_client_budget_raises_1024_before_opening_sockets(tmp_path):
    """The solo path must not depend on cmd_run having raised NOFILE."""
    script = textwrap.dedent(
        """
        import json, os, resource, socket, sys
        sys.path.insert(0, "@V2@")
        from types import SimpleNamespace
        from sn120_v2.gpu.evaluator import ServeSettings, VllmEvaluator, socket_room

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard < 4736:
            raise SystemExit(77)
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))
        seam = VllmEvaluator.__new__(VllmEvaluator)
        seam.settings = ServeSettings()
        seam.room = socket_room
        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == 1024
        clients = seam.solo_client_budget(8, 512)
        raised = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        held = []
        try:
            for _ in range(clients * 8):
                held.append(socket.socket())
            error = None
        except OSError as exc:
            error = exc.errno
        finally:
            for item in held:
                item.close()
        print(json.dumps({"clients": clients, "soft": raised, "held": len(held), "error": error, "baseline": len(os.listdir("/proc/self/fd"))}))
        """
    ).replace("@V2@", str(Path(__file__).resolve().parents[1]))
    done = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=120, cwd=os.fspath(tmp_path))
    if done.returncode == 77:
        pytest.skip("hard NOFILE is too low for the normal solo client budget")
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["clients"] == 512 and report["soft"] >= 4096
    assert report["held"] == 4096 and report["error"] is None


def test_solo_budget_caps_all_servers_and_leaves_admin_and_file_room():
    seam = VllmEvaluator.__new__(VllmEvaluator)
    seam.settings = ServeSettings(spare_files=512, admin_sockets_per_server=8)
    calls = []
    seam.room = lambda sockets, spare: calls.append((sockets, spare)) or 512
    assert seam.solo_client_budget(8, 512) == 56
    assert calls == [(4160, 576)]
    assert 8 * (56 + 8) <= 512
    seam.room = lambda sockets, spare: 1560
    assert seam.solo_client_budget(3, 512) == 512  # spare capacity does not exceed the configured client count
    seam.room = lambda sockets, spare: 64
    with pytest.raises(Exception, match="open file limit"):
        seam.solo_client_budget(8, 512)


def test_solo_round_applies_one_client_cap_to_every_server(build, tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    base = manifest_digest({"config.json": sha256_hex(b"{}")})
    master = manifest_digest({"model.part00.safetensors": sha256_hex(b"master")})
    store.put(f"{layout.public_base_prefix(base)}/config.json", b"{}")
    store.put(f"{layout.public_master_prefix(master)}/model.part00.safetensors", b"master")
    head = HeadVersion.make(parent_version_id=None, base_hash=base, base_fp32_master_digest=master, loras=(), round_id="genesis", applied_at=1)
    params = build.params(n_tasks=8, n_gpus=8, looks=(1.0,), eval_chunk_tasks=4)
    ticket = dataclasses.replace(build.ticket(n_entries=1, n_tasks=8, params=params), head=head, issued_at=int(time.time()), expires_at=int(time.time()) + 5000)
    artifact = build.artifact(head, ticket.entries[0], params)
    files = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"lora"}
    identity = manifest_digest({name: sha256_hex(data) for name, data in files.items()})
    for name, data in files.items():
        store.put(f"pending/loras/{identity}/{name}", data)
    artifact = dataclasses.replace(artifact, lora_hash=identity, prefix=f"pending/loras/{identity}", files={name: sha256_hex(data) for name, data in files.items()})
    farm = FakeServers()
    pool = SimpleNamespace(task=lambda task: SimpleNamespace(messages=lambda: [{"role": "user", "content": task}]))
    grader = SimpleNamespace(grade_many=lambda jobs: [SimpleNamespace(code="P", machine_cause=None) for _ in jobs])
    evaluator = VllmEvaluator(store, plan.BoxConfig(tmp_path / "work", build.image), pool, grader, spawn=farm.spawn, http=farm.http, gpus_free=lambda limit, timeout: None, sleep=lambda _: None, room=lambda sockets, spare: 512)
    monkeypatch.setattr("sn120_v2.gpu.evaluator.nvidia_smi", lambda: "\n".join(f"{gpu}, 0" for gpu in range(8)))
    trainer = SimpleNamespace(train=lambda request: artifact, close=lambda: None)
    runner = SoloRunner(trainer, evaluator, lambda: int(time.time()), build.image, settings=SoloSettings(prewarm=True, steal_block_tasks=1), poll_s=0.01)
    result = runner.run(ticket)
    assert result.null_outcomes == ("P" * 8,)
    assert len(farm.argv) == 8
    assert {argv[argv.index("--max-num-seqs") + 1] for argv in farm.argv.values()} == {"56"}
    assert 8 * (56 + evaluator.settings.admin_sockets_per_server) <= 512
    phase = json.loads((tmp_path / "work" / "logs" / "phases.jsonl").read_text().splitlines()[-1])
    assert (phase["phase"], phase["n_gpus"], phase["clients_per_server"], phase["configured_clients_per_server"]) == ("solo_eval", 8, 56, 256)
    assert phase["open_files_limit"] == resource.getrlimit(resource.RLIMIT_NOFILE)[0]
