"""Solo vLLM lifecycle with a fake HTTP server and real request bodies."""

from __future__ import annotations

import dataclasses
import threading
import time
from types import SimpleNamespace

from sn120_v2.gpu.evaluator import HttpError, ServeSettings, VllmEvaluator, server_argv
from sn120_v2.gpu.solo import SoloRunner, SoloSettings
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
    evaluator = VllmEvaluator(store, plan.BoxConfig(tmp_path / "work", build.image), pool, grader, spawn=farm.spawn, http=farm.http, gpus_free=lambda limit, timeout: None, settings=ServeSettings(start_timeout_s=10.0, clients_per_server=8), sleep=lambda _: None, room=lambda sockets, spare: sockets)
    # Solo uses the real _serve_one path with the normal server argv.
    monkeypatch.setattr("sn120_v2.gpu.evaluator.nvidia_smi", lambda: "0, 0\n1, 0\n")
    trainer = SimpleNamespace(train=lambda request: loras[request.entry.queue_pos], close=lambda: None)
    runner = SoloRunner(trainer, evaluator, lambda: int(time.time()), build.image, settings=SoloSettings(prewarm=True, steal_block_tasks=2), poll_s=0.01)
    result = runner.run(ticket)
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
