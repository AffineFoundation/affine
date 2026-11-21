"""Merged candidates are tested with exactly the bytes that install will fold."""

from __future__ import annotations

import dataclasses
import json
import os
import runpy
import shutil
import textwrap
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sn120_v2.canonical import manifest_digest
from sn120_v2.errors import InfraError
from sn120_v2.fold_kernel import FIXED_ORDER, TORCH_MATMUL, read_pairs, replay_fold
from sn120_v2.gpu.evaluator import server_argv
from sn120_v2.gpu.folder import fold_files, read_lora
from sn120_v2.gpu.merged import TestedBase as MergedBase
from sn120_v2.gpu.merged import assert_fold_matches, build_candidate_base
from sn120_v2.gpu.resident import hash_tree
from sn120_v2.gpu.solo import SoloRunner, SoloScheduler, SoloSettings, SoloWorker, choose_claim
from sn120_v2.sampling import rollout_seed
from sn120_v2.types import EntryResult, NOT_RUN, NULL_MODEL, PASS, SCORED


def tiny_model(root: Path, rank: int = 2):
    base, master, lora = root / "base", root / "master", root / "lora"
    base.mkdir(), master.mkdir(), lora.mkdir()
    tensors = {
        "model.layers.0.q_proj.weight": (torch.arange(48).reshape(6, 8) / 97).float(),
        "model.layers.0.down_proj.weight": (torch.arange(24).reshape(4, 6) / 53).float(),
    }
    save_file({name: weight.bfloat16() for name, weight in tensors.items()}, str(base / "model.safetensors"), metadata={"format": "pt"})
    save_file(tensors, str(master / "model.part00.safetensors"), metadata={"format": "pt"})
    (base / "config.json").write_text('{"model":"tiny"}\n', encoding="utf-8")
    a = torch.arange(rank * 8).reshape(rank, 8).float() / 1000
    b = torch.arange(6 * rank).reshape(6, rank).float() / 2000
    save_file({"base_model.model.model.layers.0.q_proj.lora_A.weight": a, "base_model.model.model.layers.0.q_proj.lora_B.weight": b}, str(lora / "adapter_model.safetensors"))
    (lora / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": 4}), encoding="utf-8")
    return master, base, lora, rank


@pytest.mark.parametrize("kernel", [FIXED_ORDER, TORCH_MATMUL])
def test_candidate_build_is_bit_identical_to_both_fold_implementations(tmp_path, kernel):
    master, base, lora, rank = tiny_model(tmp_path)
    tested = build_candidate_base(master, base, lora, rank, tmp_path / "tested", kernel=kernel, threads=1)
    # The install writer, including its fp32 master output.
    fold_files(master, base, [read_lora(lora, rank)], tmp_path / "install-master", tmp_path / "install-base")
    # The replay implementation used by the head chain and by candidate builds.
    replay = replay_fold(master, base, [read_pairs(lora, rank)], tmp_path / "replay-master", tmp_path / "replay-base", kernel=kernel, threads=1)
    assert tested.base_hash == manifest_digest(hash_tree(tmp_path / "install-base")) == replay.base_hash
    assert tested.base_files == hash_tree(tmp_path / "install-base") == replay.base_files
    assert all((tested.path / name).read_bytes() == (tmp_path / "install-base" / name).read_bytes() == (tmp_path / "replay-base" / name).read_bytes() for name in tested.base_files)
    assert not (tested.path / "master").exists() and len(list(tested.path.glob("*.safetensors"))) == 1


def test_install_hash_assertion_fails_closed(build):
    ticket = build.ticket(n_entries=1, params=build.params(eval_merged=True))
    artifact = build.artifact(ticket.head, ticket.entries[0], ticket.params)
    entry = EntryResult(0, SCORED, lora=artifact, outcomes=(PASS * 8,), tested_base_hash=build.h("merged"), tested_fold_kernel=FIXED_ORDER)
    assert_fold_matches(ticket.params, entry, build.h("merged"))
    with pytest.raises(InfraError, match="differs from the winner's tested base hash"):
        assert_fold_matches(ticket.params, entry, build.h("other"))
    with pytest.raises(InfraError, match="needs the tested base hash"):
        assert_fold_matches(ticket.params, dataclasses.replace(entry, tested_base_hash=""), build.h("merged"))
    with pytest.raises(InfraError, match="runtime-LoRA"):
        assert_fold_matches(build.params(), dataclasses.replace(entry, outcomes=(PASS * 8,)), build.h("merged"))


def _candidate_build_observer(shim: Path, observation: Path, fail: bool) -> None:
    """Observe the real builder, with CPU standing in for a driver."""
    shim.mkdir()
    package = Path(__file__).parents[1]
    (shim / "sitecustomize.py").write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(package)!r})
        import runpy
        import json

        import os
        from pathlib import Path

        from sn120_v2 import fold_kernel
        from sn120_v2.gpu import merged

        real_fold = fold_kernel.replay_fold
        def observe(*args, **kwargs):
            record = {{
                "pid": os.getpid(), "ppid": os.getppid(), "thread": __import__("threading").current_thread().name,
                "device": kwargs.get("device", "cpu"), "threads": kwargs.get("threads"), "kernel": kwargs.get("kernel"),
            }}
            Path({str(observation)!r}).write_text(json.dumps(record))
            assert kwargs.get("device") == "cuda:4"
            if {fail!r}:
                Path(kwargs["out_base"] if "out_base" in kwargs else args[4]).mkdir(parents=True, exist_ok=True)
                raise RuntimeError("driver failed after a partial candidate")
            kwargs["device"] = "cpu"  # no CUDA device exists in this unit-test environment
            return real_fold(*args, **kwargs)
        fold_kernel.replay_fold = observe
        real_build = merged.build_candidate_base
        def reserved(*args, **kwargs):
            assert kwargs["reserved"] == 37 and kwargs["threads"] == 1
            return real_build(*args, **kwargs)
        merged.build_candidate_base = reserved
        """), encoding="utf-8")
    # For a control run of the unpatched release, observe the already-running
    # parent too. The fixed test does not set this: only its spawned child is patched.
    if os.environ.get("SN120_TEST_OBSERVE_INLINE") == "1":
        runpy.run_path(str(shim / "sitecustomize.py"))


def _candidate_tiny(root: Path):
    root.mkdir()
    return tiny_model(root)


def _candidate_evaluator(tmp_path: Path, build, master: Path, base: Path, lora: Path):
    from sn120_v2.gpu.evaluator import VllmEvaluator
    from sn120_v2.gpu.plan import BoxConfig
    from sn120_v2.store import LocalStore

    ticket = build.ticket(n_entries=1, params=build.params(eval_merged=True))
    artifact = build.artifact(ticket.head, ticket.entries[0], ticket.params)
    artifact = dataclasses.replace(artifact, rank=2)
    paths = {ticket.head.base_hash: base, ticket.head.base_fp32_master_digest: master, artifact.lora_hash: lora}
    evaluator = VllmEvaluator(LocalStore(tmp_path / "store"), BoxConfig(tmp_path / "work", build.image), None, None)
    evaluator.capture = None
    evaluator.resident = SimpleNamespace(tree=lambda prefix, identity: paths[identity])
    return evaluator, ticket, artifact


def test_merged_build_runs_real_builder_in_a_spawned_child_and_releases_the_gpu(tmp_path, build, monkeypatch):
    master, base, lora, rank = _candidate_tiny(tmp_path / "tiny")
    expected = build_candidate_base(master, base, lora, rank, tmp_path / "inline", kernel=FIXED_ORDER, threads=1)
    observer = tmp_path / "observer"
    _candidate_build_observer(observer, tmp_path / "observation.json", False)
    monkeypatch.setenv("PYTHONPATH", str(observer))  # inherited by production's ordinary spawn boundary
    monkeypatch.setenv("SN120_HEAD_PUBLISH", "chain")
    monkeypatch.setenv("SN120_FOLD_THREADS", "1")
    monkeypatch.setenv("SN120_FOLD_TWICE", "0")
    evaluator, ticket, artifact = _candidate_evaluator(tmp_path, build, master, base, lora)
    evaluator._merged_reserved[ticket.round_id] = 37  # an overlapping build's reservation reaches the real child
    controller_pid = os.getpid()
    with pytest.raises(InfraError, match=r"GPUs \[4\] are still in use"):
        from sn120_v2.gpu.procs import wait_gpus_free
        wait_gpus_free(1024, 0.0, report=lambda: "4, 1316\n")  # controller_pid owns this simulated driver context

    def in_solo_gpu_thread():
        nonlocal tested
        tested = evaluator.build_merged(ticket.round_id, ticket.head, artifact, ticket.params, gpu=4)

    tested = None
    worker = threading.Thread(target=in_solo_gpu_thread, name="solo-gpu-4")
    worker.start(), worker.join(timeout=120)
    assert not worker.is_alive()
    observation = json.loads((tmp_path / "observation.json").read_text())
    assert tested is not None and observation["ppid"] == controller_pid and observation["pid"] != controller_pid
    assert observation["thread"] == "MainThread"  # the buggy builder instead runs in solo-gpu-4
    assert observation["device"] == "cuda:4" and observation["kernel"] == FIXED_ORDER and observation["threads"] == 1
    assert tested.base_hash == expected.base_hash and tested.base_files == expected.base_files
    assert all((tested.path / name).read_bytes() == (expected.path / name).read_bytes() for name in expected.base_files)
    assert evaluator._merged_reserved[ticket.round_id] == 37  # only this build's reservation is released
    # With the build child's PID/context gone, the same aggregate check passes.
    from sn120_v2.gpu.procs import wait_gpus_free
    wait_gpus_free(1024, 0.0, report=lambda: "\n".join(f"{gpu}, {6 if gpu != 4 else 0}" for gpu in range(8)))
    evaluator.cleanup_merged(ticket.round_id)
    assert not tested.path.exists() and not evaluator._merged_reserved


def test_spawned_merged_build_keeps_infra_error_and_partial_cleanup(tmp_path, build, monkeypatch):
    master, base, lora, rank = _candidate_tiny(tmp_path / "tiny")
    observer = tmp_path / "observer"
    _candidate_build_observer(observer, tmp_path / "observation.json", True)
    monkeypatch.setenv("PYTHONPATH", str(observer))
    monkeypatch.setenv("SN120_HEAD_PUBLISH", "chain")
    monkeypatch.setenv("SN120_FOLD_THREADS", "1")
    monkeypatch.setenv("SN120_FOLD_TWICE", "0")
    evaluator, ticket, artifact = _candidate_evaluator(tmp_path, build, master, base, lora)
    evaluator._merged_reserved[ticket.round_id] = 37  # an overlapping build's reservation reaches the real child
    with pytest.raises(InfraError, match="merged candidate build failed: RuntimeError: driver failed after a partial candidate"):
        evaluator.build_merged(ticket.round_id, ticket.head, artifact, ticket.params, gpu=4)
    assert not (tmp_path / "work" / "merged-candidates" / ticket.round_id / artifact.lora_hash).exists()
    assert evaluator._merged_reserved[ticket.round_id] == 37  # only this build's reservation is released


def test_spawned_merged_build_classifies_a_dead_child_as_infra_error(tmp_path, build, monkeypatch):
    master, base, lora, _ = tiny_model(tmp_path)
    evaluator, ticket, artifact = _candidate_evaluator(tmp_path, build, master, base, lora)
    observer = tmp_path / "observer"
    observer.mkdir()
    (observer / "sitecustomize.py").write_text("import os\nos._exit(73)\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(observer))
    monkeypatch.setenv("SN120_HEAD_PUBLISH", "chain")
    monkeypatch.setenv("SN120_FOLD_THREADS", "1")
    monkeypatch.setenv("SN120_FOLD_TWICE", "0")
    with pytest.raises(InfraError, match="merged candidate build failed: BrokenProcessPool"):
        evaluator.build_merged(ticket.round_id, ticket.head, artifact, ticket.params, gpu=4)
    assert evaluator._merged_reserved[ticket.round_id] == 0


def test_merged_build_checks_free_disk_and_never_serves_a_partial_directory(tmp_path, monkeypatch):
    master, base, lora, rank = tiny_model(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(InfraError, match="not enough free disk for merged candidate base"):
        build_candidate_base(master, base, lora, rank, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_runtime_lora_rounding_loses_the_small_update_that_merging_preserves():
    generator = torch.Generator().manual_seed(7)
    master = torch.randn(64, 64, generator=generator) * 0.02
    delta = torch.randn(64, 64, generator=generator)
    delta *= master.norm() * 1e-5 / delta.norm()
    x = torch.randn(128, 64, generator=generator)
    x = x.bfloat16()
    rounded_base = (x @ master.bfloat16().T).bfloat16()
    delta_x = x.float() @ delta.T
    old_change = (rounded_base + (x @ delta.bfloat16().T).bfloat16()).bfloat16().float() - rounded_base.float()
    bf16_master = master.bfloat16()
    merged_change = (x @ (master + delta).bfloat16().T).bfloat16().float() - rounded_base.float()

    def projection(change):
        return ((change * delta_x).sum() / delta_x.square().sum()).item()

    assert projection(old_change) < 0.10
    assert projection(merged_change) > 0.80


def planned_claims(params, loras):
    from sn120_v2.gpu.solo import SoloScheduler

    ticket = SimpleNamespace(params=params, draw=SimpleNamespace(task_ids=tuple(f"t{i}" for i in range(params.n_tasks))))
    runner = SimpleNamespace(settings=SoloSettings(steal_block_tasks=2), evaluator=SimpleNamespace(http=None))
    scheduler = SoloScheduler(runner, ticket, 10_000)
    for pos, lora in enumerate(loras):
        scheduler.add_lora(pos, lora, gpu=pos, tested=SimpleNamespace(path=Path(__file__).parent) if params.eval_merged else None)
    scheduler.training_finished()
    plan = []
    for gpu in range(8):
        owner = scheduler.model_for_gpu(gpu)
        while (claim := scheduler.steal(gpu, owner)) is not None:
            model, start, stop = claim
            plan.extend((model, task, sample, rollout_seed("seed-0", task, sample)) for task in ticket.draw.task_ids[start:stop] for sample in range(params.null_samples if model == NULL_MODEL else params.samples_per_task))
    return Counter(plan)


def test_eight_gpu_six_candidate_merged_plan_matches_lora_and_null_stays_on_idle_gpus(build):
    params = build.params(n_gpus=8, n_tasks=24, null_samples_per_task=2, eval_merged=True)
    ticket = build.ticket(n_entries=6, n_tasks=24, params=params)
    loras = [build.artifact(ticket.head, entry, params) for entry in ticket.entries]
    merged = planned_claims(params, loras)
    runtime = planned_claims(dataclasses.replace(params, eval_merged=False), loras)
    assert merged == runtime
    assert len(merged) == 24 * (6 + 2)
    remaining = {NULL_MODEL: 0, **{lora.lora_hash: 0 for lora in loras}}
    owner = loras[0].lora_hash
    assert choose_claim(remaining, set(remaining) - {NULL_MODEL}, 24, 2, owner, own_first=True) == (owner, 0, 2)
    assert choose_claim(remaining, set(remaining) - {NULL_MODEL}, 24, 2, NULL_MODEL, own_first=True) == (NULL_MODEL, 0, 2)


class MergedServers:
    """A server accepts only its bound merged model, or null on idle GPUs."""

    def __init__(self):
        self.argv = []
        self.bound = {}
        self.up = set()
        self.bodies = []
        self.first_model = {}
        self.models_bound = set()

    def spawn(self, argv, env, log):
        port = int(argv[argv.index("--port") + 1])
        model = argv[argv.index("--served-model-name") + 1]
        self.argv.append(list(argv))
        self.first_model.setdefault(port - 8100, model)
        self.bound[port] = model
        self.models_bound.add(model)
        self.up.add(port)
        return SimpleNamespace(alive=lambda: port in self.up, stop=lambda: self.up.discard(port))

    def http(self, url, body, timeout_s):
        port = int(url.split(":")[2].split("/")[0])
        if port not in self.up:
            raise ConnectionRefusedError(url)
        if url.endswith("/v1/chat/completions"):
            assert body["model"] == self.bound[port]
            self.bodies.append((port, dict(body)))
            return {"choices": [{"message": {"content": PASS}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}
        return {}


def test_real_solo_runner_serves_merged_models_and_cleans_round_dirs(build, tmp_path, monkeypatch):
    from sn120_v2.gpu import plan
    from sn120_v2.gpu.evaluator import ServeSettings, VllmEvaluator
    from sn120_v2.store import LocalStore

    params = build.params(n_gpus=8, n_tasks=24, null_samples_per_task=2, looks=(1.0,), eval_chunk_tasks=4, eval_merged=True, batch_cap=6)
    ticket = dataclasses.replace(build.ticket(n_entries=6, n_tasks=24, params=params), issued_at=int(__import__("time").time()), expires_at=int(__import__("time").time()) + 5000)
    servers = MergedServers()
    pool = SimpleNamespace(task=lambda task: SimpleNamespace(messages=lambda: [{"role": "user", "content": task}]))
    grader = SimpleNamespace(grade_many=lambda jobs: [SimpleNamespace(code=PASS, machine_cause=None) for _ in jobs], close=lambda: None)
    evaluator = VllmEvaluator(LocalStore(tmp_path / "store"), plan.BoxConfig(tmp_path / "work", build.image), pool, grader, spawn=servers.spawn, http=servers.http, gpus_free=lambda limit, timeout: None, settings=ServeSettings(start_timeout_s=1, clients_per_server=2), sleep=lambda _: None, room=lambda sockets, spare: sockets)
    evaluator.capture = None
    evaluator.resident = SimpleNamespace(tree=lambda prefix, identity: tmp_path / "king-base", prune=lambda keep: None)
    (tmp_path / "king-base").mkdir()
    monkeypatch.setattr("sn120_v2.gpu.evaluator.nvidia_smi", lambda: "\n".join(f"{gpu}, 0" for gpu in range(8)))
    tested = {}

    def merge(round_id, head, lora, params, gpu):
        path = tmp_path / "work" / "merged-candidates" / round_id / lora.lora_hash
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(lora.lora_hash.encode())
        made = MergedBase(path, build.h(lora.lora_hash), {"model.safetensors": build.h(lora.lora_hash)}, FIXED_ORDER)
        tested[lora.lora_hash] = made
        return made

    evaluator.build_merged = merge
    evaluator._load_one = lambda *args: (_ for _ in ()).throw(AssertionError("merged evaluation loaded a runtime LoRA"))
    trainer = SimpleNamespace(train=lambda request: build.artifact(ticket.head, request.entry, params), close=lambda: None)
    training = __import__("threading").Barrier(6)

    def train(request):
        training.wait(timeout=10)
        return build.artifact(ticket.head, request.entry, params)

    trainer.train = train
    result = SoloRunner(trainer, evaluator, __import__("time").time, build.image, settings=SoloSettings(steal_block_tasks=2, clients_while_training=2), poll_s=0.01).run(ticket)
    assert all(entry.status == SCORED and entry.outcomes == (PASS * 24,) and entry.tested_base_hash for entry in result.entries)
    assert result.null_outcomes == (PASS * 24, PASS * 24)
    assert all("--enable-lora" not in argv and "--max-loras" not in argv for argv in servers.argv)
    assert [servers.first_model[gpu] for gpu in (6, 7)] == [NULL_MODEL, NULL_MODEL]
    assert {servers.first_model[gpu] for gpu in range(6)} <= set(tested)
    assert set(tested) <= servers.models_bound
    planned = {(body["model"], body["messages"][-1]["content"], body["seed"]) for _, body in servers.bodies}
    assert len(planned) == 24 * 8
    assert not (tmp_path / "work" / "merged-candidates" / ticket.round_id).exists()


def test_merged_server_switch_waits_for_inflight_requests(build):
    import threading

    params = build.params(n_gpus=1, n_tasks=8, eval_merged=True, looks=(1.0,), eval_chunk_tasks=4)
    ticket = build.ticket(n_entries=1, n_tasks=8, params=params)
    lora = build.artifact(ticket.head, ticket.entries[0], params)
    scheduler = SimpleNamespace(trained={0: lora}, owner_by_model={}, merged_dir=lambda model: Path(__file__).parent, training_done=True, runner=SimpleNamespace(settings=SoloSettings()))
    starts, active, release = [], set(), threading.Event()
    evaluator = SimpleNamespace(_solo_clients=2, _solo_scheduler=scheduler, settings=SimpleNamespace(clients_per_server=2), clients_for=lambda params: 2, http=None, _solo_servers={0: SimpleNamespace(alive=lambda: True)})
    evaluator._serve_one = lambda *args, **kwargs: starts.append((kwargs["model"], len(active), release.is_set())) or f"http://{kwargs['model']}"
    evaluator._grade = lambda tasks, jobs, replies, request: {job: PASS for job in jobs}

    def rollout(request, url, job, written):
        active.add(job)
        assert release.wait(timeout=10), "the null request must drain before the merged server switches"
        active.remove(job)
        return PASS, False

    evaluator._rollout = rollout
    worker = SoloWorker(evaluator, ticket, scheduler.trained, 10_000, 0)
    answers, claims = [], [(NULL_MODEL, 0, 1), (lora.lora_hash, 1, 3)]
    thread = threading.Thread(target=lambda: worker.run(steal=lambda gpu, owner: claims.pop(0) if claims else None, result=lambda *job: answers.append(job), judge=lambda *args: None, alive=lambda: True, deadline=0))
    thread.start()
    deadline = __import__("time").monotonic() + 10
    while len(active) < 1 and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.001)
    assert len(active) == 1 and len(starts) == 1
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert starts == [(NULL_MODEL, 0, False), (lora.lora_hash, 0, True)]
    assert answers[0] == (NULL_MODEL, 0, 0, PASS)
    assert sorted(answers[1:]) == [(lora.lora_hash, 0, 1, PASS), (lora.lora_hash, 0, 2, PASS)]


def test_merged_deadline_keeps_the_paired_complete_block_prefix(build):
    from sn120_v2.judging import check_result

    params = build.params(n_gpus=1, n_tasks=8, eval_merged=True, looks=(0.5, 1.0), eval_chunk_tasks=4, null_samples_per_task=2)
    ticket = build.ticket(n_entries=1, n_tasks=8, params=params)
    lora = build.artifact(ticket.head, ticket.entries[0], params)
    runner = SimpleNamespace(settings=SoloSettings(steal_block_tasks=1), evaluator=SimpleNamespace(http=None), clock=lambda: 1100, image_digest=build.image)
    scheduler = SoloScheduler(runner, ticket, 10_000)
    runner.evaluator._solo_scheduler = scheduler
    scheduler.add_lora(0, lora, gpu=0, tested=MergedBase(Path(__file__).parent, build.h("merged"), {}, FIXED_ORDER))
    scheduler.training_finished()
    for index in range(6):
        for model in (NULL_MODEL, lora.lora_hash):
            for sample in range(2 if model == NULL_MODEL else 1):
                scheduler.result(model, sample, index, PASS)
    scheduler.result(NULL_MODEL, 0, 6, PASS)
    scheduler.deadline_cut()
    outcome = scheduler.finish()
    result = SoloRunner._solo_result(runner, ticket, {0: lora}, *outcome, 1000, 1100)
    assert result.stopped_at_limit
    assert result.null_outcomes == (PASS * 4 + NOT_RUN * 4,) * 2
    assert result.entries[0].outcomes == (PASS * 4 + NOT_RUN * 4,)
    assert result.entries[0].tested_base_hash == build.h("merged")
    check_result(ticket, result)



def test_merged_publication_uses_the_real_snapshot_and_explains_both_paths(tmp_path):
    from sn120_v2 import layout, publisher
    from sn120_v2.canonical import sha256_hex
    from sn120_v2.sim import SimWorld, sim_params

    class MemStore(dict):
        put, get, exists = dict.__setitem__, dict.__getitem__, dict.__contains__
        sha256 = lambda self, key: sha256_hex(self[key])

    world = SimWorld(tmp_path / "world", sim_params(eval_merged=True))
    store = MemStore()
    store[layout.head_manifest(world.controller.state.head.version_id)] = b"{}"
    publisher.StorePublisher(store, "https://files.example/").publish(world.publisher.snapshots[-1])
    text = " ".join(store[layout.LLMS_TXT].decode().split())
    doc = json.loads(store[layout.MINING_JSON])
    assert doc["contract"]["candidate_evaluation"]["tested_equals_installed"]
    for text in (text, doc["contract"]["candidate_evaluation"]["explanation"]):
        assert "Each candidate is tested as the merged model round_bf16(fp32 king master + your LoRA update), built with the same arithmetic as the install fold; the model that is tested is bit-for-bit the model that is installed." in text
        assert "Before this change the LoRA was applied at serving time on top of the rounded bf16 king, which rounded most of a small update away." in text
    assert "candidate_evaluation" not in publisher.contract(world.params.__class__(), "", "")


@pytest.mark.parametrize("mode,order,kernel", [("full", "serial", TORCH_MATMUL), ("chain", "serial", FIXED_ORDER), ("chain", "overlap", FIXED_ORDER)])
def test_real_controller_installs_the_tested_base_and_rejects_a_mismatch(tmp_path, mode, order, kernel):
    from sn120_v2 import layout
    from sn120_v2.headfold import FoldSettings, inline
    from sn120_v2.sim import sim_params
    from sn120_v2.sim_head import ChainWorld
    from sn120_v2.state import ABANDONED, APPLIED, HEAD_APPLIED

    params = sim_params(eval_merged=True)
    world = ChainWorld(tmp_path / f"{mode}-{order}", params, mode=mode, order=order, settings=FoldSettings(twice=False))
    if mode == "chain":
        world.start(mode)
    evaluated = []

    def with_proof(ticket):
        result = world.runner.__class__.run(world.runner, ticket)
        entries = []
        for entry in result.entries:
            if entry.status == SCORED:
                lora_dir = world.folder.resident.tree(entry.lora.prefix, entry.lora.lora_hash)
                built = build_candidate_base(world.local(world.head.head.base_fp32_master_digest), world.local(world.head.head.base_hash), lora_dir, entry.lora.rank, tmp_path / mode / f"candidate-{len(evaluated)}", kernel=kernel, threads=1)
                entry = dataclasses.replace(entry, tested_base_hash=built.base_hash, tested_fold_kernel=kernel)
                evaluated.append(entry)
            entries.append(entry)
        return dataclasses.replace(result, entries=tuple(entries))

    world.runner.run = with_proof
    world.folder.resident.tree(layout.public_base_prefix(world.head.head.base_hash), world.head.head.base_hash)
    world.folder.resident.tree(layout.public_master_prefix(world.head.head.base_fp32_master_digest), world.head.head.base_fp32_master_digest)
    manifest = world.king()
    assert evaluated[0].tested_base_hash == manifest.head.base_hash
    before = world.state.head.version_id
    world.register(world.uid, 0.3)
    real = world.runner.run

    def wrong(ticket):
        result = real(ticket)
        return dataclasses.replace(result, entries=tuple(dataclasses.replace(entry, tested_base_hash="0" * 64) if entry.status == SCORED else entry for entry in result.entries))

    world.runner.run = wrong
    assert world.round() is None
    assert not world.state.head.folded and world.state.head.version_id != before
    assert not world.store.list(f"pending/folds/{world.state.head.version_id}/")
    assert "differs from the winner's tested base hash" in world.controller.last_error
    world.now += world.params.apply_retry_s
    world.now += world.params.apply_retry_s
    for _ in range(2):
        assert world.round() is None
        world.now += world.params.apply_retry_s
    assert world.state.round is not None and world.state.round.applied
    assert not world.state.head.folded and not world.state.round.outcome
    assert world.store.exists(layout.sealed_result(world.state.round.round_id, world.state.round.leases[0].lease_id))
    assert "differs from the winner's tested base hash" in world.controller.last_error
    world.wait()


def test_applied_winner_recovers_its_tested_proof_for_a_deferred_fold(tmp_path):
    from sn120_v2 import layout
    from sn120_v2.headfold import FoldSettings, inline
    from sn120_v2.sim import sim_params
    from sn120_v2.sim_head import ChainWorld
    from sn120_v2.state import APPLIED, HEAD_APPLIED

    world = ChainWorld(tmp_path / "world", sim_params(eval_merged=True), mode="chain", settings=FoldSettings(twice=False))
    world.start("chain")
    proof = []

    def with_proof(ticket):
        result = world.runner.__class__.run(world.runner, ticket)
        entries = []
        for entry in result.entries:
            if entry.status == SCORED:
                lora_dir = world.folder.resident.tree(entry.lora.prefix, entry.lora.lora_hash)
                built = build_candidate_base(world.local(world.head.head.base_fp32_master_digest), world.local(world.head.head.base_hash), lora_dir, entry.lora.rank, tmp_path / "candidate", kernel=FIXED_ORDER, threads=1)
                entry = dataclasses.replace(entry, tested_base_hash=built.base_hash, tested_fold_kernel=FIXED_ORDER)
                proof.append(entry)
            entries.append(entry)
        return dataclasses.replace(result, entries=tuple(entries))

    world.runner.run = with_proof
    world.folder.run = lambda jobs: (_ for _ in ()).throw(InfraError("deferred test fold"))
    world.register(2, 0.3)
    assert world.round() is None
    stacked = world.state.head
    assert not stacked.folded and world.controller.journal.records[-1].kind == HEAD_APPLIED
    world.folder.run = inline
    world.now += world.params.apply_retry_s
    world.start()
    assert world.round() is None
    assert world.state.rounds[-1].outcome == APPLIED and world.state.round is None
    assert world.state.head.parent_version_id == stacked.version_id
    assert world.state.head.base_hash == proof[0].tested_base_hash
    assert not world.store.list("pending/")
    world.wait()


def test_eval_merged_off_keeps_old_server_argv_and_params_hash(build):
    import subprocess

    params = build.params()
    # Compare with the release implementation, not this branch's own HEAD.
    source = subprocess.check_output(["git", "show", "4d3379545a3fab9836f193b40314f1a43d2dfa7e:v2/sn120_v2/gpu/evaluator.py"], text=True)
    namespace = dict(__name__="old_evaluator", sys=__import__("sys"), Params=params.__class__, NULL_MODEL=NULL_MODEL, ContractError=ValueError, lora_table=lambda params, batches=0: batches or params.max_loras_per_box)
    prefix = source[source.index("def server_argv("):source.index("\n\ndef server_env(")]
    exec(prefix, namespace)
    argv = server_argv("/base", 8100, params)
    assert argv == namespace["server_argv"]("/base", 8100, params)
    assert dataclasses.replace(params, eval_merged=False).params_hash == params.params_hash
    assert "eval_merged" not in params.to_dict()
