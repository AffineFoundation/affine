"""The GPU side without a GPU and without torch: checks, seeds, outcome codes, processes and the cache."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import manifest_digest, sha256_hex
from sn120_v2.errors import BatchRejected, ContractError, InfraError, VerifyError
from sn120_v2.gpu import batch, plan, procs
from sn120_v2.gpu.evaluator import HttpError, ServeSettings, VllmEvaluator, chat_body, read_reply, server_argv, server_env
from sn120_v2.gpu import resident as resident_module
from sn120_v2.gpu.resident import Resident, hash_tree, resident_head, upload_tree
from sn120_v2.gpu.trainer import LORA_FILES, LoraTrainer
from sn120_v2.params import Recipe
from sn120_v2.sampling import rollout_seed
from sn120_v2.store import LocalStore
from sn120_v2.types import NULL_MODEL, Entry, EvalRequest, FrozenBatch, HeadVersion, LoRARef, TrainRequest

GROUPS = [["p1", [["good", 1.0], ["bad", 0.0]]], ["p2", [["alone", 1.0]]]]


def put_tree(store, prefix_of, files: dict[str, bytes]) -> str:
    identity = manifest_digest({name: sha256_hex(data) for name, data in files.items()})
    for name, data in files.items():
        store.put(f"{prefix_of(identity)}/{name}", data)
    return identity


@pytest.fixture
def box(tmp_path, build):
    """A store that holds a head (base and fp32 master) and one frozen batch, plus the config of a box."""
    store = LocalStore(tmp_path / "store")
    base = put_tree(store, layout.public_base_prefix, {"config.json": b"{}", "model.safetensors": b"bf16"})
    master = put_tree(store, layout.public_master_prefix, {"model.part00.safetensors": b"fp32"})
    head = HeadVersion.make(parent_version_id=None, base_hash=base, base_fp32_master_digest=master, loras=(), round_id="genesis", applied_at=1)
    return SimpleNamespace(store=store, head=head, config=plan.BoxConfig(tmp_path / "work", build.image), build=build)


def train_request(box, train: dict | None = None, groups=GROUPS, **params) -> TrainRequest:
    raw = json.dumps({"groups": groups}).encode()
    reg = box.build.reg(7)
    frozen = FrozenBatch(reg.hotkey, sha256_hex(raw), layout.frozen_batch("r000001", reg.hotkey), "affine-batch-v2", 2, 3, train or {})
    box.store.put(frozen.uri, raw)
    return TrainRequest("r000001", box.head, Entry(0, reg, frozen), box.build.params(**params), 11, (2, 3), "pending/loras/000")


class FakeJob:
    """Stands in for the training child process: it writes what ``report`` says, or hangs."""

    def __init__(self, report: dict | None = None, code: int = 0, hang: bool = False, files=LORA_FILES):
        self.report, self.code, self.hang, self.files = report or {"status": "ok", "loss": -0.2, "step_norm": 0.5, "tokens": 9, "gpu_class": "fake"}, code, hang, files
        self.calls, self.stopped, self.released = [], 0, threading.Event()

    def __call__(self, argv, env, log):
        job_dir = Path(argv[-1])
        self.calls.append((list(argv), dict(env), json.loads((job_dir / "job.json").read_text())))
        for name in self.files:
            (job_dir / "lora" / name).write_text(name)
        if self.report != "none":
            (job_dir / "report.json").write_text(json.dumps(self.report))
        return self

    def alive(self):
        return self.hang and not self.stopped

    def wait(self, timeout_s=None):
        if self.hang:
            self.released.wait(min(timeout_s or 5.0, 5.0))
            return None
        return self.code

    def stop(self):
        self.stopped += 1
        self.released.set()


def trainer(box, job: FakeJob) -> LoraTrainer:
    return LoraTrainer(box.store, box.config, lambda raw, hotkey, params: json.loads(raw)["groups"], spawn=job)


# ---- pins and config ----


def test_wrong_image_or_code_pin_is_a_box_failure_not_a_stall(box):
    params = box.build.params()
    plan.check_pins(box.build.image, params)
    with pytest.raises(InfraError):
        plan.check_pins("other-image", params)
    pinned = box.build.params(recipe=Recipe(image_digests=(box.build.image,), code_digest=plan.code_digest()))
    plan.check_pins(box.build.image, pinned)
    with pytest.raises(InfraError):
        plan.check_pins(box.build.image, box.build.params(recipe=Recipe(image_digests=(box.build.image,), code_digest="0" * 64)))
    with pytest.raises(InfraError):  # every seam checks the pin before it does any work
        trainer(box, FakeJob()).train(train_request(box, recipe=Recipe(image_digests=("other",))))


def test_box_config_has_no_default():
    with pytest.raises(Exception, match="SN120_IMAGE_DIGEST"):
        plan.BoxConfig.from_env({"SN120_WORK_DIR": "/tmp/x"})
    assert plan.BoxConfig.from_env({"SN120_WORK_DIR": "/tmp/x", "SN120_IMAGE_DIGEST": "d"}).work_dir == Path("/tmp/x")
    assert plan.install_digest() == plan.install_digest() and len(plan.install_digest()) == 64  # the image digest of a box without a baked image


# ---- batch rows, token caps, derail mask ----


def test_advantage_is_reward_minus_the_mean_of_its_group():
    assert batch.advantages([1.0, 0.0, 0.0, 1.0]) == [0.5, -0.5, -0.5, 0.5]
    assert batch.advantages([0.75]) == [0.0]  # a group of one is its own mean
    rows = batch.training_rows(GROUPS)
    assert [(row["prompt"], row["output"], row["advantage"]) for row in rows] == [("p1", "good", 0.5), ("p1", "bad", -0.5), ("p2", "alone", 0.0)]


def test_group_size_cuts_the_rollouts_of_a_prompt_in_file_order_and_keeps_a_short_last_group():
    rewards = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert batch.advantages(rewards, 0) == pytest.approx([0.4, -0.6, 0.4, 0.4, -0.6])  # 0: all rollouts of the prompt
    assert batch.advantages(rewards, 2) == [0.5, -0.5, 0.0, 0.0, 0.0]  # (1, 0), (1, 1), (0)
    assert batch.advantages(rewards, 4) == [0.25, -0.75, 0.25, 0.25, 0.0]
    assert batch.advantages(rewards, 64) == batch.advantages(rewards, 0)  # larger than the prompt: one group
    two_prompts = [["p1", [["a", 1.0], ["b", 0.0], ["c", 0.0]]], ["p2", [["d", 1.0], ["e", 0.0]]]]
    assert [row["advantage"] for row in batch.training_rows(two_prompts, 2)] == [0.5, -0.5, 0.0, 0.5, -0.5]  # a group never spans two prompts


@pytest.mark.parametrize(
    "groups",
    [[], [["p", []]], [["", [["o", 1.0]]]], [["p", [["o", float("nan")]]]], [["p", [["o", "1"]]]], [["p", [["a<|im_end|>b", 1.0]]]], [["<|im_start|>p", [["o", 1.0]]]]],
)
def test_bad_rows_reject_the_file(groups):
    with pytest.raises(BatchRejected):
        batch.training_rows(groups)


def test_end_token_only_when_the_output_ended_and_long_rows_are_cut_not_rejected():
    assert batch.fit_tokens(10, [5, 6], 99, max_output=4, max_context=100) == ([5, 6, 99], False, False)
    assert batch.fit_tokens(10, [5, 6, 7, 8], 99, max_output=4, max_context=100) == ([5, 6, 7, 8], True, False)  # exactly at the cap: cut, not ended
    assert batch.fit_tokens(10, [5, 6, 7, 8, 9, 9], 99, max_output=4, max_context=100) == ([5, 6, 7, 8], True, True)
    assert batch.fit_tokens(10, [5, 6, 7, 8], 99, max_output=4, max_context=12) == ([5, 6], True, True)  # a long prompt leaves less room
    for n_prompt, output in ((10, []), (100, [5]), (97, [5])):  # no output token left: the caller drops the row
        assert batch.fit_tokens(n_prompt, output, 99, max_output=4, max_context=97) == ([], False, False)


def test_derail_mask_keeps_the_tail_after_the_first_bad_window():
    fine, bad = [-0.1] * 10, [-9.0] * 10
    assert batch.derail_keep(fine, window=4, threshold=-3.0, tail=2) == 10
    assert batch.derail_keep(fine + bad, window=4, threshold=-3.0, tail=2) == 14  # derails at token 11; two more tokens stay
    assert batch.derail_keep(bad[:3], window=4, threshold=-3.0, tail=2) == 3  # shorter than one window: nothing is masked


# ---- processes and the GPU hand-over ----


def test_stopping_a_child_ends_its_whole_process_group(tmp_path):
    child = procs.Child(["sh", "-c", "sleep 300 & sleep 300"], os.environ, tmp_path / "log")
    assert child.alive() and child.wait(0.05) is None
    child.stop()
    assert not child.alive()
    with pytest.raises(ProcessLookupError):  # the grandchild is gone too: nothing can still hold a GPU
        for _ in range(100):
            os.killpg(child.proc.pid, 0)
            time.sleep(0.05)


def test_stop_all_reaches_every_process_and_reports_a_failure():
    class Stuck(FakeJob):
        def stop(self):
            super().stop()
            raise OSError("no")

    jobs = [Stuck(), FakeJob()]
    with pytest.raises(InfraError):
        procs.stop_all(jobs)
    assert jobs == [] and procs.gpu_env([4, 5], {"A": "1"}) == {"A": "1", "CUDA_VISIBLE_DEVICES": "4,5"}


def test_gpus_count_as_free_only_when_the_driver_says_so():
    assert procs.busy_gpus("0, 3\n1, 81000\n2, 900\n", 1024) == [1]
    assert procs.busy_gpus("0, 3\ngarbage\n", 1024) == [1]  # a line that cannot be read fails closed
    reports = iter(["0, 5000\n", "0, 5000\n", "0, 10\n"])
    naps = []
    procs.wait_gpus_free(1024, 60.0, report=lambda: next(reports), sleep=naps.append)
    assert naps == [5.0, 5.0]
    with pytest.raises(InfraError, match="still in use"):
        procs.wait_gpus_free(1024, 10.0, report=lambda: "0, 5000\n", sleep=naps.append)


# ---- the verified cache ----


def test_the_cache_serves_only_files_that_hash_to_the_identity(box, tmp_path, monkeypatch):
    resident = Resident(box.store, tmp_path / "w")
    base_dir = resident_head(resident, box.head)
    assert sorted(p.name for p in base_dir.iterdir()) == ["config.json", "model.safetensors"]  # the master is not in the base directory
    hashed = []
    monkeypatch.setattr(resident_module, "hash_tree", lambda path: hashed.append(path) or hash_tree(path))
    assert resident_head(Resident(box.store, tmp_path / "w"), box.head) == base_dir and hashed == []  # the trainer, the evaluator and the folder hash a head one time
    (base_dir / "model.safetensors").write_bytes(b"rot")  # a damaged local copy is found by a new process and replaced
    resident_module._SHARED.clear()  # a new process
    assert (resident_head(Resident(box.store, tmp_path / "w"), box.head) / "model.safetensors").read_bytes() == b"bf16"
    box.store.put(f"{layout.public_master_prefix(box.head.base_fp32_master_digest)}/model.part00.safetensors", b"upcast copy")
    fresh = Resident(box.store, tmp_path / "w2")
    with pytest.raises(VerifyError):  # a master that differs: fail closed, and keep nothing of it
        resident_head(fresh, box.head)
    assert list(fresh.root.iterdir()) == []


def test_a_missing_master_or_a_stacked_head_stops_the_box(box, tmp_path):
    ref = LoRARef(box.build.h("lora"), "r000001", "hk7", 7, box.build.h("batch"), box.build.params().recipe_id, 16)
    stacked = HeadVersion.make(
        parent_version_id=box.head.version_id, base_hash=box.head.base_hash, base_fp32_master_digest=box.head.base_fp32_master_digest, loras=(ref,), round_id="r000001", applied_at=2
    )
    with pytest.raises(InfraError):
        resident_head(Resident(box.store, tmp_path / "w"), stacked)
    box.store.delete(f"{layout.public_master_prefix(box.head.base_fp32_master_digest)}/model.part00.safetensors")
    with pytest.raises(VerifyError):
        resident_head(Resident(box.store, tmp_path / "w"), box.head)
    job = FakeJob()
    with pytest.raises(VerifyError):  # so nothing is trained on a head that cannot be folded
        trainer(box, job).train(train_request(box))
    assert job.calls == []


def test_prune_keeps_what_is_named_and_adopt_checks_the_hash(box, tmp_path):
    resident = Resident(box.store, tmp_path / "w")
    resident_head(resident, box.head)
    resident.prune({box.head.base_hash})
    assert [p.name for p in resident.root.iterdir()] == [box.head.base_hash]
    made = tmp_path / "made"
    made.mkdir()
    (made / "f").write_bytes(b"new")
    with pytest.raises(VerifyError):
        resident.adopt(made, "0" * 64)
    identity = manifest_digest({"f": sha256_hex(b"new")})
    resident.adopt(made, identity)
    assert resident.tree("nowhere", identity).name == identity  # served from disk: no download


def test_an_upload_is_checked_against_the_store(box, tmp_path):
    (tmp_path / "f").write_bytes(b"data")

    class Lossy(LocalStore):
        def put_file(self, key, path):
            self.put(key, b"other bytes")

    with pytest.raises(InfraError, match="differs"):
        upload_tree(Lossy(tmp_path / "lossy"), tmp_path, "p")


# ---- the Trainer seam ----


def test_train_uses_the_miner_values_the_request_gpus_and_uploads_what_it_hashed(box):
    job = FakeJob()
    request = train_request(box, train={"rank": 8, "lora_lr": 2e-5, "steps": 2, "sample_temperature": 0.8})
    artifact = trainer(box, job).train(request)
    argv, env, sent = job.calls[0]
    assert argv[:3] == [sys.executable, "-m", "sn120_v2.gpu.train_job"] and env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["TRITON_CACHE_DIR"] == str(box.config.work_dir / "triton" / "gpu2")  # a kernel cache per GPU slot: jobs of one wave never share it
    recipe = Recipe.from_dict(sent["recipe"])
    assert (recipe.rank, recipe.lora_lr, recipe.steps, recipe.sample_temperature) == (8, 2e-5, 2, 0.8)
    assert sent["seed"] == 11 and sent["base_hash"] == box.head.base_hash and [row["advantage"] for row in sent["rows"]] == [0.5, -0.5, 0.0]
    assert (artifact.rank, artifact.alpha, artifact.target_modules) == (8, request.params.recipe.alpha, request.params.recipe.target_modules)
    assert artifact.recipe_id == request.params.recipe_id and artifact.head_version_id == box.head.version_id
    assert artifact.lora_hash == manifest_digest(artifact.files) and sorted(artifact.files) == sorted(LORA_FILES)
    assert all(box.store.sha256(f"{artifact.prefix}/{name}") == sha for name, sha in artifact.files.items())
    assert (artifact.report.seed, artifact.report.steps, artifact.report.step_norm, artifact.report.image_digest) == (11, 2, 0.5, box.build.image)
    assert not Path(argv[-1]).exists()  # the box keeps no copy of the LoRA or of the miner's rows
    logged = [json.loads(line) for line in (box.config.work_dir / "logs" / "phases.jsonl").read_text().splitlines()]
    assert [(line["phase"], line["round_id"], line["queue_pos"], line["step_norm"]) for line in logged] == [("train", "r000001", 0, 0.5)] and logged[0]["wall_s"] >= 0
    assert (logged[0]["steps"], logged[0]["tokens"]) == (2, 9) and {"forward_tokens", "step_s", "n_gpus", "rows", "step_norm_before_cap"} <= set(logged[0])  # the speed can be read


def test_every_key_a_miner_may_set_trains_at_both_ends_of_its_bound(box):
    for bound in box.build.params().recipe.miner_bounds:  # the published bounds: no number is copied here
        ends = [list(bound.choices), [bound.choices[-1]]] if bound.choices else [bound.low, bound.high]
        for value in ends:
            value, job = int(value) if bound.integer else value, FakeJob()
            trainer(box, job).train(train_request(box, train={bound.key: value}))
            assert getattr(Recipe.from_dict(job.calls[0][2]["recipe"]), bound.key) == bound.check(value)
    job = FakeJob()
    trainer(box, job).train(train_request(box, train={"group_size": 2}, groups=[["p1", [["a", 1.0], ["b", 0.0], ["c", 1.0]]]]))
    assert [row["advantage"] for row in job.calls[0][2]["rows"]] == [0.5, -0.5, 0.0]  # groups of 2 in file order; the last group is short


@pytest.mark.parametrize(
    "report, error",
    [
        ({"status": "rejected", "reason": "no gradient"}, BatchRejected),
        ({"status": "ok", "loss": float("nan"), "step_norm": 0.1, "tokens": 1, "gpu_class": "x"}, BatchRejected),
        ({"status": "ok", "loss": 0.1, "step_norm": float("inf"), "tokens": 1, "gpu_class": "x"}, BatchRejected),
        ({"status": "ok", "loss": 0.1}, InfraError),
        ("none", InfraError),
    ],
)
def test_a_bad_training_report_never_becomes_a_lora(box, report, error):
    with pytest.raises(error):
        trainer(box, FakeJob(report)).train(train_request(box))
    assert box.store.list("pending") == []


def test_training_failures_are_sorted_into_miner_and_box(box):
    class Crash(FakeJob):
        def __call__(self, argv, env, log):
            Path(log).write_text("Traceback ...\ntorch.OutOfMemoryError: CUDA out of memory\n")
            return super().__call__(argv, env, log)

    with pytest.raises(InfraError, match="code 1; log: .*train-r000001-000.log; last line: torch.OutOfMemoryError: CUDA out of memory"):
        trainer(box, Crash(code=1)).train(train_request(box))
    assert "CUDA out of memory" in (box.config.work_dir / "logs" / "train-r000001-000.log").read_text()  # the cause outlives the job directory
    with pytest.raises(InfraError):
        trainer(box, FakeJob(files=LORA_FILES[:2])).train(train_request(box))
    with pytest.raises(InfraError, match="global_norm_cap"):
        trainer(box, FakeJob()).train(train_request(box, recipe=Recipe(image_digests=(box.build.image,), global_norm_cap=0.1)))
    with pytest.raises(BatchRejected):  # outside the published bound: never clamped
        trainer(box, FakeJob()).train(train_request(box, train={"rank": 4096}))
    with pytest.raises(BatchRejected):
        trainer(box, FakeJob()).train(train_request(box, groups=[["p", [["x<|im_end|>", 1.0]]]]))
    bad_reader = LoraTrainer(box.store, box.config, lambda raw, hotkey, params: (_ for _ in ()).throw(ValueError("not a batch")), spawn=FakeJob())
    with pytest.raises(BatchRejected, match="not a batch"):
        bad_reader.train(train_request(box))
    request = train_request(box)
    box.store.put(request.entry.batch.uri, b"swapped after the freeze")
    with pytest.raises(VerifyError):
        trainer(box, FakeJob()).train(request)
    box.store.delete(request.entry.batch.uri)
    with pytest.raises(InfraError):
        trainer(box, FakeJob()).train(request)


def test_close_stops_a_running_training_job_and_a_late_job_is_stopped(box):
    job, errors = FakeJob(hang=True), []
    seam = trainer(box, job)
    thread = threading.Thread(target=lambda: errors.append(pytest.raises(InfraError, seam.train, train_request(box))))
    thread.start()
    while not job.calls:
        time.sleep(0.01)
    seam.close()  # the GPUs of the round are free before evaluation starts
    assert job.stopped == 1
    thread.join(10)
    assert errors and job.stopped == 2 and seam._procs == []


# ---- the Evaluator seam ----


class Farm:
    """Fake vLLM servers. ``script(model, task) -> (text, finish_reason)`` or an ``HttpError``."""

    def __init__(self, script, boot_ok: bool = True):
        self.script, self.boot_ok = script, boot_ok
        self.servers, self.bodies, self.loads, self.freed, self.lock = [], [], [], [], threading.Lock()

    def spawn(self, argv, env, log):
        self.servers.append(SimpleNamespace(argv=list(argv), env=dict(env), up=self.boot_ok, alive=lambda: self.servers[index].up, stop=lambda: setattr(self.servers[index], "up", False)))
        index = len(self.servers) - 1
        return self.servers[index]

    def http(self, url, body, timeout_s):
        port = int(url.split(":")[2].split("/")[0])
        if not any(s.up and str(port) in s.argv for s in self.servers):
            raise ConnectionRefusedError(url)
        if url.endswith("/health"):
            return {}
        with self.lock:
            if url.endswith("/v1/load_lora_adapter"):
                self.loads.append((port, body["lora_name"], sorted(p.name for p in Path(body["lora_path"]).iterdir())))
                return {}
            self.bodies.append((port, body))
        reply = self.script(body["model"], body["messages"][-1]["content"])
        if isinstance(reply, Exception):
            raise reply
        return {"choices": [{"message": {"content": reply[0]}, "finish_reason": reply[1]}]}


class Grades:
    """A fake official grader: the text is the verdict. ``died`` texts lose their worker on the first tries."""

    def __init__(self, deaths: int = 0):
        self.deaths, self.seen = deaths, []

    def grade_many(self, jobs):
        self.seen.append([text for _, text, capped in jobs if not capped])
        dying = self.deaths > 0
        self.deaths -= 1
        return [SimpleNamespace(code="E", machine_cause="worker-died") if text == "F died" and dying else SimpleNamespace(code=text[0], machine_cause=None) for _, text, _ in jobs]


POOL = SimpleNamespace(task=lambda task_id: SimpleNamespace(messages=lambda: [{"role": "system", "content": "MANDATE"}, {"role": "user", "content": task_id}]))


def evaluator(box, farm: Farm, grades: Grades | None = None, **settings) -> VllmEvaluator:
    kwargs = dict(spawn=farm.spawn, http=farm.http, gpus_free=lambda limit, timeout: farm.freed.append(len(farm.servers)), sleep=lambda s: None)
    return VllmEvaluator(box.store, box.config, POOL, grades or Grades(), settings=ServeSettings(start_timeout_s=10.0, **settings), **kwargs)


def eval_request(box, n_loras: int = 2, tasks=("t0", "t1", "t2"), look: int = 0, **params) -> EvalRequest:
    params = box.build.params(n_gpus=2, **params)
    loras = []
    for index in range(n_loras):
        files = {"adapter_config.json": b"{}", "adapter_model.safetensors": f"lora-{index}".encode()}
        identity = put_tree(box.store, lambda digest: f"pending/loras/{digest}", files)
        made = box.build.artifact(box.head, box.build.entry(index), params)
        loras.append(box.build.replace(made, lora_hash=identity, prefix=f"pending/loras/{identity}", files={n: sha256_hex(d) for n, d in files.items()}))
    return EvalRequest("r000001", box.head, tuple(loras), tuple(tasks), "seed-0", look, params)


def test_null_and_every_lora_get_the_same_tasks_seeds_caps_and_server(box):
    farm = Farm(lambda model, task: ("P ok", "stop"))
    request = eval_request(box, samples_per_task=2)
    chunk = evaluator(box, farm).evaluate(request)
    models = [NULL_MODEL, *(lora.lora_hash for lora in request.loras)]
    assert list(chunk.outcomes) == models and all(chunk.outcomes[m] == ("PPP", "PPP") for m in models)  # null M is always scored
    assert (chunk.task_ids, chunk.look, chunk.image_digest) == (request.task_ids, 0, box.build.image)
    by_job: dict = {}
    for port, body in farm.bodies:
        by_job.setdefault((body["messages"][-1]["content"], body["seed"]), []).append((port, body))
    assert len(by_job) == 6 and {seed for _, seed in by_job} == {rollout_seed("seed-0", task, s) for task in request.task_ids for s in (0, 1)}
    for sent in by_job.values():  # one task and seed: every model, one server, and a body that differs only in "model"
        assert sorted(body["model"] for _, body in sent) == sorted(models) and len({port for port, _ in sent}) == 1
        assert len({json.dumps({**body, "model": ""}, sort_keys=True) for _, body in sent}) == 1
    body = farm.bodies[0][1]
    params = request.params
    assert (body["max_tokens"], body["temperature"], body["top_p"], body["top_k"]) == (params.max_output_tokens, params.temperature, params.top_p, params.top_k)
    assert body["messages"][0] == {"role": "system", "content": "MANDATE"} and body["skip_special_tokens"] is False


def test_outcome_codes_capped_is_never_graded_and_errors_stay_in_the_count(box):
    replies = {"t0": ("P", "stop"), "t1": ("F", "stop"), "t2": ("P cut", "length"), "t3": HttpError(400), "t4": ("E", "stop"), "t5": ("?", "stop")}
    grades = Grades()
    chunk = evaluator(box, Farm(lambda model, task: replies[task]), grades).evaluate(eval_request(box, n_loras=1, tasks=tuple(replies)))
    assert set(chunk.outcomes.values()) == {("PFCEEE",)}
    assert len(grades.seen) == 1 and "P cut" not in grades.seen[0] and len(grades.seen[0]) == 8  # an E of the grader is graded one time
    logged = json.loads((box.config.work_dir / "logs" / "phases.jsonl").read_text().splitlines()[-1])
    assert (logged["phase"], logged["look"], logged["models"], logged["rollouts"], logged["capped"]) == ("eval", 0, 2, 12, 2) and logged["wall_s"] >= logged["generate_s"]
    assert logged["generate_s"] >= logged["longest_rollout_s"] > 0 and {"output_tokens", "longest_rollout_tokens", "most_output_tokens", "n_gpus"} <= set(logged)
    assert read_reply({"choices": [{"message": {"content": "<think>x</think>a<|im_end|>"}, "finish_reason": "stop"}]}) == ("<think>x</think>a", False)


def test_a_server_error_is_retried_and_a_dead_server_voids_the_look(box):
    failures = iter([HttpError(503), ConnectionResetError()])
    lock = threading.Lock()

    def flaky(model, task):
        with lock:
            return next(failures, ("P", "stop"))

    assert set(evaluator(box, Farm(flaky)).evaluate(eval_request(box)).outcomes.values()) == {("PPP",)}
    farm = Farm(None)
    farm.script = lambda model, task: setattr(farm.servers[0], "up", False) or ("P", "stop")
    with pytest.raises(InfraError, match="died"):  # a dead server is not a row of E: nothing of the look is scored
        evaluator(box, farm, retries=0).evaluate(eval_request(box))
    assert not any(server.up for server in farm.servers)


def test_a_chunk_that_the_round_time_limit_cut_is_not_scored(box):
    budgets = []
    farm = Farm(lambda model, task: ("P", "stop"))
    http = farm.http
    farm.http = lambda url, body, timeout_s: budgets.append(timeout_s) or http(url, body, timeout_s)
    request = box.build.replace(eval_request(box), deadline=int(time.time()) + 50)
    assert set(evaluator(box, farm).evaluate(request).outcomes.values()) == {("PPP",)}
    assert 0 < min(budgets) <= 50, "no rollout may run past the deadline"
    late = box.build.replace(eval_request(box), deadline=int(time.time()) - 1)
    with pytest.raises(InfraError, match="time limit"):  # never a row of E: a cut rollout is not an error of its model
        evaluator(box, Farm(lambda model, task: ("P", "stop"))).evaluate(late)


def test_a_dead_grader_worker_is_graded_again_and_never_writes_a_code(box):
    farm = Farm(lambda model, task: ("F died" if task == "t1" else "P", "stop"))
    grades = Grades(deaths=2)
    chunk = evaluator(box, farm, grades).evaluate(eval_request(box, n_loras=1))
    assert chunk.outcomes[NULL_MODEL] == ("PFP",) and [len(batch_) for batch_ in grades.seen] == [6, 2, 2]  # only the lost replies go again
    with pytest.raises(InfraError, match="grader worker died"):
        evaluator(box, farm, Grades(deaths=99)).evaluate(eval_request(box, n_loras=1))

    class Broken:
        def grade_many(self, jobs):
            raise RuntimeError("pool is gone")

    with pytest.raises(InfraError, match="grader failed"):
        evaluator(box, farm, Broken()).evaluate(eval_request(box, n_loras=1))


def test_servers_start_after_the_gpus_are_free_one_per_gpu_and_close_frees_everything(box):
    farm = Farm(lambda model, task: ("P", "stop"))
    seam = evaluator(box, farm)
    first = eval_request(box)
    seam.evaluate(first)
    assert farm.freed == [0] and [s.env["CUDA_VISIBLE_DEVICES"] for s in farm.servers] == ["0", "1"]
    assert len({s.env["VLLM_CACHE_ROOT"] for s in farm.servers}) == 2  # v1: a shared compile cache killed servers
    assert sorted(farm.loads) == sorted((port, lora.lora_hash, ["adapter_config.json", "adapter_model.safetensors"]) for port in (8100, 8101) for lora in first.loras)
    seam.evaluate(box.build.replace(first, loras=first.loras[:1], look=1))  # the next look: same servers, nothing loaded again
    assert len(farm.servers) == 2 and len(farm.loads) == 4 and {body["model"] for _, body in farm.bodies[-6:]} == {NULL_MODEL, first.loras[0].lora_hash}
    seam.close()
    assert not any(server.up for server in farm.servers)
    assert sorted(p.name for p in seam.resident.root.iterdir()) == sorted([box.head.base_hash, box.head.base_fp32_master_digest])  # LoRAs leave; the head stays


def test_server_command_line_and_refused_requests(box):
    request = eval_request(box)
    argv = server_argv("/base", 8100, request.params)
    for flags in (["--served-model-name", NULL_MODEL], ["--host", "127.0.0.1"], ["--max-model-len", str(request.params.max_context_tokens)], ["--dtype", request.params.dtype]):
        assert flags[1] == argv[argv.index(flags[0]) + 1]
    assert argv[argv.index("--max-lora-rank") + 1] == str(int(request.params.recipe.bound("rank").high)) and "--enable-lora" in argv
    assert argv[argv.index("--limit-mm-per-prompt") + 1] == '{"image":0,"video":0}'  # v1: vLLM does not load the checkpoint without it
    assert server_env(3, box.config.work_dir)["HF_HUB_OFFLINE"] == "1"
    assert chat_body("m", [], 5, request.params)["chat_template_kwargs"] == {"enable_thinking": True}
    other_head = box.build.replace(request.loras[0], head_version_id="0" * 64)
    with pytest.raises(ContractError):
        evaluator(box, Farm(None)).evaluate(box.build.replace(request, loras=(other_head,)))
    with pytest.raises(ContractError):
        evaluator(box, Farm(None)).evaluate(box.build.replace(request, loras=request.loras[:1] * 2))
    farm = Farm(None, boot_ok=False)
    with pytest.raises(InfraError, match="did not start"):
        evaluator(box, farm).evaluate(request)
