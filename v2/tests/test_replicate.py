"""The miner's local round on CPU: the real trainer, evaluator, runner and judging of the box, with a fake training job, fake vLLM servers and a fake grader."""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sn120_v2.gpu.trainer")

from sn120_v2 import assembly, layout, replicate  # noqa: E402
from sn120_v2.canonical import manifest_digest, sha256_hex  # noqa: E402
from sn120_v2.envs.math_pool import POOL_V2_0_FILE, POOL_V2_0_SHA256, load_pool, pool_for_draw  # noqa: E402
from sn120_v2.errors import ConfigError, ContractError, InfraError, VerifyError  # noqa: E402
from sn120_v2.gpu import plan  # noqa: E402
from sn120_v2.gpu import resident as resident_module  # noqa: E402
from sn120_v2.gpu.evaluator import ServeSettings  # noqa: E402
from sn120_v2.gpu.procs import wait_gpus_free  # noqa: E402
from sn120_v2.gpu.resident import Resident  # noqa: E402
from sn120_v2.gpu.trainer import LORA_FILES  # noqa: E402
from sn120_v2.irt import IRTSampler, draw_from_snapshot  # noqa: E402
from sn120_v2.params import Recipe  # noqa: E402
from sn120_v2.sampling import rollout_seed, train_seed  # noqa: E402
from sn120_v2.store import LocalStore  # noqa: E402
from sn120_v2.types import NULL_MODEL, SCORED, EntryResult, HeadManifest, HeadVersion  # noqa: E402

HOTKEY = "hk1"
BASE = {"config.json": b"{}", "tokenizer.json": b"tokens", "model-00001.safetensors": b"weights"}
POOL = load_pool()
TASKS = POOL.task_ids[:40]
OUTPUT = "step " * 40 + "\\boxed{2}"
SMI = "0, 3\n1, 5\n2, 90000\n"  # three GPUs; the third is in use


def h(text: str) -> str:
    return sha256_hex(text.encode())


class WordTokens:
    prompt_tokens = output_tokens = staticmethod(lambda text: len(text.split()))


class FakeJob:
    """Stands in for the training child process: it writes the LoRA files and the report."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, env, log):
        job_dir = Path(argv[-1])
        self.calls.append((dict(env), json.loads((job_dir / "job.json").read_text())))
        for name in LORA_FILES:
            (job_dir / "lora" / name).write_text(name)
        (job_dir / "report.json").write_text(json.dumps({"status": "ok", "loss": -0.2, "step_norm": 0.5, "tokens": 9, "gpu_class": "fake"}))
        return SimpleNamespace(wait=lambda timeout_s=None: 0, alive=lambda: False, stop=lambda: None)


class Farm:
    """Fake vLLM servers. Null M fails every task; the LoRA passes every task but the ones in ``fails``."""

    def __init__(self):
        self.servers, self.bodies, self.loads, self.lock = [], [], [], threading.Lock()
        self.fails: set[str] = set()  # problem texts
        self.capped = False

    def spawn(self, argv, env, log):
        server = SimpleNamespace(argv=list(argv), env=dict(env), up=True)
        server.alive, server.stop = (lambda: server.up), (lambda: setattr(server, "up", False))
        self.servers.append(server)
        return server

    def http(self, url, body, timeout_s):
        port = int(url.split(":")[2].split("/")[0])
        if not any(s.up and str(port) in s.argv for s in self.servers):
            raise ConnectionRefusedError(url)
        if url.endswith("/health"):
            return {}
        with self.lock:
            if url.endswith("/v1/models"):
                return {"data": [{"id": NULL_MODEL}, *({"id": name} for p, name in self.loads if p == port)]}
            if url.endswith("/v1/load_lora_adapter"):
                self.loads.append((port, body["lora_name"]))
                return {}
            self.bodies.append((port, body))
        passed = body["model"] != NULL_MODEL and body["messages"][-1]["content"] not in self.fails and not self.capped
        content = "C cap" if self.capped and body["model"] != NULL_MODEL else "P ok" if passed else "F no"
        return {"choices": [{"message": {"content": content}, "finish_reason": "length" if content.startswith("C") else "stop"}]}


class Grades:
    """A fake official grader for both of its users. Scoring: the first letter is the code. File check: a text that ends with ``}`` is right."""

    made: list[dict] = []

    def __init__(self, **options):
        self.options, self.closed = options, False
        Grades.made.append(options)

    def grade_many(self, jobs):
        return [SimpleNamespace(code=text[0], machine_cause=None, reward=float(text.endswith("}"))) for _, text, _ in jobs]

    def close(self):
        self.closed = True


def batch_bytes(train: dict | None = None, hotkey: str = HOTKEY) -> bytes:
    rollouts = [{"output": OUTPUT, "reward": 1.0}, {"output": OUTPUT + " again", "reward": 0.0}]
    body = {"version": "affine-batch-v2", "hotkey": hotkey, "groups": [{"prompt": POOL.task(TASKS[0]).batch_prompt(), "rollouts": rollouts}]}
    return json.dumps(body if train is None else body | {"train": train}).encode()


@pytest.fixture
def world(tmp_path, build, monkeypatch):
    """What a miner has: mining.json, the public objects (as files), the base files in a directory, a batch file, and fakes for the GPU."""
    resident_module._SHARED.clear()
    Grades.made = []
    public, source = tmp_path / "public", tmp_path / "hf"
    source.mkdir()
    for name, data in BASE.items():
        (source / name).write_bytes(data)
    files = {name: sha256_hex(data) for name, data in BASE.items()}
    head = HeadVersion.make(parent_version_id=None, base_hash=manifest_digest(files), base_fp32_master_digest=h("master"), loras=(), round_id="genesis", applied_at=1)
    params = build.params(n_gpus=8, n_tasks=8, looks=(0.5, 1.0), eval_chunk_tasks=4)
    snapshot = dict(IRTSampler(list(TASKS), params).public_snapshot())
    store = LocalStore(public)
    store.put(layout.irt_snapshot(snapshot["snapshot_id"])[len("public/") :], json.dumps(snapshot).encode())
    view = {"round_id": "r000007", "round_seed": "seed-7", "irt_snapshot_id": snapshot["snapshot_id"], "n_tasks": 8, "head_version_id": head.version_id, "params_hash": params.params_hash}
    farm, job = Farm(), FakeJob()
    serve = dict(spawn=farm.spawn, http=farm.http, sleep=lambda s: None, settings=ServeSettings(start_timeout_s=10.0))
    real_parts = replicate.parts
    monkeypatch.setattr(replicate, "parts", lambda work, gpus=None, **kwargs: real_parts(work, gpus, train_spawn=job, serve=serve, **kwargs))
    monkeypatch.setattr(replicate, "GradePool", Grades)
    monkeypatch.setattr(replicate, "nvidia_smi", lambda: SMI)
    monkeypatch.setattr(replicate, "head_tokens", lambda directory: WordTokens)
    batch = tmp_path / "batch.json"
    batch.write_bytes(batch_bytes())
    made = SimpleNamespace(tmp=tmp_path, public=public, source=source, head=head, files=files, params=params, snapshot=snapshot, view=view, farm=farm, job=job, batch=batch, work=tmp_path / "work")
    made.write = lambda **changes: write_contract(made, **changes)
    made.write()
    return made


def write_contract(world, params=None, round="view", history=(), manifest=None) -> str:
    """mining.json with the keys the tool reads."""
    params = params or world.params
    doc = {
        "contract": {"public_url": world.tmp.as_uri(), "read_links": "public/read_links.json"},
        "head": (manifest or HeadManifest(world.head, world.files, {"master.safetensors": h("fp32")}, {})).to_dict(),
        "params": params.to_dict(),
        "round": world.view if round == "view" else round,
        "history": list(history),
    }
    path = world.tmp / "mining.json"
    path.write_text(json.dumps(doc))
    return str(path)


def test_public_contract_ignores_future_params_fields_but_keeps_strict_values(world, capsys):
    contract = json.loads(Path(world.write()).read_text())
    contract["params"]["future_param"] = {"future": True}
    contract["params"]["recipe"]["future_recipe"] = {"future": True}
    contract["params"]["recipe"]["miner_bounds"][0]["future_bound"] = "new"
    path = world.tmp / "future.json"
    path.write_text(json.dumps(contract))
    assert replicate.Contract.load(str(path)).params == world.params
    with pytest.raises(ContractError, match="unknown field"):
        from sn120_v2.params import Params
        Params.from_dict(contract["params"])
    contract["params"]["n_tasks"] = "8"
    path.write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="expected an integer"):
        replicate.Contract.load(str(path))


def cli(capsys, *argv: str) -> tuple[int, dict]:
    code = replicate.main(list(argv))
    return code, json.loads(capsys.readouterr().out)


def fetch(world, capsys) -> dict:
    code, report = cli(capsys, "fetch", "--mining-json", str(world.tmp / "mining.json"), "--work", str(world.work), "--from", str(world.source))
    assert code == 0
    return report


def run(world, capsys, *more: str) -> tuple[int, dict]:
    argv = ["run", "--mining-json", str(world.tmp / "mining.json"), "--hotkey", HOTKEY, "--work", str(world.work), "--gpus", "2", *more, str(world.batch)]
    return cli(capsys, *argv)


# ---- what differs from the box ----


def test_a_run_changes_only_the_values_this_machine_cannot_keep(build):
    params = build.params(n_gpus=8, recipe=Recipe(image_digests=("box-image",), code_digest="c" * 64))
    same, notes = replicate.local_params(params, "box-image", "c" * 64, 8, round_clock=True)
    assert same is params and [note["what"] for note in notes] == ["one-batch", "master", "free-gpus"]  # the box's own params, untouched
    local, notes = replicate.local_params(params, "my-image", plan.code_digest(), 1)
    assert [note["what"] for note in notes] == ["install", "code", "gpus", "round-clock", "one-batch", "master", "free-gpus"]
    assert (local.recipe.image_digests, local.recipe.code_digest, local.n_gpus) == (("my-image",), plan.code_digest(), 1)
    assert local.eval_time_budget_s == params.eval_time_budget_s and local.round_time_limit_s == replicate.NO_ROUND_LIMIT_S  # one reply keeps its limit
    changed = {"recipe", "n_gpus", "round_time_limit_s", "lease_timeout_s", "promote_reserve_s", "train_time_budget_s"}
    assert {key for key, value in local.to_dict().items() if value != params.to_dict()[key]} == changed
    assert local.recipe.to_dict() | {"image_digests": [], "code_digest": ""} == params.recipe.to_dict() | {"image_digests": [], "code_digest": ""}  # every training value is the box's
    text = {note["what"]: note["text"] for note in notes}
    assert "box-image" in text["install"] and "c" * 64 in text["code"] and "The box has 8" in text["gpus"]
    stop = params.round_time_limit_s - params.promote_reserve_s
    assert f"{params.train_time_budget_s} s" in text["round-clock"] and f"{stop} s" in text["round-clock"]  # each number comes from the params
    plan.check_pins("my-image", local)  # the box code accepts the local pins
    with pytest.raises(InfraError):
        plan.check_pins("my-image", params)


def test_the_file_check_grades_as_the_box_does():
    assert (replicate.GRADE_WORKERS, replicate.GRADE_TIMEOUT_S) == (assembly.ADMISSION_GRADE_WORKERS, assembly.ADMISSION_GRADE_TIMEOUT_S)


def test_compare_starts_nothing_and_names_every_difference(world, build, capsys, monkeypatch):
    code, report = cli(capsys, "compare", "--mining-json", str(world.tmp / "mining.json"), "--gpus", "1")
    assert code == 0 and report["box"] == {"params_hash": world.params.params_hash, "recipe_id": world.params.recipe_id, "image_digests": [build.image], "code_digest": "", "n_gpus": 8}
    assert report["here"] == {"image_digest": plan.install_digest(), "code_digest": plan.code_digest(), "gpus_found": 3, "n_gpus": 1}
    assert [note["what"] for note in report["differences"]] == ["install", "gpus", "round-clock", "one-batch", "master", "free-gpus"]
    assert world.farm.servers == [] and world.job.calls == [] and not world.work.exists()

    def no_driver():
        raise InfraError("nvidia-smi is not there")

    monkeypatch.setattr(replicate, "nvidia_smi", no_driver)
    assert cli(capsys, "compare", "--mining-json", str(world.tmp / "mining.json"))[1]["here"]["gpus_found"] is None


# ---- the local cache fails closed ----


def test_the_cache_never_answers_with_a_missing_master_or_searches_the_local_store(world, tmp_path):
    store = LocalStore(tmp_path / "s")
    cache = replicate.BaseOnly(store, tmp_path / "w")
    cache.merged = True
    master = world.head.base_fp32_master_digest
    for _ in range(2):
        with pytest.raises(ConfigError, match="tens of GB.*fetch command") as caught:
            cache.tree(layout.public_master_prefix(master), master)
        assert "eval_merged" in str(caught.value) and not (cache.root / master).exists()
    with pytest.raises(ConfigError, match="base files.*fetch command"):
        cache.tree(layout.public_base_prefix(world.head.base_hash), world.head.base_hash)
    with pytest.raises(VerifyError):  # the box's own cache still verifies through its real object store
        Resident(store, tmp_path / "w").tree(layout.public_master_prefix(master), master)
    with pytest.raises(VerifyError):  # a LoRA is verified too
        cache.tree("pending/loras/x", h("lora"))


def test_the_cache_verifies_the_real_master_and_refuses_a_corrupt_one(world, tmp_path):
    master_dir = world.tmp / "master-source"
    master_dir.mkdir()
    (master_dir / "master.safetensors").write_bytes(b"fp32")
    master_files = {"master.safetensors": h("fp32")}
    master = manifest_digest(master_files)
    resident_module._SHARED.clear()
    cache = replicate.BaseOnly(LocalStore(tmp_path / "s"), tmp_path / "w")
    cache.merged = True
    path = cache.root / master
    path.mkdir(parents=True)
    (path / "master.safetensors").write_bytes(b"fp32")
    assert cache.tree(layout.public_master_prefix(master), master) == path
    resident_module._SHARED.clear()  # a new process: do not trust the previous verification
    cache = replicate.BaseOnly(LocalStore(tmp_path / "s"), tmp_path / "w")
    cache.merged = True
    (path / "master.safetensors").write_bytes(b"bf16, not the master")
    with pytest.raises(ConfigError, match="missing files or does not hash.*tens of GB"):
        cache.tree(layout.public_master_prefix(master), master)
    assert not path.exists()  # the official Resident deletes a bad cache, never serves it


# ---- fetch ----


def test_fetch_from_a_directory_checks_each_file_and_fills_the_cache_in_one_move(world, capsys):
    (world.source / "model-00001.safetensors").write_bytes(b"other weights")
    code, report = cli(capsys, "fetch", "--mining-json", str(world.tmp / "mining.json"), "--work", str(world.work), "--from", str(world.source))
    cache = world.work / "cache" / world.head.base_hash
    assert code == 2 and report["error"] == "VerifyError" and "model-00001.safetensors is not the file of the head manifest" in report["reason"]
    assert not cache.exists()  # a half-filled set never reaches the cache: the box code would delete it
    (world.source / "model-00001.safetensors").write_bytes(BASE["model-00001.safetensors"])
    (world.source / "README.md").write_text("not a file of the head")  # more files in the source are ignored
    assert fetch(world, capsys) == {
        "base_hash": world.head.base_hash,
        "base_dir": str(cache),
        "dir": str(cache),
        "base_files": 3,
        "files": 3,
        "base_bytes": sum(map(len, BASE.values())),
        "bytes": sum(map(len, BASE.values())),
        "base_fetched": 2,
        "fetched": 2,
        "base_fetched_bytes": len(BASE["tokenizer.json"]) + len(BASE["model-00001.safetensors"]),
        "fetched_bytes": len(BASE["tokenizer.json"]) + len(BASE["model-00001.safetensors"]),
    }  # the first file was kept
    assert {path.name: path.read_bytes() for path in cache.iterdir()} == BASE
    assert fetch(world, capsys)["fetched"] == 0  # a second fetch only verifies
    (cache / "extra.txt").write_text("x")
    code, report = cli(capsys, "fetch", "--mining-json", str(world.tmp / "mining.json"), "--work", str(world.work), "--from", str(world.source))
    assert code == 2 and "does not match its manifest and id" in report["reason"] and (cache / "extra.txt").exists()  # fetch deletes nothing of yours


def links_file(world, **changes) -> dict:
    rows = [{"name": name, "key": f"public/heads/base/x/{name}", "sha256": world.files[name], "size": len(data), "url": (world.source / name).as_uri()} for name, data in BASE.items()]
    doc = {"schema": "sn120-v2-read-links-1", "expires_at": 2000, "head_version_id": world.head.version_id, "base": {"id": world.head.base_hash, "files": rows}}
    return doc | changes


def test_fetch_downloads_through_the_signed_links_and_fails_closed(world, capsys):
    contract = replicate.Contract.load(str(world.tmp / "mining.json"))
    path = world.public / "read_links.json"

    def go(doc: dict, now: int = 1000) -> dict:
        path.write_text(json.dumps(doc))
        return replicate.fetch(contract, world.work, now=lambda: now)  # the links file is found through mining.json

    with pytest.raises(ConfigError, match="expired"):
        go(links_file(world), now=2000)
    with pytest.raises(ConfigError, match="two different heads"):
        go(links_file(world, head_version_id=h("another head")))
    wrong = links_file(world)
    wrong["base"]["files"][0]["sha256"] = h("another file")
    with pytest.raises(VerifyError, match="does not list the base files"):
        go(wrong)
    (world.source / "tokenizer.json").write_bytes(b"swapped")
    with pytest.raises(VerifyError, match="tokenizer.json"):
        go(links_file(world))
    assert not (world.work / "cache" / world.head.base_hash).exists()
    signed = links_file(world)
    signed["base"]["files"][1]["url"] += "?X-Amz-Signature=secret"  # a file address with a query is a file that is not there
    with pytest.raises(InfraError, match="tokenizer.json") as failure:
        go(signed)
    assert "secret" not in str(failure.value) and "Signature" not in str(failure.value)  # a signed link never shows in an error
    (world.source / "tokenizer.json").write_bytes(BASE["tokenizer.json"])
    report = go(links_file(world))
    assert report["fetched"] == 1 and {p.name: p.read_bytes() for p in Path(report["dir"]).iterdir()} == BASE  # the files that were complete were kept
    assert list((world.work / "fetch").iterdir()) == []


def test_an_address_in_an_error_has_no_query(world):
    with pytest.raises(InfraError) as failure:
        replicate.read((world.tmp / "nothing.json").as_uri() + "?X-Amz-Signature=secret")
    assert "nothing.json" in str(failure.value) and "secret" not in str(failure.value)


# ---- the round to repeat ----


def test_the_draw_is_the_one_of_the_round(world):
    contract = replicate.Contract.load(world.write())
    expected = draw_from_snapshot(world.snapshot, "seed-7", 8)
    the = replicate.the_round(contract)
    assert the.draw == expected and the.shown == {"round_id": "r000007", "round_seed": "seed-7", "irt_snapshot_id": world.snapshot["snapshot_id"], "n_tasks": 8, "scored_tasks": 8}
    assert replicate.round_notes(contract, the) == []
    older = world.view | {"round_id": "r000005", "round_seed": "seed-5", "head_version_id": h("older head"), "params_hash": h("older params")}
    contract = replicate.Contract.load(world.write(round=None, history=[older, world.view | {"round_id": "r000006", "round_seed": "seed-6"}]))
    assert replicate.the_round(contract).shown["round_id"] == "r000006"  # no open round: the last closed one
    the = replicate.the_round(contract, round_id="r000005")
    assert the.draw == draw_from_snapshot(world.snapshot, "seed-5", 8)
    assert [note["what"] for note in replicate.round_notes(contract, the)] == ["round-head", "round-params"]
    with pytest.raises(ConfigError, match="r000005, r000006|r000006, r000005"):
        replicate.the_round(contract, round_id="r000009")
    with pytest.raises(ConfigError, match="it shows: none"):
        replicate.the_round(replicate.Contract.load(world.write(round=None)))


def test_a_draw_with_a_removed_task_uses_the_archived_pool(world, monkeypatch):
    old = load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)
    removed = next(task_id for task_id in old.task_ids if task_id not in POOL.task_ids)
    selected = pool_for_draw((removed,))
    assert selected.task(removed) == old.task(removed)
    with pytest.raises(ContractError, match="not in the current math pool"):
        pool_for_draw((removed,), "current")
    # The evaluator and admission of a past round receive the same archive.
    # No GPU is needed to check that selection at the construction seam.
    class Evaluator:
        def __init__(self, store, box, pool, grades, **kwargs):
            self.pool, self.grader = pool, grades
    class Trainer:
        resident = None
    monkeypatch.setattr(replicate, "VllmEvaluator", Evaluator)
    monkeypatch.setattr(replicate, "LoraTrainer", lambda *args, **kwargs: Trainer())
    monkeypatch.setattr(replicate, "BaseOnly", lambda *args: SimpleNamespace())
    parts = replicate.parts(world.work, pool=selected, grades=Grades())
    assert parts.evaluator.pool is selected


def test_the_cli_replicates_a_draw_on_the_archived_pool(world, capsys, monkeypatch):
    old = load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)
    old_draw = draw_from_snapshot(IRTSampler(list(old.task_ids), world.params).public_snapshot(), world.view["round_seed"], world.view["n_tasks"])
    snapshot = dict(IRTSampler(list(old.task_ids), world.params).public_snapshot())
    for seed in range(1000):
        if set(old_draw.task_ids) - set(POOL.task_ids):
            break
        old_draw = draw_from_snapshot(snapshot, f"old-pool-{seed}", world.view["n_tasks"])
    removed = next(task_id for task_id in old_draw.task_ids if task_id not in POOL.task_ids)
    assert removed in draw_from_snapshot(snapshot, old_draw.round_seed, world.view["n_tasks"]).task_ids
    world.store = LocalStore(world.public)
    world.store.put(layout.irt_snapshot(snapshot["snapshot_id"])[len("public/"):], json.dumps(snapshot).encode())
    world.view = world.view | {"irt_snapshot_id": snapshot["snapshot_id"], "round_seed": old_draw.round_seed}
    world.write()
    fetch(world, capsys)
    real_parts = replicate.parts
    seen = []
    def capture(work, gpus=None, **kwargs):
        made = real_parts(work, gpus, **kwargs)
        seen.append(made.evaluator.pool)
        return made
    monkeypatch.setattr(replicate, "parts", capture)
    code, report = run(world, capsys)
    assert code == 0
    assert seen[0].sha256 == POOL_V2_0_SHA256 and removed in seen[0].task_ids
    assert next(note for note in report["differences"] if note["what"] == "pool")["text"].find("archived v2.0") >= 0


def test_first_tasks_is_a_prefix_of_the_round_and_a_seed_of_your_choice_needs_a_snapshot(world, tmp_path):
    contract = replicate.Contract.load(world.write())
    whole = draw_from_snapshot(world.snapshot, "seed-7", 8)
    the = replicate.the_round(contract, first_tasks=3)
    assert (the.draw.task_ids, the.draw.weights) == (whole.task_ids[:3], whole.weights[:3])  # what the box scores first, not a new draw of 3
    assert the.draw.task_ids != draw_from_snapshot(world.snapshot, "seed-7", 3).task_ids or the.draw.weights != draw_from_snapshot(world.snapshot, "seed-7", 3).weights
    assert the.shown["n_tasks"] == 8 and the.shown["scored_tasks"] == 3 and [n["what"] for n in replicate.round_notes(contract, the)] == ["first-tasks"]
    for bad in (0, 9):
        with pytest.raises(ConfigError, match="from 1 to 8"):
            replicate.the_round(contract, first_tasks=bad)
    with pytest.raises(ConfigError, match="needs --irt-snapshot"):
        replicate.the_round(contract, round_seed="mine")
    file = tmp_path / "irt.json"
    file.write_text(json.dumps(world.snapshot))
    the = replicate.the_round(contract, round_seed="mine", irt_snapshot=str(file))
    assert the.draw == draw_from_snapshot(world.snapshot, "mine", world.params.n_tasks) and the.shown["round_id"] is None
    other = dict(IRTSampler(list(TASKS[:20]), world.params).public_snapshot())
    file.write_text(json.dumps(other))
    with pytest.raises(ConfigError, match="the snapshot given is"):  # the round names its snapshot
        replicate.the_round(contract, irt_snapshot=str(file))
    file.write_text(json.dumps(world.snapshot | {"snapshot_id": "irt-forged"}))
    with pytest.raises(VerifyError):  # a snapshot is its own hash
        replicate.the_round(contract, round_seed="mine", irt_snapshot=str(file))


# ---- run ----


def test_run_is_the_round_of_the_box_for_one_batch(world, capsys):
    fetch(world, capsys)
    draw = draw_from_snapshot(world.snapshot, "seed-7", 8)
    world.farm.fails = {POOL.task(task).problem for task in draw.task_ids[-2:]}
    code, report = run(world, capsys, "--out", str(world.tmp / "report.json"))
    assert code == 0, report
    assert json.loads((world.tmp / "report.json").read_text()) == report
    assert report["status"] == "winner" and report["score"]["eligible"] and report["score"]["n_tasks"] == 8 and report["score"]["z"] > world.params.z_bar
    assert report["win_vs_null"] | {"score": None} == {"won": True, "status": "winner", "reason": report["reason"], "score": None}
    assert report["cap_hit"]["passed"] and report["cap_hit"]["rate"] == 0 and report["cap_hit"]["rollouts"] == 8
    assert report["null_outcomes"] == ["F" * 8] and report["outcomes"] == ["PPPPPPFF"]  # in the order of the draw
    assert report["rules_run"] == ["format", "tokens", "grader"] and report["valid"] and report["n_rollouts"] == 2
    assert report["round"] == {"round_id": "r000007", "round_seed": "seed-7", "irt_snapshot_id": world.snapshot["snapshot_id"], "n_tasks": 8, "scored_tasks": 8}
    assert report["box"] == {"params_hash": world.params.params_hash, "recipe_id": world.params.recipe_id, "n_gpus": 8}
    assert report["here"] == {"image_digest": plan.install_digest(), "code_digest": plan.code_digest(), "n_gpus": 2}
    assert [note["what"] for note in report["differences"]] == ["install", "gpus", "round-clock", "one-batch", "master", "free-gpus"]
    assert "box_round" not in report and not report["stopped_at_limit"] and set(report["phase_s"]) == {"training", "evaluation"}

    # The training job: the seed, the rows and the recipe of the box, on the GPUs of this run.
    ((env, job),), raw = world.job.calls, world.batch.read_bytes()
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1" and job["seed"] == train_seed("seed-7", sha256_hex(raw)) and report["batch_sha256"] == sha256_hex(raw)
    assert job["base_dir"] == str(world.work.resolve() / "cache" / world.head.base_hash) and job["recipe"]["image_digests"] == [plan.install_digest()]
    assert {key: value for key, value in job["recipe"].items() if key != "image_digests"} == {key: value for key, value in world.params.recipe.to_dict().items() if key != "image_digests"}
    assert [(row["output"], row["advantage"]) for row in job["rows"]] == [(OUTPUT, 0.5), (OUTPUT + " again", -0.5)]

    # The scoring: one server on each GPU of this run, every task of the draw in its order, the seeds of the round, null M and the LoRA.
    assert [server.env["CUDA_VISIBLE_DEVICES"] for server in world.farm.servers] == ["0", "1"]
    sent = {(body["model"], body["messages"][-1]["content"]): body for _, body in world.farm.bodies}
    assert {content for _, content in sent} == {POOL.task(task).problem for task in draw.task_ids}
    lora_hash = report["lora"]["lora_hash"]
    for task in draw.task_ids:
        mine, null = sent[lora_hash, POOL.task(task).problem], sent[NULL_MODEL, POOL.task(task).problem]
        assert mine["seed"] == null["seed"] == rollout_seed("seed-7", task, 0) and mine["max_tokens"] == world.params.max_output_tokens
    lora_dir = Path(report["lora"]["dir"])
    assert lora_dir == world.work.resolve() / "store" / "pending" / "rounds" / "local" / "local-l0" / "loras" / "000"
    assert sorted(path.name for path in lora_dir.iterdir()) == sorted(LORA_FILES) and report["lora"]["train_report"]["seed"] == job["seed"]
    assert (world.work / "cache" / world.head.base_hash).is_dir()  # the base stays for the next run

    # The two graders are the box's: 8 or fewer workers with the short limit for the file check, the default pool for the scoring.
    assert Grades.made == [{}, {"workers": replicate.GRADE_WORKERS, "timeout_s": replicate.GRADE_TIMEOUT_S}]


@pytest.mark.parametrize("kl_seq,kl_tok,floors,status,won", [
    (0.5, 0.2, {"min_kl_seq": 0.25}, "winner", True),
    (0.1, 0.2, {"min_kl_seq": 0.25}, "null_update", False),
    (0.5, 0.0014, {"min_kl_seq": 0.0, "min_kl_tok": 0.0014}, "winner", True),
    (0.5, 0.001399, {"min_kl_seq": 0.0, "min_kl_tok": 0.0014}, "null_update", False),
])
def test_merged_local_replication_uses_solo_and_the_official_kl_gate(world, build, monkeypatch, kl_seq, kl_tok, floors, status, won):
    """The official merged runner measures and judges a sequence or token floor."""
    from sn120_v2.types import TrainReport

    params = build.params(n_gpus=1, n_tasks=8, looks=(1.0,), eval_merged=True, kl_max_tokens=8192, **floors)
    local = dataclasses.replace(params, n_gpus=1)
    ticket = build.ticket(n_entries=1, params=params)
    artifact = build.artifact(ticket.head, ticket.entries[0], params)
    batch_sha256 = sha256_hex(batch_bytes())
    report = TrainReport(seed=train_seed(ticket.draw.round_seed, batch_sha256, params.train_seed_scope), steps=1, loss=0.1, step_norm=0.2, tokens=10, wall_s=1.0, gpu_class="cpu", image_digest=build.image)
    artifact = dataclasses.replace(artifact, report=report)
    work = world.work.resolve()
    lora_prefix = layout.pending_lora_prefix(replicate.ROUND_ID, replicate.LEASE_ID, 0)
    files = {}
    for name in artifact.files:
        path = work / "store" / lora_prefix.removeprefix("pending/") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = sha256_hex(path.read_bytes())
    artifact = dataclasses.replace(artifact, prefix=lora_prefix, files=files, lora_hash=manifest_digest(files), batch_sha256=sha256_hex(batch_bytes()))
    events = []

    class Trainer:
        def train(self, request):
            events.append("trained")
            return artifact

        def close(self):
            pass

    class Evaluator:
        http = None

        def solo_worker(self, ticket, trained, deadline, gpu):
            class Worker:
                def run(self, *, steal, result, judge, alive, deadline):
                    events.append("scored")
                    while claim := steal(gpu, NULL_MODEL):
                        model, start, stop = claim
                        for index in range(start, stop):
                            code = ("P" if index == 0 else "F") if model == NULL_MODEL else ("P" if index < 6 else "F")
                            result(model, 0, index, code)

            return Worker()

        def build_merged(self, round_id, head, artifact, params, gpu):
            events.append("merged")
            path = work / "merged"
            path.mkdir(exist_ok=True)
            return SimpleNamespace(path=path, base_hash=head.base_hash, kernel="fixed-order-v1")

        def _serve_one(self, *args, **kwargs):
            return "http://local"

        def close(self):
            pass

        def measure_kl(self, ticket, entries, null, tested):
            assert all(isinstance(entry, EntryResult) and entry.status == SCORED for entry in entries)
            assert set(tested) == {0}
            events.append("official-KL")
            return {"entries": {0: {"kl_seq": kl_seq, "kl_seq_se": 0.01, "kl_tok": kl_tok, "kl_k": 1, "kl_tokens": 100}}, "null": {}, "wall_s": 1.0}

    from sn120_v2.gpu.solo import SoloSettings
    monkeypatch.setattr(replicate.SoloRunner, "__init__", lambda self, *args, **kwargs: replicate.LocalRunner.__init__(self, *args, **kwargs) or setattr(self, "settings", SoloSettings(prewarm=False)))
    built = SimpleNamespace(store=LocalStore(work / "store"), trainer=Trainer(), evaluator=Evaluator(), image=build.image)
    raw = batch_bytes()
    result = replicate.run_round(raw, HOTKEY, params, local, ticket.head, ticket.draw, built, tokens=WordTokens(), clock=lambda: 1000)
    assert events == ["trained", "merged", "scored", "official-KL"]
    assert result["status"] == status and result["win_vs_null"]["won"] is won
    assert result["kl_gate"]["passed"] is won and result["kl_gate"]["measured"]
    assert result["kl_gate"]["kl_seq"] == kl_seq and result["kl_gate"]["kl_tok"] == kl_tok
    assert result["kl_gate"]["min_kl_seq"] == floors.get("min_kl_seq", 0.0)
    assert result["kl_gate"]["min_kl_tok"] == floors.get("min_kl_tok", 0.0)
    assert result["kl_gate"]["kl_max_tokens"] == 8192
    assert result["kl_gate"]["kl_seq_se"] == 0.01 and result["kl_gate"]["kl_k"] == 1 and result["kl_gate"]["kl_tokens"] == 100
    if not won:
        floor, value = ("min_kl_tok", kl_tok) if "min_kl_tok" in floors else ("min_kl_seq", kl_seq)
        assert f"below {floor} {floors[floor]:g}" in result["reason"] and result["score"] is None
    assert result["outcomes"] == ["PPPPPPFF"] and result["null_outcomes"] == ["PFFFFFFF"]
    assert result["phase_s"]["kl"] == 1


def test_the_report_exposes_the_cap_hit_gate_and_a_loss_against_null(world, capsys):
    fetch(world, capsys)
    draw = draw_from_snapshot(world.snapshot, "seed-7", 8)
    world.farm.fails = {POOL.task(task).problem for task in draw.task_ids}
    code, report = run(world, capsys)
    assert code == 0 and report["status"] == "lost" and report["score"]["n_tasks"] == 8
    assert report["win_vs_null"]["won"] is False and report["win_vs_null"]["status"] == "lost"
    assert report["cap_hit"] == {
        "passed": True, "rate": 0.0, "null_rate": 0.0,
        "limit": world.params.cap_hit_floor, "margin": world.params.cap_hit_margin,
        "floor": world.params.cap_hit_floor, "hits": 0, "rollouts": 8,
    }
    world.farm.fails = set()
    world.farm.capped = True
    code, report = run(world, capsys)
    assert code == 0 and report["status"] == "lost"
    assert report["cap_hit"] == {
        "passed": False, "rate": 1.0, "null_rate": 0.0,
        "limit": world.params.cap_hit_floor, "margin": world.params.cap_hit_margin,
        "floor": world.params.cap_hit_floor, "hits": 8, "rollouts": 8,
    }


def test_the_official_kl_selector_and_token_cap_use_scored_sample_zero_captures(tmp_path):
    from sn120_v2.gpu.kl import selected_rollouts, tokenize_rows
    from sn120_v2.types import ERROR
    outcomes = ("PF" + ERROR,)
    rows = [
        {"task_pos": 0, "sample_index": 0, "outcome": "P", "prompt": [{"role": "user", "content": "p0"}], "completion_text": "zero"},
        {"task_pos": 1, "sample_index": 1, "outcome": "F", "prompt": [{"role": "user", "content": "p1"}], "completion_text": "not sample zero"},
        {"task_pos": 1, "sample_index": 0, "outcome": "E", "prompt": [], "completion_text": "an error"},
        {"task_pos": 1, "sample_index": 0, "outcome": "F", "prompt": [{"role": "user", "content": "p1"}], "completion_text": "x"},
        {"task_pos": 2, "sample_index": 0, "outcome": "E", "prompt": [], "completion_text": "not eligible"},
        {"task_pos": 3, "sample_index": 0, "outcome": "-", "prompt": [], "completion_text": "not scored"},
    ]
    capture = tmp_path / "rollouts.jsonl"
    capture.write_text("\n".join(json.dumps(row) for row in reversed(rows)) + "\nnot-json\n")
    selected = selected_rollouts(capture, outcomes, "round-seed", "lora-hash", 128)
    assert [row["task_pos"] for row in selected] == [0, 1]
    assert selected_rollouts(capture, outcomes, "round-seed", "lora-hash", 1)[0]["task_pos"] in {0, 1}
    assert selected_rollouts(capture, outcomes, "round-seed", "lora-hash", 1) == selected_rollouts(capture, outcomes, "round-seed", "lora-hash", 1)

    class Tokens:
        def convert_tokens_to_ids(self, token):
            return 99

        def apply_chat_template(self, prompt, **kwargs):
            assert kwargs["add_generation_prompt"] is True and kwargs["enable_thinking"] is True
            return "prompt"

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return [10] * len(text)

    encoded = tokenize_rows(Tokens(), selected, max_tokens=2)
    assert [row["completion_ids"] for row in encoded] == [[10, 10], [10, 99]]
    assert len(tokenize_rows(Tokens(), selected[:1], max_tokens=0)[0]["completion_ids"]) == len("zero") + 1


def test_merged_fetch_requires_and_verifies_the_actual_fp32_master(world, capsys):
    params = dataclasses.replace(world.params, eval_merged=True)
    master_dir = world.tmp / "fp32"
    master_dir.mkdir()
    master_data = b"real fp32 weights"
    (master_dir / "master.safetensors").write_bytes(master_data)
    master_files = {"master.safetensors": sha256_hex(master_data)}
    master_hash = manifest_digest(master_files)
    head = HeadVersion.make(parent_version_id=None, base_hash=world.head.base_hash, base_fp32_master_digest=master_hash, loras=(), round_id="genesis", applied_at=1)
    manifest = HeadManifest(head, world.files, master_files, {})
    world.write(params=params, manifest=manifest)
    code, report = cli(capsys, "fetch", "--mining-json", str(world.tmp / "mining.json"), "--work", str(world.work), "--from", str(world.source))
    assert code == 2 and "--master-from" in report["reason"]
    code, report = cli(capsys, "fetch", "--mining-json", str(world.tmp / "mining.json"), "--work", str(world.work), "--from", str(world.source), "--master-from", str(master_dir))
    assert code == 0 and report["fetched"] == 4 and report["master_hash"] == master_hash
    assert report["base_hash"] == world.head.base_hash and report["master_digest"] == master_hash
    assert (report["base_files"], report["master_files"], report["fetched_bytes"]) == (3, 1, sum(map(len, BASE.values())) + len(master_data))
    assert report["base_bytes"] == sum(map(len, BASE.values())) and report["master_bytes"] == len(master_data)
    assert Path(report["master_dir"]) == world.work.resolve() / "cache" / master_hash
    assert (Path(report["master_dir"]) / "master.safetensors").read_bytes() == master_data
    code, again = cli(capsys, "fetch", "--mining-json", str(world.tmp / "mining.json"), "--work", str(world.work), "--from", str(world.source), "--master-from", str(master_dir))
    assert code == 0 and again["fetched"] == 0
    assert (again["base_fetched"], again["master_fetched"], again["fetched_bytes"]) == (0, 0, 0)
    assert again["base_dir"] == str(world.work.resolve() / "cache" / world.head.base_hash)
    assert again["master_dir"] == str(world.work.resolve() / "cache" / master_hash)

    rows = [{"name": name, "sha256": sha, "size": len(master_data), "url": (master_dir / name).as_uri()} for name, sha in master_files.items()]
    links = links_file(world, head_version_id=head.version_id, master={"id": master_hash, "files": rows})
    path = world.public / "read_links.json"
    path.write_text(json.dumps(links))
    assert set(replicate.base_links(replicate.Contract.load(str(world.tmp / "mining.json")), links, 1000)) == set(world.files) | {f"../{master_hash}/master.safetensors"}
    broken = dict(links)
    broken["master"] = {"id": h("wrong"), "files": rows}
    with pytest.raises(ConfigError, match="fp32 master"):
        replicate.base_links(replicate.Contract.load(str(world.tmp / "mining.json")), broken, 1000)

    # The ordinary public path downloads BOTH file sets from read_links.json.
    linked_work = world.tmp / "linked-work"
    contract = replicate.Contract.load(str(world.tmp / "mining.json"))
    linked = replicate.fetch(contract, linked_work, now=lambda: 1000)
    assert (linked["base_files"], linked["base_bytes"], linked["base_fetched"]) == (3, sum(map(len, BASE.values())), 3)
    assert (linked["master_files"], linked["master_bytes"], linked["master_fetched"]) == (1, len(master_data), 1)
    assert linked["master_digest"] == master_hash and linked["fetched"] == 4
    assert (Path(linked["base_dir"]) / "config.json").read_bytes() == BASE["config.json"]
    assert (Path(linked["master_dir"]) / "master.safetensors").read_bytes() == master_data
    assert not (linked_work / "fetch" / master_hash).exists()
    assert replicate.fetch(contract, linked_work, now=lambda: 1000)["fetched"] == 0

    # A broken resumed download never publishes a partial master directory.
    incomplete = world.tmp / "incomplete-work"
    master_dir.joinpath("master.safetensors").write_bytes(b"wrong fp32 weights")
    with pytest.raises(VerifyError, match="master.safetensors"):
        replicate.fetch(contract, incomplete, now=lambda: 1000)
    assert not (incomplete / "cache" / master_hash).exists()
    assert (incomplete / "cache" / world.head.base_hash).is_dir()
    master_dir.joinpath("master.safetensors").write_bytes(master_data)
    resumed = replicate.fetch(contract, incomplete, now=lambda: 1000)
    assert (resumed["base_fetched"], resumed["master_fetched"]) == (0, 1)
    assert (Path(resumed["master_dir"]) / "master.safetensors").read_bytes() == master_data


def test_merged_cli_constructs_fold_inputs_after_verifying_the_real_master(world, capsys, monkeypatch):
    params = dataclasses.replace(world.params, eval_merged=True)
    master_data = b"real fp32 master"
    master_files = {"master.safetensors": sha256_hex(master_data)}
    master_hash = manifest_digest(master_files)
    head = HeadVersion.make(parent_version_id=None, base_hash=world.head.base_hash, base_fp32_master_digest=master_hash, loras=(), round_id="genesis", applied_at=1)
    world.write(params=params, manifest=HeadManifest(head, world.files, master_files, {}))
    contract = replicate.Contract.load(str(world.tmp / "mining.json"))
    args = replicate.parser().parse_args(["run", "--mining-json", str(world.tmp / "mining.json"), "--hotkey", HOTKEY, "--work", str(world.work), "--gpus", "2", str(world.batch)])
    monkeypatch.setattr(replicate, "gpu_count", lambda: 2)
    monkeypatch.setattr(replicate, "parts", lambda *args, **kwargs: pytest.fail("parts/training must not start before the master is verified"))

    code, report = run(world, capsys)
    assert code == 2 and report["error"] == "ConfigError"
    assert "base files" in report["reason"] and "fetch command" in report["reason"]
    code, fetched = cli(capsys, "fetch", "--mining-json", str(world.tmp / "mining.json"), "--work", str(world.work), "--from", str(world.source))
    assert code == 2 and "--master-from" in fetched["reason"]
    master_source = world.tmp / "master-source"
    master_source.mkdir()
    (master_source / "master.safetensors").write_bytes(master_data)
    code, fetched = cli(capsys, "fetch", "--mining-json", str(world.tmp / "mining.json"), "--work", str(world.work), "--from", str(world.source), "--master-from", str(master_source))
    assert code == 0
    (world.work / "cache" / master_hash).rename(world.tmp / "saved-master")  # keep the verified fixture for the final case
    code, report = run(world, capsys)
    assert code == 2 and report["error"] == "ConfigError"
    assert "eval_merged" in report["reason"] and "tens of GB" in report["reason"] and "fetch command" in report["reason"]
    assert world.job.calls == [] and world.farm.servers == []

    master_dir = world.work.resolve() / "cache" / master_hash
    master_dir.mkdir(parents=True)
    (master_dir / "master.safetensors").write_bytes(b"not fp32")
    code, report = run(world, capsys)
    assert code == 2 and report["error"] == "ConfigError"
    assert "does not hash" in report["reason"] and "fetch command again" in report["reason"]
    assert not master_dir.exists() and world.job.calls == []

    (world.tmp / "saved-master").rename(master_dir)
    seen = {}

    def fake_parts(work, gpus=None, **kwargs):
        seen["fold_input"] = kwargs["fold_input"]
        evaluator = SimpleNamespace(pool=load_pool(), grader=None)
        return SimpleNamespace(trainer=SimpleNamespace(), evaluator=evaluator, resident=replicate.BaseOnly(LocalStore(work / "store"), work), image="image")

    monkeypatch.setattr(replicate, "parts", fake_parts)
    monkeypatch.setattr(replicate, "run_round", lambda *args, **kwargs: {"status": "winner"})
    assert replicate.run(args, contract)["status"] == "winner"
    assert isinstance(seen["fold_input"], replicate.FoldInputs)
    assert seen["fold_input"].root == world.work.resolve() / "fold-inputs"


def test_non_merged_run_does_not_require_the_master(world, capsys):
    fetch(world, capsys)
    assert not (world.work / "cache" / world.head.base_fp32_master_digest).exists()
    code, report = run(world, capsys)
    assert code == 0 and report["status"] == "lost"  # the fake grader rejects the file's output; the run still trains and scores
    assert not report["kl_gate"]["enabled"] and not report["kl_gate"]["measured"]


def test_run_stops_at_an_early_look_like_the_box_and_first_tasks_scores_the_start_of_the_round(world, capsys):
    fetch(world, capsys)
    code, report = run(world, capsys, "--first-tasks", "4")
    draw = draw_from_snapshot(world.snapshot, "seed-7", 8)
    assert code == 0 and report["round"]["scored_tasks"] == 4 and report["score"]["n_tasks"] == 4 and len(report["outcomes"][0]) == 4
    assert {body["messages"][-1]["content"] for _, body in world.farm.bodies} == {POOL.task(task).problem for task in draw.task_ids[:4]}
    assert report["differences"][-1]["what"] == "first-tasks"


def test_the_round_file_of_the_box_gives_the_round_and_a_comparison_of_null_m(world, capsys):
    fetch(world, capsys)
    draw = draw_from_snapshot(world.snapshot, "seed-7", 8)
    box_file = world.view | {"draw": draw.to_dict(), "null_outcomes": {"dublin": ["FFFPFFF-"]}}
    path = world.public / "rounds" / "r000007" / "round.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(box_file))
    world.write(round=None)  # mining.json shows no round: the file alone names the seed and the snapshot
    code, report = run(world, capsys, "--round-file", path.as_uri())
    assert code == 0 and report["round"]["round_seed"] == "seed-7"
    assert report["box_round"] == {"round_id": "r000007", "same_head": True, "same_draw": True, "null_m": {"rollouts_compared": 7, "same_outcome": 6, "passed_here": 0, "passed_on_box": 1}}
    null = report["null_outcomes"]
    other = replicate.against_box(box_file | {"head_version_id": h("older head")}, world.head, draw, null)
    assert other == {"round_id": "r000007", "same_head": False, "same_draw": True}  # another head: another model, nothing to compare
    moved = replicate.against_box(box_file | {"draw": draw.to_dict() | {"task_ids": list(reversed(draw.task_ids))}}, world.head, draw, null)
    assert moved["same_draw"] is False and "null_m" not in moved
    two = replicate.against_box(box_file | {"null_outcomes": {"a": ["F" * 8], "b": ["F" * 8]}}, world.head, draw, null)
    assert "null_m" not in two  # two boxes: the file does not say which tasks each one scored


def test_run_refuses_before_any_gpu_work(world, capsys):
    code, report = run(world, capsys)
    assert code == 2 and "run the fetch command first" in report["reason"]
    fetch(world, capsys)
    code, report = run(world, capsys, "--strict")
    assert code == 2 and "--strict" in report["reason"] and plan.install_digest() in report["reason"] and "The box has 8" in report["reason"]
    code, report = cli(capsys, "run", "--mining-json", str(world.tmp / "mining.json"), "--hotkey", HOTKEY, "--work", str(world.work), str(world.batch))
    assert code == 2 and "this machine has 3 GPU(s)" in report["reason"] and "the box has 8" in report["reason"]  # the default is the count of the box
    world.batch.write_bytes(batch_bytes(hotkey="someone-else"))
    code, report = run(world, capsys)
    assert code == 1 and report == {"valid": False, "rule": "hotkey", "reason": report["reason"]}
    world.batch.write_bytes(batch_bytes().replace(b'"reward": 0.0', b'"reward": 0.5'))
    code, report = run(world, capsys)
    assert code == 1 and report["rule"] == "grader"  # the official grade, as on the validator
    assert world.job.calls == [] and world.farm.servers == []
    (world.work / "cache" / world.head.base_hash / "config.json").write_bytes(b"changed")
    resident_module._SHARED.clear()  # a new process
    world.batch.write_bytes(batch_bytes())
    code, report = run(world, capsys)
    assert code == 2 and "Run the fetch command again" in report["reason"] and not (world.work / "cache" / world.head.base_hash).exists()


def test_a_train_object_of_the_file_reaches_the_trainer(world, capsys):
    fetch(world, capsys)
    world.batch.write_bytes(batch_bytes({"rank": 32}))
    code, report = run(world, capsys)
    assert code == 0 and report["train_applied"]["rank"] == 32 and world.job.calls[0][1]["recipe"]["rank"] == 32
    assert report["train_applied"]["steps"] == world.params.recipe.steps


def test_the_free_gpu_check_looks_only_at_the_gpus_of_the_run(world):
    built = replicate.parts(world.work, 2)  # GPU 2 is in use: not a GPU of this run
    built.evaluator.gpus_free(1024, 0.0)
    assert replicate.gpu_report(2) == "0, 3\n1, 5" and replicate.gpu_report() == SMI.strip()
    with pytest.raises(InfraError, match="still in use"):
        wait_gpus_free(1024, 0.0, report=replicate.gpu_report)


def test_the_backend_for_the_miner_tool_is_the_box_without_the_master(world, monkeypatch):
    monkeypatch.undo()  # the real builder
    monkeypatch.setattr(replicate, "GradePool", Grades)
    store, trainer, evaluator, image = replicate.backend(str(world.work))
    assert isinstance(store, LocalStore) and store.root == world.work.resolve() / "store" and image == plan.install_digest()
    assert isinstance(trainer.resident, replicate.BaseOnly) and trainer.resident is evaluator.resident
    assert evaluator.gpus_free is wait_gpus_free and evaluator.pool.sha256 == POOL.sha256  # the box's own check of all GPUs, and the public pool


def test_the_document_names_every_command_option_and_difference(world):
    text = (Path(replicate.__file__).parents[1] / "REPLICATE.md").read_text(encoding="utf-8")
    commands = [line.split()[3:] for line in text.splitlines() if line.startswith("python -m sn120_v2.replicate ")]
    assert {argv[0] for argv in commands} == {"compare", "fetch", "run"}
    for argv in commands:
        argv = ["8" if word == "<N>" else word for word in argv]
        replicate.parser().parse_args(argv)  # exits on an argument the tool does not have
    for warning in (
        "git clone https://github.com/AffineFoundation/affine", "git checkout v2", "python3.12 -m venv .venv",
        "python -m pip install -e './v2[gpu,envs,r2]'", "tens of GB", "`master` / `master_files`",
        "max_train_token_steps = steps × tokens", "Minibatches do **not** multiply", "22,130,000", "steps=16", "1,383,125",
        "does NOT train and does NOT measure `kl_seq` or `kl_tok`", "passing file check is NOT a passing KL gate",
        "params.min_kl_seq", "params.min_kl_tok", "kl_rollouts` (default **128**)", "kl_max_tokens` (**8192**", "sample-0",
        "win_vs_null", "cap_hit", "--first-tasks N", "one-GPU smoke test", "decided: true", "recipe_id",
        "vllm==0.32.0", "torch 2.13", "clean virtual environment", "eval_merged=false",
    ):
        assert warning in text
    assert "no-master" not in text and "this tool never folds" not in text
    documented_check = "python v2/miner/affine_miner.py check --mining-json https://affine.io/mining.json --hotkey <ss58> --tokenizer /data/affine/cache --grade batch.json"
    assert documented_check in text
    options = {option for name in ("compare", "fetch", "run") for option in long_options(name)}
    assert len(options) == 16 and all(f"`{option}" in text for option in options)
    pinned = dataclasses.replace(world.params, recipe=dataclasses.replace(world.params.recipe, code_digest=h("the box's code")))
    local, notes = replicate.local_params(pinned, "another install", "another code", 1)
    view = world.view | {"head_version_id": h("another head"), "params_hash": h("other params")}
    contract = replicate.Contract.load(world.write(round=view))
    notes += replicate.round_notes(contract, replicate.the_round(contract, first_tasks=2))
    assert len(notes) == 10 and all(f"| `{note['what']}` |" in text for note in notes)
    changed = {field.name for field in dataclasses.fields(local) if field.name != "recipe" and getattr(local, field.name) != getattr(pinned, field.name)}
    assert changed == {"n_gpus", "round_time_limit_s", "lease_timeout_s", "promote_reserve_s", "train_time_budget_s"} and all(f"`{name}`" in text for name in changed)

def long_options(name: str) -> set[str]:
    """The long options of one command, read from its parser."""
    command = replicate.parser()._subparsers._group_actions[0].choices[name]
    return {option for option in command._option_string_actions if option.startswith("--") and option != "--help"}
