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
from sn120_v2.envs.math_pool import load_pool  # noqa: E402
from sn120_v2.errors import ConfigError, InfraError, VerifyError  # noqa: E402
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
from sn120_v2.types import NULL_MODEL, HeadManifest, HeadVersion  # noqa: E402

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
        passed = body["model"] != NULL_MODEL and body["messages"][-1]["content"] not in self.fails
        return {"choices": [{"message": {"content": "P ok" if passed else "F no"}, "finish_reason": "stop"}]}


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
    monkeypatch.setattr(replicate, "parts", lambda work, gpus=None: real_parts(work, gpus, train_spawn=job, serve=serve))
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
    assert same is params and [note["what"] for note in notes] == ["one-batch", "no-master", "free-gpus"]  # the box's own params, untouched
    local, notes = replicate.local_params(params, "my-image", plan.code_digest(), 1)
    assert [note["what"] for note in notes] == ["install", "code", "gpus", "round-clock", "one-batch", "no-master", "free-gpus"]
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
    assert [note["what"] for note in report["differences"]] == ["install", "gpus", "round-clock", "one-batch", "no-master", "free-gpus"]
    assert world.farm.servers == [] and world.job.calls == [] and not world.work.exists()

    def no_driver():
        raise InfraError("nvidia-smi is not there")

    monkeypatch.setattr(replicate, "nvidia_smi", no_driver)
    assert cli(capsys, "compare", "--mining-json", str(world.tmp / "mining.json"))[1]["here"]["gpus_found"] is None


# ---- the cache without the master ----


def test_the_cache_answers_for_the_master_and_verifies_everything_else(world, tmp_path):
    store = LocalStore(tmp_path / "s")
    cache = replicate.BaseOnly(store, tmp_path / "w")
    master = world.head.base_fp32_master_digest
    assert cache.tree(layout.public_master_prefix(master), master) == cache.root / master and not (cache.root / master).exists()
    with pytest.raises(VerifyError):  # the base is not there and not in the store
        cache.tree(layout.public_base_prefix(world.head.base_hash), world.head.base_hash)
    with pytest.raises(VerifyError):  # the box's own cache does not answer for a master
        Resident(store, tmp_path / "w").tree(layout.public_master_prefix(master), master)
    with pytest.raises(VerifyError):  # a LoRA is verified too
        cache.tree("pending/loras/x", h("lora"))


# ---- fetch ----


def test_fetch_from_a_directory_checks_each_file_and_fills_the_cache_in_one_move(world, capsys):
    (world.source / "model-00001.safetensors").write_bytes(b"other weights")
    code, report = cli(capsys, "fetch", "--mining-json", str(world.tmp / "mining.json"), "--work", str(world.work), "--from", str(world.source))
    cache = world.work / "cache" / world.head.base_hash
    assert code == 2 and report["error"] == "VerifyError" and "model-00001.safetensors is not the file of the head manifest" in report["reason"]
    assert not cache.exists()  # a half-filled set never reaches the cache: the box code would delete it
    (world.source / "model-00001.safetensors").write_bytes(BASE["model-00001.safetensors"])
    (world.source / "README.md").write_text("not a file of the head")  # more files in the source are ignored
    assert fetch(world, capsys) == {"base_hash": world.head.base_hash, "dir": str(cache), "files": 3, "fetched": 2}  # the first file was kept
    assert {path.name: path.read_bytes() for path in cache.iterdir()} == BASE
    assert fetch(world, capsys)["fetched"] == 0  # a second fetch only verifies
    (cache / "extra.txt").write_text("x")
    code, report = cli(capsys, "fetch", "--mining-json", str(world.tmp / "mining.json"), "--work", str(world.work), "--from", str(world.source))
    assert code == 2 and "does not hash to base_hash" in report["reason"] and (cache / "extra.txt").exists()  # fetch deletes nothing of yours


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
    assert code == 0 and json.loads((world.tmp / "report.json").read_text()) == report
    assert report["status"] == "winner" and report["score"]["eligible"] and report["score"]["n_tasks"] == 8 and report["score"]["z"] > world.params.z_bar
    assert report["null_outcomes"] == ["F" * 8] and report["outcomes"] == ["PPPPPPFF"]  # in the order of the draw
    assert report["rules_run"] == ["format", "tokens", "grader"] and report["valid"] and report["n_rollouts"] == 2
    assert report["round"] == {"round_id": "r000007", "round_seed": "seed-7", "irt_snapshot_id": world.snapshot["snapshot_id"], "n_tasks": 8, "scored_tasks": 8}
    assert report["box"] == {"params_hash": world.params.params_hash, "recipe_id": world.params.recipe_id, "n_gpus": 8}
    assert report["here"] == {"image_digest": plan.install_digest(), "code_digest": plan.code_digest(), "n_gpus": 2}
    assert [note["what"] for note in report["differences"]] == ["install", "gpus", "round-clock", "one-batch", "no-master", "free-gpus"]
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
        replicate.parser().parse_args(argv)  # exits on an argument the tool does not have
    options = {option for name in ("compare", "fetch", "run") for option in long_options(name)}
    assert len(options) == 14 and all(f"`{option}" in text for option in options)
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
