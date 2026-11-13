"""The miner tools: the validator's own check and round logic, driven through the public contract."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from sn120_v2 import layout, public, publisher, sampling, sim
from sn120_v2.admission import Invalid, StoreAdmission
from sn120_v2.canonical import sha256_hex
from sn120_v2.envs.math_pool import load_pool
from sn120_v2.errors import BatchRejected, InfraError
from sn120_v2.irt import IRTSampler, draw_from_snapshot
from sn120_v2.params import MinerBound, Recipe
from sn120_v2.types import NULL_MODEL, Draw, EvalChunk
from sn120_v2.worker import LocalRunner

_SPEC = importlib.util.spec_from_file_location("affine_miner_tool", Path(__file__).resolve().parents[1] / "miner" / "affine_miner.py")
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)

HOTKEY = "hk1"
OUTPUT = "step " * 40 + "\\boxed{2}"


class Box(dict):
    """One fake for the store, the trainer and the evaluator, with one event log."""

    put, get, exists = dict.__setitem__, dict.__getitem__, dict.__contains__

    def __init__(self, build, null: str = "PFFFFFFF", lora: str = "PPPPPPFF", image: str = "", fail: bool = False) -> None:
        super().__init__()
        self.build, self.null, self.lora, self.image, self.fail = build, null, lora, image or build.image, fail
        self.events, self.requests, self.done = [], [], 0

    def size(self, key: str) -> int:
        return len(self[key])

    def sha256(self, key: str) -> str:
        return sha256_hex(self[key])

    def train(self, request):
        self.events.append("train")
        self.requests.append(request)
        if self.fail:
            raise InfraError("out of memory")
        return self.build.artifact(request.head, request.entry, request.params)

    def evaluate(self, request):
        self.events.append("evaluate")
        self.requests.append(request)
        cut = slice(self.done, self.done + len(request.task_ids))
        self.done = cut.stop
        outcomes = {NULL_MODEL: (self.null[cut],), request.loras[0].lora_hash: (self.lora[cut],)}
        return EvalChunk(request.look, request.task_ids, outcomes, self.image)

    def close(self) -> None:
        self.events.append("close")


class WordTokens:
    prompt_tokens = output_tokens = staticmethod(lambda text: len(text.split()))


def batch_bytes(train: dict | None = None, hotkey: str = HOTKEY) -> bytes:
    body = {"version": "affine-batch-v2", "hotkey": hotkey, "groups": [{"prompt": "What is 1+1?", "rollouts": [{"output": OUTPUT, "reward": 1.0}]}]}
    return json.dumps(body if train is None else body | {"train": train}).encode()


def draw(n: int = 8) -> Draw:
    return Draw("snap-0", "seed-0", tuple(f"t{i:03d}" for i in range(n)), (1.0,) * n)


def run(build, box: Box, raw: bytes | None = None, params=None, n: int = 8, **options) -> dict:
    runner = LocalRunner(box, box, lambda: 7, box.image)  # the validator's own box code
    return tool.reproduce(raw or batch_bytes(), HOTKEY, params or build.params(), build.head(), draw(n), box, runner, **options)


def test_reproduce_is_the_validators_round_for_one_batch(build):
    box, raw = Box(build), batch_bytes()
    report = run(build, box, raw)
    assert report["status"] == "winner" and report["score"]["z"] > 1 and report["score"]["eligible"]
    assert box.events == ["train", "close", "evaluate", "close"]  # the trainer frees the GPUs before scoring starts
    train, scoring = box.requests
    assert train.seed == sampling.train_seed("seed-0", sha256_hex(raw)) and train.gpu_ids == tuple(range(8))
    assert box[train.entry.batch.uri] == raw and train.entry.batch.batch_sha256 == report["batch_sha256"]
    assert (scoring.task_ids, scoring.round_seed, scoring.params) == (draw().task_ids, "seed-0", train.params)  # one draw for M and the LoRA
    assert scoring.loras[0].lora_hash == report["lora_hash"]


@pytest.mark.parametrize(
    "null, lora, image, status, why",
    [
        ("PPPPFFFF", "PPPPFFFF", "", "lost", "is not above 1"),
        ("PFFFFFFF", "CCCCCCCC", "", "lost", "cap-hit"),  # a capped reply is not a pass
        ("PFFFFFFF", "PEEEEEEE", "", "no_decision", "resolved"),  # an infrastructure failure is not a loss
        ("PFFFFFFF", "PPPPPPFF", "home-built", "unofficial", "image is not in the pinned set"),
    ],
)
def test_reproduce_reports_the_validators_decision(build, null, lora, image, status, why):
    report = run(build, Box(build, null, lora, image))
    assert report["status"] == status and why in report["reason"]


def test_reproduce_stops_at_an_early_look_like_the_validator(build):
    box = Box(build, "PPPPPPPP" * 2, "FFFFFFFP" * 2)
    report = run(build, box, params=build.params(n_tasks=16, looks=(0.5, 1.0)), n=16)
    assert (report["status"], report["eliminated_at_look"]) == ("lost", 0) and "scored on 8 tasks" in report["reason"]
    assert box.events == ["train", "close", "evaluate", "close"]


def test_train_values_pass_the_validators_gate_and_are_never_clamped(build):
    box = Box(build)
    report = run(build, box, batch_bytes({"rank": 32}))
    assert box.requests[0].entry.batch.train == {"rank": 32}
    assert report["train_applied"]["rank"] == 32 and report["train_applied"]["steps"] == Recipe().steps
    box = Box(build)
    assert run(build, box, train={"alpha": 64.0})["train_applied"]["alpha"] == 64.0
    assert box.requests[0].entry.batch.train == {"alpha": 64.0}
    for bad in ({"steps": 99}, {"lr": 1e-5}, {"target_modules": ["embed_tokens"]}):
        box = Box(build)
        with pytest.raises(BatchRejected):
            run(build, box, train=bad)
        assert box.events == [] and not box


def test_a_failed_trainer_is_no_decision_and_frees_the_gpus(build):
    box = Box(build, fail=True)
    report = run(build, box)
    assert report["status"] == "no_decision" and "out of memory" in report["reason"] and report["score"] is None
    assert box.events == ["train", "train", "train", "close"]  # tried again, then freed; nothing is scored


def mining_json(build, tmp_path, recipe: Recipe | None = None) -> str:
    world = sim.SimWorld(tmp_path / "world", build.params(recipe=recipe or Recipe(image_digests=(build.image,))))
    path = tmp_path / "mining.json"
    path.write_bytes(publisher.render(world.publisher.snapshots[-1])[0])  # the controller's own snapshot
    return str(path)


def cli(capsys, *argv: str) -> tuple[int, str]:
    code = tool.main(list(argv))
    return code, capsys.readouterr().out


def test_cli_bounds_and_check_read_everything_from_mining_json(build, tmp_path, capsys):
    bounds = tuple(MinerBound("steps", 2, 7, integer=True) if b.key == "steps" else b for b in Recipe().miner_bounds)
    contract = mining_json(build, tmp_path, Recipe(image_digests=(build.image,), steps=2, miner_bounds=bounds))
    code, out = cli(capsys, "bounds", "--mining-json", contract)
    assert code == 0 and "steps: default 2; allowed: a whole number from 2 to 7" in out
    assert all(f"  {bound.key}: default " in out for bound in bounds)
    batch = tmp_path / "batch.json"
    batch.write_bytes(batch_bytes({"steps": 7}))
    code, out = cli(capsys, "check", "--mining-json", contract, "--hotkey", HOTKEY, str(batch))
    report = json.loads(out)
    assert code == 0 and report["valid"] and report["train_applied"]["steps"] == 7
    assert report["rules_not_run"] == ["tokens", "grader"]  # the tool says what it could not check
    code, out = cli(capsys, "check", "--mining-json", contract, "--hotkey", "someone-else", str(batch))
    assert code == 1 and json.loads(out)["rule"] == "hotkey"
    batch.write_bytes(batch_bytes({"steps": 8}))
    code, out = cli(capsys, "check", "--mining-json", contract, "--hotkey", HOTKEY, str(batch))
    assert code == 1 and json.loads(out)["rule"] == "train"
    with pytest.raises(SystemExit):  # no contract, no answer: the tool has no built-in limits
        tool.main(["check", "--hotkey", HOTKEY, str(batch)])


def test_cli_reproduce_rebuilds_the_draw_from_the_public_irt_snapshot(build, tmp_path, capsys, monkeypatch):
    box = Box(build)
    monkeypatch.setitem(sys.modules, "fake_backend", types.SimpleNamespace(make=lambda workdir: (box, box, box, box.image)))
    batch, irt_file = tmp_path / "batch.json", tmp_path / "irt.json"
    batch.write_bytes(batch_bytes())
    snapshot = IRTSampler([f"t{i:03d}" for i in range(40)], build.params()).public_snapshot()
    irt_file.write_text(json.dumps(snapshot))
    argv = ["reproduce", "--mining-json", mining_json(build, tmp_path), "--hotkey", HOTKEY, "--irt-snapshot", str(irt_file)]
    code, out = cli(capsys, *argv, "--round-seed", "seed-9", "--backend", "fake_backend:make", "--train", '{"rank": 8}', str(batch))
    assert code == 0 and json.loads(out)["status"] == "winner" and json.loads(out)["train_applied"]["rank"] == 8
    expected = draw_from_snapshot(snapshot, "seed-9", build.params().n_tasks)  # n_tasks comes from the contract
    assert (box.requests[1].task_ids, box.requests[1].round_seed) == (expected.task_ids, "seed-9")


def test_the_reward_rule_runs_with_the_official_grader(build):
    pytest.importorskip("math_verify")
    grader, task = tool.official_grader(build.params()), load_pool().task("math:0")
    rows = lambda reward: [{"output": "step " * 40 + f"\\boxed{{{task.answer}}}", "reward": reward}]
    raw = lambda reward: json.dumps({"version": "affine-batch-v2", "hotkey": HOTKEY, "groups": [{"prompt": task.batch_prompt(), "rollouts": rows(reward)}]}).encode()
    assert tool.check(raw(1.0), HOTKEY, build.params(), WordTokens(), grader)["rules_not_run"] == []
    with pytest.raises(Invalid) as caught:
        tool.check(raw(0.0), HOTKEY, build.params(), WordTokens(), grader)
    assert caught.value.rule == "grader"


def test_upload_checks_first_and_sends_the_signed_header(build, tmp_path, capsys, monkeypatch):
    sent, batch, link = [], tmp_path / "batch.json", "https://signed.example/put?sig=secret"
    monkeypatch.setattr(tool.urllib.request, "urlopen", lambda request, timeout: sent.append(request) or contextlib.nullcontext())
    batch.write_bytes(batch_bytes())
    argv = ["upload", "--mining-json", mining_json(build, tmp_path), str(batch), "--hotkey"]
    assert cli(capsys, *argv, HOTKEY)[0] == 2 and not sent  # no link, no upload
    monkeypatch.setenv(tool.PUT_URL_ENV, link)
    assert cli(capsys, *argv, "someone-else")[0] == 1 and not sent  # an invalid file is never sent
    assert cli(capsys, *argv, HOTKEY, "--grade")[0] == 2 and not sent  # no token count: not the validator's verdict
    code, out = cli(capsys, *argv, HOTKEY)
    assert code == 0 and json.loads(out)["uploaded"] and "secret" not in out
    assert (sent[0].full_url, sent[0].get_method(), sent[0].data) == (link, "PUT", batch_bytes())
    assert dict(sent[0].header_items()) == {"Content-type": "application/octet-stream"}


def test_a_miner_needs_only_the_two_public_files(build, tmp_path, monkeypatch):
    """Read the contract, check, upload, see the verdict, see the queue: with llms.txt and mining.json alone."""
    world = sim.SimWorld(tmp_path / "world", build.params())
    world.register(11, None)  # hk11 has a seat and no file yet
    admission = StoreAdmission(world.store, world.params, tokens=WordTokens(), grader=None, clock=lambda: 5)
    contract = publisher.StorePublisher(world.store)
    files = lambda: (world.store.get(layout.MINING_JSON), world.store.get(layout.LLMS_TXT))
    contract.publish(public.snapshot(world.controller.state, world.params, 5, admission.verdicts(["hk11"])))
    (tmp_path / "mining.json").write_bytes(files()[0])
    doc = json.loads(files()[0])
    assert doc["queue"]["next"][0]["hotkey"] == "hk11" and not doc["queue"]["next"][0]["file_valid"]
    miner_params, _ = tool.load_contract(str(tmp_path / "mining.json"))
    train = {row["key"]: row["choices"][:1] or (int(row["high"]) if row["integer"] else row["high"]) for row in doc["contract"]["miner_settable"]}
    raw = batch_bytes(train, "hk11")
    assert tool.check(raw, "hk11", miner_params, WordTokens())["rules_not_run"] == ["grader"]
    key = doc["contract"]["batch_file"]["private_object"].replace("HOTKEY", "hk11")
    monkeypatch.setattr(tool.urllib.request, "urlopen", lambda request, timeout: world.store.put(key, request.data) or contextlib.nullcontext())
    tool.upload(raw, "https://signed.example/put")  # the link leads to the miner's private object
    contract.publish(public.snapshot(world.controller.state, world.params, 6, admission.verdicts(["hk11"])))
    doc, text = json.loads(files()[0]), files()[1].decode()
    assert doc["queue"]["next"][0]["file_valid"] and "| 0 | 11 | hk11 | 1011 | yes |" in text and doc["snapshot_digest"] in text
    assert b"step step" not in files()[0] + files()[1]  # the batch itself stays private
