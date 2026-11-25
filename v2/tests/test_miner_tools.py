"""The miner tools: the validator's own check and round logic, driven through the public contract."""

from __future__ import annotations

import copy
import dataclasses
import contextlib
import importlib.util
import io
import json
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from sn120_v2 import layout, public, publisher, sampling, sim
from sn120_v2.admission import Invalid, StoreAdmission
from sn120_v2.canonical import sha256_hex
from sn120_v2.envs.math_pool import MathPool, Task, ENV_ID, load_pool
from sn120_v2.errors import BatchRejected, ConfigError, ContractError, InfraError
from sn120_v2.irt import IRTSampler, draw_from_snapshot
from sn120_v2.params import MinerBound, Params, Recipe
from sn120_v2.types import NOT_RUN, NULL_MODEL, Draw, EvalChunk
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
        artifact = self.build.artifact(request.head, request.entry, request.params)
        report = dataclasses.replace(artifact.report, seed=request.seed)
        return dataclasses.replace(artifact, report=report)

    def evaluate(self, request):
        self.events.append("evaluate")
        self.requests.append(request)
        cut = slice(self.done, self.done + len(request.task_ids))
        self.done = cut.stop
        outcomes = {NULL_MODEL: (self.null[cut],), request.loras[0].lora_hash: (self.lora[cut],)}
        return EvalChunk(request.look, request.task_ids, outcomes, self.image)

    def close(self) -> None:
        self.events.append("close")


class CandidateBox(Box):
    """The standard evaluator's seams, without the heavyweight GPU packages."""

    def __init__(self, build, null="PFFFFFFF", lora="PPPPPPFF", image="", fail=False) -> None:
        super().__init__(build, null=null, lora=lora, image=image, fail=fail)
        self.config = types.SimpleNamespace(image_digest=self.image, work_dir=None)
        self._servers = [types.SimpleNamespace(alive=lambda: True)]
        self._urls = ["local"]
        self.rollouts = []
        self._grade_requests = []

    def _prepare(self, request):
        return [NULL_MODEL, *(lora.lora_hash for lora in request.loras)], 1, 1

    def clients_for(self, params):
        return 1

    def evaluate(self, request):
        chunk = self._evaluate(request)
        self.requests.append(request)
        return chunk

    def _evaluate(self, request):
        from sn120_v2.gpu.evaluator import samples_for_model

        models, _, _ = self._prepare(request)
        jobs = [(model, sample, index) for model in models for sample in range(samples_for_model(request.params, model)) for index in range(len(request.task_ids))]
        replies = [self._rollout(request, self._urls[0], job, []) for job in jobs]
        codes = self._grade(request.task_ids, jobs, replies, request)
        outcomes = {model: tuple("".join(codes[model, sample, index] for index in range(len(request.task_ids))) for sample in range(samples_for_model(request.params, model))) for model in models}
        return EvalChunk(request.look, request.task_ids, outcomes, self.image)

    def _rollout(self, request, url, job, written):
        model, sample, index = job
        text = self.lora if model != NULL_MODEL else self.null
        position = int(request.task_ids[index][1:]) % len(text)
        self.rollouts.append(job)
        return text[position], False

    def stream(self, request, results, eliminated, done):
        """One ordered callback per task, like the standard streaming evaluator."""
        from sn120_v2.gpu.evaluator import samples_for_model

        self._prepare(request)
        models = [NULL_MODEL, *(lora.lora_hash for lora in request.loras)]
        for index in range(len(request.task_ids)):
            if done():
                return index
            for model in models:
                if eliminated(model):
                    continue
                for sample in range(samples_for_model(request.params, model)):
                    job = (model, sample, index)
                    reply = self._rollout(request, self._urls[0], job, [])
                    code = self._grade(request.task_ids, [job], [reply], request)[job]
                    results(model, sample, index, code)
        return len(request.task_ids)

    def _grade(self, tasks, jobs, replies, request=None):
        self._grade_requests.append((tuple(jobs), request.task_ids))
        return {job: text[0] for job, (text, capped) in zip(jobs, replies)}

    def _cut(self, *args):
        raise InfraError("unexpected deadline in cache test")


class WordTokens:
    prompt_tokens = output_tokens = staticmethod(lambda text: len(text.split()))


def batch_bytes(train: dict | None = None, hotkey: str = HOTKEY) -> bytes:
    body = {"version": "affine-batch-v2", "hotkey": hotkey, "groups": [{"prompt": "What is 1+1?", "rollouts": [{"output": OUTPUT, "reward": 1.0}, {"output": OUTPUT + " again", "reward": 0.0}]}]}
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
    code, out = cli(capsys, "check", "--no-live-pool", "--mining-json", contract, "--hotkey", HOTKEY, str(batch))
    report = json.loads(out)
    assert code == 0 and report["valid"] and report["train_applied"]["steps"] == 7
    assert report["rules_not_run"] == ["tokens", "grader"]  # the tool says what it could not check
    code, out = cli(capsys, "check", "--no-live-pool", "--mining-json", contract, "--hotkey", "someone-else", str(batch))
    assert code == 1 and json.loads(out)["rule"] == "hotkey"
    batch.write_bytes(batch_bytes({"steps": 8}))
    code, out = cli(capsys, "check", "--no-live-pool", "--mining-json", contract, "--hotkey", HOTKEY, str(batch))
    assert code == 1 and json.loads(out)["rule"] == "train"
    with pytest.raises(SystemExit):  # no contract, no answer: the tool has no built-in limits
        tool.main(["check", "--hotkey", HOTKEY, str(batch)])


def test_check_warns_but_does_not_reject_an_unofficial_pool_prompt(build, monkeypatch):
    problem = "A synthetic problem: what is 17 + 25?"
    task = Task("synthetic:0", "synthetic", problem, "42")
    monkeypatch.setattr(tool, "load_pool", lambda: MathPool((task,), "synthetic-pool"))

    def raw(prompt: str) -> bytes:
        return json.dumps({"version": "affine-batch-v2", "hotkey": HOTKEY, "groups": [
            {"prompt": prompt, "rollouts": [{"output": OUTPUT, "reward": 1.0}, {"output": OUTPUT + " again", "reward": 0.0}]}
        ]}).encode()

    bare = tool.check(raw(task.problem), HOTKEY, build.params())
    assert bare["valid"] is True
    assert any("1 stored prompt differs" in warning and "task.batch_prompt()" in warning for warning in bare["warnings"])
    assert any("does NOT train and does NOT measure kl_seq" in warning for warning in bare["warnings"])
    official = tool.check(raw(task.batch_prompt()), HOTKEY, build.params())
    assert official["valid"] is True
    assert not any("task.batch_prompt()" in warning for warning in official["warnings"])


def test_check_skips_prompt_warning_when_the_bundled_pool_is_unavailable(build, monkeypatch):
    task = Task("synthetic:0", "synthetic", "A synthetic problem: what is 17 + 25?", "42")
    monkeypatch.setattr(tool, "load_pool", lambda: (_ for _ in ()).throw(OSError("pool unavailable")))
    raw = json.dumps({"version": "affine-batch-v2", "hotkey": HOTKEY, "groups": [
        {"prompt": task.problem, "rollouts": [{"output": OUTPUT, "reward": 1.0}, {"output": OUTPUT + " again", "reward": 0.0}]}
    ]}).encode()
    report = tool.check(raw, HOTKEY, build.params())
    assert report["valid"] is True
    assert len(report["warnings"]) == 1 and "does NOT train" in report["warnings"][0]


def test_cli_live_pool_warning_is_strictly_over_half_and_never_changes_validity(build, tmp_path, capsys, monkeypatch):
    task = Task("gen:1", "affine_math", "public one", "1")
    active = tmp_path / "tasks.jsonl"
    active.write_text(json.dumps({"task_id": task.task_id, "problem": task.problem, "answer": task.answer}) + "\n")
    body = {"version": "affine-batch-v2", "hotkey": HOTKEY, "groups": [
        {"prompt": task.batch_prompt(), "rollouts": [{"output": OUTPUT, "reward": 1.0}, {"output": OUTPUT, "reward": 0.0}]},
        *({"prompt": tool.load_pool().task(tool.load_pool().task_ids[i]).batch_prompt(),
           "rollouts": [{"output": OUTPUT, "reward": 1.0}, {"output": OUTPUT, "reward": 0.0}]} for i in range(2)),
    ]}
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(body))
    contract = mining_json(build, tmp_path)
    code, out = cli(capsys, "check", "--mining-json", contract, "--hotkey", HOTKEY, "--active", str(active), str(path))
    report = json.loads(out)
    assert code == 0 and report["valid"] is True
    assert any("2 mapped prompts are outside" in warning and "(1 in, 2 out of 3 mapped" in warning for warning in report["warnings"])
    code, out = cli(capsys, "check", "--no-live-pool", "--mining-json", contract, "--hotkey", HOTKEY, str(path))
    assert code == 0 and not any("outside the current official-active pool" in warning for warning in json.loads(out)["warnings"])


def test_live_pool_warning_threshold_and_local_active_file(build, tmp_path):
    from sn120_v2.admission import check_batch
    tasks = [Task("gen:0", "affine_math", "public 0", "0"),
             *(tool.load_pool().task(tool.load_pool().task_ids[i]) for i in range(2))]

    def raw(*prompts):
        return json.dumps({"version": "affine-batch-v2", "hotkey": HOTKEY, "groups": [
            {"prompt": prompt, "rollouts": [{"output": OUTPUT, "reward": 1.0}, {"output": OUTPUT, "reward": 0.0}]} for prompt in prompts
        ]}).encode()

    active = tmp_path / "tasks.jsonl"
    active.write_text("".join(json.dumps({"task_id": task.task_id, "problem": task.problem, "answer": task.answer}) + "\n" for task in tasks[:1]))
    for prompts, warning in (
        ((tasks[0].batch_prompt(), tasks[1].batch_prompt(), tasks[2].batch_prompt()), True),
        ((tasks[0].batch_prompt(), tasks[0].problem, tasks[1].batch_prompt(), tasks[2].batch_prompt()), False),
        ((tasks[0].batch_prompt(), "unmapped custom prompt", "another custom prompt"), False),
    ):
        report = tool.check(raw(*prompts), HOTKEY, build.params())
        batch = check_batch(raw(*prompts), HOTKEY, build.params())
        extra = tool.live_pool_guidance(batch, build.params(), public_url="https://unused", active=str(active))
        assert bool(extra) == warning and report["valid"] is True
        if warning:
            assert "(1 in, 2 out" in extra[0] and "active SHA" in extra[0] and "guidance only" in extra[0]


def test_live_pool_remote_check_maps_base_and_public_reserve_prompts(build, tmp_path, monkeypatch):
    from test_live_pool import public_files, mocked_public, tool as live_pool  # sn120-standalone-source-path
    from sn120_v2.admission import check_batch

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "miner"))
    monkeypatch.setitem(sys.modules, "live_pool", live_pool)
    files = public_files(tmp_path)
    requests = mocked_public(monkeypatch, files)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    base_task = tool.load_pool().task(tool.load_pool().task_ids[1])
    pointer, active_raw, mining = live_pool._pointer("https://files.example")
    catalog = live_pool._catalog("https://files.example", pointer, mining, tmp_path / "verify-cache")
    outside = catalog.task("gen:9")  # high-retired, hence mapped by the full catalog but absent from the active list
    assert outside.batch_prompt() not in tool.load_pool()._by_problem
    body = {"version": "affine-batch-v2", "hotkey": HOTKEY, "groups": [
        {"prompt": task.batch_prompt(), "rollouts": [{"output": OUTPUT, "reward": 1.0}, {"output": OUTPUT, "reward": 0.0}]}
        for task in (outside, base_task)
    ]}
    batch = check_batch(json.dumps(body).encode(), HOTKEY, build.params())
    warnings = tool.live_pool_guidance(batch, build.params(), public_url="https://files.example/mining.json")
    assert catalog.task_for_prompt(outside.batch_prompt()).task_id == "gen:9"
    assert catalog.task_for_prompt(base_task.batch_prompt()).task_id == base_task.task_id
    assert "gen:9" not in live_pool.active_ids(active_raw)
    assert warnings, "the live check returned no diagnostics"
    assert any("(0 in, 2 out of 2 mapped" in warning and "outside the current official-active pool" in warning
               for warning in warnings), warnings
    assert not any("could not check live pool" in warning for warning in warnings), warnings
    assert sum(key.startswith("public/pool/reserve/shards/") for key, _ in requests) >= 1


def test_live_pool_fetch_failure_adds_only_a_warning(build, monkeypatch):
    from sn120_v2.admission import check_batch
    batch = check_batch(batch_bytes(), HOTKEY, build.params())
    warnings = tool.live_pool_guidance(batch, build.params(), public_url="file:///definitely/missing")
    assert len(warnings) == 1 and "could not check live pool" in warnings[0] and "file stays valid" in warnings[0]
    assert tool.check(batch_bytes(), HOTKEY, build.params())["valid"] is True


def test_cli_reproduce_rebuilds_the_draw_from_the_public_irt_snapshot(build, tmp_path, capsys, monkeypatch):
    box = Box(build)
    monkeypatch.setitem(sys.modules, "fake_backend", types.SimpleNamespace(make=lambda workdir: (box, box, box, box.image)))
    batch, irt_file = tmp_path / "batch.json", tmp_path / "irt.json"
    batch.write_bytes(batch_bytes())
    snapshot = IRTSampler([f"t{i:03d}" for i in range(40)], build.params()).public_snapshot()
    irt_file.write_text(json.dumps(snapshot))
    argv = ["reproduce", "--mining-json", mining_json(build, tmp_path), "--hotkey", HOTKEY, "--irt-snapshot", str(irt_file)]
    code, out = cli(capsys, *argv, "--round-seed", "seed-9", "--backend", "fake_backend:make", "--train", '{"rank": 8}', str(batch))
    assert code == 0 and json.loads(out)["status"] == "winner" and json.loads(out)["train_applied"]["rank"] == 8 and json.loads(out)["train_report"]["seed"] == box.requests[0].seed
    expected = draw_from_snapshot(snapshot, "seed-9", build.params().n_tasks)  # n_tasks comes from the contract
    assert (box.requests[1].task_ids, box.requests[1].round_seed) == (expected.task_ids, "seed-9")


def test_null_cache_fills_then_reuses_the_same_null_for_separate_candidates(build, tmp_path):
    params = build.params(n_tasks=8, eval_chunk_tasks=4, looks=(0.5, 1.0), null_samples_per_task=1)
    cache = tmp_path / "null-cache"

    def cached_run(box):
        runner = LocalRunner(box, box, lambda: 7, box.image)
        runner = tool.cached_null_runner(runner, str(cache), build.head(), params, draw())
        return tool.reproduce(batch_bytes(), HOTKEY, params, build.head(), draw(), box, runner)

    first_box = CandidateBox(build, null="PFFFFFFF", lora="PPPPPPFF")
    first = cached_run(first_box)
    saved = json.loads((cache / tool.NULL_CACHE_MANIFEST).read_text())
    assert saved["null_outcomes"] == ["PFFFFFFF"]
    assert saved["draw"]["task_ids"] == list(draw().task_ids)
    assert saved["draw"]["weights"] == list(draw().weights)
    assert saved["head"] == build.head().to_dict()
    assert saved["params"]["null_samples_per_task"] == 1
    assert [job[0] for job in first_box.rollouts].count(NULL_MODEL) == 8

    second_box = CandidateBox(build, null="FFFFFFFF", lora="PPPPPPPP")
    second = cached_run(second_box)
    third_box = CandidateBox(build, null="FFFFFFFF", lora="PPPPPPFF")
    third = cached_run(third_box)
    assert second["status"] == "winner" and third["score"] == first["score"]
    for box in (second_box, third_box):
        assert len(box.rollouts) == 8 and len({model for model, _, _ in box.rollouts}) == 1
        assert all(model != NULL_MODEL for model, _, _ in box.rollouts)
        assert all(len(jobs) == 4 for jobs, _ in box._grade_requests)
    assert [task for _, tasks in second_box._grade_requests for task in tasks] == list(draw().task_ids)


def test_null_cache_keeps_full_null_when_the_fill_candidate_drops_early(build, tmp_path):
    params = build.params(n_tasks=16, eval_chunk_tasks=8, looks=(0.5, 1.0), null_samples_per_task=1)
    cache = tmp_path / "early-drop"
    first_box = Box(build, "PPPPPPPP" * 2, "FFFFFFFP" * 2)
    runner = tool.cached_null_runner(LocalRunner(first_box, first_box, lambda: 7, first_box.image), str(cache), build.head(), params, draw(16))
    first = tool.reproduce(batch_bytes(), HOTKEY, params, build.head(), draw(16), first_box, runner)
    assert first["eliminated_at_look"] == 0
    assert json.loads((cache / tool.NULL_CACHE_MANIFEST).read_text())["null_outcomes"] == ["PPPPPPPP" * 2]

    second_box = CandidateBox(build, "FFFFFFFF" * 2, "PPPPPPPP" * 2)
    runner = tool.cached_null_runner(LocalRunner(second_box, second_box, lambda: 7, second_box.image), str(cache), build.head(), params, draw(16))
    second = tool.reproduce(batch_bytes(), HOTKEY, params, build.head(), draw(16), second_box, runner)
    assert second["score"]["n_tasks"] == 16 and second["eliminated_at_look"] is None
    assert all(model != NULL_MODEL for model, _, _ in second_box.rollouts)
    assert [task for _, tasks in second_box._grade_requests for task in tasks] == list(draw(16).task_ids)


def test_solo_null_cache_fills_then_reuses_without_null_claims(build, tmp_path):
    """The real solo runner and scheduler, with a CPU worker in place of vLLM."""
    from sn120_v2.gpu.solo import SoloRunner, SoloSettings
    from sn120_v2.headchain import FIXED_ORDER
    from sn120_v2.judging import check_result

    params = build.params(n_tasks=16, n_gpus=2, looks=(0.5, 1.0), eval_chunk_tasks=4,
                          null_samples_per_task=2, eval_merged=True)
    cache = tmp_path / "solo-cache"

    def cached_run(null, lora):
        box = CandidateBox(build, null=null, lora=lora)

        class Evaluator:
            http = None

            def solo_worker(self, ticket, trained, deadline, gpu):
                class Worker:
                    def run(self, *, steal, result, judge, alive, deadline):
                        while claim := steal(gpu, NULL_MODEL):
                            model, start, stop = claim
                            for index in range(start, stop):
                                text = box.null if model == NULL_MODEL else box.lora
                                for sample in range(ticket.params.null_samples if model == NULL_MODEL else 1):
                                    box.rollouts.append((model, sample, index))
                                    result(model, sample, index, text[index])

                return Worker()

            def build_merged(self, round_id, head, artifact, params, gpu):
                return types.SimpleNamespace(path=tmp_path, base_hash=head.base_hash, kernel=FIXED_ORDER)

            def close(self):
                pass

        runner = SoloRunner(box, Evaluator(), lambda: 1000, box.image,
                            settings=SoloSettings(prewarm=False, steal_block_tasks=4), poll_s=0.01)
        runner = tool.cached_null_runner(runner, str(cache), build.head(), params, draw(16))
        result = runner.run(build.ticket(n_entries=1, n_tasks=16, params=params))
        check_result(build.ticket(n_entries=1, n_tasks=16, params=params), result)
        return result, box

    first, first_box = cached_run("P" * 16, "F" * 7 + "P" * 9)
    assert first.entries[0].eliminated_at_look == 0
    assert sum(model == NULL_MODEL for model, _, _ in first_box.rollouts) == 32
    assert json.loads((cache / tool.NULL_CACHE_MANIFEST).read_text())["null_outcomes"] == ["P" * 16] * 2

    second, second_box = cached_run("F" * 16, "P" * 16)
    assert second.entries[0].outcomes == ("P" * 16,)
    assert second.entries[0].eliminated_at_look is None
    assert not any(model == NULL_MODEL for model, _, _ in second_box.rollouts)
    assert all(text == "P" * 16 for text in second.null_outcomes)

    third, third_box = cached_run("P" * 16, "F" * 7 + "P" * 9)
    assert third.entries[0].eliminated_at_look == 0
    assert third.entries[0].outcomes == ("F" * 7 + "P" + NOT_RUN * 8,)
    assert all(text == "P" * 8 + NOT_RUN * 8 for text in third.null_outcomes)
    assert not any(model == NULL_MODEL for model, _, _ in third_box.rollouts)


def test_an_incomplete_or_unscored_fill_never_publishes_a_cache(build, tmp_path):
    cache = tool.NullCache(str(tmp_path / "incomplete"), build.head(), build.params(null_samples_per_task=1), draw())
    with pytest.raises(InfraError, match="outcome string"):
        cache.save(("PFFF",))
    with pytest.raises(InfraError, match="whole draw"):
        cache.save(("PFFFFFF-",))
    with pytest.raises(InfraError, match="wrong number"):
        cache.save(("PFFFFFFF", "PFFFFFFF"))
    assert not cache.directory.exists()


def test_null_cache_publish_is_atomic_and_never_replaces_a_fill(build, tmp_path):
    params, directory = build.params(null_samples_per_task=1), tmp_path / "atomic"
    first = tool.NullCache(str(directory), build.head(), params, draw())
    second = tool.NullCache(str(directory), build.head(), params, draw())
    first.save(("PFFFFFFF",))
    saved = (directory / tool.NULL_CACHE_MANIFEST).read_bytes()
    with pytest.raises(ConfigError, match="filled by another run"):
        second.save(("FFFFFFFF",))
    assert (directory / tool.NULL_CACHE_MANIFEST).read_bytes() == saved
    assert list(directory.iterdir()) == [directory / tool.NULL_CACHE_MANIFEST]
    assert tool.NullCache(str(directory), build.head(), params, draw()).outcomes == ("PFFFFFFF",)


def test_null_cache_refuses_mismatched_draw_head_or_knobs_and_bad_files(build, tmp_path):
    params, cache = build.params(null_samples_per_task=1), tmp_path / "cache"
    original = tool.NullCache(str(cache), build.head(), params, draw())
    original.save(("PFFFFFFF",))
    before = (cache / tool.NULL_CACHE_MANIFEST).read_bytes()
    assert tool.NullCache(str(cache), build.head(), params, draw()).outcomes == ("PFFFFFFF",)
    changed_draw = dataclasses.replace(draw(), task_ids=("t001", "t000", *draw().task_ids[2:]))
    other_head = type(build.head()).make(
        parent_version_id=None, base_hash=build.head().base_hash,
        base_fp32_master_digest=build.head().base_fp32_master_digest,
        loras=(), round_id="genesis", applied_at=1001,
    )
    for different in (
        lambda: tool.NullCache(str(cache), build.head(), params, changed_draw),
        lambda: tool.NullCache(str(cache), build.head(), params, dataclasses.replace(draw(), weights=(2.0,) + draw().weights[1:])),
        lambda: tool.NullCache(str(cache), other_head, params, draw()),
        lambda: tool.NullCache(str(cache), build.head(), dataclasses.replace(params, temperature=0.5), draw()),
        lambda: tool.NullCache(str(cache), build.head(), dataclasses.replace(params, null_samples_per_task=2), draw()),
    ):
        with pytest.raises(ConfigError, match="another draw, head, or scoring configuration"):
            different()
        assert (cache / tool.NULL_CACHE_MANIFEST).read_bytes() == before
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "unrelated").write_text("do not overwrite")
    with pytest.raises(ConfigError, match="nonempty.*manifest"):
        tool.NullCache(str(nonempty), build.head(), params, draw())
    (cache / tool.NULL_CACHE_MANIFEST).write_text("not json")
    with pytest.raises(ConfigError, match="cannot read"):
        tool.NullCache(str(cache), build.head(), params, draw())
    saved = json.loads(before)
    for outcomes in (["PFFF"], ["PFFFFFF-"], ["PFFFFFFX"]):
        (cache / tool.NULL_CACHE_MANIFEST).write_text(json.dumps(saved | {"null_outcomes": outcomes}))
        with pytest.raises(ConfigError, match="invalid null outcomes"):
            tool.NullCache(str(cache), build.head(), params, draw())
    (cache / tool.NULL_CACHE_MANIFEST).write_text(json.dumps(saved | {"cache_id": "wrong"}))
    with pytest.raises(ConfigError, match="cache_id"):
        tool.NullCache(str(cache), build.head(), params, draw())
    missing = dict(saved)
    del missing["null_outcomes"]
    (cache / tool.NULL_CACHE_MANIFEST).write_text(json.dumps(missing))
    with pytest.raises(ConfigError, match="missing its null outcomes"):
        tool.NullCache(str(cache), build.head(), params, draw())

    plain = tmp_path / "plain"
    plain.write_text("occupied")
    with pytest.raises(ConfigError, match="not a directory"):
        tool.NullCache(str(plain), build.head(), params, draw())


def test_cli_reproduce_accepts_a_cache_and_the_other_commands_refuse_it(build, tmp_path, capsys, monkeypatch):
    box = CandidateBox(build)
    monkeypatch.setitem(sys.modules, "fake_backend", types.SimpleNamespace(make=lambda workdir: (box, box, box, box.image)))
    batch, irt_file, cache = tmp_path / "batch.json", tmp_path / "irt.json", tmp_path / "cache"
    batch.write_bytes(batch_bytes())
    snapshot = IRTSampler([f"t{i:03d}" for i in range(40)], build.params()).public_snapshot()
    irt_file.write_text(json.dumps(snapshot))
    monkeypatch.setattr(tool.time, "time", lambda: 1000)
    argv = ["reproduce", "--mining-json", mining_json(build, tmp_path), "--hotkey", HOTKEY, "--irt-snapshot", str(irt_file)]
    code, out = cli(capsys, *argv, "--round-seed", "seed-9", "--backend", "fake_backend:make", "--null-cache", str(cache), str(batch))
    first_null_jobs = sum(model == NULL_MODEL for model, _, _ in box.rollouts)
    assert code == 0 and (cache / tool.NULL_CACHE_MANIFEST).is_file()
    assert first_null_jobs > 0
    code, out = cli(capsys, *argv, "--round-seed", "seed-9", "--backend", "fake_backend:make", "--null-cache", str(cache), str(batch))
    assert code == 0
    assert sum(model == NULL_MODEL for model, _, _ in box.rollouts) == first_null_jobs
    first_outcomes = json.loads((cache / tool.NULL_CACHE_MANIFEST).read_text())["null_outcomes"]
    assert json.loads((cache / tool.NULL_CACHE_MANIFEST).read_text())["draw"] == draw_from_snapshot(snapshot, "seed-9", build.params().n_tasks).to_dict()
    for name in ("check", "upload"):
        with pytest.raises(SystemExit):
            tool.main([name, "--mining-json", "unused", "--hotkey", HOTKEY, "--null-cache", str(cache), str(batch)])
        capsys.readouterr()
    assert cli(capsys, *argv, "--round-seed", "seed-9", "--backend", "fake_backend:make", str(batch))[0] == 0
    assert not (tmp_path / "manifest.json").exists()


def test_null_cache_flag_is_the_unchanged_runner_and_no_cache_io(build, tmp_path):
    runner = LocalRunner(Box(build), Box(build), lambda: 7, build.image)
    assert tool.cached_null_runner(runner, None, build.head(), build.params(), draw()) is runner
    assert list(tmp_path.iterdir()) == []


def test_the_reward_rule_runs_with_the_official_grader(build):
    pytest.importorskip("math_verify")
    grader, task = tool.official_grader(build.params()), load_pool().task("math:0")
    rows = lambda reward: [{"output": "step " * 40 + f"\\boxed{{{task.answer}}}", "reward": reward}, {"output": "step " * 40 + "\\boxed{no answer}", "reward": 1.0 - reward}]
    raw = lambda reward: json.dumps({"version": "affine-batch-v2", "hotkey": HOTKEY, "groups": [{"prompt": task.batch_prompt(), "rollouts": rows(reward)}]}).encode()
    assert tool.check(raw(1.0), HOTKEY, build.params(), WordTokens(), grader)["rules_not_run"] == []
    with pytest.raises(Invalid) as caught:
        tool.check(raw(0.0), HOTKEY, build.params(), WordTokens(), grader)
    assert caught.value.rule == "grader"


def test_tokenizer_cache_parent_resolves_only_the_contract_head(tmp_path, capsys, monkeypatch):
    cache, head_hash = tmp_path / "work" / "cache", "a" * 64
    head = cache / head_hash
    head.mkdir(parents=True)
    assert tool.resolve_tokenizer(str(cache), head_hash) == str(head.resolve())
    assert f"resolved cache parent {cache} to head {head}" in capsys.readouterr().out
    missing = tmp_path / "other-cache"
    missing.mkdir()
    assert tool.resolve_tokenizer(str(missing), head_hash) == str(missing)
    seen = []
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: seen.append(path) or object())))
    assert tool.load_tokens(str(cache), head_hash) is not None
    assert seen == [str(head.resolve())]


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
    assert dict(sent[0].header_items()) == {"Content-type": "application/octet-stream", "User-agent": tool.CLIENT["User-Agent"]}


def test_the_live_public_params_and_future_fields_are_readable_only_by_the_public_loader(build, tmp_path):
    """Frozen 2026-11-23 public params: a miner must not need a same-day code upgrade."""
    from sn120_v2.replicate import Contract

    frozen = json.loads((Path(__file__).parent / "data" / "params_live_073ef680.json").read_text())
    raw = frozen["params"]
    assert frozen["params_hash"] == frozen["contract_params_hash"] == "073ef680cf338de5e085100704e60f80ba96657d6d4e19d12995e4cb971ef6f8"
    params = Params.from_dict(raw)
    assert params.params_hash == frozen["params_hash"]
    assert (params.crown_gate, params.z_bar, params.z_field_bar) == ("argmax_symmetric", 0.0, 0.0)
    future = copy.deepcopy(raw)
    future.update(min_kl_seq=0.0, min_kl_tok=0.0, confirm_next_round=False)
    future["recipe"]["future_recipe"] = {"anything": [1, 2]}
    future["recipe"]["miner_bounds"][0]["future_bound"] = "new"
    with pytest.raises(ContractError, match="unknown field"):
        Params.from_dict(future)
    assert Params.from_public_dict(future) == params
    with pytest.raises(ContractError, match="expected an integer"):
        Params.from_public_dict(raw | {"n_tasks": "2000"})

    world = sim.SimWorld(tmp_path / "world", build.params())
    mining = tmp_path / "mining.json"
    mining.write_bytes(publisher.render(world.publisher.snapshots[-1])[0])
    document = json.loads(mining.read_text())
    document["params"] = future
    mining.write_text(json.dumps(document))
    assert tool.load_contract(str(mining))[0] == params
    assert Contract.load(str(mining)).params == params


def test_every_read_of_an_address_names_its_client(build, tmp_path, monkeypatch):
    """The public site answers 403 to the default client name of urllib on some files. The tool never sends that name."""
    sent, path = [], mining_json(build, tmp_path)
    raw = Path(path).read_bytes()

    def reply(request, timeout):
        sent.append(request)
        return contextlib.closing(io.BytesIO(raw))

    monkeypatch.setattr(tool.urllib.request, "urlopen", reply)
    params, head = tool.load_contract("https://files.example/public/mining.json")
    tool.upload(b"{}", "https://signed.example/put?sig=secret")
    assert (params, head) == tool.load_contract(path) and len(sent) == 2, "a file is read with no request"
    for request in sent:
        assert isinstance(request, urllib.request.Request), "a bare address gets the default client name of urllib"
        assert request.get_header("User-agent") == "sn120-v2-miner/1"
    opener = urllib.request.build_opener()
    assert dict(opener.addheaders)["User-agent"].startswith("Python-urllib/"), "the name that the site refuses"


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
    # A small two-rollout example cannot exercise the maximum recipe's active-rollout floor.
    train["minibatches"] = 1
    raw = batch_bytes(train, "hk11")
    assert tool.check(raw, "hk11", miner_params, WordTokens())["rules_not_run"] == ["grader"]
    key = doc["contract"]["batch_file"]["private_object"].replace("HOTKEY", "hk11")
    monkeypatch.setattr(tool.urllib.request, "urlopen", lambda request, timeout: world.store.put(key, request.data) or contextlib.nullcontext())
    tool.upload(raw, "https://signed.example/put")  # the link leads to the miner's private object
    contract.publish(public.snapshot(world.controller.state, world.params, 6, admission.verdicts(["hk11"])))
    doc, text = json.loads(files()[0]), files()[1].decode()
    assert doc["queue"]["next"][0]["file_valid"] and "| 0 | 11 | hk11 | 1011 | yes |" in text and doc["snapshot_digest"] in text
    assert b"step step" not in files()[0] + files()[1]  # the batch itself stays private
