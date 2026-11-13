"""The smoke ramp on CPU: the exact parameter files of each stage, and the small copies of real miner files."""

from __future__ import annotations

import dataclasses
import json
import shutil
import types
from pathlib import Path

import pytest

from sn120_v2 import layout, smoke
from sn120_v2.admission import StoreAdmission, final_answer
from sn120_v2.canonical import manifest_digest, sha256_hex
from sn120_v2.cli import main
from sn120_v2.irt import IRTSampler
from sn120_v2.journal import JOURNAL_FILE
from sn120_v2.params import GenesisSpec, Params, load_params
from sn120_v2.sim import SIM_IMAGE, STORE_DIR, SimAdmission
from sn120_v2.state import load
from sn120_v2.store import LocalStore
from sn120_v2.types import ChainSnapshot, Registration

OLD_STATE = Path(__file__).parent / "data" / "state_515d7e9f" / "idle"
RUN = ["--rounds", "1", "--max-idle", "1", "--idle-sleep", "0"]
STAGES = ["1-tiny", "2-two", "3-eight", "4-multi", "5-full"]


def bootstrap(directory: Path, params: Path) -> None:
    """A first start through the command line with the CPU stack and 80 registered miners."""
    inputs = directory.parent / "inputs"
    inputs.mkdir()
    base_files, master_files = {"model.bin": sha256_hex(b"base")}, {"master.bin": sha256_hex(b"master")}
    genesis = GenesisSpec("org/model", "rev", manifest_digest(base_files), sha256_hex(b"digest"), "5Validator")
    chain = ChainSnapshot(120, 50, 60, (Registration(0, "owner", 1), *(Registration(uid, f"hk{uid}", 100 + uid, 1) for uid in range(1, 81))))
    (inputs / "genesis.json").write_bytes(genesis.to_json())
    (inputs / "chain.json").write_bytes(chain.to_json())
    (inputs / "genesis_files.json").write_text(json.dumps({"base_files": base_files, "master_files": master_files}))
    store = LocalStore(directory / STORE_DIR)
    store.put(layout.public_base_file(genesis.base_hash, "model.bin"), b"base")
    store.put(layout.public_master_file(genesis.base_fp32_master_digest, "master.bin"), b"master")
    files = [f"--{name}={inputs / file}" for name, file in (("genesis", "genesis.json"), ("genesis-files", "genesis_files.json"), ("chain", "chain.json"))]
    assert main(["bootstrap", "--state-dir", str(directory), "--stack", "sn120_v2.sim:stack", "--params", str(params), *files]) == 0


@pytest.mark.parametrize("start", ["a new bootstrap", "the state of 515d7e9f"])
def test_the_five_parameter_files_of_the_ramp_run_one_round_each_on_one_bootstrap(tmp_path, start, capsys):
    (tmp_path / "plan.json").write_text(json.dumps({"image_digest": SIM_IMAGE, "code_digest": ""}))
    assert smoke.main(["params", "--plan", str(tmp_path / "plan.json"), "--out", str(tmp_path)]) == 0
    files = {stage: tmp_path / f"params-{stage}.json" for stage in STAGES}
    assert capsys.readouterr().out.split() == [str(path) for path in files.values()]
    directory = tmp_path / "state"
    stack = ["--state-dir", str(directory), "--stack", "sn120_v2.sim:stack"]
    if start == "a new bootstrap":
        bootstrap(directory, files["1-tiny"])
    else:
        shutil.copytree(OLD_STATE, directory)  # bootstrapped by the older commit with the first params.json of SMOKE.md
    admission = SimAdmission(LocalStore(directory / STORE_DIR))
    heads = len(load(directory / JOURNAL_FILE).heads)
    seen: dict[str, tuple] = {}
    for stage in STAGES:
        assert main(["set-params", *stack, "--params", str(files[stage])]) == 0
        state = load(directory / JOURNAL_FILE)
        params = state.params
        assert params.params_hash == load_params(files[stage]).params_hash, "the file is what is pinned"
        seats = [seat.reg.hotkey for seat in sorted(state.queue.values(), key=lambda seat: (seat.reg.registration_block, seat.reg.uid))]
        for index, hotkey in enumerate(seats[: params.batch_cap]):
            admission.upload(hotkey, 0.3 if index == 0 else 0.0)  # the first batch of each round is a good one
        assert main(["run", *stack, *RUN]) == 0
        state = load(directory / JOURNAL_FILE)
        record = state.rounds[-1]
        statuses = [entry.status for entry in record.decision.entries]
        seen[stage] = (record.outcome, record.n_tasks, len(statuses), params.n_gpus)
        assert state.round is None and state.head.folded and "no_decision" not in statuses and "not_trained" not in statuses
        draw = IRTSampler([f"math:{i}" for i in range(3000)], params).draw(state.head, record.round_seed, params.n_tasks)
        assert len(draw.task_ids) == params.n_tasks and params.look_sizes()[-1] == params.n_tasks, "the real sampler draws a round of this size"
    assert seen == {
        "1-tiny": ("null", 1, 1, 1),  # one task has no standard error: this round can never have a winner
        "2-two": (seen["2-two"][0], 8, 2, 2),
        "3-eight": ("applied", 32, 8, 8),
        "4-multi": (seen["4-multi"][0], 8, 2, 8),
        "5-full": ("applied", 2000, 32, 8),
    }
    applied = sum(outcome == "applied" for outcome, *_ in seen.values())
    assert len(load(directory / JOURNAL_FILE).heads) == heads + 2 * applied, "each winner was applied and folded"
    full = load_params(files["5-full"])
    assert full == Params(recipe=full.recipe) and (full.z_bar, full.min_scored_tasks, full.batch_cap) == (1.0, 100, 32), "the last file is the launch values"
    small = load_params(files["1-tiny"])
    assert (small.z_bar, small.min_scored_tasks, small.max_output_tokens) == (0.0, 2, 2048)
    assert smoke.main(["params", "--plan", str(tmp_path / "plan.json"), "--out", str(tmp_path), "--short-reply", "4096"]) == 0
    caps = [load_params(files[stage]).max_output_tokens for stage in STAGES]
    assert caps == [4096, 4096, 65536, 4096, 65536], "a higher low cap, for real files with no group short enough"


# --- small copies of real miner files ---

THINK = "step " * 40  # 40 reasoning words: above the least of 32


class WordTokens:
    def prompt_tokens(self, prompt: str) -> int:
        return 5 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        return len(text.split())


def grade(prompt: str, output: str, capped: bool) -> float | None:
    """A toy official grader: the task "double N" has the answer 2N."""
    return float(not capped and final_answer(output) == str(2 * int(prompt.split()[1])))


def group(n: int, wrong: int = 1, right: int = 1, words: int = 0) -> dict:
    """One prompt with right and wrong rollouts. ``words`` more words make the first rollout long."""
    rows = [{"output": f"{THINK}{'more ' * words}r{k} \\boxed{{{2 * n}}}", "reward": 1} for k in range(right)]
    rows += [{"output": f"{THINK}w{k} \\boxed{{-1}}", "reward": 0} for k in range(wrong)]
    return {"prompt": f"double {n}", "rollouts": rows}


def miner_file(hotkey: str, groups: list[dict], train: dict | None = None) -> bytes:
    return json.dumps({"version": "affine-batch-v3", "hotkey": hotkey, "groups": groups} | ({"train": train} if train else {})).encode()


@pytest.fixture
def stores(tmp_path, build):
    """v1's bucket with five miner files, and the smoke store with its admission under a cap of 60 tokens."""
    src, dst = LocalStore(tmp_path / "v1"), LocalStore(tmp_path / "smoke")
    long = dict(words=100)  # 142 tokens: above the cap
    files = {
        "hkA": [group(1, **long), group(2, wrong=0, right=2), group(3), group(4, **long), group(5, right=2), group(6)],
        "hkB": [group(7, **long), group(8, wrong=0, right=2)],  # its only group with two rewards is too long
        "hkC": [group(9, wrong=0, right=3)],  # one reward only
        # hkD has no file.
        "hkE": [group(10, **long), group(11)],
        "hkF": [group(12)],
    }
    for hotkey, groups in files.items():
        src.put(layout.miner_object(hotkey), miner_file(hotkey, groups, train={"steps": 4} if hotkey == "hkA" else None))
    params = build.params(max_output_tokens=60)
    return src, dst, StoreAdmission(dst, params, tokens=WordTokens(), grader=grade), files


def test_a_small_copy_keeps_whole_short_groups_of_the_real_file_and_passes_admission(stores):
    src, dst, admission, files = stores
    hotkeys = ["hkA", "hkB", "hkC", "hkD", "hkE", "hkF"]
    before = {key: src.sha256(key) for key in src.list("")}
    copied, passed = smoke.copy_files(src, dst, hotkeys, 2, smoke.admission_cut(admission, groups=2))
    assert [line.split()[1] for line in copied] == ["hkA", "hkE"], "the first two files with a usable group, in the given order"
    assert "2 of 6 groups, 5 rollouts, longest 42 tokens" in copied[0] and "1 of 2 groups, 2 rollouts" in copied[1] and "default recipe" in copied[0]
    assert passed == [
        "passed over hkB: no group with two different rewards has every rollout within 60 tokens; the nearest group needs 142",
        "passed over hkC: no group has two different rewards",
    ]
    assert {key: src.sha256(key) for key in src.list("")} == before, "v1's bucket is only read"
    small = json.loads(dst.get(layout.miner_object("hkA")))
    assert small["groups"] == [files["hkA"][2], files["hkA"][4]] or small["groups"] == [files["hkA"][2], files["hkA"][5]], "whole groups, unchanged"
    assert set(small) == {"version", "hotkey", "groups"}, "the train object is left out: the copy trains with the default recipe"
    assert not dst.exists(layout.miner_object("hkB")) and not dst.exists(layout.miner_object("hkF"))
    verdicts = admission.verdicts(hotkeys)
    assert {hotkey: verdict.valid for hotkey, verdict in verdicts.items()} == {"hkA": True, "hkE": True}
    # The whole file of the same miner is not valid under the low cap: the trap that the small copy avoids.
    copied, _ = smoke.copy_files(src, dst, hotkeys, 1)
    assert "whole file" in copied[0] and 'train={"steps": 4}' in copied[0] and dst.get(layout.miner_object("hkA")) == src.get(layout.miner_object("hkA"))
    verdict = admission.verdicts(["hkA"])["hkA"]
    assert (verdict.valid, verdict.rule) == (False, "output_tokens")


def test_a_small_copy_is_refused_when_its_short_groups_fail_a_share_rule(stores):
    src, dst, admission, _ = stores
    answer_only = {"prompt": "double 20", "rollouts": [{"output": "\\boxed{40}", "reward": 1}, {"output": "\\boxed{-1}", "reward": 0}]}
    src.put(layout.miner_object("hkG"), miner_file("hkG", [group(21, words=100), answer_only]))
    copied, passed = smoke.copy_files(src, dst, ["hkG"], 1, smoke.admission_cut(admission, groups=2))
    assert copied == [] and passed == ["passed over hkG: a copy of its short groups fails the check: answer_only"] and dst.list("") == []


def test_the_copy_command_serves_the_next_round_and_only_a_smoke_prefix(stores, monkeypatch, capsys):
    src, dst, admission, _ = stores
    state = load(OLD_STATE / JOURNAL_FILE)
    order = [seat.reg.hotkey for seat in sorted(state.queue.values(), key=lambda seat: (seat.reg.registration_block, seat.reg.uid))][:3]
    for hotkey, n in zip(order, (30, 31, 32)):
        src.put(layout.miner_object(hotkey), miner_file(hotkey, [group(n)]))
    params = dataclasses.replace(admission.params, batch_cap=2)
    monkeypatch.setattr(smoke, "read_state", lambda args, command: (state, params, types.SimpleNamespace(store=dst, admission=admission)))
    monkeypatch.setattr(smoke, "R2Store", lambda **settings: src)
    for name in ("V1_R2_ENDPOINT", "V1_R2_BUCKET", "V1_R2_ACCESS_KEY_ID", "V1_R2_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("SN120_R2_PREFIX", "prod")
    assert smoke.main(["copy", "--state-dir", "unused"]) == 2 and "must start with 'smoke'" in capsys.readouterr().err and dst.list("") == []
    monkeypatch.setenv("SN120_R2_PREFIX", "smoke-1")
    assert smoke.main(["copy", "--state-dir", "unused"]) == 0, "without --count: as many files as the pinned batch_cap"
    out = capsys.readouterr()
    assert [line.split()[1] for line in out.out.splitlines()] == order[:2] and "2 of 2 files copied" in out.err
    assert smoke.main(["copy", "--state-dir", "unused", "--count", "5"]) == 1, "fewer usable files than asked"
    assert "too few files have a short group within 60 tokens" in capsys.readouterr().err
