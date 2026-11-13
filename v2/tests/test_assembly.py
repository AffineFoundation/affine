"""The builder of the real parts, on CPU: R2 is replaced by a local store, and no GPU process starts."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("math_verify")
pytest.importorskip("sn120_v2.gpu.trainer")

from sn120_v2 import assembly, layout, smoke  # noqa: E402
from sn120_v2.admission import StoreAdmission  # noqa: E402
from sn120_v2.canonical import manifest_digest, sha256_hex  # noqa: E402
from sn120_v2.cli import main  # noqa: E402
from sn120_v2.controller import StackContext  # noqa: E402
from sn120_v2.errors import ConfigError, InfraError, VerifyError  # noqa: E402
from sn120_v2.gpu.evaluator import VllmEvaluator  # noqa: E402
from sn120_v2.gpu.folder import LoraFolder  # noqa: E402
from sn120_v2.gpu.trainer import LoraTrainer  # noqa: E402
from sn120_v2.irt import IRTSampler  # noqa: E402
from sn120_v2.journal import JOURNAL_FILE, read_journal  # noqa: E402
from sn120_v2.params import GenesisSpec, Params, Recipe  # noqa: E402
from sn120_v2.state import load  # noqa: E402
from sn120_v2.store import LocalStore  # noqa: E402
from sn120_v2.types import ChainSnapshot, Registration  # noqa: E402
from sn120_v2.worker import LocalRunner  # noqa: E402

IMAGE = "sha256:" + "ab" * 32
BASE = {"config.json": b"{}", "tokenizer.json": b"tokens", "model-00001.safetensors": b"weights"}
MASTER = {"master-00001.safetensors": b"fp32"}


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return messages[0]["content"].split()


@pytest.fixture
def box(tmp_path, monkeypatch):
    """The environment of a box, with a local directory in the place of the R2 bucket."""
    store = LocalStore(tmp_path / "bucket")
    monkeypatch.setattr(assembly.R2Store, "from_env", classmethod(lambda cls: store))
    monkeypatch.setattr(assembly, "GradePool", lambda: "grade-pool")  # the real one starts 8 worker processes
    loaded = []
    monkeypatch.setattr(assembly.AutoTokenizer, "from_pretrained", lambda path, **options: loaded.append(path) or FakeTokenizer())
    for name, value in (("SN120_PUBLIC_URL", "none"), ("SN120_V2_CHAIN_NETWORK", "none"), ("SN120_WORK_DIR", str(tmp_path / "work")), ("SN120_IMAGE_DIGEST", IMAGE)):
        monkeypatch.setenv(name, value)
    store.loaded = loaded
    return store


def start(tmp_path, store) -> list[str]:
    """Stage made-up genesis files, bootstrap through the command line, and return the common arguments."""
    base_files = {name: sha256_hex(data) for name, data in BASE.items()}
    master_files = {name: sha256_hex(data) for name, data in MASTER.items()}
    genesis = GenesisSpec("org/model", "rev", manifest_digest(base_files), sha256_hex(b"master digest"), "5Validator")
    for name, data in BASE.items():
        store.put(layout.public_base_file(genesis.base_hash, name), data)
    for name, data in MASTER.items():
        store.put(layout.public_master_file(genesis.base_fp32_master_digest, name), data)
    chain = ChainSnapshot(120, 50, 60, (Registration(0, "owner", 1), Registration(1, "hk1", 101, 1), Registration(2, "hk2", 102, 1)))
    params = Params(n_tasks=200, recipe=Recipe(image_digests=(IMAGE,), global_norm_cap=3.5))
    for name, data in (("genesis.json", genesis.to_json()), ("chain.json", chain.to_json()), ("params.json", params.to_json())):
        (tmp_path / name).write_bytes(data)
    (tmp_path / "genesis_files.json").write_text(json.dumps({"base_files": base_files, "master_files": master_files}))
    common = ["--state-dir", str(tmp_path / "state"), "--stack", "sn120_v2.assembly:stack"]
    files = [f"--{name}={tmp_path / file}" for name, file in (("genesis", "genesis.json"), ("genesis-files", "genesis_files.json"), ("chain", "chain.json"))]
    assert main(["bootstrap", *common, "--params", str(tmp_path / "params.json"), *files]) == 0
    return common


def test_bootstrap_with_the_real_parts_pins_params_mirrors_the_journal_and_publishes_before_any_round(tmp_path, box, capsys):
    common = start(tmp_path, box)
    state = load(tmp_path / "state" / JOURNAL_FILE)
    assert state.params.recipe.global_norm_cap == 3.5 and state.params.n_tasks == 200, "set at bootstrap, with no code change"
    assert set(state.queue) == {(1, 101), (2, 102)}
    assert len(box.list(layout.JOURNAL)) == len(read_journal(tmp_path / "state" / JOURNAL_FILE))
    llms, mining = box.get(layout.LLMS_TXT).decode(), json.loads(box.get(layout.MINING_JSON))
    assert "global_norm_cap" in llms and json.dumps(mining).count("hk1") >= 1, "llms.txt and mining.json are out before the first round"
    capsys.readouterr()
    box.put(layout.miner_object("hk1"), b"not json")
    assert main(["status", *common]) == 0
    counts = json.loads(capsys.readouterr().out)["queued_files"]
    assert counts.pop("no file") == 1 and sum(counts.values()) == 1 and "valid" not in counts
    before = len(box.list(layout.JOURNAL))
    assert main(["publish", *common]) == 0 and len(box.list(layout.JOURNAL)) == before


def test_a_second_machine_reads_the_same_state_from_the_store_alone(tmp_path, box):
    start(tmp_path, box)
    context = StackContext(tmp_path / "another-machine", Params(), "status")
    stack = assembly.stack(context)
    assert context.params.recipe.global_norm_cap == 3.5 and stack.mirror and stack.runner is None and stack.folder is None
    assert isinstance(stack.admission, StoreAdmission) and stack.admission.params == context.params
    assert isinstance(stack.sampler, IRTSampler) and stack.chain is None and len(stack.entropy()) == 32
    assert load(tmp_path / "another-machine" / JOURNAL_FILE).head == load(tmp_path / "state" / JOURNAL_FILE).head


def test_the_run_command_gets_the_gpu_parts_and_needs_the_box_settings(tmp_path, box, monkeypatch):
    start(tmp_path, box)
    stack = assembly.stack(StackContext(tmp_path / "state", Params(), "run"))
    runner = stack.runner
    assert isinstance(runner, LocalRunner) and isinstance(runner.trainer, LoraTrainer) and isinstance(runner.evaluator, VllmEvaluator)
    assert isinstance(stack.folder, LoraFolder) and runner.image_digest == IMAGE and runner.evaluator.grader == "grade-pool"
    monkeypatch.delenv("SN120_IMAGE_DIGEST")
    with pytest.raises(ConfigError):
        assembly.stack(StackContext(tmp_path / "state", Params(), "run"))


def test_the_checker_counts_tokens_with_the_verified_tokenizer_files_of_the_head(tmp_path, box):
    tokens = assembly.HeadTokens(box, tmp_path / "tokenizer")
    with pytest.raises(ConfigError):
        tokens.output_tokens("a b c")  # no head in the store yet
    start(tmp_path, box)
    assert tokens.output_tokens("a b c") == 3 and tokens.prompt_tokens("one two") == 2
    directory = tmp_path / "tokenizer" / load(tmp_path / "state" / JOURNAL_FILE).head.base_hash
    assert box.loaded == [str(directory)] and sorted(path.name for path in directory.iterdir()) == ["config.json", "tokenizer.json"], "no weight file is fetched"
    key = layout.public_base_file(directory.name, "tokenizer.json")
    box.put(key, b"changed")
    (directory / "tokenizer.json").unlink()
    with pytest.raises(VerifyError):
        assembly.HeadTokens(box, tmp_path / "tokenizer").output_tokens("a")
    assert not (directory / "tokenizer.json").exists()


def test_the_round_seed_uses_the_block_hash_of_the_last_chain_read():
    class Reader:
        def snapshot(self):
            return ChainSnapshot(120, 777, 1, (Registration(0, "owner", 1),))

        def block_hash(self, block):
            return f"hash-of-{block}"

    feed = assembly.ChainFeed(120, "finney")
    feed._reader = Reader()
    with pytest.raises(InfraError):
        feed.entropy()
    assert feed.snapshot().block == 777 and feed.entropy() == "hash-of-777"


def test_the_smoke_copy_with_the_real_parts_writes_a_small_valid_file_and_needs_no_head_pointer(tmp_path, box, monkeypatch, capsys):
    common = start(tmp_path, box)
    think = "step " * 40
    rows = lambda words: [{"output": f"{think}{'more ' * words}\\boxed{{2}}", "reward": 1}, {"output": f"{think}\\boxed{{3}}", "reward": 0}]
    groups = [{"prompt": f"what is 1 + {n}", "rollouts": rows(words)} for n, words in enumerate((500, 0, 3))]
    v1 = LocalStore(tmp_path / "v1")
    v1.put(layout.miner_object("hk1"), json.dumps({"version": "affine-batch-v3", "hotkey": "hk1", "train": {"steps": 2}, "groups": groups}).encode())
    monkeypatch.setattr(smoke, "R2Store", lambda **settings: v1)
    for name in ("V1_R2_ENDPOINT", "V1_R2_BUCKET", "V1_R2_ACCESS_KEY_ID", "V1_R2_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("SN120_R2_PREFIX", "smoke-test")
    (tmp_path / "small.json").write_text(json.dumps({"batch_cap": 1, "max_output_tokens": 100, "recipe": {"image_digests": [IMAGE]}}))
    assert main(["set-params", *common, "--params", str(tmp_path / "small.json")]) == 0
    box.delete(layout.CURRENT)  # the checker takes the head from the journal, so the place of the pointer in the store does not matter
    capsys.readouterr()
    assert smoke.main(["copy", "--state-dir", str(tmp_path / "state")]) == 0
    out = capsys.readouterr()
    assert out.out.startswith("copied hk1 2 of 3 groups, 4 rollouts, longest 44 tokens") and "1 of 1 files copied" in out.err
    assert json.loads(box.get(layout.miner_object("hk1")))["groups"] == [groups[1], groups[2]], "whole groups, unchanged; the long group is left out"
    assert main(["status", *common]) == 0
    assert json.loads(capsys.readouterr().out)["queued_files"] == {"valid": 1, "no file": 1}
