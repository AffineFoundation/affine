"""The builder of the real parts, on CPU: R2 is replaced by a local store, and no GPU process starts."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("math_verify")
pytest.importorskip("sn120_v2.gpu.trainer")

from sn120_v2 import assembly, controller, headchain, headfold, layout, smoke  # noqa: E402
from sn120_v2.gpu.batch import MANDATE  # noqa: E402
from sn120_v2.admission import STREAMS_ENV, WORKERS_ENV, StoreAdmission  # noqa: E402
from sn120_v2.afterking import ORDER_ENV  # noqa: E402
from sn120_v2.canonical import manifest_digest, sha256_hex  # noqa: E402
from sn120_v2.cli import main  # noqa: E402
from sn120_v2.controller import HeadPublish, StackContext  # noqa: E402
from sn120_v2.errors import ConfigError, InfraError, VerifyError  # noqa: E402
from sn120_v2.gpu.evaluator import VllmEvaluator  # noqa: E402
from sn120_v2.gpu import plan  # noqa: E402
from sn120_v2.gpu.folder import LoraFolder  # noqa: E402
from sn120_v2.gpu.plan import code_digest  # noqa: E402
from sn120_v2.gpu.trainer import LoraTrainer  # noqa: E402
from sn120_v2.headfold import ChainFolder, FoldSettings, HeadKeeper  # noqa: E402
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

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        return " ".join(message["content"] for message in messages)


@pytest.fixture
def box(tmp_path, monkeypatch):
    """The environment of a box, with a local directory in the place of the R2 bucket."""
    store = LocalStore(tmp_path / "bucket")
    monkeypatch.setattr(assembly.R2Store, "from_env", classmethod(lambda cls: store))
    monkeypatch.setattr(assembly, "GradePool", lambda: "grade-pool")  # the real one starts 8 worker processes at first grade
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


def test_stack_start_enforces_a_nonempty_env_pin_and_names_both_values(tmp_path, box, monkeypatch):
    start(tmp_path, box)
    from sn120_v2.envs.math_grader import env_pin

    common = ["--state-dir", str(tmp_path / "state"), "--stack", "sn120_v2.assembly:stack"]
    wrong_file = tmp_path / "wrong.json"
    wrong_file.write_bytes(Params(env_pin="0" * 64).to_json())
    assert main(["set-params", *common, "--params", str(wrong_file)]) == 2
    # The context's load has the same contract as journal loading: it makes
    # the pinned params active. No store or service is involved in this check.
    context = StackContext(tmp_path / "wrong-pin", Params(), "status")
    def loaded(store=None):
        context.params = Params(env_pin="0" * 64)
        return load(tmp_path / "state" / JOURNAL_FILE)
    monkeypatch.setattr(context, "load", loaded)
    with pytest.raises(ConfigError) as failure:
        assembly.stack(context)
    assert "0" * 64 in str(failure.value) and env_pin() in str(failure.value)
    pinned = tmp_path / "pinned.json"
    pinned.write_bytes(Params(env_pin=env_pin()).to_json())
    assert main(["set-params", *common, "--params", str(pinned)]) == 0
    assert assembly.stack(StackContext(tmp_path / "correct-pin", Params(), "status")) is not None


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
    assert isinstance(stack.folder, LoraFolder) and runner.image_digest == IMAGE
    assert runner.evaluator.grader._pool is None and runner.evaluator.grader._start() == "grade-pool"
    assert stack.pins is not None and assembly.stack(StackContext(tmp_path / "state", Params(), "end-round")).pins is None
    monkeypatch.delenv("SN120_IMAGE_DIGEST")
    with pytest.raises(ConfigError):
        assembly.stack(StackContext(tmp_path / "state", Params(), "run"))


def test_with_the_chain_setting_the_box_commands_get_the_parts_that_keep_the_folded_head_on_the_box(tmp_path, box, monkeypatch, capsys):
    common = start(tmp_path, box)

    def stack(command: str):
        return assembly.stack(StackContext(tmp_path / "state", Params(), command))

    full = stack("run").head
    assert full == HeadPublish() and full.promote is controller.promote_fold and full.chain is None, "the default: each head is uploaded in full"
    assert main(["checkpoint", *common]) == 2 and "this stack makes no checkpoints" in capsys.readouterr().err
    monkeypatch.setenv(headchain.MODE_ENV, headchain.CHAIN)
    run, publish = stack("run"), stack("publish")
    assert isinstance(run.folder, ChainFolder) and isinstance(run.runner, LocalRunner) and run.head.promote is headchain.promote_fold
    assert isinstance(run.head.chain, headchain.ChainWriter) and None not in (run.head.prepare, run.head.after_round, run.head.checkpoint)
    keeper = run.head.checkpoint.__self__
    assert isinstance(keeper, HeadKeeper) and keeper.every == headchain.DEFAULT_EVERY and keeper.settings == FoldSettings.from_env({})
    assert run.folder.settings.implementation == headfold.REFERENCE and run.folder.settings.twice, "the defaults of the box"
    assert stack("checkpoint").head.checkpoint is not None and stack("checkpoint").folder is None
    assert publish.folder is None and isinstance(publish.head.chain, headchain.ChainWriter) and publish.head.promote is headchain.promote_fold
    assert (publish.head.prepare, publish.head.after_round, publish.head.checkpoint) == (None, None, None), "no command but run and checkpoint works on the local head"
    monkeypatch.setenv(headchain.EVERY_ENV, "1")
    monkeypatch.setenv(headfold.IMPLEMENTATION_ENV, headfold.BLAS)
    monkeypatch.setenv(headfold.DEVICES_ENV, "cuda:0,cuda:1")
    run = stack("run")
    assert run.head.checkpoint.__self__.every == 1 and run.folder.settings.devices == ("cuda:0", "cuda:1") and run.folder.settings.implementation == headfold.BLAS
    monkeypatch.setenv(headchain.MODE_ENV, "sometimes")
    with pytest.raises(ConfigError):
        stack("publish")


def test_the_count_of_check_streams_is_a_setting_and_a_setting_that_the_shell_does_not_hold_is_shown(tmp_path, box, monkeypatch):
    start(tmp_path, box)

    def stack():
        return assembly.stack(StackContext(tmp_path / "state", Params(), "run"))

    default = stack()
    assert default.admission.streams == default.admission.workers == 1, "one file after the other, as before"
    assert [(name in line, "NOT SET" in line) for name, line in zip((headchain.MODE_ENV, STREAMS_ENV, WORKERS_ENV, ORDER_ENV), default.settings, strict=True)] == [(True, True)] * 4
    assert "'full'" in default.settings[0] and "one file after the other" in default.settings[1] and "original serial path" in default.settings[2] and "'serial'" in default.settings[3]
    monkeypatch.setenv(STREAMS_ENV, "8")
    monkeypatch.setenv(headchain.MODE_ENV, headchain.CHAIN)
    chosen = stack()
    assert chosen.admission.streams == 8 and code_digest() == plan.code_digest(), "a setting of the shell: no file under gpu/ reads it"
    assert chosen.settings == (
        f"the head goes public in the mode 'chain' ({headchain.MODE_ENV}=chain)",
        f"the check of the waiting files reads 8 files at one time ({STREAMS_ENV}=8)",
        f"the check of the waiting files uses the original serial path ({WORKERS_ENV} is NOT SET: this is the default)",
        f"after a king, each step waits for the one before it (the order 'serial') ({ORDER_ENV} is NOT SET: this is the default)",
    )
    monkeypatch.setenv(STREAMS_ENV, "0")
    with pytest.raises(ConfigError, match=STREAMS_ENV):
        stack()


@pytest.mark.parametrize("value", ["\u00b2", "\u2460", "\uff18", "08", "0", "33", "abc"])
def test_a_bad_count_of_check_streams_is_one_refused_line_with_exit_code_2_for_run_and_for_end_round(tmp_path, box, monkeypatch, capsys, value):
    """Exit code 2 stops the run loop at once. A traceback (exit code 1) is a crash for it: it then runs ``end-round`` and starts ``run`` again."""
    common = start(tmp_path, box)
    records = len(read_journal(tmp_path / "state" / JOURNAL_FILE))
    monkeypatch.setenv(STREAMS_ENV, value)
    capsys.readouterr()
    for command in (["run", *common, "--max-idle", "1", "--idle-sleep", "0"], ["end-round", *common]):
        assert main(command) == 2
        shown = capsys.readouterr()
        assert shown.err.splitlines() == [f"refused: ConfigError: {STREAMS_ENV} must be a whole number from 1 to 32, written in the digits 0 to 9 with no zero in front (8, not 08)"]
        assert shown.out == "" and "Traceback" not in shown.err
    assert len(read_journal(tmp_path / "state" / JOURNAL_FILE)) == records, "refused before the journal is taken over: no record, no write"


def test_the_order_after_a_king_is_a_setting_of_run_in_chain_mode_and_serial_is_the_default(tmp_path, box, monkeypatch):
    start(tmp_path, box)

    def stack(command="run"):
        return assembly.stack(StackContext(tmp_path / "state", Params(), command))

    default = stack()
    assert not default.overlap and default.head.proven is None, "not set: the order of always, in full mode"
    monkeypatch.setenv(ORDER_ENV, "serial")
    assert not stack().overlap and stack().settings[3] == f"after a king, each step waits for the one before it (the order 'serial') ({ORDER_ENV}=serial)"
    monkeypatch.setenv(ORDER_ENV, "overlap")
    with pytest.raises(ConfigError, match=f"{ORDER_ENV}=overlap needs {headchain.MODE_ENV}=chain"):
        stack()  # in full mode the upload of the head is the wait: the order would move nothing
    monkeypatch.setenv(headchain.MODE_ENV, headchain.CHAIN)
    run = stack()
    assert run.overlap and run.head.proven.__self__ is run.folder and run.entropy_now is None, "this box reads no chain (the smoke test), so no block hash"
    assert run.settings[3] == f"after a king, only the fold is before the next open (the order 'overlap') ({ORDER_ENV}=overlap)"
    assert not stack("publish").overlap and not stack("end-round").overlap and not stack("checkpoint").overlap, "only run takes steps out of the line"
    assert code_digest() == plan.code_digest(), "a setting of the shell: no file under gpu/ reads it"
    monkeypatch.setenv(ORDER_ENV, "fast")
    with pytest.raises(ConfigError, match=ORDER_ENV):
        stack("publish")


def test_run_on_a_box_that_differs_from_the_pinned_digests_is_refused_before_any_round(tmp_path, box, monkeypatch, capsys):
    common = start(tmp_path, box)
    run = ["run", *common, "--max-idle", "1", "--idle-sleep", "0"]
    assert main(run) == 0, "the box has the pinned image: run waits for a file"
    records = len(box.list(layout.JOURNAL))
    monkeypatch.setenv("SN120_IMAGE_DIGEST", "sha256:" + "cd" * 32)  # as after a package change with no set-params
    capsys.readouterr()
    assert main(run) == 2, "exit code 2: the restart loop does not start run again"
    refused = "refused: ConfigError: this box differs from the pinned digests, so no round was opened. The difference: "
    values = f"image digest: pinned {IMAGE}, this box has sha256:{'cd' * 32} (the value of SN120_IMAGE_DIGEST). Nothing is counted for a seat. Next: the digest block, then set-params with the whole parameter file in which only the two digests are new (set-params replaces every value: a key that the file does not hold goes back to its default), then start again"
    assert refused + values in capsys.readouterr().err, "the line shows both values: no command shows the pinned digests"
    assert len(box.list(layout.JOURNAL)) == records + 1, "only the record of the start; nothing else was written"
    assert main(["end-round", *common]) == 2 and "no round is open" in capsys.readouterr().err, "the commands that need no GPU part are not refused for the pins"


def test_the_pin_check_of_the_box_is_the_one_of_the_trainer_and_names_each_digest_that_differs_with_both_values(monkeypatch):
    code = code_digest()
    params = Params(recipe=Recipe(image_digests=(IMAGE, "sha256:second"), code_digest=code))
    assembly.box_pins(IMAGE, params)
    assembly.box_pins(IMAGE, Params(recipe=Recipe(image_digests=(IMAGE,))))  # no code digest is pinned: as check_pins, only the image counts
    with pytest.raises(InfraError, match=f"^image digest: pinned {IMAGE} or sha256:second, this box has sha256:other \\(the value of SN120_IMAGE_DIGEST\\)$"):
        assembly.box_pins("sha256:other", params)
    monkeypatch.setattr(assembly, "code_digest", lambda: "the-code-after-a-fix")
    monkeypatch.setattr(plan, "code_digest", lambda: "the-code-after-a-fix")
    with pytest.raises(InfraError, match=f"^code digest: pinned {code}, this box has the-code-after-a-fix$"):
        assembly.box_pins(IMAGE, params)
    with pytest.raises(InfraError, match="^image digest: .*; code digest: pinned"):
        assembly.box_pins("sha256:other", params)


def test_the_checker_counts_tokens_with_the_verified_tokenizer_files_of_the_head(tmp_path, box):
    tokens = assembly.HeadTokens(box, tmp_path / "tokenizer")
    with pytest.raises(ConfigError):
        tokens.output_tokens("a b c")  # no head in the store yet
    start(tmp_path, box)
    assert tokens.output_tokens("a b c") == 3 and tokens.prompt_tokens("one two") == len(MANDATE.split()) + 2
    directory = tmp_path / "tokenizer" / load(tmp_path / "state" / JOURNAL_FILE).head.base_hash
    assert box.loaded == [str(directory)] and sorted(path.name for path in directory.iterdir()) == ["config.json", "tokenizer.json"], "no weight file is fetched"
    key = layout.public_base_file(directory.name, "tokenizer.json")
    box.put(key, b"changed")
    (directory / "tokenizer.json").unlink()
    with pytest.raises(VerifyError):
        assembly.HeadTokens(box, tmp_path / "tokenizer").output_tokens("a")
    assert not (directory / "tokenizer.json").exists()


def test_a_store_bootstrapped_with_the_old_pointer_key_needs_no_new_bootstrap(tmp_path, box, capsys):
    """The head pointer moved from ``public/current.json``. ``status`` and ``publish`` write no pointer: they take the head from the journal."""
    common = start(tmp_path, box)
    box.put(layout.miner_object("hk1"), b'{"rows": []}')
    box.put("public/current.json", box.get(layout.CURRENT))
    box.delete(layout.CURRENT)
    assert main(["status", *common]) == 0 and "queued_files" in capsys.readouterr().out
    assert main(["publish", *common]) == 0 and not box.exists(layout.CURRENT)
    stack = assembly.stack(StackContext(tmp_path / "another-machine", Params(), "status"))
    assert stack.admission.tokens.output_tokens("a b c") == 3 and len(box.loaded) == 1, "the tokenizer of the journal's head"


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
    rows = lambda words, n: [{"output": f"{think}{'more ' * words}\\boxed{{{2 * n}}}", "reward": 1}, {"output": f"{think}\\boxed{{{2 * n + 1}}}", "reward": 0}]
    groups = [{"prompt": f"what is 1 + {n}", "rollouts": rows(words, n)} for n, words in enumerate([500, *([0] * 31), *([3] * 32)], start=1)]
    # Supply a small, deterministic local grader: this test exercises the real assembly/copy path without a math-pool process.
    monkeypatch.setattr(assembly, "pool_grader", lambda pool, grades, chunk=32:
                        lambda prompt, output, capped: float(output.rsplit("boxed{", 1)[-1].split("}", 1)[0] == str(2 * int(prompt.rsplit(" ", 1)[-1]))))
    v1 = LocalStore(tmp_path / "v1")
    v1.put(layout.miner_object("hk1"), json.dumps({"version": "affine-batch-v3", "hotkey": "hk1", "train": {"steps": 2}, "groups": groups}).encode())
    monkeypatch.setattr(smoke, "R2Store", lambda **settings: v1)
    for name in ("V1_R2_ENDPOINT", "V1_R2_BUCKET", "V1_R2_ACCESS_KEY_ID", "V1_R2_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("SN120_R2_PREFIX", "smoke-test")
    (tmp_path / "small.json").write_text(json.dumps({"batch_cap": 1, "max_output_tokens": 100, "min_pool_prompt_frac": 1.0, "recipe": {"image_digests": [IMAGE]}}))
    assert main(["set-params", *common, "--params", str(tmp_path / "small.json")]) == 0
    box.delete(layout.CURRENT)  # the checker takes the head from the journal, so the place of the pointer in the store does not matter
    capsys.readouterr()
    assert smoke.main(["copy", "--state-dir", str(tmp_path / "state")]) == 0
    out = capsys.readouterr()
    assert out.out.startswith("copied hk1 32 of 64 groups, 64 rollouts, longest 44 tokens") and "1 of 1 files copied" in out.err
    assert json.loads(box.get(layout.miner_object("hk1")))["groups"] == groups[1:33], "whole groups, unchanged; stop once the 64-active floor is met"
    assert main(["status", *common]) == 0
    assert json.loads(capsys.readouterr().out)["queued_files"] == {"valid": 1, "no file": 1}
