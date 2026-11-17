"""The trainer relax on a tiny model of the head's architecture (CPU): minibatches, the update cap as a miner value, the larger file caps.

The first test holds the new training job against the job of commit
``a5c2b6a6``: ``tests/data/train_job_a5c2b6a6.py.txt`` is that file, byte
for byte (``git show a5c2b6a6:v2/sn120_v2/gpu/train_job.py``). With one
minibatch and only rows that train, the two write the same bytes. Zero-advantage
filler rows no longer shrink the loss: see the filler-row test below.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import importlib.machinery
import importlib.util
import json
import math
import random
import string
import sys
import threading
from pathlib import Path

import pytest

# As in test_gpu_torch.py: the tiny model runs on the CPU, so transformers must take its plain torch path.
sys.modules["fla"] = None  # type: ignore[assignment]

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from safetensors.torch import load_file  # noqa: E402 - after the skip guards

from sn120_v2.admission import Invalid, check_batch  # noqa: E402
from sn120_v2.canonical import digest, sha256_hex  # noqa: E402
from sn120_v2.errors import BatchRejected, ConfigError, InfraError  # noqa: E402
from sn120_v2.gpu import train_job  # noqa: E402
from sn120_v2.gpu.resident import hash_tree  # noqa: E402
from sn120_v2.gpu.trainer import LoraTrainer  # noqa: E402
from sn120_v2.params import DEFAULT_MINER_BOUNDS, MinerBound, Params, Recipe  # noqa: E402

HOTKEY = "5MinerA"
THINK = "step " * 40  # 40 reasoning tokens: above the least of 32
TEMPLATE = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

DATA = Path(__file__).parent / "data"
OLD_TRAINER, OLD_TRAINER_SHA256 = DATA / "train_job_a5c2b6a6.py.txt", "e300c9dadc9ac47ad659fee20e71998ac6f06fd0ae07844e18ce92e48d06237e"
OLD_PARAMS, OLD_PARAMS_HASH = DATA / "params_default_a5c2b6a6.json", "65ce6ef3bc867172eb87673bdb39461d03a8c85aac743c61e8b3d233b9f26d54"
NOT_A_CLOCK = ("status", "loss", "step_norm", "step_norm_before_cap", "tokens", "gpu_class", "truncated_rows", "dropped_rows", "n_gpus", "rows", "forward_tokens")

# 12 rows, 10 of them train. Two have advantage 0, the lengths differ, and one prompt has three rollouts.
OUTPUTS = ["abc", "ba", "ccdd", "d", "abcabcab", "zz", "qrs", "tuvwx", "mnopqrstu", "ef", "gh", "ijklmn"]
ADVANTAGES = [0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.0, 0.75, -0.75, 0.0, 0.4, -0.4]
ROWS = [{"prompt": f"p {'abcd'[i // 3]}", "output": output, "reward": float(a > 0), "advantage": a} for i, (output, a) in enumerate(zip(OUTPUTS, ADVANTAGES))]
N_TRAIN = sum(a != 0 for a in ADVANTAGES)
TODAY_BOUNDS = tuple(bound for bound in DEFAULT_MINER_BOUNDS if bound.key not in ("minibatches", "global_norm_cap"))  # the 7 bounds of a5c2b6a6


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def head_dir(tmp_path_factory) -> Path:
    """The tiny head of test_gpu_torch.py: a bf16 checkpoint with the class and the layer types of the launch model, and a character tokenizer."""
    path = tmp_path_factory.mktemp("head")
    text = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2, head_dim=16)
    text.update(linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4, max_position_embeddings=256, eos_token_id=2, dtype="bfloat16")
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32)
    config = transformers.Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    torch.manual_seed(0)
    transformers.Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(path)
    words = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "Ġ", "Ċ", *string.ascii_lowercase]
    special = [dict(id=i, content=words[i], single_word=False, lstrip=False, rstrip=False, normalized=False, special=True) for i in range(3)]
    model = dict(type="BPE", dropout=None, unk_token=None, continuing_subword_prefix="", end_of_word_suffix="", fuse_unk=False, byte_fallback=False, vocab={w: i for i, w in enumerate(words)}, merges=[])
    byte_level = dict(type="ByteLevel", add_prefix_space=False, trim_offsets=True, use_regex=True)
    tokenizer = dict(version="1.0", truncation=None, padding=None, added_tokens=special, normalizer=None, pre_tokenizer=byte_level, post_processor=None, decoder=byte_level, model=model)
    (path / "tokenizer.json").write_text(json.dumps(tokenizer))
    (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|im_end|>"}))
    (path / "chat_template.jinja").write_text(TEMPLATE)
    return path


def run(head_dir: Path, out: Path, devices=("cpu",), dtype: str = "bfloat16", seed: int = 7, rows=None, **recipe) -> tuple[dict, dict, dict]:
    """Run the training job. Returns (result, LoRA tensors, the train report file)."""
    recipe = Recipe(**({"steps": 2, "lora_lr": 1e-4} | recipe))
    made = {"base_dir": str(head_dir), "base_hash": "b" * 64, "dtype": dtype, "recipe": recipe.to_dict(), "seed": seed, "rows": ROWS if rows is None else rows}
    result = train_job.train(made, out, list(devices))
    return result, load_file(str(out / "adapter_model.safetensors")), json.loads((out / "train_report.json").read_text())


def update_norm(lora: dict, scale: float) -> float:
    pairs = [(lora[key], lora[key.replace("lora_A", "lora_B")]) for key in lora if "lora_A" in key]
    return math.sqrt(sum(float(((scale * b @ a) ** 2).sum()) for a, b in pairs))


def file(train=None) -> bytes:
    """A valid file of 10 prompts with 2 rollouts each. All are right but the last rollout of the last prompt."""
    right = lambda n, k: {"output": f"{THINK}{k} \\boxed{{{2 * n}}}", "reward": 1}  # noqa: E731
    groups = [{"prompt": f"double {n}", "rollouts": [right(n, 0), right(n, 1) if n < 9 else {"output": f"{THINK}1 \\boxed{{-1}}", "reward": 0}]} for n in range(10)]
    return json.dumps({"version": "affine-batch-v3", "hotkey": HOTKEY, "groups": groups} | ({} if train is None else {"train": train})).encode()


def rule(raw: bytes, params) -> str:
    """The code of the rule that fails, or "" for a valid file."""
    try:
        check_batch(raw, HOTKEY, params)
    except Invalid as exc:
        return exc.rule
    return ""


@pytest.fixture(scope="module")
def old_job():
    """The training job of a5c2b6a6 as a module of the package, so its own imports find the package of today."""
    assert sha256_hex(OLD_TRAINER.read_bytes()) == OLD_TRAINER_SHA256, "this file must stay the train_job.py of a5c2b6a6"
    loader = importlib.machinery.SourceFileLoader("sn120_v2.gpu.train_job_a5c2b6a6", str(OLD_TRAINER))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


def job(head: Path, dtype: str = "float32", seed: int = 7, rows=ROWS, **recipe) -> dict:
    if recipe.get("global_norm_cap", 6.0) is None:  # a recipe with no cap is a recipe of before: it has no miner bound for the cap
        recipe = {"miner_bounds": TODAY_BOUNDS} | recipe
    recipe = Recipe(**({"steps": 2, "lora_lr": 1e-4} | recipe))
    return {"base_dir": str(head), "base_hash": "b" * 64, "dtype": dtype, "recipe": recipe.to_dict(), "seed": seed, "rows": rows}


def tree(head: Path, out: Path, n: int = 1, **settings) -> dict:
    """The sha256 of each file of the LoRA (weights, config, train report) of one run."""
    train_job.train(job(head, **settings), out, ["cpu"] * n)
    return hash_tree(out)


@pytest.fixture
def watch(monkeypatch):
    """Record what one run of the job does: each Adam and its updates, each minibatch, each call of ``train_rows``,
    and for each row the count of updates that the model had when the row got its old policy, its mask and its count."""
    seen = {"adams": [], "first_pass": [], "passes": [], "train_rows": [], "rows": []}

    class Adam(torch.optim.Adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.updates = 0
            seen["adams"].append(self)

        def step(self, *args, **kwargs):
            out = super().step(*args, **kwargs)
            self.updates += 1
            return out

    encode, score_old, cut, train_rows = train_job.encode, train_job.score_old, train_job.minibatches_of, train_job.Replica.train_rows

    def kept_rows(tokenizer, rows, recipe):
        out = encode(tokenizer, rows, recipe)
        seen["rows"] = out[0]
        return out

    def index(row: dict) -> int:
        return next(i for i, other in enumerate(seen["rows"]) if other is row)

    def first_pass(rows) -> None:
        seen["first_pass"] += [(index(row), seen["adams"][-1].updates) for row in rows if "old" not in row]

    def watched_score_old(replica, rows):
        first_pass(rows)
        return score_old(replica, rows)

    def watched_cut(rows, minibatches, order):
        groups = cut(rows, minibatches, order)
        seen["passes"].append([[index(row) for row in group] for group in groups])
        return groups

    def watched_train_rows(self, rows, n_rows):
        first_pass(rows)  # a row with no old policy yet gets it in this call
        seen["train_rows"].append((len(rows), n_rows))
        return train_rows(self, rows, n_rows)

    monkeypatch.setattr(torch.optim, "Adam", Adam)
    monkeypatch.setattr(train_job, "encode", kept_rows)
    monkeypatch.setattr(train_job, "score_old", watched_score_old)
    monkeypatch.setattr(train_job, "minibatches_of", watched_cut)
    monkeypatch.setattr(train_job.Replica, "train_rows", watched_train_rows)
    return seen


# ---- A, one minibatch: the job of a5c2b6a6, byte for byte, without zero-advantage rows ----

SAME_AS_BEFORE = {
    "today's default recipe, one step": dict(steps=1, lora_lr=1e-5, global_norm_cap=None),
    "the new default recipe": dict(steps=1, lora_lr=1e-5),
    "two steps": dict(),
    "four steps, rank 64, alpha 128, lr 8e-5": dict(steps=4, rank=64, alpha=128.0, lora_lr=8e-5),
    "the cap cuts the update": dict(steps=4, rank=64, alpha=128.0, global_norm_cap=1e-4),
    "two copies of the model": dict(n=2),
    "three copies, MLP layers only, three steps": dict(n=3, steps=3, target_modules=("gate_proj", "up_proj", "down_proj")),
    "rows cut at the training cap, the derail rule": dict(steps=3, train_max_output_tokens=3, derail_window=2, derail_threshold=-0.5, derail_tail=0),
    "groups of the recipe, another temperature and seed": dict(seed=11, sample_temperature=0.7, support_top_k=5, support_top_p=0.8),
    "bf16, as the box": dict(dtype="bfloat16"),
}


@pytest.mark.parametrize("case", SAME_AS_BEFORE)
def test_one_minibatch_gives_the_bytes_of_the_trainer_of_a5c2b6a6(case, old_job, head_dir, tmp_path, amx_off):
    settings = dict(SAME_AS_BEFORE[case])
    devices = ["cpu"] * settings.pop("n", 1)
    made = job(head_dir, rows=[row for row in ROWS if row["advantage"] != 0], **settings)
    assert "minibatches" not in made["recipe"] and Recipe.from_dict(made["recipe"]).minibatches == 1
    before = old_job.run(json.loads(json.dumps(made)), tmp_path / "before", devices)
    now = train_job.train(json.loads(json.dumps(made)), tmp_path / "now", devices)
    # Each byte of each tensor, of the LoRA config and of the train report: so the same lora_hash.
    assert hash_tree(tmp_path / "now") == hash_tree(tmp_path / "before") and sorted(hash_tree(tmp_path / "now")) == sorted(["adapter_config.json", "adapter_model.safetensors", "train_report.json"])
    # What the box takes from the job into the journal (TrainReport: loss, step_norm, norm_before_cap, tokens, the row counts) and into its log.
    assert {key: now[key] for key in NOT_A_CLOCK} == {key: before[key] for key in NOT_A_CLOCK}
    assert set(now) == set(before)
    report = json.loads((tmp_path / "now" / "train_report.json").read_text())
    assert "minibatches" not in report and "minibatches" not in report["recipe_values"] and len(report["losses"]) == made["recipe"]["steps"]


def test_the_job_of_one_minibatch_is_the_source_of_a5c2b6a6(monkeypatch):
    """No GPU needed: the old file is unchanged apart from dispatch and the active-row loss denominator."""
    before, now = OLD_TRAINER.read_text(), Path(train_job.__file__).read_text()
    old = {node.name: ast.get_source_segment(before, node) for node in ast.parse(before).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    new = {node.name: ast.get_source_segment(now, node) for node in ast.parse(now).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert {"run", "save", "Replica", "token_scores", "ppo_loss", "encode", "add_lora", "Replay", "main"} <= set(old)
    assert {name for name in old if old[name] != new.get(name)} == {"main", "run"}
    assert sorted(set(new) - set(old)) == ["bring", "minibatches_of", "run_minibatches", "score_old", "train"]
    diff = list(difflib.unified_diff(before.splitlines(), now.splitlines(), n=0, lineterm=""))
    gone = [line for line in diff if line.startswith("-") and not line.startswith("---")]
    assert sorted(gone) == sorted(['-        report = run(job, job_dir / "lora")', "-            results = list(pool.map(lambda pair: pair[0].train_rows(pair[1], len(rows)), zip(replicas, shares)))"]), "only main dispatch and the trained-row denominator changed"
    assert "+            results = list(pool.map(lambda pair: pair[0].train_rows(pair[1], len(active)), zip(replicas, shares)))" in diff
    # The one place that picks the job: one minibatch is ``run``, more are ``run_minibatches``.
    called = []
    monkeypatch.setattr(train_job, "run", lambda job, out, devices=None: called.append("run"))
    monkeypatch.setattr(train_job, "run_minibatches", lambda job, out, devices=None: called.append("run_minibatches"))
    for recipe in (Recipe(), Recipe(steps=4), Recipe(minibatches=2), Recipe(minibatches=8000)):
        train_job.train({"recipe": recipe.to_dict()}, Path("unused"))
    assert called == ["run", "run", "run_minibatches", "run_minibatches"]


def test_one_minibatch_scores_no_row_ahead_and_mixes_no_row(head_dir, tmp_path, watch):
    train_job.train(job(head_dir, steps=3), tmp_path / "lora", ["cpu"])
    assert watch["passes"] == [] and [adam.updates for adam in watch["adams"]] == [3]
    assert watch["train_rows"] == [(N_TRAIN, N_TRAIN)] * 3, "the trained rows in each update, and the loss divided by their count"
    assert sorted(watch["first_pass"]) == [(i, 0) for i, a in enumerate(ADVANTAGES) if a != 0]


def test_zero_advantage_filler_does_not_change_loss_or_gradient_scale(head_dir, tmp_path, monkeypatch):
    gradients, clip = [], torch.nn.utils.clip_grad_norm_

    def capture(weights, *args, **kwargs):
        gradients.append([weight.grad.detach().clone() for weight in weights if weight.grad is not None])
        return clip(weights, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
    _, plain_lora, plain = run(head_dir, tmp_path / "plain", dtype="float32", steps=2)
    plain_gradients = gradients.copy()
    gradients.clear()
    filler = [{"prompt": f"filler {i}", "output": "zero advantage", "reward": 1.0, "advantage": 0.0} for i in range(9)]
    _, padded_lora, padded = run(head_dir, tmp_path / "padded", dtype="float32", steps=2, rows=ROWS + filler)

    assert padded["rows"] == plain["rows"] + len(filler)
    for key in ("losses", "grad_norms", "trained_tokens", "step_norm", "step_norm_before_cap"):
        assert padded[key] == plain[key]
    assert gradients and len(gradients) == plain["recipe_values"]["steps"]
    for plain_step, padded_step in zip(plain_gradients, gradients):
        assert len(plain_step) == len(padded_step)
        assert all(torch.equal(a, b) for a, b in zip(plain_step, padded_step))
    assert all(torch.equal(plain_lora[key], padded_lora[key]) for key in plain_lora)


# ---- A, more minibatches ----


def test_steps_times_minibatches_updates_and_no_update_without_rows(head_dir, tmp_path, watch):
    result, _, report = run(head_dir, tmp_path / "a", dtype="float32", rows=ROWS, steps=2, minibatches=4)
    assert result["updates"] == len(report["losses"]) == len(report["grad_norms"]) == 8 and report["recipe_values"]["minibatches"] == 4
    assert [len(groups) for groups in watch["passes"]] == [4, 4] and watch["adams"][-1].updates == 8
    assert sorted(size for size, _ in watch["train_rows"]) == [2] * 4 + [3] * 4 and all(size == n_rows for size, n_rows in watch["train_rows"]), "the loss of a minibatch is divided by its own rows"
    # More minibatches than rows that train: each row is a minibatch of its own, one update for each row in each pass.
    result, _, report = run(head_dir, tmp_path / "b", dtype="float32", rows=ROWS, steps=2, minibatches=8000)
    assert result["updates"] == len(report["losses"]) == 2 * N_TRAIN and watch["train_rows"][-2 * N_TRAIN :] == [(1, 1)] * (2 * N_TRAIN)
    # The same tokens train as with one minibatch, each counted one time. The first pass is one more run of every row.
    one = run(head_dir, tmp_path / "c", dtype="float32", rows=ROWS, steps=2)[0]
    assert result["tokens"] == one["tokens"] > 0 and result["forward_tokens"] * 2 == one["forward_tokens"] * 3


def test_the_order_of_each_pass_comes_from_the_seed_and_a_replay_gives_the_same_lora(head_dir, tmp_path, watch):
    first = tree(head_dir, tmp_path / "a", steps=3, minibatches=4)
    passes, rows = watch["passes"], watch["rows"]
    assert len(passes) == 3 and passes[0] != passes[1] != passes[2] != passes[0], "each pass has an order of its own"
    train = sorted(i for i, a in enumerate(ADVANTAGES) if a != 0)
    for groups in passes:
        assert sorted(i for group in groups for i in group) == train and sorted(len(group) for group in groups) == [2, 2, 3, 3]
        assert all(group == sorted(group, key=lambda i: -len(rows[i]["tokens"])) for group in groups), "the longest row of a minibatch comes first"
    # The rule in words that a miner can follow: one random.Random(seed) for the job; each pass shuffles the rows that
    # train (the longest first, ties in file order); row k of the new order goes to minibatch k modulo minibatches.
    order, expected = random.Random(7), []
    for _ in range(3):
        mixed = sorted(train, key=lambda i: -len(rows[i]["tokens"]))
        order.shuffle(mixed)
        expected.append([sorted(mixed[k::4], key=lambda i: -len(rows[i]["tokens"])) for k in range(4)])
    assert passes == expected
    watch["passes"].clear()
    assert tree(head_dir, tmp_path / "b", steps=3, minibatches=4) == first and watch["passes"] == expected, "a replay: the same order and the same bytes"
    watch["passes"].clear()
    assert tree(head_dir, tmp_path / "c", steps=3, minibatches=4, seed=8) != first and watch["passes"] != expected


def test_the_adam_state_goes_through_the_minibatches_of_a_file_and_the_next_file_starts_a_new_one(head_dir, tmp_path, watch):
    first = tree(head_dir, tmp_path / "a", steps=2, minibatches=3)
    (adam,) = watch["adams"]
    b = next(weight for weight in adam.param_groups[0]["params"] if weight.shape[1] == 16 and float(weight.abs().sum()) > 0)
    assert adam.updates == 6 and int(adam.state[b]["step"]) == 6, "one optimizer for the file: its state has seen each update"
    assert tree(head_dir, tmp_path / "other", steps=1, minibatches=2, seed=9) != first
    assert tree(head_dir, tmp_path / "b", steps=2, minibatches=3) == first, "the file between left nothing behind"
    assert len(watch["adams"]) == 3 and len({id(adam) for adam in watch["adams"]}) == 3 and [adam.updates for adam in watch["adams"]] == [6, 2, 6]


def test_the_old_policy_the_mask_and_the_count_come_from_the_head_before_the_first_update(head_dir, tmp_path, watch, monkeypatch):
    train = sorted(i for i, a in enumerate(ADVANTAGES) if a != 0)
    train_job.train(job(head_dir, steps=1), tmp_path / "head", ["cpu"])  # one update: each row is scored under the head
    under_head = {i: (row["old"].clone(), row["mask"].clone(), row["count"]) for i, row in enumerate(watch["rows"]) if "old" in row}
    assert sorted(under_head) == train
    watch["first_pass"].clear()
    true = tree(head_dir, tmp_path / "true", steps=2, minibatches=5)
    assert sorted(watch["first_pass"]) == [(i, 0) for i in train], "each row one time, with no update before it"
    for i, (old, mask, count) in under_head.items():
        row = watch["rows"][i]
        assert torch.equal(row["old"], old) and torch.equal(row["mask"], mask) and row["count"] == count
    # The same job without the pass ahead: a row of a later minibatch takes its values from a model that has moved.
    # The lines above fail on it, and the LoRA is another one.
    watch["first_pass"].clear()
    monkeypatch.setattr(train_job, "score_old", lambda replica, rows: None)
    monkeypatch.setattr(train_job, "bring", lambda replica, rows: rows)
    late = tree(head_dir, tmp_path / "late", steps=2, minibatches=5)
    assert sorted(updates for _, updates in watch["first_pass"]) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] and late != true
    moved = [i for i, (old, _, _) in under_head.items() if not torch.equal(watch["rows"][i]["old"], old)]
    assert moved and all(updates > 0 for i, updates in watch["first_pass"] if i in moved)


@pytest.mark.parametrize("copies", [1, 2])
def test_the_scoring_pass_builds_no_graph(copies, head_dir, tmp_path, monkeypatch):
    """Each forward of ``score_old`` gives a hidden state with no graph behind it. The forwards of the training are the control: they have one."""
    seen, scoring = {"scoring": [], "training": []}, set()
    hidden, score_old = train_job.Replica.hidden, train_job.score_old

    def watched_hidden(self, ids):
        out = hidden(self, ids)
        seen["scoring" if threading.get_ident() in scoring else "training"].append(out.requires_grad or out.grad_fn is not None)
        return out

    def watched_score_old(replica, rows):
        scoring.add(threading.get_ident())  # each copy scores in a thread of its own
        try:
            return score_old(replica, rows)
        finally:
            scoring.discard(threading.get_ident())
            assert torch.is_grad_enabled(), "the pass leaves the gradient mode of its thread as it found it"

    monkeypatch.setattr(train_job.Replica, "hidden", watched_hidden)
    monkeypatch.setattr(train_job, "score_old", watched_score_old)
    train_job.train(job(head_dir, steps=2, minibatches=3), tmp_path / "lora", ["cpu"] * copies)
    assert seen["scoring"] == [False] * N_TRAIN, "one forward for each row that trains, and none of them builds a graph"
    assert seen["training"] == [True] * (copies + 2 * N_TRAIN), "the warm-up of each copy and each training forward build one"


def test_the_report_holds_minibatches_among_its_recipe_values_and_only_above_one_minibatch(head_dir, tmp_path):
    _, _, one = run(head_dir, tmp_path / "one", dtype="float32", steps=2)
    _, _, four = run(head_dir, tmp_path / "four", dtype="float32", steps=2, minibatches=4)
    assert "minibatches" not in one and "minibatches" not in one["recipe_values"], "one minibatch: the report of before"
    # The job of more minibatches names the recipe values itself: they are the ones that ``save`` names, and ``minibatches``.
    assert four["recipe_values"] == one["recipe_values"] | {"minibatches": 4} and set(four) == set(one)
    # ``run`` called by itself does not know minibatches: it makes one update in each pass and its report names no ``minibatches``.
    train_job.run(job(head_dir, steps=2, minibatches=4), tmp_path / "alone", ["cpu"])
    assert json.loads((tmp_path / "alone" / "train_report.json").read_text()) == one


def test_minibatches_on_more_copies_of_the_model_give_the_one_copy_result(head_dir, tmp_path):
    _, one, _ = run(head_dir, tmp_path / "one", dtype="float32", rows=ROWS, steps=2, minibatches=3)
    _, two, _ = run(head_dir, tmp_path / "two", dtype="float32", rows=ROWS, steps=2, minibatches=3, devices=("cpu", "cpu"))
    _, many, _ = run(head_dir, tmp_path / "many", dtype="float32", rows=ROWS, steps=2, minibatches=8000, devices=("cpu", "cpu", "cpu"))  # one row for three copies
    _, each, _ = run(head_dir, tmp_path / "each", dtype="float32", rows=ROWS, steps=2, minibatches=8000)
    assert all(torch.allclose(one[key], two[key], atol=1e-7) for key in one) and all(torch.allclose(each[key], many[key], atol=1e-7) for key in each)


# ---- B: the cap on the whole update ----


def test_the_job_scales_a_lora_over_the_cap_and_leaves_one_under_it(head_dir, tmp_path):
    big = dict(dtype="float32", rows=ROWS, steps=4, rank=64, alpha=128.0, lora_lr=1e-3)
    free, lora_free, _ = run(head_dir, tmp_path / "free", global_norm_cap=None, miner_bounds=TODAY_BOUNDS, **big)
    norm = free["step_norm"]
    assert 0 < norm < 6.0 and free["step_norm_before_cap"] == norm
    under, lora_under, report = run(head_dir, tmp_path / "under", **big)  # the default cap: 6
    assert report["recipe_values"]["global_norm_cap"] == 6.0 and (under["step_norm"], under["step_norm_before_cap"]) == (norm, norm)
    assert all(torch.equal(lora_under[key], lora_free[key]) for key in lora_free), "a LoRA under the cap keeps each byte"
    cap = Recipe().for_miner({"global_norm_cap": norm / 4}).global_norm_cap
    over, lora_over, _ = run(head_dir, tmp_path / "over", global_norm_cap=cap, **big)
    assert over["step_norm"] == cap and over["step_norm_before_cap"] == norm and update_norm(lora_over, 2.0) == pytest.approx(cap, rel=1e-4)
    for key in lora_free:  # the direction stays: each B is cap / norm times the B of before, each A is the same
        assert torch.equal(lora_over[key], lora_free[key]) if "lora_A" in key else torch.allclose(lora_over[key], lora_free[key] * (cap / norm), rtol=1e-5, atol=0)
    capped, _, _ = run(head_dir, tmp_path / "many", minibatches=5, global_norm_cap=cap, **big)
    assert capped["step_norm"] == cap < capped["step_norm_before_cap"], "the cap holds, however many updates there are"


def test_the_trainer_refuses_a_report_over_the_cap():
    report = {"status": "ok", "loss": 0.1, "step_norm": 6.0}
    assert LoraTrainer._check(dict(report), Recipe()) == report  # the default cap is 6 now
    with pytest.raises(InfraError, match="larger than recipe.global_norm_cap"):
        LoraTrainer._check(report | {"step_norm": 6.001}, Recipe())
    mine = Recipe().for_miner({"global_norm_cap": 0.05})
    assert LoraTrainer._check(report | {"step_norm": 0.05}, mine)["step_norm"] == 0.05
    with pytest.raises(InfraError, match="larger than recipe.global_norm_cap"):
        LoraTrainer._check(report | {"step_norm": 0.051}, mine)
    assert LoraTrainer._check(report | {"step_norm": 99.0}, Recipe(global_norm_cap=None, miner_bounds=()))["step_norm"] == 99.0  # null: no cap


# ---- the bounds of the two new miner values ----

REFUSED = {
    "minibatches": [0, 8001, 1.5, 1.0, True, False, "4", None, -1, [2], 10**400],
    "global_norm_cap": [0.009, 6.01, 0, 0.0, -1, -0.5, math.nan, math.inf, -math.inf, None, True, "1", [1.0], 10**400],
}
TAKEN = {"minibatches": [1, 2, 8000], "global_norm_cap": [0.01, 0.05, 1, 6, 6.0]}


@pytest.mark.parametrize("key,value", [(key, value) for key, values in REFUSED.items() for value in values])
def test_a_value_outside_a_new_bound_is_refused(key, value, build):
    recipe, params = Recipe(), build.params()
    text = "train.minibatches must be a whole number from 1 to 8000" if key == "minibatches" else "train.global_norm_cap must be a number from 0.01 to 6"
    with pytest.raises(BatchRejected) as refusal:
        recipe.for_miner({key: value})
    assert str(refusal.value) == text
    raw = json.dumps(json.loads(file()) | {"train": {key: value}}).encode()  # NaN and Infinity: the file fails before the bound
    assert rule(raw, params) == ("nonfinite" if isinstance(value, float) and not math.isfinite(value) else "train")


@pytest.mark.parametrize("key,value", [(key, value) for key, values in TAKEN.items() for value in values])
def test_a_value_inside_a_new_bound_is_taken_and_never_clamped(key, value, build):
    assert getattr(Recipe().for_miner({key: value}), key) == value
    batch = check_batch(file(train={key: value}), HOTKEY, build.params())
    assert batch.train == {key: value} and getattr(build.params().recipe.for_miner(batch.train), key) == value


def test_a_missing_field_gives_the_default_and_the_bounds_of_today_stay():
    recipe = Recipe()
    assert recipe.for_miner({}) is recipe and (recipe.minibatches, recipe.global_norm_cap, recipe.steps) == (1, 6.0, 1)
    assert recipe.for_miner({"minibatches": 16}).global_norm_cap == 6.0 and recipe.for_miner({"global_norm_cap": 0.5}).minibatches == 1
    new = {bound.key: bound for bound in DEFAULT_MINER_BOUNDS}
    assert new["minibatches"] == MinerBound("minibatches", 1, 8000, integer=True, decided=True) and new["global_norm_cap"] == MinerBound("global_norm_cap", 0.01, 6.0, decided=True)
    assert new["steps"] == MinerBound("steps", 1, 4, integer=True), "steps stays 1 to 4"
    assert new["target_modules"].choices[:7] == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), "the 7 names of today stay, in their order"


def test_the_recipe_and_the_params_of_a5c2b6a6_still_parse_and_no_bound_of_today_gets_tighter():
    """``params_default_a5c2b6a6.json`` is ``Params().to_json()`` of a5c2b6a6: no ``minibatches``, a null cap, 7 bounds, the old caps."""
    raw = OLD_PARAMS.read_bytes()
    assert sha256_hex(raw) == OLD_PARAMS_HASH and "minibatches" not in json.loads(raw)["recipe"]
    old = Params.from_json(raw)
    assert (old.recipe.minibatches, old.recipe.global_norm_cap) == (1, None), "a stored recipe with no new field: one minibatch, and its cap stays as stored"
    assert (old.max_train_token_steps, old.max_prompts, old.max_rollouts, old.max_batch_bytes) == (8_000_000, 250, 2000, 64 * 2**20)
    assert [bound.key for bound in old.recipe.miner_bounds] == ["lora_lr", "steps", "group_size", "sample_temperature", "rank", "alpha", "target_modules"]
    for key in ("minibatches", "global_norm_cap"):  # the stored bounds are the bounds: such a recipe gives a miner no new key
        with pytest.raises(BatchRejected, match="a key that a miner may not set"):
            old.recipe.for_miner({key: 1})
    # The stored form of today stays, byte for byte, and so does each id: a recipe with one minibatch does not write the field.
    assert old.to_json() == raw and old.params_hash == OLD_PARAMS_HASH == digest(json.loads(raw))
    assert old.recipe_id == "8cab33a2a1b4b459d7a42ada287823d1c062c2e2a2ed7102009c8475f0748ded"
    more = dataclasses.replace(old.recipe, minibatches=2)
    assert more.to_dict() == old.recipe.to_dict() | {"minibatches": 2} and Recipe.from_dict(more.to_dict()) == more and more.recipe_id != old.recipe_id
    assert Recipe.from_dict(old.recipe.to_dict() | {"minibatches": 1}) == old.recipe, "a file that writes the 1 is the same recipe"
    # Each bound of today is a bound of the new defaults, unchanged, and no cap of the new defaults is lower.
    # ``target_modules`` is wider: its 7 choices of today are the first 7 of the new defaults.
    new = Params()
    assert all(bound in new.recipe.miner_bounds for bound in old.recipe.miner_bounds if bound.key != "target_modules")
    assert new.recipe.bound("target_modules").choices[:7] == old.recipe.bound("target_modules").choices
    assert (new.max_train_token_steps, new.max_prompts, new.max_rollouts, new.max_batch_bytes) == (16_000_000, 2000, 8000, 128 * 2**20)
    assert Params.from_json(new.to_json()) == new and new.recipe.steps == old.recipe.steps == 1


def test_the_four_caps_and_the_bounds_are_values_of_the_params_file():
    """What ``set-params`` reads: a file of changes on the defaults. The token cap can stay at 8M, and a bound can be lowered, with no new code."""
    kept = Params.from_dict({"max_train_token_steps": 8_000_000})
    assert (kept.max_train_token_steps, kept.max_prompts, kept.max_rollouts, kept.max_batch_bytes) == (8_000_000, 2000, 8000, 128 * 2**20)
    bounds = [bound.to_dict() | ({"high": 2000.0} if bound.key == "minibatches" else {"high": 2.0} if bound.key == "global_norm_cap" else {}) for bound in DEFAULT_MINER_BOUNDS]
    lower = Params.from_dict({"recipe": {"global_norm_cap": 2.0, "miner_bounds": bounds}}).recipe
    assert lower.for_miner({"minibatches": 2000, "global_norm_cap": 2}).global_norm_cap == 2.0 and lower.for_miner({}).global_norm_cap == 2.0
    for train in ({"minibatches": 2001}, {"global_norm_cap": 2.01}):
        with pytest.raises(BatchRejected):
            lower.for_miner(train)


@pytest.mark.parametrize(
    "change",
    [
        {"minibatches": 0}, {"minibatches": -1}, {"global_norm_cap": 0.0}, {"global_norm_cap": -1.0}, {"global_norm_cap": math.nan}, {"global_norm_cap": math.inf},
        {"global_norm_cap": 6.5},  # above the high end of the miner bound: a file that sets no cap would get more than a file may set
        {"global_norm_cap": None},  # no cap for a file that sets none, under a bound that gives each file a ceiling
    ],
)
def test_a_recipe_that_breaks_the_new_rules_fails_closed(change):
    with pytest.raises(ConfigError):
        dataclasses.replace(Recipe(), **change)
    assert Recipe(global_norm_cap=None, miner_bounds=DEFAULT_MINER_BOUNDS[:-1]).global_norm_cap is None, "with no miner bound for the cap, null is still no cap"
