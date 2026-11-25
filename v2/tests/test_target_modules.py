"""The projections of the linear-attention layers as choices of ``target_modules``: the rules, the public text, the trainer and the fold (CPU).

No test here runs the server that scores a LoRA. That the server applies an
adapter on these layers as the trainer does is a check on a GPU.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import json
import string
import sys
from pathlib import Path

import pytest

# As in test_gpu_torch.py: the tiny model runs on the CPU, so transformers must take its plain torch path.
sys.modules["fla"] = None  # type: ignore[assignment]

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from safetensors.torch import load_file  # noqa: E402 - after the skip guards

from sn120_v2 import publisher, sim  # noqa: E402
from sn120_v2.admission import Invalid, check_batch, verdict  # noqa: E402
from sn120_v2.errors import BatchRejected, ConfigError  # noqa: E402
from sn120_v2.fold_kernel import FIXED_ORDER, lora_update, read_pairs, replay_fold, upcast_master  # noqa: E402
from sn120_v2.gpu import train_job  # noqa: E402
from sn120_v2.gpu.folder import fold_files, read_lora, upcast  # noqa: E402
from sn120_v2.gpu.resident import hash_tree  # noqa: E402
from sn120_v2.params import ATTENTION_AND_MLP_PROJECTIONS, OPTIONAL_TARGET_MODULES, TARGET_MODULE_CHOICES, MinerBound, Params, Recipe  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("affine_miner_tool_targets", Path(__file__).resolve().parents[1] / "miner" / "affine_miner.py")
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)

HOTKEY = "5MinerA"
THINK = "step " * 40  # 40 reasoning tokens: above the least of 32
TEMPLATE = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
SEVEN = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NEW = ("in_proj_qkv", "in_proj_z", "out_proj")
OLD_PARAMS = Path(__file__).parent / "data" / "params_default_a5c2b6a6.json"  # ``Params().to_json()`` of a5c2b6a6: 7 choices
SEVEN_CHOICES = tuple(MinerBound("target_modules", choices=SEVEN, decided=True) if bound.key == "target_modules" else bound for bound in Recipe().miner_bounds)
ROWS = [
    {"prompt": "ab c", "output": "abc", "reward": 1.0, "advantage": 0.5},
    {"prompt": "ab c", "output": "ba", "reward": 0.0, "advantage": -0.5},
    {"prompt": "x", "output": "ccdd", "reward": 1.0, "advantage": 1.0},
]
PAIR_TEXT = "train.target_modules with in_proj_z must hold in_proj_qkv too"
SUBSET_TEXT = f"train.target_modules must be a non-empty subset of {list(SEVEN + NEW)}"


def file(train: dict) -> bytes:
    """A valid file of 10 prompts with 2 rollouts each, with a ``train`` object."""
    right = lambda n, k: {"output": f"{THINK}{k} \\boxed{{{2 * n}}}", "reward": 1}  # noqa: E731
    groups = [{"prompt": f"double {n}", "rollouts": [right(n, 0), right(n, 1) if n < 9 else {"output": f"{THINK}1 \\boxed{{-1}}", "reward": 0}]} for n in range(10)]
    return json.dumps({"version": "affine-batch-v3", "hotkey": HOTKEY, "groups": groups, "train": train}).encode()


# ---- the rules ----


def test_the_three_names_are_choices_and_the_default_keeps_the_seven_names():
    recipe = Recipe()
    assert ATTENTION_AND_MLP_PROJECTIONS == SEVEN and OPTIONAL_TARGET_MODULES == NEW and TARGET_MODULE_CHOICES == SEVEN + NEW
    assert recipe.target_modules == SEVEN, "the default recipe does not hold a new name"
    assert recipe.bound("target_modules") == MinerBound("target_modules", choices=SEVEN + NEW, decided=True)
    assert recipe.for_miner({}) is recipe and recipe.for_miner({"rank": 8}).target_modules == SEVEN, "a file that names no list trains the 7 names"


TAKEN = [
    (["in_proj_qkv"], ("in_proj_qkv",)),
    (["out_proj"], ("out_proj",)),
    (["in_proj_z", "in_proj_qkv"], ("in_proj_qkv", "in_proj_z")),
    (["out_proj", "in_proj_z", "in_proj_qkv"], NEW),
    (["out_proj", "in_proj_qkv", "down_proj", "q_proj"], ("q_proj", "down_proj", "in_proj_qkv", "out_proj")),
    ([*NEW, *SEVEN], SEVEN + NEW),
    (["v_proj", "q_proj"], ("q_proj", "v_proj")),
]


@pytest.mark.parametrize("names,stored", TAKEN)
def test_a_list_with_new_names_is_taken_and_stored_in_one_fixed_order(names, stored, build):
    assert Recipe().for_miner({"target_modules": names}).target_modules == stored
    batch = check_batch(file({"target_modules": names}), HOTKEY, build.params())
    assert batch.train == {"target_modules": names} and build.params().recipe.for_miner(batch.train).target_modules == stored


Z_AND_NO_QKV = [["in_proj_z"], ["in_proj_z", "out_proj"], ["q_proj", "in_proj_z"], [*SEVEN, "in_proj_z", "out_proj"]]


@pytest.mark.parametrize("names", Z_AND_NO_QKV)
def test_a_list_with_in_proj_z_and_no_in_proj_qkv_is_refused(names, build):
    with pytest.raises(BatchRejected) as refusal:
        Recipe().for_miner({"target_modules": names})
    assert str(refusal.value) == PAIR_TEXT
    with pytest.raises(Invalid) as invalid:  # the whole file is invalid, before any training
        check_batch(file({"target_modules": names}), HOTKEY, build.params())
    assert invalid.value.rule == "train" and str(invalid.value) == PAIR_TEXT
    with pytest.raises(ConfigError, match="recipe.target_modules with in_proj_z must hold in_proj_qkv too"):  # the validator's own value too
        Recipe(target_modules=tuple(names))
    assert Recipe(target_modules=("in_proj_qkv", "in_proj_z")).target_modules == ("in_proj_qkv", "in_proj_z")


@pytest.mark.parametrize("names", Z_AND_NO_QKV)
def test_the_validator_gives_such_a_file_the_verdict_not_valid_and_the_cause_names_the_rule(names, build):
    """The verdict is the record that the validator keeps of a stored file. Such a file never gets a place in a round."""
    bad = verdict(file({"target_modules": names}), HOTKEY, build.params(), now=7)
    assert (bad.valid, bad.rule, bad.reason, bad.checked_at) == (False, "train", PAIR_TEXT, 7)
    assert (bad.format, bad.n_prompts, bad.n_rollouts) == ("", 0, 0) and len(bad.batch_sha256) == 64
    good = verdict(file({"target_modules": [*names, "in_proj_qkv"]}), HOTKEY, build.params(), now=7)
    assert (good.valid, good.rule, good.reason) == (True, "", ""), "the same list with in_proj_qkv is valid"


def test_of_the_1023_lists_of_the_ten_names_the_bound_refuses_each_one_with_in_proj_z_and_no_in_proj_qkv_and_no_other():
    """The server refuses the load of a LoRA that changes ``in_proj_z`` and not ``in_proj_qkv``. The bound must refuse each such list."""
    recipe, refused = Recipe(), []
    lists = [list(names) for size in range(1, 11) for names in itertools.combinations(SEVEN + NEW, size)]
    for names in lists:
        try:
            assert recipe.for_miner({"target_modules": names}).target_modules == tuple(names)
        except BatchRejected as exc:
            assert str(exc) == PAIR_TEXT
            refused.append(names)
    assert len(lists) == 1023 and refused == [names for names in lists if "in_proj_z" in names and "in_proj_qkv" not in names] and len(refused) == 256


@pytest.mark.parametrize(
    "names",
    [["in_proj_a"], ["in_proj_b"], ["in_proj_qkv", "in_proj_a"], [*NEW, "in_proj_b"], ["in_proj_qkvz"], ["linear_attn"], ["lm_head"], ["embed_tokens"], ["IN_PROJ_QKV"]],
)
def test_in_proj_a_in_proj_b_and_each_other_name_are_refused(names, build):
    with pytest.raises(BatchRejected) as refusal:
        Recipe().for_miner({"target_modules": names})
    assert str(refusal.value) == SUBSET_TEXT
    with pytest.raises(Invalid) as invalid:
        check_batch(file({"target_modules": names}), HOTKEY, build.params())
    assert invalid.value.rule == "train"


@pytest.mark.parametrize("value", [[], ["in_proj_qkv", "in_proj_qkv"], "in_proj_qkv", [["in_proj_qkv"]], None, 3, {"in_proj_qkv": 1}])
def test_a_wrong_form_of_the_list_is_refused_as_before(value):
    with pytest.raises(BatchRejected) as refusal:
        Recipe().for_miner({"target_modules": value})
    assert str(refusal.value) == SUBSET_TEXT


def test_a_recipe_with_the_seven_choices_gives_no_new_name_and_keeps_its_id():
    """The stored bounds are the bounds. A recipe that holds the 7 choices (the stored form of a5c2b6a6) offers no new name."""
    old = Params.from_json(OLD_PARAMS.read_bytes()).recipe
    assert old.bound("target_modules").choices == SEVEN
    for names in (["in_proj_qkv"], list(NEW), ["q_proj", "out_proj"]):
        with pytest.raises(BatchRejected, match="must be a non-empty subset of"):
            old.for_miner({"target_modules": names})
    assert old.recipe_id == "8cab33a2a1b4b459d7a42ada287823d1c062c2e2a2ed7102009c8475f0748ded", "the id of a stored recipe does not change"
    seven, ten = Recipe(miner_bounds=SEVEN_CHOICES), Recipe()
    assert seven.recipe_id != ten.recipe_id, "the choices are part of the recipe: the wider list gives a new recipe_id"
    assert dataclasses.replace(ten, miner_bounds=SEVEN_CHOICES) == seven, "the list of choices is the one difference"
    assert seven.for_miner({"target_modules": ["v_proj", "q_proj"]}).target_modules == ten.for_miner({"target_modules": ["v_proj", "q_proj"]}).target_modules


# ---- the public text ----


def published(tmp_path, recipe: Recipe) -> tuple[dict, str]:
    """mining.json as a document and llms.txt as one line, for a controller with this recipe."""
    snap = sim.SimWorld(tmp_path / "world", sim.sim_params(recipe=recipe)).publisher.snapshots[-1]
    mining, llms = publisher.render(snap, "https://files.example/")
    return json.loads(mining), " ".join(llms.decode().split())


def test_the_public_files_state_the_new_names_and_the_pairing_rule(tmp_path):
    doc, text = published(tmp_path / "ten", Recipe(image_digests=(sim.SIM_IMAGE,)))
    row = next(row for row in doc["contract"]["miner_settable"] if row["key"] == "target_modules")
    allowed = "a list: a non-empty subset of q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, in_proj_qkv, in_proj_z, out_proj; a list with in_proj_z must hold in_proj_qkv too"
    assert (row["choices"], row["default"], row["allowed"]) == (list(SEVEN + NEW), list(SEVEN), allowed)
    assert f"| `train.target_modules` | {', '.join(SEVEN)} | {allowed} |" in text
    assert f"`target_modules`: the default is {', '.join(SEVEN)}. You may also name `in_proj_qkv`, `in_proj_z`, `out_proj`: projections of the linear-attention layers of the head." in text
    assert "`in_proj_a` and `in_proj_b` are not choices." in text
    assert "A list with `in_proj_z` must hold `in_proj_qkv` too." in text and "it does not load an adapter for `in_proj_z` alone" in text
    assert "A list with `in_proj_z` and no `in_proj_qkv` makes the whole file invalid. `in_proj_qkv` without `in_proj_z` is allowed." in text
    assert f"each linear layer whose name is one of `target_modules` ({', '.join(SEVEN)})" in text, "the recipe value is the default list"
    assert allowed in tool.bounds_text(Params.from_dict(doc["params"])), "the miner tool reads the same text"
    with pytest.raises(BatchRejected, match=PAIR_TEXT):
        tool.applied_train(Params.from_dict(doc["params"]), {"target_modules": ["in_proj_z"]})
    assert tool.applied_train(Params.from_dict(doc["params"]), {"target_modules": list(NEW)})["target_modules"] == list(NEW)


@pytest.mark.parametrize("names", Z_AND_NO_QKV)
def test_the_check_of_the_miner_tool_refuses_such_a_file_with_the_words_of_the_validator(names, tmp_path, capsys):
    snap = sim.SimWorld(tmp_path / "world", sim.sim_params(recipe=Recipe(image_digests=(sim.SIM_IMAGE,)))).publisher.snapshots[-1]
    mining, _ = publisher.render(snap, "https://files.example/")
    params = Params.from_dict(json.loads(mining)["params"])
    with pytest.raises(Invalid) as invalid:
        tool.check(file({"target_modules": names}), HOTKEY, params)
    assert invalid.value.rule == "train" and str(invalid.value) == PAIR_TEXT == verdict(file({"target_modules": names}), HOTKEY, params).reason
    # The command of the tool, with the published mining.json: exit code 1 and the same words.
    (tmp_path / "mining.json").write_bytes(mining)
    (tmp_path / "bad.json").write_bytes(file({"target_modules": names}))
    (tmp_path / "good.json").write_bytes(file({"target_modules": [*names, "in_proj_qkv"]}))
    command = ["check", "--mining-json", str(tmp_path / "mining.json"), "--hotkey", HOTKEY]
    assert tool.main([*command, str(tmp_path / "bad.json")]) == 1
    assert json.loads(capsys.readouterr().out) == {"valid": False, "rule": "train", "reason": PAIR_TEXT}
    assert tool.main([*command, str(tmp_path / "good.json")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True and report["train_applied"]["target_modules"] == [name for name in SEVEN + NEW if name in [*names, "in_proj_qkv"]]


def test_a_recipe_with_the_seven_choices_publishes_no_word_on_the_new_names(tmp_path):
    doc, text = published(tmp_path / "seven", Recipe(image_digests=(sim.SIM_IMAGE,), miner_bounds=SEVEN_CHOICES))
    row = next(row for row in doc["contract"]["miner_settable"] if row["key"] == "target_modules")
    assert (row["choices"], row["allowed"]) == (list(SEVEN), "a list: a non-empty subset of " + ", ".join(SEVEN))
    assert "in_proj" not in text and "out_proj" not in text and "You may also name" not in text
    assert "in_proj" not in json.dumps(doc) and "out_proj" not in json.dumps(doc)
    only_two = tuple(MinerBound("target_modules", choices=SEVEN + ("in_proj_qkv", "out_proj"), decided=True) if b.key == "target_modules" else b for b in Recipe().miner_bounds)
    _, text = published(tmp_path / "two", Recipe(image_digests=(sim.SIM_IMAGE,), miner_bounds=only_two))
    assert "You may also name `in_proj_qkv`, `out_proj`:" in text and "in_proj_z" not in text, "the text follows the choices of the recipe"


# ---- the trainer and the fold, on a tiny model of the head's architecture ----


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


def run(head_dir: Path, out: Path, dtype: str = "bfloat16", **recipe) -> tuple[dict, dict]:
    """Run the training job. Returns (result, LoRA tensors)."""
    recipe = Recipe(**({"steps": 2, "lora_lr": 1e-4} | recipe))
    job = {"base_dir": str(head_dir), "base_hash": "b" * 64, "dtype": dtype, "recipe": recipe.to_dict(), "seed": 7, "rows": ROWS}
    return train_job.train(job, out, ["cpu"]), load_file(str(out / "adapter_model.safetensors"))


def paths(lora: dict) -> set[str]:
    return {key[len("base_model.model.") : -len(".lora_A.weight")] for key in lora if key.endswith(".lora_A.weight")}


def test_the_trainer_puts_an_adapter_on_each_layer_with_a_new_name_and_trains_it(head_dir, tmp_path):
    stored = load_file(str(head_dir / "model.safetensors"))
    in_head = {key[: -len(".weight")] for key in stored if key.split(".")[-2] in NEW}
    assert len(in_head) == 9 and all(".linear_attn." in path for path in in_head) and not any("visual" in path for path in in_head), "3 linear-attention layers x 3 names"
    result, lora = run(head_dir, tmp_path / "new", target_modules=NEW, rank=8, alpha=16.0)
    assert result["status"] == "ok" and result["step_norm"] > 0 and paths(lora) == in_head
    assert all(float(lora[f"base_model.model.{path}.lora_B.weight"].abs().max()) > 0 for path in in_head), "each new layer got an update"
    for names in (("in_proj_qkv",), ("out_proj",), ("in_proj_qkv", "in_proj_z")):
        _, one = run(head_dir, tmp_path / "-".join(names), target_modules=names, rank=8, alpha=16.0)
        assert {path.split(".")[-1] for path in paths(one)} == set(names) and len(paths(one)) == 3 * len(names)
    _, both = run(head_dir, tmp_path / "ten", target_modules=SEVEN + NEW)
    _, seven = run(head_dir, tmp_path / "seven")
    assert paths(both) == paths(seven) | in_head and len(paths(seven)) == 16 and not paths(seven) & in_head


def test_the_default_lora_has_the_same_bytes_with_seven_and_with_ten_choices(head_dir, tmp_path):
    # float32, as the byte tests of test_gpu_torch.py: float32 on a CPU uses no AMX.
    for name, bounds in (("seven", SEVEN_CHOICES), ("ten", Recipe().miner_bounds)):
        run(head_dir, tmp_path / name, dtype="float32", miner_bounds=bounds)
        run(head_dir, tmp_path / f"{name}-mlp", dtype="float32", miner_bounds=bounds, target_modules=("gate_proj", "up_proj", "down_proj"), steps=3)
    assert hash_tree(tmp_path / "seven") == hash_tree(tmp_path / "ten"), "the LoRA, its config and the train report"
    assert hash_tree(tmp_path / "seven-mlp") == hash_tree(tmp_path / "ten-mlp")


def test_a_lora_on_the_new_names_folds_into_the_weights_of_those_layers(head_dir, tmp_path):
    run(head_dir, tmp_path / "lora", target_modules=NEW, rank=8, alpha=16.0, lora_lr=1e-2, steps=4, global_norm_cap=None, miner_bounds=())
    updates = read_lora(tmp_path / "lora", 8)
    old = load_file(str(head_dir / "model.safetensors"))
    assert len(updates) == 9 and all(key.split(".")[-2] in NEW and update.shape == old[key].shape and update.dtype == torch.float32 for key, update in updates.items())
    upcast(head_dir, tmp_path / "master")
    fold_files(tmp_path / "master", head_dir, [updates], tmp_path / "new_master", tmp_path / "new_base")
    master = {}
    for part in sorted((tmp_path / "new_master").glob("*.safetensors")):
        master.update(load_file(str(part)))
    new = load_file(str(tmp_path / "new_base" / "model.safetensors"))
    assert sorted(new) == sorted(old) == sorted(master)
    changed = 0
    for key, weight in old.items():
        if key in updates:
            assert torch.equal(master[key], weight.float() + updates[key]) and torch.equal(new[key], master[key].to(weight.dtype)) and new[key].dtype == weight.dtype
            changed += int(not torch.equal(new[key], weight))
        else:
            assert torch.equal(new[key], weight), f"{key} has no adapter and does not change"
    assert changed == 9, "the update of each new layer is large enough to change its bf16 weight"


def test_the_fold_of_the_box_folds_a_lora_on_the_new_names_with_the_fixed_order_kernel(head_dir, tmp_path):
    # The box folds with ``fold_kernel.replay_fold``; ``folder.fold_files`` of the test above is the older path.
    run(head_dir, tmp_path / "lora", target_modules=NEW, rank=8, alpha=16.0, lora_lr=1e-2, steps=4, global_norm_cap=None, miner_bounds=())
    lora = read_pairs(tmp_path / "lora", 8)
    old = load_file(str(head_dir / "model.safetensors"))
    assert len(lora.pairs) == 9 and all(key.split(".")[-2] in NEW and key in old for key in lora.pairs)
    upcast_master(head_dir, tmp_path / "master")
    folded = replay_fold(tmp_path / "master", head_dir, [lora], tmp_path / "new_master", tmp_path / "new_base", kernel=FIXED_ORDER)
    assert folded == replay_fold(tmp_path / "master", head_dir, [lora], None, None, kernel=FIXED_ORDER, threads=1), "hash only, one thread: the same files"
    assert folded.base_files == hash_tree(tmp_path / "new_base") and folded.master_files == hash_tree(tmp_path / "new_master")
    new = load_file(str(tmp_path / "new_base" / "model.safetensors"))
    assert sorted(new) == sorted(old)
    changed = 0
    for key, weight in old.items():
        if key in lora.pairs:
            want = weight.float() + lora_update(*lora.pairs[key], lora.scale, FIXED_ORDER)
            assert new[key].dtype == weight.dtype and torch.equal(new[key], want.to(weight.dtype))
            changed += int(not torch.equal(new[key], weight))
        else:
            assert torch.equal(new[key], weight), f"{key} has no adapter and does not change"
    assert changed == 9
