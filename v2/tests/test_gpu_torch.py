"""The GRPO step, the fold and the genesis command on a tiny model of the head's own architecture (CPU)."""

from __future__ import annotations

import json
import math
import shutil
import string
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from safetensors.torch import load_file, save_file  # noqa: E402 - after the skip guards

from sn120_v2 import layout  # noqa: E402
from sn120_v2.canonical import manifest_digest  # noqa: E402
from sn120_v2.errors import BatchRejected, ContractError, InfraError, VerifyError  # noqa: E402
from sn120_v2.gpu import genesis, train_job  # noqa: E402
from sn120_v2.gpu.folder import LoraFolder, read_lora, upcast  # noqa: E402
from sn120_v2.gpu.plan import BoxConfig  # noqa: E402
from sn120_v2.gpu.resident import hash_tree, upload_tree  # noqa: E402
from sn120_v2.params import Recipe  # noqa: E402
from sn120_v2.store import LocalStore  # noqa: E402
from sn120_v2.types import HeadVersion, LoRARef  # noqa: E402

ROWS = [
    {"prompt": "ab c", "output": "abc", "reward": 1.0, "advantage": 0.5},
    {"prompt": "ab c", "output": "ba", "reward": 0.0, "advantage": -0.5},
    {"prompt": "x", "output": "ccdd", "reward": 1.0, "advantage": 1.0},
]
TEMPLATE = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"


@pytest.fixture(autouse=True)
def one_thread():
    """bf16 on CPU is not repeatable when torch runs threads inside threads; one thread is."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def head_dir(tmp_path_factory) -> Path:
    """A bf16 checkpoint with the class, layer types and tensor names of the launch model, and a character tokenizer."""
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


def run(head_dir: Path, out: Path, devices=("cpu",), dtype: str = "bfloat16", seed: int = 7, rows=ROWS, **recipe) -> tuple[dict, dict, dict]:
    """Run the training job. Returns (result, LoRA tensors, the train report file)."""
    recipe = Recipe(**({"steps": 2, "lora_lr": 1e-4} | recipe))
    job = {"base_dir": str(head_dir), "base_hash": "b" * 64, "dtype": dtype, "recipe": recipe.to_dict(), "seed": seed, "rows": rows}
    result = train_job.run(job, out, list(devices))
    return result, load_file(str(out / "adapter_model.safetensors")), json.loads((out / "train_report.json").read_text())


def update_norm(lora: dict, scale: float) -> float:
    pairs = [(lora[key], lora[key.replace("lora_A", "lora_B")]) for key in lora if "lora_A" in key]
    return math.sqrt(sum(float(((scale * b @ a) ** 2).sum()) for a, b in pairs))


# ---- the step ----


def test_the_lora_has_the_names_of_the_weight_files_and_the_reported_norm(head_dir, tmp_path):
    result, lora, report = run(head_dir, tmp_path / "lora")
    stored = set(load_file(str(head_dir / "model.safetensors")))
    targets = {key[len("base_model.model.") : -len(".lora_A.weight")] for key in lora if key.endswith(".lora_A.weight")}
    assert len(targets) == 16 and all(f"{path}.weight" in stored for path in targets)  # so every LoRA can be folded
    assert {path.split(".")[-1] for path in targets} == set(Recipe().target_modules) and not any("visual" in path for path in targets)
    assert {tensor.dtype for tensor in lora.values()} == {torch.float32} and all(a.shape[0] == 16 for key, a in lora.items() if "lora_A" in key)
    assert result["status"] == "ok" and 0 < result["step_norm"] == pytest.approx(update_norm(lora, 32 / 16), rel=1e-4)
    assert report["optimizer"].startswith("adam, fresh per LoRA") and report["truncated_rows"] == 0 and len(report["losses"]) == 2
    assert report["recipe_values"]["lora_lr"] == 1e-4 and all(math.isfinite(n) and n > 0 for n in report["grad_norms"])
    config = json.loads((tmp_path / "lora" / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"], config["peft_type"]) == (16, 32.0, "LORA")


def test_same_seed_same_lora_and_replicas_give_the_one_gpu_result(head_dir, tmp_path):
    first = hash_tree(run(head_dir, tmp_path / "a") and tmp_path / "a")
    assert hash_tree(run(head_dir, tmp_path / "b") and tmp_path / "b") == first
    assert hash_tree(run(head_dir, tmp_path / "c", seed=8) and tmp_path / "c") != first
    _, one, _ = run(head_dir, tmp_path / "one", dtype="float32")
    _, two, _ = run(head_dir, tmp_path / "two", dtype="float32", devices=("cpu", "cpu"))  # the elastic split: more GPUs, the same step
    assert all(torch.allclose(one[key], two[key], atol=1e-7) for key in one)


def test_miner_values_change_the_step_and_the_norm_cap_scales_the_whole_update(head_dir, tmp_path):
    base, _, _ = run(head_dir, tmp_path / "base")
    hot, _, _ = run(head_dir, tmp_path / "hot", sample_temperature=1.5)
    small, _, report = run(head_dir, tmp_path / "small", rank=8, alpha=8.0, steps=1, target_modules=("q_proj",))
    assert hot["loss"] != base["loss"] and small["step_norm"] < base["step_norm"] and len(report["losses"]) == 1
    capped, lora, _ = run(head_dir, tmp_path / "cap", global_norm_cap=base["step_norm"] / 4)
    assert capped["step_norm"] == pytest.approx(base["step_norm"] / 4) == pytest.approx(update_norm(lora, 2.0), rel=1e-4)


def test_long_rows_are_cut_and_counted_and_a_batch_without_gradient_is_rejected(head_dir, tmp_path):
    result, _, report = run(head_dir, tmp_path / "cut", train_max_output_tokens=3)
    assert report["truncated_rows"] == 1 and result["status"] == "ok"  # "ccdd" is 4 tokens: cut to 3, not rejected
    flat = [dict(row, advantage=0.0) for row in ROWS]
    with pytest.raises(BatchRejected):
        run(head_dir, tmp_path / "flat", rows=flat)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({"base_dir": str(head_dir), "base_hash": "b" * 64, "dtype": "bfloat16", "recipe": Recipe().to_dict(), "seed": 1, "rows": flat}))
    train_job.main(job_dir)
    assert json.loads((job_dir / "report.json").read_text())["status"] == "rejected"
    with pytest.raises(InfraError, match="match no tensor"):
        train_job.check_names(str(head_dir), ["model.layers.0.mlp.gate_proj"])  # the text-only class would give this name


def test_support_renorm_temperature_and_the_clips():
    recipe = Recipe(support_top_k=3, support_top_p=0.9, sample_temperature=2.0)
    logits = torch.tensor([[4.0, 2.0, 0.0, -8.0], [4.0, 2.0, 0.0, -8.0]])
    log_p, log_q, inside = train_job.token_scores(logits, torch.tensor([1, 3]), recipe)
    kept = torch.tensor([4.0, 2.0, 0.0]) / 2.0  # top-3 at temperature 2; top-p 0.9 keeps all three
    assert log_q[0] == pytest.approx(float(1.0 - kept.logsumexp(0))) and log_p[0] == pytest.approx(float(2.0 - logits[0].logsumexp(0)))
    assert inside.tolist() == [True, False]  # a token the head could not sample does not train
    ratio = torch.tensor([0.0, math.log(2.0)])
    assert train_job.ppo_loss(ratio, torch.zeros(2), 1.0, Recipe()).tolist() == pytest.approx([-1.0, -1.2])  # PPO clip at 1 + 0.2
    assert train_job.ppo_loss(torch.tensor([math.log(9.0)]), torch.zeros(1), -1.0, Recipe()).tolist() == pytest.approx([3.0])  # dual clip


# ---- the fold and genesis ----


@pytest.fixture
def folded(head_dir, tmp_path, build):
    """A head in the store (base and fp32 master), one trained LoRA on it, and the fold of that head."""
    store, config = LocalStore(tmp_path / "store"), BoxConfig(tmp_path / "work", build.image)
    master_digest = upcast(head_dir, tmp_path / "master", part_bytes=40_000)
    base_files, _ = upload_tree(store, head_dir, "tmp/base")
    base_hash = manifest_digest(base_files)
    upload_tree(store, head_dir, layout.public_base_prefix(base_hash))
    upload_tree(store, tmp_path / "master", layout.public_master_prefix(master_digest))
    genesis_head = HeadVersion.make(parent_version_id=None, base_hash=base_hash, base_fp32_master_digest=master_digest, loras=(), round_id="genesis", applied_at=1)
    run(Path(head_dir), tmp_path / "lora", lora_lr=1e-3)
    lora_files, _ = upload_tree(store, tmp_path / "lora", "tmp/lora")
    ref = LoRARef(manifest_digest(lora_files), "r000001", "hk7", 7, build.h("batch"), build.params().recipe_id, 16)
    upload_tree(store, tmp_path / "lora", layout.public_lora_prefix(ref.lora_hash))
    head = HeadVersion.make(parent_version_id=genesis_head.version_id, base_hash=base_hash, base_fp32_master_digest=master_digest, loras=(ref,), round_id="r000001", applied_at=2)
    return store, config, head, build.params(), tmp_path


def test_fold_starts_from_the_fp32_master_and_rounds_one_time(folded, head_dir):
    store, config, head, params, tmp = folded
    assert len(list((tmp / "master").iterdir())) > 1 and all(f.stat().st_size < 60_000 for f in (tmp / "master").iterdir())  # parts stay under one upload
    artifact = LoraFolder(store, config).fold(head, params, "pending/folds/v1")
    assert artifact.parent_version_id == head.version_id and artifact.base_hash == manifest_digest(artifact.base_files) != head.base_hash
    assert artifact.base_fp32_master_digest == manifest_digest(artifact.master_files) != head.base_fp32_master_digest
    assert all(store.sha256(f"pending/folds/v1/base/{name}") == sha for name, sha in artifact.base_files.items())
    assert not set(artifact.base_files) & set(artifact.master_files) and "tokenizer.json" in artifact.base_files  # two file sets, two prefixes
    cache = config.work_dir / "cache"
    old = load_file(str(head_dir / "model.safetensors"))
    new_base = load_file(str(cache / artifact.base_hash / "model.safetensors"))  # adopted: the next round downloads nothing
    new_master = {name: t for part in sorted((cache / artifact.base_fp32_master_digest).iterdir()) for name, t in load_file(str(part)).items()}
    deltas = read_lora(tmp / "lora", 16)
    assert sorted(new_master) == sorted(old) == sorted(new_base) and len(deltas) == 16
    for name, weight in old.items():
        want = weight.float() + deltas[name] if name in deltas else weight.float()
        assert torch.equal(new_master[name], want) and new_master[name].dtype == torch.float32  # exact fp32 sum, never a bf16 round trip
        assert torch.equal(new_base[name], want.to(torch.bfloat16)) and new_base[name].dtype == weight.dtype
    ids = torch.tensor([[1, 5, 6, 7, 8, 9]])
    with torch.no_grad():  # the folded base is the model that training and scoring saw: base plus LoRA
        folded_logits = train_job.load_head(str(cache / artifact.base_hash), "float32")(input_ids=ids).logits
        stacked = train_job.load_head(str(head_dir), "float32")
        layers = train_job.add_lora(stacked, Recipe(), 7)
        lora = load_file(str(tmp / "lora" / "adapter_model.safetensors"))
        for path, layer in layers.items():
            layer.lora_A.data, layer.lora_B.data = lora[f"base_model.model.{path}.lora_A.weight"], lora[f"base_model.model.{path}.lora_B.weight"]
        plain = train_job.load_head(str(head_dir), "float32")(input_ids=ids).logits
        assert float((folded_logits - stacked(input_ids=ids).logits).abs().max()) < 0.2 * float((folded_logits - plain).abs().max())


@pytest.mark.parametrize("damage", ["missing part", "bf16 master", "other master", "foreign lora", "wrong rank", "no lora"])
def test_fold_fails_closed(folded, damage):
    store, config, head, params, tmp = folded
    master, lora = layout.public_master_prefix(head.base_fp32_master_digest), layout.public_lora_prefix(head.loras[0].lora_hash)
    error = VerifyError
    if damage == "missing part":
        store.delete(store.list(master)[0])
    elif damage == "other master":
        store.put(store.list(master)[0], b"not the master")
    elif damage == "no lora":
        head, error = HeadVersion.make(parent_version_id=None, base_hash=head.base_hash, base_fp32_master_digest=head.base_fp32_master_digest, loras=(), round_id="g", applied_at=1), ContractError
    else:  # a file set that hashes right but must not be folded
        work = tmp / "bad"
        shutil.copytree(tmp / ("master" if damage == "bf16 master" else "lora"), work)
        target = sorted(work.glob("*.safetensors"))[0]
        tensors = load_file(str(target))
        if damage == "bf16 master":
            tensors = {name: t.to(torch.bfloat16) for name, t in tensors.items()}
        elif damage == "foreign lora":
            tensors = {key.replace("layers.0.", "layers.9."): t for key, t in tensors.items()}
        save_file(tensors, str(target))
        ref = head.loras[0] if damage == "bf16 master" else LoRARef(manifest_digest(hash_tree(work)), "r000001", "hk7", 7, head.loras[0].batch_sha256, head.loras[0].recipe_id, 8 if damage == "wrong rank" else 16)
        digest = manifest_digest(hash_tree(work)) if damage == "bf16 master" else head.base_fp32_master_digest
        upload_tree(store, work, layout.public_master_prefix(digest) if damage == "bf16 master" else layout.public_lora_prefix(ref.lora_hash))
        head = HeadVersion.make(parent_version_id=None, base_hash=head.base_hash, base_fp32_master_digest=digest, loras=(ref,), round_id="r000001", applied_at=2)
    with pytest.raises(error):
        LoraFolder(store, config).fold(head, params, "pending/folds/v1")
    assert store.list("pending") == []  # nothing of a failed fold is uploaded


def test_genesis_command_checks_the_base_hash_then_prints_the_master_digest(head_dir, tmp_path, monkeypatch, capsys):
    asked = []

    def download(repo, revision, local_dir, allow_patterns):
        asked.append((repo, revision, allow_patterns))
        shutil.copytree(head_dir, local_dir)
        (Path(local_dir) / ".cache").mkdir()

    monkeypatch.setattr(genesis, "snapshot_download", download)
    with pytest.raises(VerifyError):  # the default pin is the launch model: other files are refused, and no master is written
        genesis.main(["--out", str(tmp_path / "wrong")])
    assert asked[0][:2] == (genesis.GENESIS_MODEL_REPO, genesis.GENESIS_MODEL_REVISION) and "*.safetensors" in asked[0][2]
    assert not (tmp_path / "wrong" / "master").exists()
    base_hash = manifest_digest(hash_tree(head_dir))
    genesis.main(["--out", str(tmp_path / "out"), "--base-hash", base_hash])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    master = tmp_path / "out" / "master"
    assert printed["base_hash"] == base_hash and printed["master_digest"] == manifest_digest(hash_tree(master))
    tensors = load_file(str(master / "model.part00.safetensors"))
    old = load_file(str(head_dir / "model.safetensors"))
    assert {t.dtype for t in tensors.values()} == {torch.float32} and all(torch.equal(tensors[name], old[name].float()) for name in old)
