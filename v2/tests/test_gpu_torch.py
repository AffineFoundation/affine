"""The GRPO step, the fold and the genesis command on a tiny model of the head's own architecture (CPU)."""

from __future__ import annotations

import json
import math
import shutil
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

# The tiny model of these tests runs on the CPU. The fast kernels of
# flash-linear-attention run only on a GPU. With this entry, transformers finds
# no such package in this process and takes its plain torch path, on any machine.
sys.modules["fla"] = None  # type: ignore[assignment]

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
    assert report["step_norm_before_cap"] == report["step_norm"] == result["step_norm"]  # no cap: one norm
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


def test_the_warm_up_row_changes_no_byte_of_what_training_makes(head_dir, tmp_path, monkeypatch):
    # float32: on a busy CPU a bf16 run is not always repeatable, with or without the warm-up row, and this test compares bytes.
    def tree(name: str, n: int) -> dict:
        run(head_dir, tmp_path / name, devices=("cpu",) * n, dtype="float32")
        return hash_tree(tmp_path / name)  # the LoRA, its config and the train report

    warm = {n: tree(f"warm{n}", n) for n in (1, 2)}
    monkeypatch.setattr(train_job.Replica, "warm_up", lambda self, ids: None)  # the job as it was before the warm-up row
    assert {n: tree(f"cold{n}", n) for n in (1, 2)} == warm


def test_gradient_checkpointing_changes_no_byte_of_what_training_makes(head_dir, tmp_path, monkeypatch):
    # float32, as the test above: this test compares bytes.
    cases = {"one": {}, "two": {"n": 2}, "mlp": {"n": 2, "steps": 3, "target_modules": ("gate_proj", "up_proj", "down_proj")}}
    cases["capped"] = {"steps": 4, "rank": 64, "alpha": 128.0, "global_norm_cap": 1e-4, "train_max_output_tokens": 3}

    def trees(name: str) -> dict:
        for case, recipe in cases.items():
            run(head_dir, tmp_path / name / case, devices=("cpu",) * recipe.get("n", 1), dtype="float32", **{k: v for k, v in recipe.items() if k != "n"})
        return {case: hash_tree(tmp_path / name / case) for case in cases}  # the LoRA, its config and the train report

    replayed = trees("replayed")
    monkeypatch.setattr(train_job, "keep_one_graph", lambda model, layers: 1)  # the job with the graph of every layer kept
    assert trees("kept") == replayed and len({json.dumps(tree, sort_keys=True) for tree in replayed.values()}) == len(cases)


def test_a_row_keeps_the_graph_of_no_decoder_layer_and_each_layer_runs_a_second_time_in_the_backward(head_dir, monkeypatch):
    def graph(tensor: torch.Tensor) -> list[str]:
        seen, todo = set(), [tensor.grad_fn]
        while todo:
            node = todo.pop()
            if node is not None and node not in seen:
                seen.add(node)
                todo.extend(parent for parent, _ in node.next_functions)
        return [type(node).__name__ for node in seen if type(node).__name__ != "AccumulateGrad"]

    ids = [1, 5, 6, 7, 8, 9, 10, 11]
    replica = train_job.Replica(str(head_dir), "cpu", "float32", Recipe(), 7)
    blocks, runs = replica.model.base_model.language_model.layers, []
    for index, block in enumerate(blocks):
        block.mlp.register_forward_pre_hook(lambda module, args, index=index: runs.append((index, torch.is_grad_enabled())))
    hidden = replica.hidden(ids)
    assert runs == [(index, False) for index in range(4)]  # the first run of each layer: no graph
    nodes = graph(hidden)
    assert nodes.count("ReplayBackward") == 4 and len(nodes) < 12  # one node per layer, and the last norm of the head
    hidden.float().square().sum().backward()
    assert runs[4:] == [(index, True) for index in (3, 2, 1, 0)]  # the second run: one layer at a time, the last layer first
    assert all(weight.grad is not None and bool(torch.isfinite(weight.grad).all()) for weight in replica.weights)
    assert any(bool(layer.lora_B.grad.any()) for path, layer in replica.layers.items() if ".layers.0." in path)  # the gradient reaches the first layer
    with pytest.raises(InfraError, match="without a cache"):  # a cache would be written two times
        replica.model.base_model(input_ids=torch.tensor([ids]), use_cache=True)
    monkeypatch.setattr(train_job, "keep_one_graph", lambda model, layers: 1)
    assert len(graph(train_job.Replica(str(head_dir), "cpu", "float32", Recipe(), 7).hidden(ids))) > 40 * len(nodes)


def test_the_warm_up_row_runs_once_per_replica_before_any_row_and_leaves_nothing_behind(head_dir, tmp_path, monkeypatch):
    warm_up, train_rows, events = train_job.Replica.warm_up, train_job.Replica.train_rows, []

    def watched(self, ids):
        weights, state = [weight.detach().clone() for weight in self.weights], torch.get_rng_state()
        warm_up(self, ids)
        events.append(("warm-up", id(self), len(ids)))
        assert all(weight.grad is None for weight in self.weights) and all(torch.equal(a, b) for a, b in zip(weights, self.weights))
        assert torch.equal(state, torch.get_rng_state())  # no draw from the random source of the process

    def rows(self, *args):
        events.append(("rows", id(self), 0))
        return train_rows(self, *args)

    monkeypatch.setattr(train_job.Replica, "warm_up", watched)
    monkeypatch.setattr(train_job.Replica, "train_rows", rows)
    monkeypatch.setattr(train_job, "WARM_UP_TOKENS", 5)
    run(head_dir, tmp_path / "out", devices=("cpu", "cpu"))
    assert [kind for kind, _, _ in events] == ["warm-up", "warm-up", "rows", "rows", "rows", "rows"]  # 2 replicas, then 2 steps
    assert len({replica for _, replica, _ in events[:2]}) == 2 and {size for _, _, size in events[:2]} == {5}  # the longest row, cut


def test_torch_leaves_a_part_of_each_gpu_to_the_driver_and_the_cpu_is_not_asked(monkeypatch):
    gib = 2**30
    assert train_job.torch_share(288 * gib) == 1 - 8 / 288 and train_job.torch_share(128 * gib) == 15 / 16
    assert train_job.torch_share(24 * gib) == 15 / 16  # a small card: a sixteenth, not 8 GiB
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=160 * gib))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda share, device: calls.append((share, device)))
    train_job.leave_room_for_the_driver("cpu")
    train_job.leave_room_for_the_driver("cuda:3")
    assert calls == [(0.95, "cuda:3")]


def test_one_thread_at_a_time_is_in_the_kernel_tuner():
    class Tuner:  # Triton's tuner in small: the arguments live in the object for the time of the call
        nargs, inside, most = None, 0, 0

        def run(self, value):
            self.nargs = {"x": value}
            Tuner.inside += 1
            Tuner.most = max(Tuner.most, Tuner.inside)
            time.sleep(0.002)
            out = {**self.nargs}["x"]  # without the lock: a TypeError when another call has ended, or the other call's value
            Tuner.inside -= 1
            self.nargs = None
            return out

    plain = Tuner.run
    train_job.one_thread_in_the_tuner(Tuner)
    locked = Tuner.run
    train_job.one_thread_in_the_tuner(Tuner)  # a second call adds no second lock
    assert Tuner.run is locked is not plain and locked.__wrapped__ is plain
    tuner = Tuner()
    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(tuner.run, range(64))) == list(range(64))
    assert Tuner.most == 1
    train_job.one_thread_in_the_tuner(None)  # no Triton on this machine: nothing to lock


def test_miner_values_change_the_step_and_the_norm_cap_scales_the_whole_update(head_dir, tmp_path):
    base, _, _ = run(head_dir, tmp_path / "base")
    hot, _, _ = run(head_dir, tmp_path / "hot", sample_temperature=1.5)
    small, _, report = run(head_dir, tmp_path / "small", rank=8, alpha=8.0, steps=1, target_modules=("q_proj",))
    assert hot["loss"] != base["loss"] and small["step_norm"] < base["step_norm"] and len(report["losses"]) == 1
    capped, lora, report = run(head_dir, tmp_path / "cap", global_norm_cap=base["step_norm"] / 4)
    assert report["step_norm_before_cap"] == pytest.approx(base["step_norm"]) and report["step_norm"] == capped["step_norm"]  # both norms are stated
    assert capped["step_norm"] == pytest.approx(base["step_norm"] / 4) == pytest.approx(update_norm(lora, 2.0), rel=1e-4)


def test_both_train_caps_cut_a_row_and_a_row_with_no_room_is_dropped_and_counted(head_dir, tmp_path):
    _, _, report = run(head_dir, tmp_path / "out", train_max_output_tokens=3)
    assert (report["truncated_rows"], report["dropped_rows"], report["rows"]) == (1, 0, 3)  # "ccdd" is 4 tokens: cut to 3, not rejected
    tokenizer = transformers.AutoTokenizer.from_pretrained(head_dir)
    chat = tokenizer.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False)
    short = len(tokenizer.encode(chat, add_special_tokens=False))  # the prompt "ab c" is 3 tokens longer
    result, _, report = run(head_dir, tmp_path / "context", train_max_output_tokens=3, train_max_context_tokens=short + 2)
    # The context cap leaves 2 output tokens after "x", and none after "ab c": that prompt's two rows are dropped.
    assert (report["truncated_rows"], report["dropped_rows"], report["rows"], result["status"]) == (1, 2, 1, "ok")
    with pytest.raises(BatchRejected):  # no row keeps an output token
        run(head_dir, tmp_path / "none", train_max_output_tokens=3, train_max_context_tokens=short)


def test_the_job_runs_at_both_ends_of_every_number_a_miner_may_set(head_dir, tmp_path):
    bounds = [bound for bound in Recipe().miner_bounds if not bound.choices]
    for end in ("low", "high"):
        values = {bound.key: int(getattr(bound, end)) if bound.integer else getattr(bound, end) for bound in bounds}
        result, lora, report = run(head_dir, tmp_path / end, **values)
        assert result["status"] == "ok" and math.isfinite(result["step_norm"]) and len(report["losses"]) == values["steps"]
        assert {key: report["recipe_values"][key] for key in values} == values
        assert all(a.shape[0] == values["rank"] for key, a in lora.items() if "lora_A" in key)


def test_a_batch_without_gradient_is_rejected_and_a_wrong_layer_name_stops_the_job(head_dir, tmp_path):
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
    kept = torch.tensor([4.0, 2.0]) / 2.0  # logits / T; of the top 3, the first two hold 0.9 of the probability
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
    master_digest = upcast(head_dir, tmp_path / "master", part_bytes=160_000)
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
    assert len(list((tmp / "master").iterdir())) > 1 and all(f.stat().st_size < 200_000 for f in (tmp / "master").iterdir())  # parts stay under one upload
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


class BadStore(LocalStore):
    """A store that returns other bytes for one weight file."""

    def sha256(self, key: str) -> str:
        return "0" * 64 if key.endswith("model.part00.safetensors") else super().sha256(key)


def test_genesis_command_checks_the_base_hash_writes_the_manifests_and_stages_checked_files(head_dir, tmp_path, monkeypatch, capsys):
    asked = []

    def download(repo, revision, local_dir, allow_patterns):
        asked.append((repo, revision, allow_patterns))
        shutil.copytree(head_dir, local_dir)
        (Path(local_dir) / ".cache").mkdir()

    monkeypatch.setattr(genesis, "snapshot_download", download)
    with pytest.raises(VerifyError):  # the default pin is the launch model: other files are refused, and no master is written
        genesis.main(["--out", str(tmp_path / "wrong")])
    assert asked[0][:2] == (genesis.GENESIS_MODEL_REPO, genesis.GENESIS_MODEL_REVISION) and "*.safetensors" in asked[0][2]
    assert not (tmp_path / "wrong" / "master").exists() and not (tmp_path / "wrong" / "genesis_files.json").exists()
    base_hash, out = manifest_digest(hash_tree(head_dir)), tmp_path / "out"
    genesis.main(["--out", str(out), "--base-hash", base_hash])  # no --upload: no store is opened
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["base_hash"] == base_hash and printed["master_digest"] == manifest_digest(hash_tree(out / "master")) and printed["uploaded_files"] == 0
    files = json.loads((out / "genesis_files.json").read_text())
    assert files == {"base_files": hash_tree(head_dir), "master_files": hash_tree(out / "master")} and printed["genesis_files"] == str(out / "genesis_files.json")
    tensors = load_file(str(out / "master" / "model.part00.safetensors"))
    old = load_file(str(head_dir / "model.safetensors"))
    assert {t.dtype for t in tensors.values()} == {torch.float32} and all(torch.equal(tensors[name], old[name].float()) for name in old)

    store = LocalStore(tmp_path / "store")
    genesis.main(["--out", str(out), "--base-hash", base_hash, "--upload"], open_store=lambda: store)
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    base_keys = {layout.public_base_file(base_hash, name): sha for name, sha in files["base_files"].items()}
    master_keys = {layout.public_master_file(printed["master_digest"], name): sha for name, sha in files["master_files"].items()}
    assert printed["uploaded_files"] == len(base_keys) + len(master_keys) and all(store.sha256(key) == sha for key, sha in (base_keys | master_keys).items())
    # A box finds the staged files under the prefixes it reads.
    assert store.list(layout.public_base_prefix(base_hash)) == sorted(base_keys) and store.list(layout.public_master_prefix(printed["master_digest"])) == sorted(master_keys)
    bad = BadStore(tmp_path / "bad")
    with pytest.raises(VerifyError, match="differs"):
        genesis.main(["--out", str(out), "--base-hash", base_hash, "--upload"], open_store=lambda: bad)
    assert not bad.exists(layout.public_master_file(printed["master_digest"], "model.part00.safetensors"))  # the wrong object is removed
