"""CPU/tensor tests of affine-batch-v4 staged GRPO and verified SFT."""
from __future__ import annotations

import json
import string
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.modules["fla"] = None  # the tiny model takes transformers' plain torch path

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sn120_v2.gpu import staged, train_job  # noqa: E402
from sn120_v2.gpu.resident import hash_tree  # noqa: E402
from sn120_v2.params import Recipe  # noqa: E402

TEMPLATE = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
GRPO_ROWS = [
    {"prompt": "p a", "output": "abc", "reward": 1.0, "advantage": 0.5, "kind": "grpo"},
    {"prompt": "p a", "output": "ba", "reward": 0.0, "advantage": -0.5, "kind": "grpo"},
]
SFT_ROWS = [{"prompt": "p b", "output": "ccdd", "reward": 1.0, "advantage": 0.0, "kind": "sft"}]


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def head_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("staged-head")
    text = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                num_attention_heads=2, num_key_value_heads=2, head_dim=16)
    text.update(linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
                linear_num_value_heads=4, max_position_embeddings=256, eos_token_id=2, dtype="float32")
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32)
    config = transformers.Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=60,
                                        video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    torch.manual_seed(0)
    transformers.Qwen3_5ForConditionalGeneration(config).save_pretrained(path)
    words = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "Ġ", "Ċ", *string.ascii_lowercase]
    special = [dict(id=i, content=words[i], single_word=False, lstrip=False, rstrip=False,
                    normalized=False, special=True) for i in range(3)]
    model = dict(type="BPE", dropout=None, unk_token=None, continuing_subword_prefix="",
                 end_of_word_suffix="", fuse_unk=False, byte_fallback=False,
                 vocab={word: i for i, word in enumerate(words)}, merges=[])
    byte = dict(type="ByteLevel", add_prefix_space=False, trim_offsets=True, use_regex=True)
    (path / "tokenizer.json").write_text(json.dumps(dict(version="1.0", truncation=None, padding=None,
        added_tokens=special, normalizer=None, pre_tokenizer=byte, post_processor=None,
        decoder=byte, model=model)))
    (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|im_end|>"}))
    (path / "chat_template.jinja").write_text(TEMPLATE)
    return path


def recipe(**changes) -> Recipe:
    return Recipe(**({"rank": 2, "alpha": 4.0, "steps": 1, "lora_lr": 1e-4} | changes))


def job(head: Path, devices=1, **changes) -> dict:
    top = recipe()
    stages = [
        {"recipe": recipe().to_dict(), "rows": json.loads(json.dumps(GRPO_ROWS))},
        {"recipe": recipe(sft_weight=0.5).to_dict(), "rows": json.loads(json.dumps(SFT_ROWS))},
    ]
    return {"base_dir": str(head), "base_hash": "b" * 64, "dtype": "float32",
            "recipe": top.to_dict(), "seed": 7, "stages": stages} | changes


def test_exact_sft_loss_includes_every_output_and_end_token():
    recipe_ = recipe(sft_weight=0.25)
    output_head = torch.nn.Linear(3, 5, bias=False)
    replica = SimpleNamespace(device="cpu", recipe=recipe_, model=SimpleNamespace(
        get_output_embeddings=lambda: output_head))
    output_head.weight.data.copy_(torch.tensor([
        [0.1, -0.2, 0.3], [-0.4, 0.5, -0.6], [0.7, -0.8, 0.9],
        [0.2, 0.1, -0.1], [-0.3, 0.4, 0.5],
    ]))
    states = torch.tensor([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]])
    replica.hidden = lambda ids: states[None] * torch.ones((), requires_grad=True)
    row = {"prompt_ids": [1], "tokens": [2]}  # this target stands for the appended end token
    loss, count = staged.sft_loss(replica, row, 4)
    expected = -(replica.model.get_output_embeddings()(states[:1]).float().log_softmax(-1)
                 .gather(-1, torch.tensor([[2]])).mean()) * 0.25 / 4
    assert count == 1 and torch.equal(loss, expected.detach())
    # A separate two-token call proves token-mean rather than sum normalization.
    row["tokens"] = [2, 3]
    replica.hidden = lambda ids: torch.cat([states[None], torch.zeros(1, 1, 3)], dim=1) * torch.ones((), requires_grad=True)
    loss, count = staged.sft_loss(replica, row, 4)
    expected = -(replica.model.get_output_embeddings()(states).float().log_softmax(-1)
                 .gather(-1, torch.tensor([[2], [3]])).mean()) * 0.25 / 4
    assert count == 2 and torch.allclose(loss, expected.detach(), rtol=0, atol=1e-7)


def test_sft_unequal_chunks_preserve_token_mean_loss_and_gradient(monkeypatch):
    monkeypatch.setattr(staged, "LOGIT_CHUNK", 2)
    head = torch.nn.Linear(3, 5, bias=False)
    head.weight.data.copy_(torch.tensor([
        [0.1, -0.2, 0.3], [-0.4, 0.5, -0.6], [0.7, -0.8, 0.9],
        [0.2, 0.1, -0.1], [-0.3, 0.4, 0.5],
    ]))
    states = torch.tensor([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1], [0.6, 0.2, -0.3], [-0.2, 0.7, 0.4], [0.3, -0.3, 0.8]])
    scale = torch.ones((), requires_grad=True)
    replica = SimpleNamespace(device="cpu", recipe=recipe(sft_weight=0.25), model=SimpleNamespace(get_output_embeddings=lambda: head))
    replica.hidden = lambda ids: torch.cat([states[None], torch.zeros(1, 1, 3)], dim=1) * scale
    targets = torch.tensor([0, 1, 2, 3, 4])
    loss, count = staged.sft_loss(replica, {"prompt_ids": [1], "tokens": targets.tolist()}, 4)
    reference = head(states).float().log_softmax(-1).gather(-1, targets[:, None])
    expected = -reference.mean() * 0.25 / 4
    expected_gradient, = torch.autograd.grad(-reference.sum() * 0.25 / (len(targets) * 4), head.weight, retain_graph=True)
    assert count == 5 and torch.allclose(loss, expected.detach(), rtol=0, atol=1e-7)
    assert torch.allclose(head.weight.grad, expected_gradient, rtol=0, atol=1e-7)
    assert scale.grad is not None and torch.isfinite(scale.grad)


def test_stages_fresh_adam_rescore_and_deterministic_order(head_dir, tmp_path, monkeypatch):
    seen = {"adam": [], "old": [], "order": [], "seed": []}

    class Adam(torch.optim.Adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.updates = 0
            seen["adam"].append(self)

        def step(self):
            self.updates += 1
            return super().step()

    old, cut = staged.score_old, staged.minibatches_of

    def watched_old(replica, rows):
        seen["old"].append((len(seen["adam"]), len(seen["adam"]) and seen["adam"][-1].updates, len(rows)))
        return old(replica, rows)

    def watched_cut(rows, minibatches, order):
        seen["order"].append(order)
        return cut(rows, minibatches, order)

    original_manual_seed = staged.torch.manual_seed
    monkeypatch.setattr(torch.optim, "Adam", Adam)
    monkeypatch.setattr(staged, "score_old", watched_old)
    monkeypatch.setattr(staged, "minibatches_of", watched_cut)
    monkeypatch.setattr(staged.torch, "manual_seed", lambda seed: seen["seed"].append(seed) or original_manual_seed(seed))
    made = job(head_dir)
    result = train_job.train(made, tmp_path / "a", ["cpu"])
    report = json.loads((tmp_path / "a" / "train_report.json").read_text())
    assert seen["seed"] == [7, staged.stage_seed(7, 0), staged.stage_seed(7, 1)]
    assert [optimizer.updates for optimizer in seen["adam"]] == [1, 1]
    assert seen["old"] == [(1, 0, 2), (2, 0, 0)]
    assert len({id(order) for order in seen["order"]}) == 2
    assert [row["seed"] for row in report["stages"]] == [staged.stage_seed(7, 0), staged.stage_seed(7, 1)]
    assert [row["updates"] for row in report["stages"]] == [1, 1]
    assert [row["seed"] for row in report["stages"]] == [staged.stage_seed(7, 0), staged.stage_seed(7, 1)]
    assert len(report["losses"]) == 2 and len(report["grad_norms"]) == 2
    assert report["recipe_values"]["stages"] == [row["recipe"] for row in report["stages"]]
    assert [row["recipe"]["sft_weight"] for row in report["stages"]] == [1.0, 0.5]
    assert result["updates"] == 2 and result["rows"] == 3
    assert "rows" not in made["stages"][0] and "rows" not in made["stages"][1]
    first = hash_tree(tmp_path / "a")
    train_job.train(job(head_dir), tmp_path / "b", ["cpu"])
    assert hash_tree(tmp_path / "b") == first


def test_two_replicas_match_one_for_grpo_then_sft(head_dir, tmp_path):
    one = train_job.train(job(head_dir), tmp_path / "one", ["cpu"])
    two = train_job.train(job(head_dir), tmp_path / "two", ["cpu", "cpu"])
    assert one["loss"] == pytest.approx(two["loss"], abs=1e-7)
    assert one["step_norm"] == pytest.approx(two["step_norm"], abs=1e-7)


def test_stage_reports_last_pass_tokens_and_restores_later_replicas(head_dir, tmp_path, monkeypatch):
    made = job(head_dir)
    later_rows = json.loads(json.dumps(GRPO_ROWS))
    later_rows.append(dict(later_rows[0], output="cab", advantage=0.25))
    made["stages"] = [
        {"recipe": recipe(sft_weight=1.0).to_dict(), "rows": json.loads(json.dumps(SFT_ROWS))},
        {"recipe": recipe(steps=3, minibatches=2).to_dict(), "rows": later_rows},
    ]
    seen = []
    original = staged.StageReplica.mixed_rows
    monkeypatch.setattr(staged.StageReplica, "mixed_rows",
                        lambda replica, rows, n: seen.append((replica.device, len(rows), n)) or original(replica, rows, n))
    result = train_job.train(made, tmp_path / "restored", ["cpu", "cpu"])
    report = json.loads((tmp_path / "restored" / "train_report.json").read_text())
    assert result["n_gpus"] == 2
    assert sum(row["updates"] for row in report["stages"]) == 7
    assert report["stages"][-1]["trained_tokens"] == report["trained_tokens"]
    assert any(row[2] == 2 for row in seen), seen
    assert [row["recipe"]["steps"] for row in report["stages"]] == [1, 3]
    assert report["recipe_values"]["top_level_shape"]["rank"] == 2


@pytest.mark.parametrize("minibatches", [1, 2])
def test_resource_metrics_leave_legacy_lora_bytes_unchanged(head_dir, tmp_path, monkeypatch, minibatches):
    made = {"base_dir": str(head_dir), "base_hash": "b" * 64, "dtype": "float32", "recipe": recipe(minibatches=minibatches).to_dict(), "seed": 7, "rows": GRPO_ROWS}
    train_job.train(made, tmp_path / f"plain-{minibatches}", ["cpu"])
    before = hash_tree(tmp_path / f"plain-{minibatches}")
    monkeypatch.setattr(train_job, "resource_metrics", lambda: {"cuda_max_memory_allocated": 1, "cuda_max_memory_reserved": 2, "process_max_rss_bytes": 3})
    report_dir = tmp_path / f"legacy-{minibatches}"
    (report_dir / "lora").mkdir(parents=True)
    (report_dir / "job.json").write_text(json.dumps(made))
    original_train = train_job.train
    captured = {}
    def cpu_train(job, out, devices=None, **kwargs):
        captured.update(job)
        return original_train(job, out, ["cpu"], **kwargs)
    monkeypatch.setattr(train_job, "train", cpu_train)
    train_job.main(report_dir)
    assert hash_tree(report_dir / "lora") == before
    report = json.loads((report_dir / "report.json").read_text())
    assert (report["cuda_max_memory_allocated"], report["cuda_max_memory_reserved"], report["process_max_rss_bytes"]) == (1, 2, 3)
    assert not (report_dir / "job.json").exists()
    assert all("prompt" not in row and "output" not in row for row in captured["rows"])
    (report_dir / "job.json").write_text(json.dumps({"recipe": recipe(minibatches=minibatches).to_dict(), "rows": []}))
    with pytest.raises(Exception):
        train_job.main(report_dir)
    assert not (report_dir / "job.json").exists()


def test_resource_metrics_are_not_in_hashed_lora_files(head_dir, tmp_path, monkeypatch):
    train_job.train(job(head_dir), tmp_path / "lora", ["cpu"])
    before = hash_tree(tmp_path / "lora")
    monkeypatch.setattr(train_job, "resource_metrics", lambda: {"cuda_max_memory_allocated": 1,
        "cuda_max_memory_reserved": 2, "process_max_rss_bytes": 3})
    report_dir = tmp_path / "job"
    (report_dir / "lora").mkdir(parents=True)
    (report_dir / "job.json").write_text(json.dumps(job(head_dir)))
    monkeypatch.setattr(train_job, "train", lambda job, out, devices=None, **kwargs: staged.run_stages(job, out, ["cpu"]))
    train_job.main(report_dir)
    report = json.loads((report_dir / "report.json").read_text())
    assert (report["cuda_max_memory_allocated"], report["process_max_rss_bytes"]) == (1, 3)
    assert hash_tree(tmp_path / "lora") == before
    assert not (report_dir / "job.json").exists()


def test_job_json_is_removed_on_rejection(tmp_path):
    directory = tmp_path / "job"
    directory.mkdir()
    made = {"recipe": recipe().to_dict(), "rows": []}
    (directory / "job.json").write_text(json.dumps(made))
    with pytest.raises(Exception):
        train_job.main(directory)
    assert not (directory / "job.json").exists()
