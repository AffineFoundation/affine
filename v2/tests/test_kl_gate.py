"""CPU checks for measured model-change KL and its sealed, public decisions."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sn120_v2.errors import ContractError
from sn120_v2.gpu.kl import measure_rows, selected_rollouts, sequence_kl, summarize
from sn120_v2.judging import check_result, decide
from sn120_v2.params import Params, load_params
from sn120_v2.public import round_report
from sn120_v2.publisher import contract, kl_rule
from sn120_v2.types import (
    LOST,
    NO_DECISION,
    NOT_SCORED,
    NULL_UPDATE,
    SCORED,
    BoxResult,
    EntryDecision,
    EntryResult,
    PASS,
)

LIVE = Path(__file__).parents[1] / "ops" / "train64k_params_t3600_prune_open.json"
PRODUCTION = Path(__file__).parents[1] / "ops" / "train64k_params_t3600_prune_open_kl8k.json"
FULL_LENGTH = Path(__file__).parents[1] / "ops" / "train64k_params_t3600_prune_open_kl.json"
OPEN_PIN = "073ef680cf338de5e085100704e60f80ba96657d6d4e19d12995e4cb971ef6f8"
METRICS = dict(kl_seq=0.5, kl_seq_se=0.02, kl_tok=0.01, kl_k=8, kl_tokens=50)


def test_identical_logits_have_zero_kl_and_known_case_is_exact():
    logits = torch.tensor([[math.log(0.5), math.log(0.25), math.log(0.25)]])
    other = torch.tensor([[math.log(0.25), math.log(0.25), math.log(0.5)]])
    from sn120_v2.gpu.kl import kl_from_log_probs

    assert float(kl_from_log_probs(logits, logits)) == pytest.approx(0.0, abs=1e-7)
    assert float(kl_from_log_probs(logits, other)) == pytest.approx(0.25 * math.log(2))


class Tiny(nn.Module):
    def __init__(self, vocab=7, hidden=4, layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        trunk = nn.Sequential(*[nn.Linear(hidden, hidden) for _ in range(layers)])
        self.base_model = SimpleNamespace()
        self.output = nn.Linear(hidden, vocab)
        self.base_model = _Trunk(trunk)
        self.vocab = vocab

    def get_output_embeddings(self):
        return self.output


class _Trunk(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, input_ids, use_cache=False):
        # Discrete ids become features, then follow a small teacher-forced trunk.
        values = input_ids % self.layers[0].in_features
        values = torch.nn.functional.one_hot(values, self.layers[0].in_features).float()
        return SimpleNamespace(last_hidden_state=self.layers(values))


def test_chunked_full_vocab_forward_equals_one_shot_on_a_tiny_random_model():
    new, head = Tiny(seed=1), Tiny(seed=2)
    row = {"prompt_ids": [0, 1, 2], "completion_ids": [3, 4, 5, 6, 1]}
    whole, tokens = sequence_kl(new, head, row, chunk=100)
    for chunk in (1, 2, 3, 5):
        chunked, count = sequence_kl(new, head, row, chunk=chunk)
        assert chunked == pytest.approx(whole, rel=1e-6, abs=1e-7)
        assert count == tokens == 5


def test_tiny_model_measurement_and_null_calibration_use_separate_forwards(monkeypatch):
    new, head = Tiny(seed=3), Tiny(seed=3)
    with torch.no_grad():
        new.output.weight[0].add_(1.0)
    rows = [{"prompt_ids": [0, 1], "completion_ids": [2, 3, 4]}]
    metrics = measure_rows(new, head, rows)
    assert metrics["kl_seq"] > 0 and metrics["kl_k"] == 1 and metrics["kl_tokens"] == 3
    assert metrics["kl_tok"] == pytest.approx(metrics["kl_seq"] / 3)
    calls = []
    from sn120_v2.gpu import kl

    real = kl._completion_hidden

    def observed(model, *args, **kwargs):
        calls.append(model)
        return real(model, *args, **kwargs)

    monkeypatch.setattr(kl, "_completion_hidden", observed)
    assert measure_rows(head, head, rows)["kl_seq"] == pytest.approx(0.0, abs=1e-7)
    assert calls == [head, head]


def test_deterministic_subsample_uses_only_scored_nonerror_sample_zero(tmp_path):
    capture = tmp_path / "model.jsonl"
    outcomes = ("PPCE-", "PFPPP")
    rows = []
    for sample in (0, 1):
        for pos, outcome in enumerate(outcomes[0]):
            for suffix in ("", "-other"):
                rows.append(dict(sample_index=sample, task_pos=pos, outcome=outcome,
                                 prompt=[{"role": "user", "content": f"p{pos}"}], completion_text=f"c{pos}{suffix}"))
    rows.append(dict(sample_index=0, task_pos=3, outcome="E", prompt=[], completion_text="error"))
    capture.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    chosen = selected_rollouts(capture, outcomes, "round-seed", "lora-hash", 8)
    # The repeated final task_pos is the latest capture; samples other than zero are excluded.
    assert [row["task_pos"] for row in chosen] == [0, 1, 2]
    assert all(row["completion_text"].endswith("-other") for row in chosen)
    assert chosen == selected_rollouts(capture, outcomes, "round-seed", "lora-hash", 8)
    assert [row["task_pos"] for row in selected_rollouts(capture, outcomes, "changed", "lora-hash", 1)] != chosen[:1]
    assert selected_rollouts(capture, outcomes, "round-seed", "lora-hash", 1) == random.Random(
        int.from_bytes(hashlib.sha256(b"round-seedlora-hashkl").digest(), "big")
    ).sample(chosen, 1)


def test_live_params_hash_unchanged_and_validation():
    live = load_params(LIVE)
    assert live.params_hash == OPEN_PIN and live.to_dict() == json.loads(LIVE.read_text())
    assert "min_kl_seq" not in live.to_dict() and "kl_rollouts" not in live.to_dict()
    assert "kl_max_tokens" not in live.to_dict() and "kl_time_limit_s" not in live.to_dict()
    assert (live.min_kl_seq, live.kl_rollouts, live.kl_max_tokens, live.kl_time_limit_s) == (0.0, 128, 0, 1200)
    for changes in ({"min_kl_seq": -1.0}, {"min_kl_seq": math.inf}, {"kl_rollouts": 7},
                    {"kl_max_tokens": -1}, {"kl_max_tokens": 1.5}, {"kl_time_limit_s": 0}):
        with pytest.raises(Exception):
            dataclasses.replace(live, **changes)
    with pytest.raises(ContractError):
        Params.from_dict(live.to_dict() | {"kl_rollouts": 1.5})


def test_production_params_exact_identity_and_effective_diff():
    old, new = load_params(LIVE), load_params(PRODUCTION)
    full = load_params(FULL_LENGTH)
    assert old.params_hash == OPEN_PIN
    assert new.params_hash == "dcab97d47d2f4b02ef7a14d5d8786a0f5a14885ddbf2634845435612fb9b4b1e"
    assert new.recipe_id == "b1f7cda18368657aebf33ee5097244ca41cccd6bcee6713459799680a6ef2d44"
    assert new.recipe.code_digest == "5e3aef5849ef6903c9fc1ab67f89eddbd2ea2cac9748621fc42b75224bf58419"
    assert new.recipe.image_digests == old.recipe.image_digests == (
        "9a4b16ff282b21baf44350e54c852ed59586b9efb6fe09e5cd2f1e288f4c108a",
    )
    assert (new.min_kl_seq, new.kl_rollouts, new.kl_max_tokens, new.kl_time_limit_s) == (0.000225, 128, 8192, 1200)
    assert dataclasses.replace(new, kl_max_tokens=0) == full
    # Compare effective values, including fields omitted at their defaults.
    assert dataclasses.asdict(old) | {"_legacy_fields": None} == dataclasses.asdict(new) | {
        "min_kl_seq": 0.0,
        "kl_max_tokens": 0,
        "recipe": dataclasses.asdict(old.recipe),
        "_legacy_fields": None,
    }
    assert json.loads(FULL_LENGTH.read_text()) == old.to_dict() | {
        "min_kl_seq": 0.000225,
        "kl_rollouts": 128,
        "recipe": old.to_dict()["recipe"] | {"code_digest": new.recipe.code_digest},
    }
    assert json.loads(PRODUCTION.read_text()) == json.loads(FULL_LENGTH.read_text()) | {"kl_max_tokens": 8192}


def gated(build, kl, floor=0.25):
    params = build.params(min_kl_seq=floor)
    ticket = build.ticket(n_entries=1, params=params)
    result = build.result(ticket, PASS * 8, {0: PASS * 8})
    entry = dataclasses.replace(result.entries[0], **({} if kl is None else METRICS | {"kl_seq": kl}))
    return ticket, dataclasses.replace(result, entries=(entry,))


def test_judging_below_above_missing_and_off(build):
    below = decide([gated(build, 0.1)])
    assert below.entries[0].status == NULL_UPDATE and below.entries[0].charged
    assert below.winner_queue_pos is None
    assert below.entries[0].reason == "measured KL 0.1 nats/seq is below min_kl_seq 0.25"
    above = decide([gated(build, 0.25)])
    assert above.entries[0].status == LOST and above.entries[0].charged
    missing = decide([gated(build, None)])
    assert missing.entries[0].status == NO_DECISION and not missing.entries[0].charged
    ticket, result = gated(build, None, floor=0.0)
    assert decide([(ticket, result)]).digest() == decide([(ticket, result)]).digest()
    assert decide([(ticket, result)]).entries[0].status == LOST
    with pytest.raises(ContractError, match="while min_kl_seq is off"):
        check_result(ticket, dataclasses.replace(result, entries=(dataclasses.replace(result.entries[0], **METRICS),)))


def test_sealed_result_round_trip_and_validation(build):
    ticket, result = gated(build, 0.5)
    result = dataclasses.replace(result, **{f"null_{key}": value for key, value in METRICS.items()})
    check_result(ticket, result)
    encoded = result.to_json()
    decoded = BoxResult.from_json(encoded)
    assert decoded.to_json() == encoded and decoded.entries[0].kl_seq == 0.5 and decoded.null_kl_k == 8
    assert all(name in decoded.entries[0].to_dict() for name in METRICS)
    assert "kl_seq" not in build.result(build.ticket(n_entries=1), PASS * 4, {0: PASS * 8}).entries[0].to_dict()
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ContractError, match="finite and nonnegative"):
            dataclasses.replace(decoded.entries[0], kl_seq=bad)
    with pytest.raises(ContractError, match="present together"):
        dataclasses.replace(decoded.entries[0], kl_seq_se=None)
    with pytest.raises(ContractError, match="KL samples exceed"):
        invalid = dataclasses.replace(decoded, entries=(dataclasses.replace(decoded.entries[0], kl_k=100, kl_tokens=500),))
        check_result(ticket, invalid)


def test_entry_decision_and_public_round_fields(build):
    ticket, result = gated(build, 0.5)
    result = dataclasses.replace(result, **{f"null_{key}": value for key, value in METRICS.items()})
    decision = decide([(ticket, result)])
    for name, value in METRICS.items():
        assert getattr(decision.entries[0], name) == value
    assert EntryDecision.from_dict(decision.entries[0].to_dict()) == decision.entries[0]
    from sn120_v2.state import Lease, RoundRecord

    lease = Lease(ticket.lease_id, ticket.box_id, ticket.issued_at, ticket.expires_at, ticket.digest(), ticket.entries)
    record = RoundRecord(ticket.round_id, ticket.head.version_id, ticket.params.params_hash, ticket.params.recipe_id,
                         ticket.draw.round_seed, ticket.draw.snapshot_id, len(ticket.draw.task_ids), ticket.issued_at, (lease,), decision=decision)
    report = round_report(record, [(ticket, result)])
    assert report["null_kl_seq"] == 0.5 and report["null_kl_k"] == 8
    assert report["entry_outcomes"][0]["kl_seq"] == 0.5
    assert report["entries"][0]["kl_tok"] == 0.01
    published = contract(ticket.params, "https://example.org")["duplicate_policy"]
    rule = "Your LoRA must change the model: the mean KL(new||head) over 128 of its own eval rollouts must be at least 0.25 nats per sequence, else the entry is a charged null_update."
    assert published["min_kl_seq_rule"] == rule == kl_rule(ticket.params)
    from sn120_v2.public import snapshot
    from sn120_v2.publisher import render
    from sn120_v2.state import State
    from sn120_v2.canonical import manifest_digest
    from sn120_v2.types import HeadManifest, HeadVersion

    base_files = {"model.bin": build.h("model")}
    head = HeadVersion.make(parent_version_id=None, base_hash=manifest_digest(base_files),
                            base_fp32_master_digest=build.h("master"), loras=(), round_id="genesis", applied_at=1000)
    manifest = HeadManifest(head=head, base_files=base_files, master_files={"master.bin": build.h("master")}, lora_files={})
    state = State(params=ticket.params, heads=[head], manifest=manifest)
    text = render(snapshot(state, ticket.params, 2000), "https://example.org")[1].decode()
    assert rule in " ".join(text.split())
    assert "kl_seq" in json.dumps(report["entries"])


def test_solo_seals_kl_and_one_failure_is_uncharged(build):
    from sn120_v2.gpu.solo import SoloRunner

    ticket = build.ticket(n_entries=2, params=build.params(min_kl_seq=0.25))
    reports = {seat.queue_pos: build.artifact(ticket.head, seat, ticket.params) for seat in ticket.entries}
    failed = SoloRunner._kl_reports(ticket, reports, {0: METRICS, 1: {"error": "CUDA out of memory"}}, (PASS * 8,), False)
    assert failed[1].status == NO_DECISION and "CUDA out of memory" in failed[1].reason
    outcome = ((PASS * 8,), {pos: (PASS * 8,) for pos in reports}, {}, False)
    result = SoloRunner._solo_result(SimpleNamespace(clock=lambda: 2000, image_digest=build.image, _tested={}),
                                     ticket, {**reports, **failed}, *outcome, 1000, 1100, METRICS, 30.0, {0: METRICS})
    assert result.entries[0].kl_seq == 0.5 and result.null_kl_seq == 0.5
    assert result.entries[1].status == NO_DECISION and result.phase_s["kl"] == 30
    assert result.phase_s["evaluation"] == 900  # 100 seconds of eval + 30 seconds of KL
    check_result(ticket, result)


def test_time_limit_before_first_look_stays_not_scored(build):
    from sn120_v2.gpu.solo import SoloRunner

    ticket = build.ticket(n_entries=1, params=build.params(min_kl_seq=0.25))
    reports = {0: build.artifact(ticket.head, ticket.entries[0], ticket.params)}
    failed = SoloRunner._kl_reports(ticket, reports, {}, ("P-------",), True)
    assert failed == {}
    outcome = (("P-------",), {0: ("P-------",)}, {}, True)
    runner = SimpleNamespace(clock=lambda: 2000, image_digest=build.image, _tested={})
    result = SoloRunner._solo_result(runner, ticket, reports, *outcome, 1000, 1100, {}, 0.0, {})
    assert result.entries[0].status == NOT_SCORED and not decide([(ticket, result)]).entries[0].charged


def test_worker_oom_does_not_stop_peer(tmp_path, monkeypatch):
    from sn120_v2.gpu.kl import run_worker

    job_dir = tmp_path / "worker"
    job_dir.mkdir()
    head_dir, candidate_dir = tmp_path / "head", tmp_path / "candidate"
    head_dir.mkdir()
    candidate_dir.mkdir()
    rows = [{"task_pos": 0, "prompt": [], "completion_text": "ok", "outcome": "P"}]
    (job_dir / "job.json").write_text(json.dumps({"head_dir": str(head_dir), "dtype": "bfloat16", "items": [
        {"key": "0", "shard": "0:0", "candidate_dir": str(candidate_dir), "rows": rows},
        {"key": "1", "shard": "1:0", "candidate_dir": str(candidate_dir), "rows": rows},
    ]}))
    from sn120_v2.gpu import kl
    import transformers

    monkeypatch.setattr(kl, "_load_model", lambda path, dtype: Tiny(seed=3 if path == head_dir else 4))
    monkeypatch.setattr(kl, "tokenize_rows", lambda tokenizer, rows, max_tokens=0: [{"prompt_ids": [0], "completion_ids": [1, 2]}])
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []
    real_measure = kl.sequence_kl

    def measure(new, head, row, device, chunk=2048):
        calls.append(new)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return real_measure(new, head, row, "cpu")

    monkeypatch.setattr(kl, "sequence_kl", measure)
    run_worker(job_dir)
    assert "CUDA out of memory" in json.loads((job_dir / "result-0:0.json").read_text())["error"]
    metrics = json.loads((job_dir / "result-1:0.json").read_text())["metrics"]
    assert metrics["kl_tokens"] == 2 and len(metrics["sequences"]) == 1
    assert summarize(metrics["sequences"], metrics["kl_tokens"])["kl_seq"] > 0


def test_token_cap_is_a_strict_completion_prefix_without_new_eos():
    from sn120_v2.gpu.kl import tokenize_rows

    class Tokenizer:
        convert_tokens_to_ids = lambda self, token: 99
        apply_chat_template = lambda self, *args, **kwargs: "prompt"
        encode = lambda self, text, **kwargs: [0] if text == "prompt" else [1, 2, 3, 4]

    rows = [dict(task_pos=0, prompt=[], completion_text="answer", outcome="P"),
            dict(task_pos=1, prompt=[], completion_text="capped", outcome="C")]
    encoded = tokenize_rows(Tokenizer(), rows, max_tokens=2)
    assert [row["completion_ids"] for row in encoded] == [[1, 2], [1, 2]]
    assert tokenize_rows(Tokenizer(), rows)[0]["completion_ids"] == [1, 2, 3, 4, 99]


def sharded_runner(tmp_path, monkeypatch, build, n_entries=2, n_gpus=8, n_rows=20, **param_changes):
    from sn120_v2.gpu import kl
    from sn120_v2.gpu.plan import BoxConfig

    params = build.params(min_kl_seq=0.25, eval_merged=True, n_gpus=n_gpus, kl_rollouts=n_rows, **param_changes)
    ticket = build.ticket(n_entries=n_entries, n_tasks=n_rows, params=params)
    reports = [build.artifact(ticket.head, seat, params, tag=str(seat.queue_pos)) for seat in ticket.entries]
    head_dir, candidate_dir = tmp_path / "head", tmp_path / "candidate"
    head_dir.mkdir()
    candidate_dir.mkdir()
    rows = [dict(sample_index=0, task_pos=pos, outcome="P", prompt=[], completion_text="ok") for pos in range(n_rows)]
    for name in ("null", *(report.lora_hash for report in reports)):
        path = tmp_path / "rollouts" / ticket.round_id / "captures" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(kl, "resident_head", lambda resident, head: head_dir)
    runner = kl.KLRunner(BoxConfig(tmp_path, build.image), object())
    proof = SimpleNamespace(path=candidate_dir)
    entries = [SimpleNamespace(queue_pos=pos, status=SCORED, lora=report, outcomes=("P" * n_rows,))
               for pos, report in enumerate(reports)]
    return runner, ticket, entries, {pos: proof for pos in range(n_entries)}


def test_rollout_shards_use_all_gpus_and_combine_float64_metrics(tmp_path, monkeypatch, build):
    runner, ticket, entries, tested = sharded_runner(tmp_path, monkeypatch, build)
    spawned, waits, stopped = [], [], []
    loaded = {}

    def child(command, env, log):
        directory = Path(command[-1])
        gpu = directory.name.removeprefix("gpu")
        spawned.append(gpu)
        items = json.loads((directory / "job.json").read_text())["items"]
        loaded[gpu] = items
        for item in items:
            assert len(item["rows"]) <= 3
            sequences = [float(row["task_pos"]) + 0.000225 for row in item["rows"]]
            tokens = 1000 + len(sequences)
            metrics = {"sequences": sequences, "kl_tokens": tokens}
            (directory / f"result-{item['shard']}.json").write_text(json.dumps({"metrics": metrics, "wall_s": 0.125}))
        return SimpleNamespace(wait=lambda timeout: waits.append(timeout) or 0, stop=lambda: stopped.append(gpu))

    runner.spawn = child
    measured = runner.measure(ticket, entries, "P" * 20, tested)
    assert sorted(spawned) == sorted(["null", *map(str, range(8))]) and sorted(stopped) == sorted(spawned)
    assert {item["key"] for item in loaded["null"]} == {"null"}
    assert all(any(item["key"] == "0" for item in loaded[str(gpu)]) for gpu in range(8))
    assert all(sorted(row["task_pos"] for items in loaded.values() for item in items if item["key"] == key
                      for row in item["rows"]) == list(range(20)) for key in ("0", "1"))
    assert measured["entries"][0]["kl_k"] == measured["entries"][1]["kl_k"] == 20
    assert measured["null"]["kl_k"] == 1  # null is log-only and cheap; entries own the full fan-out
    assert measured["entries"][0] == measured["entries"][1]
    assert measured["entries"][0]["kl_seq"] == math.fsum(pos + 0.000225 for pos in range(20)) / 20
    logs = [json.loads(line) for line in (tmp_path / "logs" / "phases.jsonl").read_text().splitlines()]
    gpu_logs = [row for row in logs if row["phase"] == "kl_gpu"]
    assert sorted(row["gpu"] for row in gpu_logs if row["worker"] != "null") == list(range(8))
    assert all(row["tokens"] == sum(1000 + len(item["rows"]) for item in loaded[row["worker"]])
               and row["tokens_s"] == pytest.approx(row["tokens"] / row["wall_s"])
               for row in gpu_logs)
    assert all(row["wall_s"] > 0 and row["model_tokens"] == 2 * row["tokens"] for row in gpu_logs)


@pytest.mark.parametrize("metrics", [
    {"sequences": [math.nan], "kl_tokens": 10},
    {"sequences": [], "kl_tokens": 10},
    {"sequences": [0.1], "kl_tokens": 0},
    {"sequences": [-0.1], "kl_tokens": 10},
])
def test_malformed_shard_metrics_are_no_decision(tmp_path, monkeypatch, build, metrics):
    runner, ticket, entries, tested = sharded_runner(tmp_path, monkeypatch, build, n_entries=1, n_rows=8)
    entries[0].outcomes = (PASS * 8,)

    def child(command, env, log):
        directory = Path(command[-1])
        for item in json.loads((directory / "job.json").read_text())["items"]:
            (directory / f"result-{item['shard']}.json").write_text(json.dumps({"metrics": metrics}))
        return SimpleNamespace(wait=lambda timeout: 0, stop=lambda: None)

    runner.spawn = child
    measured = runner.measure(ticket, entries, PASS * 8, tested)
    assert measured["null"] is None
    assert "error" in measured["entries"][0] and "kl_seq" not in measured["entries"][0]


def test_single_entry_uses_all_gpus(tmp_path, monkeypatch, build):
    runner, ticket, entries, tested = sharded_runner(tmp_path, monkeypatch, build, n_entries=1, n_rows=128)
    seen = {}

    def child(command, env, log):
        directory = Path(command[-1])
        gpu = directory.name.removeprefix("gpu")
        items = json.loads((directory / "job.json").read_text())["items"]
        seen[gpu] = items
        for item in items:
            metrics = {"sequences": [0.000225123456789] * len(item["rows"]), "kl_tokens": len(item["rows"]) * 30000}
            (directory / f"result-{item['shard']}.json").write_text(json.dumps({"metrics": metrics, "wall_s": 1.0}))
        return SimpleNamespace(wait=lambda timeout: 0, stop=lambda: None)

    runner.spawn = child
    measured = runner.measure(ticket, entries, "P" * 128, tested)
    assert all(str(gpu) in seen for gpu in range(8))
    assert all(item["key"] == "0" for gpu, items in seen.items() if gpu != "null" for item in items)
    assert measured["entries"][0]["kl_k"] == 128 and measured["entries"][0]["kl_tokens"] == 128 * 30000
    assert measured["entries"][0]["kl_seq"] == 0.000225123456789


def test_phase_timeout_stops_children_and_only_unfinished_entries_are_no_decision(tmp_path, monkeypatch, build):
    from sn120_v2.gpu.solo import SoloRunner

    runner, ticket, entries, tested = sharded_runner(tmp_path, monkeypatch, build, n_entries=2)
    monkeypatch.setattr(time, "time", lambda: ticket.issued_at + ticket.params.round_time_limit_s - 60)
    stops = []

    def child(command, env, log):
        directory = Path(command[-1])
        # One null shard finishes before the hard cap. Its completed work still
        # contributes to this GPU's tokens/s; no incomplete entry is accepted.
        if directory.name == "gpunull":
            item = json.loads((directory / "job.json").read_text())["items"][0]
            assert item["key"] == "null"
            metrics = {"sequences": [0.0] * len(item["rows"]), "kl_tokens": len(item["rows"]) * 2}
            (directory / f"result-{item['shard']}.json").write_text(json.dumps({"metrics": metrics}))
        return SimpleNamespace(wait=lambda timeout: None, stop=lambda: stops.append(env["CUDA_VISIBLE_DEVICES"]))

    runner.spawn = child
    measured = runner.measure(ticket, entries, "P" * 20, tested)
    assert sorted(stops) == ["0", "0", *map(str, range(1, 8))]
    assert all("TimeoutError" in measured["entries"][pos]["error"] for pos in (0, 1))
    assert measured["null"] is None  # calibration never substitutes for an entry
    reports = {pos: build.artifact(ticket.head, seat, ticket.params) for pos, seat in enumerate(ticket.entries)}
    failed = SoloRunner._kl_reports(ticket, reports, measured["entries"], ("P" * 20,), False)
    assert all(report.status == NO_DECISION and not decide([(ticket, build.result(ticket, "P" * 20, {pos: report}))]).entries[0].charged
               for pos, report in failed.items())
    gpu_logs = [json.loads(line) for line in (tmp_path / "logs" / "phases.jsonl").read_text().splitlines()
                if json.loads(line)["phase"] == "kl_gpu"]
    assert len(gpu_logs) == 9 and all(row["tokens"] == 0 and row["tokens_s"] == 0.0 for row in gpu_logs)
    assert next(row for row in gpu_logs if row["worker"] == "null")["gpu"] == 0
    assert all("TimeoutError" in row["error"] for row in gpu_logs)


def test_small_real_kl_and_publication_keep_float64_precision(build):
    ticket, result = gated(build, 0.000225123456789)
    decision = decide([(ticket, result)])
    assert decision.entries[0].kl_seq == 0.000225123456789
    assert EntryDecision.from_dict(decision.entries[0].to_dict()).kl_seq == decision.entries[0].kl_seq
    assert json.dumps(decision.entries[0].to_dict()).find("0.000225123456789") >= 0


def test_parent_isolates_missing_capture_and_uses_tested_merged_dir(tmp_path, monkeypatch, build):
    from sn120_v2.gpu import kl
    from sn120_v2.gpu.plan import BoxConfig

    ticket = build.ticket(n_entries=2, params=build.params(min_kl_seq=0.25, eval_merged=True, n_gpus=1))
    reports = [build.artifact(ticket.head, seat, ticket.params) for seat in ticket.entries]
    head_dir, candidate_dir = tmp_path / "head", tmp_path / "candidate"
    head_dir.mkdir()
    candidate_dir.mkdir()
    row = {"sample_index": 0, "task_pos": 0, "outcome": "P", "prompt": [], "completion_text": "ok"}
    null_capture = tmp_path / "rollouts" / ticket.round_id / "captures" / "null.jsonl"
    null_capture.parent.mkdir(parents=True)
    null_capture.write_text(json.dumps(row) + "\n")
    first_capture = null_capture.with_name(f"{reports[0].lora_hash}.jsonl")
    first_capture.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(kl, "resident_head", lambda resident, head: head_dir)
    proof = SimpleNamespace(path=candidate_dir)
    runner = kl.KLRunner(BoxConfig(tmp_path, build.image), object())
    outcomes = {pos: (PASS * 8,) for pos in (0, 1)}

    def child(command, env, log):
        directory = Path(command[-1])
        for item in json.loads((directory / "job.json").read_text())["items"]:
            metrics = {"sequences": [METRICS["kl_seq"]] * len(item["rows"]), "kl_tokens": len(item["rows"]) * 10}
            (directory / f"result-{item['shard']}.json").write_text(json.dumps({"metrics": metrics, "wall_s": 0.1}))
        return SimpleNamespace(wait=lambda timeout: 0, stop=lambda: None)

    runner.spawn = child
    measured = runner.measure(ticket, [SimpleNamespace(queue_pos=pos, status=SCORED, lora=report, outcomes=outcomes[pos]) for pos, report in enumerate(reports)], (PASS * 8,), {pos: proof for pos in (0, 1)})
    assert measured["entries"][0]["kl_seq"] == 0.5 and measured["null"]["kl_seq"] == 0.5
    assert "cannot read rollout capture" in measured["entries"][1]["error"]


def test_normal_solo_handoff_measures_scored_entries_and_seals_failures(build):
    from sn120_v2.gpu.solo import SoloRunner, SoloSettings
    from sn120_v2.types import NULL_MODEL

    params = build.params(min_kl_seq=0.25, eval_merged=True, n_gpus=1)
    ticket = build.ticket(n_entries=2, params=params)
    artifacts = {seat.queue_pos: build.artifact(ticket.head, seat, params) for seat in ticket.entries}
    seen = []

    class Evaluator:
        http = None

        def solo_worker(self, ticket, trained, deadline, gpu):
            runner = self

            class Worker:
                def run(self, *, steal, result, judge, alive, deadline):
                    while claim := steal(gpu, NULL_MODEL):
                        model, start, stop = claim
                        for index in range(start, stop):
                            result(model, 0, index, PASS)

            return Worker()

        def build_merged(self, round_id, head, artifact, params, gpu):
            return SimpleNamespace(path=Path("/"), base_hash=head.base_hash, kernel="fixed-order-v1")

        def _serve_one(self, *args, **kwargs):
            return None

        def close(self):
            pass

        def measure_kl(self, ticket, entries, null, tested):
            seen.extend(entries)
            return {"entries": {0: METRICS, 1: {"error": "CUDA out of memory"}},
                    "null": METRICS, "wall_s": 30.0}

    trainer = SimpleNamespace(train=lambda request: artifacts[request.entry.queue_pos], close=lambda: None)
    runner = SoloRunner(trainer, Evaluator(), lambda: 1000, build.image,
                        settings=SoloSettings(prewarm=False), poll_s=0.01)
    result = runner.run(ticket)
    assert len(seen) == 2 and all(entry.status == SCORED and entry.lora is not None and entry.outcomes for entry in seen)
    assert result.entries[0].kl_seq == 0.5 and result.entries[1].status == NO_DECISION
    assert result.phase_s == {"training": 0, "evaluation": 0, "kl": 30}
    check_result(ticket, result)


def test_evaluator_top_level_kl_failure_is_per_entry(build):
    from sn120_v2.gpu.evaluator import VllmEvaluator

    ticket = build.ticket(n_entries=1, params=build.params(min_kl_seq=0.25))
    entry = EntryResult(0, SCORED, lora=build.artifact(ticket.head, ticket.entries[0], ticket.params), outcomes=(PASS * 8,))
    evaluator = SimpleNamespace(config=SimpleNamespace(work_dir=Path("/tmp/kl-test")), resident=object())
    measured = VllmEvaluator.measure_kl(evaluator, ticket, [entry], (PASS * 8,), {})
    assert measured["entries"][0]["error"] and measured["null"] is None


def test_pooled_null_token_kl_uses_token_mass(build):
    ticket, result = gated(build, 0.5)
    from sn120_v2.state import Lease, RoundRecord
    from sn120_v2.judging import decide

    first = dict(kl_seq=100.0, kl_seq_se=0.0, kl_tok=1.0, kl_k=2, kl_tokens=100)
    second = dict(kl_seq=1.0, kl_seq_se=0.0, kl_tok=0.1, kl_k=1, kl_tokens=10)
    result = dataclasses.replace(result, **{f"null_{key}": value for key, value in first.items()})
    other_ticket = dataclasses.replace(ticket, lease_id=ticket.lease_id + "b", box_id="box-b")
    other = dataclasses.replace(build.result(other_ticket, PASS * 8, {0: PASS * 8}), **{f"null_{key}": value for key, value in second.items()})
    decision = decide([(ticket, result)])
    lease = Lease(ticket.lease_id, ticket.box_id, ticket.issued_at, ticket.expires_at, ticket.digest(), ticket.entries)
    record = RoundRecord(ticket.round_id, ticket.head.version_id, ticket.params.params_hash, ticket.params.recipe_id,
                         ticket.draw.round_seed, ticket.draw.snapshot_id, len(ticket.draw.task_ids), ticket.issued_at,
                         (lease,), decision=decision)
    report = round_report(record, [(ticket, result), (other_ticket, other)])
    assert report["null_kl_seq"] == pytest.approx(201 / 3)
    assert report["null_kl_tok"] == pytest.approx(101 / 110)


def test_kl_deadline_uses_remaining_promotion_reserve(tmp_path, monkeypatch, build):
    from sn120_v2.gpu import kl
    from sn120_v2.gpu.plan import BoxConfig

    ticket = build.ticket(n_entries=1, params=build.params(min_kl_seq=0.25, eval_merged=True, n_gpus=1,
                                                              round_time_limit_s=6000, promote_reserve_s=600, lease_timeout_s=14400))
    ticket = dataclasses.replace(ticket, expires_at=ticket.issued_at + ticket.params.lease_timeout_s)
    artifact = build.artifact(ticket.head, ticket.entries[0], ticket.params)
    head_dir, candidate_dir = tmp_path / "head", tmp_path / "candidate"
    head_dir.mkdir()
    candidate_dir.mkdir()
    row = {"sample_index": 0, "task_pos": 0, "outcome": "P", "prompt": [], "completion_text": "ok"}
    for name in ("null", artifact.lora_hash):
        path = tmp_path / "rollouts" / ticket.round_id / "captures" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(kl, "resident_head", lambda resident, head: head_dir)
    monkeypatch.setattr(time, "time", lambda: ticket.issued_at + 5401)
    waits = []
    def child(command, env, log):
        directory = Path(command[-1])
        for item in json.loads((directory / "job.json").read_text())["items"]:
            metrics = {"sequences": [METRICS["kl_seq"]] * len(item["rows"]), "kl_tokens": len(item["rows"]) * 10}
            (directory / f"result-{item['shard']}.json").write_text(json.dumps({"metrics": metrics}))
        return SimpleNamespace(wait=lambda timeout: waits.append(timeout) or 0, stop=lambda: None)
    runner = kl.KLRunner(BoxConfig(tmp_path, build.image), object(), spawn=child)
    entry = EntryResult(0, SCORED, lora=artifact, outcomes=(PASS * 8,))
    measured = runner.measure(ticket, [entry], (PASS * 8,), {0: SimpleNamespace(path=candidate_dir)})
    assert waits and waits[0] == pytest.approx(539)
    assert measured["entries"][0]["kl_seq"] == 0.5


def test_independent_phase_cap_applies_before_a_later_promotion_deadline(tmp_path, monkeypatch, build):
    runner, ticket, entries, tested = sharded_runner(tmp_path, monkeypatch, build, n_entries=1, n_rows=8, kl_time_limit_s=120)
    ticket = dataclasses.replace(ticket, issued_at=1000, expires_at=5000)
    entries[0].outcomes = (PASS * 8,)
    monkeypatch.setattr(time, "time", lambda: 1000)
    waits = []

    def child(command, env, log):
        directory = Path(command[-1])
        for item in json.loads((directory / "job.json").read_text())["items"]:
            metrics = {"sequences": [0.5] * len(item["rows"]), "kl_tokens": len(item["rows"]) * 10}
            (directory / f"result-{item['shard']}.json").write_text(json.dumps({"metrics": metrics}))
        return SimpleNamespace(wait=lambda timeout: waits.append(timeout) or 0, stop=lambda: None)

    runner.spawn = child
    measured = runner.measure(ticket, entries, PASS * 8, tested)
    assert measured["entries"][0]["kl_seq"] == 0.5
    assert waits and waits[0] == pytest.approx(120)
