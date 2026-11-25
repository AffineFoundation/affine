from __future__ import annotations

import dataclasses
import json

from sn120_v2 import layout
from sn120_v2.params import Params
from sn120_v2.canonical import canonical_json
from sn120_v2.dedup import dryrun
from sn120_v2.errors import ContractError
from sn120_v2.public import snapshot
from sn120_v2.sim import SimAdmission, SimWorld, sim_params
from sn120_v2.types import Verdict

import pytest


def affine_batch(hotkey: str, rows: list[tuple[str, float]], prompt: str = "p") -> bytes:
    return canonical_json({"version": "affine-batch-v3", "hotkey": hotkey, "valid": True,
                           "quality": 0.4 if hotkey == "hk1" else 0.0, "cap_rate": 0.0, "train": {},
                           "groups": [{"prompt": prompt, "rollouts": [{"output": out, "reward": reward} for out, reward in rows]}]})


def sim_read(raws, qualities):
    return lambda hotkey: (raws[hotkey], {"valid": True, "quality": qualities.get(hotkey, 0.0),
                                          "cap_rate": 0.0, "train": {}}) if hotkey in raws else None


def test_stage_a_charges_only_the_copy_and_persists_keeper_first_seen(tmp_path):
    world = SimWorld(tmp_path, sim_params(dedup_enabled=True, train_seed_scope="round"))
    world.register(1, 0.4)
    world.register(2, 0.0)
    world.register(3, -0.1)
    world.controller.state.queue.pop((3, 1003), None)
    raws = {hotkey: affine_batch(hotkey, [("  answer   one " if hotkey == "hk2" else "answer one", 1.0),
                                         ("answer two", 0.0)])
            for hotkey in ("hk1", "hk2")}
    # Equivalent normalized rollouts, different bytes: the earliest file keeps its ticket.
    world.admission._read = sim_read(raws, {"hk1": 0.4, "hk2": 0.0})
    # Keep the third seat for a later assertion without changing the selected pair.
    original = world.controller._rollout_hashes
    world.controller._rollout_hashes = lambda sha, uri="": original(sha, uri)
    record = world.round()
    assert [row.status for row in record.decision.entries] == ["winner", "duplicate"]
    assert not world.controller.state.queue
    assert record.duplicate_decisions[0].charged
    assert record.duplicate_clusters[0]["keeper"]["uid"] == 1
    assert record.duplicate_clusters[0]["keeper"]["first_seen_unix"] == record.first_seen[record.duplicate_clusters[0]["keeper"]["batch_sha256"]]
    assert record.duplicate_decisions[0].reason.endswith("similarity 1")
    public = snapshot(world.controller.state, world.params, world.clock())
    assert public["history"][-1]["duplicate_clusters"] == list(record.duplicate_clusters)
    assert public["history"][-1]["duplicates"][0]["status"] == "duplicate"
    assert not public["history"][-1]["duplicates"][0].get("rollouts")
    world.reopen()
    assert world.controller.state.rounds[-1].duplicate_clusters == record.duplicate_clusters


def test_an_exact_copy_of_a_folded_past_king_never_trains(tmp_path):
    world = SimWorld(tmp_path, sim_params(dedup_enabled=True, train_seed_scope="round"))
    world.register(1, 0.3)
    first = world.round()
    assert first.outcome == "applied" and world.controller.state.head.folded
    world.admission.upload("hk2", 0.0)
    world.register(2, 0.0)
    # Same bytes as the winner, but a different owner: the exact batch SHA decides, not hotkey.
    original_read = world.admission._read
    winner_raw = world.store.get(layout.winner_batch(first.round_id))
    world.admission._read = lambda hotkey: (winner_raw, original_read(hotkey)[1]) if hotkey == "hk2" else original_read(hotkey)
    second = world.round()
    assert second.leases == () and second.decision.entries[0].status == "duplicate"
    assert second.duplicate_clusters[0]["dropped"][0]["stage"] == "past_king_exact"
    assert world.controller.state.queue == {}


def test_first_seen_is_only_absent_insert_and_survives_a_changed_file(tmp_path):
    world = SimWorld(tmp_path, sim_params(dedup_enabled=True))
    world.register(1, 0.0)
    verdict = Verdict("hk1", True, batch_sha256="a" * 64)
    world.controller._update_first_seen({"hk1": verdict}, 100)
    world.controller._update_first_seen({"hk1": verdict}, 200)
    other = Verdict("hk1", True, batch_sha256="b" * 64)
    world.controller._update_first_seen({"hk1": other}, 300)
    world.controller._update_first_seen({"hk1": Verdict("hk1", False, rule="invalid")}, 400)
    found = json.loads(world.store.get(layout.DEDUP_FIRST_SEEN))
    assert found == {"a" * 64: {"hotkey": "hk1", "first_seen_unix": 100},
                     "b" * 64: {"hotkey": "hk1", "first_seen_unix": 300}}


def test_damaged_or_unavailable_first_seen_never_blocks_file_checks(tmp_path, capsys):
    world = SimWorld(tmp_path, sim_params(dedup_enabled=True))
    world.register(1, 0.0)
    verdict = Verdict("hk1", True, batch_sha256="a" * 64)
    world.admission.verdicts = lambda hotkeys: {"hk1": verdict}
    for bad in (b"not JSON", b"[]", b"null"):
        world.store.put(layout.DEDUP_FIRST_SEEN, bad)
        assert world.controller.check_files(123)["hk1"] == verdict
        repaired = json.loads(world.store.get(layout.DEDUP_FIRST_SEEN))
        assert repaired["a" * 64]["first_seen_unix"] == 123

    original = world.store.get

    def unavailable(key):
        if key == layout.DEDUP_FIRST_SEEN:
            raise OSError("simulated ledger outage")
        return original(key)

    world.store.get = unavailable
    assert world.controller.check_files(456)["hk1"] == verdict
    assert "leaving it unchanged" in capsys.readouterr().err


def test_stage_a_rollout_and_cache_failures_keep_exact_duplicate_detection(tmp_path, monkeypatch):
    from sn120_v2.errors import InfraError

    world = SimWorld(tmp_path, sim_params(dedup_enabled=True, train_seed_scope="round"))
    world.register(1, 0.4)
    world.register(2, 0.0)
    raw = affine_batch("hk1", [("answer", 1.0), ("tail", 0.0)])
    world.admission._read = sim_read({"hk1": raw, "hk2": raw}, {"hk1": 0.4})
    monkeypatch.setattr(world.controller, "_rollout_hashes", lambda *args: (_ for _ in ()).throw(InfraError("store outage")))
    record = world.round()
    assert [row.status for row in record.decision.entries] == ["winner", "duplicate"]
    assert record.duplicate_clusters[0]["dropped"][0]["stage"] == "exact_batch"

    import hashlib

    sha = hashlib.sha256(raw).hexdigest()
    world2 = SimWorld(tmp_path / "second", sim_params(dedup_enabled=True, train_seed_scope="round"))
    world2.register(1, 0.4)
    world2.register(2, 0.0)
    world2.admission._read = sim_read({"hk1": raw, "hk2": raw}, {"hk1": 0.4})
    world2.controller._update_first_seen({
        "hk1": Verdict("hk1", True, batch_sha256=sha),
        "hk2": Verdict("hk2", True, batch_sha256=sha),
    }, 1)

    def evidence(candidate, uri=""):
        if candidate == sha:
            raise OSError("simulated batch read outage")
        return None

    world2.controller._rollout_hashes = evidence
    second = world2.round()
    assert [row.status for row in second.decision.entries] == ["winner", "duplicate"]
    assert second.duplicate_clusters[0]["dropped"][0]["stage"] == "exact_batch"


def test_dryrun_is_read_only_and_counts_real_advantages(tmp_path, capsys):
    from sn120_v2.cli import main
    raw = canonical_json({"version": "affine-batch-v3", "hotkey": "miner", "train": {"group_size": 4},
                          "groups": [{"prompt": f"p{i}", "rollouts": [{"output": f"first {i}", "reward": 1.0},
                                                                            {"output": f"second {i}", "reward": 0.0}]}
                                     for i in range(31)] + [{"prompt": "last", "rollouts": [{"output": "one", "reward": 1.0},
                                                                                              {"output": "two", "reward": 0.0},
                                                                                              {"output": "three", "reward": 0.0}]}]})
    batch = tmp_path / "batch.json"
    batch.write_bytes(raw)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert main(["dedup-dryrun", "--batch", str(batch)]) == 0
    report = json.loads(capsys.readouterr().out)
    row = report["entries"][0]
    assert row["rollouts"] == 65 and row["nonzero_advantage_rollouts"] == 65
    assert row["would_fail_min_rollouts"] is False and report["read_only"]
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    with pytest.raises(ContractError):
        dryrun([batch, batch], Params(), first_seen_unix=(1,))
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_partial_duplicate_selection_charges_copies_and_returns_survivors(tmp_path):
    world = SimWorld(tmp_path, sim_params(dedup_enabled=True, train_seed_scope="round", n_min_start=3))
    world.register(1, 0.4)
    world.register(2, 0.0)
    world.register(3, -0.1)
    raws = {
        "hk1": affine_batch("hk1", [("answer", 1.0), ("honest-tail", 0.0)], prompt="copy"),
        "hk2": affine_batch("hk2", [("  answer  ", 1.0), ("copy-tail", 0.0)], prompt="copy"),
        "hk3": affine_batch("hk3", [("independent", 1.0), ("independent-tail", 0.0)], prompt="honest"),
    }
    world.admission._read = sim_read(raws, {"hk1": 0.4})
    record = world.round()
    assert record is not None and record.leases == () and record.outcome == "null"
    assert [(row.uid, row.status) for row in record.decision.entries] == [(1, "no_decision"), (2, "duplicate"), (3, "no_decision")]
    public = snapshot(world.controller.state, world.params, world.clock())
    assert public["history"][-1]["duplicate_clusters"] == list(record.duplicate_clusters), "Stage-A evidence appears once"
    assert record.duplicate_decisions[0].charged
    assert set(world.controller.state.queue) == {(1, 1001), (3, 1003)}
    world.reopen()
    assert set(world.controller.state.queue) == {(1, 1001), (3, 1003)}


def test_first_seen_deployment_tie_and_malformed_private_cache_fail_open(tmp_path, capsys):
    world = SimWorld(tmp_path, sim_params(dedup_enabled=True))
    world.register(1, 0.0)
    world.register(2, 0.0)
    first = Verdict("hk1", True, batch_sha256="a" * 64)
    second = Verdict("hk2", True, batch_sha256="b" * 64)
    world.admission.verdicts = lambda hotkeys: {"hk1": first, "hk2": second}
    world.controller.check_files(123)
    found = json.loads(world.store.get(layout.DEDUP_FIRST_SEEN))
    assert found == {"a" * 64: {"hotkey": "hk1", "first_seen_unix": 123},
                     "b" * 64: {"hotkey": "hk2", "first_seen_unix": 123}}
    world.controller.check_files(456)
    assert json.loads(world.store.get(layout.DEDUP_FIRST_SEEN)) == found

    # A damaged row never invents a timestamp or overwrites another valid row.
    damaged = dict(found) | {"not-a-sha": {"hotkey": "bad", "first_seen_unix": 1},
                             "b" * 64: {"hotkey": "hk2", "first_seen_unix": "old"}}
    world.store.put(layout.DEDUP_FIRST_SEEN, canonical_json(damaged))
    assert world.controller._first_seen() == {"a" * 64: 123}
    world.controller._update_first_seen({"hk2": second, "hk3": Verdict("hk3", True, batch_sha256="c" * 64)}, 789)
    repaired = json.loads(world.store.get(layout.DEDUP_FIRST_SEEN))
    assert repaired["a" * 64] == found["a" * 64]
    assert repaired["b" * 64] == {"hotkey": "hk2", "first_seen_unix": 789}
    assert repaired["c" * 64]["first_seen_unix"] == 789 and "not-a-sha" not in repaired

    sha = "d" * 64
    uri = layout.frozen_batch("r000001", "hk1")
    raw = affine_batch("hk1", [("answer", 1.0), ("other", 0.0)])
    import hashlib
    sha = hashlib.sha256(raw).hexdigest()
    world.store.put(uri, raw)
    key = layout.dedup_rollout_hashes(sha)
    for bad in (b"not JSON", b'["short"]', canonical_json(["a" * 64, "a" * 64]), b"[]", b"[{}]", b"[[]]"):
        world.store.put(key, bad)
        assert world.controller._rollout_hashes(sha, uri) is not None
        rebuilt = json.loads(world.store.get(key))
        assert len(rebuilt) == 2 and len(set(rebuilt)) == 2
    world.store.delete(uri)
    world.store.put(key, b"not JSON")
    assert world.controller._rollout_hashes(sha) is None
import dataclasses

from sn120_v2.types import DUPLICATE, NULL_UPDATE
from sn120_v2.canonical import manifest_digest
from test_dedup import lora_bytes


def _synthetic_lora(world, artifact, value):
    raw = lora_bytes([[value]], [[value]])
    world.store.put(f"{artifact.prefix}/adapter_model.safetensors", raw)
    files = dict(artifact.files) | {"adapter_model.safetensors": __import__("hashlib").sha256(raw).hexdigest()}
    return dataclasses.replace(artifact, files=files, sizes={name: len(raw) for name in files},
                               lora_hash=manifest_digest(files), rank=1, alpha=1.0)


def test_controller_null_cannot_keep_stage_b_and_duplicate_cannot_win(tmp_path, monkeypatch):
    world = SimWorld(tmp_path, sim_params(dedup_enabled=True, train_seed_scope="round", min_step_norm=0.02))
    world.register(1, 0.5)
    world.register(2, 0.4)
    world.register(3, 0.3)
    params = dataclasses.replace(world.params, recipe=dataclasses.replace(world.params.recipe, rank=1, alpha=1.0))
    world.controller._params = params
    world.controller.state.params = params
    world.params = params
    original = world.trainer.train

    def train(request):
        return _synthetic_lora(world, original(request), 2.0 + request.entry.queue_pos)


    world.trainer.train = train
    report = world.round()
    # An identical real LoRA is a Stage-B duplicate even with a stronger simulated outcome.
    assert [row.status for row in report.decision.entries] == ["winner", "duplicate", "duplicate"]
    assert report.decision.duplicate_clusters[0]["keeper"]["uid"] == 1
    assert report.duplicate_decisions == ()
    public = snapshot(world.controller.state, params, world.clock())
    assert len(public["history"][-1]["duplicate_clusters"]) == 1
    assert not any("rollouts" in member for cluster in public["history"][-1]["duplicate_clusters"] for member in [cluster["keeper"], *cluster["dropped"]])
    world.reopen()
    assert world.controller.state.rounds[-1].decision.duplicate_clusters == report.decision.duplicate_clusters

    # A non-finite stored norm never reaches keeper selection; the next real candidate can win.
    world2 = SimWorld(tmp_path / "null", sim_params(dedup_enabled=True, train_seed_scope="round", min_step_norm=0.02))
    world2.register(1, 0.9)
    world2.register(2, 0.2)
    calls = []

    def stage_b(entries, artifacts, first_seen, params, get_file, expected_seeds, excluded=(), log=None):
        calls.append(set(excluded))
        assert 0 in excluded
        return [], []

    import sn120_v2.controller as module
    monkeypatch.setattr(module, "stage_b", stage_b)
    real_train = world2.trainer.train

    def null_first(request):
        artifact = real_train(request)
        if request.entry.queue_pos == 0:
            artifact = dataclasses.replace(artifact, report=dataclasses.replace(artifact.report, step_norm=float("nan")))
        return artifact

    world2.trainer.train = null_first
    record = world2.round()
    assert calls == [{0}]
    assert [row.status for row in record.decision.entries] == [NULL_UPDATE, "winner"]
    assert record.decision.winner.uid == 2
    assert record.decision.duplicate_clusters == ()


def test_controller_whole_stage_b_failure_keeps_real_candidates(tmp_path, monkeypatch):
    world = SimWorld(tmp_path, sim_params(dedup_enabled=True, train_seed_scope="round", min_step_norm=0.02))
    world.register(1, 0.4)
    world.register(2, 0.0)
    import sn120_v2.controller as module
    monkeypatch.setattr(module, "stage_b", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("simulated comparison outage")))
    record = world.round()
    assert [row.status for row in record.decision.entries] == ["winner", "lost"]
    assert record.decision.duplicate_clusters == ()
    world.reopen()
    assert world.controller.state.rounds[-1].decision.winner.uid == 1


def test_all_failed_leases_keep_duplicates_charged_and_survivors_uncharged(tmp_path):
    world = SimWorld(tmp_path, sim_params(dedup_enabled=True, train_seed_scope="round", n_min_start=2))
    world.register(1, 0.4)
    world.register(2, 0.0)
    world.register(3, 0.3)
    raws = {
        "hk1": affine_batch("hk1", [("answer", 1.0), ("tail", 0.0)]),
        "hk2": affine_batch("hk2", [(" answer ", 1.0), (" tail ", 0.0)]),
        "hk3": affine_batch("hk3", [("independent", 1.0), ("other", 0.0)]),
    }
    world.admission._read = sim_read(raws, {"hk1": 0.4, "hk3": 0.3})
    tickets = world.controller.open_round(world.clock(), lambda: "after-freeze")
    assert len(tickets[0].entries) == 2
    failed_at = world.clock()
    world.controller.fail_lease(tickets[0].lease_id, "simulated box outage", failed_at)
    record = world.controller.state.rounds[-1]
    assert [row.status for row in record.decision.entries] == ["no_decision", "duplicate", "no_decision"]
    assert record.outcome == "null", "charged duplicates close as null; survivor rows remain no_decision"
    assert world.controller.state.failed_leases == 1, "a duplicate-bearing all-failed round retains lease backoff"
    assert world.controller.state.last_failure_at == failed_at
    assert not record.decision.entries[0].charged and record.decision.entries[1].charged and not record.decision.entries[2].charged
    assert set(world.controller.state.queue) == {(1, 1001), (3, 1003)}
    world.reopen()
    assert set(world.controller.state.queue) == {(1, 1001), (3, 1003)}
    assert world.controller.state.failed_leases == 1 and world.controller.state.last_failure_at == failed_at


def test_disabled_dedup_never_reads_lora_files_at_settlement(tmp_path):
    world = SimWorld(tmp_path, sim_params(dedup_enabled=False))
    world.register(1, 0.4)
    world.register(2, 0.0)
    reads = []
    get = world.store.get

    def watched(key):
        reads.append(key)
        return get(key)

    world.store.get = watched
    record = world.round()
    assert record is not None and record.decision is not None
    assert not any(key.endswith("adapter_model.safetensors") for key in reads)


def test_default_recipe_global_cap_passes_static_null_floor():
    params = Params()
    assert params.recipe.global_norm_cap >= params.min_step_norm
