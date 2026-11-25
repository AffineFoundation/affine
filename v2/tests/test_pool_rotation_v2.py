"""The public, informativeness-ranked active pool (rotation rule v2)."""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.errors import VerifyError
from sn120_v2.irt import IRTSampler, snapshot_digest
from sn120_v2.params import Params
from sn120_v2.pool_runtime import active_file, replay_rotation
from sn120_v2.rotation import LOW, Reserve, Rotation, rotate_info
from sn120_v2.store import LocalStore


def info_params(**changes) -> Params:
    return Params(rotation_enabled=True, rotation_rule="info", **changes)


def snapshot(tasks: dict[str, list], ability: float = 0.0) -> dict:
    body = IRTSampler(list(tasks), Params()).public_snapshot()
    body["tasks"] = {task: [*row[:3], 1] for task, row in tasks.items()}
    body["heads"] = {"head": {"ability": ability}}
    body["head_version_id"] = "head"
    body["snapshot_id"] = snapshot_digest(body)
    return body


def reserve(tasks: dict[str, float] | None = None, withdrawn=()) -> Reserve:
    tasks = tasks or {}
    meta = {task: {"prior_p": p, "source_prefix": task.split(":")[0], "env": "affine_math"} for task, p in tasks.items()}
    return Reserve(sha256_hex(canonical_json(tasks)), tuple(tasks), candidate_meta=meta,
                   withdrawn=tuple({"id": task, "reason": "bad", "added_at": "2026-11-19"} for task in withdrawn))


def rows(decision: Rotation, state: str | None = None):
    return [row for row in decision.candidate_ranking if state is None or row["state"] == state]


def test_live_v1_decisions_and_default_hashes_are_unchanged():
    p = Params.from_json((Path(__file__).parent / "data" / "params_prod_shaped.json").read_bytes())
    assert Params().params_hash == "7eb09fa89387d1d2c5db2d8d0ffd627ad87ece8ef31bdfe93f3374f0c7bb1e50"
    assert p.params_hash == "5359a448461c3f3a3e09bd9a2d8df9f387d916fb02b8d02eecea3582649c74a1"
    enabled = dataclasses.replace(p, rotation_enabled=True)
    assert enabled.params_hash == "4f270223ac2da4c6bc6682fad00c18d22d32d6ebc223397ddfd8951ff190d3e7"
    for rid in ("r000066", "r000069"):
        artifact = Path("/tmp/sn120-info-live") / f"{rid}.json"
        if artifact.exists():
            decision = Rotation.from_json(artifact.read_bytes())
            assert decision.rule_version == "affine-pool-rotation-v1"
            assert decision.candidate_ranking == decision.initial_retired_archive == ()
            assert decision.retired_ledger == decision.initial_retired_ledger == decision.initial_candidate_meta == {}


def test_transition_20120_to_10000_in_one_round_and_saturated_amendment():
    tasks = {f"math:{i}": [-10.0 if i < 13541 else 1.0, 0.25, 10] for i in range(20120)}
    ability = 2.52
    # Reserve rows carry empirical priors; no IRT fit is invented for them.
    outside = {f"new:{i}": 0.35 for i in range(2000)}
    snap = snapshot(tasks, ability)
    decision = rotate_info("r000070", reversed(list(tasks)), snap, {}, reserve(outside), info_params(
        rotation_active_size=10000, rotation_churn_frac=0.1, rotation_n_min=3))
    assert len(decision.active_ids) == 10000
    assert len(decision.retired) == 11120
    assert len(decision.admitted) == 1000
    assert decision.retire_reason_counts == {"high": 11120}
    assert set(outside) - set(decision.admitted) <= set(decision.next_up)
    assert len(decision.next_up) == 2000
    assert all(row["id"] not in reserve(withdrawn=()).withdrawn_ids for row in decision.candidate_ranking if row["id"] in decision.next_up)
    kept = [row for row in decision.candidate_ranking if row["id"] in decision.active_ids]
    saturated_kept = [row for row in kept if row["flags"] == ["high"]]
    assert len(kept) == 10000 and len(saturated_kept) == 2421
    assert all(row["p"] > 0.95 and row["score"] > 0 for row in saturated_kept)
    assert all(row["admitted"] for row in kept if row["state"] == "reserve")
    scores = [row["score"] for row in kept]
    assert all(a >= b for a, b in zip(scores, scores[1:]))
    assert any(not row["admitted"] for row in rows(decision, "reserve"))
    # The rule is independent of input order and deterministic in tie order.
    other = rotate_info("r000070", list(tasks), snap, {}, reserve(outside), info_params(
        rotation_active_size=10000, rotation_churn_frac=0.1))
    assert other == decision


def test_shrink_fill_swap_and_dead_zero_with_strict_improvement():
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(20)}
    ledger = {"a:0": (8, 0), "a:1": (8, 0)}
    outside = {"b:0": 0.5, "b:1": 0.5, "b:2": 0.5, "b:3": 0.5, "b:4": 0.5}
    p = info_params(rotation_active_size=20, rotation_churn_frac=0.1, rotation_n_min_low=8)
    decision = rotate_info("r000070", active, snapshot(active), ledger, reserve(outside), p)
    assert len(decision.admitted) == len(decision.retired) == 2
    assert all(row.reason == "low" for row in decision.retired)
    assert {row["id"] for row in rows(decision) if row["score"] == 0 and LOW in row["flags"]} == {"a:0", "a:1"}
    # No tie may replace a task; a fraction below one slot means no swaps.
    tied = rotate_info("r000070", active, snapshot(active), {}, reserve({"b:0": 0.0, "b:1": 0.0}), p)
    assert not tied.admitted and not tied.retired
    one = rotate_info("r000070", active, snapshot(active, -1.0), {}, reserve({"b:0": 0.51}), info_params(rotation_active_size=20, rotation_churn_frac=0.05))
    assert one.admitted == ("b:0",) and [row.reason for row in one.retired] == ["displaced"]


def test_fill_uses_ranked_priors_and_saturated_tasks_keep_fisher_score():
    active = {f"a:{i}": [-10.0, 0.25, 4] for i in range(4)}
    outside = {"r:0": 0.8, "r:1": 0.5, "r:2": 0.1, "r:3": 0.0}
    decision = rotate_info("r000070", active, snapshot(active), {}, reserve(outside), info_params(rotation_active_size=8))
    assert set(decision.admitted) == set(outside)
    assert len(decision.retired) == 0
    ranked = [row["id"] for row in rows(decision, "reserve")]
    assert ranked[:3] == ["r:1", "r:0", "r:2"]
    saturated = [row for row in rows(decision, "active") if row["flags"] == ["high"]]
    assert saturated and all(row["score"] == pytest.approx(row["p"] * (1 - row["p"])) for row in saturated)


def test_withdrawn_leaves_outside_cap_and_never_returns():
    active = {"a:0": [0.0, 0.25, 3], "a:1": [0.0, 0.25, 3], "a:2": [0.0, 0.25, 3]}
    archive = [{"id": "a:0", "round_id": "r000060", "reason": "excess", "event_seq": 0,
                "irt_row": [0.0, 0.25, 3], "head_rollouts": 2, "head_passes": 1}]
    first = rotate_info("r000070", active, snapshot(active), {}, reserve({"r:0": 0.5}, ["a:1"]),
                        info_params(rotation_active_size=3), retired_archive=archive)
    assert [(row.id, row.reason) for row in first.retired] == [("a:1", "withdrawn")]
    assert "a:0" in first.active_ids and "r:0" in first.admitted
    second = rotate_info("r000071", first.active_ids, snapshot({task: [0.0, 0.25, 4] for task in first.active_ids}), first.ledger,
                         reserve({"r:1": 0.5}, ["a:1"]), info_params(rotation_active_size=3), retired_archive=first.retired_archive)
    assert all(row["id"] != "a:1" for row in second.candidate_ranking)


def test_retired_task_returns_with_exact_archived_irt_and_ledger():
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(2)}
    archived = {"id": "r:0", "round_id": "r000060", "reason": "excess", "event_seq": 0,
                "irt_row": [-0.123456, 0.75, 17], "head_rollouts": 7, "head_passes": 6,
                "source_prefix": "r", "env": "affine_math", "prior_p": 0.5}
    decision = rotate_info("r000070", active, snapshot(active), {}, reserve(), info_params(rotation_active_size=3), retired_archive=[archived])
    assert decision.admitted == ("r:0",) and decision.restored_irt_rows == {"r:0": (-0.123456, 0.75, 17)}
    assert decision.ledger["r:0"] == (7, 6)
    candidate = next(row for row in decision.candidate_ranking if row["id"] == "r:0")
    from sn120_v2.irt import predicted_pass
    assert candidate["p"] == predicted_pass(0.0, -0.123456)
    assert candidate["rounds_seen"] == 17
    below_evidence = rotate_info("r000071", decision.active_ids,
                                 snapshot({task: [0.0, 0.25, 4] for task in decision.active_ids}),
                                 decision.ledger, reserve(), info_params(rotation_active_size=3),
                                 retired_archive=decision.retired_archive,
                                 initial_candidate_meta={"r:0": {"source_prefix": "r", "env": "affine_math"}})
    returned = next(row for row in below_evidence.candidate_ranking if row["id"] == "r:0")
    assert returned["p"] == 0.5  # the archive's prior survives a global metadata refresh
    # A dead archived task has score zero, rather than getting a new prior.
    dead = dict(archived, head_rollouts=8, head_passes=0, id="r:1")
    decision = rotate_info("r000070", active, snapshot(active), {}, reserve(), info_params(rotation_active_size=3), retired_archive=[dead])
    row = next(row for row in decision.candidate_ranking if row["id"] == "r:1")
    assert row["score"] == 0 and row["flags"] == ["low"]


def test_public_replay_checks_ranking_and_complete_state(tmp_path):
    store = LocalStore(tmp_path)
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(12)}
    snap = snapshot(active, -1.0)
    store.put(layout.irt_snapshot(snap["snapshot_id"]), canonical_json(snap))
    p = info_params(rotation_active_size=12, rotation_churn_frac=0.25)
    decision = rotate_info("r000070", active, snap, {}, reserve({"r:0": 0.51, "r:1": 0.49}), p, source_snapshot_id=snap["snapshot_id"])
    store.put(layout.active_pool(decision.prev_active_sha), canonical_json(sorted(active)))
    active_file(store, decision)
    report = replay_rotation(store, "r000070")
    assert report["matches"] and report["actual"] == report["expected"]
    ranking = json.loads(store.get(layout.pool_ranking("r000070")))
    assert len(ranking["candidates"]) == 14 and any(row["admitted"] for row in ranking["candidates"])
    summary = json.loads(store.get(layout.POOL_SUMMARY))
    assert summary["active_size"] == 12 and summary["reserve_size"] == 0 and summary["retired_size"] == 2
    assert sum(row["count"] for row in summary["active"]["score_histogram"]) == 12
    assert not store.exists(layout.pool_ranking("r000071"))
    # Corrupting the full ranking is not hidden by an unchanged active set.
    store.put(layout.pool_ranking("r000070"), canonical_json(ranking | {"candidates": []}))
    with pytest.raises(Exception):
        replay_rotation(store, "r000070")


def test_active_pool_reads_back_before_any_reference_is_published(tmp_path):
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(4)}
    decision = rotate_info("r000070", active, snapshot(active), {}, reserve({"r:0": 0.5}), info_params(rotation_active_size=5))

    class ObservedStore(LocalStore):
        def __init__(self, root, failure):
            super().__init__(root)
            self.failure = failure
            self.active_read_back = False
            self.reference_keys = {layout.pool_rotation(decision.round_id), layout.pool_ranking(decision.round_id),
                                   layout.pool_rotation_state(decision.digest()), layout.POOL_SUMMARY}

        def put(self, key, data):
            if self.failure == "drop" and key == layout.active_pool(decision.active_sha):
                return
            assert self.active_read_back or key not in self.reference_keys, f"reference before verified active set: {key}"
            super().put(key, data)

        def get(self, key):
            data = super().get(key)
            if key == layout.active_pool(decision.active_sha):
                if self.failure == "corrupt":
                    return b"[]"
                self.active_read_back = data == canonical_json(list(decision.active_ids))
            return data

    expected = canonical_json(list(decision.active_ids))
    for failure in (None, "drop", "corrupt"):
        store = ObservedStore(tmp_path / str(failure), failure)
        store.put(layout.active_pool(decision.initial_active_sha), canonical_json(sorted(active)))
        if failure is None:
            active_file(store, decision)
            assert store.get(layout.active_pool(decision.active_sha)) == expected
            assert sha256_hex(expected) == decision.active_sha
            assert store.active_read_back
            assert json.loads(store.get(layout.POOL_SUMMARY))["active_sha"] == decision.active_sha
            before = {key: store.get(key) for key in store.list("")}
            active_file(store, decision)
            assert {key: store.get(key) for key in store.list("")} == before, "repeat publication is byte-idempotent"
        else:
            with pytest.raises(VerifyError, match="readback"):
                active_file(store, decision)
            assert not any(store.exists(key) for key in store.reference_keys), f"{failure}: an unverified active_sha was published"


def test_churn_max_and_empty_outside_are_safe():
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(4)}
    empty = rotate_info("r000070", active, snapshot(active), {}, reserve(), info_params(rotation_active_size=4))
    assert empty.active_ids == tuple(sorted(active)) and not empty.admitted and not empty.retired and not empty.next_up
    # The fractional cap alone would allow one swap; the optional ramp cap stops it.
    capped = rotate_info("r000070", active, snapshot(active, -1.0), {}, reserve({"r:0": 0.5}),
                         info_params(rotation_active_size=4, rotation_churn_frac=0.5, rotation_churn_max=1))
    assert capped.admitted == ("r:0",) and capped.rule_params["rotation_churn_max"] == 1
    stopped = rotate_info("r000070", active, snapshot(active, -1.0), {}, reserve({"r:0": 0.5}),
                          info_params(rotation_active_size=4, rotation_churn_frac=0.5, rotation_churn_max=0))
    assert stopped.admitted == ("r:0",) and stopped.rule_params["rotation_churn_max"] == 0


def test_permuted_retired_archive_and_no_improvement_are_deterministic():
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(3)}
    archive = [{"id": f"r:{i}", "round_id": "r000060", "reason": "excess", "event_seq": i,
                "irt_row": [0.0, 0.25, 4], "head_rollouts": 0, "head_passes": 0} for i in range(3)]
    one = rotate_info("r000070", active, snapshot(active), {}, reserve(), info_params(rotation_active_size=3), retired_archive=archive)
    two = rotate_info("r000070", reversed(list(active)), snapshot(active), {}, reserve(), info_params(rotation_active_size=3), retired_archive=reversed(archive))
    assert one == two and not one.admitted and not one.retired


def test_info_controller_lifecycle_restores_table_and_replays(tmp_path):
    from test_rotation_controller import rotated_world
    from sn120_v2.state import load
    from sn120_v2.pool_runtime import PoolRuntime

    world, _ = rotated_world(tmp_path, 20, rotation_rule="info")
    first = world.round()
    assert world.store.exists(layout.pool_ranking(first.round_id))
    assert replay_rotation(world.store, first.round_id)["matches"]
    world.register(2, 0.0)
    second = world.round()
    assert second.rotation["size"] == world.params.rotation_active_size
    assert replay_rotation(world.store, second.round_id)["matches"]
    assert load(world.root / "journal.jsonl").rotation_state_sha == world.controller.state.rotation_state_sha
    restored = PoolRuntime(world.store, world.params, world.controller.state, root=tmp_path / "reserve", initial_ids=world.pool)
    assert restored.previous is not None and restored.active_ids == world.controller._rotation.active_ids
    assert set(restored.previous.ledger).isdisjoint(restored.previous.retired_ledger), (
        "a returned task keeps its ledger only in the active pool until it retires again"
    )


def test_info_rotation_no_decision_advances_live_boundary_like_replay(tmp_path):
    from test_rotation_controller import rotated_world
    from sn120_v2.journal import read_journal
    from sn120_v2.controller import stored_irt_snapshot
    from sn120_v2.pool_runtime import PoolRuntime
    from sn120_v2.state import load

    world, pool = rotated_world(tmp_path, 20, rotation_rule="info", rotation_active_size=4, rotation_churn_frac=0.25)
    first = world.round()
    assert first is not None and first.outcome == "null"
    first_archive = Rotation.from_json(world.store.get(layout.pool_rotation(first.round_id))).retired_archive
    world.register(2, 0.0)
    assert world.controller.open_round(world.clock(), "no-decision-entropy")
    opened = world.controller._rotation
    while world.controller.state.round is not None:
        lease = next(lease for lease in world.controller.state.round.leases if lease.status == "open")
        world.controller.fail_lease(lease.lease_id, "offline reproduction", world.clock())

    closed = world.controller.state.rounds[-1]
    assert closed.outcome == "no_decision" and closed.rotation is not None
    assert opened.retired_archive
    assert opened.retired_archive != first_archive
    assert json.loads(world.store.get(layout.RETIRED_POOL)) == list(opened.retired_archive)
    close = read_journal(world.root / "journal.jsonl")[-1]
    assert close.kind == "round_closed" and "rotation_state_sha" not in close.data
    assert world.controller.state.rotation_state_sha == opened.digest()
    replayed_closed = load(world.root / "journal.jsonl")
    source = stored_irt_snapshot(world.store, world.controller.state)
    fresh = PoolRuntime(world.store, world.params, replayed_closed, root=tmp_path / "reserve", initial_ids=world.pool)
    next_round = f"r{world.controller.state.round_seq + 1:06d}"
    expected, _ = fresh.open(next_round, source, world.controller.state)

    # This is a second open by the same long-lived controller, not the restart
    world.now += world.params.apply_retry_s
    assert world.controller.open_round(world.clock(), "after-no-decision-entropy")
    third = world.controller._rotation
    assert pool.admitted_counts == fresh.admitted_counts
    assert third == expected
    assert third.prev_state_sha == opened.digest()
    assert third.retired_archive[:len(opened.retired_archive)] == opened.retired_archive
    stored_archive = json.loads(world.store.get(layout.RETIRED_POOL))
    assert stored_archive == list(third.retired_archive)
    assert len(stored_archive) > len(opened.retired_archive)

    while world.controller.state.round is not None:
        lease = next(lease for lease in world.controller.state.round.leases if lease.status == "open")
        world.controller.fail_lease(lease.lease_id, "offline reproduction", world.clock())
    replayed_closed = load(world.root / "journal.jsonl")
    assert pool.admitted_counts == PoolRuntime(
        world.store, world.params, replayed_closed, root=tmp_path / "reserve", initial_ids=world.pool
    ).admitted_counts


def test_info_rotation_abandoned_advances_live_boundary(tmp_path):
    from test_rotation_controller import incident_world
    from sn120_v2.journal import read_journal
    from sn120_v2.pool_runtime import PoolRuntime

    world = incident_world(tmp_path, "info")
    pool = world.controller.pool
    world.controller.set_params(dataclasses.replace(world.params, rotation_active_size=8, min_scored_tasks=2), world.clock())
    world.register(2, 0.5)
    world.register(3, 0.0)
    world.register(4, -0.25)
    ticket, = world.controller.open_round(world.clock(), "abandoned-entropy")
    opened = world.controller._rotation
    result = world.runner.run(ticket)
    artifact = next(entry.lora for entry in result.entries if entry.lora is not None)
    world.store.put(f"{artifact.prefix}/adapter_model.bin", b"corrupted")
    world.controller.submit_result(result, world.clock())
    assert world.controller.state.rounds[-1].outcome == "abandoned", [
        (entry.uid, entry.status, entry.reason) for entry in world.controller.state.rounds[-1].decision.entries
    ]
    assert world.controller._rotation == opened and pool.previous == opened
    assert "rotation_state_sha" not in read_journal(world.root / "journal.jsonl")[-1].data

    world.register(5, 0.0)
    assert world.controller.open_round(world.clock(), "after-abandoned-entropy")
    third = world.controller._rotation
    assert third.prev_state_sha == opened.digest()
    assert third.retired_archive[:len(opened.retired_archive)] == opened.retired_archive
    assert pool.admitted_counts == PoolRuntime(
        world.store, world.params, world.controller.state, root=tmp_path / "reserve", initial_ids=world.pool
    ).admitted_counts


def test_info_rotation_open_no_decision_then_next_open_restarts(tmp_path):
    from test_rotation_controller import rotated_world
    from sn120_v2.controller import Controller, stored_irt_snapshot
    from sn120_v2.journal import Journal
    from sn120_v2.pool_runtime import PoolRuntime

    world, pool = rotated_world(tmp_path, 20, rotation_rule="info", rotation_active_size=4, rotation_churn_frac=0.25)
    # Explicit priors make one reserve row strictly outrank one active task.
    from test_rotation_controller import make_reserve
    import shutil
    shutil.rmtree(tmp_path / "reserve")
    make_reserve(tmp_path / "reserve", 20)
    pool.candidate_meta = {task: dict(meta, prior_p=1.0) for task, meta in pool.candidate_meta.items()}
    first = world.round()
    assert first is not None and first.outcome == "null"
    source = json.loads(world.store.get(layout.irt_snapshot(world.controller.state.snapshot_id)))
    world.register(2, 0.0)
    tickets = world.controller.open_round(world.clock(), "no-decision-entropy")
    while world.controller.state.round is not None:
        lease = next(lease for lease in world.controller.state.round.leases if lease.status == "open")
        world.controller.fail_lease(lease.lease_id, "offline reproduction", world.clock())
    previous = world.controller.state.rounds[-1]
    assert previous.outcome == "no_decision" and previous.rotation is not None
    assert previous.rotation["active_sha"] != first.rotation["active_sha"]
    assert set(source["tasks"]) != set(world.controller._rotation.active_ids)

    # A fresh journal, runtime and sampler are the production restart path.
    journal = Journal(world.root / "journal.jsonl", "no-decision-restart", world.clock())
    pool = PoolRuntime(world.store, world.params, world.controller.state,
                       root=tmp_path / "reserve", initial_ids=world.pool)
    sampler = IRTSampler(pool.active_ids, world.params, snapshot=source)
    controller = Controller(journal, world.store, world.admission, sampler, world.params, pool=pool)
    world.now += world.params.apply_retry_s
    tickets = controller.open_round(world.clock(), "after-no-decision-entropy")
    assert len(controller._rotation.active_ids) == 4
    assert len(tickets) == 1 and len(tickets[0].draw.task_ids) == min(world.params.n_tasks, 4)
    active = controller._rotation.active_sha
    draw = tickets[0].draw
    for task, row in source["tasks"].items():
        if task in controller._rotation.active_ids:
            assert controller.sampler.public_snapshot()["tasks"][task][:3] == row[:3]
    for task in controller._rotation.admitted:
        if task not in source["tasks"]:
            assert controller.sampler.public_snapshot()["tasks"][task][2] == 0

    pool = PoolRuntime(world.store, world.params, controller.state,
                       root=tmp_path / "reserve", initial_ids=world.pool)
    sampler = IRTSampler(pool.active_ids, world.params, snapshot=stored_irt_snapshot(world.store, controller.state))
    controller = Controller(Journal(world.root / "journal.jsonl", "open-restart", world.clock()),
                            world.store, world.admission, sampler, world.params, pool=pool)
    restarted = controller.tickets()
    assert len(restarted) == 1
    assert restarted[0].draw == draw and pool.active_rotation.active_sha == active
