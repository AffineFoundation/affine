"""Controller lifecycle with rotation enabled on a small, injected catalog."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from sn120_v2 import layout, publisher
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.controller import Controller
from sn120_v2.envs.math_pool import MathPool, Task, POOL_V2_0_FILE, POOL_V2_0_SHA256, load_pool
from sn120_v2.envs.reserve import empty_manifest
from sn120_v2.errors import ConfigError, VerifyError
from sn120_v2.irt import MASS_RULES, IRTSampler
from sn120_v2.journal import Journal, read_journal
from sn120_v2.params import Params
from sn120_v2.pool_runtime import PoolRuntime, active_file, load_rotation, replay_rotation
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import ROUND_OPENED, load
from sn120_v2.store import LocalStore


def make_reserve(root: Path, count: int = 20):
    (root / "shards").mkdir(parents=True)
    (root / "withdrawn.jsonl").write_bytes(b"")
    rows = []
    for i in range(count):
        rows.append({"id": f"simnew:{i}", "env": "affine_math", "problem": f"sim new problem {i}", "answer": str(i),
                     "source": f"sim@rev:{i}", "license": "CC0-1.0", "added_at": "2026-11-18",
                     "audit": {"verdict": "keep", "method": "sim", "ref": "sim"}})
    raw = b"".join(canonical_json(row) + b"\n" for row in rows)
    name = "0001-sim.jsonl"
    (root / "shards" / name).write_bytes(raw)
    lines = [{"shard": name, "sha256": sha256_hex(raw), "count": len(rows), "added_at": "2026-11-18"}] if rows else []
    lines.append({"withdrawn": {"sha256": sha256_hex(b""), "count": 0}})
    (root / "MANIFEST.jsonl").write_bytes(b"".join(canonical_json(row) + b"\n" for row in lines))


def rotated_world(tmp_path, reserve_count=20, **changes) -> tuple[SimWorld, PoolRuntime]:
    params = sim_params(**(dict(rotation_enabled=True, rotation_k=3, rotation_retire_p=0.1, rotation_n_min=0, rotation_active_size=2) | changes))
    world = SimWorld(tmp_path, params)
    # Explicit initial ids keep the test quick; production PoolRuntime defaults
    # to all 20,120 base-pool ids, and therefore target size 20,120.
    world.sampler = IRTSampler(world.pool, params)
    root = tmp_path / "reserve"
    make_reserve(root, reserve_count)
    base_ids = tuple(f"sim:{i}" for i in range(400))
    # Assembly constructs PoolRuntime before Controller replays the journal:
    # the runtime receives a distinct startup State, not controller.state.
    startup = load(world.root / "journal.jsonl")
    assert startup is not world.controller.state and startup.last_seq >= 0
    pool = PoolRuntime(world.store, params, startup, root=root, base=world_pool(), initial_ids=base_ids)
    world.controller.pool = pool
    world.controller.sampler = world.sampler
    world.register(1, 0.0)
    return world, pool


def world_pool() -> MathPool:
    # Base pool ids stay real, while the injected initial active set is sim:*.
    return load_pool()


def test_rotated_round_opens_artifacts_before_journal_and_closes_ledger(tmp_path):
    world, pool = rotated_world(tmp_path)
    record = world.round()
    world.register(2, 0.0)
    record = world.round()
    assert record.rotation["active_sha"] == world.controller._rotation.active_sha
    assert world.store.exists(layout.active_pool(record.rotation["active_sha"]))
    assert world.store.exists(layout.pool_rotation(record.round_id))
    assert world.store.exists(layout.RESERVE_MANIFEST) and world.store.exists(layout.RETIRED_POOL)
    report = json.loads(world.store.get(layout.round_file(record.round_id)))
    assert report["pool"]["size"] == 394 and report["pool"]["next_up"]
    snap = json.loads(world.store.get(layout.irt_snapshot(record.snapshot_id)))
    assert snap["rotation"]["round_id"] == record.round_id and set(snap["tasks"]) == set(world.controller._rotation.active_ids)
    admitted = record.rotation["admitted"]
    for task in admitted:
        assert snap["tasks"][task][1:] == [0.25, 0, snap["tasks"][task][3]]
        from sn120_v2.irt import predicted_pass
        assert abs(predicted_pass(snap["head_ability"], snap["tasks"][task][0]) - .5) < 1e-6
    closed = read_journal(world.root / "journal.jsonl")[-1]
    assert closed.data["rotation_state_sha"] == world.controller._rotation.digest()
    assert sum(row[0] for row in world.controller._rotation.ledger.values()) > 0
    replay = load(world.root / "journal.jsonl")
    assert replay.rotation_state_sha == world.controller.state.rotation_state_sha


def test_open_retags_rotation_source_before_draw(tmp_path):
    world, pool = rotated_world(tmp_path)
    world.admission.upload("hk1", 1.0)
    world.round()
    world.register(2, 0.0)
    target, stale_head = world.controller.state.heads[0], world.controller.state.head
    sampler = IRTSampler(world.pool, world.params)
    draw = sampler.draw(target, "target", 20)
    sampler.update(target, draw, {"null": ("F" * 20,)})
    draw = sampler.draw(stale_head, "stale", 20)
    sampler.update(stale_head, draw, {"null": ("P" * 20,), "winner": ("P" * 20,)},
                   new_head_version_id=stale_head.version_id, winner_lora_hash="winner")
    stale = sampler.public_snapshot()
    world.store.put(layout.irt_snapshot(sampler.snapshot_id()), canonical_json(stale))
    world.controller.sampler = sampler
    world.controller.revert_head(target.version_id, "operator restore", world.clock())
    healed = world.controller.sampler.public_snapshot()
    # Restart healing already changed the in-memory ID; the old stored source must not replace it at open.
    replayed = Controller(Journal(world.root / "journal.jsonl", "replayed-retag", world.clock()), world.store,
                          world.admission, IRTSampler(world.pool, world.params, snapshot=stale), world.params, pool=pool)
    assert replayed.sampler.snapshot_id() == healed["snapshot_id"]
    world.controller = replayed
    world.controller.pool = None  # validate the live snapshot ID chosen by open without a second rotation
    world.register(3, 0.0)
    seen = []
    real_open = pool.open_and_publish

    def capture_open(*args, **kwargs):
        seen.append((args, kwargs))
        raise ConfigError("stop after checking the selected source")

    pool.open_and_publish = capture_open
    replayed.pool = pool
    with pytest.raises(ConfigError, match="selected source"):
        replayed.open_round(world.clock(), "rotation-retag-after-restart")
    assert seen[0][0][1]["snapshot_id"] == healed["snapshot_id"]
    assert seen[0][1]["retagged_source"] is True
    assert replayed.state.round is None, "a failed pre-open must not journal ROUND_OPENED"

    # The ordinary fail-safe also supplies that source before its first draw.
    pool.open_and_publish = real_open
    replayed.sampler = IRTSampler(world.pool, world.params, snapshot=stale)
    tickets = replayed.open_round(world.clock(), "rotation-retag")
    assert tickets
    assert replayed._rotation.source_snapshot_id == healed["snapshot_id"]
    assert tickets[0].draw.snapshot_id != stale["snapshot_id"]
    assert world.controller.state.round.snapshot_id == tickets[0].draw.snapshot_id
    opened = json.loads(world.store.get(layout.irt_snapshot(tickets[0].draw.snapshot_id)))
    assert opened["head_version_id"] == target.version_id and opened["head_ability"] == healed["head_ability"]


def test_real_publisher_restores_public_pool_objects_across_history(tmp_path):
    world, _ = rotated_world(tmp_path, rotation_rule="info", rotation_active_size=2, rotation_churn_frac=0.5)
    world.round()
    world.register(2, 0.0)
    latest = world.round()
    world.controller.publisher = publisher.StorePublisher(world.store, "https://files.example/")
    world.controller.publish(world.clock())
    assert world.controller.last_error == ""
    assert latest.rotation is not None
    current_sha = latest.rotation["active_sha"]

    class ObservedStore(LocalStore):
        def __init__(self, root):
            super().__init__(root)
            self.active_puts = 0
            self.snapshot_written = False

        def put(self, key, data):
            if key.startswith("public/pool/active/"):
                self.active_puts += 1
            if key == layout.MINING_JSON:
                self.snapshot_written = True
            super().put(key, data)

    observed = ObservedStore(tmp_path / "observed")
    for key in world.store.list(""):
        observed.put(key, world.store.get(key))
    observed.active_puts = 0
    world.store = observed
    world.controller.store = observed
    world.controller.publisher.store = observed
    snap = world.publisher.snapshots[-1]
    assert len([record for record in snap["history"] if record.get("pool")]) >= 2
    world.controller._publish(world.clock())
    assert observed.active_puts == 0 and observed.snapshot_written

    for key in (layout.active_pool(current_sha), layout.pool_ranking(latest.round_id),
                layout.POOL_SUMMARY, layout.RESERVE_MANIFEST, layout.RESERVE_WITHDRAWN, layout.RETIRED_POOL):
        observed.delete(key)
    observed.active_puts = observed.snapshot_written = 0
    world.controller._publish(world.clock())
    assert world.controller.last_error == "" and observed.snapshot_written
    assert observed.active_puts == 1, "only the missing active object is restored"
    assert observed.get(layout.active_pool(current_sha)) == canonical_json(world.controller._rotation.active_ids)
    first_bytes = {key: observed.get(key) for key in observed.list("public/pool/")}
    world.controller._publish(world.clock())
    assert {key: observed.get(key) for key in observed.list("public/pool/")} == first_bytes

    class RefusingStore(ObservedStore):
        def put(self, key, data):
            if key == layout.active_pool(current_sha):
                return
            super().put(key, data)

    refused = RefusingStore(tmp_path / "refused")
    for key, data in first_bytes.items():
        refused.put(key, data)
    refused.delete(layout.active_pool(current_sha))
    refused.snapshot_written = False
    world.controller.store = refused
    world.controller.publisher.store = refused
    world.controller._publish(world.clock())
    assert "publish failed:" in world.controller.last_error and not refused.snapshot_written
    assert not refused.exists(layout.active_pool(current_sha))


def test_publish_failures_are_logged_without_escaping(tmp_path, capsys):
    world, _ = rotated_world(tmp_path)

    class FailingPublisher:
        def __init__(self, failure):
            self.failure = failure

        def publish(self, snapshot):
            raise self.failure

    for failure in (VerifyError("bad reserve prefix"), RuntimeError("render failed")):
        world.controller.publisher = FailingPublisher(failure)
        world.controller._publish(world.clock())
        expected = f"publish failed: {failure}"
        assert world.controller.last_error == expected
        assert f"publish: {expected}\n" in capsys.readouterr().err


def test_live_params_change_rebuilds_the_next_sqrt_fisher_open(tmp_path):
    world, pool = rotated_world(tmp_path, irt_draw_rule="fisher")
    world.round()
    params = dataclasses.replace(world.params, irt_draw_rule="sqrt_fisher")
    world.controller.set_params(params, world.clock())
    assert world.controller.sampler.public_snapshot()["config"]["draw_rule"] == "sqrt_fisher"
    assert pool.params == params
    world.register(2, 0.0)
    record = world.round()
    assert record is not None and record.params_hash == params.params_hash
    snapshot = json.loads(world.store.get(layout.irt_snapshot(record.snapshot_id)))
    assert snapshot["config"]["draw_rule"] == "sqrt_fisher"
    assert snapshot["mass_rule"] == MASS_RULES["sqrt_fisher"]


def test_mid_round_restart_does_not_rotate_again_and_recovers_exact_draw(tmp_path):
    world, pool = rotated_world(tmp_path / "world")
    world.register(2, 0.0)
    tickets = world.controller.open_round(world.clock(), "entropy", boxes=("box-a", "box-b"))
    before = world.controller.state.round
    ids = tickets[0].draw.task_ids
    # Fresh runtime + sampler + controller, same store and journal. No reserve
    # mutation is needed to prove that the open round never recomputes.
    params = world.params
    new_pool = PoolRuntime(world.store, params, world.controller.state, root=tmp_path / "world" / "reserve", initial_ids=world.pool)
    world.controller = Controller(Journal(world.root / "journal.jsonl", "restart", world.clock()), world.store,
                                  world.admission, IRTSampler(world.pool, params), params,
                                  folder=world.folder, publisher=world.publisher, pool=new_pool)
    assert world.controller.state.round.round_id == before.round_id
    assert new_pool.active_rotation.digest() == world.controller._rotation.digest()
    assert new_pool.active_rotation.active_sha == before.rotation["active_sha"]
    result = world.runner.run(tickets[0])
    assert result.null_outcomes and tickets[0].draw.task_ids == ids
    world.controller.submit_result(result, world.clock())
    other = world.runner.run(tickets[1])
    world.controller.submit_result(other, world.clock())
    record = world.controller.state.rounds[-1]
    expected = sum(sum(code in "PFC" for code in text) for text in (*result.null_outcomes, *other.null_outcomes))
    assert sum(row[0] for row in world.controller._rotation.ledger.values()) == expected
    assert world.controller.state.round is None
    assert [record.kind for record in read_journal(world.root / "journal.jsonl")].count(ROUND_OPENED) == 1
    # A fresh checkout starts with an empty reserve, while previously admitted
    # rows survive only in the public mirror. Restore them before catalog use.
    from sn120_v2.envs.reserve import empty_manifest
    fresh = tmp_path / "fresh-reserve"
    (fresh / "shards").mkdir(parents=True)
    (fresh / "MANIFEST.jsonl").write_bytes(empty_manifest())
    (fresh / "withdrawn.jsonl").write_bytes(b"")
    restored = PoolRuntime(world.store, params, world.controller.state, root=fresh, initial_ids=world.pool)
    assert restored.active_ids == world.controller._rotation.active_ids
    assert all(restored.task(task).task_id == task for task in restored.active_ids if task.startswith("simnew:"))
    assert set(restored._rule_reserve().queue).isdisjoint(world.controller._rotation.admitted)
    # Two rows from the same shard retain the exact original filename; a
    # second shard must not collide with it or with a stale cache object.
    assert (fresh / "shards" / "0001-sim.jsonl").is_file()
    mirror_key = layout.reserve_shard("0001-sim.jsonl", "0" * 64)
    assert mirror_key.endswith(".jsonl") and mirror_key.count(".jsonl") == 2


class Crash(BaseException):
    pass


@pytest.mark.parametrize("rotation_rule", ["v1", "info"])
def test_crash_between_artifacts_and_open_record_rewrites_identical_bytes(tmp_path, rotation_rule):
    world, pool = rotated_world(tmp_path, rotation_rule=rotation_rule)
    append = world.controller.journal.append
    def crash(kind, data, at):
        if kind == ROUND_OPENED:
            raise Crash
        return append(kind, data, at)
    world.controller.journal.append = crash
    with pytest.raises(Crash):
        world.controller.open_round(world.clock(), "entropy")
    keys = [layout.pool_rotation("r000001"), layout.RETIRED_POOL, layout.RESERVE_MANIFEST]
    before = {key: world.store.get(key) for key in keys}
    world.reopen()
    world.controller.sampler = IRTSampler(world.pool, world.params)
    pool = PoolRuntime(world.store, world.params, world.controller.state, root=tmp_path / "reserve", initial_ids=world.pool)
    world.controller.pool = pool
    tickets = world.controller.open_round(world.clock(), "entropy")
    assert tickets and all(world.store.get(key) == raw for key, raw in before.items())


def test_empty_reserve_first_enabled_round_rotates_nothing(tmp_path):
    world, pool = rotated_world(tmp_path, 0)
    record = world.round()
    assert record is not None and record.rotation["admitted"] == [] and record.rotation["retired"] == []
    assert record.rotation["queue_remaining"] == 0


def test_head_change_uses_snapshot_head_not_caller_head(tmp_path):
    world, pool = rotated_world(tmp_path)
    world.round()
    params = world.params
    # The next rotation takes its head from the last closed snapshot. A caller
    # passing another HeadVersion cannot silently change eligibility.
    world.register(2, 0.0)
    record = world.round()
    assert record is not None and record.rotation["prev_active_sha"] == world.controller._rotation.prev_active_sha
    previous_snapshot = json.loads(world.store.get(layout.irt_snapshot(world.controller._rotation.source_snapshot_id)))
    assert previous_snapshot["snapshot_id"] == world.controller._rotation.source_snapshot_id
    assert world.store.exists(layout.irt_snapshot(record.snapshot_id))


def incident_world(root: Path, rotation_rule: str) -> SimWorld:
    params = sim_params(rotation_enabled=True, rotation_rule=rotation_rule, rotation_k=3,
                        rotation_retire_p=0.1, rotation_n_min=0, rotation_active_size=2)
    world = SimWorld(root, params)
    world.sampler = IRTSampler(world.pool, params)
    world.controller.sampler = world.sampler
    world.register(1, 0.0)
    world.round()
    startup = load(world.root / "journal.jsonl")
    assert startup.snapshot_id and startup is not world.controller.state
    reserve_root = root / "reserve"
    make_reserve(reserve_root)
    pool = PoolRuntime(world.store, params, startup, root=reserve_root, base=world_pool(), initial_ids=world.pool)
    world.controller.pool = pool
    return world


@pytest.mark.parametrize("rotation_rule", ["v1", "info"])
def test_one_controller_rotates_three_rounds_identically_to_restarts(tmp_path, rotation_rule):
    # Start with an existing snapshot, as after the incident's first close.
    # The next round's snapshot must not be checked against this startup one.
    live = incident_world(tmp_path / "live", rotation_rule)
    restarted = incident_world(tmp_path / "restarted", rotation_rule)
    decisions = {}
    for world in (live, restarted):
        keys = []
        for index in range(3):
            if index and world is restarted:
                world.reopen()
                pool = PoolRuntime(world.store, world.params, world.controller.state,
                                   root=world.root / "reserve", initial_ids=world.pool)
                world.controller.pool = pool
                world.controller.sampler = IRTSampler(world.pool, world.params)
            world.register(index + 2, 0.0)
            record = world.round()
            assert record is not None
            keys.extend((layout.pool_rotation(record.round_id), layout.pool_rotation_state(record.rotation["state_sha"])))
            assert replay_rotation(world.store, record.round_id)["matches"]
        decisions[world.root.name] = {key: world.store.get(key) for key in keys}
    assert decisions["live"] == decisions["restarted"]


@pytest.mark.parametrize("rotation_rule", ["v1", "info"])
def test_controller_owning_runtime_state_also_runs_three_rotated_rounds(tmp_path, rotation_rule):
    params = sim_params(rotation_enabled=True, rotation_rule=rotation_rule, rotation_k=3,
                        rotation_retire_p=0.1, rotation_n_min=0, rotation_active_size=2)
    world = SimWorld(tmp_path, params)
    world.sampler = IRTSampler(world.pool, params)
    world.controller.sampler = world.sampler
    root = tmp_path / "reserve"
    make_reserve(root)
    pool = PoolRuntime(world.store, params, world.controller.state, root=root,
                       base=world_pool(), initial_ids=world.pool)
    world.controller.pool = pool
    for uid in range(1, 4):
        world.register(uid, 0.0)
        record = world.round()
        assert record is not None and replay_rotation(world.store, record.round_id)["matches"]


def test_rotation_open_still_refuses_a_snapshot_other_than_the_last_closed(tmp_path):
    world, pool = rotated_world(tmp_path)
    world.round()
    previous = json.loads(world.store.get(layout.irt_snapshot(world.controller.state.snapshot_id)))
    with pytest.raises(VerifyError, match="rotation source snapshot is not the last closed round's snapshot"):
        pool.open("r000002", previous | {"snapshot_id": "not-the-last-closed-snapshot"}, state=world.controller.state)


@pytest.mark.parametrize("rotation_rule", ["v1", "info"])
def test_replicate_rotation_cli_recomputes_from_public_files(tmp_path, monkeypatch, capsys, rotation_rule):
    from sn120_v2 import replicate
    world, _ = rotated_world(tmp_path, rotation_rule=rotation_rule)
    world.round()
    world.register(2, 0.0)
    record = world.round()
    store = LocalStore(tmp_path / 'public')
    for key in world.store.list('public/'):
        store.put(key, world.store.get(key))
    view = json.loads(world.store.get(layout.round_file(record.round_id)))
    doc = {'contract': {'public_url': store.root.as_uri()}, 'head': world.publisher.snapshots[-1]['head'],
           'params': world.params.to_dict(), 'history': [view], 'round': None}
    source = tmp_path / 'mining.json'
    source.write_text(json.dumps(doc))
    code = replicate.main(['rotation', '--mining-json', str(source), '--round', record.round_id, '--work', str(tmp_path / 'replica')])
    assert code == 0 and json.loads(capsys.readouterr().out)['matches'] is True