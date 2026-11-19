"""Measured IRT rows survive rotation toggles and active-pool changes."""
from __future__ import annotations

import json
from pathlib import Path

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json
from sn120_v2.controller import _recover_irt_rows, stored_irt_snapshot
from sn120_v2.irt import IRTSampler
from sn120_v2.params import Params
from sn120_v2.pool_runtime import load_rotation, replay_rotation
from sn120_v2.rotation import Rotation
from sn120_v2.state import RoundRecord, State
from sn120_v2.store import LocalStore

from test_rotation_controller import rotated_world

FIXTURES = Path(__file__).parent / "data" / "irt_dormant"


def test_controller_on_off_on_keeps_admitted_and_retired_measurements(tmp_path):
    world, runtime = rotated_world(tmp_path, 20, rotation_active_size=20)
    runtime.initial_ids = runtime.initial_ids[:20]
    runtime._active_ids = runtime.initial_ids
    first = world.round()
    assert first is not None
    first_admits = list(first.rotation["admitted"])
    assert first.rotation["retired"] and first_admits
    world.register(2, 0.0)
    second = world.round()
    assert second is not None
    rotated = json.loads(world.store.get(layout.irt_snapshot(second.snapshot_id)))
    retired = [row["id"] for row in second.rotation["retired"]]
    admitted = [task for task in first_admits if task in rotated["tasks"] and rotated["tasks"][task][2] > 0]
    assert retired and admitted
    expected = {task: rotated["tasks"][task][:3] for task in admitted}
    expected.update({task: rotated["dormant"][task] for task in retired})

    # Disabling rotation rebuilds the base pool. Retirees return with their
    # measured rows; measured admits leave the table but remain dormant.
    off_snapshot = IRTSampler(runtime.initial_ids, world.params, snapshot=rotated).public_snapshot()
    for task, row in expected.items():
        actual = off_snapshot["tasks"][task][:3] if task in off_snapshot["tasks"] else off_snapshot["dormant"][task]
        assert actual == row
    assert off_snapshot["mass_total"] == sum(row[3] for row in off_snapshot["tasks"].values())

    # A direct rebuild of the previous active set is the same transition that
    # PoolRuntime performs on re-enable; admitted-at-head priors cannot win.
    returned = IRTSampler(runtime.previous.active_ids, world.params, snapshot=off_snapshot,
                          initial={task: off_snapshot["head_ability"] for task in admitted},
                          rotation={"round_id": "r000003"}).public_snapshot()
    for task, row in expected.items():
        actual = returned["tasks"][task][:3] if task in returned["tasks"] else returned["dormant"][task]
        assert actual == row



def test_public_r000066_and_r000069_decisions_still_replay_and_recover_rows():
    # These projections contain the exact published retired rows, source
    # snapshot IDs and rule parameters, not invented IRT measurements.
    for rid in ("r000066", "r000069"):
        projected = json.loads((FIXTURES / f"{rid}.json").read_bytes())
        assert projected["round_id"] == rid
        assert len(projected["retired"]) == 32
        assert all(row["irt_row"][2] > 0 for row in projected["retired"])
        if rid == "r000066":
            assert projected["retired"][0] == {
                "head_passes": 0, "head_rollouts": 0, "id": "dapo:10781",
                "irt_row": [-1.58552, 0.867357, 10],
                "p_or_rate": 0.9822732721112238, "reason": "high", "rounds_seen": 10,
            }
    # The real full public decisions are replayed with replay_rotation. Here
    # check that dormant rebuilding does not alter the pure decision itself.
    source = json.loads((FIXTURES / "irt-r000066-source.json").read_bytes())
    assert source["snapshot_id"] == "irt-c2afc207598fe7734c3bf33cd7034784f99b81f5f6a8c69a0cb509b68188197f"
    assert source["tasks"]["dapo:10781"][:3] == [-1.58552, 0.867357, 10]


def test_public_r000066_admits_keep_their_later_measurements_on_return():
    rotation = json.loads((FIXTURES / "r000066.json").read_bytes())
    measured = json.loads((FIXTURES / "irt-r000072-measured-admits.json").read_bytes())
    tasks = {task: row for task, row in measured["tasks"].items() if task in rotation["admitted"]}
    assert len(tasks) == 9 and tasks["nmv2:67"][:3] == [2.266824, 1.526131, 3]
    # Seed the exact public rows onto a fresh table of the incident admits.
    before = IRTSampler(list(tasks), Params(), initial=tasks).public_snapshot()
    off = IRTSampler(["base:0"], Params(), snapshot=before).public_snapshot()
    assert off["dormant"] == {task: row[:3] for task, row in tasks.items()}
    returned = IRTSampler(list(tasks), Params(), snapshot=off,
                          initial={task: 0.0 for task in tasks}).public_snapshot()
    assert all(returned["tasks"][task][:3] == row[:3] for task, row in tasks.items())


def test_v2_public_decision_replays_with_dormant_support():
    store = LocalStore(FIXTURES / "v2_public")
    report = replay_rotation(store, "r000070")
    assert report["matches"] and report["actual"] == report["expected"]
    source = json.loads(store.get(layout.irt_snapshot(report["expected"]["source_snapshot_id"])))
    rebuilt = IRTSampler(report["expected"]["active_ids"], Params(), snapshot=source)
    assert rebuilt.public_snapshot()["dormant"] == {"a:0": [0.0, 0.25, 4], "a:10": [0.0, 0.25, 4]}
    # A returning v2 task keeps its exact archived row. Dormant takes priority
    # only when it has the newer measurement.
    returning = IRTSampler(["a:0", "a:1"], Params(), snapshot=rebuilt.public_snapshot(),
                           initial={"a:0": [-0.123456, 0.75, 17]}).public_snapshot()
    assert returning["tasks"]["a:0"][:3] == [0.0, 0.25, 4]
    assert returning["dormant"]["a:10"] == [0.0, 0.25, 4]
    archive = next(row for row in report["actual"]["retired_archive"] if row["id"] == "a:0")
    assert archive["irt_row"] == [0.0, 0.25, 4]


def test_legacy_recovery_uses_newest_public_measurement_and_retirement_archive(tmp_path):
    from sn120_v2.rotation import Reserve, Retired, active_sha

    store = LocalStore(tmp_path)
    older = IRTSampler(["retired", "kept"], Params()).public_snapshot()
    newer = IRTSampler(["retired", "kept"], Params(), snapshot=older,
                       initial={"retired": [-0.3725, 0.33816, 1]}).public_snapshot()
    # Construct the historical artifact with its own measured row. `initial`
    # cannot replace an active task's row, by design.
    from sn120_v2.irt import snapshot_digest
    newer["tasks"]["retired"] = [-0.3725, 0.33816, 1, 1]
    newer["snapshot_id"] = snapshot_digest(newer)
    current = IRTSampler(["retired", "kept", "base"], Params(), snapshot=newer).public_snapshot()
    for snapshot in (older, newer, current):
        store.put(layout.irt_snapshot(snapshot["snapshot_id"]), canonical_json(snapshot))
    rotation = Rotation(
        round_id="r000066", active_ids=("kept",), active_sha=active_sha(["kept"]),
        prev_active_sha=active_sha(["kept"]), admitted=(), reserve_manifest_sha="0" * 64,
        queue_remaining=0, next_up=(), reserve_history=Reserve("0" * 64).history(), ledger={},
        retired=(Retired("retired", "high", rounds_seen=10, irt_row=(-1.58552, 0.867357, 10)),),
        retired_archive=(),
    )
    store.put(layout.pool_rotation_state(rotation.digest()), rotation.to_json())
    records = [RoundRecord(rotation.round_id if index == 1 else f"r{index:06d}", "head", "params", "recipe", "seed", snapshot["snapshot_id"], 1, 0, (),
                           rotation=rotation.journal_block() | {"state_sha": rotation.digest(), "size": 1} if index == 1 else None)
               for index, snapshot in enumerate((older, newer, current), 1)]
    state = State(snapshot_id=current["snapshot_id"], rounds=records)
    recovered = _recover_irt_rows(store, state, current)
    assert recovered["retired"] == [-0.3725, 0.33816, 1]
    # Existing dormant state does not suppress recovery of a different lost row.
    rebuilt = IRTSampler(["kept", "new"], Params(), snapshot=current, recover=recovered).public_snapshot()
    assert rebuilt["dormant"]["retired"] == [-0.3725, 0.33816, 1]


def test_legacy_recovery_unchanged_pool_and_partial_dormant_state(tmp_path):
    from sn120_v2.irt import snapshot_digest
    from sn120_v2.rotation import Reserve, active_sha

    store = LocalStore(tmp_path)
    historical = IRTSampler(["lost", "other"], Params()).public_snapshot()
    historical["tasks"]["lost"] = [-1.58552, 0.867357, 10, 1]
    historical["snapshot_id"] = snapshot_digest(historical)
    current = IRTSampler(["lost", "other", "new"], Params(), snapshot=historical).public_snapshot()
    current["dormant"] = {"elsewhere": [0.4, 0.5, 2]}
    current["snapshot_id"] = snapshot_digest(current)
    store.put(layout.irt_snapshot(historical["snapshot_id"]), canonical_json(historical))
    store.put(layout.irt_snapshot(current["snapshot_id"]), canonical_json(current))
    record = RoundRecord("r000001", "head", "params", "recipe", "seed", historical["snapshot_id"], 1, 0, ())
    state = State(snapshot_id=current["snapshot_id"], rounds=[record])
    recovered = _recover_irt_rows(store, state, current)
    rebuilt = IRTSampler(["lost", "other", "new"], Params(), snapshot=current, recover=recovered).public_snapshot()
    assert rebuilt["tasks"]["lost"][:3] == [-1.58552, 0.867357, 10]
    assert rebuilt["dormant"]["elsewhere"] == [0.4, 0.5, 2]
    # Without migration, an unchanged pool retains its exact source bytes.
    same = IRTSampler(["lost", "other", "new"], Params(), snapshot=current).public_snapshot()
    assert same == current


def test_off_mode_startup_recovers_legacy_rows_only_between_rounds(tmp_path):
    from sn120_v2.irt import snapshot_digest

    store = LocalStore(tmp_path)
    historical = IRTSampler(["lost", "kept", "old"], Params()).public_snapshot()
    historical["tasks"]["lost"] = [-1.58552, 0.867357, 10, 1]
    historical["snapshot_id"] = snapshot_digest(historical)
    current = IRTSampler(["lost", "kept"], Params()).public_snapshot()
    for snapshot in (historical, current):
        store.put(layout.irt_snapshot(snapshot["snapshot_id"]), canonical_json(snapshot))
    record = RoundRecord("r000001", "head", "params", "recipe", "seed", historical["snapshot_id"], 1, 0, ())
    state = State(snapshot_id=current["snapshot_id"], rounds=[record])
    saved = stored_irt_snapshot(store, state, ["lost", "kept", "base"])
    recovery = saved.pop("recover_legacy")
    rebuilt = IRTSampler(["lost", "kept", "base"], Params(), snapshot=saved, recover=recovery).public_snapshot()
    assert rebuilt["tasks"]["lost"][:3] == [-1.58552, 0.867357, 10]
    assert "recover_legacy" not in rebuilt and "recover" not in rebuilt
    # An open round must not change its frozen table on restart.
    state.round = record
    assert stored_irt_snapshot(store, state, ["lost", "kept"]) == current


def test_recovery_validation_and_archived_row_precedence():
    from sn120_v2.irt import snapshot_digest, verify_snapshot
    import pytest
    from sn120_v2.errors import VerifyError

    snapshot = IRTSampler(["task", "other"], Params()).public_snapshot()
    snapshot["tasks"].pop("task")
    snapshot["snapshot_id"] = snapshot_digest(snapshot)
    archived = [-0.123456, 0.75, 17]
    restored = IRTSampler(["task"], Params(), snapshot=snapshot,
                          initial={"task": archived}, recover={"task": [-1.0, 0.5, 4]}).public_snapshot()
    assert restored["tasks"]["task"][:3] == archived
    for bad in ([1e308, 0.5, 1], [0.0, 0.0, 1], [0.0, 0.25, 0]):
        recovered = IRTSampler(["task"], Params(), snapshot=snapshot, recover={"task": bad}).public_snapshot()
        assert recovered["tasks"]["task"][:3] == [0.0, 0.25, 0]
        forged = {**snapshot, "dormant": {"other": bad}}
        forged["snapshot_id"] = snapshot_digest(forged)
        with pytest.raises(VerifyError, match="dormant"):
            verify_snapshot(forged)


def test_corrupt_or_misbound_optional_history_is_skipped(tmp_path):
    from sn120_v2.irt import snapshot_digest

    store = LocalStore(tmp_path)
    historical = IRTSampler(["lost"], Params()).public_snapshot()
    current = IRTSampler(["lost"], Params()).public_snapshot()
    store.put(layout.irt_snapshot(historical["snapshot_id"]), b"{")
    record = RoundRecord("r000001", "head", "params", "recipe", "seed", historical["snapshot_id"], 1, 0, ())
    state = State(snapshot_id=current["snapshot_id"], rounds=[record])
    assert _recover_irt_rows(store, state, current) == {}
    other = IRTSampler(["lost"], Params(), initial={"lost": [-1.0, 0.5, 4]}).public_snapshot()
    store.put(layout.irt_snapshot(historical["snapshot_id"]), canonical_json(other))
    assert _recover_irt_rows(store, state, current) == {}

    from sn120_v2.rotation import Reserve, Retired, active_sha

    store = LocalStore(tmp_path)
    current = IRTSampler(["retired"], Params()).public_snapshot()
    store.put(layout.irt_snapshot(current["snapshot_id"]), canonical_json(current))
    retired = Retired("retired", "high", rounds_seen=10, irt_row=(-1.58552, 0.867357, 10))
    rotation = Rotation("r000066", ("kept",), active_sha(["kept"]), active_sha(["kept"]), (),
                        (retired,), "0" * 64, 0, (), Reserve("0" * 64).history(), {})
    store.put(layout.pool_rotation_state(rotation.digest()), rotation.to_json())
    # Same snapshot id (a no-decision round) and a missing historical snapshot
    # must not suppress the independently available retirement evidence.
    record = RoundRecord("r000066", "head", "params", "recipe", "seed", current["snapshot_id"], 1, 0, (),
                         rotation=rotation.journal_block() | {"state_sha": rotation.digest(), "size": 1})
    missing = RoundRecord("r000065", "head", "params", "recipe", "seed", "irt-missing", 1, 0, ())
    state = State(snapshot_id=current["snapshot_id"], rounds=[missing, record])
    recovered = _recover_irt_rows(store, state, current)
    assert recovered["retired"] == [-1.58552, 0.867357, 10]
