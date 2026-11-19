"""Cap-zero byte compatibility with an actual 6c3d5b64-produced decision."""
from __future__ import annotations

import json
from pathlib import Path

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json
from sn120_v2.params import Params
from sn120_v2.pool_runtime import active_file, replay_rotation
from sn120_v2.rotation import Reserve, Rotation, rotate_info
from sn120_v2.store import LocalStore


def test_6c3d5b64_history_round_trip_and_zero_cap_bytes_and_replay(tmp_path):
    """The old bytes, not a newly generated expectation, are the authority."""
    fixture = json.loads((Path(__file__).parent / "data" / "rotation_zero_cap_6c3d5b64.json").read_bytes())
    assert fixture["producer"] == "6c3d5b64"
    old = Rotation.from_dict(fixture["decision"])
    assert old.rule_version == "affine-pool-rotation-v2"
    assert old.rule_params.get("rotation_unscreened_max") is None
    assert old.to_json() == canonical_json(fixture["decision"])

    # A history with neither new optional map must not grow an empty dictionary.
    old_history = {"manifest_sha": "a" * 64, "queue": ["r:0"], "shards": [],
                   "withdrawn": [], "withdrawn_sha": "", "withdrawn_count": 0}
    assert Reserve.from_dict(old_history).to_dict() == old_history
    assert Reserve.from_dict(old_history).history() == old_history
    assert Reserve.from_dict(old.reserve_history).to_dict() == old.reserve_history
    assert old.reserve_history == {
        "manifest_sha": "b" * 64,
        "queue": ["r:0", "r:1"],
        "shards": [], "withdrawn": [], "withdrawn_sha": "", "withdrawn_count": 0,
        "candidate_meta": {
            "r:0": {"prior_p": 0.51, "source_prefix": "bigmath", "env": "affine_math"},
            "r:1": {"prior_p": 0.49, "source_prefix": "bigmath", "env": "affine_math"},
        },
    }

    snap = fixture["snapshot"]
    reserve = Reserve.from_dict(old.reserve_history)
    params = Params.from_dict(old.rule_params)
    assert params.rotation_unscreened_max == 0
    new = rotate_info(old.round_id, fixture["previous_active"], snap, {}, reserve, params,
                      source_snapshot_id=old.source_snapshot_id)
    assert new.to_json() == canonical_json(fixture["decision"])
    assert canonical_json(new.journal_block()) == canonical_json(fixture["journal_block"])

    store = LocalStore(tmp_path)
    store.put(layout.irt_snapshot(snap["snapshot_id"]), canonical_json(snap))
    store.put(layout.active_pool(new.prev_active_sha), canonical_json(fixture["previous_active"]))
    active_file(store, new)
    assert store.get(layout.pool_rotation(new.round_id)) == canonical_json(fixture["decision"])
    assert store.get(layout.pool_rotation_state(new.digest())) == canonical_json(fixture["state"])
    assert store.get(layout.pool_ranking(new.round_id)) == canonical_json(fixture["ranking"])

    # Replay the old decision, rather than replaying a fresh decision merely
    # serialized into the old fixture. Its exact bytes remain the expectation.
    store.put(layout.pool_rotation(new.round_id), canonical_json(fixture["decision"]))
    store.put(layout.pool_rotation_state(new.digest()), canonical_json(fixture["state"]))
    report = replay_rotation(store, new.round_id)
    assert report["matches"] and report["actual"] == report["expected"]
