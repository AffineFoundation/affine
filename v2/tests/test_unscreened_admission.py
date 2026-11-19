"""Per-round raw-admission cap, retired provenance and strict public replay."""
from __future__ import annotations

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.pool_runtime import active_file, replay_rotation
from sn120_v2.rotation import RANKING_LIMIT, RANKING_OUTSIDE_LIMIT, Reserve, rotate_info
from sn120_v2.store import LocalStore

from test_pool_rotation_v2 import info_params, snapshot


def test_new_cap_default_and_live_param_hashes_are_compatible():
    import dataclasses
    from sn120_v2.params import Params, load_params
    from pathlib import Path
    live_file = Path("/tmp/sn120-info-live/params-9b0afb60.json")
    assert Params().params_hash == "10a121ec5a4ba830057e5510665fee1be7aad34dae259d3427e4547c393c2cf1"
    assert "rotation_unscreened_max" not in Params().to_dict()
    assert "rotation_unscreened_max" not in info_params().to_dict()
    if live_file.exists():
        params = load_params(live_file)
        params = dataclasses.replace(params, recipe=dataclasses.replace(params.recipe, code_digest="64e7bd9c10bba7eeb139ca70a68c41711da882dbee42763890f85be59a4ea3f9"),
                                     rotation_enabled=True, rotation_rule="info", rotation_active_size=10000, rotation_churn_frac=.1)
        assert params.params_hash == "94f931588114f335814231409938f39f2056893708bc798e90f1f7676920a35c"
        assert dataclasses.replace(params, rotation_unscreened_max=500).params_hash == "605583e24db4401a66dbaecb5eb5dfce6da8bac79746b1ab909f273d44e72d62"


def reserve(prior: dict[str, float], raw: set[str]) -> Reserve:
    return Reserve(sha256_hex(canonical_json(prior)), tuple(prior),
                   candidate_meta={task: {"prior_p": p} for task, p in prior.items()},
                   audit_verdicts={task: "unscreened" for task in raw})


def test_fill_cap_skips_raw_but_admits_screened_and_stays_deterministic():
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(2)}
    pool = {"raw:0": .5, "raw:1": .49, "screened:0": .48}
    params = info_params(rotation_active_size=4, rotation_churn_frac=0.0, rotation_unscreened_max=1)
    one = rotate_info("r000070", active, snapshot(active, 2.0), {}, reserve(pool, {"raw:0", "raw:1"}), params)
    assert set(one.admitted) == {"raw:0", "screened:0"}
    assert one.rule_params["rotation_unscreened_max"] == 1
    assert one.queue_remaining == 1 and one.next_up == ("raw:1",)
    assert sum(row.get("unscreened", False) for row in one.candidate_ranking) == 2
    two = rotate_info("r000070", list(active), snapshot(active, 2.0), {}, reserve(dict(reversed(list(pool.items()))), {"raw:0", "raw:1"}), params)
    assert (two.active_ids, two.admitted, two.retired, two.next_up) == (one.active_ids, one.admitted, one.retired, one.next_up)
    uncapped = rotate_info("r000070", active, snapshot(active, 2.0), {}, reserve(pool | {"screened:1": .5}, {"raw:0", "raw:1"}), info_params(rotation_active_size=5, rotation_churn_frac=0.0))
    assert "rotation_unscreened_max" not in uncapped.rule_params and set(uncapped.admitted) == {"raw:0", "raw:1", "screened:1"}


def test_swap_cap_counts_fill_and_swap_together_and_leaves_raw_queued():
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(3)}
    snap = snapshot(active, -2.0)
    pool = {"fill:0": .5, "swap:0": .51, "swap:1": .49}
    params = info_params(rotation_active_size=4, rotation_churn_frac=1.0, rotation_unscreened_max=1)
    decision = rotate_info("r000070", active, snap, {}, reserve(pool, set(pool)), params)
    assert decision.admitted == ("fill:0",) and not decision.retired
    assert set(decision.next_up) == {"swap:0", "swap:1"}
    # A different round may admit the previously skipped raw candidate.
    later = rotate_info("r000071", active, snap, {}, reserve(pool, set(pool)), params)
    assert later.admitted == ("fill:0",)  # tie hashes differ, the same cap holds


def test_swap_cap_blocks_raw_but_not_screened():
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(4)}
    pool = {"raw:0": .5, "screened:0": .49}
    params = info_params(rotation_active_size=5, rotation_churn_frac=1.0, rotation_unscreened_max=1)
    decision = rotate_info("r000070", active, snapshot(active, -2), {}, reserve(pool | {"fill:0": .5}, {"raw:0", "fill:0"}), params)
    assert set(decision.admitted) == {"fill:0", "screened:0"}
    assert [row.reason for row in decision.retired] == ["displaced"]
    assert "raw:0" in decision.next_up


def test_retired_raw_provenance_caps_readmission_and_public_replay(tmp_path):
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(2)}
    snap = snapshot(active, 2.0)
    params = info_params(rotation_active_size=4, rotation_churn_frac=0.0, rotation_unscreened_max=1)
    pool = {"raw:0": .5, "screened:0": .49, "screened:1": .5}
    first = rotate_info("r000070", active, snap, {}, reserve(pool, {"raw:0"}), params, source_snapshot_id=snap["snapshot_id"])
    assert set(first.admitted) == {"raw:0", "screened:1"}
    archive = [{"id": "raw:1", "round_id": "r000060", "reason": "displaced", "event_seq": 0,
                "audit_verdict": "unscreened", "prior_p": .51, "irt_row": [0.0, .25, 4],
                "head_rollouts": 0, "head_passes": 0}]
    capped = rotate_info("r000071", first.active_ids, snapshot({task: [0.0, .25, 4] for task in first.active_ids}, 2.0),
                         first.ledger, reserve({}, set()), params, retired_archive=first.retired_archive + tuple(archive))
    assert "raw:1" not in capped.admitted
    raw = next(row for row in capped.candidate_ranking if row["id"] == "raw:1")
    assert raw["unscreened"] and raw["state"] == "retired"

    combined = rotate_info("r000071", active, snap, {}, reserve({"raw:0": .5}, {"raw:0"}),
                           info_params(rotation_active_size=4, rotation_churn_frac=0.0, rotation_unscreened_max=1), retired_archive=archive)
    assert combined.admitted == ("raw:0",)
    assert "raw:1" not in combined.active_ids

    store = LocalStore(tmp_path)
    store.put(layout.irt_snapshot(snap["snapshot_id"]), canonical_json(snap))
    store.put(layout.active_pool(first.prev_active_sha), canonical_json(sorted(active)))
    active_file(store, first)
    report = replay_rotation(store, "r000070")
    assert report["matches"] and report["actual"] == report["expected"]


def test_retirement_preserves_raw_provenance_for_later_rounds():
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(3)}
    pool = {"raw:0": .5, "screened:0": .49}
    params = info_params(rotation_active_size=4, rotation_churn_frac=0.0, rotation_unscreened_max=1)
    first = rotate_info("r000070", active, snapshot(active, 2.0), {}, reserve(pool, {"raw:0"}), params)
    assert set(first.admitted) == {"raw:0"}
    later_reserve = Reserve.from_dict(first.reserve_history | {"queue": (), "audit_verdicts": {}, "candidate_meta": {"raw:0": {"unscreened": True}}})
    shrink = rotate_info("r000071", first.active_ids, snapshot({task: [0.0, .25, 4] for task in first.active_ids}, 2.0),
                         first.ledger, later_reserve, info_params(rotation_active_size=1, rotation_churn_frac=0.0, rotation_unscreened_max=1),
                         retired_archive=first.retired_archive)
    assert shrink.admitted == ()
    archived = next(row for row in shrink.retired_archive if row["id"] == "raw:0")
    assert archived["audit_verdict"] == "unscreened"
    later = rotate_info("r000072", shrink.active_ids, snapshot({task: [0.0, .25, 4] for task in shrink.active_ids}, 2.0),
                        shrink.ledger, reserve({}, set()), info_params(rotation_active_size=4, rotation_churn_frac=0.0, rotation_unscreened_max=1),
                        retired_archive=shrink.retired_archive)
    row = next(row for row in later.candidate_ranking if row["id"] == "raw:0")
    assert row["unscreened"] and "raw:0" in later.admitted


def test_large_ranking_is_bounded_without_dropping_replay_inputs():
    count = RANKING_LIMIT + 1
    active = {"a:0": [0.0, 0.25, 4]}
    # 100,001 candidates; no expensive IRT fit is needed for outside rows.
    pool = {f"new:{i}": .5 for i in range(count - 1)}
    retired = [{"id": f"old:{i}", "round_id": "r000060", "reason": "excess", "event_seq": i,
                "irt_row": [0.0, .25, 0], "head_rollouts": 0, "head_passes": 0, "prior_p": 0.0} for i in range(10)]
    pool = dict(list(pool.items())[:count - len(active) - len(retired)])
    active.update({f"anchor:{i}": [0.0, .25, 4] for i in range(9999)})
    pool = dict(list(pool.items())[:count - len(active) - len(retired) - 9999 + 1])
    pool.update({f"extra:{i}": .0 for i in range(10000)})
    decision = rotate_info("r000070", active, snapshot(active, 2.0), {}, reserve(pool, set()),
                           info_params(rotation_active_size=10000, rotation_churn_frac=0.0), retired_archive=retired)
    assert len(decision.candidate_ranking) < RANKING_LIMIT
    assert sum(row["id"] in decision.active_ids for row in decision.candidate_ranking) == 10000
    assert {row["id"] for row in retired} <= {row["id"] for row in decision.candidate_ranking if row["state"] == "retired"}
    assert len(decision.reserve_history["queue"]) == len(pool)
    assert sum(row["state"] == "reserve" and not row["admitted"] for row in decision.candidate_ranking) <= RANKING_OUTSIDE_LIMIT
    assert all(row["admitted"] for row in decision.candidate_ranking if row["id"] in decision.admitted)
    assert set(decision.admitted) <= set(decision.active_ids)
    assert not decision.ranking_complete


def test_bounded_ranking_replays_complete_rotation(tmp_path):
    active = {f"a:{i}": [0.0, 0.25, 4] for i in range(3)}
    snap = snapshot(active, 2.0)
    outside = {f"new:{i}": .5 for i in range(7)}
    import sn120_v2.rotation as rotation
    import sn120_v2.pool_runtime as runtime
    original = rotation.RANKING_LIMIT
    original_outside = rotation.RANKING_OUTSIDE_LIMIT
    original_runtime_outside = runtime.RANKING_OUTSIDE_LIMIT
    rotation.RANKING_LIMIT = 5
    rotation.RANKING_OUTSIDE_LIMIT = 2
    runtime.RANKING_OUTSIDE_LIMIT = 2
    try:
        decision = rotate_info("r000070", active, snap, {}, reserve(outside, set()),
                               info_params(rotation_active_size=3, rotation_churn_frac=0.0), source_snapshot_id=snap["snapshot_id"])
        assert not decision.ranking_complete and len(decision.candidate_ranking) == 5
        store = LocalStore(tmp_path)
        store.put(layout.irt_snapshot(snap["snapshot_id"]), canonical_json(snap))
        store.put(layout.active_pool(decision.prev_active_sha), canonical_json(sorted(active)))
        active_file(store, decision)
        assert replay_rotation(store, "r000070")["matches"]
    finally:
        rotation.RANKING_LIMIT = original
        rotation.RANKING_OUTSIDE_LIMIT = original_outside
        runtime.RANKING_OUTSIDE_LIMIT = original_runtime_outside