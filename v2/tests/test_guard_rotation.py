"""The fail-closed admission gate (rotation rule ``info-guard``, ``affine-pool-rotation-v3``).

The gate only ever refuses an admission. It never removes an active task, never
changes the ranking or ``next_up``, and never reaches the network: its single
input is the sealed public guard ledger.
"""
from __future__ import annotations

import pytest

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.errors import ConfigError, VerifyError
from sn120_v2.params import Params
from sn120_v2.rotation import (EMPTY_GUARD, GUARD_RULE_VERSION, GuardLedger, Reserve, extends_guard,
                               extends_guard_bytes, parse_guard_ledger, rotate_info, rule_version)
from sn120_v2.guard.store import public_row
from tests.test_pool_rotation_v2 import info_params, snapshot

VERSION = "guard-v1:0123456789abcdef"


def body_sha(task_id: str) -> str:
    return sha256_hex(canonical_json({"problem": f"problem {task_id}", "answer": "1"}))


def guard_params(**changes) -> Params:
    return Params(rotation_enabled=True, rotation_rule="info-guard", **changes)


def reserve(tasks: dict[str, float], withdrawn=()) -> Reserve:
    meta = {task: {"prior_p": p, "source_prefix": task.split(":")[0], "env": "affine_math",
                   "content_sha": body_sha(task)} for task, p in tasks.items()}
    return Reserve(sha256_hex(canonical_json(tasks)), tuple(tasks), candidate_meta=meta,
                   withdrawn=tuple({"id": task, "reason": "quality:bad_key", "added_at": "2026-11-27"}
                                   for task in withdrawn))


def ledger(passes=(), rejects=(), stale=()) -> GuardLedger:
    rows = [public_row(task, body_sha(task), True, "ok", VERSION, "2026-11-27") for task in passes]
    rows += [public_row(task, body_sha(task), False, "bad_key", VERSION, "2026-11-27") for task in rejects]
    rows += [public_row(task, sha256_hex(b"older body " * 3)[:64], True, "ok", VERSION, "2026-11-27")
             for task in stale]
    return parse_guard_ledger(b"".join(canonical_json(row) + b"\n" for row in rows))


def setup(active=6, outside=("new:1", "new:2", "new:3", "new:4"), size=8):
    tasks = {f"a:{i}": [0.0, 0.25, 4] for i in range(active)}
    return snapshot(tasks, 0.0), tasks, reserve({task: 0.5 for task in outside}), guard_params(
        rotation_active_size=size, rotation_churn_frac=0.0)


def test_the_rule_version_and_params_name_the_gate():
    assert rule_version(guard_params()) == GUARD_RULE_VERSION == "affine-pool-rotation-v3"
    assert rule_version(info_params()) == "affine-pool-rotation-v2"


def test_only_a_pass_for_the_current_body_is_admitted():
    snap, tasks, res, params = setup()
    decision = rotate_info("r1", tasks, snap, {}, res, params, guard=ledger(passes=("new:1", "new:3")))
    assert decision.rule_version == GUARD_RULE_VERSION
    assert decision.admitted == tuple(sorted(("new:1", "new:3"))) or set(decision.admitted) == {"new:1", "new:3"}
    assert decision.guard_shortfall == 0  # two seats, two eligible candidates


def test_a_rejected_or_undecided_candidate_is_never_admitted():
    snap, tasks, res, params = setup()
    decision = rotate_info("r1", tasks, snap, {}, res, params,
                           guard=ledger(passes=("new:1",), rejects=("new:2",)))
    assert set(decision.admitted) == {"new:1"}      # new:3 and new:4 have no row at all
    assert decision.guard_shortfall == 1


def test_a_pass_for_an_older_body_does_not_admit_the_current_one():
    snap, tasks, res, params = setup()
    decision = rotate_info("r1", tasks, snap, {}, res, params, guard=ledger(stale=("new:1",)))
    assert decision.admitted == ()
    assert decision.guard_shortfall == 2


def test_an_empty_ledger_admits_nobody_and_keeps_every_active_task():
    snap, tasks, res, params = setup()
    decision = rotate_info("r1", tasks, snap, {}, res, params, guard=EMPTY_GUARD)
    assert decision.admitted == ()
    assert set(decision.active_ids) == set(tasks)        # the gate removes nothing
    assert decision.guard_shortfall == 2
    assert decision.retire_reason_counts == {}


def test_a_withdrawn_active_task_still_leaves_whatever_the_ledger_says():
    snap, tasks, res, params = setup()
    res = reserve({"new:1": 0.5, "new:2": 0.5}, withdrawn=("a:0",))
    decision = rotate_info("r1", tasks, snap, {}, res, params, guard=EMPTY_GUARD)
    assert "a:0" not in decision.active_ids
    assert decision.retire_reason_counts == {"withdrawn": 1}
    assert decision.admitted == ()  # and nothing replaces it while the ledger is empty


def test_next_up_keeps_every_candidate_and_next_up_eligible_counts_the_admissible_ones():
    snap, tasks, res, params = setup()
    decision = rotate_info("r1", tasks, snap, {}, res, params, guard=ledger(passes=("new:1",)))
    assert len(decision.next_up) + len(decision.admitted) == 4  # the pre-judge worker still sees them all
    assert decision.next_up_eligible == 0  # new:1 was admitted; the rest are undecided
    full = rotate_info("r1", tasks, snap, {}, res, params, guard=ledger(passes=("new:1", "new:2", "new:3", "new:4")))
    assert full.next_up_eligible == len(full.next_up)


def test_the_guard_block_records_the_exact_ledger_the_round_admitted_from():
    snap, tasks, res, params = setup()
    guard = ledger(passes=("new:1", "new:2"))
    decision = rotate_info("r1", tasks, snap, {}, res, params, guard=guard)
    assert decision.guard_block() == {"guard_ledger_sha": guard.sha256, "guard_ledger_count": guard.count,
                                      "guard_pass": 2, "guard_reject": 0, "guard_shortfall": 0,
                                      "next_up_eligible": decision.next_up_eligible}


def test_the_decision_is_deterministic_and_independent_of_input_order():
    snap, tasks, res, params = setup()
    guard = ledger(passes=("new:1", "new:2", "new:3"))
    first = rotate_info("r1", tasks, snap, {}, res, params, guard=guard)
    second = rotate_info("r1", reversed(list(tasks)), snap, {}, res, params, guard=guard)
    assert first == second


def test_the_rule_and_the_ledger_must_agree():
    snap, tasks, res, params = setup()
    with pytest.raises(ConfigError):
        rotate_info("r1", tasks, snap, {}, res, params)  # info-guard without a ledger
    with pytest.raises(ConfigError):
        rotate_info("r1", tasks, snap, {}, res, info_params(rotation_active_size=8), guard=EMPTY_GUARD)


def test_a_v2_round_carries_no_guard_fields():
    snap, tasks, res, params = setup()
    decision = rotate_info("r1", tasks, snap, {}, res, info_params(rotation_active_size=8, rotation_churn_frac=0.0))
    assert decision.rule_version == "affine-pool-rotation-v2"
    assert decision.guard_ledger_sha == "" and decision.guard_ledger_count == 0
    assert decision.guard_shortfall == 0 and decision.next_up_eligible == 0
    assert decision.guard_block() == {}


def test_an_existing_v2_round_replays_byte_identically_under_the_new_code():
    """The regression that matters: old rounds keep their own rule version and bytes."""
    snap, tasks, res, _params = setup()
    params = info_params(rotation_active_size=8, rotation_churn_frac=0.0)
    decision = rotate_info("r000070", tasks, snap, {}, res, params)
    again = rotate_info("r000070", tasks, snap, {}, res, Params.from_dict(decision.rule_params))
    assert again.to_json() == decision.to_json()
    assert again.rule_version == "affine-pool-rotation-v2"


def test_the_ledger_is_append_only_on_its_bytes_and_on_its_counts():
    first = b"".join(canonical_json(public_row(task, body_sha(task), True, "ok", VERSION, "2026-11-27")) + b"\n"
                     for task in ("new:1",))
    second = first + canonical_json(public_row("new:2", body_sha("new:2"), True, "ok", VERSION, "2026-11-27")) + b"\n"
    extends_guard_bytes(first, second)
    extends_guard(parse_guard_ledger(first), parse_guard_ledger(second))
    with pytest.raises(VerifyError):
        extends_guard_bytes(second, first)
    with pytest.raises(VerifyError):
        extends_guard(parse_guard_ledger(second), parse_guard_ledger(first))


def test_the_ledger_refuses_rows_the_rule_cannot_trust():
    good = public_row("new:1", body_sha("new:1"), True, "ok", VERSION, "2026-11-27")
    for broken in ({**good, "added_at": "27-11-2026"}, {**good, "category": "OK"},
                   {**good, "verdict": "maybe"}, {**good, "content_sha": "short"},
                   {**good, "extra": "x"}):
        with pytest.raises(VerifyError):
            parse_guard_ledger(canonical_json(broken) + b"\n")
    with pytest.raises(VerifyError):  # not newline-terminated
        parse_guard_ledger(canonical_json(good))
    with pytest.raises(VerifyError):  # the same body decided twice
        parse_guard_ledger((canonical_json(good) + b"\n") * 2)
