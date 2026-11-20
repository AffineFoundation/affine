"""Once-ever Generator attribution, trailing payout and fail-to-burn weights."""

from __future__ import annotations

import copy

import pytest

from sn120_v2.errors import ContractError
from sn120_v2.generator import constants, payout

NOW = 1_800_000_000
T1, T2 = "01" * 10, "02" * 10


def upload(task_id=T1, uid=7, hotkey="hk7", uploaded_at=100.0, sha="a" * 64, valid=True):
    return payout.Upload(task_id, uid, hotkey, uploaded_at, sha, valid)


def claim(task_id=T1, uid=7, hotkey="hk7", uploaded_at=NOW - 60, sha="a" * 64):
    return payout.Claim(task_id, uid, hotkey, uploaded_at, sha)


def test_earliest_valid_upload_wins_even_if_listed_last():
    ledger = payout.first_accepted_claims([
        upload(uploaded_at=102, uid=8, hotkey="hk8", sha="0" * 64),
        upload(uploaded_at=101, uid=7, hotkey="hk7"),
        upload(uploaded_at=103, uid=9, hotkey="hk9"),
    ])
    assert ledger[T1] == claim(uploaded_at=101)


def test_timestamp_tie_uses_file_sha256_and_is_order_independent():
    early = upload(uploaded_at=101, sha="b" * 64)
    tie = upload(uploaded_at=101, uid=8, hotkey="hk8", sha="a" * 64)
    assert payout.first_accepted_claims([early, tie])[T1].uid == 8
    assert payout.first_accepted_claims([tie, early])[T1].uid == 8
    assert payout.first_accepted_claims([tie, early])[T1].file_sha256 == "a" * 64


def test_invalid_and_late_uploads_never_win():
    ledger = payout.first_accepted_claims([
        upload(uploaded_at=1, valid=False),
        upload(uploaded_at=100, uid=9, hotkey="hk9"),
    ])
    assert ledger[T1].uid == 9
    assert payout.first_accepted_claims([upload(valid=False)]) == {}


def test_once_ever_ledger_survives_trailing_window_and_new_upload():
    old = claim(uploaded_at=NOW - 3 * 86400)
    ledger = {T1: old}
    new = upload(uploaded_at=NOW - 10, uid=99, hotkey="hk99")
    assert payout.first_accepted_claims([new], ledger)[T1] == old
    assert payout.trailing_claims(ledger.values(), NOW) == ()
    assert payout.payout_weights(ledger.values(), {7: "hk7", 99: "hk99"}, now=NOW, updated_at=NOW) == {0: 1.0}


def test_two_unique_tasks_pay_equal_task_shares_to_same_or_different_uids():
    one = claim(T1, uid=7, hotkey="hk7", uploaded_at=NOW - 100)
    two = claim(T2, uid=7, hotkey="hk7", uploaded_at=NOW - 90)
    assert payout.payout_weights([one, two], {7: "hk7"}, now=NOW, updated_at=NOW) == {7: 1.0}
    three = payout.Claim("03" * 10, 8, "hk8", NOW - 80, "c" * 64)
    assert payout.payout_weights([one, two, three], {7: "hk7", 8: "hk8"}, now=NOW, updated_at=NOW) == pytest.approx({7: 2 / 3, 8: 1 / 3})


def test_registration_churn_excludes_old_hotkey_but_never_returns_zero():
    one = claim(T1, uid=7, hotkey="old")
    two = claim(T2, uid=8, hotkey="current")
    assert payout.payout_weights([one, two], {7: "new", 8: "current"}, now=NOW, updated_at=NOW) == {8: 1.0}
    assert payout.payout_weights([one], {7: "new"}, now=NOW, updated_at=NOW) == {0: 1.0}
    assert payout.payout_weights([claim(uid=0)], {0: "hk7"}, now=NOW, updated_at=NOW) == {0: 1.0}


def test_trailing_window_is_exactly_24_hours_and_never_future():
    at_edge = claim(uploaded_at=NOW - constants.PAYOUT_WINDOW_SECONDS)
    inside = claim(T2, uploaded_at=NOW - constants.PAYOUT_WINDOW_SECONDS + 0.001)
    future = claim("03" * 10, uploaded_at=NOW + 0.001)
    assert payout.trailing_claims([at_edge, inside, future], NOW) == (inside,)
    assert payout.payout_weights([at_edge], {7: "hk7"}, now=NOW, updated_at=NOW) == {0: 1.0}


def test_payout_six_hour_staleness_is_inclusive_future_is_invalid():
    rows = [claim()]
    registered = {7: "hk7"}
    assert payout.payout_weights(rows, registered, now=NOW, updated_at=NOW - 6 * 3600) == {7: 1.0}
    assert payout.payout_weights(rows, registered, now=NOW, updated_at=NOW - 6 * 3600 - 0.001) == {0: 1.0}
    assert payout.payout_weights(rows, registered, now=NOW, updated_at=NOW + 0.001) == {0: 1.0}
    assert payout.payout_weights([], registered, now=NOW, updated_at=NOW) == {0: 1.0}


def test_weights_are_normalized_and_never_all_zero():
    rows = [claim(f"{number + 1:02d}" * 10, uid=number + 1, hotkey=f"hk{number + 1}") for number in range(7)]
    weights = payout.payout_weights(rows, {row.uid: row.hotkey for row in rows}, now=NOW, updated_at=NOW)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert any(value > 0 for value in weights.values())
    assert payout.payout_weights(rows, {}, now=NOW, updated_at=NOW) == {0: 1.0}


def test_payout_document_is_data_only_and_has_explicit_freshness():
    rows = [claim(), claim(T2, uid=8, hotkey="hk8")]
    document = payout.payout_document(rows, now=NOW, updated_at=NOW - 30)
    assert document["version"] == "affine-generator-payout-v1"
    assert document["updated_at"] == NOW - 30
    assert document["window_seconds"] == 24 * 3600
    assert document["max_age_seconds"] == 6 * 3600
    assert [row["task_id"] for row in document["claims"]] == [T1, T2]
    assert "weights" not in document


def test_invalid_evidence_and_duplicate_ledger_fail_closed():
    with pytest.raises(ContractError):
        payout.first_accepted_claims([upload(task_id="mp:123")])
    with pytest.raises(ContractError):
        payout.first_accepted_claims([upload(uid=True)])
    with pytest.raises(ContractError):
        payout.first_accepted_claims([upload(valid=1)])
    with pytest.raises(ContractError):
        payout.first_accepted_claims([upload(sha="bad")])
    with pytest.raises(ContractError):
        payout.first_accepted_claims([{"task_id": T1}])
    with pytest.raises(ContractError):
        payout.trailing_claims([claim(), claim()], NOW)
    with pytest.raises(ContractError):
        payout.payout_weights([], {}, now=NOW, updated_at=NOW, burn_uid=1)
    with pytest.raises(ContractError):
        payout.trailing_claims([], NOW, window_s=True)


def test_no_mutation_of_permanent_ledger_input():
    ledger = {T1: claim(uploaded_at=NOW - 3 * 86400)}
    original = copy.deepcopy(ledger)
    payout.first_accepted_claims([upload(T2, uploaded_at=NOW)], ledger)
    assert ledger == original
