"""Public after-the-fact crown confirmation records, from replayed State and through publication."""

from __future__ import annotations

import dataclasses
import json

import pytest

from sn120_v2 import layout, public, publisher
from sn120_v2.journal import JOURNAL_FILE
from sn120_v2.payout import paid_set
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import APPLIED, RoundRecord, State, load
from sn120_v2.types import WINNER, Confirmation, EntryDecision, King, PendingKing, RoundDecision, Score


def crown(round_id: str, uid: int, z: float, delta: float | None = None) -> RoundRecord:
    delta = z / 100 if delta is None else delta
    score = Score(64, delta, 0.01, z, 0.75, 0.5, 0.0, 0.0, 1.0, True)
    winner = EntryDecision(0, uid, f"hk{uid}", 1000 + uid, "a" * 64, WINNER, score=score)
    decision = RoundDecision(round_id, "p" * 64, 0, (winner,), 64, 0.5, 0.0)
    return RoundRecord(round_id, "p" * 64, "params", "recipe", "seed", "", 64, 1, (), decision=decision,
                       applied=True, outcome=APPLIED, closed_at=2)


def resolution(round_id: str, pending_round_id: str, uid: int, *, confirmed: bool, z: float = 1.0,
               delta: float = 0.1, n_tasks: int = 64, reason: str = "fresh paired sample-mean delta is positive") -> RoundRecord:
    confirmation = Confirmation("p" * 64, pending_round_id, uid, "n" * 64, delta=delta, se=0.01, z=z,
                                n_tasks=n_tasks, confirmed=confirmed, reason=reason)
    return RoundRecord(round_id, "n" * 64, "params", "recipe", "seed", "", n_tasks, 3, (),
                       confirmation=confirmation, outcome="null", closed_at=4)


def stats_for(records, *, pending: PendingKing | None = None, params=None, null_rounds=()) -> dict:
    confirmed = {record.confirmation.pending_round_id: record.confirmation for record in records if record.confirmation}
    crowns = {record.round_id: record for record in records if record.outcome == APPLIED}
    kings = [King(confirmation.pending_uid, f"hk{confirmation.pending_uid}", round_id, "n" * 64, 2,
                  crowns[round_id].decision.winner.score.z)
             for round_id, confirmation in confirmed.items() if confirmation.confirmed]
    return public.confirmation_stats(State(params=params or sim_params(confirm_next_round=True), kings=kings,
                                          pending_king=pending, rounds=list(records), null_rounds=list(null_rounds)))


def test_paid_unpaid_pending_and_pre_rule_crowns():
    pre_rule = crown("r000001", 1, 3.0)  # immediate apply-and-pay: no confirmation record ever
    paid_crown = crown("r000002", 2, 1.25, 0.03125)
    unpaid_crown = crown("r000004", 3, 0.75, -0.01234)
    pending_crown = crown("r000006", 4, 2.0)
    pending = PendingKing(4, "hk4", pending_crown.round_id, "p" * 64, "n" * 64, "a" * 64, 2, 2.0)
    stats = stats_for([
        pre_rule, paid_crown, resolution("r000003", "r000002", 2, confirmed=True, z=1.5, delta=0.04),
        unpaid_crown, resolution("r000005", "r000004", 3, confirmed=False, z=0.1, delta=-0.02), pending_crown,
    ], pending=pending)
    assert [row["round_id"] for row in stats["rows"]] == ["r000006", "r000004", "r000002"]
    assert pre_rule.round_id not in json.dumps(stats)
    newest, unpaid, paid = stats["rows"]
    assert (newest["status"], newest["confirmed"], newest["paid"]) == ("pending", None, False)
    assert (newest["confirmation_round_id"], newest["confirmation_z"], newest["confirmation_delta"]) == (None, None, None)
    assert (unpaid["status"], unpaid["confirmed"], unpaid["paid"]) == ("unpaid", False, False)
    assert (unpaid["z"], unpaid["delta"]) == (0.75, -0.01234)
    assert (unpaid["confirmation_round_id"], unpaid["confirmation_z"], unpaid["confirmation_delta"]) == ("r000005", 0.1, -0.02)
    assert (paid["status"], paid["confirmed"], paid["paid"]) == ("paid", True, True)
    assert (paid["z"], paid["delta"]) == (1.25, 0.03125)
    assert (paid["confirmation_round_id"], paid["confirmation_z"], paid["confirmation_delta"]) == ("r000003", 1.5, 0.04)
    assert stats["total"] == {"n": 2, "paid": 1, "rate": 0.5, "pending": 1, "no_decision": 0}
    assert (stats["first_round_id"], stats["last_round_id"]) == ("r000002", "r000006")


def test_band_edges_belong_to_the_upper_band_and_cumulative_thresholds():
    edges = (0.999, 1.0, 1.249, 1.25, 1.499, 1.5, 1.999, 2.0)
    records = []
    for offset, edge in enumerate(edges, 10):
        round_id, next_id = f"r{offset:06d}", f"r{offset + 1:06d}"
        records += [crown(round_id, offset, edge), resolution(next_id, round_id, offset, confirmed=offset in (11, 15))]
    stats = stats_for(records)
    assert [row["n"] for row in stats["bands"]] == [1, 2, 2, 2, 1]
    assert [row["paid"] for row in stats["bands"]] == [0, 1, 0, 1, 0]
    assert [row["rate"] for row in stats["bands"]] == [0.0, 0.5, 0.0, 0.5, 0.0]
    assert [row["n"] for row in stats["cumulative"]] == [8, 7, 5, 3, 1]
    assert [row["paid"] for row in stats["cumulative"]] == [2, 2, 1, 1, 0]
    assert stats["total"]["n"] == 8 and stats["lowest_paid_z"] == 1.0 and stats["highest_unpaid_z"] == 2.0


def test_infrastructure_no_decision_is_visible_but_not_a_rate_denominator():
    early = [crown(f"r{number:06d}", number, 0.6) for number in (114, 115)]
    resolutions = [resolution(f"r{number + 1:06d}", f"r{number:06d}", number, confirmed=False, z=0, delta=0,
                              n_tasks=0 if number == 114 else 1,
                              reason="no_decision: parent scoring failed: simulated infrastructure failure")
                   for number in (114, 115)]
    paid = crown("r000120", 120, 1.373)
    pending = crown("r000122", 122, 2.0)
    pending_king = PendingKing(122, "hk122", pending.round_id, "p" * 64, "n" * 64, "a" * 64, 2, 2.0)
    stats = stats_for([early[0], resolutions[0], early[1], resolutions[1], paid,
                       resolution("r000121", paid.round_id, 120, confirmed=True), pending], pending=pending_king)
    assert stats["total"] == {"n": 1, "paid": 1, "rate": 1.0, "pending": 1, "no_decision": 2}
    assert sum(band["n"] for band in stats["bands"]) == 1
    for round_id in ("r000114", "r000115"):
        row = next(row for row in stats["rows"] if row["round_id"] == round_id)
        assert (row["status"], row["paid"], row["confirmed"]) == ("no_decision", False, False)
        assert (row["confirmation_z"], row["confirmation_delta"]) == (None, None)


def test_empty_stats_and_paid_field_uses_existing_payout_rule():
    empty = stats_for([])
    assert empty == {
        "rows": [], "bands": [{"band": band, "lo": lo, "hi": hi, "n": 0, "paid": 0, "rate": None}
                             for band, lo, hi in public.CONFIRMATION_BANDS],
        "cumulative": [{"min_z": z, "n": 0, "paid": 0, "rate": None} for z in (0.5, 1.0, 1.25, 1.5, 2.0)],
        "total": {"n": 0, "paid": 0, "rate": None, "pending": 0, "no_decision": 0},
        "lowest_paid_z": None, "highest_unpaid_z": None, "first_round_id": None, "last_round_id": None,
    }
    records = [crown("r000001", 1, 1.0), resolution("r000002", "r000001", 1, confirmed=True),
               crown("r000003", 2, 2.0), resolution("r000004", "r000003", 2, confirmed=True)]
    params = sim_params(confirm_next_round=True, kings_window=1, null_prune_from_round="r000001")
    stats = stats_for(records, params=params, null_rounds=["r000002"])
    row = next(row for row in stats["rows"] if row["round_id"] == "r000001")
    assert row["paid"] is True and row["status"] == "paid"
    # No new rule is derived: paid_set owns the current paid window, independently of this after-the-fact flag.
    state = State(params=params, rounds=records, null_rounds=["r000002"],
                  kings=[King(2, "hk2", "r000003", "n" * 64, 2, 2.0)])
    assert [king.round_id for king in paid_set(state.kings, state.null_rounds, state.params)] == ["r000003"]


def confirming_world(tmp_path) -> SimWorld:
    return SimWorld(tmp_path / "world", sim_params(confirm_next_round=True, null_samples_per_task=2,
                                                 crown_gate="argmax_symmetric", z_bar=0.5, z_field_bar=0.0))


def test_replayed_state_and_publisher_write_matching_public_record_and_llms(tmp_path):
    made = confirming_world(tmp_path)
    made.register(11, 0.4)
    first = made.round()
    made.evaluator.parent_quality = -0.4
    made.register(12, -1.0)
    second = made.round()
    assert first.outcome == APPLIED and second.confirmation.confirmed
    replayed = load(made.root / JOURNAL_FILE)
    snapshot = made.publisher.snapshots[-1]
    expected = public.confirmation_stats(replayed)
    assert snapshot["confirmations"] == expected
    assert expected["rows"][0]["status"] == "paid"

    store = made.store
    publisher.StorePublisher(store, "https://files.example/").publish(snapshot)
    mining = json.loads(store.get(layout.MINING_JSON))
    text = store.get(layout.LLMS_TXT).decode()
    assert mining["confirmations"] == expected
    assert json.loads(store.get(layout.CONFIRMATION_STATS)) == expected
    assert publisher.check_pair(store.get(layout.MINING_JSON), store.get(layout.LLMS_TXT)) == []
    payout = text.split("## 7. Payout\n", 1)[1].split("## 8.", 1)[0]
    assert "### Crown confirmation record" in payout
    flat = " ".join(payout.split())
    assert "After-the-fact record: each applied crown, its original paired z and delta, and the fresh paired test in the next round that decides payment. A crown is paid only if that test passes." in flat
    assert "descriptive, not a rule change. The crown bar is unchanged: strict `z > 0.5`" in flat
    assert "z < 1.0" in payout and "minimum original z" in payout and "Last 30 crowns" in payout
    assert "| minimum original z | n | paid | rate |" in payout
    cumulative = [line for line in payout.splitlines() if line.startswith("| ") and line.split("|")[1].strip()
                  in {"0.5", "1.0", "1.25", "1.5", "2.0"}]
    assert [line.split("|")[1].strip() for line in cumulative] == ["0.5", "1.0", "1.25", "1.5", "2.0"]
    assert "| round_id | uid | z | delta | status | confirmed | confirmation z | confirmation delta | paid |" in payout
    assert f"| {first.round_id} | 11 | " in payout and " | paid | yes | " in payout
    assert all(word not in text for word in ("Costa Rica", "UTC-6", "CR", "CST"))
    assert layout.CONFIRMATION_STATS == "public/confirmation-stats.json"


def test_report_shows_pending_and_no_decision_status_and_direct_public_snapshot_publishes(tmp_path):
    records = [crown("r000114", 114, 0.6),
               resolution("r000115", "r000114", 114, confirmed=False, z=0, delta=0, n_tasks=0,
                          reason="no_decision: parent scoring failed: simulated infrastructure failure"),
               crown("r000116", 116, 1.0)]
    pending = PendingKing(116, "hk116", "r000116", "p" * 64, "n" * 64, "a" * 64, 2, 1.0)
    made = confirming_world(tmp_path)
    made.controller.state = dataclasses.replace(made.controller.state, rounds=records, pending_king=pending)
    snapshot = public.snapshot(made.controller.state, made.params, 5)
    expected = public.confirmation_stats(made.controller.state)
    store = made.store
    publisher.StorePublisher(store, "https://files.example/").publish(snapshot)
    text = store.get(layout.LLMS_TXT).decode()
    assert json.loads(store.get(layout.CONFIRMATION_STATS)) == expected
    assert "| r000116 | 116 | 1.000 | 0.0100 | pending | none | none | none | no |" in text
    assert "| r000114 | 114 | 0.600 | 0.0060 | no_decision | no | none | none | no |" in text


def test_confirmation_file_fails_closed_on_tampering(tmp_path):
    snapshot = confirming_world(tmp_path).publisher.snapshots[-1]
    public.validate_confirmation_stats(snapshot["confirmations"], snapshot)
    with pytest.raises(ValueError, match="same object"):
        changed = {**snapshot["confirmations"], "total": {**snapshot["confirmations"]["total"], "n": 1}}
        public.validate_confirmation_stats(changed, snapshot)
    with pytest.raises(ValueError, match="not an object|same object"):
        public.validate_confirmation_stats({}, snapshot)
