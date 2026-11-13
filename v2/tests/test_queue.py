from __future__ import annotations

import pytest

from sn120_v2.errors import InfraError
from sn120_v2.queue import plan_sync, select_round, split_among_boxes, squat_trim, waiting
from sn120_v2.types import ChainSnapshot, Verdict

DAY = 86400


def uids(seats) -> list[int]:
    return [seat.reg.uid for seat in seats]


def freeze_all(build):
    return lambda reg: build.batch(reg)


def test_order_is_registration_block_then_uid(build):
    queue = build.queue(build.seat(9, block=20), build.seat(5, block=10), build.seat(3, block=10), build.seat(7, block=15))
    assert uids(waiting(queue)) == [3, 5, 7, 9]


def test_seats_sent_back_to_the_front_go_first_in_their_original_order(build):
    queue = build.queue(
        build.seat(1, block=10),
        build.seat(2, block=20),
        build.seat(8, block=80, front=True),
        build.seat(6, block=60, front=True),
    )
    assert uids(waiting(queue)) == [6, 8, 1, 2]


def test_demoted_seats_go_last_in_registration_order(build):
    queue = build.queue(build.seat(1, block=10, demoted=True), build.seat(4, block=5, demoted=True), build.seat(2, block=20))
    assert uids(waiting(queue)) == [2, 4, 1]


def test_a_front_seat_stays_first_even_if_it_was_demoted_before(build):
    queue = build.queue(build.seat(1, block=10), build.seat(2, block=20, demoted=True, front=True))
    assert uids(waiting(queue)) == [2, 1]


def test_leased_seats_are_not_waiting(build):
    queue = build.queue(build.seat(1, leased=True), build.seat(2))
    assert uids(waiting(queue)) == [2]


def test_round_takes_up_to_the_cap_in_queue_order(build):
    queue = build.queue(*(build.seat(uid) for uid in (5, 1, 4, 2, 3)))
    picked = select_round(queue, build.verdicts("hk1", "hk2", "hk3", "hk4", "hk5"), freeze_all(build), cap=3)
    assert [e.reg.uid for e in picked.entries] == [1, 2, 3]
    assert [e.queue_pos for e in picked.entries] == [0, 1, 2]
    assert picked.skipped == ()


def test_a_short_queue_runs_with_what_is_ready(build):
    queue = build.queue(build.seat(1), build.seat(2))
    picked = select_round(queue, build.verdicts("hk1", "hk2"), freeze_all(build), cap=32)
    assert [e.reg.uid for e in picked.entries] == [1, 2]


def test_a_seat_without_a_valid_file_is_skipped_and_keeps_its_place(build):
    queue = build.queue(build.seat(1), build.seat(2), build.seat(3))
    verdicts = build.verdicts("hk1", "hk3") | {"hk2": Verdict("hk2", False, reason="bad json")}
    picked = select_round(queue, verdicts, freeze_all(build), cap=2)
    assert [e.reg.uid for e in picked.entries] == [1, 3]
    assert picked.skipped == ((2, 102),)
    assert uids(waiting(queue)) == [1, 2, 3], "selection never changes the queue"


def test_freeze_is_only_called_for_files_the_live_check_calls_valid(build):
    calls = []

    def freeze(reg):
        calls.append(reg.uid)
        return build.batch(reg)

    queue = build.queue(build.seat(1), build.seat(2))
    select_round(queue, build.verdicts("hk2"), freeze, cap=5)
    assert calls == [2]


def test_a_file_that_fails_at_freeze_time_is_a_skip(build):
    queue = build.queue(build.seat(1), build.seat(2))
    freeze = lambda reg: None if reg.uid == 1 else build.batch(reg)
    picked = select_round(queue, build.verdicts("hk1", "hk2"), freeze, cap=5)
    assert [e.reg.uid for e in picked.entries] == [2]
    assert picked.skipped == ((1, 101),)


def test_no_ready_seat_gives_an_empty_selection(build):
    queue = build.queue(build.seat(1), build.seat(2))
    picked = select_round(queue, {}, freeze_all(build), cap=5)
    assert picked.entries == () and len(picked.skipped) == 2


def test_one_entry_per_hotkey_per_round(build):
    queue = build.queue(build.seat(1, block=10, hotkey="same"), build.seat(2, block=20, hotkey="same"), build.seat(3, block=30))
    picked = select_round(queue, build.verdicts("same", "hk3"), freeze_all(build), cap=5)
    assert [e.reg.uid for e in picked.entries] == [1, 3]
    assert picked.skipped == (), "the second seat is not a skip: it waits for a later round"


def test_one_entry_per_coldkey_per_round(build):
    queue = build.queue(build.seat(1, coldkey="owner"), build.seat(2, coldkey="owner"), build.seat(3, coldkey="other"))
    picked = select_round(queue, build.verdicts("hk1", "hk2", "hk3"), freeze_all(build), cap=5)
    assert [e.reg.uid for e in picked.entries] == [1, 3]


def test_seats_without_a_coldkey_are_not_grouped_together(build):
    queue = build.queue(build.seat(1), build.seat(2))
    picked = select_round(queue, build.verdicts("hk1", "hk2"), freeze_all(build), cap=5)
    assert len(picked.entries) == 2


def test_held_and_leased_seats_are_passed_over(build):
    queue = build.queue(build.seat(1, held=True), build.seat(2, leased=True), build.seat(3))
    picked = select_round(queue, build.verdicts("hk1", "hk2", "hk3"), freeze_all(build), cap=5)
    assert [e.reg.uid for e in picked.entries] == [3]
    assert picked.skipped == ()


def test_front_seats_are_offered_first(build):
    queue = build.queue(build.seat(1), build.seat(9, front=True))
    picked = select_round(queue, build.verdicts("hk1", "hk9"), freeze_all(build), cap=1)
    assert [e.reg.uid for e in picked.entries] == [9]


@pytest.mark.parametrize("n,boxes,sizes", [(5, 1, [5]), (5, 2, [3, 2]), (6, 3, [2, 2, 2]), (2, 4, [1, 1]), (1, 1, [1])])
def test_boxes_get_consecutive_blocks_of_seats(build, n, boxes, sizes):
    entries = tuple(build.entry(i) for i in range(n))
    blocks = split_among_boxes(entries, boxes)
    assert [len(block) for block in blocks] == sizes
    assert [e.queue_pos for block in blocks for e in block] == list(range(n))


def trim(build, queue, ready=(), now=10 * DAY, turns=24):
    return squat_trim(queue, build.verdicts(*ready), now, turns, min_wait_s=DAY, age_factor=2.0)


def test_squat_trim_moves_an_old_seat_with_no_file(build):
    queue = build.queue(build.seat(1, registered_at=7 * DAY))
    assert trim(build, queue) == [(1, 101)], "age 3 days is more than 2 x the 1 day minimum wait"


def test_squat_trim_never_moves_a_seat_with_a_valid_file(build):
    queue = build.queue(build.seat(1, registered_at=1 * DAY))
    assert trim(build, queue, ready=("hk1",)) == []


def test_squat_trim_keeps_a_seat_that_is_not_old_enough(build):
    queue = build.queue(build.seat(1, registered_at=int(8.5 * DAY)))
    assert trim(build, queue) == [], "age 1.5 days is less than 2 x 1 day"


def test_squat_trim_expected_wait_grows_with_seats_ahead(build):
    seats = [build.seat(uid, block=uid, registered_at=5 * DAY) for uid in range(1, 102)]
    queue = build.queue(*seats)
    moved = trim(build, queue, turns=24)
    # 24 turns per day = 1 hour per turn. Age is 5 days = 120 hours.
    # A seat with k seats ahead waits max(24, k) hours, and moves when 120 > 2 * that, so k < 60.
    assert moved == [(uid, uid) for uid in range(1, 61)]


def test_squat_trim_does_nothing_without_a_turn_rate(build):
    queue = build.queue(build.seat(1, registered_at=1))
    assert trim(build, queue, turns=0) == []


def test_squat_trim_ignores_seats_with_an_unknown_age(build):
    queue = build.queue(build.seat(1, registered_at=0))
    assert trim(build, queue) == []


def test_squat_trim_only_looks_at_the_normal_zone(build):
    queue = build.queue(build.seat(1, registered_at=1, demoted=True), build.seat(2, registered_at=1, front=True))
    assert trim(build, queue) == []


def snapshot(build, *regs, netuid=120):
    return ChainSnapshot(netuid, 5000, 9000, tuple(regs))


def sync(build, queue, snap, served_regs=(), served_hotkeys=(), **kw):
    return plan_sync(queue, served_regs, served_hotkeys, snap, netuid=120, burn_uid=0, **kw)


def test_sync_adds_new_registrations_in_turn_order(build):
    plan = sync(build, {}, snapshot(build, build.reg(7, 30), build.reg(3, 10), build.reg(5, 10)))
    assert [reg.uid for reg in plan.added] == [3, 5, 7]


def test_sync_never_queues_uid_zero(build):
    plan = sync(build, {}, snapshot(build, build.reg(0, 1), build.reg(4, 2)))
    assert [reg.uid for reg in plan.added] == [4]


def test_sync_never_requeues_a_served_seat_or_a_served_hotkey(build):
    snap = snapshot(build, build.reg(1, 10), build.reg(2, 20, hotkey="used"), build.reg(3, 30))
    plan = sync(build, {}, snap, served_regs=[(1, 10)], served_hotkeys=["used"])
    assert [reg.uid for reg in plan.added] == [3]


def test_a_new_registration_block_on_the_same_uid_is_a_new_seat(build):
    plan = sync(build, {}, snapshot(build, build.reg(1, 99, hotkey="fresh")), served_regs=[(1, 10)])
    assert [reg.key for reg in plan.added] == [(1, 99)]


def test_sync_removes_a_seat_that_left_the_chain_but_not_a_leased_one(build):
    queue = build.queue(build.seat(1), build.seat(2, leased=True), build.seat(3))
    plan = sync(build, queue, snapshot(build, build.reg(3)))
    assert plan.removed == ((1, 101),)


def test_sync_follows_a_hotkey_swap_on_a_waiting_seat(build):
    queue = build.queue(build.seat(1))
    plan = sync(build, queue, snapshot(build, build.reg(1, hotkey="swapped")))
    assert [reg.hotkey for reg in plan.rehotkeyed] == ["swapped"] and not plan.added and not plan.removed


def test_sync_with_nothing_to_do_is_empty(build):
    queue = build.queue(build.seat(1))
    assert sync(build, queue, snapshot(build, build.reg(1))).empty


def test_sync_refuses_a_snapshot_of_another_subnet_or_an_empty_one(build):
    with pytest.raises(InfraError):
        sync(build, {}, snapshot(build, build.reg(1), netuid=1))
    with pytest.raises(InfraError):
        sync(build, {}, snapshot(build))


def test_sync_refuses_to_drop_most_of_the_paid_queue_on_one_read(build):
    queue = build.queue(*(build.seat(uid) for uid in range(1, 21)))
    partial_read = snapshot(build, *(build.reg(uid) for uid in range(1, 11)))
    with pytest.raises(InfraError):
        sync(build, queue, partial_read)
    assert len(sync(build, queue, partial_read, force=True).removed) == 10
