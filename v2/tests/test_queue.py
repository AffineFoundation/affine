from __future__ import annotations

import pytest

from sn120_v2.errors import InfraError
from sn120_v2.queue import plan_sync, select_round, split_among_boxes, squat_trim, waiting
from sn120_v2.types import ChainSnapshot, Verdict

DAY = 86400


def uids(seats) -> list[int]:
    return [seat.reg.uid for seat in seats]


def pick(build, queue, ready, cap, freeze=None):
    return select_round(queue, build.verdicts(*ready), freeze or (lambda reg: build.batch(reg)), cap)


def test_order_is_registration_block_then_uid_with_front_seats_first_and_demoted_seats_last(build):
    queue = build.queue(build.seat(9, block=20), build.seat(5, block=10), build.seat(3, block=10), build.seat(7, block=15))
    assert uids(waiting(queue)) == [3, 5, 7, 9]
    queue = build.queue(
        build.seat(1, block=10), build.seat(2, block=20), build.seat(8, block=80, front=True), build.seat(6, block=60, front=True),
        build.seat(4, block=5, demoted=True), build.seat(12, block=1, demoted=True, front=True), build.seat(13, leased=True),
    )
    assert uids(waiting(queue)) == [12, 6, 8, 1, 2, 4], "front seats keep their original order; a leased seat is not waiting"


def test_a_round_takes_up_to_the_cap_in_queue_order_and_a_short_queue_runs_with_what_is_ready(build):
    queue = build.queue(*(build.seat(uid) for uid in (5, 1, 4, 2, 3)))
    picked = pick(build, queue, ("hk1", "hk2", "hk3", "hk4", "hk5"), cap=3)
    assert [e.reg.uid for e in picked.entries] == [1, 2, 3] and [e.queue_pos for e in picked.entries] == [0, 1, 2]
    assert [e.reg.uid for e in pick(build, queue, ("hk4", "hk5"), cap=32).entries] == [4, 5]
    assert pick(build, queue, (), cap=5).entries == ()


def test_a_seat_without_a_valid_file_is_skipped_and_keeps_its_place(build):
    queue = build.queue(build.seat(1), build.seat(2), build.seat(3), build.seat(4))
    calls = []

    def freeze(reg):
        calls.append(reg.uid)
        return None if reg.uid == 3 else build.batch(reg)

    verdicts = build.verdicts("hk1", "hk3", "hk4") | {"hk2": Verdict("hk2", False, reason="bad json")}
    picked = select_round(queue, verdicts, freeze, cap=2)
    assert [e.reg.uid for e in picked.entries] == [1, 4]
    assert picked.skipped == ((2, 102), (3, 103)), "an invalid live check and a failed freeze are both skips"
    assert calls == [1, 3, 4], "freeze is called only for files the live check calls valid"
    assert uids(waiting(queue)) == [1, 2, 3, 4], "selection never changes the queue"


def test_one_entry_per_miner_per_round_and_held_or_leased_seats_are_passed_over(build):
    same_hotkey = build.queue(build.seat(1, block=10, hotkey="same"), build.seat(2, block=20, hotkey="same"), build.seat(3, block=30))
    picked = pick(build, same_hotkey, ("same", "hk3"), cap=5)
    assert [e.reg.uid for e in picked.entries] == [1, 3] and picked.skipped == (), "the second seat waits; it is not a skip"
    same_owner = build.queue(build.seat(1, coldkey="owner"), build.seat(2, coldkey="owner"), build.seat(3), build.seat(4))
    assert [e.reg.uid for e in pick(build, same_owner, ("hk1", "hk2", "hk3", "hk4"), cap=5).entries] == [1, 3, 4]
    parked = build.queue(build.seat(1, held=True), build.seat(2, leased=True), build.seat(3), build.seat(9, front=True))
    assert [e.reg.uid for e in pick(build, parked, ("hk1", "hk2", "hk3", "hk9"), cap=5).entries] == [9, 3]


@pytest.mark.parametrize("n,boxes,sizes", [(5, 1, [5]), (5, 2, [3, 2]), (6, 3, [2, 2, 2]), (2, 4, [1, 1])])
def test_boxes_get_consecutive_blocks_of_seats(build, n, boxes, sizes):
    blocks = split_among_boxes(tuple(build.entry(i) for i in range(n)), boxes)
    assert [len(block) for block in blocks] == sizes
    assert [e.queue_pos for block in blocks for e in block] == list(range(n))


def trim(build, queue, ready=(), now=10 * DAY, turns=24):
    return squat_trim(queue, build.verdicts(*ready), now, turns, min_wait_s=DAY, age_factor=2.0)


def test_squat_trim_moves_only_an_old_seat_with_no_valid_file(build):
    old = build.queue(build.seat(1, registered_at=7 * DAY))
    assert trim(build, old) == [(1, 101)], "age 3 days is more than 2 x the 1 day minimum wait"
    assert trim(build, old, ready=("hk1",)) == [], "a seat with a valid file is never moved"
    assert trim(build, old, turns=0) == [], "no turn rate, no move"
    assert trim(build, build.queue(build.seat(1, registered_at=int(8.5 * DAY)))) == [], "age 1.5 days is not old enough"
    assert trim(build, build.queue(build.seat(1, registered_at=0))) == [], "unknown age"
    other_zones = build.queue(build.seat(1, registered_at=1, demoted=True), build.seat(2, registered_at=1, front=True))
    assert trim(build, other_zones) == []


def test_squat_trim_expected_wait_grows_with_the_seats_ahead(build):
    queue = build.queue(*(build.seat(uid, block=uid, registered_at=5 * DAY) for uid in range(1, 102)))
    # 24 turns per day = 1 hour per turn. Age is 5 days = 120 hours.
    # A seat with k seats ahead waits max(24, k) hours, and moves when 120 > 2 * that, so k < 60.
    assert trim(build, queue, turns=24) == [(uid, uid) for uid in range(1, 61)]


def sync(build, queue, regs, served_regs=(), served_hotkeys=(), netuid=120, **kw):
    return plan_sync(queue, served_regs, served_hotkeys, ChainSnapshot(netuid, 5000, 9000, tuple(regs)), netuid=120, burn_uid=0, **kw)


def test_sync_adds_new_seats_in_turn_order_and_never_a_served_seat_a_served_hotkey_or_uid_zero(build):
    regs = [build.reg(0, 1), build.reg(7, 30), build.reg(3, 10), build.reg(5, 10), build.reg(1, 10), build.reg(2, 20, hotkey="used")]
    plan = sync(build, {}, regs, served_regs=[(1, 10)], served_hotkeys=["used"])
    assert [reg.uid for reg in plan.added] == [3, 5, 7]
    again = sync(build, {}, [build.reg(1, 99, hotkey="fresh")], served_regs=[(1, 10)])
    assert [reg.key for reg in again.added] == [(1, 99)], "a new registration block on the same uid is a new seat"


def test_sync_removes_a_seat_that_left_the_chain_and_follows_a_hotkey_swap(build):
    queue = build.queue(build.seat(1), build.seat(2, leased=True), build.seat(3))
    plan = sync(build, queue, [build.reg(3, hotkey="swapped")])
    assert plan.removed == ((1, 101),), "a leased seat stays until its round ends"
    assert [reg.hotkey for reg in plan.rehotkeyed] == ["swapped"] and not plan.added
    assert sync(build, build.queue(build.seat(1)), [build.reg(1)]).empty


def test_sync_refuses_a_bad_chain_read_so_it_cannot_drop_the_paid_queue(build):
    queue = build.queue(*(build.seat(uid) for uid in range(1, 21)))
    half = [build.reg(uid) for uid in range(1, 11)]
    for bad in (dict(regs=half), dict(regs=[]), dict(regs=half, netuid=1)):
        with pytest.raises(InfraError):
            sync(build, queue, **bad)
    assert len(sync(build, queue, half, force=True).removed) == 10
