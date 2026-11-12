"""Deadlines, retry spacing and cost are a fold of the journal."""
from __future__ import annotations

import pytest

from .timers import SystemClock, TimerConfig, Timers


def timers(**over) -> Timers:
    cfg = dict(lease_ttl_s=1000.0, heartbeat_grace_s=100.0, rent_backoff_s=30.0, renew_every_s=50.0, absent_polls=2)
    cfg.update(over)
    return Timers(TimerConfig(**cfg))


def bound(fence=1, box='box-1', price=12.0):
    return {'fence': fence, 'box_id': box, 'price': price, 'gpu': 'RTX5090'}


def test_lease_deadline_is_the_sooner_of_ttl_and_heartbeat_grace():
    t = timers()
    assert t.deadline(1) is None
    t.fold('box_bound', bound(), 500.0)
    assert t.deadline(1) == 600.0

    # A heartbeat with a new sequence number moves the deadline.
    assert t.heartbeat(1, 1, 550.0) == 650.0
    assert t.deadline(1) == 650.0
    # The same or an older sequence number is no progress.
    assert t.heartbeat(1, 1, 640.0) is None and t.heartbeat(1, 0, 640.0) is None
    assert t.deadline(1) == 650.0
    # Progress is kept in memory at once, but a journal record is due only
    # every ``renew_every_s``.
    assert t.heartbeat(1, 2, 560.0) is None
    assert t.deadline(1) == 660.0
    assert t.heartbeat(1, 3, 601.0) == 701.0

    # The hard deadline: no heartbeat can move the lease past bound + ttl.
    assert t.heartbeat(1, 9, 1450.0) == 1500.0
    assert t.deadline(1) == 1500.0
    assert t.heartbeat(1, 10, 1499.0) is None
    assert t.deadline(1) == 1500.0


def test_a_heartbeat_for_an_unknown_fence_is_ignored():
    t = timers()
    assert t.heartbeat(5, 1, 10.0) is None
    assert t.deadline(5) is None


def test_the_lease_ends_with_the_attempt():
    for kind in ('attempt_failed', 'decided'):
        t = timers()
        t.fold('box_bound', bound(), 500.0)
        t.heartbeat(1, 1, 510.0)
        t.fold(kind, {'fence': 1}, 520.0)
        assert t.deadline(1) is None and t.heartbeat(1, 2, 530.0) is None


def test_a_restart_gives_one_fresh_grace_period_but_never_more_than_the_ttl():
    records = [
        ('controller_started', {}, 0.0),
        ('box_bound', bound(), 500.0),
        ('lease_renewed', {'fence': 1, 'hb_seq': 4, 'until': 640.0}, 540.0),
    ]
    t = timers()
    for kind, body, at in records:
        t.fold(kind, body, at)
    assert t.deadline(1) == 640.0

    # The controller was down from 560 to 900. It could not see heartbeats.
    t = timers()
    for kind, body, at in records + [('controller_started', {}, 900.0)]:
        t.fold(kind, body, at)
    assert t.deadline(1) == 1000.0
    # A heartbeat that is not newer than the journal knows is still no progress.
    assert t.heartbeat(1, 4, 950.0) is None
    assert t.heartbeat(1, 5, 950.0) == 1050.0

    # A restart close to the hard deadline does not extend it.
    t = timers()
    for kind, body, at in records + [('controller_started', {}, 1490.0)]:
        t.fold(kind, body, at)
    assert t.deadline(1) == 1500.0


def test_a_box_is_gone_only_after_enough_listings_in_a_row_miss_it():
    t = timers(absent_polls=3)
    key = ('poll', 1)
    assert [t.presence(key, False), t.presence(key, False)] == [False, False]
    assert t.presence(key, True) is False
    assert [t.presence(key, False), t.presence(key, False), t.presence(key, False)] == [False, False, True]
    # A restart forgets the count: the next listing starts at one.
    t.fold('controller_started', {}, 10.0)
    assert t.presence(key, False) is False


def test_rent_backoff_comes_from_the_journal():
    t = timers()
    t.fold('rent_try', {'fence': 1, 'try': 3, 'rung': 1, 'backoff': False}, 100.0)
    assert t.due(('rent', 1, 3), 100.0)
    # The ladder wrapped around: wait before the next round.
    t.fold('rent_try', {'fence': 1, 'try': 4, 'rung': 0, 'backoff': True}, 100.0)
    assert not t.due(('rent', 1, 4), 129.9)
    assert t.due(('rent', 1, 4), 130.0)


def test_retry_spacing_and_forget():
    t = timers()
    assert t.due(('poll', 1), 0.0)
    t.defer(('poll', 1), 50.0)
    t.defer(('gc', 'b', 1), 50.0)
    assert not t.due(('poll', 1), 49.0) and t.due(('poll', 1), 50.0)
    t.forget(frozenset({('poll', 1)}))
    assert not t.due(('poll', 1), 49.0) and t.due(('gc', 'b', 1), 0.0)


def test_cost_is_price_times_the_time_a_box_was_open():
    t = timers()
    assert t.cost(0.0) == {
        'usd_spent_estimate': 0.0, 'usd_per_hour_now': 0.0, 'boxes_alive': 0,
        'boxes_alive_unpriced': 0, 'boxes_total': 0,
    }
    t.fold('box_bound', bound(1, 'a', 12.0), 0.0)
    t.fold('box_bound', bound(2, 'b', 40.0), 1800.0)
    cost = t.cost(3600.0)
    assert cost['usd_spent_estimate'] == 12.0 + 20.0
    assert cost['usd_per_hour_now'] == 52.0 and cost['boxes_alive'] == 2 and cost['boxes_total'] == 2

    # The end of the lease does not stop the meter. Only "the provider does
    # not list the box any more" does.
    t.fold('decided', {'fence': 1}, 3600.0)
    t.fold('box_doomed', {'box_id': 'a'}, 3600.0)
    assert t.cost(7200.0)['usd_spent_estimate'] == 24.0 + 60.0
    t.fold('box_gone', {'box_id': 'a'}, 7200.0)
    cost = t.cost(7200.0)
    assert cost['usd_spent_estimate'] == 24.0 + 60.0
    assert cost['usd_per_hour_now'] == 40.0 and cost['boxes_alive'] == 1 and cost['boxes_total'] == 2
    # A second "gone" for the same box changes nothing.
    t.fold('box_gone', {'box_id': 'a'}, 9000.0)
    assert t.cost(7200.0)['usd_spent_estimate'] == 24.0 + 60.0


def test_a_box_without_a_lease_is_counted_but_not_priced():
    t = timers()
    t.fold('box_doomed', {'box_id': 'orphan'}, 10.0)
    cost = t.cost(3610.0)
    assert cost['boxes_alive'] == 1 and cost['boxes_alive_unpriced'] == 1 and cost['boxes_total'] == 1
    assert cost['usd_spent_estimate'] == 0.0
    t.fold('box_gone', {'box_id': 'orphan'}, 3610.0)
    assert t.cost(3610.0)['boxes_alive'] == 0


def test_last_record_time_never_goes_back():
    t = timers()
    t.fold('controller_started', {}, 100.0)
    t.fold('batch_admitted', {}, 90.0)
    assert t.last_at == 100.0


@pytest.mark.parametrize('over', [{'lease_ttl_s': 0}, {'heartbeat_grace_s': -1}, {'absent_polls': 0}])
def test_bad_timer_settings_are_refused(over):
    with pytest.raises(ValueError):
        timers(**over)


def test_system_clock_does_not_go_back():
    clock = SystemClock()
    first = clock.now()
    assert clock.now() >= first
