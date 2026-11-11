"""Fault-injection tests: the real controller in a hostile fake world.

The code under test is real: ``Controller``, ``Effects``, ``Journal``,
``core``, ``guard``, ``layout`` and ``timers``. Only the world around it is
fake (``sim.py``). ``Auditor`` in ``sim.py`` holds the invariants.

More seeds: ``BATCHCTL_SIM_SEEDS=20000``. Every crash point of every sweep:
``BATCHCTL_SIM_FULL_SWEEP=1``.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import os
import random

import pytest

from . import controller as K
from . import core as C
from . import effects as E
from . import guard as G
from . import journal as J
from . import sim as S
from . import timers as T
from .layout import parse_head
from .sim import AuditFailure, Harness, Scenario

SEEDS = int(os.environ.get('BATCHCTL_SIM_SEEDS', '2000'))
BLOCKS = 20
FULL_SWEEP = bool(os.environ.get('BATCHCTL_SIM_FULL_SWEEP'))

JOURNAL_STEPS = {
    'journal.temp.before', 'journal.temp.after',
    'journal.rename.before', 'journal.rename.after',
    'journal.dirsync.before', 'journal.dirsync.after',
    'journal.write.before', 'journal.write.torn', 'journal.write.after',
    'journal.fsync.before', 'journal.fsync.after',
}
TERMINAL = {phase.value for phase in C.TERMINAL}


def run(tmp_path, seed: int, sc: Scenario, tag: str = '') -> Harness:
    """Run one world. A broken invariant fails the test with the seed."""
    harness = Harness(tmp_path, seed, sc, tag)
    try:
        harness.report = harness.run()
    except AuditFailure as exc:
        raise AssertionError('seed=%d %r: %s' % (seed, sc, exc)) from exc
    return harness


def scenario_for(seed: int) -> Scenario:
    """A random world. The seed fixes everything, so a failure repeats."""
    rng = random.Random(seed * 1000003 + 17)
    return Scenario(
        batches=rng.randint(1, 7),
        bad_refs=rng.choice((0, 0, 2, 5)),
        max_boxes=rng.randint(1, 4),
        max_infra_failures=rng.randint(1, 3),
        green_pct=rng.choice((0, 30, 60, 90, 100)),
        deterministic_training=rng.random() < 0.5,
        dry_rungs=rng.choice((0, 0, 1, 3)),
        fault_rate=rng.choice((0.0, 0.02, 0.05, 0.1, 0.2)),
        fault_budget=rng.choice((10, 40, 120)),
        crash_rate=rng.choice((0.0, 0.002, 0.01, 0.03)),
        crash_budget=rng.choice((0, 3, 10)),
        apply_verify=rng.choice((E.APPLY_VERIFY_SHA256, E.APPLY_VERIFY_ETAG)),
        source_forgets=rng.random() < 0.5,
        chaos_steps=rng.choice((200, 500, 900)),
        max_jump_s=rng.choice((5.0, 40.0, 120.0)),
    )


# ----------------------------------------------------------------------------
# A world that works
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('chaos', [False, True])
def test_clean_world_decides_every_batch(tmp_path, chaos):
    for seed in range(20):
        sc = Scenario(batches=6, max_boxes=3, chaos=chaos)
        report = run(tmp_path, seed, sc, '-%s' % chaos).report
        assert report.halted is None and report.crashes == 0 and not report.faults
        assert sum(report.outcomes.values()) == 6 and set(report.outcomes) <= TERMINAL
        # Each apply moved the HEAD by one. Genesis is HEAD 0.
        assert report.head_seq == report.outcomes.get('applied', 0)
        if not chaos:
            # Every box got its turn in time, so no attempt is lost. (The
            # random scheduler can starve a box until its lease expires.)
            assert 'lost_infra' not in report.outcomes and 'attempt_failed' not in report.kinds
            assert report.kinds['box_bound'] == report.kinds['lease_issued'] == 6


def test_same_seed_gives_the_same_journal(tmp_path):
    sc = Scenario(batches=5, fault_rate=0.08, fault_budget=40, crash_rate=0.004, crash_budget=6)
    for seed in range(5):
        first = run(tmp_path, seed, sc, '-a').world
        second = run(tmp_path, seed, sc, '-b').world
        assert bytes(first.files.files[first.journal_path].data) == \
            bytes(second.files.files[second.journal_path].data)


def test_all_greens_on_one_head_are_paid_and_one_is_applied(tmp_path):
    # Every batch is GREEN and all boxes run at the same time.
    applied = credited = 0
    for seed in range(30):
        report = run(tmp_path, seed, Scenario(batches=6, max_boxes=6, green_pct=100)).report
        # (The random scheduler can starve a box until its lease expires.)
        assert set(report.outcomes) <= {'applied', 'credited', 'lost_infra'}
        assert report.outcomes['applied'] == report.head_seq >= 1
        applied += report.outcomes['applied']
        credited += report.outcomes.get('credited', 0)
    assert credited > applied


def test_bad_batch_references_are_recorded_and_do_not_block_the_queue(tmp_path):
    for seed in range(20):
        harness = run(tmp_path, seed, Scenario(batches=4, bad_refs=5, source_forgets=seed % 2 == 0))
        report, world = harness.report, harness.world
        assert report.kinds['batch_rejected'] == 5
        assert sum(report.outcomes.values()) == 4 and set(report.outcomes) <= TERMINAL
        records, _valid = J.scan(bytes(world.files.files[world.journal_path].data))
        reasons = sorted(rec['b']['reason'].split(':')[0] for rec in records if rec['t'] == 'batch_rejected')
        assert reasons == ['bad_batch_ref'] * 4 + ['batch_id_reused']
        # The batch keeps its first owner. The second use of its id is on record.
        first = harness.ctl.state.batch('b000')
        assert (first.miner, first.ref) == ('miner0', 'ref-0')


def test_dry_gpu_classes_fall_through_to_the_next_rung(tmp_path):
    for seed in range(20):
        harness = run(tmp_path, seed, Scenario(batches=4, dry_rungs=3, chaos=False))
        report = harness.report
        assert sum(report.outcomes.values()) == 4 and set(report.outcomes) <= TERMINAL
        # A rental that finds no offer is not an infra loss of the batch.
        assert 'lost_infra' not in report.outcomes and 'attempt_failed' not in report.kinds
        assert report.kinds['rent_try'] >= 3 * report.kinds['lease_issued']
        gpus = {box.gpu for p in harness.world.providers.values() for box in p.boxes.values() if box.ours}
        assert gpus == {'B300'}


# ----------------------------------------------------------------------------
# Thousands of random hostile worlds
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('block', range(BLOCKS))
def test_random_hostile_worlds_keep_every_invariant(tmp_path, block):
    size = max(1, SEEDS // BLOCKS)
    totals: collections.Counter = collections.Counter()
    for seed in range(block * size, (block + 1) * size):
        report = run(tmp_path, seed, scenario_for(seed)).report
        totals.update(report.outcomes)
        totals['crashes'] += report.crashes
        totals['faults'] += sum(report.faults.values())
        assert report.halted is None
    if size >= 50:
        # The block is not empty of events: it had every kind of outcome.
        for name in ('applied', 'credited', 'red', 'lost_infra', 'crashes', 'faults'):
            assert totals[name] > 0, (name, dict(totals))


# ----------------------------------------------------------------------------
# A crash at every step
# ----------------------------------------------------------------------------

SWEEPS = {
    'fair': (0, Scenario(batches=3, chaos=False), 1),
    'faulty': (3, Scenario(batches=3, fault_rate=0.06, fault_budget=25, chaos_steps=250), 4),
}


@pytest.mark.parametrize('name', sorted(SWEEPS))
def test_controller_crash_at_every_step_recovers(tmp_path, name):
    seed, sc, stride = SWEEPS[name]
    stride = 1 if FULL_SWEEP else stride
    base = run(tmp_path, seed, sc, '-base').report
    assert base.points > 300
    labels = set()
    points = range(1, base.points + 1, stride)
    twice = 0
    for point in points:
        # A process kill, a power loss, and a second crash during recovery.
        for power_loss, again in ((False, None), (True, None), (point % 2 == 0, 1 + point % 12)):
            crashed = dataclasses.replace(sc, crash_at=point, crash_again=again, power_loss=power_loss)
            report = run(tmp_path, seed, crashed, '-%d-%d-%s' % (point, power_loss, again)).report
            assert report.crashes >= 1
            assert report.halted is None
            assert sum(report.outcomes.values()) == 3 and set(report.outcomes) <= TERMINAL
            labels.update(report.crash_labels)
            twice += report.crashes >= 2
    # Nearly every run of the third kind died twice. (A crash at the very
    # end leaves too few steps for a second one.)
    assert twice >= 0.9 * len(points)
    if stride == 1:
        # The sweep died in every kind of journal step.
        assert JOURNAL_STEPS <= labels
    assert {'journal.write.torn', 'journal.fsync.before', 'journal.fsync.after'} <= labels


# ----------------------------------------------------------------------------
# The journal on a real disk
# ----------------------------------------------------------------------------

def check_real_journal(harness: Harness) -> None:
    path = harness.root_path / G.JOURNAL
    count, _last = J.verify_file(path)
    assert count == harness.report.records
    assert path.read_bytes() == bytes(harness.world.files.files[str(path)].data)
    # No temporary file is left, and no file that the controller does not own.
    assert {entry.name for entry in harness.root_path.iterdir()} <= G.STATE_FILES


def test_random_hostile_worlds_with_the_journal_on_a_real_disk(tmp_path):
    crashes = 0
    for seed in range(max(40, SEEDS // 10)):
        harness = run(tmp_path, seed, dataclasses.replace(scenario_for(seed), real_disk=True))
        crashes += harness.report.crashes
        check_real_journal(harness)
    assert crashes > 0


def test_controller_crash_at_every_step_recovers_on_a_real_disk(tmp_path):
    seed, sc, _stride = SWEEPS['fair']
    sc = dataclasses.replace(sc, real_disk=True)
    base = run(tmp_path, seed, sc, '-base').report
    labels = set()
    for point in range(1, base.points + 1, 1 if FULL_SWEEP else 3):
        for power_loss in (False, True):
            crashed = dataclasses.replace(sc, crash_at=point, crash_again=1 + point % 7, power_loss=power_loss)
            harness = run(tmp_path, seed, crashed, '-%d-%d' % (point, power_loss))
            assert harness.report.crashes >= 1 and harness.report.halted is None
            assert sum(harness.report.outcomes.values()) == 3
            check_real_journal(harness)
            labels.update(harness.report.crash_labels)
    assert {'journal.write.torn', 'journal.fsync.before', 'journal.rename.before'} <= labels


# ----------------------------------------------------------------------------
# A bug in an effect
# ----------------------------------------------------------------------------

def test_a_transient_bug_in_an_effect_is_retried_and_decides_nothing(tmp_path):
    for seed in range(30):
        sc = Scenario(batches=4, judge_bug=S.JUDGE_BUG_TRANSIENT, chaos=seed % 2 == 0)
        harness = run(tmp_path, seed, sc)
        report = harness.report
        assert report.halted is None and 'halted' not in report.kinds
        assert sum(report.outcomes.values()) == 4 and set(report.outcomes) <= TERMINAL
        assert harness.world.judge.raised == S.SimJudge.TRANSIENT_CALLS
        assert harness.ctl.status()['alerts']['counts']['effect_error'] == S.SimJudge.TRANSIENT_CALLS
        if not sc.chaos:
            # The bug cost no attempt: each batch needed one box only.
            assert 'attempt_failed' not in report.kinds and report.kinds['box_bound'] == 4


def test_a_bug_on_one_batch_halts_loudly_and_judges_nothing(tmp_path):
    for seed in range(30):
        sc = Scenario(batches=5, max_boxes=3, judge_bug=S.JUDGE_BUG_POISON, chaos=seed % 2 == 0)
        harness = run(tmp_path, seed, sc)
        report, world = harness.report, harness.world
        records, _valid = J.scan(bytes(world.files.files[world.journal_path].data))
        halts = [rec['b']['reason'] for rec in records if rec['t'] == 'halted']
        # The controller stopped new work and showed why. The operator
        # fixed the judge and resumed. Then every batch got its true verdict
        # (the auditor compares each verdict with the truth).
        # (In the random part, a report of the bug can arrive after the
        # resume and halt again, and a starved box can lose the batch before
        # the judge sees it.)
        assert set(halts) <= {'effect_error:settle'} and report.kinds.get('resumed', 0) == len(halts)
        assert report.halted is None
        assert sum(report.outcomes.values()) == 5 and set(report.outcomes) <= TERMINAL
        if not sc.chaos:
            # The bug cost the batch no attempt and gave it no verdict: its
            # first box led to its one decision.
            assert len(halts) == 1 and world.judge.raised == harness.config.effect_error_limit
            poisoned = [rec['t'] for rec in records if rec['b'].get('batch_id') == S.SimJudge.POISON_BATCH]
            assert poisoned.count('decided') == poisoned.count('box_bound') == 1
            assert 'attempt_failed' not in poisoned


# ----------------------------------------------------------------------------
# Someone else writes into the prefix
# ----------------------------------------------------------------------------

FOREIGN = {
    'between-steps': Scenario(batches=5, foreign_write_at=150),
    'between-steps-faulty': Scenario(
        batches=5, foreign_write_at=150, fault_rate=0.05, fault_budget=40, crash_rate=0.004, crash_budget=6,
    ),
    'race-with-the-swap': Scenario(batches=5, foreign_race=True),
    'race-with-the-swap-faulty': Scenario(
        batches=5, foreign_race=True, fault_rate=0.05, fault_budget=40, crash_rate=0.004, crash_budget=6,
    ),
}


@pytest.mark.parametrize('name', sorted(FOREIGN))
def test_foreign_pointer_write_halts_and_is_never_overwritten(tmp_path, name):
    halted = 0
    for seed in range(100):
        harness = run(tmp_path, seed, FOREIGN[name])
        world, report = harness.world, harness.report
        if not world.audit.foreign:
            assert report.halted is None
            continue
        # The foreign pointer is still there: the controller did not win.
        assert parse_head(world.raw.objs[world.layout.head]).link == 'e' * 64
        if report.halted is not None:
            halted += 1
            assert report.halted == 'pointer_moved'
            if report.head_seq >= 0:
                # The first GREEN that could not be applied is still on record.
                assert report.outcomes.get('applying', 0) == 1
            else:
                # The foreign writer made the pointer before the first HEAD.
                assert set(report.outcomes) <= {'queued'}
        else:
            # No apply was tried after the foreign write.
            assert 'applying' not in report.outcomes and 'green' not in report.outcomes
    assert halted >= 20


TAMPERS = {
    S.TAMPER_PENDING: {'state_unverified'},
    S.TAMPER_CANONICAL: {'canonical_object_conflict', 'state_unverified'},
    S.TAMPER_MANIFEST: {'base_manifest_mismatch'},
}


@pytest.mark.parametrize('kind', sorted(TAMPERS))
def test_damaged_state_in_r2_is_never_applied_and_halts(tmp_path, kind):
    reasons: collections.Counter = collections.Counter()
    for seed in range(150):
        sc = Scenario(batches=5, tamper=kind, fault_rate=0.02, fault_budget=10)
        harness = run(tmp_path, seed, sc)
        if harness.report.halted is not None:
            assert harness.world.tampered
            reasons[harness.report.halted.split(':')[0]] += 1
    assert set(reasons) <= TAMPERS[kind], reasons
    assert sum(reasons.values()) >= 30, reasons


def test_damaged_head_state_never_penalises_a_miner(tmp_path):
    lost = 0
    for seed in range(60):
        sc = Scenario(batches=5, tamper=S.TAMPER_HEAD)
        harness = run(tmp_path, seed, sc)
        report = harness.report
        # No box can train on a damaged HEAD. The batches end as lost_infra,
        # never as RED, and the queue is not blocked.
        assert report.halted is None and set(report.outcomes) <= TERMINAL
        lost += report.outcomes.get('lost_infra', 0)
    assert lost > 60


def test_breaker_halts_on_a_damaged_head_state(tmp_path):
    halted = 0
    for seed in range(60):
        sc = Scenario(batches=10, max_infra_failures=3, tamper=S.TAMPER_HEAD, breaker_failures=2)
        report = run(tmp_path, seed, sc).report
        if report.halted is not None:
            halted += 1
            assert report.halted == 'infra_breaker:2_lost_in_a_row'
            # The halt came before any batch could reach the terminal rule
            # through the damaged HEAD: the batches wait in the queue.
            assert report.outcomes.get('queued', 0) >= 1
    # (No halt: a GREEN that was already in flight replaced the damaged HEAD.)
    assert halted >= 20


def test_operator_resume_after_a_breaker_halt_finishes_every_batch(tmp_path):
    tripped = 0
    for seed in range(60):
        sc = Scenario(
            batches=6, max_infra_failures=3, fault_rate=0.3, fault_budget=300,
            breaker_failures=2, operator_resumes=True,
        )
        report = run(tmp_path, seed, sc).report
        assert report.halted is None
        assert sum(report.outcomes.values()) == 6 and set(report.outcomes) <= TERMINAL
        assert report.kinds.get('halted', 0) == report.kinds.get('resumed', 0)
        tripped += report.kinds.get('halted', 0) > 0
    assert tripped >= 20


# ----------------------------------------------------------------------------
# Status: boxes and cost
# ----------------------------------------------------------------------------

def test_status_shows_boxes_and_cost(tmp_path):
    harness = Harness(tmp_path, 7, Scenario(batches=5, max_boxes=3, chaos=False))
    harness._boot()
    harness.fair(4)
    status = harness.ctl.status()
    assert status['mode'] == 'shadow' and status['halted'] is None
    assert status['boxes']['max_boxes'] == 3
    assert 1 <= status['boxes']['slots_in_use'] <= 3
    assert status['cost']['boxes_alive'] == status['boxes']['running'] >= 1
    assert status['cost']['usd_per_hour_now'] == 12.0 * status['cost']['boxes_alive']
    assert status['cost']['new_rentals_on_hold'] is False
    assert status['batches']['total'] == 5 and status['batches']['first_queued'] is not None

    report = harness.finish()
    status = harness.ctl.status()
    assert status['cost']['boxes_alive'] == 0 and status['cost']['usd_per_hour_now'] == 0
    assert status['cost']['boxes_total'] == report.kinds['box_bound'] == 5
    assert status['cost']['usd_spent_estimate'] > 0
    assert status['batches']['by_phase'] == report.outcomes
    assert status['head']['seq'] == report.head_seq

    harness.ctl.write_status()
    written = json.loads((harness.root_path / G.STATUS).read_text())
    assert written['head'] == status['head'] and written['cost'] == status['cost']


def test_cost_cap_holds_new_rentals_and_loses_no_batch(tmp_path):
    harness = Harness(tmp_path, 3, Scenario(batches=6, max_boxes=2, chaos=False, max_cost_usd=0.3))
    harness._boot()
    harness.fair(200)
    status = harness.ctl.status()
    assert status['cost']['new_rentals_on_hold'] is True
    assert status['cost']['usd_spent_estimate'] >= 0.3
    # The boxes that ran are finished and gone. The rest of the queue waits.
    assert status['cost']['boxes_alive'] == 0 and status['boxes']['slots_in_use'] == 0
    waiting = status['batches']['by_phase'].get('queued', 0)
    assert 1 <= waiting < 6 and status['batches']['total'] == 6
    records = harness.ctl.journal.count
    harness.fair(50)
    assert harness.ctl.journal.count == records

    # The operator starts the controller again with a higher cap.
    harness.config = dataclasses.replace(harness.config, max_cost_usd=None)
    harness._restart()
    report = harness.finish()
    assert sum(report.outcomes.values()) == 6 and set(report.outcomes) <= TERMINAL


# ----------------------------------------------------------------------------
# The harness has teeth: a seeded bug in the real code is found
# ----------------------------------------------------------------------------

FAULTY = Scenario(batches=5, fault_rate=0.08, fault_budget=60)
CRASHY = Scenario(batches=5, fault_rate=0.05, fault_budget=40, crash_rate=0.01, crash_budget=8)
CLEAN = Scenario(batches=5)


def expect_audit_failure(tmp_path, sc: Scenario, needles, seeds: int = 300) -> None:
    for seed in range(seeds):
        try:
            Harness(tmp_path, seed, sc).run()
        except AuditFailure as exc:
            assert any(needle in str(exc) for needle in needles), str(exc)
            return
    raise AssertionError('the seeded bug was not found in %d seeds' % seeds)


def test_harness_finds_a_copy_without_digest_check(tmp_path, monkeypatch):
    def adopt(self, src, dst, size, sha):
        staged = self._digest(src)
        if staged is E.RETRY:
            return E.RETRY, None
        if staged is None:
            return E.INVALID, 'object_missing'
        try:
            self.store.copy(src, dst, src_etag=staged.etag)
        except E.StoreError:
            return E.RETRY, None
        return E.OK, staged.etag

    monkeypatch.setattr(E.Effects, '_adopt_object', adopt)
    expect_audit_failure(tmp_path, FAULTY, ['copied bytes that the manifest does not name'])


def test_harness_finds_a_finding_from_one_bad_read(tmp_path, monkeypatch):
    # One wrong digest is taken as proof that the object is wrong.
    monkeypatch.setattr(E.Effects, '_same_again', lambda self, key, first: True)
    expect_audit_failure(tmp_path, dataclasses.replace(FAULTY, fault_rate=0.2, fault_budget=200), [
        'a valid upload was called invalid', 'a valid uploaded state was not recognized',
        'the controller halted without a cause', 'a package with a valid result was rejected',
    ])


def test_harness_finds_a_reused_batch_id_that_is_dropped_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(K.Controller, '_known_refs', lambda self: {})
    expect_audit_failure(
        tmp_path, Scenario(batches=4, bad_refs=2, source_forgets=True),
        ['a refused batch reference was not recorded'],
    )


def test_harness_finds_an_apply_of_a_partial_state(tmp_path, monkeypatch):
    # The state is not verified at all: not at settle, not before the swap.
    def adopt(self, src, dst, size, sha):
        staged = self._digest(src)
        if staged is E.RETRY or staged is None:
            return E.OK, 'none'
        try:
            self.store.copy(src, dst, src_etag=staged.etag)
        except E.StoreError:
            return E.RETRY, None
        return E.OK, staged.etag

    def verify(self, state_id, *, batch_id, parent, evidence, full):
        return E.OK, tuple(evidence)

    monkeypatch.setattr(E.Effects, '_adopt_object', adopt)
    monkeypatch.setattr(E.Effects, '_verify_state', verify)
    monkeypatch.setattr(S.Auditor, 'controller_copy', lambda self, src, dst: None)
    expect_audit_failure(tmp_path, FAULTY, [
        'state manifest written before file', 'does not hold in full',
    ])


def test_harness_finds_a_journal_without_fsync(tmp_path, monkeypatch):
    monkeypatch.setattr(S.MemOps, 'fsync', lambda self, fd: None)
    expect_audit_failure(tmp_path, CLEAN, ['without a durable'])


def _drop_pointer_condition(monkeypatch) -> None:
    real = G.GuardedStore.put

    def put(self, key, data, *, if_none_match=False, if_match=None):
        if key == self.prefix + 'HEAD.json':
            return real(self, key, data)
        return real(self, key, data, if_none_match=if_none_match, if_match=if_match)

    monkeypatch.setattr(G.GuardedStore, 'put', put)


def test_harness_finds_a_pointer_write_without_compare_and_swap(tmp_path, monkeypatch):
    _drop_pointer_condition(monkeypatch)
    expect_audit_failure(
        tmp_path, Scenario(batches=5, foreign_race=True), ['moved the pointer after a foreign write'],
    )


def test_harness_finds_a_late_pointer_write_without_compare_and_swap(tmp_path, monkeypatch):
    _drop_pointer_condition(monkeypatch)
    expect_audit_failure(
        tmp_path, Scenario(batches=6, fault_rate=0.15, fault_budget=80), ['HEAD chain is not linear'],
    )


def test_harness_finds_an_apply_that_is_not_idempotent(tmp_path, monkeypatch):
    real = E.parse_head

    def blind(raw):
        # The effect no longer sees that the pointer is already its own.
        pointer = real(raw)
        return dataclasses.replace(pointer, link='0' * 64) if pointer.seq > 0 else pointer

    monkeypatch.setattr(E, 'parse_head', blind)
    expect_audit_failure(tmp_path, CRASHY, ['halted without a cause', 'not every batch became terminal'])


def test_harness_finds_sloppy_box_ownership(tmp_path, monkeypatch):
    real = G.parse_box_name

    def sloppy(run_id, name):
        got = real(run_id, name)
        if got is None and isinstance(name, str) and name.startswith(G.own_prefix(run_id)):
            return (1, 0)
        return got

    monkeypatch.setattr(E, 'parse_box_name', sloppy)
    expect_audit_failure(tmp_path, CLEAN, ['destroyed a box that is not its own'])


def test_harness_finds_a_sweep_that_deletes_live_staging(tmp_path, monkeypatch):
    real = K.Controller._fence_view

    def blind(self):
        view = real(self)
        return E.FenceView(frozenset(), view.dirty, view.next_fence)

    monkeypatch.setattr(K.Controller, '_fence_view', blind)
    expect_audit_failure(tmp_path, CLEAN, ['deleted staging of a live fence'])


def test_harness_finds_a_complete_marker_of_another_fence(tmp_path, monkeypatch):
    real = E.parse_complete

    def any_fence(raw, *, batch_id, fence, base_state):
        doc = json.loads(raw)
        doc['fence'] = fence
        return real(json.dumps(doc).encode(), batch_id=batch_id, fence=fence, base_state=base_state)

    monkeypatch.setattr(E, 'parse_complete', any_fence)
    expect_audit_failure(tmp_path, FAULTY, ['invalid package'])


def test_harness_finds_a_rental_that_breaks_the_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(E.Effects, '_box_problem', lambda self, box, name, rung: None)
    expect_audit_failure(tmp_path, FAULTY, ['breaks the rental rules'])


def test_harness_finds_a_repair_that_cuts_a_durable_record(tmp_path, monkeypatch):
    real = J.scan

    def cut(data):
        records, valid = real(data)
        if len(records) > 3:
            return records[:-1], data.rfind(b'\n', 0, valid - 1) + 1
        return records, valid

    monkeypatch.setattr(J, 'scan', cut)
    expect_audit_failure(tmp_path, CRASHY, ['lost journal records that were fsynced'])


def test_harness_finds_a_lease_that_never_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(T.Timers, 'deadline', lambda self, fence: None)
    sc = Scenario(batches=4, fault_rate=0.1, fault_budget=60, max_rounds=400)
    expect_audit_failure(tmp_path, sc, ['not every batch became terminal'])


def test_harness_finds_a_missing_follow_up(tmp_path, monkeypatch):
    monkeypatch.setattr(K.C, 'resolve', lambda cfg, state: None)
    expect_audit_failure(tmp_path, Scenario(batches=3, max_rounds=300), ['not every batch became terminal'])


def _patch_decide(monkeypatch, change) -> None:
    real = C.decide

    def decide(cfg, state, ev):
        return change(real, cfg, state, ev)

    monkeypatch.setattr(K.C, 'decide', decide)


def test_harness_finds_a_miner_penalised_for_a_bad_upload(tmp_path, monkeypatch):
    def change(real, cfg, state, ev):
        if type(ev) is C.SettleInvalid:
            return real(cfg, state, C.Settled(ev.fence, C.VERDICT_RED, 'a' * 64))
        return real(cfg, state, ev)

    _patch_decide(monkeypatch, change)
    expect_audit_failure(tmp_path, FAULTY, ['invalid package', 'never finished an upload', 'names another result'])


def test_harness_finds_a_green_turned_red(tmp_path, monkeypatch):
    def change(real, cfg, state, ev):
        if type(ev) is C.Settled and ev.verdict == C.VERDICT_GREEN:
            return real(cfg, state, C.Settled(ev.fence, C.VERDICT_RED, ev.result_sha))
        return real(cfg, state, ev)

    _patch_decide(monkeypatch, change)
    expect_audit_failure(tmp_path, CLEAN, ['the verdict differs from the truth'])


def test_harness_finds_a_missing_terminal_rule(tmp_path, monkeypatch):
    def change(real, cfg, state, ev):
        rec = real(cfg, state, ev)
        if rec is not None and rec[0] == 'attempt_failed':
            rec[1]['terminal'] = False
        return rec

    _patch_decide(monkeypatch, change)
    expect_audit_failure(tmp_path, FAULTY, ['went on after N lost attempts'])


def test_harness_finds_a_lease_out_of_queue_order(tmp_path, monkeypatch):
    def last_queued(state):
        queued = [b for b in state.active.values() if b.phase is C.Phase.QUEUED]
        return max(queued, key=lambda b: b.seq) if queued else None

    monkeypatch.setattr(C, '_first_queued', last_queued)
    expect_audit_failure(tmp_path, Scenario(batches=6, max_boxes=1), ['not first in the queue'])


def test_core_refuses_an_apply_intent_for_a_stale_green(tmp_path, monkeypatch):
    # Defence in depth: even with a wrong ``resolve``, the record for an
    # apply of a GREEN that was not trained on the HEAD cannot be written.
    real = C.resolve

    def resolve(cfg, state):
        rec = real(cfg, state)
        if rec is None or rec[0] != 'credited' or rec[1]['reason'] != 'stale_base':
            return rec
        if state.pending is not None:
            return rec
        batch = state.active[rec[1]['batch_id']]
        decision = batch.decision
        seq = state.head.seq + 1
        link = C.compute_link_id(
            seq=seq, state_id=decision.state_id, parent=state.head.link_id,
            batch_id=batch.batch_id, result_sha=decision.result_sha, fence=decision.fence,
        )
        return 'apply_intent', {
            'batch_id': batch.batch_id, 'seq': seq, 'state_id': decision.state_id,
            'parent_link': state.head.link_id, 'link_id': link,
            'result_sha': decision.result_sha, 'fence': decision.fence,
            'evidence': [list(item) for item in decision.evidence],
        }

    monkeypatch.setattr(K.C, 'resolve', resolve)
    refused = 0
    for seed in range(40):
        harness = Harness(tmp_path, seed, Scenario(batches=6, max_boxes=3, green_pct=100))
        try:
            harness.run()
        except C.CoreError as exc:
            assert 'not trained on the HEAD' in str(exc)
            refused += 1
            # The controller is dead and the bad record is not in the journal.
            with pytest.raises(K.ControllerError):
                harness.ctl.tick()
            kinds = [rec['t'] for rec in J.scan(bytes(
                harness.world.files.files[harness.world.journal_path].data,
            ))[0]]
            assert kinds.count('apply_intent') == kinds.count('applied')
    assert refused >= 10
