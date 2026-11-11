"""Exhaustive model checking of the real core, plus seeded-bug detection.

The quick configurations run in every test run. The deep ones need
``BATCHCTL_DEEP=1``; their results are recorded in the pull request.
"""
from __future__ import annotations

import json
import os

import pytest

from ops.parallel_eval.pit.batchctl import core as C
from ops.parallel_eval.pit.batchctl import modelcheck as M

DEEP = os.environ.get('BATCHCTL_DEEP') == '1'

QUICK = {
    'full-1-batch-2-faults': dict(batches=1, max_boxes=1, faults=2),
    'full-1-batch-1-fault-results-of-cleanup-in-flight': dict(
        batches=1, max_boxes=1, faults=1, atomic_housekeeping=False),
    'coarse-2-batches-2-faults': dict(batches=2, max_boxes=2, faults=2, coarse=True),
    'coarse-2-batches-1-box-2-faults': dict(batches=2, max_boxes=1, faults=2, coarse=True),
    'coarse-3-batches-1-fault': dict(batches=3, max_boxes=2, faults=1, coarse=True),
    'coarse-2-batches-foreign-pointer-write': dict(
        batches=2, max_boxes=2, faults=2, foreign_writes=1, coarse=True),
    'full-1-batch-fault-report': dict(batches=1, max_boxes=1, faults=2, fault_reports=1),
    'coarse-2-batches-fault-report': dict(
        batches=2, max_boxes=2, faults=2, fault_reports=1, coarse=True),
}

DEEP_CONFIGS = {
    'full-1-batch-3-faults': dict(batches=1, max_boxes=1, faults=3),
    'full-1-batch-3-infra-failures': dict(batches=1, max_boxes=1, faults=3, max_infra_failures=3),
    'full-2-batches-1-fault': dict(batches=2, max_boxes=2, faults=1),
    'full-1-batch-2-faults-results-of-cleanup-in-flight': dict(
        batches=1, max_boxes=1, faults=2, atomic_housekeeping=False),
    'coarse-2-batches-3-faults-foreign-pointer-write': dict(
        batches=2, max_boxes=2, faults=3, foreign_writes=1, coarse=True),
    'coarse-3-batches-2-faults': dict(batches=3, max_boxes=2, faults=2, coarse=True),
    'coarse-3-batches-3-boxes-2-faults': dict(batches=3, max_boxes=3, faults=2, coarse=True),
    'coarse-4-batches-1-fault': dict(batches=4, max_boxes=2, faults=1, coarse=True),
    'coarse-3-batches-2-faults-fault-report': dict(
        batches=3, max_boxes=2, faults=2, fault_reports=1, coarse=True),
}


def _assert_proof(report):
    assert report.ok, '%s\n%s' % (report.violation, ' -> '.join(report.trace or []))
    assert report.violation is None and report.states > 0 and report.sinks > 0


@pytest.mark.parametrize('name', sorted(QUICK))
def test_every_reachable_state_keeps_the_invariants(name):
    report = M.run(M.ModelConfig(**QUICK[name]))
    _assert_proof(report)
    outcomes = set()
    for tally in report.outcomes:
        outcomes.update(tally.split(','))
    # The search is not empty-handed: each outcome is really reached.
    assert {'applied', 'red', 'credited'} <= outcomes
    if 'foreign' in name or 'fault-report' in name:
        assert 'halted' in outcomes
    if QUICK[name]['faults'] >= 2:
        assert 'lost_infra' in outcomes


@pytest.mark.skipif(not DEEP, reason='deep model check; set BATCHCTL_DEEP=1')
@pytest.mark.parametrize('name', sorted(DEEP_CONFIGS))
def test_deep_configurations(name):
    report = M.run(M.ModelConfig(**DEEP_CONFIGS[name]))
    _assert_proof(report)
    print(json.dumps({'config': name, 'states': report.states, 'transitions': report.transitions,
                      'sinks': report.sinks, 'seconds': round(report.seconds, 1)}))


def test_state_limit_is_reported_as_no_proof():
    report = M.run(M.ModelConfig(batches=2, faults=2, max_states=500))
    assert not report.ok and 'not a proof' in report.violation


def test_cli_prints_the_report(capsys):
    code = M.main(['--batches', '1', '--max-boxes', '1', '--faults', '1', '--coarse', 'true'])
    doc = json.loads(capsys.readouterr().out)
    assert code == 0 and doc['report']['ok'] and doc['config']['coarse'] is True
    assert len(doc['invariants']) == len(M.INVARIANTS)


# ----------------------------------------------------------------------------
# Seeded bugs. Each test breaks one rule of the core and expects the checker
# to report it. This shows that a pass above is not an empty pass.
# ----------------------------------------------------------------------------

def _expect_violation(config, *needles):
    report = M.run(M.ModelConfig(**config))
    assert not report.ok, 'the checker missed a seeded bug'
    assert any(n in report.violation for n in needles), report.violation
    assert report.trace
    return report


def test_checker_catches_credit_instead_of_apply(monkeypatch):
    real = C.resolve

    def bad(cfg, state):
        rec = real(cfg, state)
        if rec is not None and rec[0] == 'apply_intent':
            return 'credited', {'batch_id': rec[1]['batch_id'], 'reason': 'bug'}
        return rec

    monkeypatch.setattr(C, 'resolve', bad)
    _expect_violation(dict(batches=1, max_boxes=1, faults=0, coarse=True), 'CREDITED, but no earlier GREEN')


def test_checker_catches_a_green_that_is_never_settled(monkeypatch):
    real = C.resolve

    def bad(cfg, state):
        rec = real(cfg, state)
        return None if rec is not None and rec[0] == 'credited' else rec

    monkeypatch.setattr(C, 'resolve', bad)
    _expect_violation(dict(batches=2, max_boxes=2, faults=0, coarse=True), 'fair sink is not done')


def test_checker_catches_apply_of_a_stale_green(monkeypatch):
    real_resolve = C.resolve

    def bad_resolve(cfg, state):
        rec = real_resolve(cfg, state)
        if rec is not None and rec[0] == 'credited' and rec[1]['reason'] == 'stale_base':
            batch = state.active[rec[1]['batch_id']]
            d = batch.decision
            return 'apply_intent', {
                'batch_id': batch.batch_id, 'seq': state.head.seq + 1, 'state_id': d.state_id,
                'parent_link': state.head.link_id, 'link_id': 'link-%s' % batch.batch_id,
                'result_sha': d.result_sha, 'fence': d.fence, 'evidence': [],
            }
        return rec

    def bad_intent(state, b):
        batch = state.active[b['batch_id']] if b['batch_id'] is not None else None
        if batch is not None:
            batch.phase = C.Phase.APPLYING
        state.pending = C.Pending(
            b['batch_id'], b['seq'], b['state_id'], b['link_id'], b['parent_link'],
            b['result_sha'], b['fence'], (),
        )

    monkeypatch.setattr(C, 'resolve', bad_resolve)
    monkeypatch.setitem(C._EVOLVE, 'apply_intent', bad_intent)
    _expect_violation(
        dict(batches=2, max_boxes=2, faults=0, coarse=True),
        'not trained on the HEAD it replaced', 'not the first GREEN',
    )


def test_checker_catches_a_second_apply_on_the_same_head(monkeypatch):
    real_resolve = C.resolve

    def bad_resolve(cfg, state):
        rec = real_resolve(cfg, state)
        if rec is not None and rec[0] == 'credited' and rec[1]['reason'] == 'not_first':
            batch = state.active[rec[1]['batch_id']]
            d = batch.decision
            return 'apply_intent', {
                'batch_id': batch.batch_id, 'seq': state.head.seq + 1, 'state_id': d.state_id,
                'parent_link': state.head.link_id, 'link_id': 'link-%s' % batch.batch_id,
                'result_sha': d.result_sha, 'fence': d.fence, 'evidence': [],
            }
        return rec

    def bad_intent(state, b):
        if b['batch_id'] is not None:
            state.active[b['batch_id']].phase = C.Phase.APPLYING
        state.pending = C.Pending(
            b['batch_id'], b['seq'], b['state_id'], b['link_id'], b['parent_link'],
            b['result_sha'], b['fence'], (),
        )

    monkeypatch.setattr(C, 'resolve', bad_resolve)
    monkeypatch.setitem(C._EVOLVE, 'apply_intent', bad_intent)
    _expect_violation(
        dict(batches=2, max_boxes=2, faults=0, coarse=True),
        'APPLYING without the pending intent', 'not the first GREEN',
    )


# The base rule. The first three tests seed the rules that the base rule
# replaced; the last two break the payout of an outcome.

def test_checker_catches_a_retry_that_trains_on_the_new_head(monkeypatch):
    # After an infra loss the fresh box takes the HEAD of its own moment.
    monkeypatch.setattr(C, '_base_of', lambda state, batch: state.base())
    report = _expect_violation(
        dict(batches=2, max_boxes=2, faults=1, coarse=True), 'a later box of a batch got another base',
    )
    assert any(label.startswith('box-dies') or 'expir' in label for label in report.trace)
    assert 'apply-cas' in report.trace


def test_checker_catches_a_stale_red_that_goes_back_to_the_queue(monkeypatch):
    # A batch that is RED against an old HEAD trains again on the new HEAD.
    real = C.decide

    def bad(cfg, state, ev):
        rec = real(cfg, state, ev)
        if (
            rec is not None and rec[0] == 'decided' and rec[1]['verdict'] == C.VERDICT_RED
            and rec[1]['base_seq'] != state.base()[1]
        ):
            return 'requeued', {'batch_id': rec[1]['batch_id']}
        return rec

    def requeue(state, b):
        # The seeded rule covers its tracks: the core's own guards all pass.
        batch = state.active[b['batch_id']]
        batch.phase = C.Phase.QUEUED
        batch.attempts = batch.infra_failures
        batch.base_state = batch.base_seq = None
        C._end_lease(state, batch)

    monkeypatch.setattr(C, 'decide', bad)
    monkeypatch.setitem(C._EVOLVE, 'requeued', requeue)
    _expect_violation(
        dict(batches=2, max_boxes=2, faults=0, coarse=True),
        'a batch went back to the queue without an infra loss',
    )


def test_checker_catches_a_box_that_is_stopped_when_the_head_moves(monkeypatch):
    # The controller gives up a healthy box as soon as its base is old.
    real = C.decide

    def bad(cfg, state, ev):
        if type(ev) is C.Tick:
            for fence in sorted(state.by_fence):
                batch = state.leased(fence)
                if batch.phase is C.Phase.RUNNING and batch.base_seq < state.base()[1]:
                    return C._fail(cfg, batch, 'stale_base')
        return real(cfg, state, ev)

    monkeypatch.setattr(C, 'decide', bad)
    _expect_violation(
        dict(batches=2, max_boxes=2, faults=0, coarse=True),
        'an attempt ended without a decision, but the world had no fault',
    )


def _requeue_green(state, b):
    batch = state.active[b['batch_id']]
    batch.phase = C.Phase.QUEUED
    batch.decision = None
    batch.attempts = batch.infra_failures
    batch.base_state = batch.base_seq = None


def test_checker_catches_a_stale_green_that_trains_again(monkeypatch):
    # A GREEN on an old HEAD is not credited: it trains again to be applied.
    real = C.resolve

    def bad(cfg, state):
        rec = real(cfg, state)
        if rec is not None and rec[0] == 'credited' and rec[1]['reason'] == 'stale_base':
            return 'requeued', {'batch_id': rec[1]['batch_id']}
        return rec

    monkeypatch.setattr(C, 'resolve', bad)
    monkeypatch.setitem(C._EVOLVE, 'requeued', _requeue_green)
    _expect_violation(
        dict(batches=2, max_boxes=2, faults=0, coarse=True),
        'a batch went back to the queue without an infra loss',
    )


def test_checker_catches_a_waiting_green_that_trains_again(monkeypatch):
    # Two GREENs wait for the pending HEAD that they trained on. One wins the
    # apply. The other is not credited: it trains again, in a later step than
    # its decision.
    real_decide, real_resolve = C.decide, C.resolve
    event = []

    def spy(cfg, state, ev):
        event[:] = [type(ev)]
        return real_decide(cfg, state, ev)

    def bad(cfg, state):
        rec = real_resolve(cfg, state)
        if (
            rec is not None and rec[0] == 'credited' and rec[1]['reason'] == 'not_first'
            and event == [C.ApplyDone]
        ):
            return 'requeued', {'batch_id': rec[1]['batch_id']}
        return rec

    monkeypatch.setattr(C, 'decide', spy)
    monkeypatch.setattr(C, 'resolve', bad)
    monkeypatch.setitem(C._EVOLVE, 'requeued', _requeue_green)
    _expect_violation(
        dict(batches=3, max_boxes=2, faults=0, coarse=True),
        'a decided batch ran again or lost its decision',
    )


def test_checker_catches_a_second_box_without_an_infra_loss(monkeypatch):
    # A lost box is not counted, so the batch can run for ever at no cost.
    real = C._EVOLVE['attempt_failed']

    def free_retry(state, b):
        batch = state.active[b['batch_id']]
        real(state, dict(b, infra_failures=batch.infra_failures + 1, terminal=False))
        batch.infra_failures -= 1
        batch.attempts -= 1

    monkeypatch.setattr(C, '_fail', lambda cfg, batch, reason: ('attempt_failed', {
        'batch_id': batch.batch_id, 'fence': batch.lease.fence, 'reason': reason,
        'infra_failures': batch.infra_failures, 'terminal': False,
    }))
    monkeypatch.setitem(C._EVOLVE, 'attempt_failed', free_retry)
    _expect_violation(
        dict(batches=1, max_boxes=1, faults=1, coarse=True),
        'a batch went back to the queue without an infra loss',
    )


@pytest.mark.parametrize('credit, needle', [
    ({C.Phase.APPLIED: C.CREDIT_GREEN, C.Phase.LOST_INFRA: C.CREDIT_LOST_INFRA}, 'the outcome of credited'),
    (dict(C.CREDIT, **{C.Phase.RED: C.CREDIT_GREEN}), 'the outcome of red'),
    ({C.Phase.APPLIED: C.CREDIT_GREEN, C.Phase.CREDITED: C.CREDIT_GREEN}, 'the outcome of lost_infra'),
])
def test_checker_catches_a_wrong_payout(monkeypatch, credit, needle):
    monkeypatch.setattr(C, 'CREDIT', credit)
    _expect_violation(dict(batches=2, max_boxes=2, faults=2, coarse=True), needle)


def test_checker_catches_an_outcome_that_names_the_new_head_as_base(monkeypatch):
    real = C.outcome_fields

    def bad(batch):
        fields = real(batch)
        if batch.note == 'stale_base':
            fields['base_seq'] += 1
        return fields

    monkeypatch.setattr(C, 'outcome_fields', bad)
    _expect_violation(dict(batches=2, max_boxes=2, faults=0, coarse=True), 'pays or names the wrong thing')


def test_checker_catches_wrong_queue_order(monkeypatch):
    def newest(state):
        queued = [b for b in state.active.values() if b.phase is C.Phase.QUEUED]
        return max(queued, key=lambda b: b.seq) if queued else None

    monkeypatch.setattr(C, '_first_queued', newest)
    _expect_violation(dict(batches=2, max_boxes=1, faults=0, coarse=True), 'queue order violated')


def test_checker_catches_too_many_boxes(monkeypatch):
    monkeypatch.setattr(C.State, 'slots_in_use', lambda self: 0)
    _expect_violation(dict(batches=2, max_boxes=1, faults=0, coarse=True), 'more boxes in use than max_boxes')


def test_checker_catches_a_miner_penalised_for_a_bad_upload(monkeypatch):
    real = C.decide

    def bad(cfg, state, ev):
        if type(ev) is C.SettleInvalid:
            ev = C.Settled(ev.fence, C.VERDICT_RED, 'none')
        return real(cfg, state, ev)

    monkeypatch.setattr(C, 'decide', bad)
    _expect_violation(dict(batches=1, max_boxes=1, faults=1, coarse=True), 'decision does not match the package')


def test_checker_catches_a_missing_terminal_rule(monkeypatch):
    real = C._fail

    def bad(cfg, batch, reason):
        kind, body = real(cfg, batch, reason)
        return kind, dict(body, terminal=False)

    monkeypatch.setattr(C, '_fail', bad)
    _expect_violation(dict(batches=1, max_boxes=1, faults=2, coarse=True), 'went on after N infra failures')


def test_checker_catches_an_ignored_pointer_divergence(monkeypatch):
    real = C.decide

    def bad(cfg, state, ev):
        return None if type(ev) is C.ApplyDiverged else real(cfg, state, ev)

    monkeypatch.setattr(C, 'decide', bad)
    _expect_violation(
        dict(batches=1, max_boxes=1, faults=1, foreign_writes=1, coarse=True), 'cycle of fair steps',
    )


def test_checker_catches_a_reused_fence(monkeypatch):
    real = C.decide

    def bad(cfg, state, ev):
        rec = real(cfg, state, ev)
        if rec is not None and rec[0] == 'lease_issued' and rec[1]['attempt'] > 1:
            return rec[0], dict(rec[1], fence=rec[1]['fence'] - 1)
        return rec

    def loose_lease(state, b):
        batch = state.active[b['batch_id']]
        batch.phase = C.Phase.RENTING
        batch.attempts = b['attempt']
        batch.lease = C.Lease(b['fence'], 0, b['rung'])
        state.by_fence[b['fence']] = batch.batch_id

    monkeypatch.setattr(C, 'decide', bad)
    monkeypatch.setitem(C._EVOLVE, 'lease_issued', loose_lease)
    _expect_violation(
        dict(batches=1, max_boxes=1, faults=1), 'a fence was issued twice', 'decision does not match',
        'dropped as dead',
    )


def test_checker_catches_a_lease_that_is_never_expired(monkeypatch):
    real = C.decide

    def bad(cfg, state, ev):
        if type(ev) is C.Polled and not ev.complete:
            return None
        return real(cfg, state, ev)

    monkeypatch.setattr(C, 'decide', bad)
    _expect_violation(dict(batches=1, max_boxes=1, faults=1, coarse=True), 'cycle of fair steps', 'fair sink is not done')


def test_checker_catches_a_box_that_is_never_released(monkeypatch):
    monkeypatch.setitem(C._EVOLVE, 'box_doomed', lambda state, b: None)
    real_observed = C._EVOLVE['upload_observed']

    def keep_box(state, b):
        real_observed(state, b)
        state.doomed.clear()

    monkeypatch.setitem(C._EVOLVE, 'upload_observed', keep_box)
    _expect_violation(dict(batches=1, max_boxes=1, faults=0), 'cycle of fair steps', 'left at a sink')
