"""Pure state machine: the HEAD rules, fencing, retries and replay."""
from __future__ import annotations

import json
import random

import pytest

from ops.parallel_eval.pit.batchctl import core as C

LADDER = (C.Rung('p', 'RTX5090', 'secure'), C.Rung('p', 'RTX5090', 'spot'), C.Rung('p', 'H100', 'secure'))
CFG = C.CoreConfig(max_boxes=2, max_infra_failures=2, ladder=LADDER)
EVIDENCE = (('obj', 10, 'sha', 'etag'),)


class Sim:
    """Drives the core the way the runtime does and keeps the record log."""

    def __init__(self, batches=2, cfg=CFG, genesis=True):
        self.cfg = cfg
        self.s = C.State()
        self.log = []
        if genesis:
            self.ev(C.GenesisVerified('S0', EVIDENCE))
            self.ev(C.ApplyDone(0, self.s.pending.link_id))
        for i in range(batches):
            self.ev(C.Arrived('b%d' % i, 'miner%d' % i, 'ref%d' % i))

    def ev(self, event):
        recs = C.step(self.cfg, self.s, event)
        self.log.extend(recs)
        return recs

    def kinds(self, event):
        return [kind for kind, _ in self.ev(event)]

    def start(self):
        recs = self.ev(C.Tick())
        assert [k for k, _ in recs] == ['lease_issued']
        fence = recs[0][1]['fence']
        assert self.kinds(C.RentOk(fence, 0, 'box%d' % fence, 'p')) == ['box_bound']
        assert self.kinds(C.Launched(fence)) == ['job_launched']
        return fence

    def green(self, fence, state_id='auto'):
        if state_id == 'auto':
            state_id = 'S%d' % fence
        self.ev(C.Polled(fence, complete=True))
        return self.kinds(C.Settled(fence, C.VERDICT_GREEN, 'R%d' % fence, state_id, EVIDENCE if state_id else ()))

    def red(self, fence):
        self.ev(C.Polled(fence, complete=True))
        return self.kinds(C.Settled(fence, C.VERDICT_RED, 'R%d' % fence))

    def applied(self):
        pending = self.s.pending
        return self.kinds(C.ApplyDone(pending.seq, pending.link_id))

    def phase(self, batch_id):
        return self.s.batch(batch_id).phase

    def replayed(self):
        other = C.State()
        for kind, body in self.log:
            C.evolve(other, kind, json.loads(json.dumps(body)))
        return other


def test_genesis_needs_a_confirmed_pointer_before_any_lease():
    sim = Sim(batches=1, genesis=False)
    assert sim.kinds(C.Tick()) == []
    assert sim.kinds(C.GenesisVerified('S0', EVIDENCE)) == ['apply_intent']
    assert sim.s.head is None and sim.s.pending.seq == 0
    assert sim.kinds(C.Tick()) == []
    assert sim.kinds(C.GenesisVerified('S0', EVIDENCE)) == []
    assert sim.applied() == ['applied']
    assert sim.s.head == C.Head(0, 'S0', sim.s.head.link_id) and sim.s.pending is None
    assert sim.kinds(C.Tick()) == ['lease_issued']


def test_single_green_is_applied_and_becomes_head():
    sim = Sim(batches=1)
    fence = sim.start()
    assert sim.green(fence) == ['decided', 'apply_intent']
    assert sim.phase('b0') is C.Phase.APPLYING
    assert sim.s.pending.parent_link == sim.s.head.link_id and sim.s.pending.seq == 1
    assert sim.applied() == ['applied']
    assert sim.s.head.seq == 1 and sim.s.head.state_id == 'S%d' % fence
    assert sim.phase('b0') is C.Phase.APPLIED
    out = C.outcome_fields(sim.s.batch('b0'))
    assert out['outcome'] == 'applied' and out['applied'] and out['credit'] == 'green'
    assert out['payout_list'] and out['head_seq'] == 1 and out['base_seq'] == 0


def test_first_green_wins_and_the_second_on_the_same_head_is_credited():
    sim = Sim(batches=2)
    f0, f1 = sim.start(), sim.start()
    assert sim.green(f1) == ['decided', 'apply_intent']
    # Same base, apply still pending: credited at once.
    assert sim.green(f0) == ['decided', 'credited']
    assert sim.phase('b0') is C.Phase.CREDITED and sim.s.batch('b0').note == 'not_first'
    sim.applied()
    assert sim.phase('b1') is C.Phase.APPLIED
    out = C.outcome_fields(sim.s.batch('b0'))
    assert out['credit'] == 'green' and out['payout_list'] and not out['applied']


def test_green_on_a_stale_head_is_credited_not_applied():
    sim = Sim(batches=2)
    f0, f1 = sim.start(), sim.start()
    sim.green(f1)
    sim.applied()
    assert sim.s.head.seq == 1
    assert sim.green(f0) == ['decided', 'credited']
    assert sim.s.batch('b0').note == 'stale_base'
    assert sim.s.head.seq == 1 and sim.s.pending is None


def test_running_attempt_is_not_cancelled_when_head_moves():
    sim = Sim(batches=2)
    f0, f1 = sim.start(), sim.start()
    sim.green(f1)
    sim.applied()
    batch = sim.s.batch('b0')
    assert batch.phase is C.Phase.RUNNING and batch.lease.fence == f0 and batch.lease.base_seq == 0
    assert batch.infra_failures == 0


def test_new_attempt_binds_the_pending_head_and_waits_for_it():
    sim = Sim(batches=3, cfg=C.CoreConfig(3, 2, LADDER))
    f0 = sim.start()
    sim.green(f0)
    assert sim.s.pending is not None
    f1 = sim.start()
    lease = sim.s.batch('b1').lease
    assert lease.base_seq == 1 and lease.base_state == sim.s.pending.state_id
    # GREEN on the pending HEAD waits; it is not credited and not applied yet.
    assert sim.green(f1) == ['decided']
    assert sim.phase('b1') is C.Phase.GREEN
    # A later GREEN on the same pending HEAD also waits.
    f2 = sim.start()
    assert sim.green(f2) == ['decided']
    assert sim.applied() == ['applied', 'apply_intent', 'credited']
    assert sim.phase('b1') is C.Phase.APPLYING and sim.phase('b2') is C.Phase.CREDITED
    assert sim.applied() == ['applied']
    assert sim.s.head.seq == 2 and sim.s.head.state_id == 'S%d' % f1


def test_red_is_final_and_unpaid():
    sim = Sim(batches=1)
    fence = sim.start()
    assert sim.red(fence) == ['decided']
    assert sim.phase('b0') is C.Phase.RED
    out = C.outcome_fields(sim.s.batch('b0'))
    assert out['credit'] is None and not out['payout_list'] and not out['applied']
    assert sim.kinds(C.Tick()) == []


def test_green_result_without_a_verified_state_is_credited_and_leaves_the_slot():
    sim = Sim(batches=2)
    f0, f1 = sim.start(), sim.start()
    assert sim.green(f0, state_id=None) == ['decided', 'credited']
    assert sim.phase('b0') is C.Phase.CREDITED and sim.s.batch('b0').note == 'no_verified_state'
    assert sim.s.pending is None
    # The next GREEN with a verified state on the same HEAD takes the slot.
    assert sim.green(f1) == ['decided', 'apply_intent']


def test_failed_attempt_keeps_queue_position_and_goes_first():
    sim = Sim(batches=3, cfg=C.CoreConfig(1, 3, LADDER))
    f0 = sim.start()
    assert sim.kinds(C.Polled(f0, gone=True)) == ['attempt_failed']
    batch = sim.s.batch('b0')
    assert batch.phase is C.Phase.QUEUED and batch.seq == 0 and batch.infra_failures == 1
    recs = sim.ev(C.Tick())
    assert recs[0][1]['batch_id'] == 'b0' and recs[0][1]['fence'] == f0 + 1 and recs[0][1]['attempt'] == 2


def test_lost_infra_after_n_failures_is_credited_never_red():
    sim = Sim(batches=1)
    for _ in range(CFG.max_infra_failures):
        fence = sim.start()
        sim.ev(C.Polled(fence, expired=True))
    batch = sim.s.batch('b0')
    assert batch.phase is C.Phase.LOST_INFRA and batch.infra_failures == 2 and batch.decision is None
    out = C.outcome_fields(batch)
    assert out['credit'] == 'lost_infra' and out['payout_list'] and out['outcome'] == 'lost_infra'
    assert sim.kinds(C.Tick()) == []


def test_invalid_upload_counts_as_infra_failure():
    sim = Sim(batches=1)
    fence = sim.start()
    sim.ev(C.Polled(fence, complete=True))
    recs = sim.ev(C.SettleInvalid(fence, 'object_mismatch'))
    assert recs[0][0] == 'attempt_failed' and recs[0][1]['reason'] == 'invalid_upload:object_mismatch'
    assert sim.phase('b0') is C.Phase.QUEUED


def test_complete_upload_wins_over_dead_box_and_deadline():
    sim = Sim(batches=1)
    fence = sim.start()
    assert sim.kinds(C.Polled(fence, complete=True, gone=True, expired=True)) == ['upload_observed']
    assert sim.phase('b0') is C.Phase.SETTLING


def test_rent_failures_walk_the_ladder_and_never_count_as_infra():
    sim = Sim(batches=1)
    fence = sim.ev(C.Tick())[0][1]['fence']
    rungs = []
    for try_no in range(4):
        op = [o for o in C.plan(CFG, sim.s) if isinstance(o, C.RentOp)][0]
        assert op.try_no == try_no
        rungs.append(op.rung)
        rec = sim.ev(C.RentFailed(fence, try_no, 'unavailable'))[0]
        assert rec[0] == 'rent_try' and rec[1]['backoff'] is (try_no == 2)
    assert rungs == [LADDER[0], LADDER[1], LADDER[2], LADDER[0]]
    batch = sim.s.batch('b0')
    assert batch.phase is C.Phase.RENTING and batch.infra_failures == 0 and batch.lease.try_no == 4
    # A rent result of an old try does not bind; its box is destroyed.
    assert sim.kinds(C.RentOk(fence, 1, 'late-box', 'p')) == ['box_doomed']
    assert sim.kinds(C.RentFailed(fence, 2, 'timeout')) == []
    assert sim.kinds(C.RentOk(fence, 4, 'good-box', 'p')) == ['box_bound']


def test_zombie_events_of_a_dead_fence_are_dropped():
    sim = Sim(batches=1, cfg=C.CoreConfig(1, 3, LADDER))
    old = sim.start()
    sim.ev(C.Polled(old, expired=True))
    new = sim.start()
    assert new != old
    for event in (
        C.Launched(old), C.Polled(old, complete=True), C.Polled(old, gone=True),
        C.Settled(old, C.VERDICT_GREEN, 'R', 'S-zombie', EVIDENCE), C.SettleInvalid(old, 'x'),
        C.RentFailed(old, 0, 'x'), C.Renewal(old, 3, 9.0),
    ):
        assert sim.kinds(event) == [], event
    assert sim.s.batch('b0').lease.fence == new and sim.phase('b0') is C.Phase.RUNNING
    # The settle of the new fence is not accepted before its upload was seen.
    assert sim.kinds(C.Settled(new, C.VERDICT_GREEN, 'R', 'S', EVIDENCE)) == []


def test_duplicate_results_are_no_ops():
    sim = Sim(batches=1)
    fence = sim.start()
    assert sim.kinds(C.Launched(fence)) == []
    assert sim.kinds(C.RentOk(fence, 0, 'box%d' % fence, 'p')) == []
    sim.ev(C.Polled(fence, complete=True))
    assert sim.kinds(C.Polled(fence, complete=True)) == []
    settled = C.Settled(fence, C.VERDICT_GREEN, 'R', 'S1', EVIDENCE)
    assert sim.kinds(settled) == ['decided', 'apply_intent']
    assert sim.kinds(settled) == []
    done = C.ApplyDone(1, sim.s.pending.link_id)
    assert sim.kinds(done) == ['applied']
    assert sim.kinds(done) == []
    assert sim.kinds(C.Arrived('b0', 'someone-else')) == []


def test_box_is_released_when_the_upload_is_seen_or_the_attempt_fails():
    sim = Sim(batches=2)
    f0, f1 = sim.start(), sim.start()
    assert sim.s.live_box_ids() == {'box%d' % f0, 'box%d' % f1}
    sim.ev(C.Polled(f0, complete=True))
    sim.ev(C.Polled(f1, gone=True))
    assert sim.s.live_box_ids() == set()
    assert set(sim.s.doomed) == {'box%d' % f0, 'box%d' % f1}
    assert sim.kinds(C.BoxGone('box%d' % f0)) == ['box_gone']
    assert sim.kinds(C.BoxGone('box%d' % f0)) == []
    assert (('b1', f1) in sim.s.dirty) and sim.kinds(C.StagingCleared('b1', f1)) == ['staging_cleared']


def test_orphan_rules():
    sim = Sim(batches=1)
    fence = sim.ev(C.Tick())[0][1]['fence']
    # The box of the open rent try is not an orphan.
    assert sim.kinds(C.OrphanSeen('x', 'p', fence, 0)) == []
    sim.ev(C.RentOk(fence, 0, 'x', 'p'))
    assert sim.kinds(C.OrphanSeen('x', 'p', fence, 0)) == []
    assert sim.kinds(C.OrphanSeen('twin', 'p', fence, 0)) == ['box_doomed']
    assert sim.kinds(C.OrphanSeen('twin', 'p', fence, 0)) == []
    assert sim.kinds(C.OrphanSeen('noname', 'p')) == ['box_doomed']
    with pytest.raises(C.CoreError):
        C.evolve(sim.s, 'box_doomed', {'box_id': 'x', 'provider': 'p', 'reason': 'bug'})


def test_slots_and_queue_order():
    sim = Sim(batches=4)
    f0, f1 = sim.start(), sim.start()
    assert sim.kinds(C.Tick()) == []
    assert sim.kinds(C.Tick(hold=False)) == []
    sim.red(f0)
    assert sim.kinds(C.Tick(hold=True)) == []
    assert sim.ev(C.Tick())[0][1]['batch_id'] == 'b2'
    assert f1 == 2


def test_divergence_halts_and_resume_is_explicit():
    sim = Sim(batches=2)
    f0, f1 = sim.start(), sim.start()
    sim.green(f0)
    pending = sim.s.pending
    assert sim.kinds(C.ApplyDiverged(pending.seq, 'other-link', 'x')) == []
    assert sim.kinds(C.ApplyDiverged(pending.seq, pending.link_id, 'pointer moved')) == ['halted']
    assert sim.s.halted == 'pointer moved'
    assert not [o for o in C.plan(CFG, sim.s) if isinstance(o, C.ApplyOp)]
    # A lease that is already in flight still reaches its decision.
    assert sim.green(f1) == ['decided']
    assert sim.phase('b1') is C.Phase.GREEN
    sim.ev(C.Arrived('b9', 'm'))
    assert sim.kinds(C.Tick()) == []
    assert sim.kinds(C.Resume('operator checked the pointer')) == ['resumed', 'credited']
    assert [o for o in C.plan(CFG, sim.s) if isinstance(o, C.ApplyOp)]
    assert sim.applied() == ['applied']


def test_plan_lists_the_effects_of_each_phase():
    sim = Sim(batches=1)
    fence = sim.ev(C.Tick())[0][1]['fence']
    assert C.plan(CFG, sim.s) == [C.RentOp(fence, 0, LADDER[0], 'b0')]
    sim.ev(C.RentOk(fence, 0, 'x', 'p'))
    assert C.plan(CFG, sim.s) == [
        C.LaunchOp(fence, 'b0', 1, 'x', 'p', 'S0', 0, 'ref0'), C.PollOp(fence, 'b0', 'x', 'p'),
    ]
    sim.ev(C.Launched(fence))
    assert C.plan(CFG, sim.s) == [C.PollOp(fence, 'b0', 'x', 'p')]
    sim.ev(C.Polled(fence, complete=True))
    assert C.plan(CFG, sim.s) == [C.SettleOp(fence, 'b0', 'S0'), C.DestroyOp('x', 'p')]
    sim.ev(C.Settled(fence, C.VERDICT_GREEN, 'R', 'S1', EVIDENCE))
    ops = C.plan(CFG, sim.s)
    assert isinstance(ops[0], C.ApplyOp) and ops[0].seq == 1 and ops[0].evidence == EVIDENCE
    assert C.GcOp('b0', fence) in ops
    sim.applied()
    assert C.PublishOp('b0') in C.plan(CFG, sim.s)
    assert sim.kinds(C.Published('b0')) == ['outcome_published']
    assert C.PublishOp('b0') not in C.plan(CFG, sim.s)


@pytest.mark.parametrize('kind,body', [
    ('lease_issued', {'batch_id': 'b0', 'fence': 99, 'attempt': 1, 'rung': 0}),
    ('lease_issued', {'batch_id': 'b0', 'fence': 1, 'attempt': 2, 'rung': 0}),
    ('batch_admitted', {'batch_id': 'b0', 'seq': 5, 'miner': 'm', 'ref': ''}),
    ('applied', {'seq': 1, 'link_id': 'x', 'state_id': 's', 'batch_id': 'b0'}),
    ('credited', {'batch_id': 'b0', 'reason': 'x'}),
    ('box_gone', {'box_id': 'nope', 'how': 'absent'}),
    ('staging_cleared', {'batch_id': 'b0', 'fence': 1}),
    ('outcome_published', {'batch_id': 'b0'}),
    ('decided', {'batch_id': 'b0', 'fence': 1, 'verdict': 'green', 'state_id': 's', 'result_sha': 'r',
                 'base_state': 'S0', 'base_seq': 0, 'order': 0, 'evidence': []}),
    ('mystery', {}),
])
def test_evolve_refuses_records_that_do_not_fit(kind, body):
    sim = Sim(batches=1)
    with pytest.raises(C.CoreError):
        C.evolve(sim.s, kind, body)


def test_evolve_refuses_a_second_intent_and_a_stale_or_stateless_apply():
    sim = Sim(batches=2)
    f0, f1 = sim.start(), sim.start()
    sim.green(f0)
    intent = dict(sim.log[-1][1])
    with pytest.raises(C.CoreError):
        C.evolve(sim.s, 'apply_intent', intent)
    sim.applied()
    sim.ev(C.Polled(f1, complete=True))
    C.evolve(sim.s, 'decided', {
        'batch_id': 'b1', 'fence': f1, 'verdict': 'green', 'state_id': 'S-stale', 'result_sha': 'r',
        'base_state': 'S0', 'base_seq': 0, 'order': 1, 'evidence': [],
    })
    with pytest.raises(C.CoreError):
        C.evolve(sim.s, 'apply_intent', {
            'batch_id': 'b1', 'seq': 2, 'state_id': 'S-stale', 'parent_link': sim.s.head.link_id,
            'link_id': 'l', 'result_sha': 'r', 'fence': f1, 'evidence': [],
        })


def test_passive_records_do_not_change_the_state():
    sim = Sim(batches=1)
    before = sim.s.key()
    for kind in C.PASSIVE_KINDS:
        C.evolve(sim.s, kind, {'anything': 1})
    assert sim.s.key() == before


def test_outcome_of_a_live_batch_is_refused():
    sim = Sim(batches=1)
    with pytest.raises(C.CoreError):
        C.outcome_fields(sim.s.batch('b0'))


def _random_event(rng, sim):
    s = sim.s
    fence = rng.randint(1, max(1, s.next_fence))
    try_no = rng.randint(0, 2)
    box = 'box%d' % rng.randint(1, 6)
    pending = s.pending
    choices = [
        lambda: C.Arrived('b%d' % rng.randint(0, 5), 'm'),
        lambda: C.Tick(hold=rng.random() < 0.1),
        lambda: C.RentOk(fence, try_no, box, 'p'),
        lambda: C.RentFailed(fence, try_no, 'x'),
        lambda: C.Launched(fence),
        lambda: C.Polled(fence, complete=rng.random() < 0.5, gone=rng.random() < 0.3, expired=rng.random() < 0.3),
        lambda: C.Renewal(fence, 1, 2.0),
        lambda: C.Settled(fence, rng.choice([C.VERDICT_GREEN, C.VERDICT_RED, 'weird']), 'R%d' % fence,
                          rng.choice([None, 'S%d' % fence]), EVIDENCE),
        lambda: C.SettleInvalid(fence, 'bad'),
        lambda: C.ApplyDone(pending.seq if pending else rng.randint(0, 3), pending.link_id if pending else 'x'),
        lambda: C.ApplyDiverged(pending.seq if pending else 1, pending.link_id if pending else 'x', 'moved'),
        lambda: C.BoxGone(box),
        lambda: C.OrphanSeen(box, 'p', fence, try_no),
        lambda: C.StagingCleared('b%d' % rng.randint(0, 5), fence),
        lambda: C.Published('b%d' % rng.randint(0, 5)),
        lambda: C.GenesisVerified('S0', EVIDENCE),
        lambda: C.Resume('go'),
        lambda: C.Rejected('d%d' % rng.randint(0, 3), 'bad id'),
    ]
    return rng.choice(choices)()


@pytest.mark.parametrize('seed', range(40))
def test_random_events_never_break_the_core_and_replay_is_exact(seed):
    rng = random.Random(seed)
    sim = Sim(batches=0, cfg=C.CoreConfig(rng.randint(1, 3), rng.randint(1, 3), LADDER), genesis=False)
    terminal = {}
    for _ in range(400):
        sim.ev(_random_event(rng, sim))
        s = sim.s
        assert s.slots_in_use() <= sim.cfg.max_boxes
        assert not (s.live_box_ids() & set(s.doomed))
        for batch_id, batch in s.done.items():
            assert batch.phase in C.TERMINAL
            assert terminal.setdefault(batch_id, batch.key()) == batch.key()
        for batch in s.active.values():
            assert batch.phase not in C.TERMINAL
            assert batch.infra_failures < sim.cfg.max_infra_failures
            if batch.decision is not None:
                assert batch.decision.verdict == C.VERDICT_GREEN
    assert sim.replayed().key() == sim.s.key()
    for cut in range(0, len(sim.log), 7):
        partial = C.State()
        for kind, body in sim.log[:cut]:
            C.evolve(partial, kind, json.loads(json.dumps(body)))
        # A journal prefix is always a consistent state: resolve and plan work.
        while True:
            rec = C.resolve(sim.cfg, partial)
            if rec is None:
                break
            C.evolve(partial, rec[0], rec[1])
        C.plan(sim.cfg, partial)
