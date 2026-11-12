"""The hook from outcomes into the gate decisions that the K1 service pays on.

The decisions are written by the real writer of the live gate
(``gate_writer.HostGateWriter``) with the official ``baseline_v1`` check, and
the records come from the real judge on the real live scorer (``test_judge``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

from ops.parallel_eval.pit.batchctl import MODE_LIVE, MODE_SHADOW
from ops.parallel_eval.pit.batchctl import settle as S
from ops.parallel_eval.pit.batchctl.guard import GuardedStore
from ops.parallel_eval.pit.batchctl.interfaces import VERDICT_GREEN, VERDICT_RED
from ops.parallel_eval.pit.batchctl.layout import Layout, outcome_body, sha256_hex
from ops.parallel_eval.pit.batchctl.sim import RawStore
from ops.parallel_eval.pit.batchctl.test_judge import BASE_STATE, CAND, EPOCH, HEAD, RUN, World
from ops.parallel_eval.pit.gate_writer import GateWriteError, HostGateWriter
from ops.parallel_eval.pit.round_fixture import RID


class FakeRounds:
    """The round store of the gate, as the writer uses it."""

    def __init__(self) -> None:
        self.added: List[Tuple[str, Dict[str, Any]]] = []
        self.closed_rounds: Dict[str, Tuple[str, int]] = {}

    def turns(self, rid: str) -> List[Dict[str, Any]]:
        return [turn for name, turn in self.added if name == rid]

    def turn_add(self, rid: str, turn: Mapping[str, Any]) -> None:
        self.added.append((rid, dict(turn)))

    def closed(self, rid: str) -> bool:
        return rid in self.closed_rounds

    def close(self, rid: str, why: str, epoch: int) -> None:
        self.closed_rounds[rid] = (why, epoch)


class Gate:
    def __init__(self, tmp: Path, *, mode: str = MODE_LIVE) -> None:
        self.tmp = tmp
        self.world = World(tmp / 'round')
        self.world.fill()
        self.raw = RawStore()
        self.store = GuardedStore(self.raw, RUN, mode)
        self.layout = Layout(self.store.prefix)
        self.rounds = FakeRounds()
        self.grants: List[Tuple[int, str]] = []
        self.epochs = tmp / 'gate-state' / 'epochs'
        self.life = tmp / 'life' / 'epochs'
        self.writer = HostGateWriter(
            epochs_root=self.epochs, rounds=self.rounds, baseline_v1=self.world.live.BV,
            k1_grant_fn=lambda epoch, why: self.grants.append((epoch, why)) or {'granted': epoch},
        )
        self.alerts: List[Tuple[str, str]] = []
        self.settler = self.make()

    def make(self, **changes: Any) -> S.Settler:
        fields: Dict[str, Any] = dict(
            run_id=RUN, mode=MODE_LIVE, store=self.store, records=self.world.records, writer=self.writer,
            marks=self.tmp / 'private' / 'settled', life_epochs=self.life,
            alert=lambda code, detail: self.alerts.append((code, str(detail))), clock=lambda: 1000.0,
        )
        fields.update(changes)
        return S.Settler(**fields)

    def turn(self, epoch: int) -> None:
        folder = self.life / str(epoch)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / S.TURN_FILE).write_text(json.dumps({'status': 'turn', 'uid': 11, 'epoch_index': epoch}))

    def judged(self, epoch: int, shift: float, *, fence: int = 7, trained: bool = True) -> Tuple[str, str]:
        """Judge one batch with the real judge. -> (verdict, sha256 of the result)."""
        self.world.sample(epoch)
        batch_id = 'e%d' % epoch
        if trained:
            body = self.world.result(batch_id, self.world.live.candidate(shift, seed=3), fence=fence)
        else:
            body = self.world.result(batch_id, {}, fence=fence, outcome='not_trained',
                                     reason='bound_to_other_checkpoint')
        return self.world.verdict(batch_id, body), sha256_hex(body)

    def outcome(self, epoch: int, kind: str, result_sha: Optional[str], *, state_id: Optional[str] = 'e' * 64,
                publish: bool = True, batch_id: Optional[str] = None) -> Dict[str, Any]:
        decided = kind != 'lost_infra'
        fields = {
            'batch_id': batch_id or 'e%d' % epoch, 'seq': epoch, 'miner': '11:5Faaaaaaaaaaaaaa', 'outcome': kind,
            'payout_list': kind != 'red', 'credit': {'red': None, 'lost_infra': 'lost_infra'}.get(kind, 'green'),
            'applied': kind == 'applied', 'head_seq': 3 if kind == 'applied' else None,
            'base_seq': 2 if decided else None, 'base_state': BASE_STATE if decided else None,
            'state_id': state_id if decided and kind != 'red' else None, 'result_sha': result_sha if decided else None,
            'fence': 7 if decided else None, 'attempts': 1, 'infra_failures': 0 if decided else 3, 'note': kind,
        }
        if publish:
            self.store.put(self.layout.outcome(fields['batch_id']), outcome_body(fields))
        return fields

    def decision(self, epoch: int) -> Dict[str, Any]:
        return json.loads((self.epochs / str(epoch) / S.DECISION_FILE).read_text())

    def files(self) -> List[str]:
        return sorted(str(path.relative_to(self.tmp / 'gate-state')) for path in (self.tmp / 'gate-state').rglob('*')
                      if path.is_file())


@pytest.fixture
def gate(tmp_path: Path) -> Gate:
    return Gate(tmp_path / 'own')


def test_each_outcome_becomes_the_decision_that_k1_reads(gate: Gate) -> None:
    green, red = EPOCH, EPOCH + 1
    credited, untrained, lost = EPOCH + 2, EPOCH + 3, EPOCH + 4
    for epoch in (green, red, credited, untrained, lost):
        gate.turn(epoch)
    verdict, sha = gate.judged(green, 0.25)
    assert verdict == VERDICT_GREEN
    gate.outcome(green, 'applied', sha)
    verdict, sha = gate.judged(red, -0.25)
    assert verdict == VERDICT_RED
    gate.outcome(red, 'red', sha)
    gate.outcome(credited, 'credited', gate.judged(credited, 0.25)[1])
    verdict, sha = gate.judged(untrained, 0.0, trained=False)
    assert verdict == VERDICT_RED
    gate.outcome(untrained, 'red', sha)
    gate.outcome(lost, 'lost_infra', None)

    done = {one.batch_id: one for one in gate.settler.run_once()}
    assert {name: (one.status, one.decision) for name, one in done.items()} == {
        'e%d' % green: (S.WRITTEN, 'green'), 'e%d' % red: (S.WRITTEN, 'red'),
        'e%d' % credited: (S.WRITTEN, 'green'), 'e%d' % untrained: (S.WRITTEN, 'no_decision'),
        'e%d' % lost: (S.WRITTEN, 'lost_infra'),
    }
    assert gate.alerts == []

    doc = gate.decision(green)
    assert (doc['decision'], doc['candidate'], doc['head'], doc['epoch']) == ('green', CAND, HEAD, green)
    assert doc['evaluated']['z'] > 0 and doc['decided_at_cr'].endswith(' CR') and doc['source'] == 'pit'
    assert doc['merge_blocked'] is False and doc['no_decision'] is False and 'settlement' not in doc
    assert doc['heldout']['round_id'] == RID and doc['heldout']['head_rollouts'] == 12
    meta = doc['parallel_eval']
    assert (meta['source'], meta['run_id'], meta['outcome'], meta['applied']) == ('batchctl', RUN, 'applied', True)
    assert meta['baseline'] == 'head-blocks' and len(meta['fingerprint']) == 64
    # A GREEN that moved HEAD closes the round of the gate, as on the serial lane.
    assert gate.rounds.closed_rounds == {RID: (gate.rounds.closed_rounds[RID][0], green)}

    doc = gate.decision(red)
    assert (doc['decision'], doc['candidate'], doc['no_decision']) == ('red', CAND, False)
    assert doc['evaluated']['z'] < 0 and doc['decided_at_cr']

    doc = gate.decision(credited)
    assert (doc['decision'], doc['merge_blocked'], doc['candidate']) == ('green', True, CAND)
    assert 'merge_blocked' in doc['reasons'] and doc['parallel_eval']['applied'] is False
    assert 'settlement' not in doc

    doc = gate.decision(untrained)
    assert (doc['decision'], doc['no_decision'], doc['chart_status']) == ('no_decision', True, 'no_decision')
    assert doc['candidate'] == '' and doc['head'] == HEAD and doc['evaluated'] == {'z': None}
    assert doc['k1_grant'] == {'granted': untrained} and 'not_trained' in doc['reasons'][0]
    assert [epoch for epoch, _ in gate.grants] == [untrained]

    doc = gate.decision(lost)
    assert (doc['decision'], doc['settlement'], doc['paid']) == ('lost_infra', 'pending_operator', False)
    assert doc['candidate'] == '' and doc['evaluated'] == {'z': None} and doc['decided_at_cr']
    assert doc['chart_status'] == 'lost_infra' and doc['parallel_eval']['terminal'] == 'lost_infra'

    # One turn of the round for each decision with numbers; none for lost_infra and no_decision.
    assert sorted((turn['epoch'], turn['decision']) for _, turn in gate.rounds.added) == [
        (green, 'green'), (red, 'red'), (credited, 'green')]
    # Decisions only: no turn file, no score file, nothing else under the gate state.
    assert gate.files() == sorted(
        ['epochs/%d/DECISION.json' % e for e in (green, red, credited, untrained, lost)]
        + ['epochs/%d/EVALUATED.private.json' % e for e in (green, red, credited)])
    assert sorted(path.name for path in gate.life.rglob('*') if path.is_file()) == [S.TURN_FILE] * 5
    # The private seeds of the round are in no decision.
    text = ''.join((gate.epochs / str(e) / S.DECISION_FILE).read_text() for e in (green, red, credited, untrained, lost))
    assert gate.world.rnd.plan_seed not in text and gate.world.secret not in text


def test_a_second_look_changes_nothing(gate: Gate) -> None:
    gate.turn(EPOCH)
    gate.outcome(EPOCH, 'applied', gate.judged(EPOCH, 0.25)[1])
    assert [one.status for one in gate.settler.run_once()] == [S.WRITTEN]
    before = (gate.decision(EPOCH), len(gate.rounds.added))
    assert gate.settler.run_once() == []
    # The marks are a cache. Without them the same decision is taken again and nothing changes.
    for mark in (gate.tmp / 'private' / 'settled').iterdir():
        mark.unlink()
    again = gate.settler.run_once()
    assert [(one.status, one.decision) for one in again] == [(S.WRITTEN, 'green')]
    assert (gate.decision(EPOCH), len(gate.rounds.added)) == before
    # ``settle`` on the document gives the same answer as the mark.
    doc = json.loads(gate.raw.objs[gate.layout.outcome('e%d' % EPOCH)])
    assert gate.settler.settle(doc).status == S.WRITTEN and gate.decision(EPOCH) == before[0]


@pytest.mark.parametrize('case', ['other_candidate', 'other_decision', 'not_json', 'red_over_green', 'lost_over_green',
                                  'untrained_over_green'])
def test_a_decision_that_is_there_is_never_replaced(gate: Gate, case: str) -> None:
    gate.turn(EPOCH)
    folder = gate.epochs / str(EPOCH)
    folder.mkdir(parents=True)
    kind, shift = ('red', -0.25) if case == 'red_over_green' else ('applied', 0.25)
    there: Dict[str, Any] = {
        'epoch': EPOCH, 'head': HEAD, 'candidate': CAND, 'decision': 'green', 'evaluated': {'z': 2.5},
        'no_decision': False, 'merge_blocked': False, 'source': 'serial',
    }
    if case == 'other_candidate':
        there['candidate'] = '9' * 64
    elif case == 'other_decision':
        there['decision'] = 'red'
    raw = b'{"decision": "gre' if case == 'not_json' else json.dumps(there).encode()
    (folder / S.DECISION_FILE).write_bytes(raw)
    if case == 'lost_over_green':
        gate.outcome(EPOCH, 'lost_infra', None)
    elif case == 'untrained_over_green':
        # No grant of a new turn for an epoch that the gate decided before.
        gate.outcome(EPOCH, 'red', gate.judged(EPOCH, 0.0, trained=False)[1])
    else:
        gate.outcome(EPOCH, kind, gate.judged(EPOCH, shift)[1])

    done = gate.settler.run_once()
    assert [(one.status, one.decision) for one in done] == [(S.REFUSED, None)]
    assert 'conflicting DECISION.json' in done[0].detail
    assert (folder / S.DECISION_FILE).read_bytes() == raw
    assert gate.files() == ['epochs/%d/DECISION.json' % EPOCH] and gate.rounds.added == [] and gate.grants == []
    assert [code for code, _ in gate.alerts] == ['settle_conflict']
    # The refusal is kept: the next look does not try again and does not alert again.
    assert gate.settler.run_once() == [] and len(gate.alerts) == 1
    assert (folder / S.DECISION_FILE).read_bytes() == raw


def test_a_shadow_run_has_no_settler_and_a_shadow_writer_is_refused(gate: Gate, tmp_path: Path) -> None:
    with pytest.raises(S.SettleError, match='only a live run'):
        gate.make(mode=MODE_SHADOW)
    with pytest.raises(S.SettleError, match='only a live run'):
        gate.make(store=GuardedStore(RawStore(), RUN, MODE_SHADOW))
    shadow = HostGateWriter(epochs_root=gate.epochs, shadow=True, rounds=gate.rounds, baseline_v1=gate.world.live.BV)
    with pytest.raises(S.SettleError, match='shadow writer'):
        gate.make(writer=shadow)
    with pytest.raises(GateWriteError, match='shadow'):
        shadow.write_lost_infra(epoch=EPOCH, head='', candidate='', why='x')
    assert gate.files() == []


def test_an_outcome_without_the_record_of_the_judge_waits_and_pays_nothing(gate: Gate) -> None:
    gate.turn(EPOCH)
    gate.outcome(EPOCH, 'applied', 'ab' * 32)
    done = gate.settler.run_once()
    assert [(one.status, one.decision) for one in done] == [(S.WAITING, None)]
    assert gate.files() == [] and [code for code, _ in gate.alerts] == ['settle_no_record']
    # It is looked at again, and it is settled when the record is there.
    assert [one.status for one in gate.settler.run_once()] == [S.WAITING]


def test_the_journal_and_the_record_must_agree(gate: Gate) -> None:
    gate.turn(EPOCH)
    gate.turn(EPOCH + 1)
    gate.turn(EPOCH + 2)
    # A red record cannot pay as green, and a green record is not written as red.
    gate.outcome(EPOCH, 'applied', gate.judged(EPOCH, -0.25)[1])
    gate.outcome(EPOCH + 1, 'red', gate.judged(EPOCH + 1, 0.25)[1])
    # The state that was applied must be the state that was judged.
    gate.outcome(EPOCH + 2, 'applied', gate.judged(EPOCH + 2, 0.25)[1], state_id='7' * 64)
    done = gate.settler.run_once()
    assert [one.status for one in done] == [S.REFUSED] * 3
    assert gate.files() == [] and [code for code, _ in gate.alerts] == ['settle_mismatch'] * 3


def test_what_is_not_an_epoch_of_the_live_queue_is_skipped(gate: Gate) -> None:
    gate.outcome(1, 'applied', 'ab' * 32, batch_id='batch-0007')
    fields = gate.outcome(1, 'applied', 'ab' * 32, batch_id='batch-0008', publish=False)
    assert gate.settler.run_once() == []
    assert gate.settler.settle(fields).status == S.SKIPPED
    assert gate.settler.settle(dict(fields, batch_id='e5', outcome='running')).status == S.SKIPPED
    gate.store.put(gate.layout.outcome('e%d' % EPOCH), b'{"schema": "other"}')
    assert gate.settler.run_once() == [] and [code for code, _ in gate.alerts] == ['settle_failed']
    assert gate.files() == []


def test_a_missing_turn_file_is_reported_and_never_written(gate: Gate) -> None:
    gate.outcome(EPOCH, 'applied', gate.judged(EPOCH, 0.25)[1])
    assert [one.status for one in gate.settler.run_once()] == [S.WRITTEN]
    assert [code for code, _ in gate.alerts] == ['settle_no_turn']
    assert not gate.life.exists()


def test_a_missing_part_of_the_host_is_tried_again(gate: Gate) -> None:
    gate.turn(EPOCH)
    gate.outcome(EPOCH, 'red', gate.judged(EPOCH, 0.0, trained=False)[1])

    def no_grant(epoch: int, why: str) -> Any:
        raise GateWriteError('k1 grant fn unavailable')

    gate.writer.k1_grant_fn = no_grant
    assert [one.status for one in gate.settler.run_once()] == [S.WAITING]
    assert gate.files() == [] and [code for code, _ in gate.alerts] == ['settle_failed']
    gate.writer.k1_grant_fn = lambda epoch, why: 'granted'
    assert [(one.status, one.decision) for one in gate.settler.run_once()] == [(S.WRITTEN, 'no_decision')]


def test_the_lost_infra_decision_of_the_writer_is_idempotent_and_refuses_a_conflict(gate: Gate) -> None:
    writer = gate.writer
    path = writer.write_lost_infra(epoch=EPOCH, head='', candidate='', why='boxes lost', parallel_eval_meta={'a': 1})
    first = path.read_bytes()
    assert writer.write_lost_infra(epoch=EPOCH, head='', candidate='', why='other words') == path
    assert path.read_bytes() == first
    with pytest.raises(GateWriteError, match='conflicting'):
        writer.write_lost_infra(epoch=EPOCH, head='', candidate='9' * 64, why='boxes lost')
    with pytest.raises(GateWriteError, match='conflicting'):
        writer.write_no_decision(epoch=EPOCH, head='', candidate='', why='x')
    path.write_bytes(b'not json')
    with pytest.raises(GateWriteError, match='conflicting'):
        writer.write_lost_infra(epoch=EPOCH, head='', candidate='', why='boxes lost')
    assert path.read_bytes() == b'not json' and gate.rounds.added == [] and gate.grants == [(EPOCH, 'x')]
