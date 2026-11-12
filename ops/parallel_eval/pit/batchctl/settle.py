"""The hook from the outcomes of a live run into the pay path of the live gate.

The pay path exists: the K1 service joins the turn file of an epoch with the
gate decision of that epoch (``<gate state>/epochs/<N>/DECISION.json``), and
rebuilds the list of the last distinct GREEN miners on its next tick. This
module does one thing for it: after a batch has its final outcome, it writes
the decision of that epoch, through the one writer of the live gate
(``gate_writer.HostGateWriter``).

What it never does:

* It does not run in shadow mode. A settler of a shadow run cannot be built.
* It submits no weights and writes no score file. The K1 service does that.
* It writes no turn file. This controller is not the dispatcher of turns; a
  missing turn file is reported, because K1 cannot pay that epoch.
* It never writes over a decision. A decision that is there and is not the
  same decision is refused, and so is a decision file that cannot be read.

Outcome of the controller -> decision of the gate:

``applied``     ``green``. The round of the gate closes, as for each GREEN
                that moves HEAD.
``credited``    ``green`` with ``merge_blocked``: the miner is paid, HEAD did
                not move.
``red``         ``red`` with the numbers of the judge. A batch that the box
                could not train (it was made for another checkpoint) is
                ``no_decision``: the miner gets the turn again.
``lost_infra``  ``lost_infra``, with no evaluation. K1 pays it only with a
                matching row of the infra credits file of the operator.

Each write is idempotent: the same outcome gives the same decision, and the
writer takes the same decision again without a change. A mark file for each
settled batch saves the work of a second look; losing the marks loses
nothing.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..gate_writer import GateWriteError
from . import MODE_LIVE
from .feed import epoch_of
from .interfaces import NotFound, ObjectStore, StoreError
from .judge import DECISION_NONE, Records
from .layout import MAX_COMPLETE_BYTES, SCHEMA_OUTCOME, Layout, is_hex64, load_json

MARK_SCHEMA = 'affine-batchctl-settled-v1'
DECISION_FILE = 'DECISION.json'
TURN_FILE = 'K1-TURN.json'
OUTCOMES = ('applied', 'credited', 'red', 'lost_infra')
# What a decision of the gate is, for the question "is this the same decision".
IDENTITY = ('epoch', 'head', 'candidate', 'decision', 'no_decision', 'merge_blocked')

WRITTEN = 'written'
REFUSED = 'refused'
WAITING = 'waiting'
SKIPPED = 'skipped'


class SettleError(RuntimeError):
    """The settler cannot be built or cannot be used."""


@dataclass(frozen=True)
class Settled:
    batch_id: str
    status: str
    decision: Optional[str] = None
    detail: str = ''


def _write_private(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name('.%s.%d.%d.tmp' % (path.name, os.getpid(), threading.get_ident()))
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as handle:
        handle.write(json.dumps(doc, sort_keys=True, default=str))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class Settler:
    """Writes the gate decision of each batch that has its final outcome. Live mode only."""

    def __init__(self, *, run_id: str, mode: str, store: ObjectStore, records: Records, writer: Any, marks: Path,
                 life_epochs: Optional[Path] = None, alert: Callable[[str, Any], None] = lambda code, detail: None,
                 clock: Callable[[], float] = time.time, log: Callable[[str], None] = lambda text: None):
        if mode != MODE_LIVE or getattr(store, 'mode', mode) != MODE_LIVE:
            raise SettleError('only a live run writes gate decisions')
        if getattr(writer, 'shadow', False):
            raise SettleError('the gate writer is a shadow writer')
        for name in ('write_decision', 'write_no_decision', 'write_lost_infra'):
            if not callable(getattr(writer, name, None)):
                raise SettleError('the gate writer has no %s' % name)
        self.run_id = run_id
        self.store = store
        self.layout = Layout(store.prefix)
        self.records = records
        self.writer = writer
        self.epochs = Path(writer.epochs_root)
        self.marks = Path(marks)
        self.life_epochs = Path(life_epochs) if life_epochs is not None else None
        self.alert = alert
        self.clock = clock
        self.log = log
        self._lock = threading.Lock()

    # -- one outcome ---------------------------------------------------------

    def _mark(self, batch_id: str) -> Path:
        return self.marks / (batch_id + '.json')

    def _marked(self, batch_id: str) -> Optional[Dict[str, Any]]:
        try:
            doc = json.loads(self._mark(batch_id).read_bytes())
        except (OSError, ValueError, RecursionError):
            return None
        if isinstance(doc, dict) and doc.get('schema') == MARK_SCHEMA and doc.get('batch_id') == batch_id:
            return doc
        return None

    def _existing(self, epoch: int) -> Optional[Mapping[str, Any]]:
        """The decision that the gate has for this epoch. None: no file. Raises when a file is there and is not a decision."""
        path = self.epochs / str(epoch) / DECISION_FILE
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise GateWriteError('conflicting DECISION.json for epoch %d: it cannot be read (%s)' % (
                epoch, type(exc).__name__)) from None
        try:
            doc = json.loads(raw)
        except (ValueError, RecursionError):
            doc = None
        if not isinstance(doc, dict) or not doc.get('decision'):
            raise GateWriteError('conflicting DECISION.json for epoch %d: it is not a decision' % epoch)
        return doc

    def _plan(self, outcome: Mapping[str, Any], epoch: int) -> Optional[Dict[str, Any]]:
        """What to write for this outcome. None: the record of the judge is not there yet."""
        kind = outcome['outcome']
        meta = {
            'source': 'batchctl', 'run_id': self.run_id, 'batch_id': outcome['batch_id'], 'outcome': kind,
            'applied': bool(outcome.get('applied')), 'head_seq': outcome.get('head_seq'),
            'base_seq': outcome.get('base_seq'), 'base_state': outcome.get('base_state'),
            'state_id': outcome.get('state_id'), 'result_sha': outcome.get('result_sha'),
            'attempts': outcome.get('attempts'), 'infra_failures': outcome.get('infra_failures'),
            'note': outcome.get('note'),
        }
        if kind == 'lost_infra':
            why = 'every box of the batch was lost before a decision (%s tries)' % outcome.get('infra_failures')
            return {
                'call': 'write_lost_infra', 'decision': 'lost_infra',
                'identity': {'epoch': epoch, 'head': '', 'candidate': '', 'decision': 'lost_infra',
                             'no_decision': False, 'merge_blocked': False, 'z': None},
                'args': {'epoch': epoch, 'head': '', 'candidate': '', 'why': why, 'parallel_eval_meta': meta},
            }
        result_sha = outcome.get('result_sha')
        record = self.records.get(outcome['batch_id'], result_sha) if is_hex64(result_sha) else None
        if record is None:
            return None
        head, candidate = str(record.get('head') or ''), str(record.get('candidate') or '')
        meta.update(baseline='head-blocks', fingerprint=record.get('fingerprint'), round_id=record.get('round_id'),
                    fence=record.get('fence'), probe_status=record.get('probe_status'))
        if record['decision'] == DECISION_NONE:
            if kind != 'red':
                raise SettleError('outcome %s of %s has a record with no decision' % (kind, outcome['batch_id']))
            why = '; '.join(str(reason)[:200] for reason in (record.get('reasons') or ['no decision'])[:3])
            return {
                'call': 'write_no_decision', 'decision': 'no_decision',
                'identity': {'epoch': epoch, 'head': head, 'candidate': candidate, 'decision': 'no_decision',
                             'no_decision': True, 'merge_blocked': False, 'z': None},
                'args': {'epoch': epoch, 'head': head, 'candidate': candidate, 'why': why},
            }
        decision = 'red' if kind == 'red' else 'green'
        if record['decision'] != decision:
            # The journal and the record of the judge disagree. Never pay on that.
            raise SettleError('outcome %s of %s has a %s record' % (kind, outcome['batch_id'], record['decision']))
        if kind == 'applied' and record.get('state_id') != outcome.get('state_id'):
            raise SettleError('the applied state of %s is not the state that was judged' % outcome['batch_id'])
        evaluated = record.get('evaluated')
        blocked = kind == 'credited'
        reasons = list(record.get('reasons') or [])
        if blocked and 'merge_blocked' not in reasons:
            reasons.append('merge_blocked')
        return {
            'call': 'write_decision', 'decision': decision,
            'identity': {'epoch': epoch, 'head': head, 'candidate': candidate, 'decision': decision,
                         'no_decision': False, 'merge_blocked': blocked,
                         'z': evaluated.get('z') if isinstance(evaluated, Mapping) else None},
            'args': {
                'epoch': epoch, 'head': head, 'candidate': candidate, 'decision': decision, 'evaluated': evaluated,
                'reasons': reasons, 'heldout': record.get('heldout'), 'plan_seed': str(record.get('plan_seed') or ''),
                'parallel_eval_meta': dict(meta, merge_blocked=blocked, training_head=head),
                'no_decision': False, 'merge_blocked': blocked,
            },
        }

    def settle(self, outcome: Mapping[str, Any]) -> Settled:
        """Write the gate decision of one final outcome. Safe to call again."""
        batch_id = outcome.get('batch_id')
        epoch = epoch_of(batch_id)
        if epoch is None or outcome.get('outcome') not in OUTCOMES:
            return Settled(str(batch_id), SKIPPED, detail='not an outcome of an epoch of the live queue')
        with self._lock:
            marked = self._marked(batch_id)
            if marked is not None and marked.get('status') in (WRITTEN, REFUSED):
                return Settled(batch_id, marked['status'], marked.get('decision'), str(marked.get('detail') or ''))
            try:
                plan = self._plan(outcome, epoch)
                if plan is None:
                    self.alert('settle_no_record', batch_id)
                    return Settled(batch_id, WAITING, detail='the record of the judge is not there')
                existing = self._existing(epoch)
                if existing is not None:
                    have = {key: existing.get(key) for key in IDENTITY}
                    have['z'] = (existing.get('evaluated') or {}).get('z') if isinstance(
                        existing.get('evaluated'), Mapping) else None
                    want = dict(plan['identity'])
                    for key in ('no_decision', 'merge_blocked'):
                        have[key] = bool(have[key])
                    if have != want:
                        raise GateWriteError('conflicting DECISION.json for epoch %d: the gate has %s for candidate %s' % (
                            epoch, existing.get('decision'), str(existing.get('candidate') or '')[:12]))
                path = getattr(self.writer, plan['call'])(**plan['args'])
            except GateWriteError as exc:
                text = str(exc)[:300]
                if 'conflicting' not in text:
                    # A part of the host is missing (rounds, the grant function). Try again later.
                    self.alert('settle_failed', '%s: %s' % (batch_id, text))
                    return Settled(batch_id, WAITING, detail=text)
                self.alert('settle_conflict', '%s: %s' % (batch_id, text))
                self._write_mark(batch_id, epoch, REFUSED, None, text)
                return Settled(batch_id, REFUSED, detail=text)
            except SettleError as exc:
                text = str(exc)[:300]
                self.alert('settle_mismatch', text)
                self._write_mark(batch_id, epoch, REFUSED, None, text)
                return Settled(batch_id, REFUSED, detail=text)
            if self.life_epochs is not None and not (self.life_epochs / str(epoch) / TURN_FILE).is_file():
                # This process is not the dispatcher: it does not write the turn. Without it K1 does not see the epoch.
                self.alert('settle_no_turn', batch_id)
            self._write_mark(batch_id, epoch, WRITTEN, plan['decision'], str(path))
            self.log('settled %s: %s' % (batch_id, plan['decision']))
            return Settled(batch_id, WRITTEN, plan['decision'], str(path))

    def _write_mark(self, batch_id: str, epoch: int, status: str, decision: Optional[str], detail: str) -> None:
        _write_private(self._mark(batch_id), {
            'schema': MARK_SCHEMA, 'run_id': self.run_id, 'batch_id': batch_id, 'epoch': epoch, 'status': status,
            'decision': decision, 'detail': detail, 'at': self.clock(),
        })

    # -- every outcome of the run --------------------------------------------

    def run_once(self) -> List[Settled]:
        """Settle each published outcome that has no mark yet. -> what was done now."""
        done: List[Settled] = []
        prefix = self.layout.outcome('')[:-len('.json')]
        try:
            listed = self.store.list(prefix)
        except StoreError as exc:
            self.alert('settle_failed', 'cannot list the outcomes: %s' % type(exc).__name__)
            return done
        for info in listed:
            name = info.key[len(prefix):]
            if not name.endswith('.json') or '/' in name:
                continue
            batch_id = name[:-len('.json')]
            marked = self._marked(batch_id)
            if epoch_of(batch_id) is None or (marked is not None and marked.get('status') in (WRITTEN, REFUSED)):
                continue
            try:
                doc = load_json(self.store.get(info.key, max_bytes=MAX_COMPLETE_BYTES).data)
            except (NotFound, StoreError, ValueError):
                continue
            if not isinstance(doc, dict) or doc.get('schema') != SCHEMA_OUTCOME or doc.get('batch_id') != batch_id:
                self.alert('settle_failed', 'outcome %s is not an outcome document' % batch_id)
                continue
            try:
                done.append(self.settle(doc))
            except Exception as exc:
                # A fault of one epoch must not stop the others.
                self.alert('settle_failed', '%s: %s: %s' % (batch_id, type(exc).__name__, str(exc)[:200]))
        return done


__all__ = ['Settled', 'Settler', 'SettleError']
