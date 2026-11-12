"""The judge of the parallel pipeline: the decision function of the live gate.

A box result holds the probe of the candidate and can hold HEAD blocks (see
``baseline.py``). The verdict is the answer of the live scorer
(``ops/parallel_eval/live/pe_score.score``, the same estimator and the same
bar as the serial gate) for the candidate against the baseline of its own
base checkpoint, made by boxes of this run.

Fail closed:

* The probe of the candidate and the baseline must each have a hardware and
  layout fingerprint, and the two must be equal. If not, there is no verdict:
  the result is an infrastructure loss and the batch runs again on a new box.
* A result with tasks that are not tasks of the round is not a result of this
  round: also an infrastructure loss.
* A probe with too many errors gets no decision from the scorer: also a loss.
* While the baseline is not complete, there is no verdict yet
  (``JudgeNotReady``). The result is durable, so the wait costs nothing.

A batch that could not be trained because it was made for another checkpoint
than its base is RED for the controller (decided batches are final, and the
box is not at fault). The record of the judge says ``no_decision`` for it.

The judge writes one record for each result that it gives a verdict for, in a
private directory of the validator host. The record holds what the gate writer
needs for the decision of the epoch. The verdict of a result is read from its
record when there is one: a result never gets two verdicts.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional

from .. import probe_fingerprint
from ..pe_score import call_live_host_score
from .baseline import BaselineError, Baselines
from .feed import epoch_of
from .interfaces import VERDICT_GREEN, VERDICT_RED, JudgeInvalid, JudgeNotReady
from .layout import is_hex64, sha256_hex
from .masking import MaskError, mask_key, unmask
from .rounds import Round, RoundError, RoundNotReady, Rounds

RESULT_SCHEMA = 'affine-batchctl-result-v1'
RECORD_SCHEMA = 'affine-batchctl-judged-v1'
OUTCOME_PROBED, OUTCOME_NOT_TRAINED = 'probed', 'not_trained'
PROBE_DONE = ('ok', 'finalized')
DECISION_NONE = 'no_decision'
MAX_RECORD_BYTES = 8 << 20


class Fleet:
    """What the judge and the box agent must know about the leases of the
    controller. The runner gives the real one; the default knows nothing."""

    def live(self) -> Collection[int]:
        """The fences of the leases that run or wait for a verdict."""
        return ()

    def others(self, fence: int, base_state: str) -> int:
        """The number of other boxes that will start a job on this base state soon."""
        return 0

    def head_state(self) -> Optional[str]:
        """The state id of the newest HEAD, or None when it is not known."""
        return None


class Records:
    """One file for each judged result: ``<root>/<batch_id>/<sha256 of the result>.json``."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, batch_id: str, result_sha: str) -> Path:
        if epoch_of(batch_id) is None or not is_hex64(result_sha):
            raise ValueError('no record name for this batch and result')
        return self.root / batch_id / (result_sha + '.json')

    def get(self, batch_id: str, result_sha: str) -> Optional[Dict[str, Any]]:
        path = self.path(batch_id, result_sha)
        try:
            if path.stat().st_size > MAX_RECORD_BYTES:
                return None
            doc = json.loads(path.read_bytes())
        except (OSError, ValueError, RecursionError):
            return None
        if (isinstance(doc, dict) and doc.get('schema') == RECORD_SCHEMA and doc.get('batch_id') == batch_id
                and doc.get('result_sha') == result_sha and doc.get('decision') in (
                    VERDICT_GREEN, VERDICT_RED, DECISION_NONE)):
            return doc
        return None

    def put(self, batch_id: str, result_sha: str, doc: Mapping[str, Any]) -> None:
        path = self.path(batch_id, result_sha)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_name('.%s.%d.%d.tmp' % (path.name, os.getpid(), threading.get_ident()))
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as handle:
            handle.write(json.dumps(doc, sort_keys=True, allow_nan=False, default=str))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise JudgeInvalid(why)


class GateJudge:
    """``Judge`` on the scorer of the live gate.

    ``score(epoch, round_id, result, baseline=...)`` and
    ``blocks_baseline(round_id, blocks, rollouts=, max_tokens=, fingerprint=,
    done_at=)`` are the two functions of the live scorer module.
    """

    def __init__(self, *, run_id: str, rounds: Rounds, baselines: Baselines, records: Records,
                 score: Callable[..., Mapping[str, Any]], blocks_baseline: Callable[..., Mapping[str, Any]],
                 fleet: Optional[Fleet] = None, clock: Callable[[], float] = time.time,
                 log: Callable[[str], None] = lambda text: None):
        self.run_id = run_id
        self.rounds = rounds
        self.baselines = baselines
        self.records = records
        self.score = score
        self.blocks_baseline = blocks_baseline
        self.fleet = fleet if fleet is not None else Fleet()
        self.clock = clock
        self.log = log

    # -- the result ----------------------------------------------------------

    def _parse(self, result: bytes, batch_id: str, base_state: str) -> Dict[str, Any]:
        try:
            doc = json.loads(result)
        except (ValueError, RecursionError):
            raise JudgeInvalid('the result is not JSON') from None
        _need(isinstance(doc, dict) and doc.get('schema') == RESULT_SCHEMA, 'not a result of this version')
        _need(doc.get('run_id') == self.run_id and doc.get('batch_id') == batch_id
              and doc.get('base_state') == base_state, 'the result is for another run, batch or base state')
        _need(doc.get('outcome') in (OUTCOME_PROBED, OUTCOME_NOT_TRAINED), 'the result has no outcome')
        base = doc.get('base')
        _need(isinstance(base, dict) and is_hex64(base.get('checkpoint_id')), 'the result names no base checkpoint')
        return doc

    def _round(self, epoch: int) -> Round:
        try:
            return self.rounds.of(epoch)
        except RoundNotReady as exc:
            raise JudgeNotReady(str(exc)) from None
        except RoundError as exc:
            # The files of the host are at fault, not the box. Ask again when they are repaired.
            raise JudgeNotReady('round of epoch %d: %s' % (epoch, exc)) from None

    def _take_blocks(self, doc: Mapping[str, Any], rnd: Round, key: str, checkpoint: str, result_sha: str) -> None:
        """Store the HEAD blocks of a result. A block that is not usable is not stored; the result stays valid."""
        payload = doc.get('payload') if isinstance(doc.get('payload'), dict) else {}
        head = payload.get('head')
        entries = []
        if isinstance(head, dict) and head.get('checkpoint_id') == checkpoint and isinstance(head.get('blocks'), list):
            entries = head['blocks']
        for entry in entries:
            if not isinstance(entry, dict) or entry.get('status') not in PROBE_DONE:
                continue
            try:
                probe = unmask(entry.get('result'), key, {rnd.env_id: rnd.indices})
                fingerprint = probe_fingerprint.of(probe)
            except (MaskError, probe_fingerprint.FingerprintError) as exc:
                self.log('head block of %s is not usable: %s' % (doc.get('batch_id'), exc))
                continue
            refused = self.baselines.add(
                rnd, checkpoint, entry.get('block'), k=entry.get('k'), max_tokens=entry.get('max_tokens'),
                cells=probe['envs'][rnd.env_id]['per_task'], fingerprint=fingerprint,
                source={'batch_id': doc.get('batch_id'), 'fence': doc.get('fence'), 'result_sha': result_sha},
            )
            if refused is not None:
                self.log('head block %r of %s is not stored: %s' % (entry.get('block'), doc.get('batch_id'), refused))
        fence = doc.get('fence')
        if isinstance(fence, int) and not isinstance(fence, bool) and fence >= 0:
            # The box is gone. A block that it claimed and did not bring is free for another box.
            self.baselines.release(rnd, checkpoint, fence)

    def _baseline(self, rnd: Round, checkpoint: str, base_state: str) -> Mapping[str, Any]:
        held = self.baselines.complete(rnd, checkpoint)
        if held is None:
            newest = self.fleet.head_state()
            if newest is not None and newest != base_state and self.baselines.free(rnd, checkpoint, self.fleet.live()):
                # HEAD moved on. No new box starts from this checkpoint, so nobody will run the missing block.
                raise JudgeInvalid('the baseline of a replaced HEAD cannot be completed')
            raise JudgeNotReady('baseline of %s for round %s: blocks %s are missing' % (
                checkpoint[:12], rnd.round_id, self.baselines.missing(rnd, checkpoint)))
        return self.blocks_baseline(
            rnd.round_id, held['blocks'], rollouts=held['rollouts'], max_tokens=held['max_tokens'],
            fingerprint=held['fingerprint'], done_at=held['done_at'],
        )

    # -- the verdict ---------------------------------------------------------

    def verdict(self, result: bytes, *, batch_id: str, base_state: str) -> str:
        doc = self._parse(result, batch_id, base_state)
        epoch = epoch_of(batch_id)
        _need(epoch is not None, 'batch %r is not an epoch of the live queue' % (batch_id,))
        result_sha = sha256_hex(result)
        known = self.records.get(batch_id, result_sha)
        if known is not None:
            return VERDICT_GREEN if known['decision'] == VERDICT_GREEN else VERDICT_RED
        checkpoint = doc['base']['checkpoint_id']
        record: Dict[str, Any] = {
            'schema': RECORD_SCHEMA, 'run_id': self.run_id, 'batch_id': batch_id, 'epoch': epoch,
            'result_sha': result_sha, 'base_state': base_state, 'head': checkpoint, 'candidate': None,
            'fence': doc.get('fence'), 'attempt': doc.get('attempt'), 'box': doc.get('box'),
        }
        if doc['outcome'] == OUTCOME_NOT_TRAINED:
            record.update(decision=DECISION_NONE, reasons=['not_trained: %s' % str(doc.get('reason') or '')[:200]],
                          judged_at=self.clock())
            self.records.put(batch_id, result_sha, record)
            return VERDICT_RED

        candidate = doc.get('candidate')
        payload = doc.get('payload')
        _need(isinstance(candidate, dict) and is_hex64(candidate.get('checkpoint_id'))
              and is_hex64(candidate.get('state_id')), 'a probed result names no new state')
        _need(isinstance(payload, dict) and payload.get('candidate_id') == candidate['checkpoint_id'],
              'the probe ran on another checkpoint than the new state holds')
        probe = payload.get('probe')
        _need(isinstance(probe, dict) and probe.get('status') in PROBE_DONE, 'the probe did not finish')

        rnd = self._round(epoch)
        try:
            key = mask_key(rnd.plan_seed, self.run_id, batch_id)
            try:
                self._take_blocks(doc, rnd, key, checkpoint, result_sha)
                baseline = self._baseline(rnd, checkpoint, base_state)
            except (BaselineError, OSError) as exc:
                raise JudgeNotReady('the baseline store: %s' % exc) from None
            unmasked = unmask(probe.get('result'), key, {rnd.env_id: rnd.indices})
        except MaskError as exc:
            raise JudgeInvalid('the probe is not a probe of round %s: %s' % (rnd.round_id, exc)) from None

        try:
            scored = call_live_host_score(
                lambda epoch, rid, result: self.score(epoch, rid, result, baseline=baseline),
                epoch=epoch, round_id=rnd.round_id, result=unmasked, baseline_of=lambda epoch, rid: baseline,
            )
        except JudgeNotReady:
            raise
        except Exception as exc:
            raise JudgeInvalid('the scorer refused the result: %s: %s' % (type(exc).__name__, str(exc)[:300])) from None
        decision = str(scored.get('decision') or '').lower()
        if decision == 'pending':
            raise JudgeNotReady('the scorer waits: %s' % str(scored.get('pending_reason'))[:200])
        if decision not in (VERDICT_GREEN, VERDICT_RED):
            reasons: List[Any] = list(scored.get('reasons') or [scored.get('guard')])
            raise JudgeInvalid('no decision for this probe: %s' % '; '.join(str(r)[:200] for r in reasons[:3]))

        fingerprint = baseline.get(probe_fingerprint.KEY) if isinstance(baseline, Mapping) else None
        record.update(
            decision=decision, candidate=candidate['checkpoint_id'], state_id=candidate['state_id'],
            round_id=rnd.round_id, evaluated=scored.get('evaluated'), reasons=list(scored.get('reasons') or []),
            heldout=scored.get('heldout'), plan_seed=rnd.plan_seed, z_bar=scored.get('z_bar'),
            probe_status=probe['status'],
            fingerprint=fingerprint.get('sha256') if isinstance(fingerprint, Mapping) else None,
            judged_at=self.clock(),
        )
        self.records.put(batch_id, result_sha, record)
        return decision


__all__ = ['DECISION_NONE', 'Fleet', 'GateJudge', 'RECORD_SCHEMA', 'Records']
