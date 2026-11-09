"""Single-writer path for PIT live gate epoch artifacts.

All live ``DECISION.json`` / ``EVALUATED.private.json`` writes go through a
``GateWriter``. Shadow mode refuses every write.
"""
from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

from .decision_emit import GATE_VERSION, _cr, map_pit_decision

HELDOUT_REQUIRED_KEYS = frozenset({
    'round_id',
    'set_size',
    'bar_z',
    'k_cand',
    'head_rollouts',
    'scoring',
    'settings_hash',
    'baseline',
    'max_tokens',
    'scoring_mode',
    'tau_run',
    'sigma_run_floor',
    'baseline_done_at_cr',
})


class GateWriteError(RuntimeError):
    pass


def _load_baseline_v1():
    for name in ('baseline_v1', 'priors.baseline_v1'):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError('baseline_v1 not available for gate publish guard')


def validate_heldout(heldout: Mapping[str, Any]) -> None:
    if not isinstance(heldout, Mapping):
        raise GateWriteError('heldout must be a mapping')
    missing = sorted(k for k in HELDOUT_REQUIRED_KEYS if k not in heldout or heldout[k] is None)
    if missing:
        raise GateWriteError('heldout missing required keys: %s' % ', '.join(missing))


def _wj(path: Path, obj: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _assert_public(bv: Any, evaluated: Mapping[str, Any], reasons: Any, heldout: Mapping[str, Any]) -> None:
    bv.assert_public(dict(evaluated=evaluated, reasons=reasons, heldout=heldout), True)


@runtime_checkable
class GateWriter(Protocol):
    def write_decision(
        self,
        *,
        epoch: int,
        head: str,
        candidate: str,
        decision: str,
        evaluated: Mapping[str, Any],
        reasons: Any,
        heldout: Mapping[str, Any],
        plan_seed: str,
        parallel_eval_meta: Mapping[str, Any],
        no_decision: bool = False,
        merge_blocked: bool = False,
    ) -> Path:
        ...

    def write_no_decision(
        self,
        *,
        epoch: int,
        head: str,
        candidate: str,
        why: str,
    ) -> Path:
        ...


def _resolve_k1_grant_fn(explicit: Optional[Callable[[int, str], Any]]) -> Optional[Callable[[int, str], Any]]:
    if explicit is not None:
        return explicit
    for name in ('gate_service', 'ops.promotion_gate.gate_service'):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        for attr in ('_k1_grant_no_decision', 'k1_grant_no_decision', '_grant_no_decision'):
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
    return None


@dataclass
class HostGateWriter:
    """Writes through host gate modules when available (BV.assert_public, R.turn_add)."""

    epochs_root: Path
    shadow: bool = False
    gate_version: str = GATE_VERSION
    baseline_v1: Any = None
    rounds: Any = None
    k1_grant_fn: Optional[Callable[[int, str], Any]] = None

    def __post_init__(self) -> None:
        self._bv = self.baseline_v1 if self.baseline_v1 is not None else _load_baseline_v1()
        self._k1_grant = _resolve_k1_grant_fn(self.k1_grant_fn)

    def _rounds(self):
        if self.rounds is not None:
            return self.rounds
        for name in ('heldout', 'miner.validator.gate.heldout'):
            try:
                mod = importlib.import_module(name)
            except ImportError:
                continue
            factory = getattr(mod, 'rounds', None)
            if callable(factory):
                return factory()
        raise GateWriteError('host heldout rounds() unavailable')

    def _refuse_shadow(self) -> None:
        if self.shadow:
            raise GateWriteError('refusing gate write in shadow mode')

    def write_decision(
        self,
        *,
        epoch: int,
        head: str,
        candidate: str,
        decision: str,
        evaluated: Mapping[str, Any],
        reasons: Any,
        heldout: Mapping[str, Any],
        plan_seed: str,
        parallel_eval_meta: Mapping[str, Any],
        no_decision: bool = False,
        merge_blocked: bool = False,
    ) -> Path:
        self._refuse_shadow()
        validate_heldout(heldout)
        reasons_list = list(reasons or [])
        _assert_public(self._bv, evaluated, reasons_list, heldout)
        gate_decision = map_pit_decision(decision)
        now = time.time()
        rid = str(heldout['round_id'])
        dec = dict(
            version=self.gate_version,
            epoch=int(epoch),
            head=head,
            candidate=candidate,
            mode='live',
            decided_at=now,
            decided_at_cr=_cr(now),
            decision=gate_decision,
            evaluated=dict(evaluated),
            reasons=reasons_list,
            heldout=dict(heldout),
            parallel_eval=dict(parallel_eval_meta),
            no_decision=bool(no_decision or gate_decision == 'no_decision'),
            merge_blocked=bool(merge_blocked),
            chart_status=(
                'no_decision' if (no_decision or gate_decision == 'no_decision') else gate_decision
            ),
            red_credit='zero',
            source='pit',
        )
        if dec['no_decision']:
            grant = self._k1_grant(epoch, '; '.join(reasons_list) if reasons_list else 'no_decision') if self._k1_grant else None
            if grant is not None:
                dec['k1_grant'] = grant
        epoch_dir = self.epochs_root / str(int(epoch))
        _wj(epoch_dir / 'EVALUATED.private.json', dict(evaluated=dec['evaluated']), 0o600)
        decision_path = epoch_dir / 'DECISION.json'
        _wj(decision_path, dec, 0o600)
        R = self._rounds()
        z_val = (dec['evaluated'] or {}).get('z')
        R.turn_add(
            rid,
            dict(
                epoch=int(epoch),
                candidate=candidate,
                decision=gate_decision,
                z=z_val,
                plan_seed=plan_seed,
                batch=[],
                evaluated=dec['evaluated'],
                reasons=dec['reasons'],
                decided_at=dec['decided_at'],
                prior_v1=True,
                parallel_eval=True,
            ),
        )
        if gate_decision == 'green' and not merge_blocked and not dec['no_decision']:
            if not R.closed(rid):
                R.close(rid, 'green e%d (candidate %s, pit)' % (epoch, candidate[:12]), epoch)
        return decision_path

    def write_no_decision(
        self,
        *,
        epoch: int,
        head: str,
        candidate: str,
        why: str,
    ) -> Path:
        self._refuse_shadow()
        grant = self._k1_grant(epoch, why) if self._k1_grant else None
        heldout = dict(round_id='', no_decision=True)
        now = time.time()
        dec = dict(
            version=self.gate_version,
            epoch=int(epoch),
            head=head,
            candidate=candidate,
            mode='live',
            decided_at=now,
            decided_at_cr=_cr(now),
            decision='no_decision',
            evaluated=dict(z=None),
            reasons=[why],
            heldout=heldout,
            no_decision=True,
            chart_status='no_decision',
            red_credit='zero',
            source='pit',
        )
        if grant is not None:
            dec['k1_grant'] = grant
        epoch_dir = self.epochs_root / str(int(epoch))
        decision_path = epoch_dir / 'DECISION.json'
        _wj(decision_path, dec, 0o600)
        return decision_path


@dataclass
class RecordingGateWriter:
    """In-memory test double with the same heldout / publish guards as live."""

    epochs_root: Path
    shadow: bool = False
    baseline_v1: Any = None
    turn_adds: list = field(default_factory=list)
    closes: list = field(default_factory=list)
    closed_rounds: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self._bv = self.baseline_v1 if self.baseline_v1 is not None else _load_baseline_v1()

    def _refuse_shadow(self) -> None:
        if self.shadow:
            raise GateWriteError('refusing gate write in shadow mode')

    def turn_add(self, rid: str, rec: Mapping[str, Any]) -> None:
        self.turn_adds.append((rid, dict(rec)))

    def close(self, rid: str, reason: str, epoch: int) -> None:
        self.closes.append((rid, reason, epoch))
        self.closed_rounds.add(rid)

    def closed(self, rid: str) -> bool:
        return rid in self.closed_rounds

    def write_decision(
        self,
        *,
        epoch: int,
        head: str,
        candidate: str,
        decision: str,
        evaluated: Mapping[str, Any],
        reasons: Any,
        heldout: Mapping[str, Any],
        plan_seed: str,
        parallel_eval_meta: Mapping[str, Any],
        no_decision: bool = False,
        merge_blocked: bool = False,
    ) -> Path:
        self._refuse_shadow()
        validate_heldout(heldout)
        reasons_list = list(reasons or [])
        _assert_public(self._bv, evaluated, reasons_list, heldout)
        gate_decision = map_pit_decision(decision)
        now = time.time()
        rid = str(heldout['round_id'])
        dec = dict(
            version=GATE_VERSION,
            epoch=int(epoch),
            head=head,
            candidate=candidate,
            mode='live',
            decided_at=now,
            decided_at_cr=_cr(now),
            decision=gate_decision,
            evaluated=dict(evaluated),
            reasons=reasons_list,
            heldout=dict(heldout),
            parallel_eval=dict(parallel_eval_meta),
            plan_seed=plan_seed,
            no_decision=bool(no_decision or gate_decision == 'no_decision'),
            merge_blocked=bool(merge_blocked),
            source='pit',
        )
        epoch_dir = self.epochs_root / str(int(epoch))
        _wj(epoch_dir / 'EVALUATED.private.json', dict(evaluated=dec['evaluated']), 0o600)
        decision_path = epoch_dir / 'DECISION.json'
        _wj(decision_path, dec, 0o600)
        z_val = (dec['evaluated'] or {}).get('z')
        self.turn_add(
            rid,
            dict(
                epoch=int(epoch),
                candidate=candidate,
                decision=gate_decision,
                z=z_val,
                plan_seed=plan_seed,
                batch=[],
                evaluated=dec['evaluated'],
                reasons=dec['reasons'],
                decided_at=dec['decided_at'],
                prior_v1=True,
                parallel_eval=True,
            ),
        )
        if gate_decision == 'green' and not merge_blocked and not dec['no_decision']:
            if not self.closed(rid):
                self.close(rid, 'green e%d (pit test)' % epoch, epoch)
        return decision_path

    def write_no_decision(
        self,
        *,
        epoch: int,
        head: str,
        candidate: str,
        why: str,
    ) -> Path:
        self._refuse_shadow()
        now = time.time()
        dec = dict(
            version=GATE_VERSION,
            epoch=int(epoch),
            head=head,
            candidate=candidate,
            mode='live',
            decided_at=now,
            decided_at_cr=_cr(now),
            decision='no_decision',
            evaluated=dict(z=None),
            reasons=[why],
            heldout=dict(no_decision=True),
            no_decision=True,
            k1_grant='recorded',
            source='pit',
        )
        epoch_dir = self.epochs_root / str(int(epoch))
        decision_path = epoch_dir / 'DECISION.json'
        _wj(decision_path, dec, 0o600)
        return decision_path
