"""Single-writer path for PIT live gate epoch artifacts.

All live ``DECISION.json`` / ``EVALUATED.private.json`` writes go through a
``GateWriter``. Shadow mode refuses every write.

Rounds / K1 grants must be wired explicitly (or resolved fail-closed). Never
write EVALUATED/DECISION before dependencies resolve — a mid-write failure must
not leave a half-published GREEN.
"""
from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

from .decision_emit import BLOCKED_DECISIONS, GATE_VERSION, _cr, map_pit_decision

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

# Core identity for idempotent re-emit. Timestamps / notices / reason text may
# differ on retry (e1055 recovery) without meaning a different verdict.
_DECISION_IDENTITY_KEYS = (
    'epoch', 'head', 'candidate', 'decision', 'no_decision', 'merge_blocked',
)


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


def _rj(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _log(**kw):
    kw = dict(kw)
    kw['at'] = round(time.time(), 1)
    kw['at_cr'] = _cr(kw['at'])
    print(json.dumps(kw, default=str), flush=True)


def _assert_public(bv: Any, evaluated: Mapping[str, Any], reasons: Any, heldout: Mapping[str, Any]) -> None:
    bv.assert_public(dict(evaluated=evaluated, reasons=reasons, heldout=heldout), True)


def gate_state_root(explicit: Optional[str] = None) -> Path:
    """Gate private state root (parent of ``epochs/`` and ``heldout/``).

    Env (first match): ``HG_GATE_STATE``, ``PE_GATE_STATE``, or parent of
    ``PE_GATE_EP``. No hardcoded host paths.
    """
    if explicit:
        return Path(explicit)
    for key in ('HG_GATE_STATE', 'PE_GATE_STATE'):
        env = os.environ.get(key)
        if env:
            return Path(env)
    ep = os.environ.get('PE_GATE_EP')
    if ep:
        return Path(ep).parent
    raise GateWriteError(
        'gate state root unset: set HG_GATE_STATE or PE_GATE_STATE '
        '(or PE_GATE_EP pointing at <gate-state>/epochs)'
    )


def heldout_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get('PE_HELDOUT')
    if env:
        return Path(env)
    return gate_state_root() / 'heldout'


def _import_gate_service():
    """Import gate_service when present on PYTHONPATH (host promotion-gate code).

    Side effects if imported from ``ops.promotion_gate.gate_service``: module-level
    ``sys.path`` insert and default path constants. Prefer an explicit
    ``rounds=`` / ``build_host_rounds()`` construction from env instead.
    """
    errors = []
    for name in (
        'gate_service',
        'ops.promotion_gate.gate_service',
        'miner.validator.gate.gate_service',
    ):
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            errors.append('%s: %s' % (name, exc))
    return None


def build_host_rounds(
    *,
    gate_state: Optional[str | Path] = None,
    heldout: Optional[str | Path] = None,
) -> Any:
    """Build ``heldout.Rounds`` the way ``gate_service.rounds()`` does.

    Prefer constructing ``Rounds(ST/'heldout', wj, rj, log, cr)`` from env so
    we do not need to import gate_service (host-path side effects). Fall back to
    ``gate_service.rounds()`` when that module is importable.
    """
    # Fallback first only when the operator has already put gate_service on the
    # path AND we cannot resolve a heldout root from env — otherwise build local.
    root: Optional[Path] = None
    try:
        root = heldout_root(str(heldout) if heldout is not None else None)
    except GateWriteError:
        root = None
    if root is None and gate_state is not None:
        root = Path(gate_state) / 'heldout'

    if root is not None:
        HO = None
        for name in ('heldout', 'miner.validator.gate.heldout'):
            try:
                HO = importlib.import_module(name)
                break
            except ImportError:
                continue
        if HO is not None and hasattr(HO, 'Rounds'):
            return HO.Rounds(root, _wj, _rj, _log, _cr)

    mod = _import_gate_service()
    if mod is not None:
        factory = getattr(mod, 'rounds', None)
        if callable(factory):
            return factory()

    raise GateWriteError(
        'host heldout rounds() unavailable: need heldout.Rounds + '
        'HG_GATE_STATE/PE_GATE_STATE/PE_HELDOUT (or importable gate_service.rounds)'
    )


def _resolve_k1_grant_fn(
    explicit: Optional[Callable[[int, str], Any]],
    *,
    required: bool,
) -> Optional[Callable[[int, str], Any]]:
    if explicit is not None:
        return explicit
    mod = _import_gate_service()
    if mod is not None:
        for attr in ('_k1_grant_no_decision', 'k1_grant_no_decision', '_grant_no_decision'):
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
    if required:
        raise GateWriteError(
            'k1 grant fn unavailable: pass k1_grant_fn= or put gate_service '
            '(_k1_grant_no_decision) on PYTHONPATH'
        )
    return None


def _decision_identity(doc: Mapping[str, Any]) -> dict:
    out = {k: doc.get(k) for k in _DECISION_IDENTITY_KEYS}
    ev = doc.get('evaluated') or {}
    out['evaluated_z'] = ev.get('z') if isinstance(ev, Mapping) else None
    return out


def _same_decision(existing: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    return _decision_identity(existing) == _decision_identity(new)


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

    def write_no_sample(
        self,
        *,
        epoch: int,
        head: str,
        candidate: str,
        why: str,
        parallel_eval_meta: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Terminal no_sample: DECISION only — no heldout turn credit, no k1 grant."""
        ...


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
        # Do not silently bind a missing k1 grant — resolve fail-closed when needed.

    def _rounds(self):
        if self.rounds is not None:
            return self.rounds
        # Fail closed: never invent a no-op rounds. Prefer explicit rounds= from
        # host_wiring / run_host; fall back to build_host_rounds / gate_service.
        return build_host_rounds()

    def _k1_grant(self, epoch: int, why: str) -> Any:
        fn = _resolve_k1_grant_fn(self.k1_grant_fn, required=True)
        assert fn is not None
        return fn(epoch, why)

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
        blocked = gate_decision in BLOCKED_DECISIONS
        # Resolve ALL side-effect deps BEFORE any epoch artifact write.
        R = self._rounds()
        need_k1 = bool(no_decision or gate_decision == 'no_decision')
        grant = None
        if need_k1:
            grant = self._k1_grant(
                epoch, '; '.join(reasons_list) if reasons_list else 'no_decision',
            )
        now = time.time()
        rid = str(heldout['round_id'])
        pe_meta = dict(parallel_eval_meta)
        if blocked:
            pe_meta.setdefault('settlement', 'pending_operator')
            pe_meta.setdefault('blocked', gate_decision)
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
            parallel_eval=pe_meta,
            no_decision=bool(no_decision or gate_decision == 'no_decision'),
            merge_blocked=bool(merge_blocked),
            chart_status=(
                'no_decision' if (no_decision or gate_decision == 'no_decision') else gate_decision
            ),
            red_credit='zero',
            source='pit',
        )
        if blocked:
            # Terminal non-GREEN: free pending via DECISION; never payout / HEAD.
            dec['settlement'] = 'pending_operator'
            dec['paid'] = False
        if grant is not None:
            dec['k1_grant'] = grant
        epoch_dir = self.epochs_root / str(int(epoch))
        decision_path = epoch_dir / 'DECISION.json'
        archived = epoch_dir / 'attempt-1' / 'DECISION.json'
        if decision_path.exists():
            existing = _rj(decision_path)
            if not isinstance(existing, Mapping):
                raise GateWriteError(
                    'existing DECISION.json unreadable for epoch %d' % int(epoch)
                )
            if not _same_decision(existing, dec):
                # Sanctioned rerun archives the prior DECISION under attempt-1/.
                # Never overwrite a live non-archived decision.
                if archived.is_file():
                    raise GateWriteError(
                        'conflicting DECISION.json for epoch %d '
                        '(attempt-1 archived but live DECISION still present; '
                        'refuse overwrite)' % int(epoch)
                    )
                raise GateWriteError(
                    'conflicting DECISION.json for epoch %d '
                    '(refuse overwrite of different decision)' % int(epoch)
                )
            # Identical content: idempotent re-emit. Still ensure turn/close.
            self._turn_and_maybe_close(
                R, rid=rid, dec=existing if isinstance(existing, dict) else dec,
                plan_seed=plan_seed, gate_decision=gate_decision,
                merge_blocked=merge_blocked,
            )
            return decision_path
        # Fresh DECISION after attempt-1 archive (or first write) is allowed.
        _wj(epoch_dir / 'EVALUATED.private.json', dict(evaluated=dec['evaluated']), 0o600)
        _wj(decision_path, dec, 0o600)
        self._turn_and_maybe_close(
            R, rid=rid, dec=dec, plan_seed=plan_seed,
            gate_decision=gate_decision, merge_blocked=merge_blocked,
        )
        return decision_path

    def _turn_and_maybe_close(
        self,
        R: Any,
        *,
        rid: str,
        dec: Mapping[str, Any],
        plan_seed: str,
        gate_decision: str,
        merge_blocked: bool,
    ) -> None:
        z_val = (dec.get('evaluated') or {}).get('z')
        epoch = int(dec['epoch'])
        candidate = str(dec.get('candidate') or '')
        # Skip duplicate turn rows for the same epoch+candidate+decision.
        try:
            prior = list(R.turns(rid) or [])
        except Exception:
            prior = []
        already = any(
            int(t.get('epoch', -1)) == epoch
            and str(t.get('candidate') or '') == candidate
            and str(t.get('decision') or '') == gate_decision
            for t in prior
            if isinstance(t, Mapping)
        )
        if not already:
            R.turn_add(
                rid,
                dict(
                    epoch=epoch,
                    candidate=candidate,
                    decision=gate_decision,
                    z=z_val,
                    plan_seed=plan_seed,
                    batch=[],
                    evaluated=dec.get('evaluated'),
                    reasons=dec.get('reasons'),
                    decided_at=dec.get('decided_at'),
                    prior_v1=True,
                    parallel_eval=True,
                ),
            )
        if gate_decision == 'green' and not merge_blocked and not dec.get('no_decision'):
            if not R.closed(rid):
                R.close(rid, 'green e%d (candidate %s, pit)' % (epoch, candidate[:12]), epoch)

    def write_no_decision(
        self,
        *,
        epoch: int,
        head: str,
        candidate: str,
        why: str,
    ) -> Path:
        self._refuse_shadow()
        # Resolve grant BEFORE writing DECISION.json.
        grant = self._k1_grant(epoch, why)
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
            k1_grant=grant,
        )
        epoch_dir = self.epochs_root / str(int(epoch))
        decision_path = epoch_dir / 'DECISION.json'
        if decision_path.exists():
            existing = _rj(decision_path)
            if isinstance(existing, Mapping) and _same_decision(existing, dec):
                return decision_path
            if isinstance(existing, Mapping):
                raise GateWriteError(
                    'conflicting DECISION.json for epoch %d' % int(epoch)
                )
        _wj(decision_path, dec, 0o600)
        return decision_path

    def write_no_sample(
        self,
        *,
        epoch: int,
        head: str,
        candidate: str,
        why: str,
        parallel_eval_meta: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Write DECISION=no_sample with settlement=pending_operator.

        No heldout validation, no rounds turn_add (no turn credit), no k1 grant,
        no APPLY/HEAD/bench. Frees lifecycle pending via DECISION + PIT-RESULT.
        """
        self._refuse_shadow()
        now = time.time()
        pe_meta = dict(parallel_eval_meta or {})
        pe_meta.setdefault('settlement', 'pending_operator')
        pe_meta.setdefault('terminal', 'no_sample')
        pe_meta.setdefault('no_sample_reason', (why or '')[:400])
        dec = dict(
            version=self.gate_version,
            epoch=int(epoch),
            head=head or '',
            candidate=candidate or '',
            mode='live',
            decided_at=now,
            decided_at_cr=_cr(now),
            decision='no_sample',
            evaluated=dict(z=None),
            reasons=['no_sample', (why or '')[:200]],
            heldout={},
            parallel_eval=pe_meta,
            no_decision=False,
            merge_blocked=False,
            chart_status='no_sample',
            red_credit='zero',
            source='pit',
            settlement='pending_operator',
            paid=False,
        )
        epoch_dir = self.epochs_root / str(int(epoch))
        decision_path = epoch_dir / 'DECISION.json'
        if decision_path.exists():
            existing = _rj(decision_path)
            if isinstance(existing, Mapping) and _same_decision(existing, dec):
                return decision_path
            if isinstance(existing, Mapping):
                raise GateWriteError(
                    'conflicting DECISION.json for epoch %d' % int(epoch)
                )
        _wj(decision_path, dec, 0o600)
        return decision_path

    def write_lost_infra(
        self,
        *,
        epoch: int,
        head: str,
        candidate: str,
        why: str,
        parallel_eval_meta: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Write DECISION=lost_infra with settlement=pending_operator.

        The boxes of the epoch were lost before a decision, so there is no evaluation.
        No heldout validation, no rounds turn_add (no turn credit), no k1 grant, no
        APPLY/HEAD/bench. K1 pays the epoch only with a matching K1-INFRA-CREDITS row.
        """
        self._refuse_shadow()
        now = time.time()
        pe_meta = dict(parallel_eval_meta or {})
        pe_meta.setdefault('settlement', 'pending_operator')
        pe_meta.setdefault('terminal', 'lost_infra')
        pe_meta.setdefault('lost_infra_reason', (why or '')[:400])
        dec = dict(
            version=self.gate_version,
            epoch=int(epoch),
            head=head or '',
            candidate=candidate or '',
            mode='live',
            decided_at=now,
            decided_at_cr=_cr(now),
            decision='lost_infra',
            evaluated=dict(z=None),
            reasons=['lost_infra', (why or '')[:200]],
            heldout={},
            parallel_eval=pe_meta,
            no_decision=False,
            merge_blocked=False,
            chart_status='lost_infra',
            red_credit='zero',
            source='pit',
            settlement='pending_operator',
            paid=False,
        )
        epoch_dir = self.epochs_root / str(int(epoch))
        decision_path = epoch_dir / 'DECISION.json'
        if decision_path.exists():
            existing = _rj(decision_path)
            if isinstance(existing, Mapping) and _same_decision(existing, dec):
                return decision_path
            # An unreadable file is a conflict too: never write over what is there.
            raise GateWriteError(
                'conflicting DECISION.json for epoch %d' % int(epoch)
            )
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
        blocked = gate_decision in BLOCKED_DECISIONS
        now = time.time()
        rid = str(heldout['round_id'])
        pe_meta = dict(parallel_eval_meta)
        if blocked:
            pe_meta.setdefault('settlement', 'pending_operator')
            pe_meta.setdefault('blocked', gate_decision)
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
            parallel_eval=pe_meta,
            plan_seed=plan_seed,
            no_decision=bool(no_decision or gate_decision == 'no_decision'),
            merge_blocked=bool(merge_blocked),
            source='pit',
            chart_status=gate_decision,
        )
        if blocked:
            dec['settlement'] = 'pending_operator'
            dec['paid'] = False
        epoch_dir = self.epochs_root / str(int(epoch))
        decision_path = epoch_dir / 'DECISION.json'
        if decision_path.exists():
            existing = _rj(decision_path)
            if isinstance(existing, Mapping) and _same_decision(existing, dec):
                return decision_path
            if isinstance(existing, Mapping):
                raise GateWriteError('conflicting DECISION.json for epoch %d' % int(epoch))
        _wj(epoch_dir / 'EVALUATED.private.json', dict(evaluated=dec['evaluated']), 0o600)
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
        # Blocked / red / no_decision never close the heldout round as GREEN.
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

    def write_no_sample(
        self,
        *,
        epoch: int,
        head: str,
        candidate: str,
        why: str,
        parallel_eval_meta: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        self._refuse_shadow()
        now = time.time()
        pe_meta = dict(parallel_eval_meta or {})
        pe_meta.setdefault('settlement', 'pending_operator')
        pe_meta.setdefault('terminal', 'no_sample')
        dec = dict(
            version=GATE_VERSION,
            epoch=int(epoch),
            head=head or '',
            candidate=candidate or '',
            mode='live',
            decided_at=now,
            decided_at_cr=_cr(now),
            decision='no_sample',
            evaluated=dict(z=None),
            reasons=['no_sample', (why or '')[:200]],
            heldout={},
            parallel_eval=pe_meta,
            chart_status='no_sample',
            settlement='pending_operator',
            paid=False,
            source='pit',
        )
        epoch_dir = self.epochs_root / str(int(epoch))
        decision_path = epoch_dir / 'DECISION.json'
        _wj(decision_path, dec, 0o600)
        # Intentionally no turn_add / close — no rounds turn credit.
        return decision_path
