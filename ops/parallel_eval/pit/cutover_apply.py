"""PIT-owned cutover head-apply driver.

Under cutover the legacy PE controller (``pe_live.tick``) does not run, so
``accept_pit_decision`` alone leaves GREENs in ``GREENS.json`` as ``pending``
forever. This module drives:

  pending → APPLY ready → (lifecycle offers/consumes) → record_applied
  → hash-sync

Idempotent and crash-safe. Refusal of incomplete fp32 masters lives inside
``pe_apply.next_green`` / ``pe_live.apply_tick`` / ``pe_apply.record_applied``
(no monkeypatching). ``head_advanced`` is true only after the candidate
appears on ``HEAD-CHAIN.json``.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional


class CutoverApplyError(RuntimeError):
    pass


def _pe_state() -> Path:
    env = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
    if not env:
        raise CutoverApplyError('PE_STATE is unset (no host-path default)')
    return Path(env)


def _ensure_live_on_path() -> None:
    live = Path(__file__).resolve().parents[1] / 'live'
    live_s = str(live)
    if live_s not in sys.path:
        sys.path.insert(0, live_s)


def _load_pe_live():
    _ensure_live_on_path()
    for name in ('pe_live', 'ops.parallel_eval.live.pe_live'):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise CutoverApplyError('pe_live module not importable')


def _load_pe_apply():
    _ensure_live_on_path()
    for name in ('pe_apply', 'ops.parallel_eval.live.pe_apply'):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise CutoverApplyError('pe_apply module not importable')


def normalize_pe_modules(
    *,
    pe_apply: Any = None,
    pe_live: Any = None,
) -> tuple[Any, Any]:
    """Return the single shared (pe_apply, pe_live) pair pe_live.apply_tick uses.

    Fail closed if the caller's pe_apply is a different module object than
    ``pe_live.PA``. Registers both import names onto that object.
    """
    PL = pe_live or _load_pe_live()
    live_pa = getattr(PL, 'PA', None)
    if live_pa is None:
        raise CutoverApplyError('pe_live.PA missing (fail closed)')
    if pe_apply is not None and pe_apply is not live_pa:
        raise CutoverApplyError(
            'pe_apply module identity mismatch with pe_live.PA '
            '(bare pe_apply vs ops.parallel_eval.live.pe_apply) — fail closed'
        )
    PA = live_pa
    sys.modules['pe_apply'] = PA
    sys.modules['ops.parallel_eval.live.pe_apply'] = PA
    if getattr(PL, 'PA', None) is not PA:
        raise CutoverApplyError('pe_live.PA is not the shared pe_apply module')
    return PA, PL


def _fp32_master_index_dir() -> Optional[Path]:
    for key in ('PIT_FP32_MASTER_INDEX', 'AFFINE_FP32_MASTER_INDEX'):
        v = os.environ.get(key)
        if v:
            return Path(v)
    return None


def master_index_is_complete(candidate_id: str, *, index_dir: Optional[Path] = None) -> bool:
    """Delegate to pe_apply when loaded; otherwise evaluate the index directly."""
    try:
        PA = _load_pe_apply()
        return bool(PA.master_index_is_complete(candidate_id, index_dir=index_dir))
    except CutoverApplyError:
        pass
    # Hermetic fallback for unit tests that never import live pe_*.
    if not isinstance(candidate_id, str) or len(candidate_id) != 64:
        return False
    root = Path(index_dir) if index_dir is not None else _fp32_master_index_dir()
    if root is None:
        return False
    path = Path(root) / ('%s.json' % candidate_id)
    if not path.is_file():
        return False
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return False
    if not isinstance(doc, Mapping):
        return False
    digest = str(doc.get('master_digest') or '')
    if len(digest) != 64:
        return False
    files = doc.get('files')
    if not isinstance(files, Mapping) or not files:
        return False
    shards = []
    for name, meta in files.items():
        n = str(name)
        if n in ('master_digest.txt', 'MASTER.json', 'FP32-MASTER.json'):
            continue
        if n.endswith('.safetensors') or n.endswith('.bin'):
            if isinstance(meta, Mapping) and meta.get('sha256'):
                shards.append(n)
    return bool(shards)


def head_chain_has_candidate(candidate_id: str, *, pe_apply: Any = None) -> bool:
    """True when HEAD-CHAIN.json lists candidate. Reads PE_STATE directly (no pe_live import)."""
    chain: Mapping[str, Any] = {}
    if pe_apply is not None:
        chain = pe_apply.chain() or {}
    else:
        try:
            path = _pe_state() / 'HEAD-CHAIN.json'
            if path.is_file():
                doc = json.loads(path.read_text())
                if isinstance(doc, Mapping):
                    chain = doc
        except Exception:
            chain = {}
    for row in chain.get('applied') or []:
        if isinstance(row, Mapping) and row.get('candidate') == candidate_id:
            return True
        if isinstance(row, Mapping) and row.get('new_head') == candidate_id:
            return True
    return False


def refuse_incomplete_pending_greens(
    *,
    pe_apply: Any = None,
    index_dir: Optional[Path] = None,
    alert_fn: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> list[str]:
    """Mark+alert incomplete pending GREENs once via pe_apply (no re-mark every tick)."""
    PA = pe_apply or normalize_pe_modules()[0]
    return list(PA.block_incomplete_pending_greens(alert_fn=alert_fn, index_dir=index_dir) or [])


def drive_cutover_apply(
    *,
    pe_apply: Any = None,
    pe_live: Any = None,
    index_dir: Optional[Path] = None,
    alert_fn: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run apply_tick + reconcile_tick under cutover. Idempotent. No monkeypatching.

    Incomplete-master refusal is inside pe_apply.next_green / pe_live.apply_tick /
    pe_apply.record_applied. This driver only marks+alerts once, then calls the
    real pe_live ticks on the shared module object.
    """
    PA, PL = normalize_pe_modules(pe_apply=pe_apply, pe_live=pe_live)
    refused = refuse_incomplete_pending_greens(
        pe_apply=PA, index_dir=index_dir, alert_fn=alert_fn,
    )
    if hasattr(PL, '_sync_life_paths'):
        PL._sync_life_paths()
    apply_out = PL.apply_tick()
    reconcile_out = PL.reconcile_tick()
    recorded = list(reconcile_out or [])
    return {
        'ok': True,
        'apply': apply_out,
        'reconcile': reconcile_out,
        'recorded_epochs': recorded,
        'refused_incomplete_master': refused,
        'pe_apply_id': id(PA),
        'pe_live_pa_id': id(getattr(PL, 'PA', None)),
    }


def confirm_head_advanced(candidate_id: str, *, pe_apply: Any = None) -> bool:
    """True only after HEAD-CHAIN append for this candidate is durable."""
    return head_chain_has_candidate(candidate_id, pe_apply=pe_apply)


def cutover_apply_alert(doc: Mapping[str, Any]) -> Optional[Path]:
    """run_host alert sink: write GREEN-APPLY-ALERT.json once per call site decision.

    De-dupe of GREENS.json re-marks is owned by pe_apply.mark_apply_blocked_incomplete;
    this only materializes the durable alert file for operators.
    """
    from .durable import pit_state_root
    root = pit_state_root()
    root.mkdir(parents=True, exist_ok=True)
    alert = root / 'GREEN-APPLY-ALERT.json'
    payload = dict(doc or {})
    payload.setdefault('at_unix', time.time())
    payload.setdefault('ok', False)
    tmp = alert.with_name('.%s.%d.tmp' % (alert.name, os.getpid()))
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + '\n')
    os.chmod(tmp, 0o600)
    os.replace(tmp, alert)
    print(
        json.dumps({
            'component': 'pit-cutover-apply',
            'level': 'error',
            'event': payload.get('event') or 'green_apply_blocked_incomplete_master',
            'candidate': str(payload.get('candidate') or '')[:16],
            'epoch': payload.get('epoch'),
            'alert': str(alert),
        }, default=str),
        file=sys.stderr,
        flush=True,
    )
    return alert
