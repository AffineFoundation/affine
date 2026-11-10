"""PIT-owned cutover head-apply driver.

Under cutover the legacy PE controller (``pe_live.tick``) does not run, so
``accept_pit_decision`` alone leaves GREENs in ``GREENS.json`` as ``pending``
forever. This module drives:

  pending → APPLY ready → (lifecycle offers/consumes) → record_applied
  → hash-sync

Idempotent and crash-safe. Refuses to promote a GREEN whose published fp32
master index is incomplete (digest-only / no shards). ``head_advanced`` is
true only after the candidate appears on ``HEAD-CHAIN.json``.
"""
from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


class CutoverApplyError(RuntimeError):
    pass


def _pe_state() -> Path:
    env = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
    if not env:
        raise CutoverApplyError('PE_STATE is unset (no host-path default)')
    return Path(env)


def _load_pe_apply():
    for name in ('pe_apply', 'ops.parallel_eval.live.pe_apply'):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise CutoverApplyError('pe_apply module not importable')


def _load_pe_live():
    for name in ('pe_live', 'ops.parallel_eval.live.pe_live'):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise CutoverApplyError('pe_live module not importable')


def _fp32_master_index_dir() -> Optional[Path]:
    for key in ('PIT_FP32_MASTER_INDEX', 'AFFINE_FP32_MASTER_INDEX'):
        v = os.environ.get(key)
        if v:
            return Path(v)
    return None


def master_index_is_complete(candidate_id: str, *, index_dir: Optional[Path] = None) -> bool:
    """True when the published master index lists real weight shards (not digest-only)."""
    if not isinstance(candidate_id, str) or len(candidate_id) != 64:
        return False
    root = index_dir if index_dir is not None else _fp32_master_index_dir()
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
    # Digest-only publish wrote solely master_digest.txt — refuse that.
    weight = []
    for name, meta in files.items():
        n = str(name)
        if n in ('master_digest.txt', 'MASTER.json', 'FP32-MASTER.json'):
            continue
        if n.endswith('.safetensors') or n.endswith('.bin') or n == 'meta.json':
            if isinstance(meta, Mapping) and meta.get('sha256'):
                weight.append(n)
    shard_like = [n for n in weight if n.endswith('.safetensors') or n.endswith('.bin')]
    return bool(shard_like)


def head_chain_has_candidate(candidate_id: str, *, pe_apply: Any = None) -> bool:
    PA = pe_apply or _load_pe_apply()
    chain = PA.chain() or {}
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
    """Skip (do not APPLY-ready) pending GREENs whose master index is incomplete.

    Marks them with ``why`` but leaves ``state=pending`` so a later complete
    publish can proceed. Returns candidate ids that were refused this pass.
    """
    PA = pe_apply or _load_pe_apply()
    refused: list[str] = []
    for item in PA.greens():
        if item.get('state') != 'pending' or item.get('merge_blocked'):
            continue
        cand = str(item.get('candidate') or '')
        if not cand or master_index_is_complete(cand, index_dir=index_dir):
            continue
        refused.append(cand)
        try:
            PA.mark(
                cand,
                apply_blocked='incomplete-fp32-master',
                why='fp32 master index incomplete (need weight shards, not digest-only)',
                blocked_at=time.time(),
            )
        except Exception:
            # mark may require locked write; still alert
            pass
        if alert_fn is not None:
            alert_fn({
                'event': 'green_apply_blocked_incomplete_master',
                'candidate': cand,
                'epoch': item.get('epoch'),
                'source': item.get('source'),
            })
    return refused


def drive_cutover_apply(
    *,
    pe_apply: Any = None,
    pe_live: Any = None,
    index_dir: Optional[Path] = None,
    alert_fn: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run apply_tick + reconcile_tick under cutover. Idempotent.

    Before apply_tick, refuse incomplete-master pending GREENs so they never
    become APPLY ready. ``record_applied`` (inside reconcile) appends HEAD-CHAIN
    and runs the env-injected hash-sync hook.
    """
    PA = pe_apply or _load_pe_apply()
    PL = pe_live or _load_pe_live()
    refused = refuse_incomplete_pending_greens(
        pe_apply=PA, index_dir=index_dir, alert_fn=alert_fn,
    )
    # Temporarily strip incomplete pending from consideration by marking them
    # unrestorable-for-apply is too strong (would strand forever). Instead we
    # monkey-patch next_green via mark apply_blocked and filter in a local wrapper
    # if pe_live.apply_tick cannot see our flag. pe_live.apply_tick uses
    # PA.next_green() which only checks state==pending && !merge_blocked.
    # So we temporarily set merge_blocked for incomplete ones? That would strand
    # for K1 pay. Better: set state aside... Actually user said first-GREEN-wins /
    # stranded semantics must hold — incomplete is NOT stranded, just blocked.
    #
    # Patch: wrap PA.next_green for this call to skip incomplete masters.
    real_next = PA.next_green

    def _next_complete():
        for x in sorted(PA.greens(), key=lambda r: (r['green_at'], r['epoch'])):
            if x.get('state') != 'pending' or x.get('merge_blocked'):
                continue
            cand = str(x.get('candidate') or '')
            if cand and not master_index_is_complete(cand, index_dir=index_dir):
                continue
            return x
        return None

    PA.next_green = _next_complete  # type: ignore[method-assign]
    try:
        if hasattr(PL, '_sync_life_paths'):
            PL._sync_life_paths()
        apply_out = PL.apply_tick()
        reconcile_out = PL.reconcile_tick()
    finally:
        PA.next_green = real_next  # type: ignore[method-assign]
    recorded = list(reconcile_out or [])
    return {
        'ok': True,
        'apply': apply_out,
        'reconcile': reconcile_out,
        'recorded_epochs': recorded,
        'refused_incomplete_master': refused,
    }


def confirm_head_advanced(candidate_id: str, *, pe_apply: Any = None) -> bool:
    """True only after HEAD-CHAIN append for this candidate is durable."""
    return head_chain_has_candidate(candidate_id, pe_apply=pe_apply)
