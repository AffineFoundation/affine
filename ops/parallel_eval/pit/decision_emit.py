"""Emit per-epoch DECISION.json in the live gate / k1 / ff_launcher schema.

k1_service, gate_ledger, ff_launcher_ntp and pe_lifecycle all read
``promotion-gate/state/epochs/<e>/DECISION.json``. PIT must write the same
shape (decision green|red|no_decision, evaluated, heldout, parallel_eval, …)
with ``source=pit``. Never writes in shadow mode.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import PitConfig
from .state import ItemRecord


GATE_VERSION = 'sn120-promotion-gate-v1'


def gate_epochs_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get('PE_GATE_EP')
    if env:
        return Path(env)
    return Path(
        '/home/const/subnet120-rewrite/state/root-audits/'
        '20261026-promotion-gate/state/epochs'
    )


def _cr(t: Optional[float] = None) -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S CR', time.gmtime((t or time.time()) - 21600))


def _wj(path: Path, obj: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def map_pit_decision(decision: Optional[str]) -> str:
    d = (decision or '').lower()
    if d in ('green_fresh', 'green_stale', 'green'):
        return 'green'
    if d == 'red':
        return 'red'
    if d in ('no_decision', ''):
        return 'no_decision'
    return d


def build_decision_doc(
    item: ItemRecord,
    *,
    scored: Mapping[str, Any],
    mode: str = 'live',
    decided_at: Optional[float] = None,
) -> dict:
    now = float(decided_at if decided_at is not None else time.time())
    gate_decision = map_pit_decision(item.decision or scored.get('decision'))
    evaluated = dict(scored.get('evaluated') or {})
    if 'z' not in evaluated and item.z is not None:
        evaluated['z'] = item.z
    reasons = list(scored.get('reasons') or [])
    if item.merge_blocked and 'merge_blocked' not in reasons:
        reasons.append('merge_blocked')
    heldout = dict(scored.get('heldout') or {})
    if item.bound_round_id and 'round_id' not in heldout:
        heldout['round_id'] = item.bound_round_id
    doc = dict(
        version=GATE_VERSION,
        epoch=int(item.epoch),
        head=item.bound_head,
        candidate=item.candidate_id,
        mode=mode,
        decided_at=now,
        decided_at_cr=_cr(now),
        decision=gate_decision,
        evaluated=evaluated,
        reasons=reasons,
        heldout=heldout,
        probe=dict(scored.get('probe') or {}),
        timing=dict(scored.get('timing') or {}),
        commitment=scored.get('commitment'),
        z_bar=scored.get('z_bar'),
        red_credit='zero',  # last-10-GREEN: REDs earn nothing
        batch_tasks=None,
        early_staged=False,
        delta_rc=None,
        source='pit',
        merge_blocked=bool(item.merge_blocked),
        fresh=item.fresh,
        paid=bool(item.paid),
        no_decision=(gate_decision == 'no_decision'),
        chart_status=('no_decision' if gate_decision == 'no_decision' else gate_decision),
        parallel_eval=dict(
            source='pit',
            item_id=item.item_id,
            training_head=item.bound_head,
            binding_H=item.bound_head,
            box_id=item.box_id,
            attempt=item.attempt,
            pit_decision=item.decision,
            merge_blocked=bool(item.merge_blocked),
            fresh=item.fresh,
        ),
    )
    return doc


def emit_decision(
    item: ItemRecord,
    *,
    scored: Mapping[str, Any],
    config: PitConfig,
    epochs_root: Optional[Path] = None,
    decided_at: Optional[float] = None,
) -> Optional[Path]:
    """Write DECISION.json for a live PIT decision. Shadow → None (refused)."""
    if config.is_shadow() or not config.may_write_live_artifacts():
        return None
    doc = build_decision_doc(
        item, scored=scored, mode=config.mode, decided_at=decided_at,
    )
    root = gate_epochs_root(str(epochs_root) if epochs_root else None)
    path = root / str(int(item.epoch)) / 'DECISION.json'
    if path.exists():
        # Idempotent: refuse overwrite of a non-pit decision.
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}
        if existing.get('source') not in (None, 'pit') and existing.get('parallel_eval', {}).get('source') != 'pit':
            if existing.get('decision') and existing.get('source') != 'pit':
                raise RuntimeError('refusing to overwrite non-PIT DECISION.json for e%d' % item.epoch)
    _wj(path, doc, 0o600)
    return path
