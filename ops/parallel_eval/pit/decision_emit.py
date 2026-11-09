"""Emit per-epoch DECISION.json in the live gate / k1 / ff_launcher schema.

k1_service, gate_ledger, ff_launcher_ntp and pe_lifecycle all read
``promotion-gate/state/epochs/<e>/DECISION.json``. PIT must write the same
shape (decision green|red|no_decision, evaluated, heldout, parallel_eval, …)
with ``source=pit``.

Live writes are **deprecated** here: call ``emit_decision`` with a
``GateWriter`` (see ``gate_writer.py``). Never writes in shadow mode.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from .config import PitConfig
from .state import ItemRecord


GATE_VERSION = 'sn120-promotion-gate-v1'


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
    ) -> Path: ...


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
    gate_writer: GateWriter,
    config: PitConfig,
    epochs_root: Optional[Path] = None,
    decided_at: Optional[float] = None,
) -> Optional[Path]:
    """Write a live PIT decision via ``gate_writer``. Shadow → None (refused).

    Direct filesystem writes without a ``GateWriter`` are no longer supported.
    """
    if epochs_root is not None:
        warnings.warn(
            'emit_decision(epochs_root=...) is ignored; use gate_writer.epochs_root',
            DeprecationWarning,
            stacklevel=2,
        )
    if config.is_shadow() or not config.may_write_live_artifacts():
        return None

    heldout = dict(scored.get('heldout') or {})
    # GateWriter.write_decision validates the full heldout block (single writer).
    doc = build_decision_doc(
        item, scored=scored, mode=config.mode, decided_at=decided_at,
    )
    gate_decision = map_pit_decision(item.decision or scored.get('decision'))
    plan_seed = item.bound_plan_seed or str(scored.get('plan_seed') or '')
    if not plan_seed:
        raise ValueError('plan_seed required for live gate emit')
    return gate_writer.write_decision(
        epoch=int(item.epoch),
        head=item.bound_head or '',
        candidate=item.candidate_id or '',
        decision=gate_decision,
        evaluated=doc['evaluated'],
        reasons=doc['reasons'],
        heldout=heldout,
        plan_seed=plan_seed,
        parallel_eval_meta=doc.get('parallel_eval') or {},
        no_decision=bool(doc.get('no_decision')),
        merge_blocked=bool(item.merge_blocked),
    )
