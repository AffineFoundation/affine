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
    gate = os.environ.get('HG_GATE_STATE')
    if gate:
        return Path(gate) / 'epochs'
    raise RuntimeError(
        'PE_GATE_EP/HG_GATE_STATE unset (fail closed; no host-path defaults)'
    )


def _cr(t: Optional[float] = None) -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S CR', time.gmtime((t or time.time()) - 21600))


BLOCKED_DECISIONS = frozenset({
    'blocked_no_candidate',
    'blocked_incomplete_master',
})

# Terminal non-GREEN settlements: never last-10 payout / HEAD / bench.
# lost_infra stays paid for K1 credit but never last-10 / HEAD.
TERMINAL_SETTLEMENT_DECISIONS = frozenset({
    *BLOCKED_DECISIONS,
    'no_sample',
    'lost_infra',
})


def map_pit_decision(decision: Optional[str]) -> str:
    d = (decision or '').lower()
    if d in ('green_fresh', 'green_stale', 'green', 'green_waiting'):
        return 'green'
    if d == 'red':
        return 'red'
    if d in ('no_decision', ''):
        return 'no_decision'
    if d == 'lost_infra':
        return 'lost_infra'
    if d in TERMINAL_SETTLEMENT_DECISIONS:
        return d
    return d


def is_blocked_decision(decision: Optional[str]) -> bool:
    return map_pit_decision(decision) in BLOCKED_DECISIONS


def is_terminal_settlement(decision: Optional[str]) -> bool:
    return map_pit_decision(decision) in TERMINAL_SETTLEMENT_DECISIONS


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
    if (item.merge_blocked or getattr(item, 'stranded', False)) and 'merge_blocked' not in reasons:
        reasons.append('merge_blocked')
    if getattr(item, 'stranded', False) and 'stranded' not in reasons:
        reasons.append('stranded')
    if gate_decision in TERMINAL_SETTLEMENT_DECISIONS and gate_decision not in reasons:
        reasons.append(gate_decision)
    heldout = dict(scored.get('heldout') or {})
    if item.bound_round_id and 'round_id' not in heldout:
        heldout['round_id'] = item.bound_round_id
    terminal = gate_decision in TERMINAL_SETTLEMENT_DECISIONS
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
        red_credit='zero',  # last-10-GREEN: REDs earn nothing; terminal never pays
        batch_tasks=None,
        early_staged=False,
        delta_rc=None,
        source='pit',
        merge_blocked=bool(item.merge_blocked or getattr(item, 'stranded', False)),
        stranded=bool(getattr(item, 'stranded', False) or item.merge_blocked),
        fresh=False if terminal else item.fresh,
        paid=False if terminal else bool(item.paid),
        no_decision=(gate_decision == 'no_decision'),
        chart_status=(
            gate_decision if terminal
            else ('no_decision' if gate_decision == 'no_decision' else gate_decision)
        ),
        parallel_eval=dict(
            source='pit',
            item_id=item.item_id,
            training_head=item.bound_head,
            binding_H=item.bound_head,
            box_id=item.box_id,
            attempt=item.attempt,
            pit_decision=item.decision,
            merge_blocked=bool(item.merge_blocked or getattr(item, 'stranded', False)),
            stranded=bool(getattr(item, 'stranded', False) or item.merge_blocked),
            fresh=False if terminal else item.fresh,
            # Cross-hardware audit: record GPU class on every DECISION.
            train_gpu=(item.meta or {}).get('train_gpu') or scored.get('train_gpu'),
            eval_gpu=(item.meta or {}).get('eval_gpu') or scored.get('eval_gpu'),
            gpu=(item.meta or {}).get('train_gpu')
                or (item.meta or {}).get('eval_gpu')
                or scored.get('train_gpu')
                or scored.get('eval_gpu'),
        ),
    )
    if terminal:
        # Operator settlement only — never a GREEN payout / last-10 credit.
        doc['settlement'] = 'pending_operator'
        doc['parallel_eval']['settlement'] = 'pending_operator'
        doc['parallel_eval']['terminal'] = gate_decision
        if gate_decision in BLOCKED_DECISIONS:
            doc['parallel_eval']['blocked'] = gate_decision
        if gate_decision == 'no_sample':
            doc['parallel_eval']['no_sample_reason'] = (
                (item.meta or {}).get('no_sample_reason') or ''
            )[:400]
    if scored.get('early_stop') is True and gate_decision == 'red':
        # RED-only early stop of the probe: ``evaluated`` is the estimate of the tasks that were finished.
        doc['parallel_eval']['early_stop'] = True
        doc['parallel_eval']['early_stop_fraction'] = scored.get('early_stop_fraction')
        doc['parallel_eval']['early_stop_detail'] = dict(scored.get('early_stop_detail') or {})
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
