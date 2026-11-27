"""PIT live cutover / revert switch (SPEC §7 step 6–7).

When ``parallel_eval.pit`` is ``{enabled:true, mode:live, cutover:true}``:

* trainer-pod path stops dequeuing (kept 24 h as hot fallback);
* warm pool is off;
* standing PE eval lanes are off;
* PIT owns the item queue.

``revert:true`` (with cutover false / enabled false) restores the pre-cutover
paths: trainer dequeue, warm pool and standing lanes may run again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .config import PitConfig, load_pit_config


@dataclass(frozen=True)
class CutoverPlan:
    pit_enabled: bool
    mode: str
    cutover: bool
    revert: bool
    trainer_dequeue: bool
    warm_pool: bool
    standing_lanes: bool
    trainer_fallback_hours: int
    item_box_cap: int
    notes: tuple

    def as_dict(self) -> dict:
        return {
            'pit_enabled': self.pit_enabled,
            'mode': self.mode,
            'cutover': self.cutover,
            'revert': self.revert,
            'trainer_dequeue': self.trainer_dequeue,
            'warm_pool': self.warm_pool,
            'standing_lanes': self.standing_lanes,
            'trainer_fallback_hours': self.trainer_fallback_hours,
            'item_box_cap': self.item_box_cap,
            'notes': list(self.notes),
        }


def plan_cutover(cfg: PitConfig) -> CutoverPlan:
    if cfg.revert or not cfg.enabled or cfg.mode != 'live' or not cfg.cutover:
        # Revert / pre-cutover: legacy paths available.
        return CutoverPlan(
            pit_enabled=bool(cfg.enabled and not cfg.revert),
            mode=cfg.mode,
            cutover=False,
            revert=bool(cfg.revert),
            trainer_dequeue=True,
            warm_pool=True,
            standing_lanes=True,
            trainer_fallback_hours=int(cfg.trainer_fallback_hours),
            item_box_cap=int(cfg.max_boxes),
            notes=('revert' if cfg.revert else 'cutover-inactive',),
        )
    notes = [
        'pit-live-cutover',
        'trainer-dequeue-off',
        'warm-pool-off',
        'standing-lanes-off',
        'trainer-fallback-%dh' % int(cfg.trainer_fallback_hours),
    ]
    return CutoverPlan(
        pit_enabled=True,
        mode='live',
        cutover=True,
        revert=False,
        trainer_dequeue=False if cfg.disable_trainer_dequeue else True,
        warm_pool=False if cfg.disable_warm_pool else True,
        standing_lanes=False if cfg.disable_standing_lanes else True,
        trainer_fallback_hours=int(cfg.trainer_fallback_hours),
        item_box_cap=int(cfg.max_boxes),
        notes=tuple(notes),
    )


def plan_from_flag(parallel_eval: Optional[Mapping[str, Any]] = None) -> CutoverPlan:
    return plan_cutover(load_pit_config(parallel_eval))


def pe_dispatch_should_noop(cfg: PitConfig) -> bool:
    """pe_live.dispatch_tick must no-op whenever PIT owns the queue."""
    if cfg.revert:
        return False
    return bool(cfg.enabled)
