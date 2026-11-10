"""Shared GPU allow-list + B300-first-then-cheapest pick (pure; no Lium I/O)."""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

from . import (
    ALLOW_SHARDED_GPUS_DEFAULT,
    ALLOWED_GPUS,
    GPU_ORDER,
    PRICE_CEILING_USD_H,
    REQUIRED_GPU_COUNT,
    SHARDED_GPUS,
)

Offer = Tuple[str, Any, float]  # (gpu_class, row, price_usd_h)


def effective_gpu_order(
    order: Optional[Sequence[str]] = None,
    *,
    allow_sharded_gpus: bool = ALLOW_SHARDED_GPUS_DEFAULT,
) -> tuple:
    """Apply the PROMOTION-GATE: drop RTX5090/H100 unless allow_sharded_gpus."""
    order_t = tuple(order) if order else GPU_ORDER
    if allow_sharded_gpus:
        return order_t
    blocked = set(SHARDED_GPUS)
    out = tuple(g for g in order_t if g not in blocked)
    if not out:
        raise ValueError('effective gpu_order empty after sharded-GPU gate')
    return out


def pick_best_offer(
    cands: Iterable[Offer],
    order: Optional[Sequence[str]] = None,
    ceiling: Optional[float] = None,
    *,
    allow_sharded_gpus: bool = ALLOW_SHARDED_GPUS_DEFAULT,
) -> Tuple[Optional[str], Any]:
    """B300-first-then-cheapest (order[0]-first-then-cheapest).

    Prefer the cheapest offer of ``order[0]`` under the optional ceiling;
    otherwise the global cheapest under the ceiling. Caller must already
    require a full 8-GPU pack. ``allow_sharded_gpus`` gates RTX5090/H100.
    """
    order_t = effective_gpu_order(order, allow_sharded_gpus=allow_sharded_gpus)
    allowed = set(order_t)
    rows = []
    for g, r, p in cands:
        try:
            price = float(p)
        except (TypeError, ValueError):
            continue
        if ceiling is not None and price > float(ceiling):
            continue
        if g not in ALLOWED_GPUS or g not in allowed:
            continue
        rows.append((g, r, price))
    if not rows:
        return None, None
    preferred = order_t[0] if order_t else None
    if preferred:
        pref = [(g, r, p) for g, r, p in rows if g == preferred]
        if pref:
            pref.sort(key=lambda x: x[2])
            return pref[0][0], pref[0][1]
    rows.sort(key=lambda x: x[2])
    return rows[0][0], rows[0][1]


__all__ = [
    'ALLOWED_GPUS',
    'GPU_ORDER',
    'SHARDED_GPUS',
    'ALLOW_SHARDED_GPUS_DEFAULT',
    'PRICE_CEILING_USD_H',
    'REQUIRED_GPU_COUNT',
    'effective_gpu_order',
    'pick_best_offer',
]
