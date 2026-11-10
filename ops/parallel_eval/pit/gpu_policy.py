"""Shared GPU allow-list + B300-first-then-cheapest pick (pure; no Lium I/O)."""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

from . import ALLOWED_GPUS, GPU_ORDER, PRICE_CEILING_USD_H, REQUIRED_GPU_COUNT

Offer = Tuple[str, Any, float]  # (gpu_class, row, price_usd_h)


def pick_best_offer(
    cands: Iterable[Offer],
    order: Optional[Sequence[str]] = None,
    ceiling: Optional[float] = None,
) -> Tuple[Optional[str], Any]:
    """B300-first-then-cheapest (order[0]-first-then-cheapest).

    Prefer the cheapest offer of ``order[0]`` under the optional ceiling;
    otherwise the global cheapest under the ceiling. Caller must already
    require a full 8-GPU pack.
    """
    order_t = tuple(order) if order else GPU_ORDER
    rows = []
    for g, r, p in cands:
        try:
            price = float(p)
        except (TypeError, ValueError):
            continue
        if ceiling is not None and price > float(ceiling):
            continue
        if g not in ALLOWED_GPUS:
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
    'PRICE_CEILING_USD_H',
    'REQUIRED_GPU_COUNT',
    'pick_best_offer',
]
