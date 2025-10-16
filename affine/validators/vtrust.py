from __future__ import annotations

from typing import Dict, Mapping, Tuple

from ..core.wilson import wilson_interval


def compute_vtrust(stats: Mapping[str, Tuple[int, int]], *, confidence: float = 0.95) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for validator, (wins, total) in stats.items():
        if total == 0:
            scores[validator] = 0.0
            continue
        lower, _ = wilson_interval(wins, total, confidence)
        scores[validator] = lower
    return scores


__all__ = ["compute_vtrust"]
