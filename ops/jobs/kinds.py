"""Job kinds and concurrent caps (SPEC §4.3).

Owner ruling: baseline / bench / prior do **not** count toward the item box_cap
(5 live / 2 shadow).
"""
from __future__ import annotations

from enum import Enum
from typing import Mapping


class JobKind(str, Enum):
    ITEM = 'item'
    BASELINE = 'baseline'
    BENCH = 'bench'
    PRIOR = 'prior'


JOB_KINDS = frozenset(k.value for k in JobKind)

# Live concurrent caps by kind. Shadow item cap is 2 (see PitConfig).
LIVE_CAPS: Mapping[JobKind, int] = {
    JobKind.ITEM: 5,
    JobKind.BASELINE: 2,
    JobKind.BENCH: 1,
    JobKind.PRIOR: 1,
}

SHADOW_CAPS: Mapping[JobKind, int] = {
    JobKind.ITEM: 2,
    JobKind.BASELINE: 2,  # uses the live round
    JobKind.BENCH: 0,
    JobKind.PRIOR: 0,
}

# Name prefixes under affine-<kind>-…
NAME_PREFIX: Mapping[JobKind, str] = {
    JobKind.ITEM: 'affine-pit-',
    JobKind.BASELINE: 'affine-baseline-',
    JobKind.BENCH: 'affine-bench-job-',
    JobKind.PRIOR: 'affine-prior-job-',
}


def kind_cap(kind: JobKind, *, live: bool) -> int:
    table = LIVE_CAPS if live else SHADOW_CAPS
    return int(table[kind])


def counts_toward_item_cap(kind: JobKind) -> bool:
    """Only item (PIT) boxes consume the 5-box / 2-box item cap."""
    return kind is JobKind.ITEM
