"""Hash-chained PIT decision log.

Each entry digests the previous entry's hash plus its own body. The log is the
durable audit of GREEN/RED/MERGE_BLOCKED outcomes in green-time order.

Shadow mode may append to a local staging log; it must never be confused with
live DECISION.json / HEAD-CHAIN writes (see shadow.ShadowGuard).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..durable import atomic, digest, read


GENESIS_PREV = '0' * 64


@dataclass(frozen=True)
class DecisionEntry:
    """One sealed decision-log row."""

    seq: int
    item_id: str
    registration_index: int
    decision: str  # green | red | merge_blocked
    bound_head: str
    candidate_id: Optional[str]
    z: Optional[float]
    paid: bool
    merge_blocked: bool
    green_at_mono: Optional[float]
    prev_hash: str
    entry_hash: str
    at_unix: float
    shadow: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def body_for_hash(self) -> dict:
        return {
            'seq': self.seq,
            'item_id': self.item_id,
            'registration_index': self.registration_index,
            'decision': self.decision,
            'bound_head': self.bound_head,
            'candidate_id': self.candidate_id,
            'z': self.z,
            'paid': self.paid,
            'merge_blocked': self.merge_blocked,
            'green_at_mono': self.green_at_mono,
            'prev_hash': self.prev_hash,
            'at_unix': self.at_unix,
            'shadow': self.shadow,
            'extra': self.extra,
        }

    def as_dict(self) -> dict:
        d = self.body_for_hash()
        d['entry_hash'] = self.entry_hash
        return d


def _seal(
    *,
    seq: int,
    item_id: str,
    registration_index: int,
    decision: str,
    bound_head: str,
    candidate_id: Optional[str],
    z: Optional[float],
    paid: bool,
    merge_blocked: bool,
    green_at_mono: Optional[float],
    prev_hash: str,
    at_unix: float,
    shadow: bool,
    extra: Optional[Dict[str, Any]],
) -> DecisionEntry:
    if decision not in ('green', 'red', 'merge_blocked'):
        raise ValueError('decision must be green|red|merge_blocked')
    if decision == 'merge_blocked' and not (paid and merge_blocked):
        raise ValueError('merge_blocked rows must be paid and merge_blocked=True')
    if decision == 'green' and merge_blocked:
        raise ValueError('decision=green cannot be merge_blocked; use merge_blocked')
    if len(prev_hash) != 64 or any(c not in '0123456789abcdef' for c in prev_hash):
        raise ValueError('prev_hash must be lowercase hex64')
    body = {
        'seq': seq,
        'item_id': item_id,
        'registration_index': registration_index,
        'decision': decision,
        'bound_head': bound_head,
        'candidate_id': candidate_id,
        'z': z,
        'paid': paid,
        'merge_blocked': merge_blocked,
        'green_at_mono': green_at_mono,
        'prev_hash': prev_hash,
        'at_unix': at_unix,
        'shadow': shadow,
        'extra': dict(extra or {}),
    }
    entry_hash = digest(body)
    return DecisionEntry(entry_hash=entry_hash, **body)


class DecisionLog:
    """Append-only hash chain of PIT decisions."""

    def __init__(self, entries: Optional[Sequence[DecisionEntry]] = None):
        self._entries: List[DecisionEntry] = list(entries or [])
        self._verify_chain(self._entries)

    @staticmethod
    def _verify_chain(entries: Sequence[DecisionEntry]) -> None:
        prev = GENESIS_PREV
        for i, e in enumerate(entries):
            if e.seq != i:
                raise ValueError('seq gap at %d' % i)
            if e.prev_hash != prev:
                raise ValueError('prev_hash break at seq %d' % i)
            sealed = _seal(
                seq=e.seq,
                item_id=e.item_id,
                registration_index=e.registration_index,
                decision=e.decision,
                bound_head=e.bound_head,
                candidate_id=e.candidate_id,
                z=e.z,
                paid=e.paid,
                merge_blocked=e.merge_blocked,
                green_at_mono=e.green_at_mono,
                prev_hash=e.prev_hash,
                at_unix=e.at_unix,
                shadow=e.shadow,
                extra=e.extra,
            )
            if sealed.entry_hash != e.entry_hash:
                raise ValueError('entry_hash mismatch at seq %d' % i)
            prev = e.entry_hash

    @property
    def tip(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS_PREV

    @property
    def entries(self) -> TupleView:
        return TupleView(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def append(
        self,
        *,
        item_id: str,
        registration_index: int,
        decision: str,
        bound_head: str,
        candidate_id: Optional[str] = None,
        z: Optional[float] = None,
        paid: bool = False,
        merge_blocked: bool = False,
        green_at_mono: Optional[float] = None,
        shadow: bool = True,
        extra: Optional[Dict[str, Any]] = None,
        at_unix: Optional[float] = None,
    ) -> DecisionEntry:
        # Enforce green-time order among green-family rows.
        if decision in ('green', 'merge_blocked'):
            if green_at_mono is None:
                raise ValueError('green_at_mono required for green-family decisions')
            for prior in self._entries:
                if prior.decision in ('green', 'merge_blocked') and prior.green_at_mono is not None:
                    if float(green_at_mono) < float(prior.green_at_mono):
                        raise ValueError(
                            'green-time order violated: new %.6f < prior %.6f (seq %d)'
                            % (float(green_at_mono), float(prior.green_at_mono), prior.seq)
                        )
                    if (
                        float(green_at_mono) == float(prior.green_at_mono)
                        and registration_index < prior.registration_index
                    ):
                        raise ValueError('green-time tie-break order violated')
        entry = _seal(
            seq=len(self._entries),
            item_id=item_id,
            registration_index=registration_index,
            decision=decision,
            bound_head=bound_head,
            candidate_id=candidate_id,
            z=z,
            paid=paid,
            merge_blocked=merge_blocked,
            green_at_mono=green_at_mono,
            prev_hash=self.tip,
            at_unix=float(at_unix if at_unix is not None else time.time()),
            shadow=shadow,
            extra=extra,
        )
        self._entries.append(entry)
        return entry

    def greens_in_order(self) -> List[DecisionEntry]:
        return [e for e in self._entries if e.decision in ('green', 'merge_blocked')]

    def as_list(self) -> List[dict]:
        return [e.as_dict() for e in self._entries]

    def save(self, path: Path) -> None:
        atomic(path, {
            'version': 'pit-decision-log-v1',
            'tip': self.tip,
            'entries': self.as_list(),
        })

    @classmethod
    def load(cls, path: Path) -> 'DecisionLog':
        doc = read(path)
        if doc.get('version') != 'pit-decision-log-v1':
            raise ValueError('unknown decision log version')
        entries = []
        for row in doc.get('entries') or []:
            entries.append(DecisionEntry(
                seq=row['seq'],
                item_id=row['item_id'],
                registration_index=row['registration_index'],
                decision=row['decision'],
                bound_head=row['bound_head'],
                candidate_id=row.get('candidate_id'),
                z=row.get('z'),
                paid=bool(row['paid']),
                merge_blocked=bool(row['merge_blocked']),
                green_at_mono=row.get('green_at_mono'),
                prev_hash=row['prev_hash'],
                entry_hash=row['entry_hash'],
                at_unix=float(row['at_unix']),
                shadow=bool(row.get('shadow', True)),
                extra=dict(row.get('extra') or {}),
            ))
        log = cls(entries)
        if log.tip != doc.get('tip'):
            raise ValueError('tip mismatch')
        return log


class TupleView:
    """Read-only sequence facade over the entry list."""

    def __init__(self, entries: List[DecisionEntry]):
        self._entries = entries

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        return self._entries[idx]
