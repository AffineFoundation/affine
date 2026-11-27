"""Hash-chained PIT decision log (BRIEF §1.7, DESIGN §3).

Append-only under an optional flock. Ordering key is host append (one clock,
one lock). Pod clocks are never used. Public slice omits draw indices.
"""
from __future__ import annotations

import fcntl
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..durable import atomic, digest, read

GENESIS_PREV = '0' * 64


@dataclass(frozen=True)
class DecisionEntry:
    seq: int
    prev_sha256: str
    entry_hash: str
    epoch: int
    uid: int
    hotkey_prefix: str
    binding_H: str
    binding_round_id: str
    candidate: str
    decision: str  # green_fresh | green_stale | red | no_decision | blocked_*
    z: Optional[float]
    gain: Optional[float]
    fresh: Optional[bool]
    staleness: Optional[int]
    merge_blocked: bool
    paid: bool
    decided_at: float
    shadow: bool = True
    source: str = 'pit'
    extra: Dict[str, Any] = field(default_factory=dict)

    def body_for_hash(self) -> dict:
        return {
            'seq': self.seq,
            'prev_sha256': self.prev_sha256,
            'epoch': self.epoch,
            'uid': self.uid,
            'hotkey_prefix': self.hotkey_prefix,
            'binding_H': self.binding_H,
            'binding_round_id': self.binding_round_id,
            'candidate': self.candidate,
            'decision': self.decision,
            'z': self.z,
            'gain': self.gain,
            'fresh': self.fresh,
            'staleness': self.staleness,
            'merge_blocked': self.merge_blocked,
            'paid': self.paid,
            'decided_at': self.decided_at,
            'shadow': self.shadow,
            'source': self.source,
            'extra': self.extra,
        }

    def as_dict(self) -> dict:
        d = self.body_for_hash()
        d['entry_hash'] = self.entry_hash
        return d

    def public_slice(self) -> dict:
        """Fields safe after round reveal. Never includes draw indices."""
        return {
            'seq': self.seq,
            'prev_sha256': self.prev_sha256,
            'entry_hash': self.entry_hash,
            'epoch': self.epoch,
            'uid': self.uid,
            'hotkey_prefix': self.hotkey_prefix,
            'binding_H': self.binding_H,
            'binding_round_id': self.binding_round_id,
            'candidate': self.candidate,
            'decision': self.decision,
            'z': self.z,
            'gain': self.gain,
            'fresh': self.fresh,
            'staleness': self.staleness,
            'merge_blocked': self.merge_blocked,
            'paid': self.paid,
            'decided_at': self.decided_at,
            'source': self.source,
        }


def _seal(**kwargs: Any) -> DecisionEntry:
    decision = kwargs['decision']
    if decision not in (
        'green_fresh', 'green_stale', 'red', 'no_decision',
        'blocked_no_candidate', 'blocked_incomplete_master', 'no_sample',
    ):
        raise ValueError('invalid decision')
    if decision == 'green_stale' and not kwargs.get('merge_blocked'):
        raise ValueError('green_stale must be merge_blocked')
    if decision == 'green_fresh' and kwargs.get('merge_blocked'):
        raise ValueError('green_fresh cannot be merge_blocked')
    if decision in (
        'blocked_no_candidate', 'blocked_incomplete_master', 'no_sample',
    ) and kwargs.get('paid'):
        raise ValueError('terminal settlement decisions must not be paid')
    prev = kwargs['prev_sha256']
    if len(prev) != 64 or any(c not in '0123456789abcdef' for c in prev):
        raise ValueError('prev_sha256 must be hex64')
    body = {k: kwargs[k] for k in (
        'seq', 'prev_sha256', 'epoch', 'uid', 'hotkey_prefix', 'binding_H',
        'binding_round_id', 'candidate', 'decision', 'z', 'gain', 'fresh',
        'staleness', 'merge_blocked', 'paid', 'decided_at', 'shadow', 'source', 'extra',
    )}
    body['extra'] = dict(body.get('extra') or {})
    return DecisionEntry(entry_hash=digest(body), **body)


class DecisionLog:
    def __init__(self, entries: Optional[Sequence[DecisionEntry]] = None, *, lock_path: Optional[Path] = None):
        self._entries: List[DecisionEntry] = list(entries or [])
        self._lock_path = Path(lock_path) if lock_path else None
        self._verify_chain(self._entries)

    @staticmethod
    def _verify_chain(entries: Sequence[DecisionEntry]) -> None:
        prev = GENESIS_PREV
        for i, e in enumerate(entries):
            if e.seq != i:
                raise ValueError('seq gap at %d' % i)
            if e.prev_sha256 != prev:
                raise ValueError('prev_sha256 break at seq %d' % i)
            sealed = _seal(**{**e.body_for_hash()})
            if sealed.entry_hash != e.entry_hash:
                raise ValueError('entry_hash mismatch at seq %d' % i)
            prev = e.entry_hash

    @contextmanager
    def _locked(self):
        if self._lock_path is None:
            yield
            return
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, 'a') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            yield

    @property
    def tip(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS_PREV

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[DecisionEntry]:
        return list(self._entries)

    def append(
        self,
        *,
        epoch: int,
        uid: int,
        hotkey_prefix: str,
        binding_H: str,
        binding_round_id: str,
        candidate: str,
        decision: str,
        z: Optional[float] = None,
        gain: Optional[float] = None,
        fresh: Optional[bool] = None,
        staleness: Optional[int] = None,
        merge_blocked: bool = False,
        paid: bool = False,
        decided_at: Optional[float] = None,
        shadow: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> DecisionEntry:
        with self._locked():
            # Green-family order: decided_at non-decreasing.
            at = float(decided_at if decided_at is not None else time.time())
            if decision in ('green_fresh', 'green_stale'):
                for prior in self._entries:
                    if prior.decision in ('green_fresh', 'green_stale') and at < prior.decided_at:
                        raise ValueError('green-time order violated')
            entry = _seal(
                seq=len(self._entries),
                prev_sha256=self.tip,
                epoch=epoch,
                uid=uid,
                hotkey_prefix=hotkey_prefix,
                binding_H=binding_H,
                binding_round_id=binding_round_id,
                candidate=candidate,
                decision=decision,
                z=z,
                gain=gain,
                fresh=fresh,
                staleness=staleness,
                merge_blocked=merge_blocked,
                paid=paid,
                decided_at=at,
                shadow=shadow,
                source='pit',
                extra=dict(extra or {}),
            )
            self._entries.append(entry)
            return entry

    def fresh_or_stale(self, binding_H: str, current_head: str) -> str:
        if binding_H == current_head:
            return 'fresh'
        return 'stale'

    def greens_in_order(self) -> List[DecisionEntry]:
        return [e for e in self._entries if e.decision in ('green_fresh', 'green_stale')]

    def public_slice(self) -> List[dict]:
        return [e.public_slice() for e in self._entries]

    def as_list(self) -> List[dict]:
        return [e.as_dict() for e in self._entries]

    def save(self, path: Path) -> None:
        atomic(path, {
            'version': 'pit-decision-log-v1',
            'tip': self.tip,
            'entries': self.as_list(),
        })

    @classmethod
    def load(cls, path: Path, *, lock_path: Optional[Path] = None) -> 'DecisionLog':
        doc = read(path)
        if doc.get('version') != 'pit-decision-log-v1':
            raise ValueError('unknown decision log version')
        entries = []
        for row in doc.get('entries') or []:
            entries.append(DecisionEntry(
                seq=row['seq'],
                prev_sha256=row['prev_sha256'],
                entry_hash=row['entry_hash'],
                epoch=row['epoch'],
                uid=row['uid'],
                hotkey_prefix=row['hotkey_prefix'],
                binding_H=row['binding_H'],
                binding_round_id=row['binding_round_id'],
                candidate=row['candidate'],
                decision=row['decision'],
                z=row.get('z'),
                gain=row.get('gain'),
                fresh=row.get('fresh'),
                staleness=row.get('staleness'),
                merge_blocked=bool(row['merge_blocked']),
                paid=bool(row['paid']),
                decided_at=float(row['decided_at']),
                shadow=bool(row.get('shadow', True)),
                source=row.get('source') or 'pit',
                extra=dict(row.get('extra') or {}),
            ))
        log = cls(entries, lock_path=lock_path)
        if log.tip != doc.get('tip'):
            raise ValueError('tip mismatch')
        return log
