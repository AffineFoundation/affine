"""Deadlines, retry spacing and cost. All of it is rebuilt from the journal.

Nothing here decides an outcome. The timers only say when an effect may
report "expired" or "gone", and how long to wait before a retry. The core
then checks that report against the fence and the phase.

Lease deadline of a running box
-------------------------------
* hard deadline: ``box_bound`` time + ``lease_ttl_s``. Journal time only, so
  a restart never extends it.
* heartbeat deadline: last heartbeat progress + ``heartbeat_grace_s``. The
  progress time is in memory. After a restart every running lease gets one
  fresh grace period, because the controller was blind while it was down.

A lease is reported as expired only if the upload was seen as not complete
at a time after the deadline.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional, Tuple


class SystemClock:
    """Wall time at start plus monotonic time since then.

    A step of the wall clock (NTP, an operator) cannot expire every lease at
    once, because the time inside one process follows the monotonic clock.
    """

    def __init__(self) -> None:
        self._wall = time.time()
        self._mono = time.monotonic()

    def now(self) -> float:
        return self._wall + (time.monotonic() - self._mono)


@dataclass(frozen=True)
class TimerConfig:
    lease_ttl_s: float
    heartbeat_grace_s: float
    rent_backoff_s: float
    renew_every_s: float
    absent_polls: int

    def __post_init__(self) -> None:
        if min(self.lease_ttl_s, self.heartbeat_grace_s) <= 0 or self.absent_polls < 1:
            raise ValueError('timer settings out of range')


class Timers:
    """Thread-safe. ``fold`` is called for every journal record, in order."""

    def __init__(self, cfg: TimerConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._started_at = 0.0
        self._bound_at: Dict[int, float] = {}
        self._progress: Dict[int, Tuple[int, float]] = {}
        self._renewed_at: Dict[int, float] = {}
        self._absent: Dict[Hashable, int] = {}
        self._not_before: Dict[Hashable, float] = {}
        self._fence_box: Dict[int, str] = {}
        # box_id -> (price per hour, start time, gpu). Price is None for a
        # box that was never bound to a lease (an orphan or a late rental).
        self._open: Dict[str, Tuple[Optional[float], float, str]] = {}
        self._spent = 0.0
        self._boxes_total = 0
        self._last_at = 0.0

    # -- journal fold --------------------------------------------------------

    def fold(self, kind: str, body: Mapping[str, Any], at: float) -> None:
        with self._lock:
            self._last_at = max(self._last_at, at)
            if kind == 'controller_started':
                self._started_at = at
                self._absent.clear()
            elif kind == 'box_bound':
                fence = body['fence']
                self._bound_at[fence] = at
                self._fence_box[fence] = body['box_id']
                self._open[body['box_id']] = (float(body.get('price') or 0.0), at, body.get('gpu') or '')
                self._boxes_total += 1
            elif kind == 'lease_renewed':
                fence = body['fence']
                known = self._progress.get(fence)
                if known is None or body['hb_seq'] > known[0]:
                    self._progress[fence] = (body['hb_seq'], at)
                self._renewed_at[fence] = at
            elif kind in ('attempt_failed', 'decided'):
                self._drop_fence(body['fence'])
            elif kind == 'rent_try':
                if body.get('backoff'):
                    key = ('rent', body['fence'], body['try'])
                    self._not_before[key] = at + self.cfg.rent_backoff_s
            elif kind == 'box_doomed':
                if body['box_id'] not in self._open:
                    self._open[body['box_id']] = (None, at, '')
                    self._boxes_total += 1
            elif kind == 'box_gone':
                opened = self._open.pop(body['box_id'], None)
                if opened is not None and opened[0] is not None:
                    self._spent += opened[0] * max(0.0, at - opened[1]) / 3600.0

    def _drop_fence(self, fence: int) -> None:
        self._bound_at.pop(fence, None)
        self._progress.pop(fence, None)
        self._renewed_at.pop(fence, None)
        self._absent.pop(('poll', fence), None)
        self._fence_box.pop(fence, None)

    @property
    def last_at(self) -> float:
        with self._lock:
            return self._last_at

    # -- lease deadlines -----------------------------------------------------

    def deadline(self, fence: int) -> Optional[float]:
        """The time after which the lease of this fence may be called expired."""
        with self._lock:
            bound = self._bound_at.get(fence)
            if bound is None:
                return None
            progress = self._progress.get(fence)
            alive_at = max(bound, self._started_at, progress[1] if progress else 0.0)
            return min(bound + self.cfg.lease_ttl_s, alive_at + self.cfg.heartbeat_grace_s)

    def heartbeat(self, fence: int, seq: int, now: float) -> Optional[float]:
        """Record a heartbeat. Returns the new deadline to put in the journal
        when a ``lease_renewed`` record is due, else None."""
        with self._lock:
            if fence not in self._bound_at:
                return None
            known = self._progress.get(fence)
            if known is not None and seq <= known[0]:
                return None
            self._progress[fence] = (seq, now)
            last = self._renewed_at.get(fence)
            if last is not None and now - last < self.cfg.renew_every_s:
                return None
            self._renewed_at[fence] = now
            bound = self._bound_at[fence]
            return min(bound + self.cfg.lease_ttl_s, now + self.cfg.heartbeat_grace_s)

    # -- presence and retry spacing -----------------------------------------

    def presence(self, key: Hashable, present: bool) -> bool:
        """Count listings in a row that miss a box. True when it is gone."""
        with self._lock:
            if present:
                self._absent.pop(key, None)
                return False
            count = self._absent.get(key, 0) + 1
            self._absent[key] = count
            return count >= self.cfg.absent_polls

    def due(self, key: Hashable, now: float) -> bool:
        with self._lock:
            return now >= self._not_before.get(key, 0.0)

    def defer(self, key: Hashable, until: float) -> None:
        with self._lock:
            self._not_before[key] = until

    def forget(self, keep: frozenset) -> None:
        """Drop retry state of effects that are not planned any more."""
        with self._lock:
            for key in [k for k in self._not_before if k not in keep]:
                del self._not_before[key]

    # -- cost ----------------------------------------------------------------

    def cost(self, now: float) -> Dict[str, Any]:
        """Money spent so far, from journal times and rental prices.

        This is an estimate for the operator. The provider's bill is the
        truth. A box without a known price is counted but not priced.
        """
        with self._lock:
            running = 0.0
            rate = 0.0
            unpriced = 0
            for price, since, _gpu in self._open.values():
                if price is None:
                    unpriced += 1
                    continue
                rate += price
                running += price * max(0.0, now - since) / 3600.0
            return {
                'usd_spent_estimate': round(self._spent + running, 4),
                'usd_per_hour_now': round(rate, 4),
                'boxes_alive': len(self._open),
                'boxes_alive_unpriced': unpriced,
                'boxes_total': self._boxes_total,
            }
