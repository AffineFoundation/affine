"""PIT rental seam (BRIEF §1.3 / §1.9).

Host RA ``pe_rent.py`` owns real Lium rental. This git port:

* defines the on-demand rent/release API PIT uses;
* counts ``affine-pit-*`` boxes toward the eval-node cap;
* refuses NEVER_NAMES / warm-pool semantics;
* loads the host module when present.

Hard rule: rent only when the controller has a bound item ready. No warm pool.
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Set

from . import GPU_ORDER, PREFIX


# Names the code must never rent or reclaim (DESIGN §7). Redacted host names
# are matched by prefix patterns only — no private hostnames embedded here.
NEVER_NAME_PREFIXES = (
    'affine-trainer-',
    'affine-probe-',
    'affine-evaluator-',
    'affine-bench-',
    'affine-prior-',
    'affine-warm-',  # pe_warm must never be claimed as PIT
)


class RentError(RuntimeError):
    pass


def is_pit_name(name: str) -> bool:
    return isinstance(name, str) and name.startswith(PREFIX)


def is_never_name(name: str) -> bool:
    if not isinstance(name, str):
        return True
    if is_pit_name(name):
        return False
    return any(name.startswith(p) for p in NEVER_NAME_PREFIXES)


def count_pit_boxes(listings: List[Mapping[str, Any]]) -> int:
    n = 0
    for row in listings:
        name = row.get('name') or row.get('pod_name') or ''
        if is_pit_name(name):
            n += 1
    return n


def eval_nodes_with_pit(
    pe_eval_count: int,
    pit_listings: List[Mapping[str, Any]],
) -> int:
    """PE eval nodes + PIT boxes share the eval-node cap (BRIEF §1.9)."""
    return int(pe_eval_count) + count_pit_boxes(pit_listings)


@dataclass
class RentalLedger:
    """In-memory view of PIT rentals. Empty queue ⇒ zero active rentals."""

    active: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # box_id -> meta
    history: List[Dict[str, Any]] = field(default_factory=list)

    def count(self) -> int:
        return len(self.active)

    def register(self, box_id: str, *, item_id: str, name: str, at: float) -> None:
        if box_id in self.active:
            raise RentError('box already registered')
        self.active[box_id] = dict(item_id=item_id, name=name, rented_at=at)
        self.history.append(dict(op='rent', box_id=box_id, item_id=item_id, name=name, at=at))

    def release(self, box_id: str, *, reason: str, at: float) -> None:
        meta = self.active.pop(box_id, None)
        self.history.append(dict(
            op='release', box_id=box_id, reason=reason, at=at,
            item_id=None if meta is None else meta.get('item_id'),
        ))


@dataclass
class RentService:
    """On-demand rent/release. Inject host pe_rent callables when available."""

    max_boxes: int = 2
    max_eval_nodes: int = 5
    gpu_order: tuple = GPU_ORDER
    ledger: RentalLedger = field(default_factory=RentalLedger)
    host_rent: Optional[Callable[..., Mapping[str, Any]]] = None
    host_release: Optional[Callable[..., Mapping[str, Any]]] = None
    list_pods: Optional[Callable[[], List[Mapping[str, Any]]]] = None
    pe_eval_count_fn: Optional[Callable[[], int]] = None
    now_fn: Callable[[], float] = time.time

    def active_count(self) -> int:
        return self.ledger.count()

    def can_rent(self, *, queued_bound_waiting: int) -> bool:
        """True only when there is demand and room under the cap.

        ``queued_bound_waiting`` = items that need a box (BOUND ready to rent).
        Empty demand ⇒ False even if cap is free (no warm pool).
        """
        if queued_bound_waiting <= 0:
            return False
        if self.ledger.count() >= self.max_boxes:
            return False
        pe = 0 if self.pe_eval_count_fn is None else int(self.pe_eval_count_fn())
        listings = [] if self.list_pods is None else list(self.list_pods())
        if eval_nodes_with_pit(pe, listings) >= self.max_eval_nodes and self.ledger.count() >= self.max_boxes:
            return False
        return True

    def rent(self, *, item_id: str, epoch: int, attempt: int) -> Dict[str, Any]:
        if self.ledger.count() >= self.max_boxes:
            raise RentError('box_cap reached')
        name = '%s%d-a%d' % (PREFIX, int(epoch), int(attempt))
        if is_never_name(name):
            raise RentError('refused NEVER_NAME')
        at = float(self.now_fn())
        if self.host_rent is not None:
            out = dict(self.host_rent(
                name=name, item_id=item_id, epoch=epoch, attempt=attempt,
                gpu_order=list(self.gpu_order), prefix=PREFIX, ttl_h=4,
            ))
            box_id = out.get('box_id') or out.get('id')
            if not box_id:
                raise RentError('host rent returned no box_id')
            pod_name = out.get('name') or name
            if is_never_name(pod_name):
                raise RentError('host rent returned NEVER_NAME')
        else:
            # Deterministic synthetic box for unit/shadow without host rental.
            box_id = 'pit-box-%s-%d' % (item_id[:12], attempt)
            pod_name = name
            out = dict(box_id=box_id, name=pod_name, synthetic=True, gpu=self.gpu_order[0])
        self.ledger.register(str(box_id), item_id=item_id, name=pod_name, at=at)
        out.update(box_id=str(box_id), name=pod_name, rented_at=at)
        return out

    def release(self, box_id: str, *, reason: str = 'done') -> Dict[str, Any]:
        at = float(self.now_fn())
        if self.host_release is not None:
            try:
                self.host_release(box_id=box_id, reason=reason)
            except Exception as exc:
                # Still drop from ledger so we do not keep a zombie rental slot;
                # controller retries release confirmation separately.
                self.ledger.release(box_id, reason='release-error:%s' % str(exc)[:80], at=at)
                raise
        self.ledger.release(box_id, reason=reason, at=at)
        return dict(ok=True, box_id=box_id, reason=reason, released_at=at)


def load_host_pe_rent(module_path: Optional[str] = None):
    for name in ([module_path] if module_path else []) + ['pe_rent']:
        if not name:
            continue
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def bind_host_rent_service(
    service: RentService,
    *,
    module_path: Optional[str] = None,
) -> RentService:
    mod = load_host_pe_rent(module_path)
    if mod is None:
        return service
    service.host_rent = getattr(mod, 'rent', None) or getattr(mod, 'rent_box', None)
    service.host_release = getattr(mod, 'release', None) or getattr(mod, 'release_box', None)
    service.list_pods = getattr(mod, 'list_pods', None) or getattr(mod, 'ps', None)
    return service
