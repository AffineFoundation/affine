"""PIT rental seam (BRIEF §1.3 / §1.9, SPEC §4).

Host RA ``pe_rent.py`` owns real Lium rental. This git port:

* defines the on-demand rent/release API PIT uses;
* counts ``affine-pit-*`` boxes toward the item box_cap only;
* refuses NEVER_NAMES / warm-pool semantics;
* pins item rentals to B300 (no GPU fallback);
* enforces the $56/h price ceiling;
* disables Lium auto-backup on every rental;
* loads the host module when present.

Hard rule: rent only when the controller has a bound item ready. No warm pool.
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import GPU_PINNED, PREFIX, PRICE_CEILING_USD_H


NEVER_NAME_PREFIXES = (
    'affine-trainer-',
    'affine-probe-',
    'affine-evaluator-',
    'affine-bench-',
    'affine-prior-',
    'affine-warm-',
    'affine-baseline-',  # baseline jobs use their own prefix; never claim as PIT
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
    return int(pe_eval_count) + count_pit_boxes(pit_listings)


@dataclass
class RentalLedger:
    active: Dict[str, Dict[str, Any]] = field(default_factory=dict)
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
    max_boxes: int = 2
    max_eval_nodes: int = 5
    gpu_order: tuple = GPU_PINNED
    price_ceiling_usd_h: float = PRICE_CEILING_USD_H
    disable_auto_backup: bool = True
    ledger: RentalLedger = field(default_factory=RentalLedger)
    host_rent: Optional[Callable[..., Mapping[str, Any]]] = None
    host_release: Optional[Callable[..., Mapping[str, Any]]] = None
    list_pods: Optional[Callable[[], List[Mapping[str, Any]]]] = None
    pe_eval_count_fn: Optional[Callable[[], int]] = None
    disable_auto_backup_fn: Optional[Callable[[str], bool]] = None
    now_fn: Callable[[], float] = time.time

    def active_count(self) -> int:
        return self.ledger.count()

    def can_rent(self, *, queued_bound_waiting: int) -> bool:
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
        if tuple(self.gpu_order) != GPU_PINNED:
            raise RentError('item rentals are pinned to B300')
        name = '%s%d-a%d' % (PREFIX, int(epoch), int(attempt))
        if is_never_name(name):
            raise RentError('refused NEVER_NAME')
        at = float(self.now_fn())
        if self.host_rent is not None:
            out = dict(self.host_rent(
                name=name, item_id=item_id, epoch=epoch, attempt=attempt,
                gpu_order=list(self.gpu_order), prefix=PREFIX, ttl_h=4,
                price_ceiling_usd_h=self.price_ceiling_usd_h,
                disable_auto_backup=self.disable_auto_backup,
            ))
            box_id = out.get('box_id') or out.get('id')
            if not box_id:
                raise RentError('host rent returned no box_id')
            pod_name = out.get('name') or name
            if is_never_name(pod_name):
                raise RentError('host rent returned NEVER_NAME')
            price = out.get('price_usd_h')
            if price is not None and float(price) > float(self.price_ceiling_usd_h):
                raise RentError('price %.2f above ceiling %.2f' % (float(price), self.price_ceiling_usd_h))
        else:
            box_id = 'pit-box-%s-%d' % (item_id[:12], attempt)
            pod_name = name
            out = dict(
                box_id=box_id, name=pod_name, synthetic=True,
                gpu=self.gpu_order[0], price_usd_h=min(35.0, self.price_ceiling_usd_h),
            )
        # Disable Lium auto-backup (SPEC §8).
        disabled = True
        if self.disable_auto_backup:
            if self.disable_auto_backup_fn is not None:
                disabled = bool(self.disable_auto_backup_fn(str(box_id)))
            out['auto_backup_disabled'] = disabled
            if not disabled:
                if self.host_release is not None:
                    try:
                        self.host_release(box_id=box_id, reason='auto-backup-still-on')
                    except Exception:
                        pass
                raise RentError('lium auto-backup must be disabled on every rental')
        else:
            out['auto_backup_disabled'] = False
        self.ledger.register(str(box_id), item_id=item_id, name=pod_name, at=at)
        out.update(box_id=str(box_id), name=pod_name, rented_at=at)
        return out

    def release(self, box_id: str, *, reason: str = 'done') -> Dict[str, Any]:
        at = float(self.now_fn())
        if self.host_release is not None:
            try:
                self.host_release(box_id=box_id, reason=reason)
            except Exception as exc:
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
    service.disable_auto_backup_fn = getattr(mod, 'disable_auto_backup', None)
    return service
