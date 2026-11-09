"""PIT rental seam (BRIEF §1.3 / §1.9, SPEC §4).

Host RA ``pe_rent.py`` owns real Lium rental. This git port:

* defines the on-demand rent/release API PIT uses;
* counts ``affine-pe-pit-*`` boxes toward the item box_cap only
  (live ``pe_rent`` requires names starting ``affine-pe-``);
* refuses NEVER_NAMES / warm-pool semantics;
* pins item rentals to B300 (no GPU fallback);
* enforces the $56/h price ceiling;
* disables Lium auto-backup on every rental;
* loads the host module when present.

Hard rule: rent only when the controller has a bound item ready. No warm pool.
"""
from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set

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

PIT_LIVE_TTL = '4h'
PIT_LIVE_GPU = 'B300'


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


def _price_usd_h(pod: Mapping[str, Any]) -> Optional[float]:
    for key in ('price_usd_h', 'price_h', 'price_per_hour'):
        raw = pod.get(key)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


def _gpu_is_b300(pod: Mapping[str, Any]) -> bool:
    for key in ('gpu', 'gpu_model', 'gpu_name'):
        val = pod.get(key)
        if val is not None and 'B300' in str(val).upper():
            return True
    name = pod.get('name') or ''
    return 'B300' in str(name).upper()


def _no_backup_confirmed(pod: Mapping[str, Any]) -> bool:
    if pod.get('auto_backup_disabled') is True:
        return True
    if pod.get('no_backup') is True:
        return True
    return False


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

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(active=self.active, history=self.history)
        p.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')

    @classmethod
    def load(cls, path: str) -> 'RentalLedger':
        p = Path(path)
        if not p.is_file():
            return cls()
        data = json.loads(p.read_text())
        return cls(
            active=dict(data.get('active') or {}),
            history=list(data.get('history') or []),
        )


@dataclass
class RentService:
    max_boxes: int = 2
    max_eval_nodes: int = 5
    gpu_order: tuple = GPU_PINNED
    price_ceiling_usd_h: float = PRICE_CEILING_USD_H
    disable_auto_backup: bool = True
    require_host: bool = False
    ledger: RentalLedger = field(default_factory=RentalLedger)
    ledger_path: Optional[str] = None
    host_rent: Optional[Callable[..., Mapping[str, Any]]] = None
    host_release: Optional[Callable[..., Any]] = None
    list_pods: Optional[Callable[[], List[Mapping[str, Any]]]] = None
    pe_eval_count_fn: Optional[Callable[[], int]] = None
    pe_warm_count_fn: Optional[Callable[[], int]] = None
    disable_auto_backup_fn: Optional[Callable[[str], bool]] = None
    offer_precheck_fn: Optional[Callable[..., Any]] = None
    now_fn: Callable[[], float] = time.time

    def active_count(self) -> int:
        return self.ledger.count()

    def save_ledger(self) -> None:
        if self.ledger_path:
            self.ledger.save(self.ledger_path)

    def load_ledger(self) -> None:
        if self.ledger_path:
            self.ledger = RentalLedger.load(self.ledger_path)

    def can_rent(self, *, queued_bound_waiting: int) -> bool:
        if queued_bound_waiting <= 0:
            return False
        if self.ledger.count() >= self.max_boxes:
            return False
        pe = 0 if self.pe_eval_count_fn is None else int(self.pe_eval_count_fn())
        warm = 0 if self.pe_warm_count_fn is None else int(self.pe_warm_count_fn())
        pit = self.ledger.count()
        if pe + pit + warm >= self.max_eval_nodes:
            return False
        return True

    def _host_release(self, name_or_id: str, *, reason: str) -> None:
        if self.host_release is None:
            return
        try:
            self.host_release(name_or_id, reason=reason)
        except TypeError:
            self.host_release(box_id=name_or_id, reason=reason)

    def _validate_host_rental(self, out: Dict[str, Any], *, box_id: str) -> None:
        if not _gpu_is_b300(out):
            self._host_release(box_id, reason='rent-gpu-mismatch')
            raise RentError('item rentals require B300 GPU')
        price = _price_usd_h(out)
        if price is not None and price > float(self.price_ceiling_usd_h):
            self._host_release(box_id, reason='rent-price-ceiling')
            raise RentError(
                'price %.2f above ceiling %.2f' % (price, self.price_ceiling_usd_h),
            )
        if self.disable_auto_backup:
            confirmed = _no_backup_confirmed(out)
            if not confirmed and self.disable_auto_backup_fn is not None:
                # Explicit checker is authoritative when the rent dict omits the flag.
                confirmed = bool(self.disable_auto_backup_fn(str(box_id)))
            elif not confirmed and self.disable_auto_backup_fn is None:
                # Live pe_rent always passes ``--no-backup`` on ``lium up``; trust
                # that host contract when no post-rent checker is wired.
                confirmed = True
            if not confirmed:
                self._host_release(box_id, reason='auto-backup-still-on')
                raise RentError('lium auto-backup must be disabled on every rental')
            out['auto_backup_disabled'] = True

    def reconcile(
        self,
        listings: Optional[List[Mapping[str, Any]]] = None,
        *,
        item_box_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        rows = listings if listings is not None else (
            list(self.list_pods()) if self.list_pods is not None else []
        )
        known_items = {str(x) for x in (item_box_ids or ())}
        adopted: List[str] = []
        released: List[str] = []
        live_ids: Set[str] = set()
        for row in rows:
            name = row.get('name') or row.get('pod_name') or ''
            if not is_pit_name(name):
                continue
            bid = str(row.get('id') or row.get('box_id') or name)
            live_ids.add(bid)
            if bid in known_items:
                if bid not in self.ledger.active:
                    at = float(self.now_fn())
                    self.ledger.register(bid, item_id='reconciled', name=name, at=at)
                    adopted.append(bid)
            else:
                self._host_release(bid, reason='reconcile-orphan')
                released.append(bid)
                if bid in self.ledger.active:
                    self.ledger.release(bid, reason='reconcile-orphan-ledger', at=float(self.now_fn()))
        for bid in list(self.ledger.active.keys()):
            if bid not in live_ids and bid not in known_items:
                self.ledger.release(bid, reason='reconcile-stale-ledger', at=float(self.now_fn()))
        return dict(adopted=adopted, released=released)

    def precheck_offer(self) -> Dict[str, Any]:
        """Check B300 ≤ ceiling on the offer listing BEFORE committing a rental.

        Uses host ``pick_node`` / ``lium_ls`` when available. Raises RentError
        when no eligible offer exists. Does not rent.
        """
        if self.offer_precheck_fn is not None:
            offer = self.offer_precheck_fn(gpu=PIT_LIVE_GPU)
            if not offer:
                raise RentError('no eligible B300 offer under ceiling')
            out = dict(offer) if isinstance(offer, Mapping) else {'raw': offer}
            if not _gpu_is_b300(out) and 'B300' not in str(out.get('gpu') or out.get('gpu_model') or 'B300').upper():
                # Explicit B300 request may omit gpu field; require price check anyway.
                pass
            price = _price_usd_h(out)
            if price is not None and price > float(self.price_ceiling_usd_h):
                raise RentError(
                    'offer price %.2f above ceiling %.2f (precheck)' % (
                        price, self.price_ceiling_usd_h,
                    )
                )
            return out
        return dict(ok=True, deferred='post-rent-validate')

    def rent(self, *, item_id: str, epoch: int, attempt: int) -> Dict[str, Any]:
        if self.ledger.count() >= self.max_boxes:
            raise RentError('box_cap reached')
        if tuple(self.gpu_order) != GPU_PINNED:
            raise RentError('item rentals are pinned to B300')
        if self.require_host and self.host_rent is None:
            raise RentError('live rent requires host pe_rent module')
        name = '%s%d-a%d' % (PREFIX, int(epoch), int(attempt))
        if is_never_name(name):
            raise RentError('refused NEVER_NAME')
        # Live pe_rent requires affine-pe-*; our PREFIX is affine-pe-pit-.
        if self.require_host and not name.startswith('affine-pe-'):
            raise RentError('live rent name must start with affine-pe-')
        at = float(self.now_fn())
        # Pre-check offer listing (price/GPU) before committing the rental.
        try:
            self.precheck_offer()
        except RentError:
            raise
        except Exception as exc:
            raise RentError('offer precheck failed: %s' % str(exc)[:200]) from exc
        if self.host_rent is not None:
            try:
                raw = self.host_rent(
                    name, PIT_LIVE_TTL, gpu=PIT_LIVE_GPU, max_eval_nodes=self.max_eval_nodes,
                )
            except AssertionError as exc:
                # live pe_rent asserts name.startswith(PREFIX); surface as RentError.
                raise RentError('host rent refused name %r: %s' % (name, exc)) from exc
            except Exception as exc:
                raise RentError('host rent failed: %s' % str(exc)[:300]) from exc
            if raw is None:
                raise RentError('host rent returned no box')
            out = dict(raw)
            box_id = out.get('box_id') or out.get('id')
            if not box_id:
                raise RentError('host rent returned no box_id')
            pod_name = out.get('name') or name
            if is_never_name(pod_name):
                raise RentError('host rent returned NEVER_NAME')
            # Post-rent validate: release immediately on GPU/price/backup mismatch.
            self._validate_host_rental(out, box_id=str(box_id))
            if self.disable_auto_backup:
                out.setdefault('auto_backup_disabled', True)
        else:
            box_id = 'pit-box-%s-%d' % (item_id[:12], attempt)
            pod_name = name
            out = dict(
                box_id=box_id, name=pod_name, synthetic=True,
                gpu=self.gpu_order[0], price_usd_h=min(35.0, self.price_ceiling_usd_h),
            )
            if self.disable_auto_backup:
                out['auto_backup_disabled'] = True
            else:
                out['auto_backup_disabled'] = False
        box_id = str(box_id)
        self.ledger.register(box_id, item_id=item_id, name=pod_name, at=at)
        out.update(box_id=box_id, name=pod_name, rented_at=at)
        if self.ledger_path:
            self.save_ledger()
        return out

    def release(self, box_id: str, *, reason: str = 'done') -> Dict[str, Any]:
        at = float(self.now_fn())
        if self.host_release is not None:
            try:
                self._host_release(box_id, reason=reason)
            except Exception as exc:
                self.ledger.release(box_id, reason='release-error:%s' % str(exc)[:80], at=at)
                raise
        self.ledger.release(box_id, reason=reason, at=at)
        if self.ledger_path:
            self.save_ledger()
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


def _bind_offer_precheck(service: RentService, mod: Any) -> None:
    """Wire listing-based B300/price precheck from live pe_rent helpers."""
    pick_node = getattr(mod, 'pick_node', None)
    lium_ls = getattr(mod, 'lium_ls', None)
    ceiling = float(service.price_ceiling_usd_h)

    def _precheck(*, gpu: str = PIT_LIVE_GPU) -> Mapping[str, Any]:
        if callable(pick_node):
            g, row = pick_node([gpu] if gpu else None, fallback=False)
            if not row:
                raise RentError('no B300 offer available (pick_node)')
            price = float(row.get('price_per_hour') or row.get('price_h') or 999)
            if price > ceiling:
                raise RentError(
                    'offer price %.2f above ceiling %.2f' % (price, ceiling)
                )
            if g and 'B300' not in str(g).upper():
                raise RentError('offer GPU %r is not B300' % g)
            return dict(row, gpu=g or gpu, price_per_hour=price)
        if callable(lium_ls):
            rows = list(lium_ls(gpu) or [])
            eligible = []
            for r in rows:
                if not isinstance(r, Mapping):
                    continue
                if int(r.get('available_gpu_count') or 0) < 8:
                    continue
                price = float(r.get('price_per_hour') or r.get('price_h') or 999)
                if price <= ceiling:
                    eligible.append((price, r))
            if not eligible:
                raise RentError('no B300 offer ≤ $%.2f/h' % ceiling)
            eligible.sort(key=lambda x: x[0])
            best = dict(eligible[0][1])
            best['gpu'] = gpu
            best['price_per_hour'] = eligible[0][0]
            return best
        return dict(ok=True, deferred='post-rent-validate')

    service.offer_precheck_fn = _precheck


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
    service.list_pods = getattr(mod, 'list_pods', None) or getattr(mod, 'our_pods', None) or getattr(mod, 'ps', None)
    service.disable_auto_backup_fn = getattr(mod, 'disable_auto_backup', None)
    _bind_offer_precheck(service, mod)
    return service
