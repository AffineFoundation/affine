"""PIT rental seam (BRIEF §1.3 / §1.9, SPEC §4).

Host RA ``pe_rent.py`` owns real Lium rental. This git port:

* defines the on-demand rent/release API PIT uses;
* counts ``affine-pe-pit-*`` boxes toward the item box_cap only
  (live ``pe_rent`` requires names starting ``affine-pe-``);
* refuses NEVER_NAMES / warm-pool semantics;
* prefers B300 then cheapest allowed GPU under the $56/h ceiling;
* requires a full 8-GPU pack of an allowed class (B300/RTX5090/B200/H200/H100);
* disables Lium auto-backup on every rental;
* loads the host module when present.

Hard rule: rent only when the controller has a bound item ready. No warm pool.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set

from . import (
    ALLOWED_GPUS,
    GPU_ORDER,
    GPU_PINNED,
    PREFIX,
    PREFIX_LIVE,
    PREFIX_SHADOW,
    PRICE_CEILING_USD_H,
)


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
# Preferred first class; pick_node falls back to cheaper allowed classes.
PIT_LIVE_GPU = 'B300'


class RentError(RuntimeError):
    pass


def is_shadow_pit_name(name: str) -> bool:
    return isinstance(name, str) and name.startswith(PREFIX_SHADOW)


def is_live_pit_name(name: str) -> bool:
    # PREFIX_SHADOW is a longer sub-prefix of PREFIX_LIVE — exclude it.
    return (
        isinstance(name, str)
        and name.startswith(PREFIX_LIVE)
        and not is_shadow_pit_name(name)
    )


def is_pit_name(name: str) -> bool:
    return is_live_pit_name(name) or is_shadow_pit_name(name)


def is_never_name(name: str) -> bool:
    if not isinstance(name, str):
        return True
    if is_pit_name(name):
        return False
    return any(name.startswith(p) for p in NEVER_NAME_PREFIXES)


def count_pit_boxes(listings: List[Mapping[str, Any]], *, shadow: Optional[bool] = None) -> int:
    n = 0
    for row in listings:
        name = row.get('name') or row.get('pod_name') or ''
        if shadow is True:
            if is_shadow_pit_name(name):
                n += 1
        elif shadow is False:
            if is_live_pit_name(name):
                n += 1
        elif is_pit_name(name):
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


def _gpu_class_of(pod: Mapping[str, Any]) -> Optional[str]:
    """Normalize a rental GPU string to an ALLOWED_GPUS member, or None."""
    chunks: List[str] = []
    for key in ('gpu', 'gpu_model', 'gpu_name'):
        val = pod.get(key)
        if val is not None:
            chunks.append(str(val).upper())
    name = pod.get('name') or ''
    if name:
        chunks.append(str(name).upper())
    text = ' '.join(chunks)
    # Check longer / more specific names first (RTX5090 before bare 5090).
    for cls in ALLOWED_GPUS:
        token = cls.upper()
        if token in text:
            return cls
        if token == 'RTX5090' and '5090' in text:
            return 'RTX5090'
    return None


def _gpu_is_allowed(pod: Mapping[str, Any]) -> bool:
    return _gpu_class_of(pod) in ALLOWED_GPUS


def _gpu_is_b300(pod: Mapping[str, Any]) -> bool:
    """Backward-compatible alias — True when the rental is specifically B300."""
    return _gpu_class_of(pod) == 'B300'


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
        # Idempotent: skip when already out of active (decided-item retries /
        # restarts must not grow history with duplicate release rows).
        if box_id not in self.active:
            return
        meta = self.active.pop(box_id)
        self.history.append(dict(
            op='release', box_id=box_id, reason=reason, at=at,
            item_id=meta.get('item_id'),
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
    gpu_order: tuple = GPU_ORDER
    price_ceiling_usd_h: float = PRICE_CEILING_USD_H
    disable_auto_backup: bool = True
    require_host: bool = False
    # Shadow: own prefix + own cap; never shares the live PE 5-node budget.
    shadow: bool = False
    name_prefix: str = PREFIX_LIVE
    share_pe_cap: bool = True
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
        # Shadow has its own cap (max_boxes); do not consume live PE budget.
        if self.shadow or not self.share_pe_cap:
            return True
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
        if not _gpu_is_allowed(out):
            self._host_release(box_id, reason='rent-gpu-mismatch')
            raise RentError(
                'item rentals require one of %s (got %r)'
                % (list(ALLOWED_GPUS), out.get('gpu') or out.get('gpu_model'))
            )
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
        """Check allowed 8-GPU ≤ ceiling BEFORE committing a rental.

        Uses host ``pick_node`` / ``lium_ls`` when available (B300-first-then-
        cheapest). Raises RentError when no eligible offer exists. Does not rent.
        """
        if self.offer_precheck_fn is not None:
            offer = self.offer_precheck_fn(gpu=None)
            if not offer:
                raise RentError('no eligible 8-GPU offer under ceiling')
            out = dict(offer) if isinstance(offer, Mapping) else {'raw': offer}
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
        order = tuple(self.gpu_order) or GPU_ORDER
        if any(g not in ALLOWED_GPUS for g in order):
            raise RentError('gpu_order has unsupported class: %r' % (order,))
        if self.require_host and self.host_rent is None:
            raise RentError('live rent requires host pe_rent module')
        prefix = self.name_prefix or (PREFIX_SHADOW if self.shadow else PREFIX_LIVE)
        name = '%s%d-a%d' % (prefix, int(epoch), int(attempt))
        if is_never_name(name):
            raise RentError('refused NEVER_NAME')
        # Live pe_rent requires affine-pe-* for rent AND release. Live + shadow
        # PIT both use affine-pe-pit-* (shadow: affine-pe-pit-shadow-*).
        if self.require_host and not name.startswith('affine-pe-'):
            raise RentError('live/shadow rent name must start with affine-pe-')
        at = float(self.now_fn())
        # Pre-check offer listing (price/GPU) before committing the rental.
        try:
            self.precheck_offer()
        except RentError:
            raise
        except Exception as exc:
            raise RentError('offer precheck failed: %s' % str(exc)[:200]) from exc
        # gpu=None → host pick_node uses full order (B300-first-then-cheapest).
        rent_kw: Dict[str, Any] = dict(gpu=None)
        if self.share_pe_cap and not self.shadow:
            rent_kw['max_eval_nodes'] = self.max_eval_nodes
        if self.host_rent is not None:
            try:
                raw = self.host_rent(name, PIT_LIVE_TTL, **rent_kw)
            except TypeError:
                # Some host seams use positional ttl only / require an explicit gpu.
                try:
                    raw = self.host_rent(
                        name, PIT_LIVE_TTL,
                        gpu=order[0], max_eval_nodes=self.max_eval_nodes,
                    )
                except Exception as exc:
                    raise RentError('host rent failed: %s' % str(exc)[:300]) from exc
            except AssertionError as exc:
                # live pe_rent asserts name.startswith(affine-pe-); surface as RentError.
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
                gpu=order[0], price_usd_h=min(35.0, self.price_ceiling_usd_h),
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
        if box_id not in self.ledger.active:
            return dict(ok=True, box_id=box_id, already_released=True, reason=reason, at=at)
        name_hint = str(self.ledger.active[box_id].get('name') or '')
        if self.host_release is not None:
            try:
                self._host_release(box_id, reason=reason)
            except Exception as exc:
                self.ledger.release(box_id, reason='release-error:%s' % str(exc)[:80], at=at)
                raise
        # Verify the pod is actually gone (lium ps). Alert loudly if still billing.
        still = self._verify_released(box_id, name=name_hint)
        if still:
            print(
                json.dumps({
                    'component': 'pit-pe-rent',
                    'level': 'error',
                    'event': 'release_verify_failed',
                    'box_id': box_id,
                    'name': name_hint or still.get('name'),
                    'reason': reason,
                    'note': 'ledger released but pod still visible in lium ps — billing continues',
                    'still': dict(still),
                }),
                file=sys.stderr,
                flush=True,
            )
            self.ledger.release(
                box_id,
                reason='release-verify-failed:%s' % reason[:60],
                at=at,
            )
            if self.ledger_path:
                self.save_ledger()
            raise RentError(
                'release verify failed: pod %s still in lium ps after release' % (
                    name_hint or box_id,
                )
            )
        self.ledger.release(box_id, reason=reason, at=at)
        if self.ledger_path:
            self.save_ledger()
        return dict(ok=True, box_id=box_id, reason=reason, released_at=at)

    def _verify_released(self, box_id: str, *, name: str = '') -> Optional[Mapping[str, Any]]:
        """Return the still-live pod row if release did not remove it; else None."""
        if self.list_pods is None:
            return None
        try:
            rows = list(self.list_pods() or [])
        except Exception as exc:
            print(
                json.dumps({
                    'component': 'pit-pe-rent',
                    'level': 'error',
                    'event': 'release_verify_ps_failed',
                    'box_id': box_id,
                    'error': str(exc)[:200],
                }),
                file=sys.stderr,
                flush=True,
            )
            return None
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            rid = str(row.get('id') or row.get('box_id') or '')
            rname = str(row.get('name') or row.get('pod_name') or '')
            if rid and rid == str(box_id):
                return row
            if name and rname == name:
                return row
            if rname and rname == str(box_id):
                return row
        return None


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
    """Wire listing-based B300-first-then-cheapest/price precheck."""
    pick_node = getattr(mod, 'pick_node', None)
    lium_ls = getattr(mod, 'lium_ls', None)
    pick_best = getattr(mod, 'pick_best_offer', None)
    ceiling = float(service.price_ceiling_usd_h)
    order = tuple(service.gpu_order) or GPU_ORDER

    def _precheck(*, gpu: Optional[str] = None) -> Mapping[str, Any]:
        prefer = [gpu] if gpu else list(order)
        if callable(pick_node):
            try:
                g, row = pick_node(prefer, fallback=True, price_ceiling=ceiling)
            except TypeError:
                # Older host pe_rent without price_ceiling kwarg.
                g, row = pick_node(prefer, fallback=True)
            if not row:
                raise RentError('no eligible 8-GPU offer under ceiling (pick_node)')
            price = float(row.get('price_per_hour') or row.get('price_h') or 999)
            if price > ceiling:
                raise RentError(
                    'offer price %.2f above ceiling %.2f' % (price, ceiling)
                )
            if g and g not in ALLOWED_GPUS:
                raise RentError('offer GPU %r not in allow-list' % g)
            return dict(row, gpu=g or gpu or order[0], price_per_hour=price)
        if callable(lium_ls):
            cands = []
            for g in prefer:
                try:
                    rows = list(lium_ls(g) or [])
                except Exception:
                    continue
                for r in rows:
                    if not isinstance(r, Mapping):
                        continue
                    if int(r.get('available_gpu_count') or 0) < 8:
                        continue
                    price = float(r.get('price_per_hour') or r.get('price_h') or 999)
                    cands.append((g, r, price))
            if callable(pick_best):
                g, row = pick_best(cands, order=prefer, ceiling=ceiling)
            else:
                under = [(g, r, p) for g, r, p in cands if p <= ceiling]
                if not under:
                    g, row = None, None
                else:
                    pref = [x for x in under if x[0] == prefer[0]]
                    pool = pref or under
                    pool.sort(key=lambda x: x[2])
                    g, row = pool[0][0], pool[0][1]
            if not row:
                raise RentError('no eligible 8-GPU offer ≤ $%.2f/h' % ceiling)
            price = float(row.get('price_per_hour') or row.get('price_h') or 999)
            best = dict(row)
            best['gpu'] = g or prefer[0]
            best['price_per_hour'] = price
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
