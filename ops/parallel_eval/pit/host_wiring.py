"""Live host wiring (PR #96 blocker 1).

``build_live_host_bundle`` imports and binds every live seam. Missing any required
host module raises ``LiveWiringError`` (fail closed).
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .box import BoxIO, ScriptedBoxIO
from .config import PitConfig, load_pit_config
from .decision_emit import gate_epochs_root
from .gate_writer import HostGateWriter
from .pe_lane import LaneService, bind_host_lane_service
from .pe_rent import RentService, bind_host_rent_service
from .signer import LIVE_ROOT_MODE, load_delegated_host_signer


class LiveWiringError(RuntimeError):
    """Live mode refused: required host module or callback missing."""


def _import_module(candidates: list[Optional[str]], label: str):
    seen: set[str] = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise LiveWiringError('%s: no importable module in %r' % (label, [c for c in candidates if c]))


def _require_attr(mod: Any, names: tuple[str, ...], label: str) -> Callable[..., Any]:
    for attr in names:
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn
    raise LiveWiringError('%s: module %r missing any of %s' % (label, mod.__name__, names))


@dataclass
class LiveHostBundle:
    config: PitConfig
    rent: RentService
    lane: LaneService
    box_io_factory: Callable[[], BoxIO]
    score_fn: Callable[..., Mapping[str, Any]]
    verify_uploads: Callable[..., Mapping[str, Any]]
    gate_writer: HostGateWriter
    root_signer: Any
    plan_seed_fn: Callable[[int], str]
    require_host: bool = True
    pe_apply_accept: Optional[Callable[..., Any]] = None
    destroy_box_state_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    upload_master: Optional[Callable[..., Mapping[str, Any]]] = None
    upload_candidate: Optional[Callable[..., Mapping[str, Any]]] = None


def build_live_host_bundle(
    parallel_eval: Mapping[str, Any],
    *,
    validate_only: bool = False,
) -> LiveHostBundle:
    """Wire all live host seams. Raises ``LiveWiringError`` if anything is missing."""
    cfg = load_pit_config(parallel_eval)
    if not cfg.is_live() or not cfg.enabled:
        raise LiveWiringError('build_live_host_bundle requires enabled live pit config')
    if cfg.root_mode != LIVE_ROOT_MODE:
        raise LiveWiringError('live ROOT path is delegated_host only')

    extra = dict(cfg.extra or {})
    rent_mod = _import_module(
        [cfg.pe_rent_module, extra.get('pe_rent_module'), 'pe_rent'],
        'pe_rent',
    )
    lane_mod = _import_module(
        [cfg.pe_lane_module, extra.get('pe_lane_module'), 'pe_lane'],
        'pe_lane',
    )
    score_mod = _import_module(
        [cfg.pe_score_module, extra.get('pe_score_module'), 'pe_score'],
        'pe_score',
    )
    box_mod = _import_module(
        [extra.get('box_io_module'), extra.get('pe_box_module'), 'pe_box', 'ops.parallel_eval.live.pe_box'],
        'box_io',
    )
    verify_mod = _import_module(
        [extra.get('verify_uploads_module'), 'pe_verify', 'pe_upload_verify'],
        'verify_uploads',
    )
    seed_mod = _import_module(
        [extra.get('round_seed_module'), extra.get('plan_seed_module'), 'pe_round_seed'],
        'round_seed',
    )

    host_rent = _require_attr(rent_mod, ('rent', 'rent_box'), 'pe_rent')
    host_release = _require_attr(rent_mod, ('release', 'release_box'), 'pe_rent release')
    list_pods = getattr(rent_mod, 'list_pods', None) or getattr(rent_mod, 'our_pods', None) or getattr(rent_mod, 'ps', None)
    if not callable(list_pods):
        raise LiveWiringError('pe_rent: list_pods/ps required for live reconcile')

    host_boot = _require_attr(lane_mod, ('boot',), 'pe_lane boot')
    host_serve = _require_attr(lane_mod, ('serve',), 'pe_lane serve')
    host_run_probe = _require_attr(lane_mod, ('run_probe',), 'pe_lane run_probe')

    host_score = getattr(score_mod, 'score', None)
    if not callable(host_score):
        raise LiveWiringError('pe_score.score required')

    verify_uploads = getattr(verify_mod, 'verify_uploads', None) or getattr(verify_mod, 'verify_green_uploads', None)
    if not callable(verify_uploads):
        raise LiveWiringError('verify_uploads callable required (fail closed)')

    plan_seed_fn = (
        getattr(seed_mod, 'plan_seed_for_epoch', None)
        or getattr(seed_mod, 'plan_seed_fn', None)
        or getattr(seed_mod, 'cand_seed_for_epoch', None)
    )
    if not callable(plan_seed_fn):
        raise LiveWiringError('round seed provider must expose plan_seed_for_epoch / plan_seed_fn')

    box_io_cls = getattr(box_mod, 'BoxIO', None) or getattr(box_mod, 'HostBoxIO', None)
    box_factory = getattr(box_mod, 'box_io_factory', None)
    if box_factory is None and box_io_cls is not None:
        box_factory = box_io_cls
    if not callable(box_factory):
        raise LiveWiringError('box_io factory/class required (not ScriptedBoxIO)')

    if box_factory is ScriptedBoxIO:
        raise LiveWiringError('ScriptedBoxIO refused in live mode')

    def _box_io_factory() -> BoxIO:
        io = box_factory() if not isinstance(box_factory, type) else box_factory()
        if isinstance(io, ScriptedBoxIO):
            raise LiveWiringError('ScriptedBoxIO refused in live mode')
        return io

    try:
        root_signer = load_delegated_host_signer()
    except Exception as exc:
        raise LiveWiringError('delegated root signer required: %s' % exc) from exc

    ledger_path = extra.get('rent_ledger_path') or os.environ.get('PIT_RENT_LEDGER')
    rent = RentService(
        max_boxes=cfg.max_boxes,
        gpu_order=tuple(cfg.gpu_order),
        price_ceiling_usd_h=float(cfg.price_ceiling_usd_h),
        disable_auto_backup=bool(cfg.disable_lium_auto_backup),
        require_host=True,
        host_rent=host_rent,
        host_release=host_release,
        list_pods=list_pods,
        ledger_path=ledger_path,
    )

    lane = LaneService(
        require_host=True,
        host_boot=host_boot,
        host_serve=host_serve,
        host_run_probe=host_run_probe,
        host_release=getattr(lane_mod, 'release', None),
    )

    def score_fn(**kwargs: Any) -> Mapping[str, Any]:
        return dict(host_score(**kwargs))

    gate_writer = HostGateWriter(
        epochs_root=gate_epochs_root(extra.get('gate_epochs_root')),
        shadow=False,
    )

    pe_apply_accept = None
    destroy_fn = getattr(box_mod, 'destroy_box_state', None)
    upload_master = getattr(box_mod, 'upload_master', None)
    upload_candidate = getattr(box_mod, 'upload_candidate', None)
    try:
        pe_apply = importlib.import_module('pe_apply')
        pe_apply_accept = getattr(pe_apply, 'accept_pit_decision', None)
    except ImportError:
        pass

    bundle = LiveHostBundle(
        config=cfg,
        rent=rent,
        lane=lane,
        box_io_factory=_box_io_factory,
        score_fn=score_fn,
        verify_uploads=verify_uploads,
        gate_writer=gate_writer,
        root_signer=root_signer,
        plan_seed_fn=plan_seed_fn,
        require_host=True,
        pe_apply_accept=pe_apply_accept,
        destroy_box_state_fn=destroy_fn if callable(destroy_fn) else None,
        upload_master=upload_master if callable(upload_master) else None,
        upload_candidate=upload_candidate if callable(upload_candidate) else None,
    )
    if validate_only:
        return bundle
    return bundle
