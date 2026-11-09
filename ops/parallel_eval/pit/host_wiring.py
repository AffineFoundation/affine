"""Live host wiring (PR #96 blocker 1 + review-2).

``build_live_host_bundle`` imports and binds every live seam. Missing any required
host module raises ``LiveWiringError`` (fail closed).

Also binds ``pe_eval_count_fn`` / ``pe_warm_count_fn`` so PIT and PE share one
global eval-node cap, and adapts live ``pe_score.score(epoch, rid, result)``
(positional) into a heldout-nested score_fn.
"""
from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .box import BoxIO, ScriptedBoxIO
from .config import PitConfig, load_pit_config
from .decision_emit import gate_epochs_root
from .gate_writer import HostGateWriter
from .pe_lane import LaneService, bind_host_lane_service
from .pe_rent import RentService, bind_host_rent_service
from .pe_score import call_live_host_score
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


def _pe_state_root() -> Path:
    env = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
    if env:
        return Path(env)
    return Path(
        '/home/const/subnet120-rewrite/state/root-audits/'
        '20261106-parallel-cutover/state'
    )


def _bind_shared_cap_counters(rent: RentService, rent_mod: Any) -> None:
    """PIT + PE share one global eval-node cap (review-2 blocker 4)."""
    eval_nodes = getattr(rent_mod, 'eval_nodes', None)
    our_pods = getattr(rent_mod, 'our_pods', None) or getattr(rent_mod, 'list_pods', None)

    def _pe_eval_count() -> int:
        if callable(eval_nodes):
            try:
                nodes = list(eval_nodes())
            except TypeError:
                nodes = list(eval_nodes(None))
            warm_names = _warm_pod_names()
            n = 0
            for p in nodes:
                name = ''
                if isinstance(p, Mapping):
                    name = str(p.get('name') or p.get('pod_name') or '')
                else:
                    name = str(p)
                if name in warm_names or name.startswith('affine-warm-'):
                    continue
                if name.startswith('affine-pe-pit-') or name.startswith('affine-pit-'):
                    continue
                n += 1
            return n
        if callable(our_pods):
            rows = list(our_pods() or [])
            return sum(
                1 for p in rows
                if str((p or {}).get('name') or '').startswith('affine-pe-')
            )
        return 0

    def _pe_warm_count() -> int:
        return len(_warm_active_pods())

    rent.pe_eval_count_fn = _pe_eval_count
    rent.pe_warm_count_fn = _pe_warm_count


def _warm_pod_names() -> set[str]:
    warm = _pe_state_root() / 'WARM.json'
    try:
        doc = json.loads(warm.read_text()) if warm.is_file() else {}
    except Exception:
        doc = {}
    pods = (doc or {}).get('pods') or {}
    return set(pods.keys()) if isinstance(pods, Mapping) else set()


def _warm_active_pods() -> list[str]:
    warm = _pe_state_root() / 'WARM.json'
    try:
        doc = json.loads(warm.read_text()) if warm.is_file() else {}
    except Exception:
        doc = {}
    pods = (doc or {}).get('pods') or {}
    active = (
        'renting', 'booting', 'ready', 'prefetching', 'served', 'claimed',
    )
    out = []
    if isinstance(pods, Mapping):
        for name, entry in pods.items():
            if isinstance(entry, Mapping) and entry.get('state') in active:
                out.append(str(name))
    return out


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
    indices_fn: Callable[[int], Sequence[int]]
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
    if not cfg.enabled or not (cfg.is_live() or cfg.is_shadow()):
        raise LiveWiringError('build_live_host_bundle requires enabled live or shadow pit config')
    if cfg.is_live() and cfg.root_mode != LIVE_ROOT_MODE:
        raise LiveWiringError('live ROOT path is delegated_host only')
    # Shadow may use fixture signer; live still requires delegated_host.

    extra = dict(cfg.extra or {})
    rent_mod = _import_module(
        [
            cfg.pe_rent_module, extra.get('pe_rent_module'),
            'pe_rent', 'ops.parallel_eval.live.pe_rent',
        ],
        'pe_rent',
    )
    lane_mod = _import_module(
        [
            cfg.pe_lane_module, extra.get('pe_lane_module'),
            'pe_lane', 'ops.parallel_eval.live.pe_lane',
        ],
        'pe_lane',
    )
    score_mod = _import_module(
        [
            cfg.pe_score_module, extra.get('pe_score_module'),
            'pe_score', 'ops.parallel_eval.live.pe_score',
        ],
        'pe_score',
    )
    box_mod = _import_module(
        [
            extra.get('box_io_module'), extra.get('pe_box_module'),
            'pe_box', 'ops.parallel_eval.live.pe_box',
        ],
        'box_io',
    )
    verify_mod = _import_module(
        [
            extra.get('verify_uploads_module'),
            'pe_verify', 'pe_upload_verify',
            'ops.parallel_eval.live.pe_verify',
            'ops.parallel_eval.live.pe_upload_verify',
        ],
        'verify_uploads',
    )
    seed_mod = _import_module(
        [
            extra.get('round_seed_module'), extra.get('plan_seed_module'),
            'pe_round_seed', 'ops.parallel_eval.live.pe_round_seed',
        ],
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

    indices_fn = getattr(seed_mod, 'indices_for_epoch', None)
    if not callable(indices_fn):
        raise LiveWiringError('round seed provider must expose indices_for_epoch (fail closed)')

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

    root_signer = None
    try:
        root_signer = load_delegated_host_signer()
    except Exception as exc:
        if cfg.is_live():
            raise LiveWiringError('delegated root signer required: %s' % exc) from exc
        # Shadow: real boxes still preferred; admission may use fixture later.
        root_signer = None

    ledger_path = extra.get('rent_ledger_path') or os.environ.get('PIT_RENT_LEDGER')
    rent = RentService(
        max_boxes=cfg.max_boxes,
        max_eval_nodes=int(extra.get('max_eval_nodes') or cfg.max_boxes or 5),
        gpu_order=tuple(cfg.gpu_order),
        price_ceiling_usd_h=float(cfg.price_ceiling_usd_h),
        disable_auto_backup=bool(cfg.disable_lium_auto_backup),
        require_host=True,
        host_rent=host_rent,
        host_release=host_release,
        list_pods=list_pods,
        ledger_path=ledger_path,
    )
    _bind_shared_cap_counters(rent, rent_mod)
    from .pe_rent import _bind_offer_precheck
    _bind_offer_precheck(rent, rent_mod)

    def _pod_of(handle):
        if handle is None:
            raise LiveWiringError('pod handle required')
        if hasattr(handle, 'run'):
            return handle
        if isinstance(handle, Mapping):
            pod_of = getattr(lane_mod, 'pod_of', None) or getattr(rent_mod, 'pod_of', None)
            if not callable(pod_of):
                try:
                    import pe_common as _pc
                    pod_of = _pc.pod_of
                except Exception:
                    from ops.parallel_eval.live import pe_common as _pc
                    pod_of = _pc.pod_of
            return pod_of(handle)
        raise LiveWiringError('pod handle must be Pod or rent dict')

    def _boot(pod, **kw):
        return host_boot(_pod_of(pod), **kw)

    def _serve(pod, cid, gmu='0.9', max_num_seqs=512, **kw):
        return host_serve(_pod_of(pod), cid, gmu=gmu, max_num_seqs=max_num_seqs, **kw)

    def _run_probe(pod, cid, indices, plan_seed, **kw):
        return host_run_probe(_pod_of(pod), cid, indices, plan_seed, **kw)

    lane = LaneService(
        require_host=True,
        host_boot=_boot,
        host_serve=_serve,
        host_run_probe=_run_probe,
        host_release=getattr(lane_mod, 'release', None),
    )

    def score_fn(**kwargs: Any) -> Mapping[str, Any]:
        # Always call live score(epoch, rid, result) positionally via adapter.
        return call_live_host_score(
            host_score,
            epoch=int(kwargs['epoch']),
            round_id=str(kwargs.get('round_id') or kwargs.get('rid') or ''),
            result=kwargs.get('result') or {},
        )

    gate_writer = HostGateWriter(
        epochs_root=gate_epochs_root(extra.get('gate_epochs_root')),
        shadow=bool(cfg.is_shadow()),
    )

    pe_apply_accept = None
    destroy_fn = getattr(box_mod, 'destroy_box_state', None)
    upload_master = getattr(box_mod, 'upload_master', None)
    upload_candidate = getattr(box_mod, 'upload_candidate', None)
    for apply_name in (
        'pe_apply',
        'ops.parallel_eval.live.pe_apply',
    ):
        try:
            pe_apply = importlib.import_module(apply_name)
            pe_apply_accept = getattr(pe_apply, 'accept_pit_decision', None)
            if callable(pe_apply_accept):
                break
        except ImportError:
            continue

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
        indices_fn=indices_fn,
        require_host=True,
        pe_apply_accept=pe_apply_accept,
        destroy_box_state_fn=destroy_fn if callable(destroy_fn) else None,
        upload_master=upload_master if callable(upload_master) else None,
        upload_candidate=upload_candidate if callable(upload_candidate) else None,
    )
    if validate_only:
        return bundle
    return bundle
