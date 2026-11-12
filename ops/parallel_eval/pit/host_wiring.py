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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from . import PREFIX_LIVE, PREFIX_SHADOW
from .box import BoxIO, ScriptedBoxIO
from .config import PitConfig, load_pit_config
from .decision_emit import gate_epochs_root
from .gate_writer import HostGateWriter, build_host_rounds
from .life_paths import pe_state_root
from .pe_lane import LaneService, bind_host_lane_service
from .pe_rent import RentService, _bind_offer_precheck, bind_host_rent_service
from . import probe_fingerprint
from .pe_score import call_live_host_score, nest_heldout_fields
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
    return pe_state_root()


def _pit_live_cutover() -> bool:
    """True when the PE flag has live PIT cutover enabled (warm pool suppressed)."""
    flag = os.environ.get('PE_FLAG')
    if not flag:
        return False
    try:
        doc = json.loads(Path(flag).read_text())
    except Exception:
        return False
    pe = (doc or {}).get('parallel_eval') or {}
    pit = pe.get('pit') or {}
    if pit.get('revert') is True:
        return False
    return (
        pe.get('enabled') is True
        and pit.get('enabled') is True
        and pit.get('mode') == 'live'
        and pit.get('cutover') is True
    )


def _warm_suppressed() -> bool:
    """WARM-OFF file or live PIT cutover — do not invent warm capacity pressure."""
    return (_pe_state_root() / 'WARM-OFF').exists() or _pit_live_cutover()


def _live_pod_ids(rent_mod: Any) -> set[str]:
    """Current Lium pod ids from pe_rent our_pods/list_pods/ps (empty if unavailable)."""
    our_pods = (
        getattr(rent_mod, 'our_pods', None)
        or getattr(rent_mod, 'list_pods', None)
        or getattr(rent_mod, 'ps', None)
    )
    if not callable(our_pods):
        return set()
    try:
        rows = list(our_pods() or [])
    except TypeError:
        try:
            rows = list(our_pods(None) or [])
        except Exception:
            return set()
    except Exception:
        return set()
    ids: set[str] = set()
    for p in rows:
        if isinstance(p, Mapping):
            pid = p.get('id')
            if pid:
                ids.add(str(pid))
    return ids


def _make_shadow_rent(rent_mod: Any, host_rent: Any) -> Callable[..., Mapping[str, Any]]:
    """Rent ``affine-pe-pit-shadow-*`` via live pe_rent (prefix satisfies affine-pe-*).

    Shadow must use the live rent/release path so ``pe_rent.release``'s
    ``affine-pe-`` assert succeeds — otherwise pods keep billing after the
    ledger says released. Does not pass ``max_eval_nodes`` (own cap).
    """
    def _rent(name: str, ttl: str = '4h', *, gpu: str = 'B300', **_kw: Any) -> Mapping[str, Any]:
        if not str(name).startswith(PREFIX_SHADOW):
            raise LiveWiringError(
                'shadow rent name must start with %s' % PREFIX_SHADOW
            )
        if not callable(host_rent):
            raise LiveWiringError('shadow rent requires live pe_rent.rent')
        # Drop max_eval_nodes — shadow has its own cap (share_pe_cap=False).
        out = host_rent(name, ttl, gpu=gpu)
        if out is None:
            raise LiveWiringError('shadow host rent returned no box')
        d = dict(out)
        d['auto_backup_disabled'] = True
        d['shadow'] = True
        return d

    return _rent


def _bind_shared_cap_counters(rent: RentService, rent_mod: Any) -> None:
    """Live PIT + PE share one global eval-node cap (review-2 blocker 4).

    Shadow boxes use PREFIX_SHADOW and never enter this shared budget.
    """
    eval_nodes = getattr(rent_mod, 'eval_nodes', None)
    our_pods = getattr(rent_mod, 'our_pods', None) or getattr(rent_mod, 'list_pods', None)

    def _pe_eval_count() -> int:
        if callable(eval_nodes):
            try:
                nodes = list(eval_nodes())
            except TypeError:
                nodes = list(eval_nodes(None))
            warm_names = _warm_pod_names()
            suppress = _warm_suppressed()
            n = 0
            for p in nodes:
                name = ''
                if isinstance(p, Mapping):
                    name = str(p.get('name') or p.get('pod_name') or '')
                else:
                    name = str(p)
                # Legacy warm prefix always excluded from the PE eval bucket.
                if name.startswith('affine-warm-'):
                    continue
                # affine-pe-warm-*: normally counted via pe_warm_count_fn. Under
                # WARM-OFF / cutover the warm counter is zeroed, so count a live
                # one once here as an eval node (do not double-count via WARM.json).
                if name.startswith('affine-pe-warm-'):
                    if not suppress:
                        continue
                elif name in warm_names:
                    continue
                # Exclude live + shadow PIT (and legacy shadow prefix).
                # Exclude on-demand GREEN bench boxes (own cap 1; not PE eval).
                if (
                    name.startswith(PREFIX_LIVE)
                    or name.startswith(PREFIX_SHADOW)
                    or name.startswith('affine-pit-shadow-')
                    or name.startswith('affine-pit-')
                    or name.startswith('affine-pe-bench-')
                ):
                    continue
                n += 1
            return n
        if callable(our_pods):
            rows = list(our_pods() or [])
            return sum(
                1 for p in rows
                if str((p or {}).get('name') or '').startswith('affine-pe-')
                and not str((p or {}).get('name') or '').startswith('affine-pe-pit-')
                and not str((p or {}).get('name') or '').startswith('affine-pe-bench-')
            )
        return 0

    def _pe_warm_count() -> int:
        return len(_warm_active_pods(live_ids=_live_pod_ids(rent_mod)))

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


def _warm_active_pods(*, live_ids: Optional[set[str]] = None) -> list[str]:
    """Active WARM.json pods that still exist on Lium.

    Under WARM-OFF or live PIT cutover return [] so stale WARM.json rows cannot
    block shared-cap rentals. Otherwise only count entries whose ``pod.id`` is
    in the current live Lium id set.
    """
    if _warm_suppressed():
        return []
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
    if not isinstance(pods, Mapping):
        return out
    # No live listing → fail closed (do not invent warm pressure from stale JSON).
    if live_ids is None:
        return out
    for name, entry in pods.items():
        if not isinstance(entry, Mapping) or entry.get('state') not in active:
            continue
        pod = entry.get('pod') or {}
        pid = pod.get('id') if isinstance(pod, Mapping) else None
        if not pid or str(pid) not in live_ids:
            continue
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
    round_id_fn: Optional[Callable[[int], str]] = None
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
    host_push_bundle = _require_attr(lane_mod, ('push_bundle',), 'pe_lane push_bundle')
    host_serve = _require_attr(lane_mod, ('serve',), 'pe_lane serve')
    host_run_probe = _require_attr(lane_mod, ('run_probe',), 'pe_lane run_probe')
    # drive_item GRADER_FIX before probe (zero-sample stale grader uv symlinks).
    try:
        from ops.parallel_eval.live.pe_box import GRADER_FIX as _GRADER_FIX
    except Exception:
        try:
            import pe_box as _pe_box
            _GRADER_FIX = _pe_box.GRADER_FIX
        except Exception:
            _GRADER_FIX = (
                'find /opt/plain/probe/.cache/uv/environments-v2 -maxdepth 4 -type l '
                '-lname "/opt/affine-grader-uv/*" -delete 2>/dev/null; true'
            )

    host_score = getattr(score_mod, 'score', None)
    if not callable(host_score):
        raise LiveWiringError('pe_score.score required')
    # The baseline that the scorer judges a candidate against (the round baseline, or the baseline of the turn).
    scoring_baseline = getattr(score_mod, 'scoring_baseline', None)

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

    round_id_fn = getattr(seed_mod, 'round_id_for_epoch', None)
    if not callable(round_id_fn):
        raise LiveWiringError(
            'round seed provider must expose round_id_for_epoch '
            '(SAMPLE.private.json round_id; fail closed)'
        )

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
    shadow = bool(cfg.is_shadow())

    # Shadow: own prefix + own cap; never share the live PE 5-node budget.
    # Live pe_rent asserts affine-pe-*, so shadow rents via a thin lium wrapper.
    effective_host_rent = host_rent
    if shadow:
        effective_host_rent = _make_shadow_rent(rent_mod, host_rent)

    rent = RentService(
        max_boxes=cfg.max_boxes,
        max_eval_nodes=int(extra.get('max_eval_nodes') or cfg.max_boxes or 5),
        gpu_order=tuple(cfg.effective_gpu_order()),
        price_ceiling_usd_h=float(cfg.price_ceiling_usd_h),
        disable_auto_backup=bool(cfg.disable_lium_auto_backup),
        require_host=True,
        shadow=shadow,
        name_prefix=PREFIX_SHADOW if shadow else PREFIX_LIVE,
        share_pe_cap=not shadow,
        host_rent=effective_host_rent,
        host_release=host_release,
        list_pods=list_pods,
        ledger_path=ledger_path,
    )
    if not shadow:
        _bind_shared_cap_counters(rent, rent_mod)
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
        # drive_item: pe_lane.boot then push_bundle (podctl + vLLM env).
        # live pe_lane.boot returns None — coerce to a mapping for LaneService.
        P = _pod_of(pod)
        out = host_boot(P, **kw)
        host_push_bundle(P)
        return out if isinstance(out, Mapping) else dict(ok=True)

    def _serve(pod, cid, gmu='0.9', max_num_seqs=512, **kw):
        out = host_serve(_pod_of(pod), cid, gmu=gmu, max_num_seqs=max_num_seqs, **kw)
        return out if isinstance(out, Mapping) else dict(ok=True, candidate_id=cid)

    def _run_probe(pod, cid, indices, plan_seed, **kw):
        P = _pod_of(pod)
        # drive_item.probe: GRADER_FIX immediately before run_probe.
        try:
            P.run(_GRADER_FIX, check=False, timeout=120)
        except Exception:
            pass
        return host_run_probe(P, cid, indices, plan_seed, **kw)

    lane = LaneService(
        require_host=True,
        host_boot=_boot,
        host_serve=_serve,
        host_run_probe=_run_probe,
        host_release=getattr(lane_mod, 'release', None),
    )

    def score_fn(**kwargs: Any) -> Mapping[str, Any]:
        # Always call live score(epoch, rid, result) positionally via adapter.
        # The baseline gives its hardware and layout fingerprint. A candidate probe with a different
        # one, or a side with none, gets no decision (probe_fingerprint.py). This is not an audit
        # record: the scorer does not run.
        rid = str(kwargs.get('round_id') or kwargs.get('rid') or '')
        result = kwargs.get('result') or {}
        if not callable(scoring_baseline):
            # This pe_score module cannot show the baseline that it scores against: nothing is scored.
            return nest_heldout_fields(probe_fingerprint.refusal(None, result), round_id=rid)
        return call_live_host_score(
            host_score,
            epoch=int(kwargs['epoch']),
            round_id=rid,
            result=result,
            baseline_of=scoring_baseline,
        )

    rounds = None
    if cfg.is_live():
        # Explicit rounds= — same construction as gate_service.rounds()
        # (heldout.Rounds under HG_GATE_STATE/PE_GATE_STATE/PE_HELDOUT).
        rounds = build_host_rounds(
            gate_state=extra.get('gate_state_root') or os.environ.get('HG_GATE_STATE')
            or os.environ.get('PE_GATE_STATE'),
            heldout=extra.get('heldout_root') or os.environ.get('PE_HELDOUT'),
        )
    gate_writer = HostGateWriter(
        epochs_root=gate_epochs_root(extra.get('gate_epochs_root')),
        shadow=bool(cfg.is_shadow()),
        rounds=rounds,
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
        round_id_fn=round_id_fn,
        require_host=True,
        pe_apply_accept=pe_apply_accept,
        destroy_box_state_fn=destroy_fn if callable(destroy_fn) else None,
        upload_master=upload_master if callable(upload_master) else None,
        upload_candidate=upload_candidate if callable(upload_candidate) else None,
    )
    if validate_only:
        return bundle
    return bundle
