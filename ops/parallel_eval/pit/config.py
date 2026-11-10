"""PIT configuration. Defaults match DESIGN §12 / BRIEF §1.8 / SPEC §3–§4.

On-demand rental is a hard rule, not a config option. Live mode is authorized
with box_cap default 5, B300-first then cheapest allowed GPU under $56/h,
masters after GREEN only.

Live ROOT path (owner 2026-11-09): ``root_mode=delegated_host`` only.
``per_round`` remains an unused alternative and is refused in live.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

from . import (
    ALLOW_SHARDED_GPUS_DEFAULT,
    ALLOWED_GPUS,
    GATE_HEAD_ROLLOUTS,
    GATE_K_CAND,
    GATE_MAX_TOKENS,
    GATE_N_DRAWN,
    GPU_ORDER,
    MAX_ATTEMPTS,
    MAX_BOXES_LIVE,
    MAX_BOXES_SHADOW,
    MAX_SEQS,
    PRICE_CEILING_USD_H,
    SHARDED_GPUS,
    TRAIN_OUTPUT_TOKENS,
    TRAINER_FALLBACK_HOURS,
)
from .signer import ALT_ROOT_MODE, LIVE_ROOT_MODE

BASELINE_OPTION_A = 'one_secret_draw_per_head_epoch'
# SPEC §3.1: masters upload only on GREEN (during_probe dropped).
MASTER_UPLOAD_AFTER_GREEN = 'after_green'
GATE_DTYPE = 'bfloat16'
AUTHORIZED_MODES = frozenset({'shadow', 'live'})


@dataclass(frozen=True)
class PitConfig:
    enabled: bool = False
    mode: str = 'shadow'
    max_boxes: int = MAX_BOXES_SHADOW
    # Lifecycle cutover backpressure: max PIT-DISPATCHED without PIT-RESULT.
    # None ⇒ same as max_boxes (pe_lifecycle reads the flag directly).
    max_pending: Optional[int] = None
    master_upload: str = MASTER_UPLOAD_AFTER_GREEN
    baseline: str = BASELINE_OPTION_A
    gpu_order: tuple = GPU_ORDER
    # PROMOTION-GATE: when false (default), drop RTX5090/H100 from the effective
    # rental order even if they appear in gpu_order. Enable after first live ZeRO-3 test.
    allow_sharded_gpus: bool = ALLOW_SHARDED_GPUS_DEFAULT
    max_attempts: int = MAX_ATTEMPTS
    decision_timeout_seconds: int = 21000
    price_ceiling_usd_h: float = PRICE_CEILING_USD_H
    # Gate replica — frozen.
    k_cand: int = GATE_K_CAND
    head_rollouts: int = GATE_HEAD_ROLLOUTS
    n_drawn: int = GATE_N_DRAWN
    max_output_tokens: int = GATE_MAX_TOKENS
    train_output_tokens: int = TRAIN_OUTPUT_TOKENS
    max_num_seqs: int = MAX_SEQS
    dtype: str = GATE_DTYPE
    z_strict_gt: bool = True
    no_fp8: bool = True
    # Cutover (SPEC §7 step 6): trainer stays 24h as hot fallback; lanes/warm off.
    cutover: bool = False
    revert: bool = False
    trainer_fallback_hours: int = TRAINER_FALLBACK_HOURS
    disable_warm_pool: bool = True
    disable_standing_lanes: bool = True
    disable_trainer_dequeue: bool = True
    disable_lium_auto_backup: bool = True
    full_redo_after_sealed: bool = True  # SPEC D3
    reds_earn: bool = False  # last-10-GREEN rule: REDs earn nothing
    # Live default: delegated host key (ROOT signs host key once).
    root_mode: str = LIVE_ROOT_MODE
    pe_rent_module: Optional[str] = None
    pe_lane_module: Optional[str] = None
    pe_score_module: Optional[str] = None
    root_signer_module: Optional[str] = None
    # On-demand GREEN bench (replaces standing affine-bench daemon). Default on.
    bench_on_green: bool = True
    # Root for publish_fleet input (`$BENCH_RESULTS_DIR`); no host-path default.
    bench_results_dir: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def box_cap(self) -> int:
        return self.max_boxes

    def is_shadow(self) -> bool:
        return self.mode == 'shadow'

    def is_live(self) -> bool:
        return self.mode == 'live'

    def may_run(self) -> bool:
        return self.enabled is True and self.mode in AUTHORIZED_MODES

    def may_write_live_artifacts(self) -> bool:
        return self.enabled is True and self.mode == 'live' and not self.revert

    def cutover_active(self) -> bool:
        """Trainer dequeue / warm / standing lanes off when live cutover is on."""
        if self.revert:
            return False
        return self.enabled and self.mode == 'live' and self.cutover

    def effective_gpu_order(self) -> tuple:
        """gpu_order with SHARDED_GPUS removed unless allow_sharded_gpus is true."""
        if self.allow_sharded_gpus:
            return tuple(self.gpu_order)
        blocked = set(SHARDED_GPUS)
        out = tuple(g for g in self.gpu_order if g not in blocked)
        if not out:
            raise ValueError('effective gpu_order empty after sharded-GPU gate')
        return out

    def as_dict(self) -> dict:
        d = asdict(self)
        d['gpu_order'] = list(self.gpu_order)
        d['effective_gpu_order'] = list(self.effective_gpu_order())
        d['extra'] = dict(self.extra)
        d['on_demand_rent_only'] = True
        d['non_item_jobs_outside_cap'] = True
        d['gpu_pinned'] = list(self.effective_gpu_order())
        return d


DEFAULT_PIT = PitConfig()


def load_pit_config(parallel_eval: Optional[Mapping[str, Any]] = None) -> PitConfig:
    parallel_eval = parallel_eval or {}
    raw = parallel_eval.get('pit')
    if raw is None:
        return DEFAULT_PIT
    if not isinstance(raw, Mapping):
        raise TypeError('parallel_eval.pit must be a mapping')

    for banned in ('warm_pool', 'warm_boxes', 'pre_rent', 'idle_pool', 'keep_warm'):
        if raw.get(banned):
            raise ValueError('PIT rents on demand only; %r is refused' % banned)

    known = {
        'enabled', 'mode', 'max_boxes', 'box_cap', 'max_pending', 'master_upload',
        'baseline',
        'gpu_order', 'allow_sharded_gpus', 'max_attempts', 'decision_timeout_seconds',
        'price_ceiling_usd_h',
        'k_cand', 'head_rollouts', 'n_drawn', 'max_output_tokens', 'train_output_tokens',
        'max_num_seqs', 'dtype', 'z_strict_gt', 'no_fp8', 'cutover', 'revert',
        'trainer_fallback_hours', 'disable_warm_pool', 'disable_standing_lanes',
        'disable_trainer_dequeue', 'disable_lium_auto_backup', 'full_redo_after_sealed',
        'reds_earn', 'root_mode', 'pe_rent_module', 'pe_lane_module', 'pe_score_module',
        'root_signer_module', 'shadow', 'trainer_fallback', 'bench_on_green',
        'bench_results_dir',
    }
    kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in raw.items():
        if key == 'box_cap':
            kwargs['max_boxes'] = value
        elif key == 'shadow' and 'mode' not in raw:
            if value is True:
                kwargs['mode'] = 'shadow'
        elif key == 'trainer_fallback':
            # Accept legacy string; live cutover uses trainer_fallback_hours.
            extra[key] = value
        elif key in known and key not in ('box_cap', 'shadow', 'trainer_fallback'):
            kwargs[key] = value
        elif key not in known:
            extra[key] = value
    if extra:
        kwargs['extra'] = extra
    if 'gpu_order' in kwargs and not isinstance(kwargs['gpu_order'], tuple):
        kwargs['gpu_order'] = tuple(kwargs['gpu_order'])

    # Mode-aware max_boxes default: live → 5 unless explicitly set.
    if 'max_boxes' not in kwargs:
        mode = kwargs.get('mode', 'shadow')
        kwargs['max_boxes'] = MAX_BOXES_LIVE if mode == 'live' else MAX_BOXES_SHADOW

    cfg = PitConfig(**kwargs)

    if cfg.enabled and cfg.mode not in AUTHORIZED_MODES:
        raise ValueError('mode must be shadow|live; refuse mode=%r' % (cfg.mode,))
    if cfg.revert and cfg.cutover:
        raise ValueError('revert and cutover cannot both be true')
    if cfg.max_boxes < 1:
        raise ValueError('max_boxes must be >= 1')
    if cfg.max_pending is not None and int(cfg.max_pending) < 1:
        raise ValueError('max_pending must be >= 1')
    if cfg.mode == 'live' and cfg.max_boxes > MAX_BOXES_LIVE:
        raise ValueError('live max_boxes cannot exceed %d' % MAX_BOXES_LIVE)
    if float(cfg.price_ceiling_usd_h) > PRICE_CEILING_USD_H + 1e-9:
        raise ValueError('price_ceiling_usd_h cannot exceed %.1f' % PRICE_CEILING_USD_H)
    if float(cfg.price_ceiling_usd_h) <= 0:
        raise ValueError('price_ceiling_usd_h must be > 0')
    order = tuple(cfg.gpu_order)
    if not order:
        raise ValueError('gpu_order must be non-empty')
    if len(set(order)) != len(order):
        raise ValueError('gpu_order must not contain duplicates')
    bad = [g for g in order if g not in ALLOWED_GPUS]
    if bad:
        raise ValueError(
            'gpu_order entries must be in %s; refuse %r'
            % (list(ALLOWED_GPUS), bad)
        )
    if cfg.k_cand != GATE_K_CAND:
        raise ValueError('refuse gate math change: k_cand')
    if cfg.head_rollouts != GATE_HEAD_ROLLOUTS:
        raise ValueError('refuse gate math change: head_rollouts')
    if cfg.n_drawn != GATE_N_DRAWN:
        raise ValueError('refuse gate math change: n_drawn')
    if cfg.max_output_tokens != GATE_MAX_TOKENS:
        raise ValueError('refuse gate math change: max_output_tokens')
    if cfg.dtype != GATE_DTYPE:
        raise ValueError('refuse gate math change: dtype')
    if cfg.z_strict_gt is not True or cfg.no_fp8 is not True:
        raise ValueError('refuse gate math change: z/fp8')
    if cfg.baseline != BASELINE_OPTION_A:
        raise ValueError('baseline must be option A (secret draw job per head/round)')
    if cfg.master_upload not in ('after_green', 'during_probe'):
        raise ValueError('master_upload must be after_green|during_probe')
    if cfg.mode == 'live' and cfg.master_upload != MASTER_UPLOAD_AFTER_GREEN:
        raise ValueError('live mode requires master_upload=after_green (SPEC §3.1)')
    if cfg.reds_earn is not False:
        raise ValueError('REDs earn nothing under the last-10-GREEN rule')
    if cfg.disable_lium_auto_backup is not True:
        raise ValueError('Lium auto-backup must stay disabled on every rental')
    # Live ROOT: delegated host key is the only live path (owner 2026-11-09).
    if cfg.mode == 'live' and cfg.enabled and cfg.root_mode != LIVE_ROOT_MODE:
        raise ValueError(
            'live ROOT path is %r only; refused root_mode=%r (unused alternative %r)'
            % (LIVE_ROOT_MODE, cfg.root_mode, ALT_ROOT_MODE)
        )
    if cfg.root_mode not in (LIVE_ROOT_MODE, ALT_ROOT_MODE):
        raise ValueError('root_mode must be delegated_host|per_round; got %r' % (cfg.root_mode,))
    return cfg
