"""PIT configuration. Defaults match DESIGN §12 / BRIEF §1.8.

On-demand rental is a hard rule, not a config option.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

from . import (
    GATE_HEAD_ROLLOUTS,
    GATE_K_CAND,
    GATE_MAX_TOKENS,
    GATE_N_DRAWN,
    MAX_ATTEMPTS,
    MAX_BOXES_DEFAULT,
    MAX_SEQS,
    TRAIN_OUTPUT_TOKENS,
)

BASELINE_OPTION_A = 'one_secret_draw_per_head_epoch'
MASTER_UPLOAD_DURING_PROBE = 'during_probe'
GATE_DTYPE = 'bfloat16'


@dataclass(frozen=True)
class PitConfig:
    enabled: bool = False
    mode: str = 'shadow'  # only shadow authorized in this PR while enabled
    max_boxes: int = MAX_BOXES_DEFAULT
    master_upload: str = MASTER_UPLOAD_DURING_PROBE
    baseline: str = BASELINE_OPTION_A
    gpu_order: tuple = ('B300', 'B200', 'H200')
    max_attempts: int = MAX_ATTEMPTS
    decision_timeout_seconds: int = 21000
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
    trainer_fallback: str = 'trainer_pod'
    pe_rent_module: Optional[str] = None
    pe_lane_module: Optional[str] = None
    pe_score_module: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def box_cap(self) -> int:
        """Alias used by older call sites / docs."""
        return self.max_boxes

    def is_shadow(self) -> bool:
        return self.mode == 'shadow'

    def may_run(self) -> bool:
        return self.enabled is True and self.mode == 'shadow'

    def may_write_live_artifacts(self) -> bool:
        return False  # this PR never authorizes live artifact writes

    def as_dict(self) -> dict:
        d = asdict(self)
        d['gpu_order'] = list(self.gpu_order)
        d['extra'] = dict(self.extra)
        d['on_demand_rent_only'] = True  # hard rule, not configurable
        return d


DEFAULT_PIT = PitConfig()


def load_pit_config(parallel_eval: Optional[Mapping[str, Any]] = None) -> PitConfig:
    parallel_eval = parallel_eval or {}
    raw = parallel_eval.get('pit')
    if raw is None:
        return DEFAULT_PIT
    if not isinstance(raw, Mapping):
        raise TypeError('parallel_eval.pit must be a mapping')

    # Reject any attempt to enable a warm pool via config.
    for banned in ('warm_pool', 'warm_boxes', 'pre_rent', 'idle_pool', 'keep_warm'):
        if raw.get(banned):
            raise ValueError('PIT rents on demand only; %r is refused' % banned)

    known = {
        'enabled', 'mode', 'max_boxes', 'box_cap', 'master_upload', 'baseline',
        'gpu_order', 'max_attempts', 'decision_timeout_seconds', 'k_cand',
        'head_rollouts', 'n_drawn', 'max_output_tokens', 'train_output_tokens',
        'max_num_seqs', 'dtype', 'z_strict_gt', 'no_fp8', 'trainer_fallback',
        'pe_rent_module', 'pe_lane_module', 'pe_score_module', 'shadow',
    }
    kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in raw.items():
        if key == 'box_cap':
            kwargs['max_boxes'] = value
        elif key == 'shadow' and 'mode' not in raw:
            # Accept legacy {enabled, shadow:true} → mode=shadow
            if value is True and raw.get('enabled') is True:
                kwargs['mode'] = 'shadow'
            elif value is True and not raw.get('enabled'):
                kwargs.setdefault('mode', 'shadow')
        elif key in known and key not in ('box_cap', 'shadow'):
            kwargs[key] = value
        elif key not in known:
            extra[key] = value
    if extra:
        kwargs['extra'] = extra
    if 'gpu_order' in kwargs and not isinstance(kwargs['gpu_order'], tuple):
        kwargs['gpu_order'] = tuple(kwargs['gpu_order'])

    cfg = PitConfig(**kwargs)

    if cfg.enabled and cfg.mode != 'shadow':
        raise ValueError('only mode=shadow is authorized in this PR; refuse mode=%r' % (cfg.mode,))
    if cfg.max_boxes < 1:
        raise ValueError('max_boxes must be >= 1')
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
        raise ValueError('this PR only ships baseline option A')
    if cfg.master_upload not in ('during_probe', 'after_green'):
        raise ValueError('master_upload must be during_probe|after_green')
    return cfg
