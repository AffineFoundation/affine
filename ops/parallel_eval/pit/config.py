"""PIT configuration. Defaults match DESIGN §12 recommended options.

Never enable live writes from this module. ``enabled`` defaults to false and
``mode`` defaults to ``shadow``. Shadow is the only mode this PR may run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


# Gate math is frozen — do not change these without an explicit operator decision.
GATE_K_CAND = 8
GATE_HEAD_ROLLOUTS = 12
GATE_N_DRAWN = 1000
GATE_MAX_OUTPUT_TOKENS = 65536
GATE_DTYPE = 'bfloat16'
GATE_Z_STRICT_GT = True  # strict z > 1
GATE_NO_FP8 = True

BASELINE_OPTION_A = 'one_secret_draw_per_head_epoch'
ADMISSION_SCHEME = 'ephemeral_ed25519_root_cert'
MC_UPLOAD = 'during_probe'
TRAINER_FALLBACK = 'trainer_pod'


@dataclass(frozen=True)
class PitConfig:
    """Immutable PIT settings loaded from ``parallel_eval.pit``."""

    enabled: bool = False
    mode: str = 'shadow'  # only 'shadow' is allowed while enabled in this PR
    box_cap: int = 2
    baseline: str = BASELINE_OPTION_A
    admission: str = ADMISSION_SCHEME
    master_upload: str = MC_UPLOAD
    trainer_fallback: str = TRAINER_FALLBACK
    # Gate replica — read-only mirrors of live settings; not knobs to retune.
    k_cand: int = GATE_K_CAND
    head_rollouts: int = GATE_HEAD_ROLLOUTS
    n_drawn: int = GATE_N_DRAWN
    max_output_tokens: int = GATE_MAX_OUTPUT_TOKENS
    dtype: str = GATE_DTYPE
    z_strict_gt: bool = GATE_Z_STRICT_GT
    no_fp8: bool = GATE_NO_FP8
    # Infra retry
    max_rent_attempts: int = 3
    max_infra_retries: int = 2
    rent_backoff_s: float = 30.0
    # Host adapters (optional absolute paths; unset -> injected stubs / missing)
    pe_rent_module: Optional[str] = None
    pe_lane_module: Optional[str] = None
    pe_score_module: Optional[str] = None
    pe_live_module: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def is_shadow(self) -> bool:
        return self.mode == 'shadow'

    def may_run(self) -> bool:
        """True only when the operator explicitly enabled PIT in shadow mode."""
        return self.enabled is True and self.mode == 'shadow'

    def may_write_live_artifacts(self) -> bool:
        """Shadow (and disabled) never write DECISION / weights / HEAD-CHAIN / R2 heads."""
        return False  # this PR never authorizes live artifact writes

    def as_dict(self) -> dict:
        d = asdict(self)
        d['extra'] = dict(self.extra)
        return d


DEFAULT_PIT = PitConfig()


def load_pit_config(parallel_eval: Optional[Mapping[str, Any]] = None) -> PitConfig:
    """Load PIT settings from a ``parallel_eval`` flag block.

    Unknown keys are kept in ``extra``. Invalid ``mode`` while enabled raises.
    Gate math fields are accepted only when they match the frozen live values.
    """
    parallel_eval = parallel_eval or {}
    raw = parallel_eval.get('pit')
    if raw is None:
        return DEFAULT_PIT
    if not isinstance(raw, Mapping):
        raise TypeError('parallel_eval.pit must be a mapping')

    known = {
        'enabled', 'mode', 'box_cap', 'baseline', 'admission', 'master_upload',
        'trainer_fallback', 'k_cand', 'head_rollouts', 'n_drawn', 'max_output_tokens',
        'dtype', 'z_strict_gt', 'no_fp8', 'max_rent_attempts', 'max_infra_retries',
        'rent_backoff_s', 'pe_rent_module', 'pe_lane_module', 'pe_score_module',
        'pe_live_module',
    }
    kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in raw.items():
        if key in known:
            kwargs[key] = value
        else:
            extra[key] = value
    if extra:
        kwargs['extra'] = extra

    cfg = PitConfig(**kwargs)

    if cfg.enabled and cfg.mode != 'shadow':
        raise ValueError('only mode=shadow is authorized in this PR; refuse mode=%r' % (cfg.mode,))
    if cfg.box_cap < 1:
        raise ValueError('box_cap must be >= 1')
    if cfg.k_cand != GATE_K_CAND:
        raise ValueError('refuse gate math change: k_cand must be %d' % GATE_K_CAND)
    if cfg.head_rollouts != GATE_HEAD_ROLLOUTS:
        raise ValueError('refuse gate math change: head_rollouts must be %d' % GATE_HEAD_ROLLOUTS)
    if cfg.n_drawn != GATE_N_DRAWN:
        raise ValueError('refuse gate math change: n_drawn must be %d' % GATE_N_DRAWN)
    if cfg.max_output_tokens != GATE_MAX_OUTPUT_TOKENS:
        raise ValueError('refuse gate math change: max_output_tokens must be %d' % GATE_MAX_OUTPUT_TOKENS)
    if cfg.dtype != GATE_DTYPE:
        raise ValueError('refuse gate math change: dtype must be %s' % GATE_DTYPE)
    if cfg.z_strict_gt is not True:
        raise ValueError('refuse gate math change: z must stay strict > 1')
    if cfg.no_fp8 is not True:
        raise ValueError('refuse gate math change: fp8 remains disallowed')
    if cfg.baseline != BASELINE_OPTION_A:
        raise ValueError('this PR only ships baseline option A (%s)' % BASELINE_OPTION_A)
    return cfg
