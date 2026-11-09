"""Per-item train+eval (PIT) — serverless rent-per-job.

DESIGN: uploads/DESIGN.public.md. BRIEF: uploads/IMPLEMENTATION-BRIEF.public.md.
SPEC: uploads/SPEC.md (serverless).

Hard rule (owner): boxes are rented **only on demand** for a bound queue item.
No warm pool, no idle/pre-rented machines. A box is released as soon as its
item's result is stored or it fails.

Live defaults (SPEC): max_boxes=5, B300 pinned (no GPU fallback), price ceiling
$56/h, masters upload only on GREEN_FRESH, REDs earn nothing under the
last-10-GREEN rule.
"""
# Live pe_rent asserts name.startswith("affine-pe-"). PIT boxes use this
# sub-prefix so they share the PE rental path without colliding with lanes.
PREFIX = 'affine-pe-pit-'
MAX_BOXES_SHADOW = 2
MAX_BOXES_LIVE = 5
MAX_BOXES_DEFAULT = MAX_BOXES_SHADOW  # disabled/shadow default; live overrides to 5
GPU_PINNED = ('B300',)  # SPEC: candidates + baselines pinned; no B200/H200 fallback
GPU_ORDER = GPU_PINNED  # alias kept for older call sites
PRICE_CEILING_USD_H = 56.0
TRAIN_OUTPUT_TOKENS = 32768
GATE_MAX_TOKENS = 65536
GATE_K_CAND = 8
GATE_HEAD_ROLLOUTS = 12
GATE_N_DRAWN = 1000
MAX_SEQS = 512
MAX_ATTEMPTS = 3
ADMISSION_VERSION = 'pit-box-admission-v1'
DECISION_LOG_VERSION = 'pit-decision-log-v1'
TRAINER_FALLBACK_HOURS = 24
BENCH_PROFILE = 'qwen3.5-family-pp0'

# Per-phase stall timeouts (seconds). Watchdog uses these; not knobs for gate math.
STALL = {
    'renting': 600,
    'provisioning': 1800,
    'training': 1200,
    'sealed': 900,
    'serving': 1200,
    'probing': 4500,
    'result': 300,
    'scored': 600,
    'decided': 300,
}

from .config import PitConfig, load_pit_config, DEFAULT_PIT
from .state import ItemState, ItemRecord, TransitionError, apply_transition, ALLOWED
from .shadow import ShadowGuard, ShadowWriteError, FORBIDDEN_SHADOW_WRITES

__all__ = [
    'PREFIX', 'MAX_BOXES_DEFAULT', 'MAX_BOXES_SHADOW', 'MAX_BOXES_LIVE',
    'GPU_ORDER', 'GPU_PINNED', 'PRICE_CEILING_USD_H', 'TRAIN_OUTPUT_TOKENS',
    'GATE_MAX_TOKENS', 'GATE_K_CAND', 'GATE_HEAD_ROLLOUTS', 'GATE_N_DRAWN',
    'MAX_SEQS', 'MAX_ATTEMPTS', 'ADMISSION_VERSION', 'DECISION_LOG_VERSION',
    'STALL', 'TRAINER_FALLBACK_HOURS', 'BENCH_PROFILE',
    'PitConfig', 'load_pit_config', 'DEFAULT_PIT',
    'ItemState', 'ItemRecord', 'TransitionError', 'apply_transition', 'ALLOWED',
    'ShadowGuard', 'ShadowWriteError', 'FORBIDDEN_SHADOW_WRITES',
]
