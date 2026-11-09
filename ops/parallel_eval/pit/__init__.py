"""Per-item train+eval (PIT) — shadow-first, disabled by default.

DESIGN: uploads/DESIGN.public.md. BRIEF: uploads/IMPLEMENTATION-BRIEF.public.md.

Hard rule (owner): boxes are rented **only on demand** for a bound queue item.
No warm pool, no idle/pre-rented machines. A box is released as soon as its
item's result is stored or the item fails.
"""
PREFIX = 'affine-pit-'
MAX_BOXES_DEFAULT = 2
GPU_ORDER = ('B300', 'B200', 'H200')
TRAIN_OUTPUT_TOKENS = 32768
GATE_MAX_TOKENS = 65536
GATE_K_CAND = 8
GATE_HEAD_ROLLOUTS = 12
GATE_N_DRAWN = 1000
MAX_SEQS = 512
MAX_ATTEMPTS = 3
ADMISSION_VERSION = 'pit-box-admission-v1'
DECISION_LOG_VERSION = 'pit-decision-log-v1'

# Per-phase stall timeouts (seconds). Watchdog uses these; not knobs for gate math.
STALL = {
    'renting': 600,
    'provisioning': 1800,
    'training': 3600,
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
    'PREFIX', 'MAX_BOXES_DEFAULT', 'GPU_ORDER', 'TRAIN_OUTPUT_TOKENS',
    'GATE_MAX_TOKENS', 'GATE_K_CAND', 'GATE_HEAD_ROLLOUTS', 'GATE_N_DRAWN',
    'MAX_SEQS', 'MAX_ATTEMPTS', 'ADMISSION_VERSION', 'DECISION_LOG_VERSION', 'STALL',
    'PitConfig', 'load_pit_config', 'DEFAULT_PIT',
    'ItemState', 'ItemRecord', 'TransitionError', 'apply_transition', 'ALLOWED',
    'ShadowGuard', 'ShadowWriteError', 'FORBIDDEN_SHADOW_WRITES',
]
