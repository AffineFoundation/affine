"""Per-item train+eval (PIT) — serverless rent-per-job.

DESIGN: uploads/DESIGN.public.md. BRIEF: uploads/IMPLEMENTATION-BRIEF.public.md.
SPEC: uploads/SPEC.md (serverless).

Hard rule (owner): boxes are rented **only on demand** for a bound queue item.
No warm pool, no idle/pre-rented machines. A box is released as soon as its
item's result is stored or it fails.

Live defaults (SPEC): max_boxes=5, B300-first then cheapest allowed GPU under
$56/h (exactly 8 GPUs), masters upload only on GREEN_FRESH, REDs earn nothing
under the last-10-GREEN rule.
"""
# Live pe_rent asserts name.startswith("affine-pe-") for BOTH rent and release.
# Live PIT boxes use this sub-prefix so they share the PE rental path + global
# eval-node cap. Shadow must ALSO satisfy affine-pe-* or pe_rent.release refuses
# the name and the box keeps billing while the ledger says released.
PREFIX = 'affine-pe-pit-'
PREFIX_LIVE = PREFIX
# Shadow: affine-pe-pit-shadow-* (own cap; still releaseable via live pe_rent).
PREFIX_SHADOW = 'affine-pe-pit-shadow-'
MAX_BOXES_SHADOW = 2
MAX_BOXES_LIVE = 5
MAX_BOXES_DEFAULT = MAX_BOXES_SHADOW  # disabled/shadow default; live overrides to 5
# Allowed Lium classes (class name 'RTX5090'). Prefer B300 when ≤ ceiling; else
# cheapest full 8-GPU offer of any allowed type under the ceiling.
ALLOWED_GPUS = ('B300', 'RTX5090', 'B200', 'H200', 'H100')
GPU_ORDER = ('B300', 'RTX5090', 'B200', 'H200', 'H100')
GPU_PINNED = GPU_ORDER  # alias kept for older call sites
PRICE_CEILING_USD_H = 56.0
REQUIRED_GPU_COUNT = 8
# Per-GPU HBM below this → shard/offload fp32 accumulator (ZeRO-style).
LOW_MEM_GPU_GIB = 100.0
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
# First job on a new sm_120 (RTX5090) class: kernel compile is slow.
STALL_FIRST_COMPILE = {
    'renting': 900,
    'provisioning': 3600,
    'training': 3600,
    'sealed': 1800,
    'serving': 2400,
    'probing': 7200,
    'result': 600,
    'scored': 900,
    'decided': 600,
}

from .config import PitConfig, load_pit_config, DEFAULT_PIT
from .state import ItemState, ItemRecord, TransitionError, apply_transition, ALLOWED
from .shadow import ShadowGuard, ShadowWriteError, FORBIDDEN_SHADOW_WRITES

__all__ = [
    'PREFIX', 'PREFIX_LIVE', 'PREFIX_SHADOW',
    'MAX_BOXES_DEFAULT', 'MAX_BOXES_SHADOW', 'MAX_BOXES_LIVE',
    'GPU_ORDER', 'GPU_PINNED', 'PRICE_CEILING_USD_H', 'TRAIN_OUTPUT_TOKENS',
    'GATE_MAX_TOKENS', 'GATE_K_CAND', 'GATE_HEAD_ROLLOUTS', 'GATE_N_DRAWN',
    'MAX_SEQS', 'MAX_ATTEMPTS', 'ADMISSION_VERSION', 'DECISION_LOG_VERSION',
    'STALL', 'STALL_FIRST_COMPILE', 'TRAINER_FALLBACK_HOURS', 'BENCH_PROFILE',
    'ALLOWED_GPUS', 'REQUIRED_GPU_COUNT', 'LOW_MEM_GPU_GIB',
    'PitConfig', 'load_pit_config', 'DEFAULT_PIT',
    'ItemState', 'ItemRecord', 'TransitionError', 'apply_transition', 'ALLOWED',
    'ShadowGuard', 'ShadowWriteError', 'FORBIDDEN_SHADOW_WRITES',
]
