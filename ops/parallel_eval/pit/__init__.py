"""Per-item train+eval (PIT) — shadow-first, disabled by default.

Each queued registration rents its own GPU box. The box downloads the bound
head and fp32 master, trains with trainer-v834e, then runs the live-gate probe
on the same box. The host scores the probe and records incentives. The first
GREEN for a head becomes the new head; later GREENs bound to an older head are
paid but merge_blocked and never applied.

Everything ships behind ``parallel_eval.pit.enabled=false``. The only enabled
mode in this PR is ``mode=shadow``, which never writes DECISION.json, weights,
HEAD-CHAIN, or R2 head pointers.
"""
from .config import PitConfig, load_pit_config, DEFAULT_PIT
from .state import ItemState, ItemRecord, TransitionError, apply_transition, ALLOWED
from .dequeue import Binding, bind_item, is_stale_green, order_greens
from .decision_log import DecisionLog, DecisionEntry
from .shadow import ShadowGuard, ShadowWriteError, FORBIDDEN_SHADOW_WRITES
from .admission import (
    AdmissionCert,
    AdmissionRequest,
    EphemeralBoxKey,
    verify_admission_cert,
    build_admission_request,
    PLACEHOLDER_ROOT_PUBKEY_HEX,
)

__all__ = [
    'PitConfig',
    'load_pit_config',
    'DEFAULT_PIT',
    'ItemState',
    'ItemRecord',
    'TransitionError',
    'apply_transition',
    'ALLOWED',
    'Binding',
    'bind_item',
    'is_stale_green',
    'order_greens',
    'DecisionLog',
    'DecisionEntry',
    'ShadowGuard',
    'ShadowWriteError',
    'FORBIDDEN_SHADOW_WRITES',
    'AdmissionCert',
    'AdmissionRequest',
    'EphemeralBoxKey',
    'verify_admission_cert',
    'build_admission_request',
    'PLACEHOLDER_ROOT_PUBKEY_HEX',
]
