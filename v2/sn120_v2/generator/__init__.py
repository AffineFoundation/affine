"""SN120 mechanism 1: Generator protocol primitives.

Public page: https://affine.io/generator. This package has no controller,
chain writer, deployment configuration or production-pool writer.
"""

from .constants import (
    ACCEPT,
    CONTINUE,
    KING,
    MAX_SAMPLES,
    MAX_SUBMISSION_BYTES,
    MECH_ID,
    NETUID,
    POOL_ID,
    PUBLIC_NAME,
    PUBLIC_URL,
    REJECT,
    SAMPLING,
    SCHEMA_VERSION,
)
from .decisions import Decision, decision, informativeness, q, q_exact
from .epoch import epoch_bounds, epoch_contains, epoch_deadline, epoch_id, epoch_start, layout
from .protocol import Submission, canonical_submission_hash, sign_submission, signing_message, verify_submission

__all__ = [
    "ACCEPT",
    "CONTINUE",
    "Decision",
    "KING",
    "MAX_SAMPLES",
    "MAX_SUBMISSION_BYTES",
    "MECH_ID",
    "NETUID",
    "POOL_ID",
    "PUBLIC_NAME",
    "PUBLIC_URL",
    "REJECT",
    "SAMPLING",
    "SCHEMA_VERSION",
    "Submission",
    "canonical_submission_hash",
    "decision",
    "epoch_bounds",
    "epoch_contains",
    "epoch_deadline",
    "epoch_id",
    "epoch_start",
    "informativeness",
    "layout",
    "q",
    "q_exact",
    "sign_submission",
    "signing_message",
    "verify_submission",
]
