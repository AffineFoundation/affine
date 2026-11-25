"""Old-code execution guard for reviewed KL history. Never edits a journal."""
from __future__ import annotations

from .errors import V2Error
from .params import KL_PARAMS_HASH


def require_closed(state, *, transition: bool = False) -> None:
    """Old code may read closed KL history, never finish or open a gated round."""
    record = state.round
    if record is not None and record.params_hash == KL_PARAMS_HASH:
        stage = "decided" if record.decision is not None else "undecided"
        raise V2Error(
            f"rollback blocked: {record.round_id} is {stage} under KL params; "
            "finish it with the reviewed KL checkout before rollback"
        )
    from . import params
    if (
        not transition
        and not params.ROLLBACK_TRANSITION
        and state.params is not None
        and getattr(state.params, "_rollback_raw_params_hash", state.params.params_hash)
        == KL_PARAMS_HASH
    ):
        raise V2Error(
            "rollback blocked: KL params are still pinned; apply the complete old "
            "params with this compatibility loader before starting the old loop"
        )
