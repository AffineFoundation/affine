"""duel_turns — the production duel-corpus turn view, as seen from the pod.

Since the trace-first cutover (2026-09-02) the view itself lives in
affine.corpus.view (duel_turns@v4: one record per rollout holding the
baked message graph + turn metas). The pod does not store turns any more;
it derives them here only to count yield for the scheduler (kept turns per
source / policy) and to keep the local Parquet index honest. The fold on
the validator box runs the very same builder over the published traces, so
what the pod counts is what D will contain.

Identity contract (reproduces both legacy pipelines exactly):

    slice instance_id   task["sid"]
    repo                task["repo"]
    model               policy["model"] (provider-qualified label)
    run_tag             trace.info.run_tag when the adapter stamped one
                        (mini_swe: sha256 of the raw traj file), else
                        sha256 of the sorted-keys trace dump (verifiers)
    generated_at        trace.info.generated_at when stamped (mini_swe:
                        file mtime), else the caller-supplied batch stamp

Resolved score is telemetry, not a keep-gate (Reason v3 policy).
"""

from __future__ import annotations

import logging

from affine.corpus.trace import ToolParityError, TraceShapeError
from affine.corpus.view import (
    VIEW_SPEC,
    build_view_record,
    validate_turns,
    view_turns,
)

from rollouts.panel import PanelKeys, panel_drop, panel_keys
from rollouts.schema import utc_now_iso

__all__ = ["VIEW_SPEC", "derive_turns", "validate_records",
           "derive_and_validate"]

log = logging.getLogger("rollouts.views.duel_turns")


def derive_turns(envelope: dict, *, panel: PanelKeys | None = None,
                 generated_at: str | None = None, baker=None) -> list[dict]:
    """Turn records from one envelope. Errored rollouts, panel-excluded
    tasks, tool traces that fail template parity (or arrive without a
    ToolBaker) and unwalkable graphs derive nothing; every record is
    tagged source + language, and carries the catalog row's explicit
    `stratum` when it has one."""
    panel = panel or panel_keys()
    task = envelope["task"]
    if panel_drop(task["uid"], task.get("repo") or "", panel):
        return []
    try:
        record = build_view_record(
            envelope, baker=baker,
            generated_at=generated_at or utc_now_iso())
    except (ToolParityError, TraceShapeError) as e:
        log.warning("%s: %s — trajectory dropped", task["uid"], e)
        return []
    if record is None:
        return []
    return view_turns(record)


def validate_records(records: list[dict], panel: PanelKeys | None = None,
                     ) -> tuple[list[dict], dict[str, int]]:
    """Fold admission contract, staging flavour: every REGISTERED dialect
    is admitted (the [dataset].allowed_action_kinds gate lives in the fold,
    corpus_push and the duel tripwire), so a not-yet-admitted dialect
    still counts toward yield during a fork's notice period."""
    return validate_turns(records, panel=panel or panel_keys(),
                          allowed_kinds=None)


def derive_and_validate(envelopes: list[dict],
                        panel: PanelKeys | None = None,
                        generated_at: str | None = None,
                        baker=None) -> tuple[list[dict], dict[str, int]]:
    """Derive + validate across envelopes."""
    panel = panel or panel_keys()
    records: list[dict] = []
    for env in envelopes:
        records.extend(derive_turns(env, panel=panel,
                                    generated_at=generated_at, baker=baker))
    return validate_records(records, panel)
