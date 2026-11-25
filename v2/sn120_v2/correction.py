"""Operator-approved offline correction of a confirmation infrastructure failure.

The command line is the only writer. Both it and journal replay use
``check_correction``: a malformed or unauthorized correction is never paid.
The original confirmation is retained as evidence, not edited.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import sha256_hex
from .errors import ContractError, JournalCorrupt, V2Error
from .judging import TIE_TOLERANCE, score_confirmation
from .params import Params
from .recheck import VERDICT_SCHEMA, bundle_sha256, load_bundle
from .types import NOT_RUN, BoxResult, Confirmation, ConfirmationCorrection, Draw, King, PendingKing, RoundTicket, check_outcomes

REASON = "offline re-check after infra no_decision"
# Only failures to obtain or compare parent-side fresh evidence are eligible.
# A new-head loss, tie or scoring failure is never in this set.
INFRA_REASON_PREFIXES = (
    "no_decision: parent scoring failed:",
    "no_decision: parent scoring is unavailable",
    "no_decision: no fresh parent result arrived",
    "no_decision: the pending crown's parent manifest is unavailable",
    "no_decision: fresh parent comparison failed (",
    "no_decision: the new-head null did not complete the full fresh draw",
)
REQUIRED_FIELDS = frozenset({
    "schema", "round_id", "crown_round_id", "uid", "hotkey", "new_version_id", "parent_version_id",
    "delta", "se", "z", "new_pass_rate", "old_pass_rate", "n_tasks", "confirmed",
    "null_outcomes", "parent_outcomes", "null_samples", "parent_samples", "sanity",
    "code_digest", "confirmation_code_digest", "code_identity_matches", "git_head",
    "excluded_training_prompts", "existing_confirmation", "bundle_sha256",
})


def evidence_rounds(state: Any) -> list[Any]:
    """The bounded public history plus the permanent, bounded-size correction index."""
    records = list(state.rounds)
    known = {record.round_id for record in records}
    records.extend(record for round_id, record in sorted(state.correction_evidence.items())
                   if round_id not in known)
    return records


@dataclass(frozen=True)
class Evidence:
    raw: bytes
    verdict: dict[str, Any]
    verdict_sha256: str
    bundle_sha256: str
    body: dict[str, Any]
    ticket: RoundTicket
    null_outcomes: tuple[str, ...]


def read_evidence(verdict_path: str | Path, bundle_path: str | Path | None = None) -> Evidence:
    """Read the real writer's verdict and verify its self-contained sealed bundle.

    ``--bundle`` is explicit. With only ``--verdict``, ``bundle.json`` beside
    it is read. Neither path is taken from the verdict's content.
    """
    verdict_path = Path(verdict_path)
    try:
        raw = verdict_path.read_bytes()
        verdict = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise V2Error(f"cannot read approved verdict {verdict_path}: {exc}") from exc
    if not isinstance(verdict, dict):
        raise V2Error("approved verdict must be one JSON object")
    missing = sorted(REQUIRED_FIELDS - verdict.keys())
    if missing:
        raise V2Error(f"approved verdict is missing required fields: {', '.join(missing)}")
    if verdict["schema"] != VERDICT_SCHEMA:
        raise V2Error(f"approved verdict schema must be {VERDICT_SCHEMA}")
    for name in ("sanity", "confirmed", "code_identity_matches"):
        _field(verdict, name, bool)
    if verdict["sanity"]:
        raise V2Error("a sanity-prefix offline re-check cannot correct a confirmation")
    path = Path(bundle_path) if bundle_path is not None else verdict_path.parent / "bundle.json"
    body, ticket, null_outcomes = load_bundle(path)
    digest = bundle_sha256(body)
    if _field(verdict, "bundle_sha256", str) != digest:
        raise V2Error("the evidence bundle does not match the approved verdict's canonical bundle_sha256")
    pending = ticket.parent.pending if ticket.parent is not None else None
    if pending is None:
        raise V2Error("the evidence bundle has no original pending crown")
    for name, expected in (
        ("round_id", ticket.round_id),
        ("crown_round_id", pending.crown_round_id),
        ("uid", pending.uid),
        ("hotkey", pending.hotkey),
        ("new_version_id", pending.new_version_id),
        ("parent_version_id", pending.parent_version_id),
        ("null_outcomes", list(null_outcomes)),
        ("n_tasks", len(ticket.draw.task_ids)),
        ("null_samples", ticket.params.null_samples),
        ("existing_confirmation", body["confirmation_resolved"]),
        ("excluded_training_prompts", body["exclusions"]["excluded_training_prompts"]),
        ("confirmation_code_digest", ticket.params.recipe.code_digest),
    ):
        if verdict[name] != expected:
            raise V2Error(f"approved verdict {name} does not match its verified evidence bundle")
    return Evidence(raw, verdict, sha256_hex(raw), digest, body, ticket, null_outcomes)


def _field(data: dict[str, Any], name: str, kind: type) -> Any:
    value = data[name]
    if type(value) is not kind:
        raise V2Error(f"approved verdict {name} must be {kind.__name__}")
    return value


def prepare(state: Any, evidence: Evidence, draw: Draw, approved_by: str, original_params: Params, original_result: BoxResult) -> ConfirmationCorrection | None:
    """Build a correction, or return None for an identical repeat."""
    data = evidence.verdict
    uid = _field(data, "uid", int)
    confirmed = _field(data, "confirmed", bool)
    identities = {name: _field(data, name, str) for name in ("crown_round_id", "hotkey", "new_version_id", "parent_version_id")}
    for name in ("delta", "se", "z"):
        if type(data[name]) not in (int, float) or not math.isfinite(data[name]):
            raise V2Error(f"approved verdict {name} must be a finite number")
    _field(data, "n_tasks", int)
    identity = (identities["crown_round_id"], uid, identities["parent_version_id"], identities["new_version_id"])
    resolved = [record for record in evidence_rounds(state) if record.confirmation is not None
                and (record.confirmation.pending_round_id, record.confirmation.pending_uid,
                     record.confirmation.parent_version_id, record.confirmation.new_version_id) == identity]
    if len(resolved) != 1 or resolved[0].parent is None:
        raise V2Error("the target crown needs one confirmation round with its original pinned pending crown")
    record = resolved[0]
    pending = record.parent.pending
    if _field(data, "existing_confirmation", dict) != record.confirmation.to_dict():
        raise V2Error("approved verdict does not preserve the original confirmation_resolved")
    if (pending.crown_round_id, pending.uid, pending.hotkey, pending.parent_version_id, pending.new_version_id) != (
            identities["crown_round_id"], uid, identities["hotkey"], identities["parent_version_id"], identities["new_version_id"]):
        raise V2Error("approved verdict does not match the original pending crown")
    new_outcomes = tuple(_field(data, "null_outcomes", list))
    parent_outcomes = tuple(_field(data, "parent_outcomes", list))
    null_samples = _field(data, "null_samples", int)
    parent_samples = _field(data, "parent_samples", int)
    if null_samples != len(new_outcomes) or parent_samples != len(parent_outcomes):
        raise V2Error("approved verdict sample counts do not match its outcome strings")
    if any(type(text) is not str for text in (*new_outcomes, *parent_outcomes)):
        raise V2Error("approved verdict outcome entries must be strings")
    try:
        check_outcomes(new_outcomes, len(draw.task_ids), "offline new head")
        check_outcomes(parent_outcomes, len(draw.task_ids), "offline parent")
        if any(NOT_RUN in text for text in (*new_outcomes, *parent_outcomes)):
            raise ContractError("offline re-check outcomes must cover every task; NOT_RUN is not evidence")
        scored = score_confirmation(draw.weights, new_outcomes, parent_outcomes)
        for name in ("delta", "se", "z", "n_tasks"):
            if data[name] != scored[name]:
                raise V2Error(f"approved verdict {name} does not match the recomputed paired result")
        if not confirmed or confirmed != scored["confirmed"]:
            raise V2Error("approved verdict must be confirmed with delta above TIE_TOLERANCE")
        verdict = Confirmation(
            parent_version_id=pending.parent_version_id, pending_round_id=pending.crown_round_id,
            pending_uid=uid, new_version_id=pending.new_version_id,
            delta=scored["delta"], se=scored["se"], z=scored["z"],
            new_pass_rate=scored["new_pass_rate"], old_pass_rate=scored["old_pass_rate"],
            n_tasks=scored["n_tasks"], confirmed=True, reason=REASON,
            excluded_training_prompts=_field(data, "excluded_training_prompts", int),
        )
        correction = ConfirmationCorrection(
            pending=pending,
            king=King(pending.uid, pending.hotkey, pending.crown_round_id, pending.new_version_id, pending.applied_at, pending.z),
            confirmation_round_id=record.round_id, original=record.confirmation,
            verdict=verdict, draw=draw, original_params=original_params, draw_sha256=draw.digest(),
            new_outcomes=new_outcomes, parent_outcomes=parent_outcomes,
            original_new_outcomes=original_result.null_outcomes,
            original_parent_outcomes=original_result.parent_outcomes,
            original_result_sha256=original_result.digest(),
            verdict_sha256=evidence.verdict_sha256, bundle_sha256=evidence.bundle_sha256,
            code_digest=_field(data, "code_digest", str), git_HEAD=_field(data, "git_head", str),
            approved_by=approved_by.strip(),
        )
    except ContractError as exc:
        raise V2Error(f"approved verdict cannot form a correction: {exc}") from exc
    previous = next((item for item in state.corrections if item.pending.crown_round_id == pending.crown_round_id), None)
    if previous is not None:
        if previous != correction or state.correction_draws.get(record.round_id) != draw.digest():
            raise V2Error("this crown already has a different confirmation correction; it cannot be replaced")
        return None
    if state.correction_draws.get(record.round_id) not in (None, draw.digest()):
        raise V2Error("the stored confirmation draw changed; no correction was written")
    check_correction(state, correction)
    return correction


def check_correction(state: Any, correction: ConfirmationCorrection) -> None:
    """Fail closed on identity, original outcome, verdict and complete paired evidence."""
    pending, verdict, original = correction.pending, correction.verdict, correction.original
    identity = (pending.crown_round_id, pending.uid, pending.parent_version_id, pending.new_version_id)
    crown = next((record for record in evidence_rounds(state) if record.round_id == pending.crown_round_id and record.applied), None)
    winner = crown.decision.winner if crown is not None and crown.decision is not None else None
    if winner is None or (winner.uid, winner.hotkey) != (pending.uid, pending.hotkey):
        raise V2Error("the target crown is not the applied winner named by the approved verdict")
    head = next((head for head in state.heads if head.version_id == pending.new_version_id), None)
    if head is None:
        raise V2Error("the target new head is not in the journal")
    if (head.parent_version_id, head.round_id) != (pending.parent_version_id, pending.crown_round_id):
        raise V2Error("the target head and parent do not match the journal's applied crown")
    if any(king.round_id == pending.crown_round_id and king.uid == pending.uid for king in state.kings):
        raise V2Error("the target crown is already paid")
    if pending.batch_sha256 != winner.batch_sha256 or pending.z != (winner.score.z if winner.score is not None else 0.0):
        raise V2Error("the correction does not preserve the crown's winner identity")
    if pending.applied_at != head.applied_at:
        raise V2Error("the correction does not preserve the crown's install time")
    resolved = [record for record in evidence_rounds(state) if record.confirmation is not None
                and (record.confirmation.pending_round_id, record.confirmation.pending_uid,
                     record.confirmation.parent_version_id, record.confirmation.new_version_id) == identity]
    if len(resolved) != 1:
        raise V2Error("the target crown needs exactly one confirmation_resolved in a retained confirmation round")
    record = resolved[0]
    if record.parent is None or record.parent.pending != pending:
        raise V2Error("the correction does not preserve the original pinned pending crown")
    if record.outcome != "null":
        raise V2Error("only a confirming null round with no promoted replacement may be corrected")
    original = record.confirmation
    assert original is not None
    if original.confirmed or not original.reason.startswith("no_decision"):
        raise V2Error("only an unconfirmed confirmation no_decision may be corrected")
    if not original.reason.startswith(INFRA_REASON_PREFIXES):
        raise V2Error(f"the confirmation no_decision is not an allowed parent-infrastructure failure: {original.reason}")
    if not verdict.confirmed or verdict.delta <= TIE_TOLERANCE:
        raise V2Error("an approved offline re-check must confirm with delta above TIE_TOLERANCE")
    if (verdict.pending_round_id, verdict.pending_uid, verdict.parent_version_id, verdict.new_version_id) != identity:
        raise V2Error("approved verdict identities do not match the target crown")
    if (correction.confirmation_round_id, correction.original) != (record.round_id, original):
        raise V2Error("correction does not preserve the original confirmation and its round")
    if record.round_id in state.correction_evidence and state.correction_evidence[record.round_id] != record:
        raise JournalCorrupt("the correction evidence index does not preserve its original confirmation round")
    for crown_record in (crown, state.correction_evidence.get(pending.crown_round_id)):
        if crown_record is not None and crown_record.decision.winner != winner:
            raise JournalCorrupt("the correction evidence index does not preserve its original crown winner")
    if correction.draw_sha256 != correction.draw.digest():
        raise JournalCorrupt("correction draw_sha256 does not match its retained draw")
    if (correction.original_params.params_hash != record.params_hash
            or correction.original_params.recipe_id != record.recipe_id):
        raise JournalCorrupt("correction does not retain the original confirmation ticket's parameters")
    if (len(correction.new_outcomes) != correction.original_params.null_samples
            or len(correction.parent_outcomes) != correction.original_params.null_samples):
        raise V2Error("offline re-check must retain exactly the original ticket's new and parent sample counts")
    if original.reason.startswith("no_decision: fresh parent comparison failed (") and not correction.original_parent_outcomes:
        raise V2Error("paired-scoring no_decision must retain the original parent's outcome evidence")
    if (correction.draw.snapshot_id, correction.draw.round_seed, len(correction.draw.task_ids)) != (
            record.snapshot_id, record.round_seed, record.n_tasks) or not record.leases:
        raise JournalCorrupt("correction draw does not match the original confirmation round")
    if state.correction_draws.get(record.round_id) not in (None, correction.draw_sha256):
        raise JournalCorrupt("a second confirmation correction changes the round's retained draw")
    try:
        check_outcomes(correction.new_outcomes, record.n_tasks, "offline new head")
        check_outcomes(correction.parent_outcomes, record.n_tasks, "offline parent")
        if any(NOT_RUN in text for text in (*correction.new_outcomes, *correction.parent_outcomes)):
            raise ContractError("offline re-check outcomes must cover every task; NOT_RUN is not evidence")
        check_outcomes(correction.original_new_outcomes, record.n_tasks, "original new head")
        if (len(correction.original_new_outcomes) != correction.original_params.null_samples
                or any(NOT_RUN in text for text in correction.original_new_outcomes)):
            raise ContractError("the original new head did not complete the full fresh draw")
        if correction.new_outcomes != correction.original_new_outcomes:
            raise JournalCorrupt("correction does not preserve the original sealed new-head outcomes")
        if correction.original_parent_outcomes and correction.parent_outcomes != correction.original_parent_outcomes:
            raise JournalCorrupt("correction does not preserve the original sealed parent outcomes")
        scored = score_confirmation(correction.draw.weights, correction.new_outcomes, correction.parent_outcomes)
    except (ContractError, TypeError) as exc:
        raise V2Error(f"approved verdict's paired outcome strings do not pass sanity checks: {exc}") from exc
    if not scored["confirmed"] or verdict.n_tasks != scored["n_tasks"] or verdict.n_tasks != record.n_tasks:
        raise V2Error("approved verdict must confirm over every task of the confirmation draw")
    for name in ("delta", "se", "z", "new_pass_rate", "old_pass_rate"):
        if getattr(verdict, name) != scored[name]:
            raise V2Error(f"approved verdict {name} does not match its paired outcome strings")
    if len(record.leases) != 1:
        raise JournalCorrupt("correction needs one unambiguous parent-bearing sealed ticket")
    if record.leases[0].result_digest != correction.original_result_sha256:
        raise JournalCorrupt("correction does not preserve the original sealed result digest")
    if verdict.excluded_training_prompts != original.excluded_training_prompts:
        raise JournalCorrupt("correction does not preserve the original training-prompt exclusions")
    if verdict.reason != REASON:
        raise V2Error("correction verdict reason is not the offline re-check reason")
    if correction.king != King(pending.uid, pending.hotkey, pending.crown_round_id, pending.new_version_id, pending.applied_at, pending.z):
        raise V2Error("correction king does not match the target crown")
    if any(item.pending.crown_round_id == pending.crown_round_id for item in state.corrections):
        raise JournalCorrupt("a crown has more than one confirmation correction")
