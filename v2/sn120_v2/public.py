"""The public view. One plain dict feeds llms.txt, mining.json, the website and the benchmark service.

Privacy rule: an entry that did not win shows its seat, its batch sha256,
its status, delta, se, z and cap rate, and its sealed eval outcome strings.
Its file, its location and its train values are never published. Only a
winner's LoRA and training batch are public.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import math
import os
import re
import statistics
import time
from collections import Counter
from typing import Any, Mapping, Sequence

from .canonical import is_sha256, to_jsonable
from .errors import V2Error
from .params import Params
from .payout import paid_set, payees
from .queue import waiting
from .state import APPLIED, RoundRecord, State
from .types import KL_FIELDS, NOT_SCORED, NOT_TRAINED, WINNER, BoxResult, ChainSnapshot, EntryDecision, King, QueueEntry, RoundTicket, Verdict

SCHEMA = "sn120-v2-public-1"
WINNER_STATS_SCHEMA = "sn120-v2-winner-stats-1"
QUEUE_PREVIEW = 50
HISTORY_PREVIEW = 20
SERVICE_STATES = frozenset({"running", "applying", "idle", "stalled"})
SERVICE_REASONS = frozenset({"waiting for seats", "publish retrying", "deploying fix", "between rounds"})
NEXT_OPEN_COARSE_S = 300
NEXT_OPEN_WINDOW_S = 300
VALID, NOT_VALID, NOT_JUDGED = "valid", "not_valid", "not_judged"  # ``file_verdict`` in ``queue.files``, and the name of its count

# ``public_text``: the rule for a text that an exception made, before it goes into a public file.
TEXT_LIMIT = 600  # characters. The longest text today has about 520: a failed round that quotes the last line of a job log.
SECRET_NAME = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|ENDPOINT|ACCOUNT", re.IGNORECASE)
SECRET_MIN = 8  # a shorter value is a flag or a number (TOKENIZERS_PARALLELISM=false): to replace it would cut plain words
CUT_SECRET, CUT_LOGIN, CUT_QUERY, CUT_ADDRESS = "[removed-secret]", "[removed-login]", "[removed-query]", "[removed-address]"
_LOGIN = re.compile(r"(?<=://)[^\s/@?#]+@")
_URL_QUERY = re.compile(r"(\b[A-Za-z][A-Za-z0-9+.-]*://[^\s?'\"<>]*)\?[^\s'\"<>]*")
_QUERY = re.compile(r"\?[\w.~%\[\]-]+=[^\s'\"<>]*")
_NAMED = re.compile(
    r"(?i)\b((?:[\w-]*[_-])?(?:api[_-]?key|access[_-]?key|secret|signature|credential|password|passwd|token|authorization)(?:[_-][\w-]*)?"
    r"[\"']?\s*[:=]\s*[\"']?(?:(?:bearer|basic)\s+)?)(?!\[removed-)[^\s,;&'\"<>)}\]]+"
)
_V6 = re.compile(r"(?<![\w:.])[0-9A-Fa-f:]*:[0-9A-Fa-f:.]*[0-9A-Fa-f:](?![\w:])")
_V4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d)")


def _address(match: re.Match[str]) -> str:
    found = match.group(0)
    try:
        ipaddress.ip_address(found)
    except ValueError:  # a time of day, a version number, a "::" of a test name
        return found
    return CUT_ADDRESS if any(char.isalnum() for char in found) else found


def public_text(text: object, environ: Mapping[str, str] | None = None) -> str:
    """A text from an exception, made safe for a public file. Every such text passes here before it is published.

    The error class and the message stay, so a miner can read the cause of a
    failure that its own file made. These parts are replaced by a marker:

    * each value of an environment variable of this process whose name says
      that it is a credential or a store location (``SECRET_NAME``);
    * the login of a URL (``user:password@``) and each URL query string: a
      signed link holds its signature there;
    * the value after a name such as ``token=`` or ``Signature:``;
    * each IPv4 and IPv6 address. A port stays.

    Then the text is cut to ``TEXT_LIMIT`` characters. The journal and the
    sealed box result keep the text as it was: they are not public.
    """
    text = " ".join(str(text).split())
    environ = os.environ if environ is None else environ
    secrets = {value.strip() for name, value in environ.items() if SECRET_NAME.search(name)}
    for value in sorted((value for value in secrets if len(value) >= SECRET_MIN), key=len, reverse=True):
        text = text.replace(value, CUT_SECRET)
    text = _LOGIN.sub(CUT_LOGIN + "@", text)
    text = _URL_QUERY.sub(lambda match: f"{match.group(1)}?{CUT_QUERY}", text)
    text = _QUERY.sub("?" + CUT_QUERY, text)
    text = _NAMED.sub(lambda match: match.group(1) + CUT_SECRET, text)
    text = _V4.sub(_address, _V6.sub(_address, text))
    if len(text) > TEXT_LIMIT:
        tail = f" [cut: {len(text) - TEXT_LIMIT} more characters]"
        text = text[:TEXT_LIMIT] + tail
    return text


def entry_view(entry: EntryDecision) -> dict[str, Any]:
    view: dict[str, Any] = {
        "queue_pos": entry.queue_pos,
        "uid": entry.uid,
        "hotkey": entry.hotkey,
        "registration_block": entry.registration_block,
        "batch_sha256": entry.batch_sha256,
        "status": entry.status,
        "eliminated_at_look": entry.eliminated_at_look,
    }
    if entry.score is not None:
        score = entry.score
        view |= {"delta": score.delta, "se": score.se, "z": score.z, "cap_rate": score.cap_rate, "n_tasks": score.n_tasks}
    view |= {name: value for name in ("kl_seq", "kl_seq_se", "kl_tok", "kl_k", "kl_tokens") if (value := getattr(entry, name)) is not None}
    if entry.status == WINNER:
        view |= {"lora_hash": entry.lora_hash, "pass_rate": entry.score.pass_rate, "null_pass_rate": entry.score.null_pass_rate}
    return view


def phase(state: State) -> str:
    """IDLE: no round. RUNNING: boxes train and score. APPLY: decided, the head and files are being published."""
    if state.round is None:
        return "IDLE"
    return "RUNNING" if state.round.decision is None else "APPLY"


def service_view(
    state: State, now: int, *, stop_asked: bool = False, close_publish_pending: bool = False, waiting_for_seats: bool | None = None,
) -> dict[str, Any]:
    """Coarse service facts only. The caller supplies flags, not errors, file checks or scheduler timings.

    ``close_publish_pending`` is set by ``close_in_public`` when a recorded close still has public
    work undone. ``waiting_for_seats`` is the actual file-valid admission result; an unknown
    count conservatively means waiting, never a new file read or scheduler prediction.
    """
    phase_now = phase(state)
    if phase_now in ("RUNNING", "APPLY"):
        return {"state": {"RUNNING": "running", "APPLY": "applying"}[phase_now], "reason": None, "next_open_at": None}
    if stop_asked:
        reason = "deploying fix"
    elif close_publish_pending:
        reason = "publish retrying"
    else:
        reason = "between rounds" if waiting_for_seats is False else "waiting for seats"
    service_state = "stalled" if close_publish_pending else "idle"
    next_open = None
    if reason in ("deploying fix", "publish retrying", "between rounds"):
        # A coarse five-minute window, never a scheduler or retry deadline.
        estimate = ((now + NEXT_OPEN_WINDOW_S + NEXT_OPEN_COARSE_S - 1) // NEXT_OPEN_COARSE_S) * NEXT_OPEN_COARSE_S
        next_open = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(estimate))
    return {"state": service_state, "reason": reason, "next_open_at": next_open}


def confirmation_view(confirmation: Any) -> dict[str, Any]:
    """The same paid/unpaid paired result in round.json and mining.json round/history."""
    return confirmation.to_dict() | {
        "new_head_pass_rate": confirmation.new_pass_rate,
        "old_head_pass_rate": confirmation.old_pass_rate,
        "paid": confirmation.confirmed,
        "payout_status": "paid" if confirmation.confirmed else "unpaid",
        "reason": public_text(confirmation.reason),
    }


def round_view(record: RoundRecord) -> dict[str, Any]:
    seats = [entry for lease in record.leases for entry in lease.entries]
    view: dict[str, Any] = {
        "round_id": record.round_id,
        "head_version_id": record.head_version_id,
        "params_hash": record.params_hash,
        "recipe_id": record.recipe_id,
        "round_seed": record.round_seed,
        "irt_snapshot_id": record.snapshot_id,
        "n_tasks": record.n_tasks,
        "opened_at": record.opened_at,
        "closed_at": record.closed_at,
        "outcome": record.outcome,
        "seats": len(seats) + len(record.duplicate_decisions),
        "skipped": len(record.skipped),
        "duplicate_clusters": list(record.duplicate_clusters),
        "duplicates": [entry_view(entry) | {"reason": public_text(entry.reason)} for entry in record.duplicate_decisions],
        **({"confirmation": confirmation_view(record.confirmation)} if record.confirmation is not None else {}),
    }
    if record.decision is None:
        view["entries"] = sorted([
            {"queue_pos": e.queue_pos, "uid": e.reg.uid, "hotkey": e.reg.hotkey, "registration_block": e.reg.registration_block, "batch_sha256": e.batch.batch_sha256}
            for e in seats
        ] + [entry_view(entry) for entry in record.duplicate_decisions], key=lambda row: row["queue_pos"])
    else:
        decision = record.decision
        view["entries"] = [entry_view(entry) for entry in decision.entries]
        view |= {"winner_uid": decision.winner.uid if decision.winner else None, "reason": public_text(decision.reason), "null_pass_rate": decision.null_pass_rate, "null_cap_rate": decision.null_cap_rate}
        view["duplicate_clusters"] = [*record.duplicate_clusters, *decision.duplicate_clusters]
        view["duplicates"] = [entry_view(entry) | {"reason": public_text(entry.reason)} for entry in decision.entries if entry.status == "duplicate"]
    if record.rotation is not None:
        from . import layout

        block = record.rotation
        view["pool"] = {key: block[key] for key in ("active_sha", "admitted", "retired", "reserve_manifest_sha", "queue_remaining", "next_up")}
        view["pool"] |= {"size": block["size"], "active": layout.active_pool(block["active_sha"]),
                         "rotation": layout.pool_rotation(record.round_id), "retired_archive": layout.RETIRED_POOL,
                         "reserve_manifest": layout.RESERVE_MANIFEST}
        if block.get("rule_version") in ("affine-pool-rotation-v2", "affine-pool-rotation-v3"):
            view["pool"] |= {"rule_version": block["rule_version"], "ranking": layout.pool_ranking(record.round_id),
                             "summary": layout.POOL_SUMMARY, "rule_params": block.get("rule_params") or {},
                             "retire_reason_counts": block.get("retire_reason_counts") or {}}
            if block["rule_version"] == "affine-pool-rotation-v3":
                view["pool"] |= {"guard_ledger": layout.pool_guard(block["guard_ledger_sha"]),
                                 **{key: block[key] for key in ("guard_ledger_sha", "guard_ledger_count", "guard_pass",
                                                                 "guard_reject", "guard_shortfall", "next_up_eligible")}}
    return view


def file_verdict(verdict: Verdict | None) -> str:
    """The last check of a hotkey's stored file, as one of three texts.

    ``file_valid`` is false for a file that breaks a rule and also when
    there is no verdict. This text tells the two apart. There is no verdict
    when no file was stored at the check, when the seat came after the
    check, and before the first check of a new process.
    """
    if verdict is None:
        return NOT_JUDGED
    return VALID if verdict.valid else NOT_VALID


def _file_facts(verdict: Verdict | None) -> dict[str, Any]:
    """SHA and the verdict's own Unix UTC time only: no file bytes, location, rule or counts."""
    sha = verdict.batch_sha256 if verdict is not None and is_sha256(verdict.batch_sha256) else None
    return {"batch_sha256": sha, "checked_at": verdict.checked_at if verdict is not None and sha and verdict.checked_at > 0 else None}


def line_view(line: Sequence[QueueEntry], verdicts: Mapping[str, Verdict], batch_cap: int) -> list[dict[str, Any]]:
    """Every waiting seat, in the real waiting order, with only its own last-check file facts.

    ``valid_ahead`` excludes this seat and every leased seat (they are not in ``line``). It counts
    valid-file waiting seats strictly ahead, regardless of hold flags. ``eta_rounds`` is only the
    public-parameter ceiling ``valid_ahead / batch_cap``: it does not predict a scheduler decision,
    an exact check/start time, or a task draw.
    """
    ahead = 0
    rows = []
    for position, seat in enumerate(line):
        verdict = verdicts.get(seat.reg.hotkey)
        valid = bool(verdict and verdict.valid)
        rows.append({
            "position": position,
            "uid": seat.reg.uid,
            "hotkey": seat.reg.hotkey,
            "registration_block": seat.reg.registration_block,
            "file_valid": valid,
            "file_verdict": file_verdict(verdict),
            **_file_facts(verdict),
            "front": seat.front,
            "demoted": seat.demoted,
            "held": seat.held,
            "valid_ahead": ahead,
            "eta_rounds": math.ceil(ahead / batch_cap) if batch_cap > 0 else None,
        })
        ahead += valid
    return rows


def files_view(line: Sequence[QueueEntry], verdicts: Mapping[str, Verdict], checked_at: int | None, batch_cap: int = 0) -> dict[str, Any]:
    """``queue.files``: one UID-ordered row per hotkey, plus counts over the whole waiting line.

    The full waiting order is in ``queue.line``; these rows deliberately do not imply that UID order
    is queue order. For a hotkey with multiple seats, ``valid_ahead``/``eta_rounds`` belong to its
    earliest waiting seat. No file reason, rule, counts, format, URI, contents or URL is exposed.
    """
    ordered = line_view(line, verdicts, batch_cap)
    earliest: dict[str, dict[str, Any]] = {}
    for row in ordered:
        earliest.setdefault(row["hotkey"], row)
    uids: dict[str, int] = {}
    for seat in line:
        uids[seat.reg.hotkey] = min(seat.reg.uid, uids.get(seat.reg.hotkey, seat.reg.uid))
    rows = [
        {"uid": uid, "hotkey": hotkey,
         "file_valid": earliest[hotkey]["file_valid"], "file_verdict": earliest[hotkey]["file_verdict"],
         "batch_sha256": earliest[hotkey]["batch_sha256"], "checked_at": earliest[hotkey]["checked_at"],
         "valid_ahead": earliest[hotkey]["valid_ahead"], "eta_rounds": earliest[hotkey]["eta_rounds"]}
        for uid, hotkey in sorted((uid, hotkey) for hotkey, uid in uids.items())
    ]
    counts = Counter(row["file_verdict"] for row in rows)
    return {
        "checked_at": checked_at,
        "seats": len(line),
        "hotkeys": len(rows),
        VALID: counts[VALID],
        NOT_VALID: counts[NOT_VALID],
        NOT_JUDGED: counts[NOT_JUDGED],
        "verdicts": rows,
    }


def pending_registrations(state: State, chain: ChainSnapshot | None, params: Params) -> list[dict[str, Any]]:
    """Chain registrations which have neither a waiting seat nor a seat leased in the open round.

    This is a view of the controller's latest complete, validated chain read. It adds no chain I/O,
    no new queue, and no journal access. A pending row has no file facts because it has no waiting seat.
    """
    if chain is None or chain.netuid != params.netuid:
        return []
    return [
        {"uid": reg.uid, "hotkey": reg.hotkey, "registration_block": reg.registration_block, "status": "pending"}
        for reg in sorted(chain.registrations, key=lambda reg: (reg.registration_block, reg.uid))
        if reg.uid != params.burn_uid and not any(reg.key == seat.reg.key for seat in state.queue.values())
        and reg.key not in state.served_regs and reg.hotkey not in state.served_hotkeys
    ]


def visible_params(params: Params) -> dict[str, Any]:
    """Public display form; canonical ``to_dict`` remains the hash/stored form.

    A reader may need the effective defaults even though old records omit them.
    Feeding this projection back through either Params decoder restores exactly
    the same canonical record and therefore the same params_hash and recipe_id.
    """
    shown = params.to_dict()
    shown.setdefault("max_stages", params.max_stages)
    recipe = dict(shown.get("recipe") or {})
    recipe.setdefault("sft_weight", params.recipe.sft_weight)
    shown["recipe"] = recipe
    return shown


def snapshot(
    state: State, params: Params, now: int, verdicts: Mapping[str, Verdict] | None = None, files_checked_at: int | None = None,
    chain: ChainSnapshot | None = None, *, stop_asked: bool = False, close_publish_pending: bool = False,
    waiting_for_seats: bool | None = None,
) -> dict[str, Any]:
    """Everything a miner, the website or the benchmark service may see.

    ``verdicts`` is admission's last check of the waiting files, taken at
    ``files_checked_at``. A file that a miner stored after that time shows
    the old verdict until the next check.

    ``queue.next`` is the first ``QUEUE_PREVIEW`` seats. ``queue.line`` is the full ordered
    waiting line with SHA/check-time-only file facts. ``queue.files`` gives UID-ordered counts.
    ``chain`` is the controller's already-read ChainSnapshot, never a new chain request.
    """
    verdicts = verdicts or {}
    line = line_view(waiting(state.queue), verdicts, params.batch_cap)
    return {
        "schema": SCHEMA,
        "generated_at": now,
        "netuid": params.netuid,
        "env": params.env,
        "phase": phase(state),
        "service": service_view(state, now, stop_asked=stop_asked, close_publish_pending=close_publish_pending,
                                waiting_for_seats=waiting_for_seats),
        "head": state.manifest.to_dict() if state.manifest else None,
        "heads": [{"version_id": h.version_id, "round_id": h.round_id, "applied_at": h.applied_at, "folded": h.folded} for h in state.heads[-HISTORY_PREVIEW:]],
        "params": visible_params(params),
        "params_hash": params.params_hash,
        "recipe_id": params.recipe_id,
        "irt_snapshot_id": state.snapshot_id,
        "round": round_view(state.round) if state.round else None,
        "history": [round_view(record) for record in state.rounds[-HISTORY_PREVIEW:]],
        "queue": {
            "length": len(line),
            "files_checked_at": files_checked_at,
            "next": line[:QUEUE_PREVIEW],
            "line": line,
            "pending": pending_registrations(state, chain, params),
            "files": files_view(waiting(state.queue), verdicts, files_checked_at, params.batch_cap),
        },
        "kings": [king.to_dict() for king in payees(state.kings, params.kings_window)],
        "paid_kings": [{"uid": king.uid, "round_id": king.round_id} for king in paid_set(state.kings, state.null_rounds, params)],
        "confirmations": confirmation_stats(state),
        "null_prune_from_round": params.null_prune_from_round,
    }



def _crowned_winner(record: RoundRecord) -> EntryDecision | None:
    """An actually crowned/applied king, never a named winner of an abandoned or null round."""
    return record.decision.winner if record.outcome == APPLIED and record.decision is not None else None


CONFIRMATION_BANDS = (
    ("z < 1.0", None, 1.0),
    ("1.0 <= z < 1.25", 1.0, 1.25),
    ("1.25 <= z < 1.5", 1.25, 1.5),
    ("1.5 <= z < 2.0", 1.5, 2.0),
    ("z >= 2.0", 2.0, None),
)


def confirmation_stats(state: State) -> dict[str, Any]:
    """Every applied crown under next-round confirmation, newest first; no loser or waiting data.

    Pre-rule crowns are omitted: a crown is in scope only when a later journal record resolved it, or
    when it is the state's current pending crown. The replayed closed rounds are not shortened to the
    public snapshot's preview. Resolved rows reuse ``confirmation_view`` for the existing paid flag and
    payout status. An infrastructure ``no_decision`` is not a failed paired test and enters no rate.
    """
    resolved = {record.confirmation.pending_round_id: (record.round_id, record.confirmation)
                for record in state.rounds if record.confirmation is not None}
    under_rule = set(resolved)
    if state.pending_king is not None:
        under_rule.add(state.pending_king.crown_round_id)
    rows = []
    for record in reversed(state.rounds):
        winner = _crowned_winner(record)
        if winner is None or record.round_id not in under_rule:
            continue
        resolution = resolved.get(record.round_id)
        if resolution is None:
            # With the confirmation rule active, only the current crown can still be awaiting its next round.
            if state.pending_king is None or record.round_id != state.pending_king.crown_round_id:
                continue
            status, confirmed, confirmation_round_id = "pending", None, None
            confirmation_z = confirmation_delta = None
            paid = False
        else:
            confirmation_round_id, confirmation = resolution
            shown = confirmation_view(confirmation)
            paid = shown["paid"]
            no_decision = confirmation.reason.startswith("no_decision") or confirmation.n_tasks == 0
            status = "no_decision" if no_decision else shown["payout_status"]
            confirmed = confirmation.confirmed
            confirmation_z = None if no_decision else confirmation.z
            confirmation_delta = None if no_decision else confirmation.delta
        rows.append({
            "round_id": record.round_id, "uid": winner.uid, "hotkey": winner.hotkey,
            "z": winner.score.z, "delta": winner.score.delta, "status": status,
            "confirmation_round_id": confirmation_round_id, "confirmed": confirmed,
            "confirmation_z": confirmation_z, "confirmation_delta": confirmation_delta, "paid": paid,
        })

    tested = [row for row in rows if row["status"] in ("paid", "unpaid")]

    def aggregate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        paid = sum(row["paid"] for row in items)
        return {"n": len(items), "paid": paid, "rate": paid / len(items) if items else None}

    bands = []
    for band, lo, hi in CONFIRMATION_BANDS:
        bands.append({"band": band, "lo": lo, "hi": hi,
                      **aggregate([row for row in tested if (lo is None or row["z"] >= lo) and (hi is None or row["z"] < hi)])})
    cumulative = [{"min_z": min_z, **aggregate([row for row in tested if row["z"] >= min_z])}
                  for min_z in (0.5, 1.0, 1.25, 1.5, 2.0)]
    paid_z = [row["z"] for row in tested if row["paid"]]
    unpaid_z = [row["z"] for row in tested if row["status"] == "unpaid"]
    return {
        "rows": rows, "bands": bands, "cumulative": cumulative,
        "total": {**aggregate(tested), "pending": sum(row["status"] == "pending" for row in rows),
                  "no_decision": sum(row["status"] == "no_decision" for row in rows)},
        "lowest_paid_z": min(paid_z) if paid_z else None,
        "highest_unpaid_z": max(unpaid_z) if unpaid_z else None,
        "first_round_id": rows[-1]["round_id"] if rows else None,
        "last_round_id": rows[0]["round_id"] if rows else None,
    }


def validate_confirmation_stats(stats: Mapping[str, Any], snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    """The standalone file is the same object embedded in this mining.json; fail closed on any mismatch."""
    if not isinstance(stats, Mapping) or not isinstance(snapshot.get("confirmations"), Mapping):
        raise ValueError("confirmation stats is not an object")
    if to_jsonable(stats) != to_jsonable(snapshot["confirmations"]):
        raise ValueError("confirmation stats is not the same object as confirmations in mining.json")
    return stats



def winner_stats(state: State, now: int, public_url: str = "") -> dict[str, Any]:
    """Safe stats for crowned/applied winners in the public closed-round history.

    This is not an all-time archive: it covers the same last ``HISTORY_PREVIEW`` closed rounds as the
    public snapshot. Only ``APPLIED`` rounds with an actual ``WINNER`` entry contribute. Batch facts
    and ``train`` knobs come from that winner's already-frozen batch, never from a waiting or losing
    entry. URLs are public object paths only.
    """
    return _winner_stats(state.rounds[-HISTORY_PREVIEW:], state.kings, [*state.reverted_heads, *state.heads], now, public_url,
                         pending=state.pending_king,
                         paid_rounds={king.round_id for king in paid_set(state.kings, state.null_rounds, state.params)})


def _winner_stats(records: Sequence[RoundRecord], kings_seq: Sequence[King], heads: Sequence[Any], now: int,
                  public_url: str = "", pending: Any = None, paid_rounds: set[str] | None = None) -> dict[str, Any]:
    from . import layout

    base = public_url.rstrip("/")
    kings = {king.round_id: king for king in kings_seq}
    if pending is not None:
        kings[pending.crown_round_id] = pending
    confirmations = {record.confirmation.pending_round_id: record.confirmation
                     for record in records if record.confirmation is not None}
    paid_rounds = set(kings) if paid_rounds is None else paid_rounds
    winners = []
    for record in reversed(records):
        winner = _crowned_winner(record)
        if winner is None:
            continue
        frozen = next((seat.batch for lease in record.leases for seat in lease.entries
                       if seat.queue_pos == winner.queue_pos and seat.batch.batch_sha256 == winner.batch_sha256), None)
        king = kings.get(record.round_id)
        new_head = king.new_version_id if king is not None and not isinstance(king, King) else king.head_version_id if king is not None else None
        head = next((head for head in reversed(heads) if new_head is not None and head.version_id == new_head), None)
        if head is None:
            head = next((head for head in reversed(heads) if any(ref.round_id == record.round_id for ref in head.loras)), None)
        if head is None:
            head = next((head for head in reversed(heads)
                         if head.round_id == record.round_id and head.parent_version_id == record.head_version_id), None)
        if head is None:
            head = next((head for head in reversed(heads) if head.round_id == record.round_id), None)
        confirmation = confirmations.get(record.round_id)
        pending_king = king is not None and not isinstance(king, King)
        applied_at = king.applied_at if king is not None else (head.applied_at if head is not None else record.closed_at)
        row = {
            "round_id": record.round_id,
            "uid": winner.uid,
            "hotkey": winner.hotkey,
            "applied_at": applied_at,
            "z": winner.score.z if winner.score is not None else None,
            "delta": winner.score.delta if winner.score is not None else None,
            "se": winner.score.se if winner.score is not None else None,
            "head": head.version_id if head is not None else new_head,
            "lora": winner.lora_hash,
            "batch_sha256": winner.batch_sha256,
            "batch_prompts": frozen.n_prompts if frozen is not None else None,
            "batch_rollouts": frozen.n_rollouts if frozen is not None else None,
            "batch_format": frozen.format if frozen is not None else None,
            "train": to_jsonable(frozen.train) if frozen is not None else None,
            "batch_url": f"{base}/{layout.winner_batch(record.round_id)}" if base else layout.winner_batch(record.round_id),
            "confirmed": (confirmation.confirmed if confirmation is not None else None if pending_king else
                          True if record.round_id in paid_rounds or record.round_id in kings else None),
            "paid": record.round_id in paid_rounds if king is not None or confirmation is not None else None,
            "payout_status": "paid" if record.round_id in paid_rounds else "unpaid" if king is not None or confirmation is not None else "unknown",
        }
        winners.append(row)
    return {"schema": WINNER_STATS_SCHEMA, "generated_at": now, "winners": winners}


ENRICHMENT_FIELDS = frozenset({"group_count", "boxed_frac", "output_chars_p50", "prompt_sources", "prompt_band"})
SOURCE_PREFIXES = ("gen", "dapo", "math", "nmv2", "other", "unmapped")


def _one_final_box(text: str) -> bool:
    """One balanced, non-empty ``\\boxed{}``, ending the output; no second box or trailing text."""
    if text.count("\\boxed{") != 1:
        return False
    begin = text.index("\\boxed{") + len("\\boxed{")
    depth = 1
    for position in range(begin, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return bool(text[begin:position].strip()) and not text[position + 1:].strip()
    return False


def _public_batch(raw: bytes) -> tuple[Any, ...]:
    """Strictly parse all training rows, without current admission caps or recipe/token rules."""
    from .admission import FORMAT_V4, FORMATS, _groups

    document = json.loads(raw)
    if not isinstance(document, dict) or not isinstance(document.get("hotkey"), str) or document.get("version") not in FORMATS:
        raise ValueError("public winner batch has an invalid header")
    if document["version"] != FORMAT_V4:
        return tuple(_groups(document, document["hotkey"]))
    stages = document.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("public winner batch has no stages")
    groups = []
    for stage in stages:
        if (not isinstance(stage, dict) or not set(stage) <= {"train", "groups", "sft"}
                or not set(stage) & {"groups", "sft"} or "train" in stage and not isinstance(stage["train"], dict)):
            raise ValueError("public winner batch has an invalid stage")
        if "groups" in stage:
            groups.extend(_groups({"hotkey": document["hotkey"], "groups": stage["groups"]}, document["hotkey"]))
        traces = stage.get("sft", [])
        if not isinstance(traces, list) or "sft" in stage and not traces:
            raise ValueError("public winner batch has invalid SFT traces")
        for trace in traces:
            if (not isinstance(trace, dict) or set(trace) != {"prompt", "output"}
                    or not isinstance(trace["prompt"], str) or not trace["prompt"].strip()
                    or not isinstance(trace["output"], str)):
                raise ValueError("public winner batch has an invalid SFT trace")
            # The same ordered singleton view used by admission's shared v4
            # checks. The placeholder reward is not an enrichment metric.
            groups.append((trace["prompt"], ((trace["output"], 1.0),)))
    return tuple(groups)


def winner_enrichment(snapshot: Mapping[str, Any], row: Mapping[str, Any], store: Any, prompt_pool: Any) -> dict[str, Any]:
    """Derive metrics from one already-public crowned batch, never from a private/frozen URI."""
    from . import layout
    from .canonical import sha256_hex

    key = layout.winner_batch(row["round_id"])
    raw = store.get(key)
    if sha256_hex(raw) != row["batch_sha256"]:
        raise ValueError("public winner batch does not match its crowned batch SHA256")
    groups = _public_batch(raw)
    outputs = [output for _, rollouts in groups for output, _ in rollouts]
    if not outputs:
        raise ValueError("public winner batch has no rollouts")
    sources = {name: 0 for name in SOURCE_PREFIXES}
    mapped: dict[str, int] = {}
    for prompt, _ in groups:
        task = prompt_pool.task_for_prompt(prompt)
        if task is None:
            sources["unmapped"] += 1
        else:
            prefix = task.task_id.split(":", 1)[0]
            sources[prefix if prefix in sources else "other"] += 1
            mapped[task.task_id] = mapped.get(task.task_id, 0) + 1
    result: dict[str, Any] = {
        "group_count": len(groups),
        "boxed_frac": sum(_one_final_box(output) for output in outputs) / len(outputs),
        "output_chars_p50": statistics.median(len(output) for output in outputs),
        "prompt_sources": sources,
    }
    record = next((record for record in snapshot.get("history", []) if record.get("round_id") == row["round_id"]), None)
    if record is None or not record.get("irt_snapshot_id"):
        return result
    try:
        irt = json.loads(store.get(layout.irt_snapshot(record["irt_snapshot_id"])))
        from .irt import predicted_pass, verify_snapshot

        verify_snapshot(irt)
        band = irt.get("config", {}).get("middle_band")
        if band is not None:
            low, high = band
            ability = irt.get("head_ability", 0.0)
            counts = {"low": 0, "middle": 0, "high": 0}
            for task_id, count in mapped.items():
                task = irt.get("tasks", {}).get(task_id)
                if task is None:
                    raise ValueError("mapped crowned-batch task is missing from its crown-round IRT")
                p = predicted_pass(ability, task[0])
                counts["low" if p < low else "high" if p > high else "middle"] += count
            result["prompt_band"] = counts
    except (KeyError, OSError, ValueError, TypeError, AttributeError, V2Error) as exc:
        # A bad IRT object removes only the optional band, not the already verified batch metrics.
        logging.getLogger(__name__).warning(
            "winner-stats prompt_band omitted for %s: %s: %s", row["round_id"], type(exc).__name__, exc
        )
    return result


def enrich_winner_stats(snapshot: Mapping[str, Any], stats: Mapping[str, Any], store: Any,
                        prompt_pool: Any, logger: Any = None) -> dict[str, Any]:
    """Best-effort per-crown enrichment; an unreadable public batch cannot stop publication."""
    logger = logging.getLogger(__name__) if logger is None else logger
    rows = []
    for original in stats["winners"]:
        row = {key: value for key, value in original.items() if key not in ENRICHMENT_FIELDS}
        try:
            row.update(winner_enrichment(snapshot, row, store, prompt_pool))
        except (KeyError, OSError, RuntimeError, ValueError, TypeError, AttributeError, V2Error) as exc:
            logger.warning("winner-stats enrichment omitted for %s: %s: %s", row["round_id"], type(exc).__name__, exc)
        rows.append(row)
    return {**stats, "winners": rows}


def validate_winner_enrichment(snapshot: Mapping[str, Any], row: Mapping[str, Any], store: Any, prompt_pool: Any) -> None:
    """Recompute optional public-batch facts; a type check alone would trust forged content."""
    derived = winner_enrichment(snapshot, row, store, prompt_pool)
    for name in ENRICHMENT_FIELDS:
        if name in row and (name not in derived or row[name] != derived[name]):
            raise ValueError(f"winner stats {name} does not match its public crowned batch and crown-round IRT")


def validate_winner_stats(snapshot: Mapping[str, Any], document: Mapping[str, Any], public_url: str = "",
                          store: Any = None, prompt_pool: Any = None) -> Mapping[str, Any]:
    """Check every stored winner fact against public history before a site mirrors or displays it.

    Batch counts and knobs are available only in the controller's crowned-only transport or its standalone public file.
    ``mining.json`` deliberately omits that transport: batch facts remain allowed, but are type/recipe checked;
    identity and every determined public-history fact still match. A lossy head-only projection may name a parent,
    so a supplied head must instead be a real version in the snapshot's public head history.
    """
    if not isinstance(document, Mapping) or set(document) != {"schema", "generated_at", "winners"} or document["schema"] != WINNER_STATS_SCHEMA:
        raise ValueError("winner stats has the wrong schema or fields")
    generated_at = document["generated_at"]
    if type(generated_at) is not int or generated_at != snapshot.get("generated_at") or not isinstance(document["winners"], list):
        raise ValueError("winner stats is not of this public snapshot")

    from . import layout
    from .params import Params

    base = (public_url or (snapshot.get("contract") or {}).get("public_url") or "").rstrip("/")
    expected = winner_stats_from_snapshot(snapshot, base)
    if len(document["winners"]) != len(expected["winners"]):
        raise ValueError("winner stats does not cover exactly the crowned winners of this public snapshot")
    allowed = WINNER_FIELDS | {"batch_url"}
    transport = {}
    crowns = _public_crowns(snapshot)
    for supplied in (snapshot.get("winner_stats") or {}).get("winners", []):
        if not isinstance(supplied, Mapping):
            continue
        winner = crowns.get(supplied.get("round_id"))
        if winner is not None and all(supplied.get(name) == winner.get(name) for name in ("uid", "hotkey", "batch_sha256")):
            transport[supplied["round_id"]] = supplied
    recipe = Params.from_dict(snapshot["params"]).recipe
    public_heads = {head.get("version_id") for head in snapshot.get("heads", [])}
    pending_round = None
    if Params.from_dict(snapshot["params"]).confirm_next_round:
        # The next round may already be open, and a null later round need not carry a confirmation. Do not
        # guess from its presence: an unresolved, unpaid projected crown is the pending one. Legacy objects
        # wrote confirmed=False here; the corrected writer emits null.
        pending = next((row for row in expected["winners"] if row["confirmed"] is None and not row["paid"]), None)
        pending_round = pending["round_id"] if pending is not None else None
    for row, public_row in zip(document["winners"], expected["winners"], strict=True):
        if not isinstance(row, Mapping) or not allowed <= set(row) <= allowed | ENRICHMENT_FIELDS:
            raise ValueError("winner stats must have exactly the public fields of a crowned winner plus optional batch-derived facts")
        if store is not None and set(row) & ENRICHMENT_FIELDS:
            validate_winner_enrichment(snapshot, row, store, prompt_pool)
        for name in allowed - {"batch_prompts", "batch_rollouts", "batch_format", "train", "batch_url", "head", "confirmed"}:
            if row[name] != public_row[name]:
                raise ValueError(f"winner stats {name} does not match public history")
        if transport and row["head"] != public_row["head"]:
            raise ValueError("winner stats head does not match public history")
        if row["head"] not in public_heads and row["head"] != public_row["head"]:
            # A lossy heads-window projection may no longer name an older crown's
            # head. The producer may still record a parent or pre-fold version that
            # has since aged out of snapshot["heads"]. Identity fields have already
            # been checked; accept that remaining well-formed version id only when
            # the projection has no head to contradict.
            head, projected = row["head"], public_row["head"]
            if projected or not (isinstance(head, str) and len(head) == 64 and set(head) <= set("0123456789abcdef")):
                raise ValueError("winner stats head is not a version in the public head history")
        supplied = transport.get(row["round_id"])
        if supplied is not None and row["confirmed"] != supplied["confirmed"]:
            raise ValueError("winner stats confirmed does not match the crowned winner's confirmation")
        elif row["round_id"] == pending_round:
            if row["confirmed"] not in (None, False):  # old objects may hold False; the corrected writer uses null
                raise ValueError("winner stats cannot call a pending crown confirmed")
        elif public_row["confirmed"] is not None and row["confirmed"] != public_row["confirmed"]:
            raise ValueError("winner stats confirmed does not match public history")
        key = layout.winner_batch(row["round_id"])
        if row["batch_url"] != (f"{base}/{key}" if base else key):
            raise ValueError("winner stats has a batch URL other than the crowned winner's configured public object")
        for name in ("batch_prompts", "batch_rollouts", "batch_format"):
            value = row[name]
            if supplied is None:
                if value is not None and type(value) is not (int if name != "batch_format" else str):
                    raise ValueError("winner stats batch facts have the wrong type")
            elif value != supplied.get(name) or type(value) is not (int if name != "batch_format" else str):
                raise ValueError("winner stats batch facts do not match the crowned winner's frozen batch")
        train = row["train"]
        if supplied is None:
            if train is not None and not isinstance(train, Mapping):
                raise ValueError("winner stats train must be an object or null")
        elif not isinstance(train, Mapping) or to_jsonable(train) != to_jsonable(supplied.get("train")):
            raise ValueError("winner stats train does not match the crowned winner's frozen batch")
        if train is not None:
            try:
                recipe.for_miner(train)
            except (TypeError, ValueError, AttributeError) as error:
                raise ValueError("winner stats train is not a valid set of public miner knobs") from error
    return document


WINNER_FIELDS = frozenset({
    "round_id", "uid", "hotkey", "applied_at", "z", "delta", "se", "head", "lora",
    "batch_sha256", "batch_prompts", "batch_rollouts", "batch_format", "train",
    "confirmed", "paid", "payout_status",
})


def _public_crowns(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The crowned entries named by public history, keyed by round, never losing or waiting seats."""
    crowned = {}
    for record in snapshot.get("history", []):
        if record.get("outcome") != APPLIED:
            continue
        winner = next((entry for entry in record.get("entries", []) if entry.get("status") == WINNER), None)
        if winner is not None:
            crowned[record["round_id"]] = winner
    return crowned


def winner_stats_from_snapshot(snapshot: Mapping[str, Any], public_url: str = "") -> dict[str, Any]:
    """Render crowned-only history from a public snapshot, without a journal lock or private reads.

    A controller snapshot includes a winner-only transport mapping made by ``winner_stats``. Every
    supplied row must match the winner in public history and is reduced to the safe field allowlist.
    External public history has no frozen batches, so their batch counts and train knobs stay null.
    """
    from . import layout

    base = public_url.rstrip("/")
    crowns = _public_crowns(snapshot)
    supplied = snapshot.get("winner_stats")
    if supplied is not None:
        if supplied.get("schema") != WINNER_STATS_SCHEMA or supplied.get("generated_at") != snapshot.get("generated_at"):
            raise ValueError("winner stats transport is not of this public snapshot")
        if not isinstance(supplied.get("winners"), list):
            raise ValueError("winner stats transport has no winners list")
        rows = []
        for supplied_row in supplied["winners"]:
            if not isinstance(supplied_row, Mapping):
                raise ValueError("winner stats transport row is not an object")
            winner = crowns.get(supplied_row.get("round_id"))
            if winner is None or any(supplied_row.get(name) != winner.get(name) for name in ("uid", "hotkey", "batch_sha256")):
                continue
            row = {name: supplied_row[name] for name in WINNER_FIELDS if name in supplied_row}
            row["batch_url"] = f"{base}/{layout.winner_batch(row['round_id'])}" if base else layout.winner_batch(row["round_id"])
            rows.append(row)
        return {"schema": WINNER_STATS_SCHEMA, "generated_at": snapshot["generated_at"], "winners": rows}

    rows = []
    history = json.loads(json.dumps(snapshot.get("history", [])))  # never attach a later confirmation to the caller's record
    for position, record in enumerate(history):
        if record.get("confirmation") is not None:
            continue
        resolved = next((later["confirmation"] for later in history[position + 1:]
                         if (later.get("confirmation") or {}).get("pending_round_id") == record["round_id"]), None)
        if resolved is not None:
            record["confirmation"] = resolved
    kings = {king["round_id"]: king for king in snapshot.get("kings", [])}
    confirmations = {confirmation["pending_round_id"]: confirmation
                     for record in history
                     if (confirmation := record.get("confirmation")) is not None}
    paid = {row["round_id"] for row in snapshot.get("paid_kings", [])}
    for record in reversed(history):
        round_id = record["round_id"]
        if round_id not in crowns:
            continue
        winner, king = crowns[round_id], kings.get(round_id)
        confirmation = confirmations.get(round_id)
        head_id = king.get("head_version_id") if king else None
        head = next((head for head in reversed(snapshot.get("heads", [])) if head_id and head["version_id"] == head_id), None)
        if head is None:
            head = next((head for head in reversed(snapshot.get("heads", [])) if head["round_id"] == round_id), None)
        rows.append({
            "round_id": round_id, "uid": winner["uid"], "hotkey": winner["hotkey"],
            "applied_at": king["applied_at"] if king else (head["applied_at"] if head else record["closed_at"]),
            "z": winner.get("z"), "delta": winner.get("delta"), "se": winner.get("se"),
            "head": head["version_id"] if head else head_id, "lora": winner.get("lora_hash"),
            "batch_sha256": winner["batch_sha256"], "batch_prompts": None, "batch_rollouts": None,
            "batch_format": None, "train": None,
            "batch_url": f"{base}/{layout.winner_batch(round_id)}" if base else layout.winner_batch(round_id),
            "confirmed": confirmation["confirmed"] if confirmation else True if round_id in paid else None,
            "paid": round_id in paid, "payout_status": "paid" if round_id in paid else "unpaid",
        })
    return {"schema": WINNER_STATS_SCHEMA, "generated_at": snapshot["generated_at"], "winners": rows}



def round_report(record: RoundRecord, leases: Sequence[tuple[RoundTicket, BoxResult]]) -> dict[str, Any]:
    """The public file of a settled round: the draw and all scored eval outcomes.

    ``entry_outcomes`` is in queue order and omits entries without outcome
    strings. Null M and the winner keep their original, separate fields.
    """
    report = round_view(record)
    winner = record.decision.winner if record.decision else None
    if leases:
        draw = leases[0][0].draw
        report["draw"] = draw.to_dict()
    report["null_outcomes"] = {ticket.box_id: list(result.null_outcomes) for ticket, result in leases}
    nulls = [{name: getattr(result, f"null_{name}") for name in ("kl_seq", "kl_seq_se", "kl_tok", "kl_k", "kl_tokens")}
             for _, result in leases if result.null_kl_seq is not None]
    if nulls:
        import math

        count = sum(row["kl_k"] for row in nulls)
        tokens = sum(row["kl_tokens"] for row in nulls)
        mean = math.fsum(row["kl_seq"] * row["kl_k"] for row in nulls) / count
        # Recover each calibration's sum of squares, then pool it without a new pass.
        spread = math.fsum(
            (row["kl_k"] - 1) * (row["kl_seq_se"] * math.sqrt(row["kl_k"])) ** 2
            + row["kl_k"] * (row["kl_seq"] - mean) ** 2 for row in nulls
        )
        report |= {"null_kl_seq": mean, "null_kl_seq_se": math.sqrt(spread / (count - 1) / count) if count > 1 else 0.0,
                   "null_kl_tok": math.fsum(row["kl_tok"] * row["kl_tokens"] for row in nulls) / tokens,
                   "null_kl_k": count, "null_kl_tokens": tokens}
    report["winner_outcomes"] = next(
        (list(entry.outcomes) for _, result in leases for entry in result.entries if winner and entry.queue_pos == winner.queue_pos),
        None,
    )
    if record.confirmation is not None:
        report["confirmation"] = confirmation_view(record.confirmation)
    parent_results = [result for ticket, result in leases if ticket.parent is not None]
    if parent_results or record.parent is not None:
        # Parent archives are optional; the fresh sealed outcomes and decision are mandatory public evidence,
        # including a no-result or duplicate-only close.
        report["parent_outcomes"] = next((list(result.parent_outcomes) for result in parent_results if result.parent_outcomes), None)
        error = next((result.parent_error for result in parent_results if result.parent_error), "")
        report["parent_error"] = public_text(error) if error else (None if report["parent_outcomes"] else "no fresh parent result arrived")
    from . import layout

    scored = {entry.queue_pos: entry for _, result in leases for entry in result.entries if entry.outcomes}
    seats = {seat.queue_pos: seat for ticket, _ in leases for seat in ticket.entries}
    decisions = {entry.queue_pos: entry for entry in record.decision.entries} if record.decision else {}
    report["entry_outcomes"] = [
        {
            "queue_pos": pos,
            "uid": seats[pos].reg.uid,
            "hotkey": seats[pos].reg.hotkey,
            "lora_hash": scored[pos].lora.lora_hash,
            "status": decisions[pos].status if pos in decisions else None,
            "outcomes": list(scored[pos].outcomes),
        } | {name: value for name in KL_FIELDS if (value := getattr(scored[pos], name)) is not None}
        for pos in sorted(scored)
    ]
    report["rollouts_meta"] = layout.public_rollout_gate_meta(record.round_id)
    report["stopped_at_limit"] = any(result.stopped_at_limit for _, result in leases)
    # The summary a person relays: per LoRA the verdict, its reason and the size of the trained update.
    trained = {entry.queue_pos: entry.lora.report for _, result in leases for entry in result.entries if entry.lora is not None}
    reasons = {entry.queue_pos: entry.reason for entry in record.decision.entries} if record.decision else {}
    for view in report["entries"]:
        view["reason"] = public_text(reasons.get(view["queue_pos"], ""))
        measured = next((entry for _, result in leases for entry in result.entries if entry.queue_pos == view["queue_pos"]), None)
        if measured is not None:
            view |= {name: value for name in ("kl_seq", "kl_seq_se", "kl_tok", "kl_k", "kl_tokens") if (value := getattr(measured, name)) is not None}
        if view["queue_pos"] in trained:
            train = trained[view["queue_pos"]]
            train_data = train.to_dict()
            view |= {"step_norm": train_data["step_norm"], "norm_before_cap": train_data["norm_before_cap"],
                     "train_wall_s": train.wall_s}
            for tag in ("step_norm_nonfinite", "norm_before_cap_nonfinite"):
                if train_data.get(tag):
                    view[tag] = train_data[tag]
    return report


def _span(seconds: float) -> str:
    return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"


def round_text(report: Mapping[str, Any]) -> str:
    """The round summary as a short text to paste. ``report`` is the public round file."""
    entries = report.get("entries", [])
    winner = next((e for e in entries if e.get("status") == WINNER), None)
    lines = [f"SN120 v2 round {report['round_id']}: {str(report['outcome']).upper()}" + (" (stopped at the time limit)" if report.get("stopped_at_limit") else "")]
    if winner:
        lines.append(f"winner: UID {winner['uid']} ({winner['hotkey']}), delta {winner['delta']:+.4f}, z {winner['z']:.2f}")
        lines.append(f"head: {report['head_version_id'][:12]} -> {str(report.get('new_head_version_id'))[:12]}")
    else:
        lines.append(f"no winner: {report.get('reason') or report.get('note') or 'the head and the kings stay as they are'}")
        lines.append(f"head: {report['head_version_id'][:12]} (unchanged)")
    scored = max((e.get("n_tasks", 0) for e in entries), default=0)
    lines.append(
        f"batches: {report['seats']} | tasks scored: {scored} of {report['n_tasks']}"
        + (f" | null M: pass {report['null_pass_rate']:.2%}, cap-hit {report['null_cap_rate']:.2%}" if "null_pass_rate" in report else "")
    )
    waiting = sum(e.get("status") == NOT_TRAINED for e in entries)
    if waiting:
        lines.append(f"not trained: {waiting} of {report['seats']} batches. The training time of the round ended first. They are not charged and are first in the next round.")
    cut = sum(e.get("status") == NOT_SCORED for e in entries)
    if cut:
        lines.append(f"not scored: {cut} of {report['seats']} batches. The time limit of the round came before the first look. They are not charged and are first in the next round.")
    lines.append(f"{'pos':>3} {'uid':>4}  {'hotkey':<12} {'verdict':<11} {'delta':>8} {'z':>6} {'cap-hit':>7} {'tasks':>5} {'norm':>9}  note")
    for e in entries:
        hotkey = e["hotkey"] if len(e["hotkey"]) <= 12 else e["hotkey"][:5] + ".." + e["hotkey"][-5:]
        numbers = f"{e['delta']:>+8.4f} {e['z']:>6.2f} {e['cap_rate']:>7.2%} {e['n_tasks']:>5}" if "z" in e else f"{'-':>8} {'-':>6} {'-':>7} {'-':>5}"
        before = e.get("norm_before_cap") or e.get("step_norm")  # the norm that training gave: the number that recipe.global_norm_cap is set from
        norm = f"{before:>9.4g}" if before is not None else f"{'-':>9}"
        if e.get("norm_before_cap_nonfinite") or e.get("step_norm_nonfinite"):
            norm = f"{'nonfinite':>9}"
        note = e.get("reason", "")
        if before and not e.get("norm_before_cap_nonfinite") and not e.get("step_norm_nonfinite") and e.get("step_norm", before) < before * (1 - 1e-9):
            note = f"update capped to norm {e['step_norm']:.4g}" + (f"; {note}" if note else "")
        if e.get("eliminated_at_look") is not None:
            note = f"dropped at look {e['eliminated_at_look']}" + (f"; {note}" if note else "")
        lines.append(f"{e['queue_pos']:>3} {e['uid']:>4}  {hotkey:<12} {e.get('status', 'no result'):<11} {numbers} {norm}  {note}".rstrip())
    wall = report.get("wall_s") or {}
    names = [("training", "train"), ("evaluation", "eval"), ("kl", "kl"), ("promote", "promote"), ("fold", "fold"), ("lease_to_promotion", "lease to promotion"), ("total", "total")]
    lines.append("wall time: " + " | ".join(f"{label} {_span(wall[key])}" for key, label in names if key in wall))
    return "\n".join(lines)
