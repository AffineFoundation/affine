"""The public view. One plain dict feeds llms.txt, mining.json, the website and the benchmark service.

Privacy rule: an entry that did not win shows only its seat, its batch
sha256, its status, and delta, se, z and cap rate. Its file, its location,
its train values and its per-task outcomes are never published. Only a
winner's LoRA and outcomes are public.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .params import Params
from .payout import payees
from .queue import waiting
from .state import RoundRecord, State
from .types import WINNER, BoxResult, EntryDecision, RoundTicket, Verdict

SCHEMA = "sn120-v2-public-1"
QUEUE_PREVIEW = 50
HISTORY_PREVIEW = 20


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
    if entry.status == WINNER:
        view |= {"lora_hash": entry.lora_hash, "pass_rate": entry.score.pass_rate, "null_pass_rate": entry.score.null_pass_rate}
    return view


def phase(state: State) -> str:
    """IDLE: no round. RUNNING: boxes train and score. APPLY: decided, the head and files are being published."""
    if state.round is None:
        return "IDLE"
    return "RUNNING" if state.round.decision is None else "APPLY"


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
        "seats": len(seats),
        "skipped": len(record.skipped),
    }
    if record.decision is None:
        view["entries"] = [
            {"queue_pos": e.queue_pos, "uid": e.reg.uid, "hotkey": e.reg.hotkey, "registration_block": e.reg.registration_block, "batch_sha256": e.batch.batch_sha256}
            for e in seats
        ]
    else:
        decision = record.decision
        view["entries"] = [entry_view(entry) for entry in decision.entries]
        view |= {"winner_uid": decision.winner.uid if decision.winner else None, "reason": decision.reason, "null_pass_rate": decision.null_pass_rate, "null_cap_rate": decision.null_cap_rate}
    return view


def snapshot(state: State, params: Params, now: int, verdicts: Mapping[str, Verdict] | None = None) -> dict[str, Any]:
    """Everything a miner, the website or the benchmark service may see."""
    verdicts = verdicts or {}
    line = waiting(state.queue)
    return {
        "schema": SCHEMA,
        "generated_at": now,
        "netuid": params.netuid,
        "env": params.env,
        "phase": phase(state),
        "head": state.manifest.to_dict() if state.manifest else None,
        "heads": [{"version_id": h.version_id, "round_id": h.round_id, "applied_at": h.applied_at, "folded": h.folded} for h in state.heads[-HISTORY_PREVIEW:]],
        "params": params.to_dict(),
        "params_hash": params.params_hash,
        "recipe_id": params.recipe_id,
        "irt_snapshot_id": state.snapshot_id,
        "round": round_view(state.round) if state.round else None,
        "history": [round_view(record) for record in state.rounds[-HISTORY_PREVIEW:]],
        "queue": {
            "length": len(line),
            "next": [
                {
                    "position": position,
                    "uid": seat.reg.uid,
                    "hotkey": seat.reg.hotkey,
                    "registration_block": seat.reg.registration_block,
                    "file_valid": bool(verdicts.get(seat.reg.hotkey) and verdicts[seat.reg.hotkey].valid),
                    "front": seat.front,
                    "demoted": seat.demoted,
                    "held": seat.held,
                }
                for position, seat in enumerate(line[:QUEUE_PREVIEW])
            ],
        },
        "kings": [king.to_dict() for king in payees(state.kings, params.kings_window)],
    }


def round_report(record: RoundRecord, leases: Sequence[tuple[RoundTicket, BoxResult]]) -> dict[str, Any]:
    """The public file of a settled round: the draw, null M's outcomes, and the winner's outcomes only."""
    report = round_view(record)
    winner = record.decision.winner if record.decision else None
    if leases:
        draw = leases[0][0].draw
        report["draw"] = draw.to_dict()
    report["null_outcomes"] = {ticket.box_id: list(result.null_outcomes) for ticket, result in leases}
    report["winner_outcomes"] = next(
        (list(entry.outcomes) for _, result in leases for entry in result.entries if winner and entry.queue_pos == winner.queue_pos),
        None,
    )
    return report
