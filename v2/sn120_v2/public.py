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
from .types import NOT_SCORED, NOT_TRAINED, WINNER, BoxResult, EntryDecision, RoundTicket, Verdict

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


def snapshot(
    state: State, params: Params, now: int, verdicts: Mapping[str, Verdict] | None = None, files_checked_at: int | None = None
) -> dict[str, Any]:
    """Everything a miner, the website or the benchmark service may see.

    ``verdicts`` is admission's last check of the waiting files, taken at
    ``files_checked_at``. A file that a miner stored after that time shows
    the old verdict until the next check.
    """
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
            "files_checked_at": files_checked_at,
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
    report["stopped_at_limit"] = any(result.stopped_at_limit for _, result in leases)
    # The summary a person relays: per LoRA the verdict, its reason and the size of the trained update.
    trained = {entry.queue_pos: entry.lora.report for _, result in leases for entry in result.entries if entry.lora is not None}
    reasons = {entry.queue_pos: entry.reason for entry in record.decision.entries} if record.decision else {}
    for view in report["entries"]:
        view["reason"] = reasons.get(view["queue_pos"], "")
        if view["queue_pos"] in trained:
            train = trained[view["queue_pos"]]
            view |= {"step_norm": train.step_norm, "norm_before_cap": train.norm_before_cap, "train_wall_s": train.wall_s}
    return report


def _span(seconds: float) -> str:
    return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"


def round_text(report: Mapping[str, Any]) -> str:
    """The round summary as a short text to paste. ``report`` is the public round file."""
    entries = report.get("entries", [])
    winner = next((e for e in entries if e["status"] == WINNER), None)
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
    waiting = sum(e["status"] == NOT_TRAINED for e in entries)
    if waiting:
        lines.append(f"not trained: {waiting} of {report['seats']} batches. The training time of the round ended first. They are not charged and are first in the next round.")
    cut = sum(e["status"] == NOT_SCORED for e in entries)
    if cut:
        lines.append(f"not scored: {cut} of {report['seats']} batches. The time limit of the round came before the first look. They are not charged and are first in the next round.")
    lines.append(f"{'pos':>3} {'uid':>4}  {'hotkey':<12} {'verdict':<11} {'delta':>8} {'z':>6} {'cap-hit':>7} {'tasks':>5} {'norm':>9}  note")
    for e in entries:
        hotkey = e["hotkey"] if len(e["hotkey"]) <= 12 else e["hotkey"][:5] + ".." + e["hotkey"][-5:]
        numbers = f"{e['delta']:>+8.4f} {e['z']:>6.2f} {e['cap_rate']:>7.2%} {e['n_tasks']:>5}" if "z" in e else f"{'-':>8} {'-':>6} {'-':>7} {'-':>5}"
        before = e.get("norm_before_cap") or e.get("step_norm")  # the norm that training gave: the number that recipe.global_norm_cap is set from
        norm = f"{before:>9.4g}" if before is not None else f"{'-':>9}"
        note = e.get("reason", "")
        if before and e.get("step_norm", before) < before * (1 - 1e-9):
            note = f"update capped to norm {e['step_norm']:.4g}" + (f"; {note}" if note else "")
        if e.get("eliminated_at_look") is not None:
            note = f"dropped at look {e['eliminated_at_look']}" + (f"; {note}" if note else "")
        lines.append(f"{e['queue_pos']:>3} {e['uid']:>4}  {hotkey:<12} {e['status']:<11} {numbers} {norm}  {note}".rstrip())
    wall = report.get("wall_s") or {}
    names = [("training", "train"), ("evaluation", "eval"), ("promote", "promote"), ("fold", "fold"), ("lease_to_promotion", "lease to promotion"), ("total", "total")]
    lines.append("wall time: " + " | ".join(f"{label} {_span(wall[key])}" for key, label in names if key in wall))
    return "\n".join(lines)
