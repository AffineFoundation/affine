"""The public view. One plain dict feeds llms.txt, mining.json, the website and the benchmark service.

Privacy rule: an entry that did not win shows only its seat, its batch
sha256, its status, and delta, se, z and cap rate. Its file, its location,
its train values and its per-task outcomes are never published. Only a
winner's LoRA and outcomes are public.

Score rule: delta, se and z are numbers only for an entry that was judged
(winner or lost). An entry that the box scored and the judge did not use
(too few rollouts resolved, null M failed, the time limit came before the
first look) has them as null. The journal keeps the numbers.
"""

from __future__ import annotations

import ipaddress
import os
import re
from typing import Any, Mapping, Sequence

from .params import Params
from .payout import payees
from .queue import waiting
from .state import RoundRecord, State
from .types import NOT_SCORED, NOT_TRAINED, WINNER, BoxResult, EntryDecision, RoundTicket, Verdict

SCHEMA = "sn120-v2-public-1"
QUEUE_PREVIEW = 50
HISTORY_PREVIEW = 20

# ``public_text``: the rule for a text that an exception made, before it goes into a public file.
TEXT_LIMIT = 600  # characters
SECRET_NAME = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|ENDPOINT|ACCOUNT", re.IGNORECASE)
SECRET_MIN = 8  # a shorter value is a flag or a number (TOKENIZERS_PARALLELISM=false): to replace it would cut plain words
CUT_SECRET, CUT_LOGIN, CUT_QUERY, CUT_ADDRESS = "[removed-secret]", "[removed-login]", "[removed-query]", "[removed-address]"
CUT_PATH = "[removed-path]"
# The tail that the trainer adds to the error of a failed job (``gpu.trainer.LoraTrainer._keep_log``): where the log
# of the job is on the box, and the last line of that log. Neither is public. ``job_cause`` says the cause in fixed words.
_JOB_LOG = re.compile(r"; log: .*?; last line: (.*)$")
_OUT_OF_MEMORY = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
_ERROR_CLASS = re.compile(r"(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*(?:Error|Exception))(?::|$)")
OUT_OF_MEMORY, IN_THE_LOG = "cause: out of memory", "the cause is in the log of the box"
# A path of a file system with two parts or more, such as /root/work/logs or ~/x/y. The path of a URL is not one.
_PATH = re.compile(r"(?<![\w/:.@~\]-])~?/(?:[\w.+@%~-]+/)+[\w.+@%~-]*")
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
    * each IPv4 and IPv6 address. A port stays;
    * each path of a file on this machine (``CUT_PATH``).

    The error of a failed training job ends with the place of its log and
    the last line of that log. Both are cut, and fixed words give the cause
    in their place (``job_cause``).

    Then the text is cut to ``TEXT_LIMIT`` characters. The journal and the
    sealed box result keep the text as it was: they are not public.
    """
    text = " ".join(str(text).split())
    text = _JOB_LOG.sub(lambda match: "; " + job_cause(match.group(1)), text)
    environ = os.environ if environ is None else environ
    secrets = {value.strip() for name, value in environ.items() if SECRET_NAME.search(name)}
    for value in sorted((value for value in secrets if len(value) >= SECRET_MIN), key=len, reverse=True):
        text = text.replace(value, CUT_SECRET)
    text = _LOGIN.sub(CUT_LOGIN + "@", text)
    text = _URL_QUERY.sub(lambda match: f"{match.group(1)}?{CUT_QUERY}", text)
    text = _QUERY.sub("?" + CUT_QUERY, text)
    text = _NAMED.sub(lambda match: match.group(1) + CUT_SECRET, text)
    text = _V4.sub(_address, _V6.sub(_address, text))
    text = _PATH.sub(CUT_PATH, text)
    if len(text) > TEXT_LIMIT:
        tail = f" [cut: {len(text) - TEXT_LIMIT} more characters]"
        text = text[:TEXT_LIMIT] + tail
    return text


def job_cause(last_line: str) -> str:
    """The cause of a failed training job in fixed words, from the last line of its log. No word of the line is copied but the name of an error class."""
    if _OUT_OF_MEMORY.search(last_line):
        return OUT_OF_MEMORY
    named = _ERROR_CLASS.match(last_line)
    return f"cause: an error of the class {named.group(1)}" if named else IN_THE_LOG


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
        score, judged = entry.score, entry.charged  # winner or lost: the only entries whose scores the judge used
        view |= {"delta": score.delta if judged else None, "se": score.se if judged else None, "z": score.z if judged else None}
        view |= {"cap_rate": score.cap_rate, "n_tasks": score.n_tasks}
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
        view |= {"winner_uid": decision.winner.uid if decision.winner else None, "reason": public_text(decision.reason), "null_pass_rate": decision.null_pass_rate, "null_cap_rate": decision.null_cap_rate}
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
        view["reason"] = public_text(reasons.get(view["queue_pos"], ""))
        if view["queue_pos"] in trained:
            train = trained[view["queue_pos"]]
            view |= {"step_norm": train.step_norm, "norm_before_cap": train.norm_before_cap, "train_wall_s": train.wall_s}
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
        judged = f"{e['delta']:>+8.4f} {e['z']:>6.2f}" if e.get("z") is not None else f"{'-':>8} {'-':>6}"
        numbers = judged + (f" {e['cap_rate']:>7.2%} {e['n_tasks']:>5}" if e.get("n_tasks") is not None else f" {'-':>7} {'-':>5}")
        before = e.get("norm_before_cap") or e.get("step_norm")  # the norm that training gave: the number that recipe.global_norm_cap is set from
        norm = f"{before:>9.4g}" if before is not None else f"{'-':>9}"
        note = e.get("reason", "")
        if before and e.get("step_norm", before) < before * (1 - 1e-9):
            note = f"update capped to norm {e['step_norm']:.4g}" + (f"; {note}" if note else "")
        if e.get("eliminated_at_look") is not None:
            note = f"dropped at look {e['eliminated_at_look']}" + (f"; {note}" if note else "")
        lines.append(f"{e['queue_pos']:>3} {e['uid']:>4}  {hotkey:<12} {e.get('status', 'no result'):<11} {numbers} {norm}  {note}".rstrip())
    wall = report.get("wall_s") or {}
    names = [("training", "train"), ("evaluation", "eval"), ("promote", "promote"), ("fold", "fold"), ("lease_to_promotion", "lease to promotion"), ("total", "total")]
    lines.append("wall time: " + " | ".join(f"{label} {_span(wall[key])}" for key, label in names if key in wall))
    return "\n".join(lines)
