"""Judging. Pure functions: a ticket and sealed results in, one decision out.

The box never picks the winner. The controller recomputes everything here
from the outcome strings, so the decision is a fixed function of the score
file and the pinned parameters.

Estimator. For each scored task t, d_t = (LoRA pass rate) - (null M pass
rate), on the same seeds. With sampling weights w_t:

    delta = sum(w_t * d_t) / sum(w_t)
    var   = n / (n - 1) * sum((w_t * (d_t - delta))^2) / sum(w_t)^2
    se    = sqrt(var),  z = delta / se

The variance uses the spread of the realised d_t. One rollout per task has
no per-task noise estimate, and the spread of d_t already holds both the
task-sampling noise and the rollout noise. It has no finite-population
term, so it is slightly on the safe side. With equal weights it is the
paired standard error: sd(d) / sqrt(n).

Win rule. A LoRA is eligible when it was scored on every task, z is strictly
above ``z_bar``, and its cap-hit rate is not above
max(cap_hit_floor, null cap-hit rate + cap_hit_margin). Only a rate strictly
greater than that limit is rejected. The eligible LoRA with the highest
delta wins. A tie goes to the earlier-queued entry. With no eligible LoRA,
M stays. The bar has no correction for the number of LoRAs in the round.

Early looks. The win rule runs only at the last look. At an earlier look a
LoRA can only be dropped, and only when it can no longer reach the bar:

* A look that has scored a share t of the draw gives z_t. The z this LoRA
  is heading for at the end of the eval is estimated by z_t / sqrt(t), with
  standard error 1 / sqrt(t).
* Its upper confidence bound is (z_t + c_k) / sqrt(t). The LoRA is dropped
  when that bound is at or below ``z_bar``.
* c_k comes from an alpha-spending function. Look k may spend
  a_k = spent(t_k) - spent(t_(k-1)), with spent(t) = drop_alpha * ln(1 + (e - 1) * t)
  (Lan-DeMets, Pocock type), and c_k is the normal quantile of 1 - a_k.
  The a_k add up to less than ``drop_alpha``. So a LoRA whose true level is
  on the bar or above it is dropped early with a total chance below
  ``drop_alpha``, however many looks there are.

Dropping cannot create a winner, so the final test needs no correction.
"""

from __future__ import annotations

import math
from fractions import Fraction
from statistics import NormalDist
from typing import Sequence

from .errors import ContractError, HeadMismatchError
from .params import Params
from .types import (
    CAP,
    ERROR,
    LOST,
    NO_DECISION,
    NOT_RUN,
    PASS,
    SCORED,
    SKIP,
    WINNER,
    BoxResult,
    EntryDecision,
    EntryResult,
    RoundDecision,
    RoundTicket,
    Score,
    check_outcomes,
)

TIE_TOLERANCE = 1e-12


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def scored_prefix(outcomes: Sequence[str]) -> int:
    """How many leading tasks were run. Every sample string must agree, and nothing may follow a gap."""
    lengths = set()
    for text in outcomes:
        run = len(text.rstrip(NOT_RUN))
        _need(NOT_RUN not in text[:run], "an outcome string has a scored task after a task that was not run")
        lengths.add(run)
    _need(len(lengths) == 1, "sample strings of one entry cover different numbers of tasks")
    return lengths.pop()


def pass_rates(outcomes: Sequence[str], n: int) -> list[float]:
    samples = len(outcomes)
    return [sum(text[t] == PASS for text in outcomes) / samples for t in range(n)]


def code_count(outcomes: Sequence[str], code: str, n: int) -> int:
    return sum(text[:n].count(code) for text in outcomes)


def code_share(outcomes: Sequence[str], code: str, n: int) -> float:
    return code_count(outcomes, code, n) / (n * len(outcomes))


def _as_written(value: float) -> Fraction:
    """A parameter as the decimal it was written as: 0.33 means exactly 33/100."""
    return Fraction(repr(value))


def cap_hit_ok(lora_caps: int, null_caps: int, rollouts: int, params: Params) -> bool:
    """Exact test of: cap rate <= max(cap_hit_floor, null cap rate + cap_hit_margin).

    A LoRA is rejected only when its cap-hit share is strictly greater than
    the limit. A share exactly on the limit passes. The test uses
    whole-number counts, so no rounding error can move the limit.
    """
    limit = max(_as_written(params.cap_hit_floor), Fraction(null_caps, rollouts) + _as_written(params.cap_hit_margin))
    return Fraction(lora_caps, rollouts) <= limit


def cover_ok(outcomes: Sequence[str], n: int, params: Params) -> bool:
    """True when enough rollouts resolved: the share without an infrastructure error is at least min_cover."""
    rollouts = n * len(outcomes)
    return Fraction(rollouts - code_count(outcomes, ERROR, n), rollouts) >= _as_written(params.min_cover)


def weighted_mean(weights: Sequence[float], values: Sequence[float]) -> float:
    return math.fsum(w * v for w, v in zip(weights, values)) / math.fsum(weights)


def estimate(weights: Sequence[float], diffs: Sequence[float]) -> tuple[float, float]:
    """(delta, se) of the weighted mean of paired differences. se is 0.0 when it cannot be estimated."""
    n = len(diffs)
    _need(n == len(weights) and n >= 1, "the estimator needs one weight per scored task")
    delta = weighted_mean(weights, diffs)
    if n < 2:
        return delta, 0.0
    total = math.fsum(weights)
    spread = math.fsum((w * (d - delta)) ** 2 for w, d in zip(weights, diffs))
    return delta, math.sqrt(n / (n - 1) * spread) / total


def cap_limit(null_cap_rate: float, params: Params) -> float:
    return max(params.cap_hit_floor, null_cap_rate + params.cap_hit_margin)


def score_entry(
    weights: Sequence[float],
    null_outcomes: Sequence[str],
    lora_outcomes: Sequence[str],
    params: Params,
    n_total: int | None = None,
) -> Score:
    """Score one LoRA against null M on the tasks the LoRA was run on.

    ``n_total`` is the size of the full draw. Leave it out to score a part
    of the draw at an early look: then the entry is never eligible.
    """
    n = scored_prefix(lora_outcomes)
    _need(n >= 1, "a scored entry needs at least one scored task")
    w = weights[:n]
    lora, null = pass_rates(lora_outcomes, n), pass_rates(null_outcomes, n)
    delta, se = estimate(w, [a - b for a, b in zip(lora, null)])
    z = delta / se if se > 0 else 0.0
    cap_rate, null_cap_rate = code_share(lora_outcomes, CAP, n), code_share(null_outcomes, CAP, n)
    reasons = []
    if n_total is None or n < n_total:
        reasons.append(f"scored on {n} tasks, not on the full draw")
    if n < params.min_scored_tasks:
        reasons.append(f"scored on {n} tasks; the rule needs {params.min_scored_tasks}")
    if not z > params.z_bar:
        reasons.append(f"z {z:.3f} is not above {params.z_bar:g}")
    rollouts = n * len(lora_outcomes)
    if not cap_hit_ok(code_count(lora_outcomes, CAP, n), code_count(null_outcomes, CAP, n), rollouts, params):
        reasons.append(f"cap-hit rate {cap_rate:.4f} is above {cap_limit(null_cap_rate, params):.4f}")
    return Score(
        n_tasks=n,
        delta=delta,
        se=se,
        z=z,
        pass_rate=weighted_mean(w, lora),
        null_pass_rate=weighted_mean(w, null),
        cap_rate=cap_rate,
        null_cap_rate=null_cap_rate,
        cover=1.0 - code_share(lora_outcomes, ERROR, n),
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def spent_alpha(share: float, alpha: float) -> float:
    """Error that early drops may have spent once ``share`` of the draw is scored (Lan-DeMets, Pocock type)."""
    return alpha * math.log(1.0 + (math.e - 1.0) * share)


def drop_margin(look: int, n_total: int, params: Params) -> float:
    """c_k: how many standard errors the upper confidence bound sits above the estimate at early look k."""
    sizes = params.look_sizes(n_total)
    earlier = sizes[look - 1] / n_total if look else 0.0
    allowed = spent_alpha(sizes[look] / n_total, params.drop_alpha) - spent_alpha(earlier, params.drop_alpha)
    return NormalDist().inv_cdf(1.0 - allowed)


def upper_final_z(score: Score, look: int, n_total: int, params: Params) -> float:
    """Upper confidence bound on the z this LoRA is heading for at the end of the eval."""
    return (score.z + drop_margin(look, n_total, params)) / math.sqrt(score.n_tasks / n_total)


def drop_at_look(score: Score, look: int, n_total: int, params: Params) -> bool:
    """True when a LoRA can no longer reach the win bar. Only an early look can drop, and it never names a winner."""
    sizes = params.look_sizes(n_total)
    if not 0 <= look < len(sizes) - 1 or score.n_tasks != sizes[look]:
        return False
    if score.n_tasks < params.min_scored_tasks or score.se <= 0:
        return False
    return upper_final_z(score, look, n_total, params) <= params.z_bar


def pick_winner(candidates: Sequence[tuple[int, Score]]) -> int | None:
    """Queue position of the eligible entry with the highest delta. A tie goes to the earlier queue position."""
    best: tuple[int, float] | None = None
    for queue_pos, score in sorted(candidates, key=lambda pair: pair[0]):
        if score.eligible and (best is None or score.delta > best[1] + TIE_TOLERANCE):
            best = (queue_pos, score.delta)
    return None if best is None else best[0]


def check_result(ticket: RoundTicket, result: BoxResult) -> None:
    """Fail closed unless the sealed result answers exactly this ticket and is well formed."""
    params, draw = ticket.params, ticket.draw
    n, samples = len(draw.task_ids), params.samples_per_task
    _need(
        (result.round_id, result.lease_id, result.box_id) == (ticket.round_id, ticket.lease_id, ticket.box_id),
        "the result names another round, lease or box",
    )
    if result.head_version_id != ticket.head.version_id:
        raise HeadMismatchError("the result was computed against another head than the ticket's")
    _need(result.ticket_digest == ticket.digest(), "the result does not answer this ticket")
    _need(result.image_digest in params.recipe.image_digests, "the worker image is not in the pinned set")
    _need(
        [e.queue_pos for e in result.entries] == [e.queue_pos for e in ticket.entries],
        "the result must report every ticket entry once, in queue order",
    )
    null_scored = 0
    if any(e.status == SCORED for e in result.entries):
        _need(len(result.null_outcomes) == samples, "null M needs one outcome string per sample")
        check_outcomes(result.null_outcomes, n, "null M")
        null_scored = scored_prefix(result.null_outcomes)
    look_sizes = params.look_sizes(n)
    for seat, entry in zip(ticket.entries, result.entries):
        what = f"entry {entry.queue_pos}"
        if entry.status != SCORED:
            _need(not entry.outcomes and entry.eliminated_at_look is None, f"{what}: an unscored entry has no outcomes")
            continue
        lora = entry.lora
        _need(lora.head_version_id == ticket.head.version_id, f"{what}: LoRA was trained on another head")
        _need(lora.batch_sha256 == seat.batch.batch_sha256, f"{what}: LoRA was trained on another batch")
        _need(lora.recipe_id == params.recipe_id, f"{what}: LoRA was trained with another recipe")
        recipe = params.recipe.for_miner(seat.batch.train)
        _need(
            (lora.rank, lora.alpha, tuple(lora.target_modules)) == (recipe.rank, recipe.alpha, recipe.target_modules),
            f"{what}: LoRA rank, alpha or target modules differ from the recipe with the miner's values",
        )
        _need(lora.report.image_digest in params.recipe.image_digests, f"{what}: trainer image is not pinned")
        _need(math.isfinite(lora.report.loss) and math.isfinite(lora.report.step_norm), f"{what}: non-finite train report")
        _need(len(entry.outcomes) == samples, f"{what}: needs one outcome string per sample")
        check_outcomes(entry.outcomes, n, what)
        scored = scored_prefix(entry.outcomes)
        _need(scored <= null_scored, f"{what}: scored on more tasks than null M")
        look = entry.eliminated_at_look
        if look is None:
            _need(scored == n, f"{what}: not dropped, so it must be scored on every task")
            continue
        _need(0 <= look < len(look_sizes) - 1, f"{what}: can only be dropped at an early look")
        _need(scored == look_sizes[look], f"{what}: dropped at look {look} but scored on {scored} tasks")
        partial = score_entry(draw.weights, result.null_outcomes, entry.outcomes, params)
        _need(drop_at_look(partial, look, n, params), f"{what}: the drop rule does not hold at look {look}")


def _entry_decision(seat, entry: EntryResult, status: str, reason: str, score: Score | None) -> EntryDecision:
    return EntryDecision(
        queue_pos=seat.queue_pos,
        uid=seat.reg.uid,
        hotkey=seat.reg.hotkey,
        registration_block=seat.reg.registration_block,
        batch_sha256=seat.batch.batch_sha256,
        status=status,
        reason=reason,
        lora_hash=entry.lora.lora_hash if entry.lora else "",
        score=score,
        eliminated_at_look=entry.eliminated_at_look,
    )


def decide(leases: Sequence[tuple[RoundTicket, BoxResult]]) -> RoundDecision:
    """Settle a round from every lease's ticket and sealed result.

    Each LoRA is compared with the null M that its own box scored. With more
    than one box, the boxes' null-M rates must agree within
    ``calibration_tol``; if not, no score is used and every scored entry is
    ``no_decision``.
    """
    _need(len(leases) >= 1, "a round needs at least one lease")
    first = leases[0][0]
    params, n = first.params, len(first.draw.task_ids)
    for ticket, result in leases:
        _need(
            (ticket.round_id, ticket.head.version_id, ticket.draw, ticket.params.params_hash)
            == (first.round_id, first.head.version_id, first.draw, params.params_hash),
            "every lease of a round shares one head, one draw and one parameter set",
        )
        check_result(ticket, result)

    null_rates, null_caps, problems = [], [], []
    for ticket, result in leases:
        if not any(e.status == SCORED for e in result.entries):
            continue
        run = scored_prefix(result.null_outcomes)  # null M stops early only when every LoRA was dropped
        null_rates.append(weighted_mean(first.draw.weights[:run], pass_rates(result.null_outcomes, run)))
        null_caps.append(code_share(result.null_outcomes, CAP, run))
        if not cover_ok(result.null_outcomes, run, params):
            cover = 1.0 - code_share(result.null_outcomes, ERROR, run)
            problems.append(f"null M on {ticket.box_id}: only {cover:.3f} of rollouts resolved")
    if len(null_rates) > 1 and max(null_rates) - min(null_rates) > params.calibration_tol:
        problems.append("null-M pass rates differ between boxes by more than calibration_tol")

    rows: list[tuple[object, EntryResult, str, str, Score | None]] = []
    candidates: list[tuple[int, Score]] = []
    for ticket, result in leases:
        for seat, entry in zip(ticket.entries, result.entries):
            if entry.status != SCORED:
                rows.append((seat, entry, entry.status, entry.reason, None))
                continue
            score = score_entry(first.draw.weights, result.null_outcomes, entry.outcomes, params, n_total=n)
            if problems:
                rows.append((seat, entry, NO_DECISION, problems[0], score))
            elif not cover_ok(entry.outcomes, score.n_tasks, params):
                rows.append((seat, entry, NO_DECISION, f"only {score.cover:.3f} of this LoRA's rollouts resolved", score))
            else:
                rows.append((seat, entry, LOST, "", score))
                candidates.append((seat.queue_pos, score))

    winner = pick_winner(candidates)
    entries = []
    for seat, entry, status, reason, score in sorted(rows, key=lambda row: row[0].queue_pos):
        if status == LOST:
            status = WINNER if seat.queue_pos == winner else LOST
            reason = "" if status == WINNER else "; ".join(score.reasons) or "another LoRA had a higher delta"
        entries.append(_entry_decision(seat, entry, status, reason, score))

    if winner is not None:
        why = f"entry {winner} beat M by more than {params.z_bar:g} standard error"
    elif problems:
        why = "no decision: " + "; ".join(problems)
    elif not candidates:
        why = "no entry was scored"
    else:
        why = f"null round: no LoRA beat M by more than {params.z_bar:g} standard error"
    return RoundDecision(
        round_id=first.round_id,
        head_version_id=first.head.version_id,
        winner_queue_pos=winner,
        entries=tuple(entries),
        n_tasks=n,
        null_pass_rate=math.fsum(null_rates) / len(null_rates) if null_rates else 0.0,
        null_cap_rate=math.fsum(null_caps) / len(null_caps) if null_caps else 0.0,
        reason=why,
    )
