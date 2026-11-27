"""Anchors: an easy-task regression check beside the gate draw.

Pool rotation retires the tasks the head solves almost always, so the gate
draw never measures them again. Each round with ``anchor_n > 0`` adds a public,
rotating sample of those retired tasks to the evaluation order:

* Eligible: a retired task (in the retired archive and not active this round),
  not withdrawn, with no ``reject`` row in the sealed guard ledger, whose body
  the task catalog resolves, and whose measured head record (passes / rollouts
  while it was active) has at least ``anchor_min_rollouts`` rollouts and a pass
  rate of at least ``anchor_min_p``. The pending crown's training prompts are
  excluded exactly as they are from the gate draw. Never an active, reserve or
  candidate task, so never a gate task.
* Draw: ``round(anchor_n * anchor_hi_share)`` anchors from records
  ``>= anchor_hi_p``, the rest from ``[anchor_min_p, anchor_hi_p)``; a short
  stratum is filled from the other. Within a stratum the rank is
  ``sha256("<round_seed>|anchor|<task_id>")``, smallest first. The round seed
  is taken only after the round's batches are frozen.
* Order: the gate draw keeps its order. The anchors, in rank order, go to the
  evenly spaced positions ``floor((j + u) * N / a)`` of the combined order of
  ``N`` tasks, where ``a`` is the number of anchors and ``u`` comes from
  ``sha256("<round_seed>|anchor-offset")``. Any scored prefix holds a
  proportional share of anchors. Each anchor has weight 1.0.

Anchors never enter delta, the field gate, early drops, confirmation, the
IRT update, the rotation ledger, retirement or admission. They only feed the
one-sided crown veto in ``judging``. All anchor logic stays outside ``gpu/``:
the box evaluates the task list it is given, and its KL sample (chosen inside
``gpu/kl.py``) may include anchor rows. That is harmless: KL measures
divergence from the head, and anchors are public retired tasks.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

from .canonical import sha256_hex
from .params import Params
from .types import Draw

ANCHOR_WEIGHT = 1.0


def anchor_rank(round_seed: str, task_id: str) -> str:
    return sha256_hex(f"{round_seed}|anchor|{task_id}".encode("utf-8"))


def _offset(round_seed: str) -> float:
    return int(sha256_hex(f"{round_seed}|anchor-offset".encode("utf-8")), 16) / float(1 << 256)


def eligible_records(
    retired_archive: Sequence[Mapping[str, Any]],
    retired_ledger: Mapping[str, Sequence[int]],
    active_ids: Iterable[str],
    params: Params,
    *,
    withdrawn: Iterable[str] = (),
    rejected: Iterable[str] = (),
    reserve: Iterable[str] = (),
    exclude: Iterable[str] = (),
    resolvable: Callable[[str], bool] = lambda task_id: True,
) -> dict[str, float]:
    """{task_id: measured record pass rate} of every eligible retired task. A pure function of public state."""
    latest: dict[str, Mapping[str, Any]] = {}
    for row in retired_archive:
        task_id = str(row.get("id", ""))
        key = (int(row.get("event_seq", -1)), str(row.get("round_id", "")))
        old = latest.get(task_id)
        if old is None or (int(old.get("event_seq", -1)), str(old.get("round_id", ""))) < key:
            latest[task_id] = row
    blocked = set(active_ids) | set(withdrawn) | set(rejected) | set(reserve) | set(exclude)
    records: dict[str, float] = {}
    for task_id in sorted(latest):
        row = latest[task_id]
        if not task_id or task_id in blocked or row.get("reason") == "withdrawn":
            continue
        if task_id in retired_ledger:
            rollouts, passes = (int(value) for value in retired_ledger[task_id][:2])
        else:
            rollouts, passes = int(row.get("head_rollouts", 0)), int(row.get("head_passes", 0))
        if rollouts <= 0 or rollouts < params.anchor_min_rollouts:
            continue
        p = passes / rollouts
        if p < params.anchor_min_p or not resolvable(task_id):
            continue
        records[task_id] = p
    return records


def select(records: Mapping[str, float], round_seed: str, params: Params) -> list[str]:
    """The round's anchors in rank order: stratified, then smallest seed rank first within each stratum."""
    n = min(params.anchor_n, len(records))
    if n <= 0:
        return []
    high = sorted((task for task, p in records.items() if p >= params.anchor_hi_p), key=lambda task: anchor_rank(round_seed, task))
    low = sorted((task for task, p in records.items() if p < params.anchor_hi_p), key=lambda task: anchor_rank(round_seed, task))
    want_high = min(round(params.anchor_n * params.anchor_hi_share), n)
    take_high = min(want_high, len(high))
    take_low = min(n - take_high, len(low))
    take_high = min(n - take_low, len(high))  # a short low stratum is filled from the high one
    chosen = high[:take_high] + low[:take_low]
    return sorted(chosen, key=lambda task: anchor_rank(round_seed, task))


def interleave(gate: Draw, anchors: Sequence[str], anchor_p: Sequence[float]) -> Draw:
    """The gate draw with the anchors at evenly spaced, seed-offset positions. No anchor: the gate draw itself."""
    if not anchors:
        return gate
    a = len(anchors)
    total = len(gate.task_ids) + a
    u = _offset(gate.round_seed)
    slots = {min(total - 1, int((j + u) * total / a)): j for j in range(a)}
    ids: list[str] = []
    weights: list[float] = []
    rest = iter(zip(gate.task_ids, gate.weights))
    for position in range(total):
        if position in slots:
            ids.append(anchors[slots[position]])
            weights.append(ANCHOR_WEIGHT)
        else:
            task, weight = next(rest)
            ids.append(task)
            weights.append(weight)
    return Draw(gate.snapshot_id, gate.round_seed, tuple(ids), tuple(weights), tuple(anchors), tuple(anchor_p))


def anchor_positions(draw: Draw) -> tuple[int, ...]:
    anchors = set(draw.anchor_ids)
    return tuple(index for index, task in enumerate(draw.task_ids) if task in anchors)


def gate_positions(draw: Draw) -> tuple[int, ...]:
    anchors = set(draw.anchor_ids)
    return tuple(index for index, task in enumerate(draw.task_ids) if task not in anchors)


def project(texts: Sequence[str], positions: Sequence[int]) -> tuple[str, ...]:
    """Outcome strings restricted to ``positions``. A scored prefix stays a prefix: positions rise."""
    return tuple("".join(text[index] for index in positions) for text in texts)


def gate_draw(draw: Draw) -> Draw:
    """The draw without its anchors: exactly what the gate, IRT, rotation and confirmation see."""
    if not draw.anchor_ids:
        return draw
    keep = gate_positions(draw)
    return Draw(draw.snapshot_id, draw.round_seed, tuple(draw.task_ids[i] for i in keep), tuple(draw.weights[i] for i in keep))


def gate_outcomes(draw: Draw, texts: Sequence[str]) -> tuple[str, ...]:
    return tuple(texts) if not draw.anchor_ids else project(texts, gate_positions(draw))
