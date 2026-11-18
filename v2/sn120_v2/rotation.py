"""Deterministic public-reserve -> active-pool rotation. Pure: no I/O.

The loader and controller are adapters around this module. Every input of
``rotate`` is published, so a third party can reproduce the whole decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .canonical import Record, canonical_json, is_sha256, sha256_hex
from .errors import ConfigError, ContractError, VerifyError
from .irt import DECIMALS, predicted_pass, verify_snapshot
from .params import Params
from .types import CAP, FAIL, PASS

RULE_VERSION = "affine-pool-rotation-v1"
BASE_ACTIVE_SIZE = 20_120
WITHDRAWN, HIGH, LOW = "withdrawn", "high", "low"


def active_sha(task_ids: Sequence[str]) -> str:
    ids = sorted(task_ids)
    if len(set(ids)) != len(ids) or any(not isinstance(task, str) or not task for task in ids):
        raise ContractError("an active set needs distinct, nonempty task ids")
    return sha256_hex(canonical_json(ids))


@dataclass(frozen=True)
class ReserveShard:
    shard: str
    sha256: str
    count: int
    added_at: str


@dataclass(frozen=True)
class Reserve(Record):
    """The public part of a reserve needed by the rule.

    ``queue`` is already in manifest order and within-shard hash order, with
    old admissions and withdrawals removed. It therefore also makes a saved
    rotation independently replayable without a git checkout.
    """
    manifest_sha: str
    queue: tuple[str, ...] = ()
    shards: tuple[ReserveShard, ...] = ()
    withdrawn: tuple[dict[str, Any], ...] = ()
    withdrawn_sha: str = ""
    withdrawn_count: int = 0

    def __post_init__(self) -> None:
        if not is_sha256(self.manifest_sha):
            raise ContractError("reserve manifest_sha must be a sha256")
        if len(set(self.queue)) != len(self.queue):
            raise ContractError("the reserve queue repeats a task id")

    @property
    def withdrawn_ids(self) -> frozenset[str]:
        return frozenset(str(row["id"]) for row in self.withdrawn)

    def history(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class Retired(Record):
    id: str
    reason: str
    p_or_rate: float | None = None
    rounds_seen: int = 0
    irt_row: tuple[float, float, int] | None = None
    head_rollouts: int = 0
    head_passes: int = 0

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "reason": self.reason}


@dataclass(frozen=True)
class Rotation(Record):
    round_id: str
    active_ids: tuple[str, ...]
    active_sha: str
    prev_active_sha: str
    admitted: tuple[str, ...]
    retired: tuple[Retired, ...]
    reserve_manifest_sha: str
    queue_remaining: int
    next_up: tuple[str, ...]
    reserve_history: dict[str, Any]
    ledger: dict[str, tuple[int, int]]
    retired_archive: tuple[dict[str, Any], ...] = ()
    source_snapshot_id: str = ""
    prev_state_sha: str = ""
    initial_active_sha: str = ""
    rule_params: dict[str, Any] = field(default_factory=dict)
    rule_version: str = RULE_VERSION

    def __post_init__(self) -> None:
        if active_sha(self.active_ids) != self.active_sha or set(self.admitted) - set(self.active_ids):
            raise VerifyError("rotation active ids do not match active_sha")
        if self.queue_remaining < 0 or self.rule_version != RULE_VERSION:
            raise VerifyError("not an affine-pool-rotation-v1 decision")

    def journal_block(self) -> dict[str, Any]:
        return {
            "active_sha": self.active_sha,
            "prev_active_sha": self.prev_active_sha,
            "admitted": list(self.admitted),
            "retired": [row.public() for row in self.retired],
            "reserve_manifest_sha": self.reserve_manifest_sha,
            "queue_remaining": self.queue_remaining,
            "next_up": list(self.next_up),
            "rule_version": self.rule_version,
            "prev_state_sha": self.prev_state_sha,
        }

    def pool_block(self) -> dict[str, Any]:
        from . import layout  # key naming only; still no I/O

        return {
            "active_sha": self.active_sha,
            "size": len(self.active_ids),
            "admitted": list(self.admitted),
            "retired": [row.public() for row in self.retired],
            "reserve_manifest_sha": self.reserve_manifest_sha,
            "queue_remaining": self.queue_remaining,
            "next_up": list(self.next_up),
            "active": layout.active_pool(self.active_sha),
            "rotation": layout.pool_rotation(self.round_id),
            "retired_archive": layout.RETIRED_POOL,
            "reserve_manifest": layout.RESERVE_MANIFEST,
        }


def _tie(round_id: str, task_id: str) -> str:
    return sha256_hex(f"{round_id}|retire|{task_id}".encode("utf-8"))


def head_ability(irt_snapshot: Mapping[str, Any]) -> float:
    """Exactly public_snapshot's head ability, including an unseen head's zero."""
    verify_snapshot(irt_snapshot)
    return float(irt_snapshot["heads"].get(irt_snapshot["head_version_id"], {}).get("ability", 0.0))


def _ledger_row(ledger: Mapping[str, Sequence[int]], task_id: str) -> tuple[int, int]:
    row = ledger.get(task_id, (0, 0))
    if len(row) != 2 or any(type(value) is not int or value < 0 for value in row) or row[1] > row[0]:
        raise ContractError(f"invalid head ledger counts for {task_id!r}")
    return int(row[0]), int(row[1])


def rotate(
    round_id: str,
    A_prev: Sequence[str],
    irt_snapshot: Mapping[str, Any],
    ledger: Mapping[str, Sequence[int]],
    reserve: Reserve,
    params: Params,
    *,
    initial_size: int = BASE_ACTIVE_SIZE,
    retired_archive: Sequence[Mapping[str, Any]] = (),
    source_snapshot_id: str = "",
    prev_state_sha: str = "",
) -> Rotation:
    """Return the active set for ``round_id``; never mutate an input.

    W = withdrawn active ids (outside k).
    H = {t: rounds_seen >= n_min and sigmoid(head_ability-difficulty) > retire_p}.
    L = {t: head_rollouts >= n_min_low and empirical_passes/rollouts < retire_low_p},
    only when retire_low_p > 0. H is p-descending, L empirical-rate-ascending,
    then sha256(round_id|retire|task_id).

    target = rotation_active_size or initial_size; avail = |Q|.
    n_retire = min(k, |H union L|, avail + max(0, |A_prev\\W|-target)).
    n_admit = clamp(target-(|A_prev|-|W|-n_retire), 0, min(avail,k+|W|)).
    """
    if not params.rotation_enabled:
        raise ConfigError("rotate requires rotation_enabled=true")
    ids = tuple(sorted(A_prev))
    previous_sha = active_sha(ids)
    if not ids or len(reserve.queue) != len(set(reserve.queue)):
        raise ContractError("rotation needs a nonempty active set and a unique reserve queue")
    ability = head_ability(irt_snapshot)
    tasks = irt_snapshot["tasks"]
    withdrawn = sorted(set(ids) & reserve.withdrawn_ids)
    retired: dict[str, Retired] = {}
    for task_id in withdrawn:
        row = tasks.get(task_id)
        rolls, passes = _ledger_row(ledger, task_id)
        retired[task_id] = Retired(task_id, WITHDRAWN, None, int(row[2]) if row else 0,
                                   tuple(row[:3]) if row else None, rolls, passes)

    eligible: list[tuple[str, float, float | None]] = []
    if reserve.queue:
        for task_id in sorted(set(ids) - set(withdrawn)):
            row = tasks.get(task_id)
            rolls, passes = _ledger_row(ledger, task_id)
            if row and row[2] >= params.rotation_n_min:
                p = predicted_pass(ability, float(row[0]))
                if p > params.rotation_retire_p:
                    eligible.append((task_id, -p, None))
                    continue
            if params.rotation_retire_low_p > 0 and rolls >= params.rotation_n_min_low:
                rate = passes / rolls
                if rate < params.rotation_retire_low_p:
                    eligible.append((task_id, 0.0, rate))
    # All high candidates precede all low candidates, as specified. The second
    # key is descending p for high and ascending empirical rate for low.
    eligible.sort(key=lambda item: (item[2] is not None, item[1] if item[2] is None else item[2], _tie(round_id, item[0])))
    target = params.rotation_active_size or initial_size
    avail = len(reserve.queue)
    n_retire = min(params.rotation_k, len(eligible), avail + max(0, len(ids) - len(withdrawn) - target))
    for task_id, negative_p, low_rate in eligible[:n_retire]:
        row = tasks[task_id]
        rolls, passes = _ledger_row(ledger, task_id)
        retired[task_id] = Retired(task_id, LOW if low_rate is not None else HIGH,
                                   low_rate if low_rate is not None else -negative_p,
                                   int(row[2]), tuple(row[:3]), rolls, passes)
    n_admit = max(0, min(target - (len(ids) - len(withdrawn) - n_retire), avail, params.rotation_k + len(withdrawn)))
    admitted = reserve.queue[:n_admit]
    overlap = (set(ids) - set(withdrawn) - set(retired)) & set(admitted)
    if overlap:
        raise VerifyError(f"reserve would admit an already active task: {sorted(overlap)[0]}")
    active = tuple(sorted((set(ids) - set(withdrawn) - set(retired)) | set(admitted)))
    if not active:
        raise ConfigError("rotation would leave an empty active set")
    archive = list(retired_archive)
    archive.extend({"id": row.id, "round_id": round_id, "reason": row.reason,
                    "p_or_rate": row.p_or_rate, "rounds_seen": row.rounds_seen,
                    "event_seq": len(retired_archive) + position} for position, row in enumerate(retired.values()))
    return Rotation(
        round_id=round_id, active_ids=active, active_sha=active_sha(active), prev_active_sha=previous_sha,
        admitted=tuple(admitted), retired=tuple(retired.values()), reserve_manifest_sha=reserve.manifest_sha,
        queue_remaining=avail - n_admit, next_up=reserve.queue[n_admit:n_admit + 2 * params.rotation_k],
        reserve_history=reserve.history(), ledger={key: tuple(_ledger_row(ledger, key)) for key in sorted(active)},
        retired_archive=tuple(archive), source_snapshot_id=source_snapshot_id,
        prev_state_sha=prev_state_sha, initial_active_sha=previous_sha if not prev_state_sha else "",
        rule_params={name: getattr(params, name) for name in Params.LEFT_OUT_AT_DEFAULT if name.startswith("rotation_")},
    )


def fold_ledger(
    ledger: Mapping[str, Sequence[int]], active_ids: Sequence[str], draw_ids: Sequence[str], null_outcomes: Sequence[str]
) -> dict[str, tuple[int, int]]:
    """Fold resolved null-M rollouts; CAP is a fail, exactly as in IRT."""
    if len(draw_ids) == 0:
        raise ContractError("a ledger fold needs a draw")
    for text in null_outcomes:
        if len(text) != len(draw_ids):
            raise ContractError("ledger outcomes do not match the draw")
    counts = {task: list(_ledger_row(ledger, task)) for task in active_ids}
    for text in null_outcomes:
        for task_id, code in zip(draw_ids, text):
            if task_id in counts and code in (PASS, FAIL, CAP):
                counts[task_id][0] += 1
                counts[task_id][1] += int(code == PASS)
    return {task: tuple(row) for task, row in sorted(counts.items())}


def reserve_from_history(history: Mapping[str, Any]) -> Reserve:
    """Strict public-artifact -> rule-input conversion."""
    return Reserve.from_dict(history)
