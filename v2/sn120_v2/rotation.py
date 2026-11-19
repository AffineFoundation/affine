"""Deterministic public-reserve -> active-pool rotation. Pure: no I/O.

The loader and controller are adapters around this module. Every input of
``rotate`` is published, so a third party can reproduce the whole decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Any, Mapping, Sequence

from .canonical import Record, canonical_json, is_sha256, sha256_hex
from .errors import ConfigError, ContractError, VerifyError
from .irt import DECIMALS, predicted_pass, verify_snapshot
from .params import Params
from .types import CAP, FAIL, PASS

RULE_VERSION = "affine-pool-rotation-v1"
INFO_RULE_VERSION = "affine-pool-rotation-v2"
RANKING_LIMIT = 100_000
RANKING_OUTSIDE_LIMIT = 20_000
BASE_ACTIVE_SIZE = 20_120
WITHDRAWN, HIGH, LOW = "withdrawn", "high", "low"
EXCESS, DISPLACED = "excess", "displaced"


def rule_version(params: Params) -> str:
    return INFO_RULE_VERSION if params.rotation_rule == "info" else RULE_VERSION


def rule_params(params: Params) -> dict[str, Any]:
    names = ("rotation_enabled", "rotation_k", "rotation_retire_p", "rotation_n_min",
             "rotation_retire_low_p", "rotation_n_min_low", "rotation_active_size")
    if params.rotation_rule == "info":
        names += ("rotation_rule", "rotation_churn_frac", "rotation_prior_p", "rotation_queue_publish", "rotation_churn_max")
        if params.rotation_unscreened_max:
            names += ("rotation_unscreened_max",)
    return {name: getattr(params, name) for name in names}


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
    ``candidate_meta`` is optional public metadata for the ranked v2 rule:
    ``{id: {prior_p?, source_prefix, env}}``. It carries no task text.
    ``audit_verdicts`` carries the reserve row's screening provenance for the
    unscreened-admission cap, independently of the display metadata.
    """
    manifest_sha: str
    queue: tuple[str, ...] = ()
    shards: tuple[ReserveShard, ...] = ()
    withdrawn: tuple[dict[str, Any], ...] = ()
    withdrawn_sha: str = ""
    withdrawn_count: int = 0
    candidate_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    LEFT_OUT_AT_DEFAULT = ("candidate_meta", "audit_verdicts")
    audit_verdicts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_sha256(self.manifest_sha):
            raise ContractError("reserve manifest_sha must be a sha256")
        if len(set(self.queue)) != len(self.queue):
            raise ContractError("the reserve queue repeats a task id")
        if any(task_id not in self.queue or verdict not in ("keep", "unscreened") for task_id, verdict in self.audit_verdicts.items()):
            raise ContractError("reserve audit verdicts must belong to queued tasks")

    @property
    def withdrawn_ids(self) -> frozenset[str]:
        return frozenset(str(row["id"]) for row in self.withdrawn)

    def history(self) -> dict[str, Any]:
        history = self.to_dict()
        if not self.candidate_meta:
            history.pop("candidate_meta", None)
        return history


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
    LEFT_OUT_AT_DEFAULT = ("candidate_ranking", "retire_reason_counts", "restored_irt_rows", "initial_retired_archive", "initial_candidate_meta", "initial_retired_ledger", "retired_ledger", "opened_at", "ranking_complete")
    candidate_ranking: tuple[dict[str, Any], ...] = ()
    ranking_complete: bool = True
    retire_reason_counts: dict[str, int] = field(default_factory=dict)
    restored_irt_rows: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    initial_retired_archive: tuple[dict[str, Any], ...] = ()
    initial_candidate_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    initial_retired_ledger: dict[str, tuple[int, int]] = field(default_factory=dict)
    retired_ledger: dict[str, tuple[int, int]] = field(default_factory=dict)
    opened_at: int = 0

    def __post_init__(self) -> None:
        if active_sha(self.active_ids) != self.active_sha or set(self.admitted) - set(self.active_ids):
            raise VerifyError("rotation active ids do not match active_sha")
        if self.queue_remaining < 0 or self.rule_version not in (RULE_VERSION, INFO_RULE_VERSION):
            raise VerifyError("unknown pool rotation rule")
        if self.rule_version == RULE_VERSION and (self.candidate_ranking or self.retire_reason_counts or self.restored_irt_rows or self.initial_retired_archive or self.initial_candidate_meta or self.initial_retired_ledger or self.retired_ledger):
            raise VerifyError("v1 rotation cannot carry v2 ranked-pool state")
        if self.rule_version == INFO_RULE_VERSION and not self.candidate_ranking:
            raise VerifyError("v2 rotation lacks its public candidate ranking")

    def journal_block(self) -> dict[str, Any]:
        block = {
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
        if self.rule_version == INFO_RULE_VERSION:
            block["retire_reason_counts"] = self.retire_reason_counts
        return block

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
            **({"ranking": layout.pool_ranking(self.round_id), "summary": layout.POOL_SUMMARY,
                "rule_params": self.rule_params, "retire_reason_counts": self.retire_reason_counts}
               if self.rule_version == INFO_RULE_VERSION else {}),
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
        rule_params=rule_params(params),
    )


def _prior(row: Mapping[str, Any], fallback: float) -> float:
    value = row.get("prior_p", fallback)
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ContractError("candidate prior_p must be a finite number in [0, 1]")
    return float(value)


def _candidate_meta(reserve: Reserve, task_id: str) -> Mapping[str, Any]:
    return reserve.candidate_meta.get(task_id, {})


def rotate_info(
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
    initial_candidate_meta: Mapping[str, Mapping[str, Any]] | None = None,
    initial_retired_ledger: Mapping[str, Sequence[int]] | None = None,
) -> Rotation:
    """Rank every non-withdrawn candidate, then shrink, fill and swap.

    The candidate list is complete and published. Its state describes the
    previous round, not the new active set. All order ties use
    sha256(round_id|rank|task_id); no iteration or file order can change a
    decision. Withdrawing is permanent: no retired row is a new candidate.
    """
    if not params.rotation_enabled or params.rotation_rule != "info":
        raise ConfigError("rotate_info requires rotation_enabled=true and rotation_rule=info")
    if initial_candidate_meta:
        reserve = Reserve.from_dict(reserve.to_dict() | {"candidate_meta": dict(initial_candidate_meta)})
    input_archive = sorted(retired_archive, key=lambda item: (int(item.get("event_seq", -1)), str(item.get("round_id", "")), str(item.get("id", ""))))
    ids = tuple(sorted(A_prev))
    previous_sha = active_sha(ids)
    if not ids or len(reserve.queue) != len(set(reserve.queue)):
        raise ContractError("rotation needs a nonempty active set and a unique reserve queue")
    ability = head_ability(irt_snapshot)
    tasks = irt_snapshot["tasks"]
    active_set = set(ids)
    if active_set & set(reserve.queue):
        raise VerifyError("a reserve candidate is already active")
    withdrawn = reserve.withdrawn_ids
    archives: dict[str, Mapping[str, Any]] = {}
    for row in input_archive:
        task_id = str(row.get("id", ""))
        if task_id in active_set:
            continue  # a historical retirement followed by a return is stale
        old = archives.get(task_id)
        if old is None or (int(old.get("event_seq", -1)), str(old.get("round_id", ""))) < (int(row.get("event_seq", -1)), str(row.get("round_id", ""))):
            archives[task_id] = row
    known = set(archives) - withdrawn
    if set(reserve.queue) & (active_set | known):
        raise VerifyError("the ranked rule needs a reserve queue disjoint from active and retired tasks")
    if set(irt_snapshot["tasks"]) != active_set:
        raise VerifyError("the ranked rule requires IRT rows exactly matching the previous active set")

    rows: dict[str, dict[str, Any]] = {}
    for task_id, state in [(task_id, "active") for task_id in ids] + [(task_id, "reserve") for task_id in reserve.queue]:
        row = tasks.get(task_id)
        meta = _candidate_meta(reserve, task_id)
        unscreened = bool(meta.get("unscreened")) or reserve.audit_verdicts.get(task_id) == "unscreened"
        rolls, passes = _ledger_row(ledger, task_id)
        if state == "active" and row is not None:
            rounds = int(row[2])
            p = predicted_pass(ability, float(row[0])) if rounds >= params.rotation_n_min else params.rotation_prior_p
        else:
            meta = _candidate_meta(reserve, task_id)
            p = _prior(meta, params.rotation_prior_p)
            rounds = 0
        flags = [HIGH] if rounds >= params.rotation_n_min and p > params.rotation_retire_p else []
        if rolls >= params.rotation_n_min_low and passes == 0:
            flags.append(LOW)
        rows[task_id] = {"id": task_id, "state": state, "p": p, "rounds_seen": rounds,
                         "score": 0.0 if LOW in flags else p * (1.0 - p), "flags": flags,
                         "source_prefix": str(meta.get("source_prefix") or task_id.split(":", 1)[0]),
                         "env": str(meta.get("env") or "affine_math"), "head_rollouts": rolls, "head_passes": passes,
                         **({"unscreened": True} if unscreened else {}),
                         "admitted": False, "retired": False, "retire_reason": None,
                         "tie": sha256_hex(f"{round_id}|rank|{task_id}".encode())}
    for task_id, old in archives.items():
        if task_id in withdrawn or old.get("reason") == WITHDRAWN:
            continue
        meta = {
            key: old[key] for key in ("source_prefix", "env", "prior_p") if key in old
        } | dict(_candidate_meta(reserve, task_id))
        unscreened = old.get("audit_verdict") == "unscreened" or reserve.audit_verdicts.get(task_id) == "unscreened"
        row = old.get("irt_row") or tasks.get(task_id)
        rounds = int(row[2]) if row and len(row) == 3 else 0
        p = predicted_pass(ability, float(row[0])) if row and rounds >= params.rotation_n_min else _prior(meta, params.rotation_prior_p)
        rolls, passes = _ledger_row(ledger, task_id) if task_id in ledger else (int(old.get("head_rollouts", 0)), int(old.get("head_passes", 0)))
        flags = [HIGH] if rounds >= params.rotation_n_min and p > params.rotation_retire_p else []
        if rolls >= params.rotation_n_min_low and passes == 0:
            flags.append(LOW)
        rows[task_id] = {"id": task_id, "state": "retired", "p": p, "rounds_seen": rounds,
                         "score": 0.0 if LOW in flags else p * (1.0 - p), "flags": flags,
                         "source_prefix": str(meta.get("source_prefix") or task_id.split(":", 1)[0]),
                         "env": str(meta.get("env") or old.get("env") or "affine_math"),
                         "head_rollouts": rolls, "head_passes": passes, "admitted": False, "retired": False,
                         "retire_reason": None, "tie": sha256_hex(f"{round_id}|rank|{task_id}".encode())}
        if unscreened:
            rows[task_id]["unscreened"] = True

    def ascending(task_id: str) -> tuple[float, str, str]:
        row = rows[task_id]
        return row["score"], row["tie"], row["id"]

    def descending(task_id: str) -> tuple[float, str, str]:
        row = rows[task_id]
        return -row["score"], row["tie"], row["id"]

    target = params.rotation_active_size or initial_size
    if target < 1:
        raise ConfigError("the ranked rule needs a positive active-set target")
    if not (active_set | set(reserve.queue) | set(archives)) - withdrawn:
        raise ConfigError("rotation would leave an empty active set")
    kept = active_set - withdrawn
    retired: dict[str, Retired] = {}
    archive = list(input_archive)

    def retire(task_id: str, reason: str) -> None:
        row = rows[task_id]
        old = archives.get(task_id, {})
        saved = tasks.get(task_id) or old.get("irt_row")
        retired[task_id] = Retired(task_id, reason, row["p"], row["rounds_seen"],
                                   tuple(saved[:3]) if saved else None, row["head_rollouts"], row["head_passes"])
        row["retired"], row["retire_reason"] = True, reason
        meta = dict(_candidate_meta(reserve, task_id))
        archive.append({"id": task_id, "round_id": round_id, "reason": reason, "p_or_rate": row["p"],
                        "rounds_seen": row["rounds_seen"], "event_seq": len(archive),
                        **({"irt_row": list(saved[:3])} if saved else {}),
                        "head_rollouts": row["head_rollouts"], "head_passes": row["head_passes"],
                        "source_prefix": row["source_prefix"], "env": row["env"],
                        **({"prior_p": meta["prior_p"]} if "prior_p" in meta else {}),
                        **({"audit_verdict": "unscreened"} if row.get("unscreened", False) else {})})

    for task_id in sorted(active_set & withdrawn):
        retire(task_id, WITHDRAWN)
    if len(kept) > target:
        for task_id in sorted(kept, key=ascending)[:len(kept) - target]:
            flags = rows[task_id]["flags"]
            retire(task_id, HIGH if HIGH in flags else LOW if LOW in flags else EXCESS)
            kept.remove(task_id)
    outside = (set(reserve.queue) | set(archives)) - withdrawn - kept - {task for task, row in archives.items() if row.get("reason") == WITHDRAWN}
    unscreened_admissions = 0

    def admit(task_id: str) -> None:
        nonlocal unscreened_admissions
        kept.add(task_id)
        outside.remove(task_id)
        rows[task_id]["admitted"] = True
        unscreened_admissions += int(rows[task_id].get("unscreened", False))

    if len(kept) < target:
        ranked = sorted(outside, key=descending)
        selected: list[str] = []
        for task_id in ranked:
            if len(selected) == target - len(kept):
                break
            if params.rotation_unscreened_max and rows[task_id].get("unscreened", False) and unscreened_admissions >= params.rotation_unscreened_max:
                continue  # stays queued for a later round
            selected.append(task_id)
            unscreened_admissions += int(rows[task_id].get("unscreened", False))
        for task_id in selected:
            admit(task_id)
    cap = int(params.rotation_churn_frac * target) if params.rotation_churn_max == 0 else min(int(params.rotation_churn_frac * target), params.rotation_churn_max)
    active_heap = [ascending(task_id) for task_id in kept]
    outside_heap = [descending(task_id) for task_id in outside]
    heapq.heapify(active_heap)
    heapq.heapify(outside_heap)
    swaps = 0
    deferred: list[tuple[float, str, str]] = []
    while swaps < cap and active_heap and outside_heap:
        task_id = active_heap[0][2]
        candidate = outside_heap[0][2]
        if rows[candidate]["score"] <= rows[task_id]["score"]:
            break
        if params.rotation_unscreened_max and rows[candidate].get("unscreened", False) and unscreened_admissions >= params.rotation_unscreened_max:
            deferred.append(heapq.heappop(outside_heap))  # capped, but still first in rank for a later round
            continue
        heapq.heappop(active_heap)
        heapq.heappop(outside_heap)
        flags = rows[task_id]["flags"]
        retire(task_id, HIGH if HIGH in flags else LOW if LOW in flags else DISPLACED)
        kept.remove(task_id)
        outside.add(task_id)
        admit(candidate)
        heapq.heappush(outside_heap, descending(task_id))
        swaps += 1
    for item in deferred:
        heapq.heappush(outside_heap, item)
    active = tuple(sorted(kept))
    ranking = tuple({key: value for key, value in row.items() if key != "tie"}
                    for row in sorted(rows.values(), key=lambda row: (-row["score"], row["tie"], row["id"]))
                    if row["retire_reason"] != WITHDRAWN)
    complete = len(rows) <= RANKING_LIMIT
    if not complete:
        published = {row["id"] for row in ranking if row["state"] in ("active", "retired") or row["retired"] or row["id"] in kept}
        outside_rows = [row for row in ranking if row["id"] not in published]
        published.update(row["id"] for row in outside_rows[:RANKING_OUTSIDE_LIMIT])
        ranking = tuple(row for row in ranking if row["id"] in published)
    admitted = tuple(task_id for task_id, row in sorted(rows.items(), key=lambda item: (-item[1]["score"], item[1]["tie"], item[0])) if row["admitted"])
    upcoming = tuple(row["id"] for row in ranking if row["id"] in outside and row["id"] not in withdrawn)
    restored = {task_id: tuple(archives[task_id]["irt_row"][:3]) for task_id in admitted if task_id in archives and archives[task_id].get("irt_row")}
    counts = {reason: sum(row.reason == reason for row in retired.values()) for reason in sorted({row.reason for row in retired.values()})}
    return Rotation(
        round_id=round_id, active_ids=active, active_sha=active_sha(active), prev_active_sha=previous_sha,
        admitted=tuple(task_id for task_id, row in sorted(rows.items(), key=lambda item: (-item[1]["score"], item[1]["tie"], item[0])) if row["admitted"]),
        retired=tuple(retired.values()), reserve_manifest_sha=reserve.manifest_sha,
        queue_remaining=len(set(reserve.queue) - kept - withdrawn), next_up=upcoming[:params.rotation_queue_publish],
        reserve_history=reserve.history(), ledger={key: (rows[key]["head_rollouts"], rows[key]["head_passes"]) for key in active},
        retired_archive=tuple(archive), source_snapshot_id=source_snapshot_id,
        prev_state_sha=prev_state_sha, initial_active_sha=previous_sha if not prev_state_sha else "",
        rule_params=rule_params(params),
        rule_version=INFO_RULE_VERSION, candidate_ranking=ranking, ranking_complete=complete, retire_reason_counts=counts,
        restored_irt_rows=restored,
        retired_ledger={key: _ledger_row(initial_retired_ledger or {}, key) for key in (initial_retired_ledger or {})} |
                       {row.id: (row.head_rollouts, row.head_passes) for row in retired.values()},
        initial_retired_archive=tuple(input_archive) if initial_candidate_meta is not None else (),
        initial_candidate_meta={key: dict(value) for key, value in (initial_candidate_meta or {}).items()},
        initial_retired_ledger={key: _ledger_row(initial_retired_ledger or {}, key) for key in (initial_retired_ledger or {})},
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
