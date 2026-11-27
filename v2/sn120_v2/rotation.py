"""Deterministic public-reserve -> active-pool rotation. Pure: no I/O.

The loader and controller are adapters around this module. Every input of
``rotate`` is published, so a third party can reproduce the whole decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import json
import math
import re
from typing import Any, Mapping, Sequence

from .canonical import Record, canonical_json, is_sha256, sha256_hex
from .errors import ConfigError, ContractError, VerifyError
from .irt import DECIMALS, predicted_pass, verify_snapshot
from .params import Params
from .types import CAP, FAIL, PASS

RULE_VERSION = "affine-pool-rotation-v1"
INFO_RULE_VERSION = "affine-pool-rotation-v2"
GUARD_RULE_VERSION = "affine-pool-rotation-v3"
INFO_RULES = ("info", "info-guard")
RANKED_RULE_VERSIONS = (INFO_RULE_VERSION, GUARD_RULE_VERSION)
BASE_ACTIVE_SIZE = 20_120
WITHDRAWN, HIGH, LOW = "withdrawn", "high", "low"
EXCESS, DISPLACED = "excess", "displaced"


def ranked(rule_version_or_rule: str) -> bool:
    """True for every ranked rule, by rule version or by ``rotation_rule``."""
    return rule_version_or_rule in RANKED_RULE_VERSIONS or rule_version_or_rule in INFO_RULES


def guarded(params: Params) -> bool:
    return params.rotation_rule == "info-guard"


def rule_version(params: Params) -> str:
    if params.rotation_rule == "info-guard":
        return GUARD_RULE_VERSION
    return INFO_RULE_VERSION if params.rotation_rule == "info" else RULE_VERSION


def rule_params(params: Params) -> dict[str, Any]:
    names = ("rotation_enabled", "rotation_k", "rotation_retire_p", "rotation_n_min",
             "rotation_retire_low_p", "rotation_n_min_low", "rotation_active_size")
    if params.rotation_rule in INFO_RULES:
        names += ("rotation_rule", "rotation_churn_frac", "rotation_prior_p", "rotation_queue_publish", "rotation_churn_max")
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
    """
    LEFT_OUT_AT_DEFAULT = ("candidate_meta",)
    manifest_sha: str
    queue: tuple[str, ...] = ()
    shards: tuple[ReserveShard, ...] = ()
    withdrawn: tuple[dict[str, Any], ...] = ()
    withdrawn_sha: str = ""
    withdrawn_count: int = 0
    candidate_meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_sha256(self.manifest_sha):
            raise ContractError("reserve manifest_sha must be a sha256")
        if len(set(self.queue)) != len(self.queue):
            raise ContractError("the reserve queue repeats a task id")

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


GUARD_ROW_FIELDS = frozenset({"id", "content_sha", "verdict", "category", "guard_version", "added_at"})
GUARD_PASS, GUARD_REJECT = "pass", "reject"
_GUARD_ID = re.compile(r"([a-z][a-z0-9_]*):([0-9]+)\Z")
_GUARD_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")  # UTC calendar date, as everywhere else


@dataclass(frozen=True)
class GuardLedger:
    """The sealed public admission ledger: one decision per (id, content_sha).

    Pure rule input. It carries no judge reasoning, no proposed answer and no
    task text; see ``sn120_v2.guard`` for how a row is produced. ``sha256`` is
    the hash of the exact published bytes, which is what a Rotation records.
    """
    sha256: str
    count: int = 0
    passes: dict[str, str] = field(default_factory=dict)   # id -> content_sha
    rejects: dict[str, str] = field(default_factory=dict)  # id -> content_sha

    def eligible(self, task_id: str, content_sha: str | None) -> bool:
        """Admission is allowed only by a recorded pass for the current body."""
        return bool(content_sha) and self.passes.get(task_id) == content_sha


EMPTY_GUARD = GuardLedger(sha256_hex(b""), 0)


def parse_guard_ledger(raw: bytes) -> GuardLedger:
    """Strict canonical-JSONL parse. One row per (id, content_sha); never two."""
    passes: dict[str, str] = {}
    rejects: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    lines = raw.splitlines(keepends=True) if raw else []
    if raw and not raw.endswith(b"\n"):
        raise VerifyError("guard ledger rows must each end with a newline")
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise VerifyError(f"guard ledger: invalid canonical JSONL: {exc}") from exc
        if not isinstance(row, dict) or set(row) != GUARD_ROW_FIELDS:
            raise VerifyError("guard ledger row has a bad schema")
        if any(not isinstance(row[name], str) for name in GUARD_ROW_FIELDS):
            raise VerifyError("every guard ledger field must be a string")
        if not _GUARD_ID.fullmatch(row["id"]) or not is_sha256(row["content_sha"]):
            raise VerifyError("guard ledger row has a bad task id or content_sha")
        if row["verdict"] not in (GUARD_PASS, GUARD_REJECT) or not row["category"].strip():
            raise VerifyError("guard ledger row has a bad verdict or category")
        if row["category"] != row["category"].lower() or not row["guard_version"].strip():
            raise VerifyError("guard ledger category must be lowercase and guard_version nonempty")
        if not _GUARD_DATE.fullmatch(row["added_at"]):
            raise VerifyError("guard ledger added_at must be a UTC YYYY-MM-DD date")
        if canonical_json(row) + b"\n" != line:
            raise VerifyError("guard ledger rows must be canonical JSONL, one newline-terminated row each")
        key = (row["id"], row["content_sha"])
        if key in seen:
            raise VerifyError(f"guard ledger decides {row['id']} twice for the same body")
        seen.add(key)
        (passes if row["verdict"] == GUARD_PASS else rejects)[row["id"]] = row["content_sha"]
    return GuardLedger(sha256_hex(raw), len(lines), passes, rejects)


def extends_guard_bytes(previous: bytes, current: bytes) -> None:
    """Append-only on the bytes, exactly like withdrawn.jsonl."""
    if not current.startswith(previous):
        raise VerifyError("guard ledger shrank or changed an earlier row")


def extends_guard(previous: GuardLedger, current: GuardLedger) -> None:
    """The same rule as far as two Rotation records can state it.

    A published rotation records the ledger's sha and count, not its rows, so
    this is the count-and-identity half; ``extends_guard_bytes`` is the full
    check and runs wherever the bytes themselves are loaded.
    """
    if current.count < previous.count:
        raise VerifyError("guard ledger shrank")
    if current.count == previous.count and current.sha256 != previous.sha256:
        raise VerifyError("guard ledger changed an earlier row")


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
    LEFT_OUT_AT_DEFAULT = ("candidate_ranking", "retire_reason_counts", "restored_irt_rows", "initial_retired_archive", "initial_candidate_meta", "initial_retired_ledger", "retired_ledger", "opened_at",
                           "guard_ledger_sha", "guard_ledger_count", "guard_pass", "guard_reject", "guard_shortfall", "next_up_eligible")
    candidate_ranking: tuple[dict[str, Any], ...] = ()
    retire_reason_counts: dict[str, int] = field(default_factory=dict)
    restored_irt_rows: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    initial_retired_archive: tuple[dict[str, Any], ...] = ()
    initial_candidate_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    initial_retired_ledger: dict[str, tuple[int, int]] = field(default_factory=dict)
    retired_ledger: dict[str, tuple[int, int]] = field(default_factory=dict)
    opened_at: int = 0
    # The sealed public guard ledger of a v3 (info-guard) decision. Only its
    # content id, size and the counts are recorded here; the rows themselves
    # are the published object at ``layout.pool_guard(guard_ledger_sha)``.
    guard_ledger_sha: str = ""
    guard_ledger_count: int = 0
    guard_pass: int = 0
    guard_reject: int = 0
    guard_shortfall: int = 0
    next_up_eligible: int = 0

    def __post_init__(self) -> None:
        if active_sha(self.active_ids) != self.active_sha or set(self.admitted) - set(self.active_ids):
            raise VerifyError("rotation active ids do not match active_sha")
        if self.queue_remaining < 0 or self.rule_version not in (RULE_VERSION, INFO_RULE_VERSION, GUARD_RULE_VERSION):
            raise VerifyError("unknown pool rotation rule")
        if self.rule_version == RULE_VERSION and (self.candidate_ranking or self.retire_reason_counts or self.restored_irt_rows or self.initial_retired_archive or self.initial_candidate_meta or self.initial_retired_ledger or self.retired_ledger):
            raise VerifyError("v1 rotation cannot carry v2 ranked-pool state")
        if self.rule_version in RANKED_RULE_VERSIONS and not self.candidate_ranking:
            raise VerifyError("a ranked rotation lacks its public candidate ranking")
        guard_state = (self.guard_ledger_sha, self.guard_ledger_count, self.guard_pass, self.guard_reject, self.guard_shortfall, self.next_up_eligible)
        if self.rule_version != GUARD_RULE_VERSION and any(guard_state):
            raise VerifyError("only an info-guard rotation may carry guard-ledger state")
        if self.rule_version == GUARD_RULE_VERSION:
            if not is_sha256(self.guard_ledger_sha):
                raise VerifyError("an info-guard rotation must seal a guard ledger by sha256")
            if min(self.guard_ledger_count, self.guard_pass, self.guard_reject, self.guard_shortfall, self.next_up_eligible) < 0:
                raise VerifyError("guard-ledger counts must not be negative")
            if self.guard_pass + self.guard_reject > self.guard_ledger_count:
                raise VerifyError("guard-ledger decisions exceed its row count")

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
        if self.rule_version in RANKED_RULE_VERSIONS:
            block["retire_reason_counts"] = self.retire_reason_counts
        if self.rule_version == GUARD_RULE_VERSION:
            block |= self.guard_block()
        return block

    def guard_block(self) -> dict[str, Any]:
        """Public counts of the sealed guard ledger. No candidate is named.

        Empty for a round under an earlier rule: a v1 or v2 round has no guard,
        and nothing it publishes may suggest it had one.
        """
        if self.rule_version != GUARD_RULE_VERSION:
            return {}
        return {"guard_ledger_sha": self.guard_ledger_sha, "guard_ledger_count": self.guard_ledger_count,
                "guard_pass": self.guard_pass, "guard_reject": self.guard_reject,
                "guard_shortfall": self.guard_shortfall, "next_up_eligible": self.next_up_eligible}

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
               if self.rule_version in RANKED_RULE_VERSIONS else {}),
            **({"guard_ledger": layout.pool_guard(self.guard_ledger_sha), **self.guard_block()}
               if self.rule_version == GUARD_RULE_VERSION else {}),
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
    allow_unscored_previous: bool = False,
    initial_candidate_meta: Mapping[str, Mapping[str, Any]] | None = None,
    initial_retired_ledger: Mapping[str, Sequence[int]] | None = None,
    guard: GuardLedger | None = None,
) -> Rotation:
    """Rank every non-withdrawn candidate, then shrink, fill and swap.

    The candidate list is complete and published. Its state describes the
    previous round, not the new active set. All order ties use
    sha256(round_id|rank|task_id); no iteration or file order can change a
    decision. Withdrawing is permanent: no retired row is a new candidate.

    Under ``rotation_rule="info-guard"`` (``affine-pool-rotation-v3``) a
    candidate from outside the active set may be admitted only when the
    sealed public guard ledger records a pass for its id *and* for the
    ``content_sha`` the catalog currently publishes for it. The gate is
    fail-closed: with too few eligible candidates the round admits fewer and
    records the shortfall. It never removes an active task by itself, and it
    never changes the ranking or ``next_up``, which keep every non-withdrawn
    outside candidate so the pre-judge worker can see what to judge next. No
    network call and no model runs here; the judge is an offline operator
    tool that only ever appends rows to the ledger.
    """
    if not params.rotation_enabled or params.rotation_rule not in INFO_RULES:
        raise ConfigError("rotate_info requires rotation_enabled=true and rotation_rule='info' or 'info-guard'")
    gate = params.rotation_rule == "info-guard"
    if gate and guard is None:
        raise ConfigError("the info-guard rule needs its sealed guard ledger")
    if not gate and guard is not None:
        raise ConfigError("only the info-guard rule takes a guard ledger")
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
    if set(irt_snapshot["tasks"]) != active_set and not allow_unscored_previous:
        raise VerifyError("the ranked rule requires IRT rows exactly matching the previous active set")

    rows: dict[str, dict[str, Any]] = {}
    for task_id, state in [(task_id, "active") for task_id in ids] + [(task_id, "reserve") for task_id in reserve.queue]:
        row = tasks.get(task_id)
        meta = _candidate_meta(reserve, task_id)
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
                         "admitted": False, "retired": False, "retire_reason": None,
                         "tie": sha256_hex(f"{round_id}|rank|{task_id}".encode())}
    for task_id, old in archives.items():
        if task_id in withdrawn or old.get("reason") == WITHDRAWN:
            continue
        meta = {
            key: old[key] for key in ("source_prefix", "env", "prior_p") if key in old
        } | dict(_candidate_meta(reserve, task_id))
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
                        **({"prior_p": meta["prior_p"]} if "prior_p" in meta else {})})

    for task_id in sorted(active_set & withdrawn):
        retire(task_id, WITHDRAWN)
    if len(kept) > target:
        for task_id in sorted(kept, key=ascending)[:len(kept) - target]:
            flags = rows[task_id]["flags"]
            retire(task_id, HIGH if HIGH in flags else LOW if LOW in flags else EXCESS)
            kept.remove(task_id)
    outside = (set(reserve.queue) | set(archives)) - withdrawn - kept - {task for task, row in archives.items() if row.get("reason") == WITHDRAWN}

    def admissible(task_id: str) -> bool:
        """Fail closed: without the gate every candidate passes; with it, only a recorded pass for the current body does."""
        if not gate:
            return True
        assert guard is not None
        return guard.eligible(task_id, _candidate_meta(reserve, task_id).get("content_sha"))

    shortfall = 0
    if len(kept) < target:
        wanted = target - len(kept)
        fill = sorted((task_id for task_id in outside if admissible(task_id)), key=descending)[:wanted]
        shortfall = wanted - len(fill)
        for task_id in fill:
            kept.add(task_id)
            rows[task_id]["admitted"] = True
        outside -= kept
    cap = int(params.rotation_churn_frac * target) if params.rotation_churn_max == 0 else min(int(params.rotation_churn_frac * target), params.rotation_churn_max)
    active_heap = [ascending(task_id) for task_id in kept]
    outside_heap = [descending(task_id) for task_id in outside if admissible(task_id)]
    heapq.heapify(active_heap)
    heapq.heapify(outside_heap)
    swaps = 0
    while swaps < cap and active_heap and outside_heap:
        task_id = active_heap[0][2]
        candidate = outside_heap[0][2]
        if rows[candidate]["score"] <= rows[task_id]["score"]:
            break
        heapq.heappop(active_heap)
        heapq.heappop(outside_heap)
        flags = rows[task_id]["flags"]
        retire(task_id, HIGH if HIGH in flags else LOW if LOW in flags else DISPLACED)
        kept.remove(task_id)
        outside.add(task_id)
        outside.remove(candidate)
        kept.add(candidate)
        rows[candidate]["admitted"] = True
        if admissible(task_id):
            # A task the gate would refuse never re-enters, not even by swap.
            heapq.heappush(outside_heap, descending(task_id))
        swaps += 1
    active = tuple(sorted(kept))
    ranking = tuple({key: value for key, value in row.items() if key != "tie"}
                    for row in sorted(rows.values(), key=lambda row: (-row["score"], row["tie"], row["id"]))
                    if row["retire_reason"] != WITHDRAWN)
    admitted = tuple(row["id"] for row in ranking if row["admitted"])
    upcoming = tuple(row["id"] for row in ranking if row["id"] in outside and row["id"] not in withdrawn)
    restored = {task_id: tuple(archives[task_id]["irt_row"][:3]) for task_id in admitted if task_id in archives and archives[task_id].get("irt_row")}
    counts = {reason: sum(row.reason == reason for row in retired.values()) for reason in sorted({row.reason for row in retired.values()})}
    next_up = upcoming[:params.rotation_queue_publish]
    guard_fields: dict[str, Any] = {}
    if gate:
        assert guard is not None
        guard_fields = {"guard_ledger_sha": guard.sha256, "guard_ledger_count": guard.count,
                        "guard_pass": len(guard.passes), "guard_reject": len(guard.rejects),
                        "guard_shortfall": shortfall,
                        "next_up_eligible": sum(admissible(task_id) for task_id in next_up)}
    return Rotation(
        round_id=round_id, active_ids=active, active_sha=active_sha(active), prev_active_sha=previous_sha,
        admitted=admitted, retired=tuple(retired.values()), reserve_manifest_sha=reserve.manifest_sha,
        queue_remaining=len(set(reserve.queue) - kept - withdrawn), next_up=next_up,
        reserve_history=reserve.history(), ledger={key: (rows[key]["head_rollouts"], rows[key]["head_passes"]) for key in active},
        retired_archive=tuple(archive), source_snapshot_id=source_snapshot_id,
        prev_state_sha=prev_state_sha, initial_active_sha=previous_sha if not prev_state_sha else "",
        rule_params=rule_params(params),
        rule_version=rule_version(params), candidate_ranking=ranking, retire_reason_counts=counts,
        restored_irt_rows=restored, **guard_fields,
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
