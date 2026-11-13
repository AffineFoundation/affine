"""Shared data types. Everything that crosses a seam is defined here.

Each type is an immutable ``Record``: it has a strict JSON round trip and a
stable digest. Types that carry a hash check it when they are built, so an
object that exists is already consistent with its own hash.

Times are whole seconds since the Unix epoch, in UTC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .canonical import Record, digest, is_sha256, manifest_digest, to_jsonable
from .errors import ContractError, VerifyError
from .params import Params

# One rollout outcome is one character. A model's outcomes on a list of tasks
# are one string, in task order.
PASS = "P"  # resolved and correct
FAIL = "F"  # resolved and wrong
CAP = "C"  # hit the token cap with no end of sequence: unresolved, counts as not passed
ERROR = "E"  # our infrastructure failed (timeout, crash): not passed, stays in the denominator
NOT_RUN = "-"  # not scored, because the entry was dropped at an earlier look
OUTCOME_CODES = frozenset((PASS, FAIL, CAP, ERROR, NOT_RUN))

NULL_MODEL = "null"  # key of the null entry: the head M served with no adapter

# Status of one entry in a box result.
SCORED = "scored"
NO_DECISION = "no_decision"  # infrastructure failed: not charged, back to the front of the queue
SKIP = "skip"  # the frozen batch failed a check: not charged, keeps its place
# Status of one entry in a round decision (NO_DECISION and SKIP also apply).
WINNER = "winner"
LOST = "lost"


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def check_outcomes(outcomes: tuple[str, ...], n_tasks: int, what: str) -> None:
    """Every outcome string must cover every task with a known code."""
    for text in outcomes:
        _need(len(text) == n_tasks, f"{what}: outcome string has {len(text)} codes for {n_tasks} tasks")
        _need(OUTCOME_CODES.issuperset(text), f"{what}: unknown outcome code")


@dataclass(frozen=True)
class Registration(Record):
    """One paid chain registration. It buys exactly one seat in one round, win or lose."""

    uid: int
    hotkey: str
    registration_block: int
    registered_at: int = 0  # time of the registration block; 0 = unknown
    coldkey: str = ""  # optional owner key, for the one-entry-per-miner rule

    def __post_init__(self) -> None:
        _need(self.uid >= 0 and self.registration_block >= 0 and bool(self.hotkey), "bad registration")

    @property
    def key(self) -> tuple[int, int]:
        """(uid, registration_block): the identity of a seat."""
        return (self.uid, self.registration_block)


@dataclass(frozen=True)
class ChainSnapshot(Record):
    """A complete read of the subnet's registrations at one block. UID 0 is included here."""

    netuid: int
    block: int
    taken_at: int
    registrations: tuple[Registration, ...]

    def hotkeys(self) -> dict[int, str]:
        return {reg.uid: reg.hotkey for reg in self.registrations}


@dataclass(frozen=True)
class Verdict(Record):
    """The live check of a miner's stored file. Public: it never holds batch content or train values."""

    hotkey: str
    valid: bool
    reason: str = ""
    format: str = ""
    n_prompts: int = 0
    n_rollouts: int = 0
    batch_sha256: str = ""
    checked_at: int = 0
    rule: str = ""  # stable code of the rule that failed; empty for a valid file


@dataclass(frozen=True)
class FrozenBatch(Record):
    """A miner's file as copied and checked at round open. What is frozen is what is trained."""

    hotkey: str
    batch_sha256: str  # sha256 of the exact stored bytes
    uri: str  # private location of the frozen copy
    format: str
    n_prompts: int
    n_rollouts: int
    train: dict[str, Any] = field(default_factory=dict)  # the miner's values; apply with Recipe.for_miner(train)

    def __post_init__(self) -> None:
        _need(is_sha256(self.batch_sha256), "FrozenBatch.batch_sha256 must be a 64-hex sha256")


@dataclass(frozen=True)
class Entry(Record):
    """One admitted seat of a round. A lower ``queue_pos`` was queued earlier and wins ties."""

    queue_pos: int
    reg: Registration
    batch: FrozenBatch

    def __post_init__(self) -> None:
        _need(self.queue_pos >= 0, "queue_pos must be zero or more")
        _need(self.batch.hotkey == self.reg.hotkey, "the frozen batch belongs to another hotkey")


@dataclass(frozen=True)
class LoRARef(Record):
    """An applied LoRA, as recorded in the head."""

    lora_hash: str
    round_id: str
    miner_hotkey: str
    miner_uid: int
    batch_sha256: str
    recipe_id: str
    rank: int


@dataclass(frozen=True)
class HeadVersion(Record):
    """The head M: a base checkpoint plus an ordered list of applied LoRAs.

    ``version_id`` is the sha256 of the canonical JSON of every other field.
    A ``HeadVersion`` with a wrong id cannot be built.
    """

    version_id: str
    parent_version_id: str | None
    base_hash: str  # manifest digest of the bf16 base files
    base_fp32_master_digest: str  # digest of the fp32 master the next fold must start from
    loras: tuple[LoRARef, ...]
    folded: bool  # True when the last winner is merged into the base and ``loras`` is empty
    round_id: str
    applied_at: int

    def __post_init__(self) -> None:
        if self.version_id != self.expected_id():
            raise VerifyError(f"head version id {self.version_id[:12]} does not match its content")
        _need(self.folded == (not self.loras), "folded must be true exactly when no LoRA is stacked")

    def expected_id(self) -> str:
        body = self.to_dict()
        del body["version_id"]
        return digest(body)

    @classmethod
    def make(
        cls,
        *,
        parent_version_id: str | None,
        base_hash: str,
        base_fp32_master_digest: str,
        loras: tuple[LoRARef, ...],
        round_id: str,
        applied_at: int,
    ) -> HeadVersion:
        body = dict(
            parent_version_id=parent_version_id,
            base_hash=base_hash,
            base_fp32_master_digest=base_fp32_master_digest,
            loras=tuple(loras),
            folded=not loras,
            round_id=round_id,
            applied_at=applied_at,
        )
        return cls(version_id=digest(to_jsonable(body)), **body)


@dataclass(frozen=True)
class HeadManifest(Record):
    """What a box or a miner downloads to fetch a head and check every file."""

    head: HeadVersion
    base_files: dict[str, str]  # {file name: sha256} of the bf16 base
    master_files: dict[str, str]  # {file name: sha256} of the fp32 master, stored apart from the base
    lora_files: dict[str, dict[str, str]]  # {lora_hash: {file name: sha256}} for each stacked LoRA

    def __post_init__(self) -> None:
        if manifest_digest(self.base_files) != self.head.base_hash:
            raise VerifyError("base files do not hash to the head's base_hash")
        stacked = [ref.lora_hash for ref in self.head.loras]
        if sorted(stacked) != sorted(self.lora_files):
            raise VerifyError("the manifest's LoRA files do not match the head's LoRA list")
        for lora_hash, files in self.lora_files.items():
            if manifest_digest(files) != lora_hash:
                raise VerifyError(f"files of LoRA {lora_hash[:12]} do not hash to its id")
        _need(bool(self.master_files), "a head manifest must list the fp32 master files")


@dataclass(frozen=True)
class Draw(Record):
    """The public task draw of a round. Looks are prefixes of ``task_ids``."""

    snapshot_id: str  # the IRT snapshot the draw was made from
    round_seed: str  # public seed of the round
    task_ids: tuple[str, ...]  # in evaluation order
    weights: tuple[float, ...]  # sampling weight of each task (1 / inclusion probability)

    def __post_init__(self) -> None:
        _need(len(self.task_ids) >= 1, "a draw needs at least one task")
        _need(len(self.task_ids) == len(self.weights), "a draw needs one weight per task")
        _need(len(set(self.task_ids)) == len(self.task_ids), "a draw must not repeat a task")
        _need(all(math.isfinite(w) and w > 0 for w in self.weights), "sampling weights must be positive numbers")


@dataclass(frozen=True)
class TrainReport(Record):
    """What the trainer measured while it made one LoRA."""

    seed: int
    steps: int
    loss: float
    step_norm: float  # norm of the adapter update; the step-size bound applies to this
    tokens: int
    wall_s: float
    gpu_class: str
    image_digest: str
    norm_before_cap: float = 0.0  # norm of the update before recipe.global_norm_cap cut it; 0 = not reported
    truncated_rows: int = 0  # rows cut at a training token cap
    dropped_rows: int = 0  # rows with no output token left after the cut


@dataclass(frozen=True)
class LoRAArtifact(Record):
    """One trained LoRA and where its files are. ``lora_hash`` is the manifest digest of ``files``."""

    lora_hash: str
    files: dict[str, str]  # {file name: sha256}
    sizes: dict[str, int]  # {file name: bytes}
    prefix: str  # object-store prefix: each file is at <prefix>/<file name>
    head_version_id: str
    batch_sha256: str
    recipe_id: str
    rank: int
    alpha: float
    target_modules: tuple[str, ...]
    report: TrainReport

    def __post_init__(self) -> None:
        if manifest_digest(self.files) != self.lora_hash:
            raise VerifyError("LoRA files do not hash to lora_hash")
        _need(sorted(self.sizes) == sorted(self.files), "a LoRA needs one size per file")


@dataclass(frozen=True)
class TrainRequest(Record):
    """Train one bounded LoRA from one frozen batch on the pinned head."""

    round_id: str
    head: HeadVersion
    entry: Entry
    params: Params  # pinned snapshot: recipe and caps
    seed: int
    gpu_ids: tuple[int, ...]
    out_prefix: str  # object-store prefix where the LoRA files must be written


@dataclass(frozen=True)
class EvalRequest(Record):
    """Score null M and the given LoRAs on one look's tasks, with the round's seeds and caps."""

    round_id: str
    head: HeadVersion
    loras: tuple[LoRAArtifact, ...]  # null M is always scored too
    task_ids: tuple[str, ...]  # the tasks of this look only
    round_seed: str  # rollout seeds come from sampling.rollout_seed(round_seed, task_id, j)
    look: int
    params: Params  # pinned snapshot: decode settings and caps
    deadline: int = 0  # Unix seconds; past it the evaluator gives up with InfraError and the chunk is not used. 0 = none


@dataclass(frozen=True)
class EvalChunk(Record):
    """Outcomes of one look. Keys are ``NULL_MODEL`` and each ``lora_hash``."""

    look: int
    task_ids: tuple[str, ...]
    outcomes: dict[str, tuple[str, ...]]  # per model: one string per sample index, one code per task
    image_digest: str

    def __post_init__(self) -> None:
        _need(NULL_MODEL in self.outcomes, "an eval chunk must score null M")
        for model, strings in self.outcomes.items():
            check_outcomes(strings, len(self.task_ids), f"eval chunk model {model[:12]}")
            _need(all(NOT_RUN not in text for text in strings), "an eval chunk must score every task it was given")


@dataclass(frozen=True)
class FoldArtifact(Record):
    """A folded head: the stacked LoRAs merged into a new bf16 base and a new fp32 master."""

    parent_version_id: str
    base_hash: str
    base_files: dict[str, str]  # at <prefix>/base/<file name>
    base_fp32_master_digest: str
    master_files: dict[str, str]  # at <prefix>/master/<file name>
    prefix: str
    image_digest: str

    def __post_init__(self) -> None:
        if manifest_digest(self.base_files) != self.base_hash:
            raise VerifyError("folded base files do not hash to base_hash")
        _need(is_sha256(self.base_fp32_master_digest) and bool(self.master_files), "a fold needs its fp32 master")


@dataclass(frozen=True)
class RoundTicket(Record):
    """Everything one box needs to run its leased part of a round. Its digest pins all of it."""

    round_id: str
    lease_id: str
    box_id: str
    head: HeadVersion
    params: Params
    draw: Draw
    entries: tuple[Entry, ...]
    issued_at: int
    expires_at: int

    def __post_init__(self) -> None:
        _need(len(self.entries) >= 1, "a ticket needs at least one entry")
        positions = [entry.queue_pos for entry in self.entries]
        _need(positions == sorted(set(positions)), "ticket entries must be in queue order, without repeats")
        _need(self.expires_at > self.issued_at, "a lease must expire after it is issued")


@dataclass(frozen=True)
class EntryResult(Record):
    """What a box reports for one entry."""

    queue_pos: int
    status: str  # SCORED, NO_DECISION or SKIP
    reason: str = ""
    lora: LoRAArtifact | None = None
    outcomes: tuple[str, ...] = ()  # one string per sample index over the round's task order
    eliminated_at_look: int | None = None
    attempts: int = 1

    def __post_init__(self) -> None:
        _need(self.status in (SCORED, NO_DECISION, SKIP), f"unknown entry status {self.status!r}")
        _need((self.status == SCORED) == (self.lora is not None), "exactly the scored entries carry a LoRA")
        _need(self.status == SCORED or bool(self.reason), "a no_decision or skip entry must give a reason")


@dataclass(frozen=True)
class BoxResult(Record):
    """The sealed result of one lease. Its digest is the seal. The box never picks the winner."""

    round_id: str
    lease_id: str
    box_id: str
    ticket_digest: str  # digest of the ticket this result answers
    head_version_id: str  # the head the box trained and scored against
    image_digest: str
    null_outcomes: tuple[str, ...]  # null M over the round's task order
    entries: tuple[EntryResult, ...]
    started_at: int
    finished_at: int
    stopped_at_limit: bool = False  # the round's time limit ended the scoring before the full draw
    phase_s: dict[str, int] = field(default_factory=dict)  # wall seconds of each phase on the box: training, evaluation


@dataclass(frozen=True)
class Score(Record):
    """One LoRA against null M on the same tasks and seeds."""

    n_tasks: int  # tasks this LoRA was scored on
    delta: float  # weighted mean of (LoRA pass rate - null M pass rate) per task
    se: float  # standard error of delta
    z: float  # delta / se, and 0.0 when se is 0
    pass_rate: float
    null_pass_rate: float
    cap_rate: float  # share of this LoRA's rollouts that hit the token cap
    null_cap_rate: float
    cover: float  # share of this LoRA's rollouts that resolved without an infrastructure error
    eligible: bool
    reasons: tuple[str, ...] = ()  # why it is not eligible; empty when it is


@dataclass(frozen=True)
class EntryDecision(Record):
    """The controller's decision for one entry."""

    queue_pos: int
    uid: int
    hotkey: str
    registration_block: int
    batch_sha256: str
    status: str  # WINNER, LOST, NO_DECISION or SKIP
    reason: str = ""
    lora_hash: str = ""
    score: Score | None = None
    eliminated_at_look: int | None = None

    def __post_init__(self) -> None:
        _need(self.status in (WINNER, LOST, NO_DECISION, SKIP), f"unknown decision status {self.status!r}")

    @property
    def charged(self) -> bool:
        """True when the entry had a fair tournament: its registration is used up."""
        return self.status in (WINNER, LOST)


@dataclass(frozen=True)
class RoundDecision(Record):
    """The settled outcome of a round. At most one winner. No winner means M stays."""

    round_id: str
    head_version_id: str
    winner_queue_pos: int | None
    entries: tuple[EntryDecision, ...]
    n_tasks: int
    null_pass_rate: float
    null_cap_rate: float
    reason: str = ""

    def __post_init__(self) -> None:
        winners = [entry.queue_pos for entry in self.entries if entry.status == WINNER]
        expected = [] if self.winner_queue_pos is None else [self.winner_queue_pos]
        _need(winners == expected, "a round decision has at most one winner, named by winner_queue_pos")

    @property
    def winner(self) -> EntryDecision | None:
        return next((entry for entry in self.entries if entry.status == WINNER), None)


@dataclass(frozen=True)
class King(Record):
    """A round winner that was applied to the head."""

    uid: int
    hotkey: str  # the hotkey that earned it; it is paid only while still registered on ``uid``
    round_id: str  # a v2 round id, or "v1-e<epoch>" for a king imported at cutover
    head_version_id: str  # the head this king produced; "" for an imported king
    applied_at: int
    z: float = 0.0


@dataclass(frozen=True)
class QueueEntry(Record):
    """One waiting seat and its queue flags."""

    reg: Registration
    leased: bool = False  # in an open round now
    front: bool = False  # sent back to the front after a no_decision
    demoted: bool = False  # moved to the back: no valid file for too long
    infra_attempts: int = 0  # how many no_decision outcomes this seat has had
    held: bool = False  # parked for the operator after too many no_decision outcomes


@dataclass(frozen=True)
class WeightReceipt(Record):
    """Result of one attempt to set chain weights."""

    status: str  # "set" or "deferred"
    block: int
    reason: str = ""

    def __post_init__(self) -> None:
        _need(self.status in ("set", "deferred"), f"unknown weight receipt status {self.status!r}")


@dataclass(frozen=True)
class V1Import(Record):
    """State carried over from v1 at cutover, already checked by the cutover import."""

    source: str  # what was read, for the journal
    chain_block: int
    kings: tuple[King, ...]  # oldest first
    served_regs: tuple[tuple[int, int], ...]  # (uid, registration_block) seats that already had their turn
    served_hotkeys: tuple[str, ...]  # hotkeys that already had their turn
    demoted: tuple[tuple[int, int], ...] = ()  # seats v1 had moved to the back
    front: tuple[tuple[int, int], ...] = ()  # seats v1 owed a retry ahead of the queue
