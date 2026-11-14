"""The controller (sequencer).

It is the only process that leases seats, judges results, changes the head
or names a king. Every change is one journal record, and the state changes
only by applying that record. ``tick`` moves any unfinished step forward.
So a controller that restarts continues from the journal: it never applies
a winner twice and never loses a seat.

One round:

  open_round     freeze up to ``batch_cap`` valid files in queue order, pin
                 head, params and the public draw, and lease them to boxes
  submit_result  check a box's sealed result; when every lease has answered
                 or failed, judge the round
  finish         verify and promote the winner's LoRA, append the new head,
                 fold it when a fold is due, then close the round

Rules:

* An infrastructure failure is ``no_decision``: nobody is charged, the
  seats go back to the front of the queue, and nothing is promoted. A lease
  that times out is such a failure, and a late result is never used.
* A hash that does not match stops the apply. The round is then abandoned
  and nobody is charged.
* No round opens while a fold is due: boxes train and score on a folded head.
* The parameters are pinned in the journal at the first start. Every later
  process reads them from there, so two machines can never use two sets.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import layout
from .canonical import canonical_json, to_jsonable
from .errors import BatchRejected, ConfigError, HeadMismatchError, InfraError, LeaseError, V2Error, VerifyError
from .headlog import apply_lora, fold_due, fold_head, genesis_head, lora_ref
from .journal import JOURNAL_FILE, Journal, pull
from .judging import check_result, decide
from .params import GenesisSpec, Params
from .public import round_report, snapshot
from .queue import SyncPlan, plan_sync, select_round, split_among_boxes, squat_trim, waiting
from .sampling import make_round_seed
from .seams import Admission, Folder, ObjectStore, Publisher, RoundRunner, TaskSampler
from .state import (
    ABANDONED,
    APPLIED,
    APPLY_DEFERRED,
    GENESIS,
    HEAD_APPLIED,
    HEAD_FOLDED,
    LEASE_FAILED,
    NO_DECISION,
    NULL,
    PARAMS_SET,
    REGISTRATIONS_SYNCED,
    ROUND_CLOSED,
    ROUND_OPENED,
    ROUND_SETTLED,
    SEAT_RELEASED,
    SEATS_DEMOTED,
    V1_IMPORTED,
    Lease,
    RoundRecord,
    State,
    apply,
    load,
    replay,
)
from .store import delete_pending, get_record, promote_fold, promote_lora, publish_head, put_record
from .types import NO_DECISION as ENTRY_NO_DECISION
from .types import (
    NULL_MODEL,
    SKIP,
    BoxResult,
    ChainSnapshot,
    FrozenBatch,
    HeadManifest,
    HeadVersion,
    King,
    Registration,
    RoundTicket,
    V1Import,
)

Leases = list[tuple[RoundTicket, BoxResult]]


@dataclasses.dataclass
class StackContext:
    """What a stack builder gets: where the state lives, the parameters, and the command that runs."""

    state_dir: Path
    params: Params  # after ``load``: the parameters pinned in the journal, when it has them
    command: str = "run"  # a builder may leave out the runner and the folder for any other command

    def load(self, store: ObjectStore | None = None) -> State:
        """The replayed state. Call it before ``params`` is read.

        Give the store when it keeps the journal copy (``Stack.mirror``):
        the local journal is first brought up to date from that copy.
        """
        path = self.state_dir / JOURNAL_FILE
        state = replay(pull(store, path)) if store is not None else load(path) if path.exists() else State()
        self.params = state.params or self.params
        return state


@dataclasses.dataclass
class Stack:
    """Every part the controller needs. ``python -m sn120_v2 run --stack module:function`` builds one."""

    store: ObjectStore
    admission: Admission
    sampler: TaskSampler
    runner: RoundRunner | None = None  # needed by ``run`` only
    folder: Folder | None = None
    publisher: Publisher | None = None
    entropy: Callable[[], str] | None = None  # a value the validator cannot choose, such as the latest block hash
    chain: Callable[[], ChainSnapshot] | None = None  # when set, the queue is synced before each round
    mirror: bool = False  # keep the durable copy of the journal in ``store``, one object per record


def stored_irt_snapshot(store: ObjectStore, state: State) -> Mapping[str, Any] | None:
    """The task-table snapshot of the last closed round. Give it to the sampler after a restart."""
    key = layout.irt_snapshot(state.snapshot_id) if state.snapshot_id else ""
    return json.loads(store.get(key)) if key and store.exists(key) else None


class Controller:
    def __init__(
        self,
        journal: Journal,
        store: ObjectStore,
        admission: Admission,
        sampler: TaskSampler,
        params: Params,
        *,
        folder: Folder | None = None,
        publisher: Publisher | None = None,
    ) -> None:
        self.journal, self.store, self.admission, self.sampler, self._params = journal, store, admission, sampler, params
        self.folder, self.publisher = folder, publisher
        self.state: State = replay(journal.records)
        self.last_error = ""
        self._retry_at = 0
        self._verdicts: dict[str, Verdict] = {}  # admission's last check of the waiting files, for the public snapshot
        self._verdicts_at: int | None = None
        self._published: object = None  # what the last snapshot showed of the queue and its files
        self._pin()

    @property
    def params(self) -> Params:
        """The parameters pinned in the journal. Before the first start: the ones this process was given."""
        return self.state.params or self._params

    def _pin(self) -> None:
        if hasattr(self.admission, "params"):
            self.admission.params = self.params

    def _log(self, kind: str, data: dict[str, Any], now: int) -> None:
        apply(self.state, self.journal.append(kind, data, now))

    # ------------------------------------------------------------ bootstrap

    def bootstrap(
        self,
        genesis: GenesisSpec,
        base_files: Mapping[str, str],
        master_files: Mapping[str, str],
        now: int,
        v1: V1Import | None = None,
    ) -> HeadVersion:
        """Write the genesis head, and the state carried over from v1 if any. Only an empty journal accepts it."""
        if self.state.heads:
            raise V2Error("this journal already has a genesis; refusing to bootstrap again")
        manifest = HeadManifest(genesis_head(genesis, now), dict(base_files), dict(master_files), {})
        self._log(GENESIS, {"genesis": genesis.to_dict(), "manifest": manifest.to_dict()}, now)
        self._log(PARAMS_SET, {"params": self._params.to_dict()}, now)
        if v1 is not None:
            self._log(V1_IMPORTED, {"v1": v1.to_dict()}, now)
        self._sync_pointer()
        self._publish(now)
        return manifest.head

    def set_params(self, params: Params, now: int) -> None:
        """Operator action: pin new parameters. Refused while a round is open, because a round pins one set.

        One case is allowed in a round: the round is decided, and only the
        image and code digests change. Training, scoring and judging are done
        then; the digests only say which code may do the fold that is left.
        Without this, a fix to the fold code could never finish the round.
        """
        record = self.state.round
        if record is not None:
            old = self.params.recipe
            same = dataclasses.replace(params, recipe=dataclasses.replace(params.recipe, image_digests=old.image_digests, code_digest=old.code_digest))
            if record.decision is None or same != self.params:
                raise V2Error("a round is open; set the parameters when it has closed (a decided round takes new digests only)")
        if not self.state.heads:
            raise V2Error("the journal has no genesis yet; give the parameters to bootstrap")
        self._log(PARAMS_SET, {"params": params.to_dict()}, now)
        self._pin()

    def sync_registrations(self, chain: ChainSnapshot, now: int, force: bool = False) -> SyncPlan:
        """Bring the queue in line with the chain. A suspicious read is refused unless ``force``."""
        state = self.state
        plan = plan_sync(state.queue, state.served_regs, state.served_hotkeys, chain, self.params.netuid, self.params.burn_uid, force=force)
        if not plan.empty:
            data = {
                "block": chain.block,
                "added": [reg.to_dict() for reg in plan.added],
                "removed": [list(key) for key in plan.removed],
                "rehotkeyed": [reg.to_dict() for reg in plan.rehotkeyed],
            }
            self._log(REGISTRATIONS_SYNCED, data, now)
        return plan

    def release_seat(self, key: tuple[int, int], now: int) -> None:
        """Operator action: let a held seat take a turn again."""
        self._log(SEAT_RELEASED, {"key": list(key)}, now)

    # ------------------------------------------------------------ open

    def _freeze(self, reg: Registration, round_id: str) -> FrozenBatch | None:
        batch = self.admission.freeze(reg, round_id)
        if batch is None:
            return None
        try:
            self.params.recipe.for_miner(batch.train)
        except BatchRejected:
            return None
        return batch if batch.hotkey == reg.hotkey else None

    def check_files(self, now: int) -> Mapping[str, Verdict]:
        """Admission's verdict on the stored file of every waiting seat, kept for the public snapshot.

        It reads every waiting file from the store. So it runs before a round's
        lease starts or between rounds, never inside a round's time.
        """
        verdicts = dict(self.admission.verdicts([seat.reg.hotkey for seat in waiting(self.state.queue)]))
        self._verdicts, self._verdicts_at = verdicts, now
        return verdicts

    def open_round(
        self, now: int, entropy: str, boxes: Sequence[str] = ("box-0",), clock: Callable[[], int] | None = None
    ) -> list[RoundTicket]:
        """Start a round with the seats that are ready now. Returns no ticket when nothing can start.

        With ``clock``, the lease time is read after the waiting files are read
        and checked, so that work does not use the round's time limit.
        """
        state, params = self.state, self.params
        if state.round is not None:
            raise V2Error("a round is already open")
        if not params.recipe.image_digests:
            raise ConfigError("no worker image is pinned in the recipe; refusing to open a round")
        if fold_due(state.head, params) or now < self._retry_at:
            return []
        if state.failed_leases and now < state.last_failure_at + params.apply_retry_s:
            return []
        verdicts = self.check_files(now)
        moved = squat_trim(state.queue, verdicts, now, state.turns_last_day(now), params.squat_min_wait_s, params.squat_age_factor)
        if moved:
            self._log(SEATS_DEMOTED, {"keys": [list(key) for key in moved]}, now)
        round_id = f"r{state.round_seq + 1:06d}"
        picked = select_round(state.queue, verdicts, lambda reg: self._freeze(reg, round_id), params.batch_cap)
        if len(picked.entries) < params.n_min_start:
            return []
        if clock is not None:
            now = clock()  # the lease starts here
        head = state.head
        draw = self.sampler.draw(head, make_round_seed(round_id, head.version_id, entropy), params.n_tasks)
        self._store_snapshot(draw.snapshot_id)
        tickets, leases = [], []
        for index, block in enumerate(split_among_boxes(picked.entries, len(boxes))):
            ticket = RoundTicket(round_id, f"{round_id}-l{index}", boxes[index], head, params, draw, block, now, now + params.lease_timeout_s)
            put_record(self.store, layout.ticket(round_id, ticket.lease_id), ticket)
            tickets.append(ticket)
            leases.append(Lease(ticket.lease_id, ticket.box_id, now, ticket.expires_at, ticket.digest(), block))
        record = RoundRecord(
            round_id, head.version_id, params.params_hash, params.recipe_id, draw.round_seed, draw.snapshot_id,
            len(draw.task_ids), now, tuple(leases), picked.skipped,
        )
        self._log(ROUND_OPENED, {"round": record.to_dict()}, now)
        self._publish(now)
        return tickets

    def _ticket(self, record: RoundRecord, lease: Lease) -> RoundTicket:
        ticket = get_record(self.store, layout.ticket(record.round_id, lease.lease_id), RoundTicket)
        if ticket.digest() != lease.ticket_digest:
            raise VerifyError(f"the stored ticket of lease {lease.lease_id} does not match the journal")
        return ticket

    def tickets(self) -> list[RoundTicket]:
        """Tickets of the leases that are still open. Use it to hand work out again after a restart."""
        record = self.state.round
        return [self._ticket(record, lease) for lease in record.leases if lease.status == "open"] if record else []

    # ------------------------------------------------------------ results

    def submit_result(self, result: BoxResult, now: int) -> None:
        """Accept one box's sealed result. A result that fails any check raises and changes nothing."""
        record = self.state.round
        if record is None or record.round_id != result.round_id or record.decision is not None:
            raise LeaseError("no open round is waiting for this result")
        lease = next((l for l in record.leases if l.lease_id == result.lease_id), None)
        key = layout.sealed_result(record.round_id, result.lease_id)
        if lease is None or lease.status != "open" or self.store.exists(key):
            raise LeaseError("this lease is unknown, has failed, or already has a result")
        if now >= lease.expires_at:
            raise LeaseError("the lease has expired; a late result is never used")
        if record.head_version_id != self.state.head.version_id:
            raise HeadMismatchError("the round's head is no longer the current head")
        check_result(self._ticket(record, lease), result)
        put_record(self.store, key, result)
        self._settle(now)

    def fail_lease(self, lease_id: str, reason: str, now: int) -> None:
        """Record that a box failed. Its seats go back to the front of the queue, not charged."""
        record = self.state.round
        if record is None or record.decision is not None:
            raise LeaseError("no open round has a running lease")
        self._log(LEASE_FAILED, {"round_id": record.round_id, "lease_id": lease_id, "reason": reason}, now)
        self._settle(now)

    def _leases(self, record: RoundRecord, status: str) -> Leases:
        pairs = []
        for lease in record.leases:
            key = layout.sealed_result(record.round_id, lease.lease_id)
            if lease.status == status and self.store.exists(key):
                pairs.append((self._ticket(record, lease), get_record(self.store, key, BoxResult)))
        return pairs

    def _settle(self, now: int) -> None:
        record = self.state.round
        if record is None or record.decision is not None:
            return
        open_leases = [lease for lease in record.leases if lease.status == "open"]
        pairs = self._leases(record, "open")
        if len(pairs) < len(open_leases):
            return
        if not pairs:
            return self._close(NO_DECISION, [], now)
        decision = decide(pairs)
        results = {ticket.lease_id: result.digest() for ticket, result in pairs}
        self._log(ROUND_SETTLED, {"round_id": record.round_id, "decision": decision.to_dict(), "results": results}, now)
        self._finish(now)

    # ------------------------------------------------------------ apply, fold, close

    def _finish(self, now: int) -> None:
        record = self.state.round
        pairs, winner = self._leases(record, "done"), record.decision.winner
        if winner is not None and not record.applied:
            try:
                self._apply(record, pairs, now)
            except VerifyError as exc:
                return self._close(ABANDONED, pairs, now, str(exc))
            except InfraError as exc:
                if record.apply_attempts + 1 >= self.params.apply_max_attempts:
                    return self._close(ABANDONED, pairs, now, f"apply failed {self.params.apply_max_attempts} times: {exc}")
                return self._log(APPLY_DEFERRED, {"round_id": record.round_id, "reason": str(exc)}, now)
        if fold_due(self.state.head, self.params) and not self.fold_if_due(now, in_round=True):
            return
        charged = any(entry.charged for entry in record.decision.entries)
        self._close(APPLIED if winner else (NULL if charged else NO_DECISION), pairs, now)

    def _apply(self, record: RoundRecord, pairs: Leases, now: int) -> None:
        """Verify and promote the winner, then append the new head. The journal record is the commit point."""
        winner = record.decision.winner
        ticket, result = next(pair for pair in pairs if any(e.queue_pos == winner.queue_pos for e in pair[0].entries))
        seat = next(e for e in ticket.entries if e.queue_pos == winner.queue_pos)
        artifact = next(e for e in result.entries if e.queue_pos == winner.queue_pos).lora
        promote_lora(self.store, artifact)
        public_batch = layout.winner_batch(record.round_id)
        try:
            self.store.copy(seat.batch.uri, public_batch)
        except KeyError:
            raise VerifyError("the winner's frozen batch is missing") from None
        if self.store.sha256(public_batch) != seat.batch.batch_sha256:
            self.store.delete(public_batch)
            raise VerifyError("the winner's frozen batch does not match its sha256")
        parent = self.state.manifest
        head = apply_lora(self.state.head, artifact, lora_ref(record.round_id, seat, artifact), now)
        manifest = HeadManifest(head, parent.base_files, parent.master_files, parent.lora_files | {artifact.lora_hash: artifact.files})
        king = King(seat.reg.uid, seat.reg.hotkey, record.round_id, head.version_id, now, winner.score.z)
        self._log(HEAD_APPLIED, {"round_id": record.round_id, "manifest": manifest.to_dict(), "king": king.to_dict()}, now)

    def fold_if_due(self, now: int, in_round: bool = False) -> bool:
        """Fold the stacked LoRAs into a new base when a fold is due. Returns True when the head is folded."""
        head = self.state.head
        if not fold_due(head, self.params) or (self.state.round is not None and not in_round) or now < self._retry_at:
            return False
        if self.folder is None:
            raise ConfigError("a fold is due, but no folder is configured")
        try:
            fold = self.folder.fold(head, self.params, layout.pending_fold_prefix(head.version_id))
            promote_fold(self.store, fold)
            folded = fold_head(head, fold, now)
        except (InfraError, VerifyError, HeadMismatchError) as exc:
            self.last_error, self._retry_at = f"fold failed: {exc}", now + self.params.apply_retry_s
            return False
        manifest = HeadManifest(folded, fold.base_files, fold.master_files, {})
        self._log(HEAD_FOLDED, {"manifest": manifest.to_dict()}, now)
        delete_pending(self.store, fold.prefix)
        return True

    def _close(self, outcome: str, pairs: Leases, now: int, reason: str = "") -> None:
        record, params, queue = self.state.round, self.params, self.state.queue
        decision = record.decision
        box_failed = {e.queue_pos for _, result in pairs for e in result.entries if e.status == ENTRY_NO_DECISION}
        served, front, attempted, held, released = [], [], [], [], []
        for entry in decision.entries if decision else ():
            key = [entry.uid, entry.registration_block]
            if entry.status == SKIP:
                released.append(key)
            elif entry.charged and outcome != ABANDONED:
                served.append(key + [entry.hotkey])
            else:
                front.append(key)
                if entry.queue_pos in box_failed:
                    attempted.append(key)
                    if queue[tuple(key)].infra_attempts + 1 >= params.max_infra_attempts:
                        held.append(key)
        snapshot_id = self.state.snapshot_id
        if outcome in (APPLIED, NULL) and pairs:
            snapshot_id = self._update_sampler(pairs, outcome)
        closed = dataclasses.replace(record, outcome=outcome, closed_at=now)
        report = round_report(closed, pairs) | {"note": reason, "wall_s": self._wall(pairs, now)}
        report["new_head_version_id"] = self.state.head.version_id if outcome == APPLIED else None
        self.store.put(layout.round_file(record.round_id), canonical_json(report))
        data = {"round_id": record.round_id, "outcome": outcome, "reason": reason, "snapshot_id": snapshot_id}
        data |= {"served": served, "front": front, "attempted": attempted, "held": held, "released": released}
        self._log(ROUND_CLOSED, data, now)
        self._sync_pointer()
        delete_pending(self.store, layout.pending_round_prefix(record.round_id))
        self._publish(now)

    def _wall(self, pairs: Leases, now: int) -> dict[str, int]:
        """Wall seconds of each phase of the open round. Box phases come from the box; the others from the journal."""
        at: dict[str, int] = {}
        for item in reversed(self.journal.records):
            at.setdefault(item.kind, item.at)
            if item.kind == ROUND_OPENED:
                break
        opened = at[ROUND_OPENED]
        wall = {name: max((result.phase_s.get(name, 0) for _, result in pairs), default=0) for name in ("training", "evaluation")}
        wall["total"] = now - opened
        if HEAD_APPLIED in at:
            wall["promote"] = at[HEAD_APPLIED] - at[ROUND_SETTLED]
            wall["lease_to_promotion"] = at[HEAD_APPLIED] - opened
        if HEAD_FOLDED in at:
            wall["fold"] = at[HEAD_FOLDED] - at[HEAD_APPLIED]
        return wall

    def _update_sampler(self, pairs: Leases, outcome: str) -> str:
        """Fold the round's outcomes into the task table and store the new public snapshot."""
        ticket, result = pairs[0]
        outcomes = {NULL_MODEL: result.null_outcomes}
        outcomes |= {e.lora.lora_hash: e.outcomes for _, r in pairs for e in r.entries if e.lora is not None}
        winner = self.state.round.decision.winner
        applied = outcome == APPLIED
        try:
            snapshot_id = self.sampler.update(
                ticket.head,
                ticket.draw,
                outcomes,
                winner_lora_hash=winner.lora_hash if applied else "",
                new_head_version_id=self.state.head.version_id if applied else "",
            )
            self._store_snapshot(snapshot_id)
            return snapshot_id
        except InfraError as exc:
            self.last_error = f"task table update failed: {exc}"
            return self.state.snapshot_id

    def _store_snapshot(self, snapshot_id: str) -> None:
        """Publish the sampler's snapshot. The sampler does no I/O, so a restart reads the table back from here."""
        self.store.put(layout.irt_snapshot(snapshot_id), canonical_json(to_jsonable(self.sampler.public_snapshot())))

    # ------------------------------------------------------------ publish

    def _sync_pointer(self) -> None:
        """Make the store's head manifest and ``current`` pointer follow the journal."""
        manifest = self.state.manifest
        try:
            if manifest and not (self.store.exists(layout.CURRENT) and self.store.get(layout.CURRENT) == manifest.to_json()):
                publish_head(self.store, manifest)
        except (InfraError, VerifyError) as exc:
            self.last_error = f"head pointer not published yet: {exc}"

    def _shown(self) -> object:
        """What a snapshot shows of the queue and its files. A change here is a reason to publish between rounds."""
        seats = [(seat.reg.key, seat.front, seat.demoted, seat.held) for seat in waiting(self.state.queue)]
        return seats, {hotkey: (verdict.valid, verdict.batch_sha256) for hotkey, verdict in self._verdicts.items()}

    def _publish(self, now: int) -> None:
        if self.publisher is None:
            return
        shown = self._shown()
        try:
            self.publisher.publish(snapshot(self.state, self.params, now, self._verdicts, self._verdicts_at))
            self._published = shown
        except InfraError as exc:
            self.last_error = f"publish failed: {exc}"

    def publish(self, now: int, check_files: bool = False, only_changes: bool = False) -> bool:
        """Publish the public snapshot outside a round's own steps. Returns True when it published.

        ``check_files`` reads the file of every waiting seat first; it is
        refused while a round is open. ``only_changes`` publishes only when
        the queue or a file verdict differs from the last published snapshot.
        """
        if check_files:
            if self.state.round is not None:
                raise V2Error("a round is open; the waiting files are not read inside a round's time")
            try:
                self.check_files(now)
            except InfraError as exc:
                self.last_error = f"file check failed: {exc}"
        if self.publisher is None or (only_changes and self._shown() == self._published):
            return False
        self._publish(now)
        return True

    # ------------------------------------------------------------ the loop

    def tick(self, now: int) -> None:
        """Move every unfinished step forward. Safe to call at any time, and first thing after a restart."""
        if not self.state.heads:
            return
        self._sync_pointer()
        record = self.state.round
        if record is not None and record.decision is None:
            for lease in record.leases:
                has_result = self.store.exists(layout.sealed_result(record.round_id, lease.lease_id))
                if lease.status == "open" and now >= lease.expires_at and not has_result:
                    self._log(LEASE_FAILED, {"round_id": record.round_id, "lease_id": lease.lease_id, "reason": "the lease expired"}, now)
            self._settle(now)
        elif record is not None:
            waiting_to_retry = record.apply_attempts and now < self.state.last_failure_at + self.params.apply_retry_s
            if not waiting_to_retry:
                self._finish(now)
        else:
            self.fold_if_due(now)

    def run_round(self, runner: RoundRunner, clock: Callable[[], int], entropy: str) -> RoundRecord | None:
        """One round on one box, start to end. Returns the closed round, or None when no round could start."""
        self.tick(clock())
        tickets = self.open_round(clock(), entropy, clock=clock) if self.state.round is None else self.tickets()
        for ticket in tickets:
            try:
                self.submit_result(runner.run(ticket), clock())
            except Exception as exc:  # a box failure of any kind must not stop the controller
                record = self.state.round
                running = record is not None and record.decision is None and record.lease(ticket.lease_id).status == "open"
                # With the result in the store the box did its work: what failed is the controller's own judging.
                # The round stays open with its result, and the next start judges it.
                if not running or self.store.exists(layout.sealed_result(record.round_id, ticket.lease_id)):
                    raise
                self.fail_lease(ticket.lease_id, f"{type(exc).__name__}: {exc}", clock())
        self.tick(clock())
        return self.state.rounds[-1] if tickets and self.state.round is None else None
