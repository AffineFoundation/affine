"""One box: train each leased batch, then score null M and every LoRA, look by look.

``LocalRunner`` is the ``RoundRunner`` for a box. It calls a ``Trainer`` and
an ``Evaluator`` and returns a sealed ``BoxResult``. It never picks the
winner: it only reports outcomes. The controller judges them.

Rules it keeps:

* One failing batch must not take the box down. A training failure is tried
  again, then that entry becomes ``no_decision`` and the others go on.
* Batches train at the same time, each on its own GPUs. With no more batches
  than GPUs every batch starts at the lease, and the GPUs go to the batches
  by their token-steps (``gpu_counts``). With more batches than GPUs each
  batch has one GPU, and a batch waits for a free GPU.
* The trainer is closed before evaluation starts, so its GPUs are free.
* Null M is scored in every look, with the same tasks and seeds as the LoRAs.
* The round's time limit is hard. It cuts every look, the first one too:
  - Training has its own time, ``train_time_budget_s`` from the lease. At its
    end no new training starts and every job that still runs is stopped. Such
    a batch is ``not_trained``: not charged, first in the next round, and no
    failure is counted for its seat. So the scoring always starts on time.
  - Past the limit the box starts no new chunk of tasks, and a chunk that
    still runs is not used. What is scored by then is the whole test.
  - A LoRA with fewer scored tasks than the first look at the limit is
    ``not_scored``: not charged, first in the next round, and no failure is
    counted for its seat.
"""

from __future__ import annotations

import queue
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Sequence

from . import layout
from .errors import BatchRejected, ContractError, InfraError
from .judging import drop_at_look, score_entry
from .params import Params
from .sampling import train_seed
from .seams import Evaluator, Trainer
from .types import (
    NO_DECISION,
    NOT_RUN,
    NOT_SCORED,
    NOT_TRAINED,
    NULL_MODEL,
    OUT_OF_TIME,
    SCORED,
    SKIP,
    BoxResult,
    Entry,
    EntryResult,
    EvalChunk,
    EvalRequest,
    LoRAArtifact,
    RoundTicket,
    TrainRequest,
)

# A pass of the box started "at the lease" when it started at most this long after the start of the lease. Only then
# did a batch of the first group have the whole training time. The first group: the batches that start with the pass
# (all of them when the round has no more batches than GPUs, else as many as there are GPUs). A pass that started
# later (run started again in the round, or the round ran again under the same lease) gave it less: a batch that does
# not finish is then not_trained, and nothing is counted. A normal pass starts a few seconds after the lease: the
# controller writes the ticket, the journal record and the public files in between. The weight hash, the downloads
# and the read of the loads (``LoadOf``) come after the start of the pass.
AT_THE_LEASE_S = 120


# A round with more batches than GPUs: each batch has one GPU, and the batches start in this order. False: queue
# order. True: the batch with the most token-steps first, and queue order between equal batches. Then the long jobs
# run side by side from the lease and the short jobs fill the GPUs after them, so the training phase ends sooner.
LONGEST_FIRST = False


@dataclass(frozen=True)
class TrainLoad:
    """What the training of one batch costs: the two numbers that the GPU plan of a round needs."""

    token_steps: int  # steps times the tokens of the rollouts that train: the cost of admission's rule ``train_cost``
    rows: int  # the rollouts that train. A training job puts no more copies of the model on GPUs than this.


# The one source of the load of a batch. The box gives the function: today it reads the frozen file and counts with
# the tokenizer of the head. When the ticket holds these numbers, only this function changes.
LoadOf = Callable[[Entry, Params], TrainLoad]


def gpu_counts(n_gpus: int, loads: Sequence[TrainLoad]) -> list[int]:
    """How many GPUs the training job of each batch has, in queue order. All integers: the same on every machine.

    More batches than GPUs: one GPU each. Else every batch gets one GPU. Then each GPU that is left, one at a time,
    goes to the batch with the most token-steps for each GPU that it has now. Between equal batches the lower queue
    position wins. A batch gets no more GPUs than it has rows that train: its job cannot use them. So the largest
    value of token-steps / GPUs over the batches is as small as whole GPUs allow, and a GPU stays free only when no
    batch can use it.
    """
    n = len(loads)
    counts = [1] * n
    if n > n_gpus:
        return counts
    for _ in range(n_gpus - n):
        best = None
        for i in range(n):
            if counts[i] < loads[i].rows and (best is None or loads[i].token_steps * counts[best] > loads[best].token_steps * counts[i]):
                best = i
        if best is None:
            break
        counts[best] += 1
    return counts


def start_order(loads: Sequence[TrainLoad], longest_first: bool) -> list[int]:
    """The order in which the batches of a round start: queue order, or the most token-steps first."""
    order = list(range(len(loads)))
    return sorted(order, key=lambda i: (-loads[i].token_steps, i)) if longest_first else order


class LocalRunner:
    def __init__(
        self,
        trainer: Trainer,
        evaluator: Evaluator,
        clock: Callable[[], int],
        image_digest: str,
        attempts: int = 3,
        poll_s: float = 1.0,
        load_of: LoadOf | None = None,
    ) -> None:
        self.trainer, self.evaluator = trainer, evaluator
        self.clock, self.image_digest, self.attempts, self.poll_s = clock, image_digest, attempts, poll_s
        self.load_of = load_of

    def run(self, ticket: RoundTicket) -> BoxResult:
        started, params = self.clock(), ticket.params
        # Two times, both counted from the lease. ``limit``: the box stops scoring, early enough that judging and
        # promotion still fit inside the round's time limit. ``train_end``: the end of the training time.
        limit = min(ticket.expires_at, ticket.issued_at + params.round_time_limit_s) - params.promote_reserve_s
        train_end = min(limit, ticket.issued_at + params.train_time_budget_s)
        reports = self._train_all(ticket, train_end, at_the_lease=started - ticket.issued_at <= AT_THE_LEASE_S)
        trained = {pos: report for pos, report in reports.items() if isinstance(report, LoRAArtifact)}
        train_done = self.clock()
        null, scored, dropped, stopped = self._evaluate(ticket, trained, limit) if trained else ((), {}, {}, False)
        first_look = params.look_sizes(len(ticket.draw.task_ids))[0]
        n_scored = len(null[0].rstrip(NOT_RUN)) if null else 0
        entries = []
        for entry in ticket.entries:
            report = reports[entry.queue_pos]
            if isinstance(report, LoRAArtifact) and stopped and n_scored < first_look:
                reason = f"the round's time limit ended the scoring after {n_scored} tasks, before the first look of {first_look}"
                report = EntryResult(entry.queue_pos, NOT_SCORED, reason=reason)
            elif isinstance(report, LoRAArtifact):
                report = EntryResult(entry.queue_pos, SCORED, lora=report, outcomes=scored[entry.queue_pos], eliminated_at_look=dropped.get(entry.queue_pos))
            entries.append(report)
        finished = self.clock()
        return BoxResult(
            round_id=ticket.round_id,
            lease_id=ticket.lease_id,
            box_id=ticket.box_id,
            ticket_digest=ticket.digest(),
            head_version_id=ticket.head.version_id,
            image_digest=self.image_digest,
            null_outcomes=null if any(entry.status == SCORED for entry in entries) else (),
            entries=tuple(entries),
            started_at=started,
            finished_at=finished,
            stopped_at_limit=stopped,
            phase_s={"training": train_done - started, "evaluation": finished - train_done},
        )

    def _loads(self, ticket: RoundTicket) -> list[TrainLoad]:
        """The load of each batch, in queue order. The same files give the same loads in every pass.

        With no source (a test stack, or a round of one batch at home) the batches count as equal. A file whose load
        cannot be read has no token-steps and keeps one GPU: its training job fails or is rejected for the same
        cause, so no GPU more is held for it. A fault here never stops the round.
        """
        if self.load_of is None:
            return [TrainLoad(1, ticket.params.n_gpus)] * len(ticket.entries)
        loads = []
        for entry in ticket.entries:
            try:
                loads.append(self.load_of(entry, ticket.params))
            except Exception as exc:  # noqa: BLE001 - a store fault, a tokenizer fault or a bad file: the trainer judges the file
                loads.append(TrainLoad(0, 1))
                print(f"{ticket.round_id}: the load of queue position {entry.queue_pos} (UID {entry.reg.uid}) was not read, so its job has 1 GPU: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return loads

    def _train_all(self, ticket: RoundTicket, train_end: int, at_the_lease: bool) -> dict[int, EntryResult | LoRAArtifact]:
        """Train the batches until ``train_end``, side by side, each job on its own GPUs.

        Returns one report per queue position. At ``train_end`` every job that
        still runs is stopped, so the scoring always starts on time.
        ``at_the_lease``: this pass started at the lease (``AT_THE_LEASE_S``).
        """
        entries, n_gpus = ticket.entries, ticket.params.n_gpus
        side_by_side = len(entries) <= n_gpus  # every batch has its own GPUs and starts at the start of the pass
        # The loads are read only when they change something: GPUs are left over, or the start order follows them.
        needed = len(entries) < n_gpus or (not side_by_side and LONGEST_FIRST)
        loads = self._loads(ticket) if needed else [TrainLoad(1, 1)] * len(entries)
        own: list[tuple[int, ...]] = []
        free: queue.SimpleQueue[tuple[int, ...]] = queue.SimpleQueue()
        if side_by_side:
            for count in gpu_counts(n_gpus, loads):
                first = own[-1][-1] + 1 if own else 0
                own.append(tuple(range(first, first + count)))
        else:
            for gpu in range(n_gpus):
                free.put((gpu,))
        if needed and self.load_of is not None:
            plan = "; ".join(
                f"queue position {entry.queue_pos} (UID {entry.reg.uid}): {load.token_steps} token-steps, {load.rows} rows that train"
                + (f", {len(gpus)} GPU(s)" if side_by_side else "")
                for entry, load, gpus in zip(entries, loads, own or [()] * len(entries))
            )
            print(f"{ticket.round_id}: the loads of the training jobs: {plan}", file=sys.stderr, flush=True)
        order = start_order(loads, LONGEST_FIRST and not side_by_side)
        width = len(entries) if side_by_side else n_gpus  # the batches of the first group: they start with the pass

        def one(place: int, index: int) -> EntryResult | LoRAArtifact:
            gpus = own[index] if side_by_side else free.get()
            try:
                # In a pass that started at the lease, the first group starts at the lease: each has the whole training time.
                return self._train(ticket, entries[index], gpus, train_end, whole_time=at_the_lease and place < width)
            finally:
                if not side_by_side:
                    free.put(gpus)

        try:
            with ThreadPoolExecutor(max(1, width)) as threads:
                jobs = {index: threads.submit(one, place, index) for place, index in enumerate(order)}
                running = set(jobs.values())
                while running:
                    _, running = wait(running, timeout=self.poll_s)
                    if running and self.clock() >= train_end:
                        self.trainer.close()  # the training time is over: stop every job that still runs
                reports = {entry.queue_pos: jobs[index].result() for index, entry in enumerate(entries)}
        finally:
            self.trainer.close()
        return reports

    def _train(
        self, ticket: RoundTicket, entry: Entry, gpu_ids: tuple[int, ...], train_end: int, whole_time: bool
    ) -> EntryResult | LoRAArtifact:
        if self.clock() >= train_end:
            return EntryResult(entry.queue_pos, NOT_TRAINED, reason="the training time of the round ended before this batch could start")
        request = TrainRequest(
            round_id=ticket.round_id,
            head=ticket.head,
            entry=entry,
            params=ticket.params,
            seed=train_seed(ticket.draw.round_seed, entry.batch.batch_sha256),
            gpu_ids=gpu_ids,
            out_prefix=layout.pending_lora_prefix(ticket.round_id, ticket.lease_id, entry.queue_pos),
        )
        reason = ""
        for attempt in range(1, self.attempts + 1):
            try:
                return self.trainer.train(request)
            except BatchRejected as exc:
                return EntryResult(entry.queue_pos, SKIP, reason=f"batch rejected at train time: {exc}", attempts=attempt)
            except InfraError as exc:
                reason = str(exc)
            if self.clock() >= train_end and whole_time and attempt == 1:
                # Admission's cost cap says that a valid batch fits. This one had the whole time in one try and did not
                # finish: a fault of ours, counted for the seat. After a try that failed, the box took part of the time
                # from the batch: then it is not_trained, and nothing is counted.
                reason = f"{OUT_OF_TIME} ({ticket.params.train_time_budget_s} s) and did not finish"
                return EntryResult(entry.queue_pos, NO_DECISION, reason=reason, attempts=attempt)
            if self.clock() >= train_end:
                return EntryResult(entry.queue_pos, NOT_TRAINED, reason="the training time of the round ended while this batch trained", attempts=attempt)
        # A training job that failed is not counted for the seat (EntryResult.counted_for_seat): the cause can be the box.
        return EntryResult(entry.queue_pos, NO_DECISION, reason=f"training failed {self.attempts} times: {reason}", attempts=self.attempts)

    def _look(self, request: EvalRequest) -> EvalChunk | None:
        """One chunk of tasks, or None when it failed after the deadline: an unfinished chunk is never used."""
        models = {NULL_MODEL} | {lora.lora_hash for lora in request.loras}
        reason = ""
        for _ in range(self.attempts):
            try:
                chunk = self.evaluator.evaluate(request)
            except InfraError as exc:
                if self.clock() >= request.deadline:
                    return None
                reason = str(exc)
                continue
            if chunk.task_ids != request.task_ids or set(chunk.outcomes) != models or chunk.image_digest != self.image_digest:
                raise ContractError("the evaluator answered another request than the one it was given")
            return chunk
        raise InfraError(f"evaluation failed {self.attempts} times: {reason}")

    def _evaluate(self, ticket: RoundTicket, trained: dict[int, LoRAArtifact], deadline: int):
        """Return (null outcomes, outcomes per queue position, look at which a position was dropped, stopped at the limit)."""
        params, draw = ticket.params, ticket.draw
        n, samples = len(draw.task_ids), params.samples_per_task
        null = [""] * samples
        scored = {pos: [""] * samples for pos in trained}
        dropped: dict[int, int] = {}
        alive, start, stopped = sorted(trained), 0, False
        try:
            for look, stop in enumerate(params.look_sizes(n)):
                while start < stop and not stopped:
                    if self.clock() >= deadline:
                        stopped = True
                        break
                    end = min(stop, start + params.eval_chunk_tasks)
                    loras = tuple(trained[pos] for pos in alive)
                    chunk = self._look(EvalRequest(ticket.round_id, ticket.head, loras, draw.task_ids[start:end], draw.round_seed, look, params, deadline, len(ticket.entries)))
                    if chunk is None:
                        stopped = True
                        break
                    for sample in range(samples):
                        null[sample] += chunk.outcomes[NULL_MODEL][sample]
                        for pos in alive:
                            scored[pos][sample] += chunk.outcomes[trained[pos].lora_hash][sample]
                    start = end
                if stopped:
                    break
                for pos in list(alive):
                    if drop_at_look(score_entry(draw.weights, null, scored[pos], params), look, n, params):
                        alive.remove(pos)
                        dropped[pos] = look
                if not alive:
                    break
        finally:
            self.evaluator.close()
        pad = lambda strings: tuple(text + NOT_RUN * (n - len(text)) for text in strings)
        return pad(null), {pos: pad(strings) for pos, strings in scored.items()}, dropped, stopped
