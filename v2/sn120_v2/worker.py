"""One box: train each leased batch, then score null M and every LoRA, look by look.

``LocalRunner`` is the ``RoundRunner`` for a box. It calls a ``Trainer`` and
an ``Evaluator`` and returns a sealed ``BoxResult``. It never picks the
winner: it only reports outcomes. The controller judges them.

Rules it keeps:

* One failing batch must not take the box down. A training failure is tried
  again, then that entry becomes ``no_decision`` and the others go on.
* Batches train at the same time, each on its own slot of GPUs. With more
  batches than slots, a batch waits for a free slot.
* The trainer is closed before evaluation starts, so its GPUs are free.
* Null M is scored in every look, with the same tasks and seeds as the LoRAs.
* The round's time limit is hard. It cuts every look, the first one too:
  - Training has its own time, ``train_time_budget_s`` from the lease. At its
    end no new training starts and every job that still runs is stopped. Such
    a batch is ``not_trained``: not charged, first in the next round, and no
    failure is counted for its seat. So the scoring always starts on time.
  - Past the limit no new rollout starts. The box uses the largest complete
    grading-block boundary shared by null M and every open LoRA.
  - A LoRA with fewer scored tasks than the first look at the limit is
    ``not_scored``: not charged, first in the next round, and no failure is
    counted for its seat.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable

from . import layout
from .errors import BatchRejected, ContractError, InfraError
from .judging import drop_at_look, score_entry
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
    EvalBlock,
    EvalRequest,
    eval_blocks,
    LoRAArtifact,
    RoundTicket,
    TrainRequest,
)

# A pass of the box started "at the lease" when it started at most this long after the start of the lease. Only then
# did a batch of the first group have the whole training time. A pass that started later (run started again in the
# round, or the round ran again under the same lease) gave it less: a batch that does not finish is then not_trained,
# and nothing is counted. A normal pass starts a few seconds after the lease: the controller writes the ticket, the
# journal record and the public files in between. The weight hash and the downloads come after the start of the pass.
AT_THE_LEASE_S = 120


def gpu_slots(n_gpus: int, n_batches: int) -> list[tuple[int, ...]]:
    """Split the box's GPUs into equal slots of floor(n_gpus / batches) GPUs, at least one GPU each."""
    per_lora = max(1, n_gpus // max(1, n_batches))
    return [tuple(range(start, start + per_lora)) for start in range(0, n_gpus - per_lora + 1, per_lora)]


class LocalRunner:
    def __init__(
        self, trainer: Trainer, evaluator: Evaluator, clock: Callable[[], int], image_digest: str, attempts: int = 3, poll_s: float = 1.0
    ) -> None:
        self.trainer, self.evaluator = trainer, evaluator
        self.clock, self.image_digest, self.attempts, self.poll_s = clock, image_digest, attempts, poll_s

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

    def _train_all(self, ticket: RoundTicket, train_end: int, at_the_lease: bool) -> dict[int, EntryResult | LoRAArtifact]:
        """Train the batches in queue order until ``train_end``, as many at a time as there are GPU slots.

        Returns one report per queue position. At ``train_end`` every job that
        still runs is stopped, so the scoring always starts on time.
        ``at_the_lease``: this pass started at the lease (``AT_THE_LEASE_S``).
        """
        slots = gpu_slots(ticket.params.n_gpus, len(ticket.entries))
        free: queue.SimpleQueue[tuple[int, ...]] = queue.SimpleQueue()
        for slot in slots:
            free.put(slot)

        def one(index: int, entry: Entry) -> EntryResult | LoRAArtifact:
            slot = free.get()
            try:
                # In a pass that started at the lease, the first batches start at the lease: each has the whole training time.
                return self._train(ticket, entry, slot, train_end, whole_time=at_the_lease and index < len(slots))
            finally:
                free.put(slot)

        try:
            with ThreadPoolExecutor(len(slots)) as threads:
                jobs = [threads.submit(one, index, entry) for index, entry in enumerate(ticket.entries)]
                running = set(jobs)
                while running:
                    _, running = wait(running, timeout=self.poll_s)
                    if running and self.clock() >= train_end:
                        self.trainer.close()  # the training time is over: stop every job that still runs
                reports = [job.result() for job in jobs]
        finally:
            self.trainer.close()
        return {entry.queue_pos: report for entry, report in zip(ticket.entries, reports)}

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

    def _evaluate(self, ticket: RoundTicket, trained: dict[int, LoRAArtifact], deadline: int):
        """Return (null outcomes, outcomes per queue position, look at which a position was dropped, stopped at the limit).

        Replies may complete and blocks may arrive out of order. Only exact prefixes enter the drop rule, so speculation
        never changes a look decision. All grading callbacks return before this generator closes.
        """
        params, draw = ticket.params, ticket.draw
        n, samples = len(draw.task_ids), params.samples_per_task
        looks = params.look_sizes(n)
        valid_blocks = {(start, end) for start, end in eval_blocks(params.eval_chunk_tasks, looks)}
        names = {NULL_MODEL: NULL_MODEL} | {trained[pos].lora_hash: pos for pos in trained}
        texts = {model: [""] * samples for model in names}
        arrived: dict[tuple[str, int], EvalBlock] = {}
        ready, dropped = {model: 0 for model in names}, {}
        alive = set(trained)
        stopped = self.clock() >= deadline
        stream_lock = threading.Lock()
        stream_done = False
        dispatch_stopped = stopped

        def limit(model: str) -> int:
            with stream_lock:
                if dispatch_stopped or stream_done:
                    return 0
                if model == NULL_MODEL:
                    # Every alive LoRA needs null M. Without one, null ends at the largest look reached by a drop.
                    wanted = max((dropped[pos] for pos in dropped if pos not in alive), default=-1)
                    return n if alive else looks[wanted] if wanted >= 0 else 0
                pos = names[model]
                return n if pos in alive else looks[dropped[pos]]

        def advance(model: str) -> None:
            while (block := arrived.pop((model, ready[model]), None)) is not None:
                if block.model != model or block.end > wanted(model) or len(block.outcomes) != samples:
                    raise ContractError("the evaluator answered another request than the one it was given")
                for sample in range(samples):
                    texts[model][sample] += block.outcomes[sample]
                if len(texts[model][0]) != block.end:
                    raise ContractError("the evaluator returned an overlapping or duplicate eval block")
                ready[model] = block.end

        def judge() -> None:
            """Apply looks in order as soon as null M and this model have the whole graded prefix."""
            for pos in sorted(alive):
                model = trained[pos].lora_hash
                for look, stop in enumerate(looks[:-1]):
                    if ready[model] >= stop and ready[NULL_MODEL] >= stop:
                        partial_null = tuple(text[:stop] for text in texts[NULL_MODEL])
                        partial_lora = tuple(text[:stop] for text in texts[model])
                        if drop_at_look(score_entry(draw.weights, partial_null, partial_lora, params), look, n, params):
                            alive.remove(pos)
                            dropped[pos] = look
                            for sample in range(samples):
                                texts[model][sample] = texts[model][sample][:stop]
                            ready[model] = stop
                            # A speculative block past this look is never an outcome.
                            for key in [key for key in arrived if key[0] == model]:
                                del arrived[key]
                            break

        def wanted(model: str) -> int:
            with stream_lock:
                if model == NULL_MODEL:
                    return max(ready.values(), default=0) if not alive and dropped else n
                pos = names[model]
                return n if pos in alive else looks[dropped[pos]]

        request = EvalRequest(ticket.round_id, ticket.head, tuple(trained[pos] for pos in sorted(trained)), draw.task_ids, draw.round_seed, 0, params, deadline, len(ticket.entries))
        try:
            for block in self.evaluator.stream(request, limit):
                stopped = stopped or self.clock() >= deadline
                dispatch_stopped = stopped
                if block.model not in names or (block.start, block.end) not in valid_blocks:
                    raise ContractError("the evaluator answered another request than the one it was given")
                if block.start < wanted(block.model):
                    arrived[block.model, block.start] = block
                    advance(block.model)
                judge()
            # An evaluator may stop dispatching at the deadline, but must still grade every complete block with a reply.
            for model in sorted(names):
                advance(model)
                judge()
            stopped = stopped or self.clock() >= deadline
        finally:
            stream_done = True
            self.evaluator.close()

        if stopped:
            # Exactly the old chunk ends: looks contain the earlier block boundaries, then eval_chunk_tasks steps onward.
            boundaries = [end for _, end in eval_blocks(params.eval_chunk_tasks, looks)]

            def boundary(end: int) -> int:
                return max((stop for stop in boundaries if stop <= end), default=0)

            common = min(boundary(ready[NULL_MODEL]), *(boundary(ready[trained[pos].lora_hash]) for pos in alive))
            for pos, look in list(dropped.items()):
                if looks[look] > common:
                    del dropped[pos]
                    alive.add(pos)
            for model, pos in names.items():
                if model != NULL_MODEL and pos not in alive:
                    continue
                for sample in range(samples):
                    texts[model][sample] = texts[model][sample][:common]
                ready[model] = common
        elif alive:
            # A stream that ends early without the clock or a server failure is a broken evaluator, not a short score.
            wanted = {model: n for model, pos in names.items() if model != NULL_MODEL and pos in alive}
            if not alive:
                wanted[NULL_MODEL] = looks[max(dropped.values())] if dropped else 0
            else:
                wanted[NULL_MODEL] = n
            if any(ready[model] < end and limit(model) >= end for model, end in wanted.items()):
                raise InfraError("the evaluator ended before every alive model completed the draw")

        null = tuple(texts[NULL_MODEL])
        if not stopped and not alive and dropped:
            null_stop = looks[max(dropped.values())]
            null = tuple(text[:null_stop] for text in null)
        scored = {pos: tuple(texts[trained[pos].lora_hash]) for pos in trained}
        pad = lambda strings: tuple(text + NOT_RUN * (n - len(text)) for text in strings)
        return pad(null), {pos: pad(strings) for pos, strings in scored.items()}, dropped, stopped
