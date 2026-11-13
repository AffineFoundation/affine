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
* The round's time limit is a backstop. The box looks at the clock before
  each training and after each chunk of tasks. Past the deadline it starts
  nothing new: what is scored by then is the whole test.
"""

from __future__ import annotations

import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from . import layout
from .errors import BatchRejected, ContractError, InfraError
from .judging import drop_at_look, score_entry
from .sampling import train_seed
from .seams import Evaluator, Trainer
from .types import (
    NO_DECISION,
    NOT_RUN,
    NULL_MODEL,
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


def gpu_slots(n_gpus: int, n_batches: int) -> list[tuple[int, ...]]:
    """Split the box's GPUs into equal slots of floor(n_gpus / batches) GPUs, at least one GPU each."""
    per_lora = max(1, n_gpus // max(1, n_batches))
    return [tuple(range(start, start + per_lora)) for start in range(0, n_gpus - per_lora + 1, per_lora)]


class LocalRunner:
    def __init__(self, trainer: Trainer, evaluator: Evaluator, clock: Callable[[], int], image_digest: str, attempts: int = 3) -> None:
        self.trainer, self.evaluator = trainer, evaluator
        self.clock, self.image_digest, self.attempts = clock, image_digest, attempts

    def run(self, ticket: RoundTicket) -> BoxResult:
        started, params = self.clock(), ticket.params
        # The box stops early enough that judging and promotion still fit inside the round's limit.
        deadline = min(ticket.expires_at, ticket.issued_at + params.round_time_limit_s) - params.promote_reserve_s
        reports = self._train_all(ticket, deadline)
        trained = {pos: report for pos, report in reports.items() if isinstance(report, LoRAArtifact)}
        train_done = self.clock()
        null, scored, dropped, stopped = self._evaluate(ticket, trained, deadline) if trained else ((), {}, {}, False)
        first_look = params.look_sizes(len(ticket.draw.task_ids))[0]
        n_scored = len(null[0].rstrip(NOT_RUN)) if null else 0
        entries = []
        for entry in ticket.entries:
            report = reports[entry.queue_pos]
            if isinstance(report, LoRAArtifact) and stopped and n_scored < first_look:
                reason = f"the round's time limit ended the scoring after {n_scored} tasks, before the first look of {first_look}"
                report = EntryResult(entry.queue_pos, NO_DECISION, reason=reason)
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

    def _train_all(self, ticket: RoundTicket, deadline: int) -> dict[int, EntryResult | LoRAArtifact]:
        """Train every batch, as many at a time as there are GPU slots. Returns one report per queue position."""
        slots = gpu_slots(ticket.params.n_gpus, len(ticket.entries))
        free: queue.SimpleQueue[tuple[int, ...]] = queue.SimpleQueue()
        for slot in slots:
            free.put(slot)

        def one(entry: Entry) -> EntryResult | LoRAArtifact:
            slot = free.get()
            try:
                if self.clock() >= deadline:
                    return EntryResult(entry.queue_pos, NO_DECISION, reason="the round's time limit came before this batch could train")
                return self._train(ticket, entry, slot)
            finally:
                free.put(slot)

        try:
            with ThreadPoolExecutor(len(slots)) as threads:
                reports = list(threads.map(one, ticket.entries))
        finally:
            self.trainer.close()
        return {entry.queue_pos: report for entry, report in zip(ticket.entries, reports)}

    def _train(self, ticket: RoundTicket, entry: Entry, gpu_ids: tuple[int, ...]) -> EntryResult | LoRAArtifact:
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
                    chunk = self._look(EvalRequest(ticket.round_id, ticket.head, loras, draw.task_ids[start:end], draw.round_seed, look, params, deadline))
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
