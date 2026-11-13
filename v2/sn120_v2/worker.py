"""One box: train each leased batch, then score null M and every LoRA, look by look.

``LocalRunner`` is the ``RoundRunner`` for a box. It calls a ``Trainer`` and
an ``Evaluator`` and returns a sealed ``BoxResult``. It never picks the
winner: it only reports outcomes. The controller judges them.

Rules it keeps:

* One failing batch must not take the box down. A training failure is tried
  again, then that entry becomes ``no_decision`` and the others go on.
* The trainer is closed before evaluation starts, so its GPUs are free.
* Null M is scored in every look, with the same tasks and seeds as the LoRAs.
"""

from __future__ import annotations

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
        started = self.clock()
        slots = gpu_slots(ticket.params.n_gpus, len(ticket.entries))
        reports: dict[int, EntryResult | LoRAArtifact] = {}
        try:
            for index, entry in enumerate(ticket.entries):
                reports[entry.queue_pos] = self._train(ticket, entry, slots[index % len(slots)])
        finally:
            self.trainer.close()
        trained = {pos: report for pos, report in reports.items() if isinstance(report, LoRAArtifact)}
        null, scored, dropped = self._evaluate(ticket, trained) if trained else ((), {}, {})
        entries = []
        for entry in ticket.entries:
            report = reports[entry.queue_pos]
            if isinstance(report, LoRAArtifact):
                report = EntryResult(entry.queue_pos, SCORED, lora=report, outcomes=scored[entry.queue_pos], eliminated_at_look=dropped.get(entry.queue_pos))
            entries.append(report)
        return BoxResult(
            round_id=ticket.round_id,
            lease_id=ticket.lease_id,
            box_id=ticket.box_id,
            ticket_digest=ticket.digest(),
            head_version_id=ticket.head.version_id,
            image_digest=self.image_digest,
            null_outcomes=null,
            entries=tuple(entries),
            started_at=started,
            finished_at=self.clock(),
        )

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

    def _look(self, request: EvalRequest) -> EvalChunk:
        models = {NULL_MODEL} | {lora.lora_hash for lora in request.loras}
        reason = ""
        for _ in range(self.attempts):
            try:
                chunk = self.evaluator.evaluate(request)
            except InfraError as exc:
                reason = str(exc)
                continue
            if chunk.task_ids != request.task_ids or set(chunk.outcomes) != models or chunk.image_digest != self.image_digest:
                raise ContractError("the evaluator answered another request than the one it was given")
            return chunk
        raise InfraError(f"evaluation failed {self.attempts} times: {reason}")

    def _evaluate(self, ticket: RoundTicket, trained: dict[int, LoRAArtifact]):
        """Return (null outcomes, outcomes per queue position, look at which a position was dropped)."""
        params, draw = ticket.params, ticket.draw
        n, samples = len(draw.task_ids), params.samples_per_task
        sizes = params.look_sizes(n)
        null = [""] * samples
        scored = {pos: [""] * samples for pos in trained}
        dropped: dict[int, int] = {}
        alive, start = sorted(trained), 0
        try:
            for look, stop in enumerate(sizes):
                loras = tuple(trained[pos] for pos in alive)
                chunk = self._look(EvalRequest(ticket.round_id, ticket.head, loras, draw.task_ids[start:stop], draw.round_seed, look, params))
                for sample in range(samples):
                    null[sample] += chunk.outcomes[NULL_MODEL][sample]
                    for pos in alive:
                        scored[pos][sample] += chunk.outcomes[trained[pos].lora_hash][sample]
                start = stop
                for pos in list(alive):
                    if drop_at_look(score_entry(draw.weights, null, scored[pos], params), look, n, params):
                        alive.remove(pos)
                        dropped[pos] = look
                if not alive:
                    break
        finally:
            self.evaluator.close()
        pad = lambda strings: tuple(text + NOT_RUN * (n - len(text)) for text in strings)
        return pad(null), {pos: pad(strings) for pos, strings in scored.items()}, dropped
