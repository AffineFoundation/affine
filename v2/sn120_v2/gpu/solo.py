"""Solo GPU execution: independent per-GPU training, serving and stealing.

A GPU never runs vLLM while its training job is active. Idle GPUs may start
the normal default-argv server immediately for null. After training exits,
that GPU starts or reuses the same default-argv server. Workers take work
from a shared, atomic claim scheduler. The training call, deadline, task
draw, sampling seed and look rule are unchanged. A resource budget may lower
vLLM's sequence limit and the matching number of HTTP clients. While any GPU
is still training, each worker's in-flight requests are capped separately.
"""

from __future__ import annotations

import queue
import resource
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Sequence

from ..errors import InfraError
from ..judging import drop_at_look, score_entry
from ..types import KL_FIELDS, NOT_RUN, NOT_SCORED, NULL_MODEL, SCORED, BoxResult, EntryResult, EvalRequest, LoRAArtifact, RoundTicket
from ..worker import AT_THE_LEASE_S, LocalRunner
from .evaluator import _Cancelled, _Deadline
from .plan import log_phase


@dataclass(frozen=True)
class SoloSettings:
    """Execution-only settings, deliberately outside the hashed Params."""

    prewarm: bool = True
    steal_block_tasks: int = 16
    poll_s: float = 0.05
    clients_while_training: int = 256


def lpt_train_all(entries: Sequence, prior: dict[int, float] | None = None) -> list:
    """Longest expected training first, with queue position breaking ties."""
    prior = prior or {}
    return sorted(entries, key=lambda entry: (-prior.get(entry.queue_pos, float(entry.batch.n_rollouts)), entry.queue_pos))


def choose_claim(remaining: Counter, alive: set[str], n_tasks: int, block: int, owner: str = "", own_first: bool = False) -> tuple[str, int, int] | None:
    """Choose null, own work, or the longest remaining live candidate.

    Values are the next unclaimed task indexes. Null leads when it is not
    ahead of the owner's next task. Otherwise the owner goes first; a worker
    without an owner steals from the candidate with the most unclaimed work.
    """
    candidates = [model for model in remaining if remaining[model] < n_tasks and (model == NULL_MODEL or model in alive)]
    if not candidates:
        return None
    if own_first and owner in candidates:
        model = owner
    elif owner not in candidates and owner != NULL_MODEL:
        model = max(candidates, key=lambda name: (n_tasks - remaining[name], name == NULL_MODEL))
    elif NULL_MODEL in candidates and (owner not in remaining or remaining[NULL_MODEL] <= remaining[owner]):
        model = NULL_MODEL
    elif owner in candidates:
        model = owner
    else:
        model = max(candidates, key=lambda name: (n_tasks - remaining[name], name == NULL_MODEL))
    start = remaining[model]
    return model, start, min(n_tasks, start + block)


class SoloScheduler:
    """Atomic work claims and ordered, idempotent outcomes for one round."""

    def __init__(self, runner: "SoloRunner", ticket: RoundTicket, deadline: int):
        self.runner, self.ticket, self.deadline = runner, ticket, deadline
        self.params, self.draw = ticket.params, ticket.draw
        self.n, self.samples = len(self.draw.task_ids), ticket.params.samples_per_task
        self.null_samples = ticket.params.null_samples
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.owner_by_model: dict[str, int] = {}
        self.trained: dict[int, LoRAArtifact] = {}
        self.tested: dict[int, object] = {}
        self.alive: set[int] = set()
        self.dropped: dict[int, int] = {}
        self.remaining = Counter({NULL_MODEL: 0})
        self.next_result = {(NULL_MODEL, sample): 0 for sample in range(self.null_samples)}
        self.buffered: dict[tuple[str, int, int], str] = {}
        self.null = [""] * self.null_samples
        self.scored: dict[int, list[str]] = {}
        self.judged: set[tuple[int, int]] = set()
        self.cut = False
        self.confirmation_needed = ticket.parent is not None
        self.failure: BaseException | None = None
        self.training_done = False
        self.seen_results: dict[tuple[str, int, int], str] = {}

    def add_lora(self, pos: int, lora: LoRAArtifact, gpu: int | None = None, tested=None) -> None:
        with self.changed:
            if pos in self.trained:
                return
            if self.params.eval_merged:
                if tested is None or not tested.path.is_dir():
                    raise InfraError("eval_merged needs a built tested base before a candidate can be scored")
                self.tested[pos] = tested
            self.trained[pos] = lora
            if gpu is not None:
                self.owner_by_model[lora.lora_hash] = gpu
            self.alive.add(pos)
            self.scored[pos] = [""] * self.samples
            self.remaining[lora.lora_hash] = 0
            self.next_result.update({(lora.lora_hash, sample): 0 for sample in range(self.samples)})
            self.changed.notify_all()

    def training_finished(self) -> None:
        with self.changed:
            self.training_done = True
            self._cancel_null_if_unused()
            self.changed.notify_all()

    def _cancel_null_if_unused(self) -> None:
        """No future LoRA can use king work after training and the last drop."""
        if self.training_done and not self.alive and not self.confirmation_needed:
            cancel = getattr(self.runner.evaluator.http, "cancel", None)
            if cancel is not None:
                cancel(NULL_MODEL)

    def model_for_gpu(self, gpu: int) -> str:
        with self.changed:
            return next((model for model, owner in self.owner_by_model.items() if owner == gpu), NULL_MODEL)

    def merged_dir(self, model: str):
        if model == NULL_MODEL:
            return None
        pos = self._pos(model)
        if pos not in self.tested:
            raise InfraError(f"no tested merged base is recorded for {model}")
        return self.tested[pos].path

    def steal(self, gpu: int, owner: str) -> tuple[str, int, int] | None:
        with self.changed:
            while True:
                if self.failure is not None or self.cut or (self.training_done and not self.alive and not self.confirmation_needed):
                    return None
                names = {lora.lora_hash for lora in self.trained.values()}
                claim = choose_claim(self.remaining, names & {self.trained[pos].lora_hash for pos in self.alive}, self.n, self.runner.settings.steal_block_tasks, owner, own_first=self.params.eval_merged)
                if claim is not None:
                    model, start, stop = claim
                    self.remaining[model] = stop
                    return claim
                if self.training_done:
                    return None
                self.changed.wait(self.runner.settings.poll_s)

    def result(self, model: str, sample: int, index: int, code: str) -> None:
        with self.changed:
            if model != NULL_MODEL and self._pos(model) not in self.alive:
                return
            job = (model, sample, index)
            previous = self.seen_results.get(job)
            if previous is not None:
                if previous != code:
                    raise InfraError("a retried rollout disagrees with its first outcome")
                return
            self.seen_results[job] = code
            self.buffered[job] = code
            self._drain(model, sample)
            self.changed.notify_all()

    def _drain(self, model: str, sample: int) -> None:
        while (model, sample, self.next_result[model, sample]) in self.buffered:
            index = self.next_result[model, sample]
            code = self.buffered.pop((model, sample, index))
            if model == NULL_MODEL:
                self.null[sample] += code
            else:
                pos = self._pos(model)
                if pos not in self.alive:
                    return
                self.scored[pos][sample] += code
            self.next_result[model, sample] += 1
        # Null can catch up after a LoRA has crossed a look. Re-check every
        # unjudged early look after every accepted result, including null.
        for pos in list(self.alive):
            for look in range(len(self.params.look_sizes(self.n)) - 1):
                self.judge(pos, look)

    def judge(self, pos: int, look: int) -> None:
        with self.changed:
            if pos not in self.alive or (pos, look) in self.judged:
                return
            stop = self.params.look_sizes(self.n)[look]
            if any(len(text) < stop for text in self.null) or any(len(text) < stop for text in self.scored[pos]):
                return
            self.judged.add((pos, look))
            exact = tuple(text[:stop] for text in self.scored[pos])
            if drop_at_look(score_entry(self.draw.weights, self.null, exact, self.params), look, self.n, self.params):
                self.alive.remove(pos)
                self.dropped[pos] = look
                self.scored[pos][:] = [text[:stop] for text in self.scored[pos]]
                if hasattr(self.runner.evaluator.http, "cancel"):
                    self.runner.evaluator.http.cancel(self.trained[pos].lora_hash)  # type: ignore[attr-defined]
                self._cancel_null_if_unused()
                self.changed.notify_all()

    def deadline_cut(self) -> None:
        with self.changed:
            self.cut = True
            self.changed.notify_all()

    def fail(self, error: BaseException) -> None:
        with self.changed:
            if self.failure is None:
                self.failure = error
            cancel = getattr(self.runner.evaluator.http, "cancel", None)
            if cancel is not None:
                cancel(NULL_MODEL)
                for lora in self.trained.values():
                    cancel(lora.lora_hash)
            self.changed.notify_all()

    def finish(self):
        with self.lock:
            if self.failure is not None:
                raise self.failure
            stopped = self.cut
            if stopped:
                common = min([len(text) for text in self.null] + [len(text) for pos in self.alive for text in self.scored[pos]])
                common = common // self.params.eval_chunk_tasks * self.params.eval_chunk_tasks
                if not common:
                    raise InfraError("solo scoring reached the time limit before null and every live LoRA completed a shared block")
                self.null[:] = [text[:common] for text in self.null]
                for pos in self.alive:
                    self.scored[pos][:] = [text[:common] for text in self.scored[pos]]
            elif not self.alive and not self.confirmation_needed:
                last = max(self.dropped.values(), default=-1)
                keep = self.params.look_sizes(self.n)[last] if last >= 0 else 0
                self.null[:] = [text[:keep] for text in self.null]
            else:
                if self.confirmation_needed and any(len(text) != self.n for text in self.null):
                    raise InfraError("confirmation scoring ended before the new-head null completed the full fresh draw")
                for pos in self.alive:
                    if any(len(text) != self.n for text in self.scored[pos]) or any(len(text) != self.n for text in self.null):
                        raise InfraError("solo scoring ended before null and every live LoRA completed")
            pad = lambda strings: tuple(text + NOT_RUN * (self.n - len(text)) for text in strings)
            return pad(self.null), {pos: pad(strings) for pos, strings in self.scored.items()}, self.dropped, stopped

    def _pos(self, model: str) -> int:
        return next(pos for pos, lora in self.trained.items() if lora.lora_hash == model)


class SoloWorker:
    """One GPU's vLLM server and its continuously fed HTTP clients."""

    def __init__(self, evaluator, ticket: RoundTicket, trained: dict[int, LoRAArtifact], deadline: int, gpu: int):
        self.evaluator, self.ticket, self.deadline, self.gpu = evaluator, ticket, deadline, gpu
        self.trained_source = trained
        self.scheduler = getattr(evaluator, "_solo_scheduler", None)
        self.owner_by_gpu: dict[int, str] = {}
        self.params = ticket.params
        self.trained = self.trained_source
        self.clients = getattr(evaluator, "_solo_clients", None) or evaluator.clients_for(self.params)
        self.owner = next((lora.lora_hash for pos, lora in self.trained_source.items() if getattr(lora, "train_gpu", None) == gpu), NULL_MODEL)

    def _client_cap(self) -> int:
        """In-flight requests: capped while any GPU is still training."""
        scheduler = self.scheduler
        if scheduler is None or scheduler.training_done:
            return self.clients
        return min(self.clients, scheduler.runner.settings.clients_while_training)

    def run(self, *, steal, result, judge, alive, deadline):
        if self.scheduler is not None:
            self.trained_source = self.scheduler.trained
            self.owner = next((model for model, gpu_id in self.scheduler.owner_by_model.items() if gpu_id == self.gpu), NULL_MODEL)
        request = EvalRequest(self.ticket.round_id, self.ticket.head, tuple(self.trained_source.values()), self.ticket.draw.task_ids, self.ticket.draw.round_seed, 0, self.params, deadline, len(self.ticket.entries))
        bound = self.owner if self.params.eval_merged else NULL_MODEL
        url = (self.evaluator._serve_one(self.ticket.head, self.params, len(self.ticket.entries), self.gpu, model=bound, base_dir=self.scheduler.merged_dir(bound))
               if self.params.eval_merged else self.evaluator._serve_one(self.ticket.head, self.params, len(self.ticket.entries), self.gpu))
        written = []
        pending: set[Future] = set()
        stopped = False
        held_claim = None

        def switch(model):
            nonlocal bound, url
            if model == bound:
                return
            if pending:
                raise InfraError("a merged server cannot switch models with rollouts in flight")
            url = self.evaluator._serve_one(self.ticket.head, self.params, len(self.ticket.entries), self.gpu, model=model, base_dir=self.scheduler.merged_dir(model))
            bound = model

        def submit(executor, model, sample, index):
            job = (model, sample, index)
            future = executor.submit(self.evaluator._rollout, request, url, job, written)
            future.solo_job = job
            return future

        with ThreadPoolExecutor(self.clients) as clients:
            try:
                while pending or not stopped:
                    while not stopped and len(pending) < self._client_cap() and held_claim is None:
                        claim = steal(self.gpu, self.owner)
                        if claim is None:
                            stopped = True
                            break
                        model, start, stop = claim
                        if self.params.eval_merged and model != bound:
                            if pending:
                                held_claim = claim
                                break
                            switch(model)
                        elif not self.params.eval_merged and model != NULL_MODEL:
                            lora = next(lora for lora in self.trained_source.values() if lora.lora_hash == model)
                            self.evaluator._load_one(self.gpu, lora)
                        for index in range(start, stop):
                            for sample in range(self.params.null_samples if model == NULL_MODEL else self.params.samples_per_task):
                                pending.add(submit(clients, model, sample, index))
                    if not pending:
                        if held_claim is not None:
                            if stopped:
                                break
                            claim, held_claim = held_claim, None
                            switch(claim[0])
                            model, start, stop = claim
                            for index in range(start, stop):
                                for sample in range(self.params.null_samples if model == NULL_MODEL else self.params.samples_per_task):
                                    pending.add(submit(clients, model, sample, index))
                            continue
                        break
                    if not pending:
                        break
                    done, pending = wait(pending, timeout=self.runner_poll_s(), return_when=FIRST_COMPLETED)
                    if not done:
                        continue
                    jobs, replies = [], []
                    for future in done:
                        job = future.solo_job
                        try:
                            reply = future.result()
                        except _Deadline:
                            stopped = True
                            if self.scheduler is not None:
                                self.scheduler.deadline_cut()
                            continue
                        except _Cancelled:
                            continue
                        if reply is None:
                            continue
                        jobs.append(job)
                        replies.append(reply)
                    if jobs:
                        codes = self.evaluator._grade(request.task_ids, jobs, replies, request)
                        for job in jobs:
                            result(job[0], job[1], job[2], codes[job])
                    if not all(self.evaluator._solo_servers[self.gpu].alive() for _ in [0]):
                        raise InfraError(f"the vLLM server on GPU {self.gpu} died during solo scoring")
            finally:
                if stopped and ((self.deadline and time.time() >= self.deadline)
                                or (self.scheduler is not None and self.scheduler.training_done and not self.scheduler.alive
                                    and not self.scheduler.confirmation_needed)):
                    cancel = getattr(self.evaluator.http, "cancel", None)
                    if cancel is not None:
                        cancel(NULL_MODEL)
                        for lora in self.trained_source.values():
                            cancel(lora.lora_hash)
                    for future in pending:
                        future.cancel()
                wait(pending)

    @staticmethod
    def runner_poll_s() -> float:
        return 0.05


class SoloRunner(LocalRunner):
    """Production runner with LPT training and solo serve/steal workers."""

    def __init__(self, trainer, evaluator, clock, image_digest, *, settings: SoloSettings = SoloSettings(), attempts: int = 3, poll_s: float = 1.0):
        super().__init__(trainer, evaluator, clock, image_digest, attempts=attempts, poll_s=poll_s)
        self.settings = settings

    def _close_resources(self) -> None:
        """Every round releases the GPU processes and the grader's worker pipes."""
        errors = []
        for resource in (self.trainer, self.evaluator):
            try:
                resource.close()
            except Exception as exc:  # noqa: BLE001 - both owners must be torn down
                errors.append(exc)
        if errors:
            raise InfraError(f"round cleanup failed: {errors[0]}") from errors[0]

    def run(self, ticket: RoundTicket):
        if hasattr(self.evaluator, "solo_worker"):
            try:
                return self._solo_run_round(ticket)
            finally:
                try:
                    self._close_resources()
                finally:
                    cleanup = getattr(self.evaluator, "cleanup_merged", None)
                    if cleanup is not None:
                        cleanup(ticket.round_id)
        return super().run(ticket)

    def _solo_run_round(self, ticket: RoundTicket):
        started, params = self.clock(), ticket.params
        wall_started = time.monotonic()
        begin_round = getattr(self.evaluator, "begin_round", None)
        if begin_round is not None:
            begin_round(ticket)
        if hasattr(self.evaluator, "solo_client_budget"):
            self.evaluator._solo_clients = self.evaluator.solo_client_budget(params.n_gpus, self.evaluator.clients_for(params))
        elif hasattr(self.evaluator, "settings"):
            self.evaluator._solo_clients = self.evaluator.clients_for(params)
        limit = min(ticket.expires_at, ticket.issued_at + params.round_time_limit_s) - params.promote_reserve_s
        train_end = min(limit, ticket.issued_at + params.train_time_budget_s)
        reports: dict[int, object] = {}
        scheduler = SoloScheduler(self, ticket, limit)
        self.evaluator._solo_scheduler = scheduler
        self._tested = scheduler.tested
        entries = lpt_train_all(ticket.entries)
        training: queue.Queue = queue.Queue()
        for index, entry in enumerate(entries):
            training.put((index, entry))
        train_done = [started]
        train_lock = threading.Lock()
        failed: queue.Queue = queue.Queue()
        remaining_training = [len(entries)]
        training_gpus: set[int] = set()
        warm_futures: dict[int, Future] = {}
        warm_pool = ThreadPoolExecutor(params.n_gpus)

        def start_idle_server(gpu: int) -> None:
            """Start the normal null server, or this GPU's trained merged model."""
            if not self.settings.prewarm:
                return
            with train_lock:
                if gpu in training_gpus:
                    return
                previous = warm_futures.get(gpu)
                if not params.eval_merged:
                    if previous is not None and previous.done():
                        try:
                            previous.result()
                        except Exception:  # the worker's ordinary startup retries a failed warm start
                            pass
                        else:
                            return
                    try:
                        self.evaluator._serve_one(ticket.head, params, len(ticket.entries), gpu)
                    except Exception:  # the worker below performs the normal startup retry
                        pass
                    return
                if previous is not None and not previous.done():
                    return
                model = scheduler.model_for_gpu(gpu)
                # A failed early start is retried here, not fatal to the round.
                warm_futures[gpu] = warm_pool.submit(self.evaluator._serve_one, ticket.head, params, len(ticket.entries), gpu, model=model, base_dir=scheduler.merged_dir(model) if model != NULL_MODEL else None)

        def finish_warm(gpu: int) -> None:
            future = warm_futures.get(gpu)
            if future is not None:
                try:
                    future.result()
                except Exception:  # noqa: BLE001 - normal post-training startup retries
                    pass

        def gpu_worker(gpu: int):
            try:
                while True:
                    # Claim the next training job before deciding to start a
                    # server. This check is atomic with the busy-state update.
                    with train_lock:
                        if self.clock() >= train_end:
                            job = None
                        else:
                            try:
                                job = training.get_nowait()
                            except queue.Empty:
                                job = None
                        if job is None:
                            training_gpus.discard(gpu)
                        else:
                            training_gpus.add(gpu)
                    if job is None:
                        break
                    index, entry = job
                    report = self._train(ticket, entry, (gpu,), train_end, whole_time=started - ticket.issued_at <= AT_THE_LEASE_S and index < params.n_gpus)
                    reports[entry.queue_pos] = report
                    if isinstance(report, LoRAArtifact):
                        tested = self.evaluator.build_merged(ticket.round_id, ticket.head, report, params, gpu) if params.eval_merged else None
                        scheduler.add_lora(entry.queue_pos, report, gpu, tested)
                    with train_lock:
                        remaining_training[0] -= 1
                        train_done[0] = max(train_done[0], self.clock())
                        if remaining_training[0] == 0:
                            scheduler.training_finished()
                # Only now can this GPU have no further training assignment.
                start_idle_server(gpu)
                finish_warm(gpu)
                worker = self.evaluator.solo_worker(ticket, scheduler.trained, limit, gpu)
                worker.run(steal=scheduler.steal, result=scheduler.result, judge=scheduler.judge, alive=lambda: bool(scheduler.alive), deadline=limit)
            except BaseException as exc:  # noqa: BLE001
                failed.put(exc)
                scheduler.fail(exc)

        try:
            with ThreadPoolExecutor(params.n_gpus) as pool:
                futures = [pool.submit(gpu_worker, gpu) for gpu in range(params.n_gpus)]
                running = set(futures)
                while running:
                    _, running = wait(running, timeout=self.poll_s)
                    if self.clock() >= train_end and running:
                        self.trainer.close()
                    if self.clock() >= limit and running:
                        scheduler.deadline_cut()
                self.trainer.close()
        finally:
            warm_pool.shutdown(wait=True, cancel_futures=True)
        if self.clock() >= limit:
            scheduler.deadline_cut()
        if not failed.empty():
            raise failed.get()
        outcome = scheduler.finish()
        if any(len(text.rstrip(NOT_RUN)) != len(ticket.draw.task_ids) for text in outcome[0]):
            parent_outcomes, parent_error = (), "the new-head null did not complete the full fresh draw"
        else:
            parent_outcomes, parent_error = self._score_parent(ticket, limit)
        tested = dict(scheduler.tested)
        eval_done = self.clock()
        eval_wall_s = time.monotonic() - wall_started
        if params.min_kl_seq > 0 and (any(isinstance(report, LoRAArtifact) for report in reports.values()) or outcome[0]):
            # All vLLM servers must release the cards before torch loads two models.
            self.evaluator.close()
            scored_reports = {
                entry.queue_pos: EntryResult(entry.queue_pos, SCORED, lora=reports[entry.queue_pos], outcomes=outcome[1][entry.queue_pos])
                for entry in ticket.entries
                if isinstance(reports[entry.queue_pos], LoRAArtifact) and entry.queue_pos in outcome[1]
            }
            measured = self.evaluator.measure_kl(ticket, list(scored_reports.values()), outcome[0], tested)
            reports.update(self._kl_reports(ticket, reports, measured["entries"], outcome[0], outcome[3]))
            null_kl = measured.get("null") or {}
            kl_wall_s = measured["wall_s"]
            kl_metrics = measured["entries"]
        else:
            null_kl, kl_wall_s, kl_metrics = {}, 0.0, {}
        if hasattr(self.evaluator, "config"):
            clients = getattr(self.evaluator, "_solo_clients", self.evaluator.clients_for(params))
            log_phase(
                self.evaluator.config.work_dir,
                "solo_eval",
                round_id=ticket.round_id,
                wall_s=time.monotonic() - wall_started,
                train_s=train_done[0] - started,
                eval_s=eval_wall_s,
                n_gpus=params.n_gpus,
                configured_clients_per_server=self.evaluator.clients_for(params),
                clients_per_server=clients,
                open_files_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0],
            )
        return self._solo_result(ticket, reports, *outcome, started, train_done[0], null_kl, kl_wall_s, kl_metrics, eval_done=eval_done,
                                 parent_outcomes=parent_outcomes, parent_error=parent_error)

    @staticmethod
    def _kl_reports(ticket, reports, measurements, null, stopped):
        """A per-entry KL failure is uncharged and cannot win; every peer is sealed."""
        from ..types import NO_DECISION

        first_look = ticket.params.look_sizes(len(ticket.draw.task_ids))[0]
        n_scored = len(null[0].rstrip(NOT_RUN)) if null else 0
        updated = {}
        for pos, report in reports.items():
            if not isinstance(report, LoRAArtifact):
                continue
            if stopped and n_scored < first_look:
                continue  # _solo_result preserves the ordinary uncharged not_scored outcome.
            measured = measurements.get(pos)
            if measured is None or "error" in measured or "kl_seq" not in measured:
                reason = "KL measurement failed: " + (measured or {}).get("error", "no measurement returned")
                updated[pos] = EntryResult(pos, NO_DECISION, reason=reason)
        return updated

    def _solo_result(self, ticket, reports, null, scored, dropped, stopped, started, train_done, null_kl=None, kl_wall_s=0.0, kl_metrics=None, eval_done=None, parent_outcomes=(), parent_error=""):
        params = ticket.params
        first_look = params.look_sizes(len(ticket.draw.task_ids))[0]
        n_scored = len(null[0].rstrip(NOT_RUN)) if null else 0
        entries = []
        for entry in ticket.entries:
            report = reports[entry.queue_pos]
            if isinstance(report, LoRAArtifact) and stopped and n_scored < first_look:
                report = EntryResult(entry.queue_pos, NOT_SCORED, reason=f"the round's time limit ended the scoring after {n_scored} tasks, before the first look of {first_look}")
            elif isinstance(report, LoRAArtifact):
                scheduler = getattr(self.evaluator, "_solo_scheduler", None) if hasattr(self, "evaluator") else None
                tested = getattr(self, "_tested", scheduler.tested if scheduler is not None else {})
                proof = tested.get(entry.queue_pos) if params.eval_merged else None
                artifact = report
                report = EntryResult(
                    entry.queue_pos, SCORED, lora=artifact, outcomes=scored[entry.queue_pos], eliminated_at_look=dropped.get(entry.queue_pos),
                    tested_base_hash=proof.base_hash if proof else "", tested_fold_kernel=proof.kernel if proof else "",
                    **{key: (kl_metrics or {}).get(entry.queue_pos, {}).get(key) for key in KL_FIELDS if key in (kl_metrics or {}).get(entry.queue_pos, {})},
                )
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
            phase_s={"training": train_done - started, "evaluation": (finished if eval_done is None else eval_done) - train_done} | ({"kl": max(0, int(kl_wall_s))} if ticket.params.min_kl_seq > 0 else {}),
            **{f"null_{key}": value for key, value in (null_kl or {}).items()},
            parent_outcomes=parent_outcomes if ticket.parent is not None else (),
            parent_error=parent_error if ticket.parent is not None else "",
        )
