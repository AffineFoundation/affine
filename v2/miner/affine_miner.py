"""Miner tools for Affine SN120 v2: see your bounds, check a file, upload it, reproduce a round at home.

    python v2/miner/affine_miner.py bounds --mining-json mining.json
    python v2/miner/affine_miner.py check --mining-json mining.json --hotkey <ss58> batch.json
    AFFINE_PUT_URL=<link> python v2/miner/affine_miner.py upload --mining-json mining.json --hotkey <ss58> batch.json
    python v2/miner/affine_miner.py reproduce --mining-json mining.json --hotkey <ss58> \
        --irt-snapshot irt.json --round-seed <seed> --backend <module>:<function> batch.json

``--backend sn120_v2.replicate:backend`` is the code of the validator's box. It runs as it is only on a machine
with the pinned install and the GPU count of the params. On any other machine use
``python -m sn120_v2.replicate run``: see ``v2/REPLICATE.md``.

Every rule and every number comes from mining.json and from the validator's
own code in ``sn120_v2``: the same batch check, the same box runner on the
``Trainer`` and ``Evaluator`` seams, the same judging. This file holds no
rule of its own.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from sn120_v2 import judging, layout
from sn120_v2.admission import Batch, Grader, HFTokens, Invalid, Tokens, check_batch, env_grader
from sn120_v2.canonical import digest, to_jsonable
from sn120_v2.envs.math_pool import ENV_ID, load_pool
from sn120_v2.errors import ConfigError, ContractError, InfraError, V2Error
from sn120_v2.irt import draw_from_snapshot
from sn120_v2.params import Params
from sn120_v2.publisher import UPLOAD_HEADERS, miner_settable
from sn120_v2.seams import Evaluator, ObjectStore, RoundRunner, Trainer
from sn120_v2.types import NOT_RUN, NULL_MODEL, BoxResult, Draw, Entry, EvalChunk, EvalRequest, FrozenBatch, HeadManifest, HeadVersion, Registration, RoundTicket, check_outcomes
from sn120_v2.worker import LocalRunner

ROUND_ID, LEASE_ID, BOX_ID = "local", "local-l0", "local-box"
PUT_URL_ENV = "AFFINE_PUT_URL"  # the opened upload link; it is a secret, so it is not a command-line argument
# The public site answers 403 to the default client name of urllib on its files under public/. So every request of this tool names itself.
CLIENT = {"User-Agent": "sn120-v2-miner/1"}
LOCAL_KL_COMMAND = "python -m sn120_v2.replicate run"
LOCAL_KL_HELP = "python -m sn120_v2.replicate run --mining-json mining.json --hotkey <ss58> --work <directory> --gpus <N> batch.json"
NULL_CACHE_SCHEMA = "affine-miner-null-cache-v1"
NULL_CACHE_MANIFEST = "manifest.json"


def kl_guidance(params: Params) -> str:
    """The file check and the model-change measurement are separate gates."""
    floors = []
    if params.min_kl_seq > 0:
        floors.append(f"params.min_kl_seq = {params.min_kl_seq:g} nats/sequence")
    if params.min_kl_tok > 0:
        floors.append(f"params.min_kl_tok = {params.min_kl_tok:g} nats/token")
    enabled = "Enabled contract floor(s): " + " and ".join(floors) + "." if floors else "Both KL floors are off now."
    return (
        "WARNING: check does NOT train and does NOT measure kl_seq or kl_tok. A passing file check is NOT a passing KL gate. "
        f"{enabled} Floors come from the contract, not a hard-coded value. "
        f"Measure the trained merged candidate and run the official KL gate locally with `{LOCAL_KL_COMMAND}`; "
        f"full command: `{LOCAL_KL_HELP}` (no separate `replicate kl` command is needed)."
    )


def train_cost_guidance(params: Params) -> str:
    """The published cap is total trained tokens multiplied by steps, never minibatches."""
    cap, bound = params.max_train_token_steps, params.recipe.bound("steps")
    text = (
        "train_cost: max_train_token_steps = steps × tokens; tokens is the sum of trained tokens "
        "(prompt + output + eos) over the kept training rows. Minibatches do NOT multiply this cost."
    )
    if cap > 0:
        text += f" The current cap is {cap:,}. With steps=16, the token budget is {cap // 16:,} (integer floor)."
    if bound is not None:
        text += f" Your allowed steps range is {bound.low:g}–{bound.high:g}."
    return text


def load_contract(source: str) -> tuple[Params, HeadVersion]:
    """Params and head from mining.json: a file or an http(s) URL. There is no built-in default."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(urllib.request.Request(source, headers=CLIENT), timeout=30) as reply:
            raw = reply.read()
    else:
        raw = Path(source).read_bytes()
    doc = json.loads(raw)
    return Params.from_public_dict(doc["params"]), HeadManifest.from_dict(doc["head"]).head


def bounds_text(params: Params) -> str:
    """Every field a miner may set under ``train``, read from ``Recipe.miner_bounds``."""
    lines = [f'Fields you may set under "train" (recipe_id {params.recipe_id}):',
             "decided means the bound is frozen into recipe_id, NOT that its value is locked. You may still set each value within its published range.",
             train_cost_guidance(params)]
    for row in miner_settable(params.recipe):
        status = " [decided: bound frozen into recipe_id; value is yours to set in range]" if row["decided"] else " [placeholder: bound can change; value is yours to set in range]"
        lines.append(f"  {row['key']}: default {json.dumps(row['default'])}; allowed: {row['allowed']}{status}")
    return "\n".join(lines)


def applied_train(params: Params, train: Mapping[str, Any]) -> dict[str, Any]:
    """The values the validator trains with for this ``train`` object. Raises ``BatchRejected``; never clamps."""
    recipe = params.recipe.for_miner(train)
    return {bound.key: to_jsonable(getattr(recipe, bound.key)) for bound in params.recipe.miner_bounds}


def prompt_guidance(batch: Batch, params: Params) -> list[str]:
    """Compare only this file's stored prompts with the bundled public pool."""
    if params.env != ENV_ID:
        return []
    try:
        pool = load_pool()
    except (OSError, ValueError, V2Error):
        return []  # An unavailable local pool adds no warning and never rejects the file.
    mismatched = sum(
        1 for prompt, _ in batch.groups
        if (task := pool.task_for_prompt(prompt)) is not None and prompt != task.batch_prompt()
    )
    if not mismatched:
        return []
    noun = "prompt differs" if mismatched == 1 else "prompts differ"
    return [
        f"WARNING: {mismatched} stored {noun} from the official pool prompt text `task.batch_prompt()` "
        '(the official instruction, a blank line, then the problem, in the one stored `prompt` field). '
        "Validator training uses exactly the stored `prompt` field; a separate sampler system message is not kept. "
        "This is guidance only: the file stays valid."
    ]


def check(raw: bytes, hotkey: str, params: Params, tokens: Tokens | None = None, grader: Grader | None = None) -> dict[str, Any]:
    """The validator's check of one file, as a report. Raises ``Invalid`` when the file breaks a rule."""
    return check_report(check_batch(raw, hotkey, params, tokens=tokens, grader=grader), params)


def check_report(batch: Batch, params: Params) -> dict[str, Any]:
    warning = kl_guidance(params)
    return {
        "valid": True,
        "format": batch.format,
        "batch_sha256": batch.sha256,
        "n_prompts": batch.n_prompts,
        "n_rollouts": batch.n_rollouts,
        "dropped_groups": batch.dropped_groups,
        "rules_run": list(batch.checked),
        "rules_not_run": [family for family in ("tokens", "grader") if family not in batch.checked],
        "train_applied": applied_train(params, batch.train),
        "train_cost_guidance": train_cost_guidance(params),
        "warnings": [warning, *prompt_guidance(batch, params)],
        "kl_gate": {"trained": False, "kl_seq_measured": False, "kl_tok_measured": False, "passing_file_check_is_passing_kl_gate": False, "enabled": params.kl_floor_on, "min_kl_seq": params.min_kl_seq, "min_kl_tok": params.min_kl_tok, "command": LOCAL_KL_COMMAND, "warning": warning},
    }


def reproduce(
    raw: bytes,
    hotkey: str,
    params: Params,
    head: HeadVersion,
    draw: Draw,
    store: ObjectStore,
    runner: RoundRunner,
    *,
    train: Mapping[str, Any] | None = None,
    tokens: Tokens | None = None,
    grader: Grader | None = None,
) -> dict[str, Any]:
    """Run one batch as a one-entry round: check, then the validator's box code and judging.

    ``train`` replaces the ``train`` object of the file. It has the same
    shape and passes the same gate (``Recipe.for_miner``).
    """
    batch = check_batch(raw, hotkey, params, tokens=tokens, grader=grader)
    train = dict(batch.train if train is None else train)
    report = check_report(batch, params) | {"train_applied": applied_train(params, train)}
    key, now = layout.frozen_batch(ROUND_ID, hotkey), int(time.time())
    store.put(key, raw)
    frozen = FrozenBatch(hotkey, batch.sha256, key, batch.format, batch.n_prompts, batch.n_rollouts, train, batch.stage_recipes())
    entry = Entry(0, Registration(0, hotkey, 0), frozen)
    ticket = RoundTicket(ROUND_ID, LEASE_ID, BOX_ID, head, params, draw, (entry,), now, now + params.lease_timeout_s)
    result = runner.run(ticket)
    mine = result.entries[0]
    if mine.lora is not None:
        report |= {"lora_hash": mine.lora.lora_hash, "train_report": mine.lora.report.to_dict()}
    try:
        decision = judging.decide([(ticket, result)])
    except ContractError as exc:  # for example a worker image that the contract does not pin
        score = judging.score_entry(draw.weights, result.null_outcomes, mine.outcomes, params, n_total=len(draw.task_ids)) if mine.outcomes else None
        return report | {"status": "unofficial", "reason": f"the validator would refuse this run: {exc}", "score": score and score.to_dict()}
    row = decision.entries[0]
    return report | {"status": row.status, "reason": row.reason or decision.reason, "eliminated_at_look": row.eliminated_at_look, "score": row.score and row.score.to_dict()}


class NullCache:
    """A fail-closed, miner-local cache of null outcomes on one exact draw."""

    def __init__(self, directory: str, head: HeadVersion, params: Params, draw: Draw) -> None:
        self.directory = Path(directory).expanduser()
        # Public params omits legacy default fields. Include the materialized
        # scoring values too: null_samples_per_task=None means samples_per_task.
        self.manifest = {
            "schema": NULL_CACHE_SCHEMA,
            "head": head.to_dict(),
            "params": to_jsonable(params) | {
                "params_hash": params.params_hash,
                "null_samples_per_task": params.null_samples_per_task,
                "null_samples": params.null_samples,
            },
            "draw": draw.to_dict(),
        }
        self.identity = digest(self.manifest)
        self.manifest["cache_id"] = self.identity
        self.task_positions = {task: index for index, task in enumerate(draw.task_ids)}
        self.outcomes: tuple[str, ...] | None = None
        if not self.directory.exists():
            return
        if not self.directory.is_dir():
            raise ConfigError(f"--null-cache is not a directory: {self.directory}")
        manifest_path = self.directory / NULL_CACHE_MANIFEST
        try:
            entries = list(self.directory.iterdir())
            saved = json.loads(manifest_path.read_text())
        except FileNotFoundError:
            saved = None
        except (OSError, ValueError, TypeError) as exc:
            raise ConfigError(f"cannot read --null-cache {self.directory}: {type(exc).__name__}") from None
        if entries and saved is None:
            raise ConfigError(f"--null-cache {self.directory} is nonempty but has no {NULL_CACHE_MANIFEST}; use an empty directory or a new one")
        if not entries:
            return
        if not isinstance(saved, dict):
            raise ConfigError(f"--null-cache {self.directory} has an invalid manifest")
        differences = sorted(key for key in self.manifest.keys() | saved.keys() if key not in ("cache_id", "null_outcomes") and saved.get(key) != self.manifest.get(key))
        if differences or saved.get("cache_id") != self.identity:
            detail = ", ".join(differences) or "cache_id"
            raise ConfigError(f"--null-cache {self.directory} belongs to another draw, head, or scoring configuration ({detail}); use a different directory")
        if "null_outcomes" not in saved:
            raise ConfigError(f"--null-cache {self.directory} is missing its null outcomes; use an empty directory or a new one")
        try:
            outcomes = tuple(saved["null_outcomes"])
            self.validate(outcomes)
        except (KeyError, TypeError, ValueError, ContractError) as exc:
            raise ConfigError(f"--null-cache {self.directory} has invalid null outcomes: {exc}") from None
        self.outcomes = outcomes

    def validate(self, outcomes: Sequence[str]) -> tuple[str, ...]:
        outcomes = tuple(outcomes)
        check_outcomes(outcomes, len(self.manifest["draw"]["task_ids"]), "null M")
        if len(outcomes) != self.manifest["params"]["null_samples"]:
            raise ValueError("wrong number of null samples")
        if any(NOT_RUN in text for text in outcomes):
            raise ValueError("null cache fill did not score the whole draw; no cache was written")
        return outcomes

    def save(self, outcomes: Sequence[str]) -> None:
        """Publish a complete fill atomically; never overwrite a filled cache."""
        if self.outcomes is not None:
            return
        try:
            outcomes = self.validate(outcomes)
        except (TypeError, ValueError, ContractError) as exc:
            raise InfraError(str(exc)) from None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            document = dict(self.manifest, null_outcomes=list(outcomes))
            encoded = (json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".manifest.", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # An atomic no-replace publish prevents two candidates from
                # mixing results if they are accidentally run concurrently.
                os.link(temporary, self.directory / NULL_CACHE_MANIFEST)
                self.outcomes = outcomes
            except FileExistsError:
                raise ConfigError(f"--null-cache {self.directory} was filled by another run; use it again to verify and reuse that cache") from None
            finally:
                temporary.unlink(missing_ok=True)
        except OSError as exc:
            raise InfraError(f"could not write --null-cache {self.directory}: {type(exc).__name__}") from None


class CachedNullEvaluator:
    """A local-only decorator; candidate generation and grading stay on the backend."""

    def __init__(self, evaluator: Evaluator, cache: NullCache, *, clock: Callable[[], int] | None = None) -> None:
        self.evaluator, self.cache = evaluator, cache
        self.clock = clock or (lambda: int(time.time()))
        self.complete: tuple[str, ...] | None = cache.outcomes
        self._observed: list[dict[int, str]] | None = None

    def __getattr__(self, name: str) -> Any:
        # Block backends use attributes such as the image digest. Do not
        # accidentally make a LocalRunner choose the separate solo protocol.
        if name == "solo_worker":
            raise AttributeError(name)
        return getattr(self.evaluator, name)

    def _positions(self, request: EvalRequest) -> list[int]:
        if (
            request.head.to_dict() != self.cache.manifest["head"]
            or request.round_seed != self.cache.manifest["draw"]["round_seed"]
            or request.params.params_hash != self.cache.manifest["params"]["params_hash"]
        ):
            raise ContractError("the cache evaluator received another head, draw, or params")
        try:
            return [self.cache.task_positions[task] for task in request.task_ids]
        except KeyError:
            raise ContractError("the cache evaluator received a task outside its draw") from None

    @property
    def manifest_head(self) -> str:
        return self.cache.manifest["head"]["version_id"]

    @manifest_head.setter
    def manifest_head(self, value: str) -> None:
        raise AttributeError("manifest_head is read-only")

    def _remember(self, request: EvalRequest, model: str, sample: int, index: int, code: str) -> None:
        """Accept each ordered null result once; reject a duplicate disagreement."""
        if model != NULL_MODEL or self.complete is not None:
            return
        if self._observed is None:
            self._observed = [{} for _ in range(request.params.null_samples)]
        if not 0 <= sample < len(self._observed):
            raise ContractError("null cache received an invalid sample index")
        positions = self._positions(request)
        if not 0 <= index < len(positions):
            raise ContractError("null cache received a task outside the request")
        position = positions[index]
        rows = self._observed[sample]
        if position in rows:
            if rows[position] != code:
                raise ContractError("a retried null rollout disagrees with the cached observation")
            return
        rows[position] = code
        n_tasks = len(self.cache.task_positions)
        if all(len(rows) == n_tasks for rows in self._observed):
            self.complete = tuple("".join(rows[index] for index in range(n_tasks)) for rows in self._observed)

    def _cached_slice(self, request: EvalRequest) -> tuple[str, ...]:
        if self.cache.outcomes is None:
            raise ContractError("cached null is not available")
        positions = self._positions(request)
        return tuple("".join(text[index] for index in positions) for text in self.cache.outcomes)

    def evaluate(self, request: EvalRequest) -> EvalChunk:
        if self.cache.outcomes is not None:
            return self._evaluate_candidates(request)
        self._positions(request)
        chunk = self.evaluator.evaluate(request)
        for sample in range(request.params.null_samples):
            for index, code in enumerate(chunk.outcomes[NULL_MODEL][sample]):
                self._remember(request, NULL_MODEL, sample, index, code)
        return chunk

    def _evaluate_candidates(self, request: EvalRequest) -> EvalChunk:
        """The standard block evaluation, with its null jobs left out."""
        from sn120_v2.gpu.evaluator import _Deadline

        evaluator = self.evaluator
        if not all(hasattr(evaluator, name) for name in ("_prepare", "_rollout", "_grade", "clients_for", "_urls", "_servers", "_cut", "config")):
            raise ConfigError("--null-cache reuse needs the standard evaluator backend")
        started = time.monotonic()
        _, table, clients = evaluator._prepare(request)
        written: list[tuple[int, float]] = []
        models = [lora.lora_hash for lora in request.loras]
        jobs = [(model, sample, index) for model in models for sample in range(request.params.samples_per_task) for index in range(len(request.task_ids))]
        served, deadline_hit = time.monotonic(), False

        def rollout(job: tuple[str, int, int]):
            nonlocal deadline_hit
            try:
                return evaluator._rollout(request, evaluator._urls[job[2] % len(evaluator._urls)], job, written)
            except _Deadline:
                deadline_hit = True
                return None

        with ThreadPoolExecutor(clients) as threads:
            replies = list(threads.map(rollout, jobs))
        if not all(server.alive() for server in evaluator._servers):
            evaluator.close()
            raise InfraError("a vLLM server died during the look; nothing of it can be scored")
        if deadline_hit:
            raise evaluator._cut(request, started, served, [NULL_MODEL, *models], table, written, len(jobs), sum(reply is not None for reply in replies))
        codes = evaluator._grade(request.task_ids, jobs, replies, request)
        outcomes = {model: tuple("".join(codes[model, sample, index] for index in range(len(request.task_ids))) for sample in range(request.params.samples_per_task)) for model in models}
        outcomes[NULL_MODEL] = self._cached_slice(request)
        return EvalChunk(request.look, request.task_ids, outcomes, evaluator.config.image_digest)

    def _stream_blocks(self, request: EvalRequest, results: Callable[[str, int, int, str], None], eliminated: Callable[[str], bool], done: Callable[[], bool]) -> int:
        """Compatibility path for simulators; every request is one bounded block."""
        start = 0
        for look, stop in enumerate(request.params.look_sizes(len(request.task_ids))):
            while start < stop:
                if (self.cache.outcomes is not None and done()) or (request.deadline and self.clock() >= request.deadline):
                    return start
                end = min(stop, start + request.params.eval_chunk_tasks)
                # A block-only fake has no way to request null by itself. On a
                # fill, keep its ordinary requests and suppress dropped results;
                # the GPU backend instead uses its native streaming protocol.
                loras = request.loras if self.cache.outcomes is None else tuple(
                    lora for lora in request.loras if not eliminated(lora.lora_hash)
                )
                chunk_request = dataclasses.replace(request, task_ids=request.task_ids[start:end], look=look, loras=loras)
                chunk = self.evaluate(chunk_request)
                for model, strings in chunk.outcomes.items():
                    if eliminated(model):
                        continue
                    for sample, text in enumerate(strings):
                        for index, code in enumerate(text):
                            results(model, sample, start + index, code)
                start = end
        return start

    def _stream_native(self, request: EvalRequest, results: Callable[[str, int, int, str], None], eliminated: Callable[[str], bool], done: Callable[[], bool]) -> int:
        """Use the native streaming protocol, but never stop the null early on a fill."""
        stream = getattr(self.evaluator, "stream", None)
        if stream is None:
            return self._stream_blocks(request, results, eliminated, done)

        def result(model: str, sample: int, index: int, code: str) -> None:
            self._remember(request, model, sample, index, code)
            results(model, sample, index, code)

        # Candidate elimination still happens normally. Only the null's early
        # stop is disabled: its complete observation is the reusable artifact.
        return stream(request, result, eliminated, lambda: False)

    def stream(self, request: EvalRequest, results: Callable[[str, int, int, str], None], eliminated: Callable[[str], bool], done: Callable[[], bool]) -> int:
        stream = getattr(self.evaluator, "stream", None)
        if self.cache.outcomes is not None or stream is None:
            return self._stream_blocks(request, results, eliminated, done)
        return self._stream_native(request, results, eliminated, done)

    def close(self) -> None:
        self.evaluator.close()


class CachedNullRunner:
    """Delegate training, timing and judging to the existing runner."""

    def __init__(self, runner: LocalRunner, cache: NullCache) -> None:
        if type(runner) is not LocalRunner:
            raise ConfigError("--null-cache needs the standard local or solo backend")
        runner.evaluator = CachedNullEvaluator(runner.evaluator, cache, clock=runner.clock)
        self.runner, self.cache = runner, cache

    def run(self, ticket: RoundTicket) -> BoxResult:
        result = self.runner.run(ticket)
        return self._finish(result)

    def _finish(self, result: BoxResult) -> BoxResult:
        complete = self.runner.evaluator.complete
        if self.cache.outcomes is None:
            if complete is not None and not result.stopped_at_limit and any(text.rstrip(NOT_RUN) for text in result.null_outcomes):
                self.cache.save(complete)
        elif complete != self.cache.outcomes:
            raise ContractError("the cached null was not replayed in full")
        if self.cache.outcomes is not None:
            for actual, expected in zip(result.null_outcomes, self.cache.outcomes, strict=True):
                prefix = actual.rstrip(NOT_RUN)
                if actual != prefix + NOT_RUN * (len(expected) - len(prefix)) or prefix != expected[:len(prefix)]:
                    raise ContractError("the result did not use the cached null outcomes")
        # Do not compare ordinary results: a candidate that drops early makes
        # the runner deliberately retain only the null prefix through that
        # look. The separate complete capture is what makes later candidates
        # comparable, while their own result still follows the ordinary rule.
        return result


class CachedSoloNullEvaluator(CachedNullEvaluator):
    """Intercept the real solo worker factory, before its first work claim."""

    def __setattr__(self, name: str, value: Any) -> None:
        # SoloRunner installs its round scheduler and client budget on the
        # evaluator. Workers must observe that state on the real backend.
        if name in ("_solo_scheduler", "_solo_clients"):
            setattr(self.evaluator, name, value)
        else:
            super().__setattr__(name, value)

    def _scheduler(self):
        scheduler = getattr(self.evaluator, "_solo_scheduler", None)
        if scheduler is None:
            raise ConfigError("--null-cache needs the standard solo scheduler")
        return scheduler

    def _begin_round(self, ticket: RoundTicket, scheduler) -> None:
        """Gate early drops for a fill, or install cached null exactly once."""
        with scheduler.lock:
            if getattr(scheduler, "_null_cache_installed", False):
                return
            scheduler._null_cache_installed = True
            if self.cache.outcomes is not None:
                for sample, text in enumerate(self.cache.outcomes):
                    for index, code in enumerate(text):
                        scheduler.result(NULL_MODEL, sample, index, code)
                # An exhausted null counter is invisible to every later claim.
                scheduler.remaining[NULL_MODEL] = len(ticket.draw.task_ids)
                return
            judge = scheduler.judge

            def judge_after_null(pos: int, look: int) -> None:
                if all(len(text) == len(ticket.draw.task_ids) for text in scheduler.null):
                    judge(pos, look)

            scheduler.judge = judge_after_null

    def solo_worker(self, ticket: RoundTicket, trained: dict, deadline: int, gpu: int):
        scheduler = self._scheduler()
        self._begin_round(ticket, scheduler)
        worker = self.evaluator.solo_worker(ticket, trained, deadline, gpu)
        if self.cache.outcomes is not None:
            return worker
        request = EvalRequest(
            ticket.round_id, ticket.head, tuple(trained.values()), ticket.draw.task_ids,
            ticket.draw.round_seed, 0, ticket.params, deadline, len(ticket.entries),
        )
        run = worker.run

        def capturing_run(*, steal, result, judge, alive, deadline):
            def capture(model: str, sample: int, index: int, code: str) -> None:
                self._remember(request, model, sample, index, code)
                result(model, sample, index, code)

            return run(steal=steal, result=capture, judge=judge, alive=alive, deadline=deadline)

        worker.run = capturing_run
        return worker


class CachedSoloNullRunner(CachedNullRunner):
    """Reuse null through the existing solo scheduler, including merged mode."""

    def __init__(self, runner: RoundRunner, cache: NullCache) -> None:
        from sn120_v2.gpu.solo import SoloRunner

        if type(runner) is not SoloRunner:
            raise ConfigError("--null-cache merged reuse needs the standard SoloRunner backend")
        runner.evaluator = CachedSoloNullEvaluator(runner.evaluator, cache, clock=runner.clock)
        self.runner, self.cache = runner, cache

    def run(self, ticket: RoundTicket) -> BoxResult:
        result = self.runner.run(ticket)
        scheduler = self.runner.evaluator._scheduler()
        complete = self.runner.evaluator.complete
        if self.cache.outcomes is None:
            # The scheduler may truncate an early-dropped candidate's null.
            # The worker capture is the separate full-draw observation.
            if complete is not None and not result.stopped_at_limit and any(text.rstrip(NOT_RUN) for text in result.null_outcomes):
                self.cache.save(complete)
        elif not result.stopped_at_limit and any(text != expected[:len(text)] for text, expected in zip(scheduler.null, self.cache.outcomes, strict=True)):
            raise ContractError("the cached null was not replayed in full")
        return self._finish(result)


def local_runner(trainer: Trainer, evaluator: Evaluator, clock: Callable[[], int], image: str, params: Params) -> RoundRunner:
    """The existing runner for this contract; the no-cache CLI path is unchanged."""
    if params.eval_merged:
        from sn120_v2.gpu.solo import SoloRunner

        return SoloRunner(trainer, evaluator, clock, image)
    return LocalRunner(trainer, evaluator, clock, image)


def cached_null_runner(runner: RoundRunner, directory: str | None, head: HeadVersion, params: Params, draw: Draw) -> RoundRunner:
    """The default is the existing local execution path."""
    if directory is None:
        return runner
    cache = NullCache(directory, head, params, draw)
    return CachedSoloNullRunner(runner, cache) if params.eval_merged else CachedNullRunner(runner, cache)


def upload(raw: bytes, url: str) -> None:
    """PUT the file to the presigned link, with the headers the link is signed for. The URL is a secret: it is never printed."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=raw, method="PUT", headers=UPLOAD_HEADERS | CLIENT), timeout=300):
            pass
    except OSError as exc:
        raise InfraError(f"the upload failed: {type(exc).__name__} {getattr(exc, 'code', '')}".strip()) from None


def resolve_tokenizer(path: str, base_hash: str) -> str:
    """Resolve ``<work>/cache`` to this head's verified inner directory when given."""
    place = Path(path).expanduser()
    candidate = place / base_hash
    if place.is_dir() and place.name == "cache" and candidate.is_dir():
        resolved = candidate.resolve()
        print(f"tokenizer: resolved cache parent {place.resolve()} to head {resolved}", flush=True)
        return str(resolved)
    return path


def load_tokens(path: str | None, base_hash: str | None = None) -> Tokens | None:
    """The head's tokenizer, or None. ``transformers`` is needed only when a tokenizer is given."""
    if path and base_hash:
        path = resolve_tokenizer(path, base_hash)
    return HFTokens(importlib.import_module("transformers").AutoTokenizer.from_pretrained(path)) if path else None


def official_grader(params: Params) -> Grader:
    """The validator's grader of the environment, built with the validator's own call. Needs the ``envs`` extra."""
    if params.env != ENV_ID:
        raise ConfigError(f"this tool has no grader for environment {params.env!r}")
    return env_grader(load_pool(), importlib.import_module("sn120_v2.envs.math_grader").grade)


def load_backend(spec: str, workdir: str) -> tuple[ObjectStore, Trainer, Evaluator, str]:
    """``module:function``. The function takes a work directory and returns (store, trainer, evaluator, image digest)."""
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)(workdir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("bounds", "check", "upload", "reproduce"):
        command = commands.add_parser(name)
        if name == "check":
            command.epilog = "WARNING: this does NOT train or measure kl_seq or kl_tok; a passing file check is NOT a passing KL gate. Enabled floors come from params.min_kl_seq and params.min_kl_tok. Measure them with `python -m sn120_v2.replicate run`. max_train_token_steps = steps × sum(prompt + output + eos over kept training rows); minibatches do NOT multiply it. Live worked example: cap 22,130,000 / steps=16 = 1,383,125 tokens; official check --tokenizer rejects a mixed batch above that budget."
        command.add_argument("--mining-json", required=True, help="mining.json: a file or an http(s) URL")
        if name == "bounds":
            continue
        command.add_argument("batch", help="your batch file")
        command.add_argument("--hotkey", required=True, help="the ss58 address of your hotkey")
        command.add_argument("--tokenizer", help="head tokenizer directory, <work>/cache from fetch, or model id; without it the token rules do not run")
        command.add_argument("--grade", action="store_true", help="match your rewards with the official grader (needs --tokenizer and the envs extra)")
        if name == "reproduce":
            command.add_argument("--irt-snapshot", required=True, help="the published IRT snapshot of the round (a JSON file)")
            command.add_argument("--round-seed", required=True, help="the public seed of the round")
            command.add_argument("--tasks", type=int, help="draw N tasks (default: n_tasks of the contract)")
            command.add_argument("--train", help='a JSON object that replaces "train" of the file')
            command.add_argument("--backend", required=True, help="module:function that returns (store, trainer, evaluator, image digest)")
            command.add_argument("--null-cache", help="reuse full null M outcomes for this exact draw, head and scoring params; a mismatch is refused")
            command.add_argument("--work", help="work directory (default: a temporary directory)")
    args = parser.parse_args(argv)
    try:
        params, head = load_contract(args.mining_json)
        if args.command == "bounds":
            print(bounds_text(params))
            return 0
        if args.grade and not args.tokenizer:  # with no token count no row counts as capped: not the validator's verdict
            raise ConfigError("--grade needs --tokenizer")
        if args.command == "upload" and not os.environ.get(PUT_URL_ENV):
            raise ConfigError(f"set {PUT_URL_ENV} to the upload URL of your sealed link")
        raw, tokens = Path(args.batch).read_bytes(), load_tokens(args.tokenizer, head.base_hash)
        grader = official_grader(params) if args.grade else None
        if args.command in ("check", "upload"):
            report = check(raw, args.hotkey, params, tokens, grader)
            if args.command == "upload":  # an invalid file is never sent
                upload(raw, os.environ[PUT_URL_ENV])
                report["uploaded"] = True
        else:
            print(bounds_text(params), file=sys.stderr)
            store, trainer, evaluator, image = load_backend(args.backend, args.work or tempfile.mkdtemp(prefix="affine-reproduce-"))
            runner = LocalRunner(trainer, evaluator, lambda: int(time.time()), image)
            train = json.loads(args.train) if args.train else None
            snapshot = json.loads(Path(args.irt_snapshot).read_bytes())
            draw = draw_from_snapshot(snapshot, args.round_seed, args.tasks or params.n_tasks)
            if args.null_cache is not None:
                runner = local_runner(trainer, evaluator, lambda: int(time.time()), image, params)
                runner = cached_null_runner(runner, args.null_cache, head, params, draw)
            report = reproduce(raw, args.hotkey, params, head, draw, store, runner, train=train, tokens=tokens, grader=grader)
    except Invalid as exc:
        print(json.dumps({"valid": False, "rule": exc.rule, "reason": str(exc)}, indent=1))
        return 1
    except V2Error as exc:
        print(json.dumps({"error": type(exc).__name__, "reason": str(exc)}, indent=1))
        return 2
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
