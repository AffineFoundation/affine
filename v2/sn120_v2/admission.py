"""Admission: the strict check of a miner's batch file, the live verdicts and the freeze at round open.

A batch file is one JSON object::

    {"version": "affine-batch-v1" | "affine-batch-v2" | "affine-batch-v3",
     "hotkey": "<the miner's ss58 hotkey>",
     "train": {...},                      # optional, never on v1: only keys in Recipe.miner_bounds
     "groups": [{"prompt": "<text>", "rollouts": [{"output": "<text>", "reward": <number>}, ...]}, ...]}

``check_batch`` applies every rule in a fixed order and stops at the first one
that fails. Nothing is ever repaired or clamped: one bad value makes the whole
file invalid. Every limit comes from ``Params``.

This module uses the standard library and plain Python modules of this package
only, so a miner can run the same check on their own machine. The tokenizer
and the official grader are inputs.

One file never stops the check of the others. When a file makes the grader,
the tokenizer or the checker itself fail, the same part is tried on a
known-good case. It passes: only that file is invalid, for this call. It
fails: the failure is ours, and ``InfraError`` is raised. Such a file is
checked again at a later call, ``FAULT_HOLD_S`` after its last check at the
earliest. After ``FAULT_TRIES`` failed checks in a row, each one with that
proof and at the same place, the verdict stays.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

from . import layout
from .canonical import sha256_hex
from .errors import BatchRejected, InfraError, VerifyError
from .gpu.batch import advantages
from .params import Params, Recipe
from .seams import ObjectStore
from .types import FrozenBatch, Registration, Verdict

FORMAT_V1 = "affine-batch-v1"
FORMATS = (FORMAT_V1, "affine-batch-v2", "affine-batch-v3")  # v2 and v3 may carry "train"
CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
_TOP_KEYS = frozenset(("version", "hotkey", "groups"))
_BOXED = "\\boxed"

Group = tuple[str, tuple[tuple[str, float], ...]]  # (prompt, ((output, reward), ...))
Sizes = dict[tuple[int, int], tuple[int, int]]  # (group, rollout) place -> (prompt tokens, output tokens)

# The official grade of one rollout: (prompt, output, capped) -> reward.
# The checker sets capped: True when the output is exactly max_output_tokens long, so it stopped at the cap.
# None means the prompt is not a task of the environment, so there is no official grade.
# It raises GraderTimeLimit when the grade did not finish in time, and any other exception when grading the row
# made it fail. A grader may also have three optional methods, as EnvGrader has:
#   many(rows) -> the outcome of each (prompt, output, capped) row, in order: the reward, None, or the exception
#   probe()    -> raises InfraError unless the grader works now
#   cases()    -> known-good (prompt, right answer) pairs, for the known-good file of the checker
Grader = Callable[[str, str, bool], float | None]
TIME_LIMIT = "grader_time_limit"  # the rule code of a row that the official grader cannot grade in time
FAULT = "grader_fault"  # the rule code of a row that makes the official grader fail, while the grader itself works
NO_SIGNAL = "no_signal"  # the rule code of a file in which no rollout can change the model
TRAIN_COST = "train_cost"  # the rule code of a file that costs more training than one seat may use
TOKENIZER_FAULT = "tokenizer_fault"  # the rule code of a text that makes the tokenizer fail, while the tokenizer itself works
CHECK_FAULT = "check_fault"  # the rule code of a file that makes the checker itself fail, while the checker still passes a known-good file
# These verdicts depend on this machine at this moment. Each holds for FAULT_HOLD_S, then the file is checked
# again at the next call, until FAULT_TRIES is reached.
TRANSIENT = (TIME_LIMIT, FAULT, TOKENIZER_FAULT, CHECK_FAULT)
# After this many failed checks in a row of the same bytes with the same params, the verdict stays like any other:
# until the file changes, the params change or the process starts again. A failed check counts only when it is
# proven to be the fault of the file (``Invalid.proven``), and only in a row of checks that fail at the same place.
FAULT_TRIES = 3
# Exceptions that tell of the state of this machine (no memory, no file handle, a dead pipe), not of the file.
# A failed check with such a cause is never counted toward FAULT_TRIES.
MACHINE = (MemoryError, OSError)
# The seconds between two checks of the same bytes with the same params, after a verdict in TRANSIENT, counted or
# not. So FAULT_TRIES counted checks need a fault that lasts (FAULT_TRIES - 1) * FAULT_HOLD_S, not one short stall.
FAULT_HOLD_S = 600
_AGAIN = f"checked again at a try to open a round, at most once each {FAULT_HOLD_S // 60} minutes; after {FAULT_TRIES} failed checks in a row the verdict stays until a new file is uploaded"
# The env's Grade.machine_cause for an expired time limit (envs.math_grader.TIME_LIMIT). This module cannot
# import the env, which needs extra packages. If the env renames the value, a time-out becomes a grader fault.
ENV_TIME_LIMIT = "time-limit"
PROBE_TEXT = "What is 1 + 1?"  # the tokenizer works when it counts at least one token in this text, as a prompt and as an output
PROBE_TASKS = 4  # the grader works when it gives reward 1 to one of the first pool tasks answered with its own answer
PROBE_HOTKEY = "known-good"  # the owner named in the known-good file of the checker
# The characters of output text that the checker hands to the tokenizer in one call (``output_tokens_many``), when
# the tokenizer can count many texts at once. One call holds the tokens of these texts in memory, so this is the
# bound of that memory: the check of one file never holds more than this and one more text.
AHEAD_CHARS = 4_000_000

# Every rule code, in the order the rules run, with its meaning. llms.txt prints this table.
RULES: dict[str, str] = {
    "size": "the file is empty or larger than the byte cap",
    "json": "the file is not UTF-8 JSON",
    "nonfinite": "the file holds NaN or Infinity",
    "top_keys": "the top level is not an object with exactly version, hotkey, groups and (v2, v3) the optional train",
    "version": "version is not a known format",
    "train": "train is not an object, has a key that is not settable, or has a value outside its published bound",
    "hotkey": "hotkey is not the owner of the stored object",
    "groups": "groups is not a non-empty list",
    "group_keys": "a group does not have exactly the keys prompt, rollouts",
    "prompt": "a prompt is not non-empty text",
    "duplicate_prompt": "two groups have the same prompt",
    "rollouts": "rollouts is not a non-empty list",
    "rollout_keys": "a rollout does not have exactly the keys output, reward",
    "output": "an output is not text",
    "reward": "a reward is not a finite number, or the rewards of one prompt are too large to take their mean",
    "cap": "the first group alone is larger than the rollout cap",
    "text": "a prompt or an output is not valid Unicode text",
    "control_token": "a prompt or an output holds a chat control token",
    TOKENIZER_FAULT: f"a prompt or an output makes the tokenizer fail; {_AGAIN}",
    "output_tokens": "an output is longer than the output token cap",
    "context_tokens": "prompt + output is longer than the context token cap",
    "answer_only": "too many rollouts have almost no reasoning before the first boxed answer",
    "reasoning_share": "too small a share of the output tokens is reasoning",
    "answer_repeat": "one final answer is in too many rollouts",
    NO_SIGNAL: "no rollout has a reward that differs from the mean reward of its training group, so nothing trains",
    TRAIN_COST: "steps times the tokens that the trainer runs through the model is above the training cost cap",
    TIME_LIMIT: f"the official grader cannot grade a rollout inside its time limit; {_AGAIN}",
    FAULT: f"a rollout makes the official grader fail; {_AGAIN}",
    "grader": "a reward is not the official grade",
    "pool": "too small a share of the prompts are tasks of the public pool",
    CHECK_FAULT: f"the file makes the checker itself fail, at any step; {_AGAIN}",
}


class GraderTimeLimit(Exception):
    """Raised by a ``Grader``: the official grade of this row did not finish inside its time limit."""


class GraderFault(Exception):
    """Raised by a ``Grader``: grading this row made the grader fail (its worker process died, or it raised)."""


class EnvGrader:
    """The ``Grader`` of an environment, over its task pool and a function that grades a list of jobs.

    ``grade_many(jobs)`` takes ``(task, output, capped)`` jobs and returns one grade for each, in order. A grade
    has ``reward`` and ``machine_cause``: an expired time limit gives ``GraderTimeLimit`` for that row, and any
    other cause (a dead worker) gives ``GraderFault``. The rows go out ``chunk`` at a time, so a caller that
    stops at the first bad row does not wait for the rest of the file.
    """

    def __init__(self, pool: Any, grade_many: Callable[[Sequence[tuple[Any, str, bool]]], Sequence[Any]], chunk: int) -> None:
        self.pool, self.grade_many, self.chunk = pool, grade_many, max(1, chunk)

    def __call__(self, prompt: str, output: str, capped: bool) -> float | None:
        outcome = next(self.many([(prompt, output, capped)]))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def many(self, rows: Sequence[tuple[str, str, bool]]) -> Iterator[float | None | Exception]:
        for start in range(0, len(rows), self.chunk):
            jobs = [(self.pool.task_for_prompt(prompt), output, capped) for prompt, output, capped in rows[start : start + self.chunk]]
            known = [job for job in jobs if job[0] is not None]
            try:
                grades = list(self.grade_many(known)) if known else []
            except InfraError:
                raise
            except Exception as exc:  # the whole call failed: every row of it has no grade, and the probe says whose fault it is
                grades = [exc] * len(known)
            if len(grades) != len(known):
                raise InfraError(f"the official grader returned {len(grades)} grades for {len(known)} rows")
            answers = iter(grades)
            for task, _, _ in jobs:
                yield None if task is None else self._outcome(next(answers))

    @staticmethod
    def _outcome(grade: Any) -> float | Exception:
        if isinstance(grade, Exception):
            return grade
        if grade.machine_cause == ENV_TIME_LIMIT:
            return GraderTimeLimit()
        if grade.machine_cause is not None:
            return GraderFault(str(grade.machine_cause))
        return float(grade.reward)

    def cases(self) -> list[tuple[str, str]]:
        """Known-good cases: the batch prompt and the reference answer of the first pool tasks."""
        tasks = [self.pool.task(task_id) for task_id in self.pool.task_ids[:PROBE_TASKS]]
        return [(task.batch_prompt(), task.answer) for task in tasks]

    def probe(self) -> bool:
        """Raise ``InfraError`` unless the grader works now: a pool task with its own answer must get reward 1.

        Returns whether every known-good case got its reward. Only then is the grader proven sound enough to
        count a failed row against its file (``FAULT_TRIES``): a pool that loses rows at random loses some of
        these cases too.
        """
        tasks = [self.pool.task(task_id) for task_id in self.pool.task_ids[:PROBE_TASKS]]
        try:
            grades = self.grade_many([(task, _BOXED + "{" + task.answer + "}", False) for task in tasks])
            works = [grade.machine_cause is None and grade.reward == 1.0 for grade in grades]
        except Exception as exc:
            raise InfraError(f"the official grader failed on a known-good case: {type(exc).__name__}") from exc
        if not any(works):
            raise InfraError("the official grader gives no known-good case its reward")
        return len(works) == len(tasks) and all(works)


def env_grader(pool: Any, grade: Callable[..., Any]) -> EnvGrader:
    """The ``Grader`` of an environment, from its task pool and its ``grade(task, output, capped)`` function.

    The grades run one at a time in the calling process. This is the miner's tool. The validator uses ``pool_grader``.
    """
    return EnvGrader(pool, lambda jobs: [grade(*job) for job in jobs], chunk=1)


class LazyPool:
    """A grade pool that starts at the first grade and serves one caller at a time.

    ``start()`` builds the pool, for example ``lambda: GradePool(workers=8, timeout_s=20.0)``. So a command that
    checks no file starts no worker process, and two threads cannot mix their jobs in one pool.
    """

    def __init__(self, start: Callable[[], Any]) -> None:
        self._start, self._pool, self._lock = start, None, threading.Lock()

    def grade_many(self, jobs: Sequence[tuple[Any, str, bool]]) -> Sequence[Any]:
        with self._lock:
            if self._pool is None:
                self._pool = self._start()
            return self._pool.grade_many(jobs)

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()


def pool_grader(pool: Any, grade_pool: Any, chunk: int = 32) -> EnvGrader:
    """The same ``Grader`` over the environment's ``GradePool``: worker processes, each grade under a hard time limit.

    A row that hangs is killed at ``GradePool.timeout_s`` and the rows of a file are graded in parallel. At most
    ``chunk`` rows are sent before the checker looks at the results, so the worst wait for one file that hangs on
    every row is ``chunk / workers * timeout_s``.
    """
    return EnvGrader(pool, grade_pool.grade_many, chunk)


class Tokens(Protocol):
    """Token counts from the head's tokenizer."""

    def prompt_tokens(self, prompt: str) -> int:
        """Tokens of the prompt as one user message in the chat template, with the generation prompt."""
        ...

    def output_tokens(self, text: str) -> int:
        """Tokens of the text alone, with no special token added."""
        ...

    # Optional: ``output_tokens_many(texts: Sequence[str]) -> Sequence[int]`` gives ``output_tokens`` of each text,
    # in order, in one call. A tokenizer that can count many texts at once (on every CPU) has it. Each count must
    # be the count that ``output_tokens`` gives. The checker falls back to ``output_tokens`` when the call fails.


class HFTokens:
    """``Tokens`` over a Hugging Face tokenizer object. The caller loads the tokenizer; nothing is imported here."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def prompt_tokens(self, prompt: str) -> int:
        message = [{"role": "user", "content": prompt}]
        ids = self._tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True)
        return len(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def output_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def output_tokens_many(self, texts: Sequence[str]) -> list[int]:
        """``output_tokens`` of each text. A fast tokenizer counts the texts of one call on every CPU at once.

        This is the call of ``output_tokens`` with a list in place of one text: the same function of the
        tokenizer with the same settings, and it encodes each text of the list apart from the others. So each
        count is the same. No thread is started here: the threads are the tokenizer's own. With
        ``TOKENIZERS_PARALLELISM=false`` in the environment it uses one thread, and nothing is gained.
        """
        texts = list(texts)
        if not texts:  # the tokenizer fails on a call with no text
            return []
        found = self._tokenizer(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        return [len(ids) for ids in found["input_ids"]]


class Invalid(ValueError):
    """The file breaks a published rule.

    ``rule`` is a stable code. ``str()`` is the public reason: it names
    positions, counts and published limits, never batch text or a train value.
    ``proven`` is for a rule in ``TRANSIENT`` only: the part that failed passed its known-good case at that
    moment, and the cause is not in ``MACHINE`` (for the grader, see ``_fault``). Only such a failure counts
    toward ``FAULT_TRIES``.
    """

    def __init__(self, rule: str, reason: str, proven: bool = False) -> None:
        super().__init__(reason)
        self.rule, self.proven = rule, proven


@dataclass(frozen=True)
class Batch:
    """A file that passed. ``groups`` holds the kept groups only: exactly what is trained."""

    sha256: str  # of the exact file bytes
    n_bytes: int
    format: str
    groups: tuple[Group, ...]
    dropped_groups: int  # whole groups past max_prompts / max_rollouts, dropped in file order
    train: dict[str, Any]  # the miner's "train" object as given; apply with Recipe.for_miner
    checked: tuple[str, ...]  # rule families that ran: "format", plus "tokens" and "grader" when given
    train_token_steps: int = 0  # the training cost that rule train_cost compares with its cap; 0 without a tokenizer

    @property
    def n_prompts(self) -> int:
        return len(self.groups)

    @property
    def n_rollouts(self) -> int:
        return sum(len(rollouts) for _, rollouts in self.groups)


def _need(ok: bool, rule: str, reason: str) -> None:
    if not ok:
        raise Invalid(rule, reason)


def _reject_constant(name: str) -> None:
    raise Invalid("nonfinite", f"non-finite number {name}")


def _finite(value: Any) -> bool:
    """A real JSON number (not true or false) that fits a float."""
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:  # a whole number too large for a float
        return False


def _load(raw: bytes, max_bytes: int) -> dict[str, Any]:
    _need(isinstance(raw, (bytes, bytearray)) and 0 < len(raw) <= max_bytes, "size", f"file must be 1..{max_bytes} bytes")
    try:
        value = json.loads(bytes(raw).decode("utf-8"), parse_constant=_reject_constant)
    except Invalid:
        raise
    except (ValueError, RecursionError) as exc:
        raise Invalid("json", f"not UTF-8 JSON: {type(exc).__name__}") from None
    _need(isinstance(value, dict), "top_keys", "the file must be one JSON object with keys version, hotkey, groups")
    return value


def _train(value: Any, recipe: Recipe) -> dict[str, Any]:
    keys = ", ".join(bound.key for bound in recipe.miner_bounds)
    _need(isinstance(value, dict), "train", f"train must be an object; its keys may be: {keys}")
    _need(all(recipe.bound(key) is not None for key in value), "train", f"train keys must be among: {keys}")
    try:
        recipe.for_miner(value)  # the same gate the trainer uses; it never clamps
    except BatchRejected as exc:
        raise Invalid("train", str(exc)) from None
    except (OverflowError, TypeError):  # a number too large for a float, or a list that holds a non-string
        raise Invalid("train", "a train value is outside its published bound") from None
    return dict(value)


def _header(value: dict[str, Any], recipe: Recipe) -> tuple[str, dict[str, Any]]:
    version = value.get("version")
    may_train = version in FORMATS[1:]
    allowed = _TOP_KEYS | {"train"} if may_train else _TOP_KEYS
    extra = "; train is optional" if may_train else ""
    _need(_TOP_KEYS <= set(value) <= allowed, "top_keys", f"top level keys must be version, hotkey, groups{extra}")
    _need(may_train or version == FORMAT_V1, "version", "version must be one of: " + ", ".join(FORMATS))
    return version, _train(value["train"], recipe) if "train" in value else {}


def _groups(value: dict[str, Any], hotkey: str) -> list[Group]:
    """The shape of every group, including groups the caps will drop."""
    _need(value["hotkey"] == hotkey, "hotkey", "hotkey does not match the owner of the stored object")
    groups = value["groups"]
    _need(isinstance(groups, list) and len(groups) > 0, "groups", "groups must be a non-empty list")
    out: list[Group] = []
    prompts: set[str] = set()
    for i, group in enumerate(groups):
        at = f"groups[{i}]"
        _need(isinstance(group, dict) and set(group) == {"prompt", "rollouts"}, "group_keys", f"{at} keys must be exactly prompt, rollouts")
        prompt, rollouts = group["prompt"], group["rollouts"]
        _need(isinstance(prompt, str) and prompt.strip() != "", "prompt", f"{at}.prompt must be a non-empty string")
        _need(prompt not in prompts, "duplicate_prompt", f"{at}.prompt repeats an earlier prompt")
        prompts.add(prompt)
        _need(isinstance(rollouts, list) and len(rollouts) > 0, "rollouts", f"{at}.rollouts must be a non-empty list")
        rows = []
        for j, rollout in enumerate(rollouts):
            here = f"{at}.rollouts[{j}]"
            _need(isinstance(rollout, dict) and set(rollout) == {"output", "reward"}, "rollout_keys", f"{here} keys must be exactly output, reward")
            _need(isinstance(rollout["output"], str), "output", f"{here}.output must be a string")
            _need(_finite(rollout["reward"]), "reward", f"{here}.reward must be a finite number")
            rows.append((rollout["output"], float(rollout["reward"])))
        out.append((prompt, tuple(rows)))
    return out


def _cap(groups: list[Group], max_prompts: int, max_rollouts: int) -> list[Group]:
    """Keep whole groups, in file order, while both caps hold."""
    kept, n_rollouts = 0, 0
    while kept < min(len(groups), max_prompts) and n_rollouts + len(groups[kept][1]) <= max_rollouts:
        n_rollouts += len(groups[kept][1])
        kept += 1
    _need(kept > 0, "cap", "the first group alone exceeds the rollout cap")
    return groups[:kept]


def _text(text: str, at: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise Invalid("text", f"{at} is not valid Unicode text") from None
    _need(not any(token in text for token in CONTROL_TOKENS), "control_token", f"{at} contains a chat control token")


def final_answer(text: str) -> str | None:
    """The content of the last ``\\boxed{...}`` with all white space removed, or None when there is none."""
    start = text.rfind(_BOXED + "{")
    if start < 0:
        return None
    begin, depth = start + len(_BOXED) + 1, 0
    for position in range(begin, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            if depth == 0:
                return "".join(text[begin:position].split())
            depth -= 1
    return None


def _count(tokens: Tokens, count: Callable[[str], int], text: str, where: str) -> int:
    """The token count of one text. A text that makes the tokenizer fail is the fault of the file, when the tokenizer still works.

    The tokenizer works when it counts ``PROBE_TEXT``. If it fails there too, the failure is ours: ``InfraError``.
    A Rust tokenizer reports a panic with a ``BaseException``, so that is caught here too.
    """
    try:
        return count(text)
    except (InfraError, KeyboardInterrupt, SystemExit):
        raise
    except BaseException as cause:
        try:
            works = tokens.prompt_tokens(PROBE_TEXT) > 0 and tokens.output_tokens(PROBE_TEXT) > 0
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            works = False
        if not works:
            raise InfraError(f"the tokenizer fails on its known-good case: {type(cause).__name__}") from cause
        raise Invalid(TOKENIZER_FAULT, f"{where}: the tokenizer fails on this text", proven=not isinstance(cause, MACHINE)) from None


def _reasoning(output: str) -> str | None:
    """The text before the first boxed answer, or None when the output has no boxed answer."""
    cut = output.find(_BOXED)
    return None if cut < 0 else output[:cut]


class _Ahead:
    """The output token counts of one file, taken in one call for the next texts of the walk in ``_kept``.

    The walk counts the same texts in the same order as before, and it stops at the same place with the same
    reason: a count from here is the count that ``tokens.output_tokens`` gives, and a text with no count from
    here is counted by ``_count``, as before. Only the time changes. The texts of one call are the next
    ``AHEAD_CHARS`` characters of the walk, so the memory is bounded and a file that fails early costs at most one
    call more than before. When a call fails or gives something that is not one count for each text, no other
    call is made for this file: the walk counts one text at a time and finds the text and the reason as before.
    """

    def __init__(self, tokens: Tokens, groups: Sequence[Group]) -> None:
        self._tokens, self._many = tokens, getattr(tokens, "output_tokens_many", None)
        self._texts, self._counts = self._walk(groups), {}

    @staticmethod
    def _walk(groups: Sequence[Group]) -> Iterator[str]:
        """Each text that ``_kept`` counts, in its order: an output, then its reasoning."""
        for _, rollouts in groups:
            for output, _ in rollouts:
                yield output
                reasoning = _reasoning(output)
                if reasoning is not None:
                    yield reasoning

    def _next(self) -> dict[str, int]:
        texts, size = [], 0
        for text in self._texts:
            texts.append(text)
            size += len(text)
            if size >= AHEAD_CHARS:
                break
        try:
            counts = list(self._many(texts)) if texts else []
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # the walk meets the text that fails and gives the reason, with the proof of ``_count``
            counts = None
        if not texts or counts is None or len(counts) != len(texts) or not all(type(n) is int and n >= 0 for n in counts):
            self._many = None
            return {}
        return dict(zip(texts, counts))

    def count(self, text: str, where: str) -> int:
        if self._many is not None and text not in self._counts:
            self._counts = self._next()
        n = self._counts.get(text)
        return _count(self._tokens, self._tokens.output_tokens, text, where) if n is None else n


def _kept(groups: Sequence[Group], params: Params, tokens: Tokens | None) -> tuple[set[tuple[int, int]], Sizes | None]:
    """Text, token and content rules on the kept groups.

    Return the (group, rollout) places of the capped outputs, and the prompt and output token counts of every
    rollout by its place. Without a tokenizer there are no counts: ``None``.
    """
    cap, context = params.max_output_tokens, params.max_context_tokens
    n_rollouts = answer_only = reasoning = total = 0
    answers: Counter[str] = Counter()
    capped: set[tuple[int, int]] = set()
    sizes: Sizes = {}
    ahead = _Ahead(tokens, groups) if tokens is not None else None
    for i, (prompt, rollouts) in enumerate(groups):
        _text(prompt, f"groups[{i}].prompt")
        n_prompt = _count(tokens, tokens.prompt_tokens, prompt, f"groups[{i}].prompt") if tokens else 0
        for j, (output, _) in enumerate(rollouts):
            here = f"groups[{i}].rollouts[{j}]"
            _text(output, f"{here}.output")
            n_rollouts += 1
            answer = final_answer(output)
            if answer is not None:
                answers[answer] += 1
            if tokens is None:
                continue
            n = ahead.count(output, f"{here}.output")
            _need(n <= cap, "output_tokens", f"{here}.output is {n} tokens; the cap is {cap}")
            used = n_prompt + n + (n < cap)  # an output below the cap gets one end-of-sequence token; at the cap it is cut
            capped.update({(i, j)} if n == cap else ())
            sizes[i, j] = (n_prompt, n)
            _need(used <= context, "context_tokens", f"{here}: prompt + output is {used} tokens; the cap is {context}")
            thought = _reasoning(output)  # reasoning comes before the first boxed answer
            before = n if thought is None else ahead.count(thought, f"{here}.output")
            answer_only += before < params.min_reasoning_tokens
            reasoning, total = reasoning + before, total + n
    if tokens is not None:
        limit = params.max_answer_only_frac
        _need(answer_only <= limit * n_rollouts, "answer_only",
              f"{answer_only} of {n_rollouts} rollouts have under {params.min_reasoning_tokens} reasoning tokens before the first boxed answer; the limit is a share of {limit:g}")
        share = params.reasoning_token_share
        _need(reasoning >= share * total, "reasoning_share", f"reasoning is {reasoning} of {total} output tokens; the least share is {share:g}")
    limit = max(params.max_answer_repeat_count, params.max_answer_repeat_frac * n_rollouts)
    worst = max(answers.values(), default=0)
    _need(worst <= limit, "answer_repeat", f"one boxed answer is the final answer of {worst} rollouts; the limit is {limit:g}")
    return capped, (sizes if tokens is not None else None)


def trained_tokens(n_prompt: int, n_output: int, recipe: Recipe) -> int:
    """The tokens of one rollout that the trainer runs through the model in one step. 0: the trainer drops the rollout.

    This is the arithmetic of ``gpu.batch.fit_tokens``. The room for the output is ``train_max_output_tokens``,
    or less when the prompt is long: ``train_max_context_tokens`` minus the prompt. An output shorter than the
    room gets one end-of-turn token. An output that fills the room or is longer is cut to the room. With no
    output token or no room, nothing is left to train on.
    """
    room = min(recipe.train_max_output_tokens, recipe.train_max_context_tokens - n_prompt)
    if n_output < 1 or room < 1:
        return 0
    return n_prompt + (room if n_output >= room else n_output + 1)


def _trains(groups: Sequence[Group], params: Params, train: Mapping[str, Any], sizes: Sizes | None) -> int:
    """The file must train something, and must not cost more training than one seat may use. Return the cost.

    Both rules use the trainer's own arithmetic. A rollout trains when its advantage is not zero and the
    trainer does not drop it. The advantage is the reward minus the mean reward of the training group
    (``gpu.batch.advantages``). A file in which no rollout trains gives no gradient: the trainer rejects it, so
    it must not take a slot. The cost is ``steps`` times the tokens of the rollouts that train
    (``trained_tokens``): the trainer runs exactly these tokens through the model in each step. Without a
    tokenizer the checker cannot know the tokens: no rollout counts as dropped and the cost rule does not run.
    """
    recipe = params.recipe.for_miner(train)
    active = tokens = 0
    for i, (_, rollouts) in enumerate(groups):
        try:
            group = advantages([reward for _, reward in rollouts], recipe.group_size)
        except OverflowError:  # finite rewards whose sum is not: the trainer cannot take the mean either
            raise Invalid("reward", f"groups[{i}]: the rewards are too large to take their mean") from None
        for j, advantage in enumerate(group):
            if advantage == 0:
                continue
            row = trained_tokens(*sizes[i, j], recipe) if sizes is not None else 1
            active, tokens = active + (row > 0), tokens + row
    _need(active > 0, NO_SIGNAL, "no rollout has a reward that differs from the mean reward of its training group, so the file trains nothing")
    if sizes is None:
        return 0
    cost, cap = recipe.steps * tokens, params.max_train_token_steps
    _need(cap <= 0 or cost <= cap, TRAIN_COST,
          f"training this file costs {cost} token-steps ({recipe.steps} steps times {tokens} tokens in {active} rollouts that train); the cap is {cap}")
    return cost


def _grades(grader: Grader, rows: Sequence[tuple[str, str, bool]]) -> Iterable[float | None | Exception]:
    """The outcome of each row, in order and on demand: the reward, None, or the exception that says why there is no grade."""
    if hasattr(grader, "many"):
        yield from grader.many(rows)
        return
    for row in rows:
        try:
            yield grader(*row)
        except InfraError:
            raise
        except Exception as exc:
            yield exc


def _fault(grader: Grader, row: tuple[str, str, bool], cause: Exception) -> bool:
    """The grader gave no grade for one row. Return whether that is proven to be the fault of the row.

    First the known-good cases of the grader. None passes: the failure is ours and ``InfraError`` is raised.
    The proof needs more: every known-good case passes now, and the cause is not in ``MACHINE``. For a row with
    no grade at all (not a time limit), the same row, graded again alone, must also get no grade: a worker that
    our machine killed does not die twice at the same row. A row over its time limit is not graded again: a slow
    machine is slow for both tries, and a file that hangs would cost its time limit twice.
    A grader with no ``probe`` method cannot give the proof: its time limit is a verdict for this call that is
    never counted, and any other exception out of it is ``InfraError``.
    """
    if not hasattr(grader, "probe"):
        if isinstance(cause, GraderTimeLimit):
            return False
        raise InfraError(f"the official grader failed: {type(cause).__name__}") from cause
    if not grader.probe() or isinstance(cause, MACHINE):
        return False
    return isinstance(cause, GraderTimeLimit) or isinstance(next(iter(_grades(grader, [row]))), Exception)


def _graded(groups: Sequence[Group], params: Params, grader: Grader, capped: set[tuple[int, int]]) -> None:
    """Every submitted reward must be the official grade. A prompt the environment does not know has no grade.

    A row that the grader cannot grade, in time or at all, is not allowed, whatever its reward. The check stops
    at the first one, so a file cannot make the checker wait at every row. That verdict is about this file
    only: if the grader fails on its known-good case too, the failure is ours and ``InfraError`` is raised.
    """
    places = [(i, j) for i, (_, rollouts) in enumerate(groups) for j in range(len(rollouts))]
    rows = [(groups[i][0], groups[i][1][j][0], (i, j) in capped) for i, j in places]
    unknown: set[int] = set()
    for (i, j), row, official in zip(places, rows, _grades(grader, rows)):
        here = f"groups[{i}].rollouts[{j}]"
        if isinstance(official, GraderTimeLimit):
            raise Invalid(TIME_LIMIT, f"{here}: the official grader did not finish this row inside its time limit", _fault(grader, row, official))
        if isinstance(official, Exception):
            raise Invalid(FAULT, f"{here}: the official grader fails on this row", _fault(grader, row, official))
        if official is None:
            unknown.add(i)
        else:
            _need(abs(groups[i][1][j][1] - official) <= params.reward_tolerance, "grader", f"{here}.reward is not the official grade")
    known, share = len(groups) - len(unknown), params.min_pool_prompt_frac
    _need(known >= share * len(groups), "pool", f"{known} of {len(groups)} prompts are tasks of the public pool; the least share is {share:g}")


def check_batch(raw: bytes, hotkey: str, params: Params, *, tokens: Tokens | None = None, grader: Grader | None = None) -> Batch:
    """Check one file against every rule. Return the ``Batch`` or raise ``Invalid``.

    Without ``tokens`` the token caps and the reasoning rules do not run, and
    no output counts as capped. Without ``grader`` the grader match does not
    run. ``Batch.checked`` says what ran. The validator always passes ``tokens``.

    A file that makes the checker fail with an exception of any other kind is invalid with rule ``check_fault``,
    when the checker still passes a known-good file. If it does not, the failure is ours: ``InfraError``.
    """
    try:
        return _check(raw, hotkey, params, tokens, grader)
    except (Invalid, InfraError, KeyboardInterrupt, SystemExit):
        raise
    except BaseException as cause:
        _still_works(params, tokens, grader, cause)
        raise Invalid(CHECK_FAULT, f"the checker fails on this file: {_where(cause)}", proven=not isinstance(cause, MACHINE)) from cause


def _where(cause: BaseException) -> str:
    """The class of an exception and the place in the code that raised it. Never its message: that can hold batch text."""
    trace = cause.__traceback__
    if trace is None:
        return type(cause).__name__
    while trace.tb_next is not None:
        trace = trace.tb_next
    code = trace.tb_frame.f_code
    return f"{type(cause).__name__} in {code.co_name}, {os.path.basename(code.co_filename)} line {trace.tb_lineno}"


def _known_good(params: Params, grader: Grader | None) -> Iterator[bytes]:
    """Small files that must pass every rule: one prompt, a right answer with reward 1 and a try with no answer with reward 0.

    With the grader of an environment the prompts are its known-good cases (the first pool tasks), one file for
    each. Without one there is one file with a plain sum, which no environment grades.
    """
    think = "".join(f"Step {k}. " for k in range(params.min_reasoning_tokens + 32))  # two words or more for each step
    for prompt, answer in grader.cases() if hasattr(grader, "cases") else [(PROBE_TEXT, "2")]:
        rollouts = [{"output": f"{think}{_BOXED}{{{answer}}}", "reward": 1.0}, {"output": think.strip(), "reward": 0.0}]
        yield json.dumps({"version": FORMAT_V1, "hotkey": PROBE_HOTKEY, "groups": [{"prompt": prompt, "rollouts": rollouts}]}).encode()


def _still_works(params: Params, tokens: Tokens | None, grader: Grader | None, cause: BaseException) -> None:
    """The checker failed on one file. Return when it still passes a known-good file, so the file is at fault. If not, raise ``InfraError``."""
    problem = "it has no known-good file"
    try:
        for raw in _known_good(params, grader):
            try:
                _check(raw, PROBE_HOTKEY, params, tokens, grader)
                return
            except Invalid as exc:
                problem = f"rule {exc.rule}"
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        problem = _where(exc)
    raise InfraError(f"the checker fails on a file ({_where(cause)}) and does not pass its known-good file ({problem})") from cause


def _check(raw: bytes, hotkey: str, params: Params, tokens: Tokens | None, grader: Grader | None) -> Batch:
    value = _load(raw, params.max_batch_bytes)
    version, train = _header(value, params.recipe)
    groups = _groups(value, hotkey)
    kept = _cap(groups, params.max_prompts, params.max_rollouts)
    capped, sizes = _kept(kept, params, tokens)
    cost = _trains(kept, params, train, sizes)
    if grader is not None:
        _graded(kept, params, grader, capped)
    checked = ("format",) + (("tokens",) if tokens is not None else ()) + (("grader",) if grader is not None else ())
    return Batch(sha256_hex(bytes(raw)), len(raw), version, tuple(kept), len(groups) - len(kept), train, checked, cost)


def verdict(raw: bytes, hotkey: str, params: Params, *, tokens: Tokens | None = None, grader: Grader | None = None, now: int = 0) -> Verdict:
    """``check_batch`` as a public ``Verdict``. It never raises ``Invalid`` and never holds batch content."""
    return _verdict(hotkey, raw, _judge(raw, hotkey, params, tokens, grader), now)


def _judge(raw: bytes, hotkey: str, params: Params, tokens: Tokens | None, grader: Grader | None) -> Batch | Invalid:
    try:
        return check_batch(raw, hotkey, params, tokens=tokens, grader=grader)
    except Invalid as exc:
        return exc


def _verdict(hotkey: str, raw: bytes, result: Batch | Invalid, now: int) -> Verdict:
    if isinstance(result, Batch):
        return Verdict(hotkey, True, "", result.format, result.n_prompts, result.n_rollouts, result.sha256, now)
    file_hash = sha256_hex(bytes(raw)) if isinstance(raw, (bytes, bytearray)) and raw else ""
    return Verdict(hotkey, False, str(result), batch_sha256=file_hash, checked_at=now, rule=result.rule)


SHA_SHOWN = 12  # characters of a sha256 in a log line


def verdict_note(verdict: Verdict | None) -> str:
    """One check of a file as the log shows it: the start of the sha256, then valid, or the rule code and its reason."""
    if verdict is None:
        return "no file"
    start = verdict.batch_sha256[:SHA_SHOWN] or "not hashed"  # an empty or oversized file is not hashed
    if verdict.valid:
        return f"{start} valid"
    return f"{start} not valid, rule {verdict.rule}: {' '.join(verdict.reason.split())}"


def verdict_log(seats: Sequence[tuple[int, str]], verdicts: Mapping[str, Verdict], before: Mapping[str, str], now: int) -> tuple[list[str], dict[str, str]]:
    """The log lines of one check of the waiting files, and the notes that the next check compares with.

    ``seats`` holds the UID and the hotkey of each waiting seat. A seat gets a line only when its note differs
    from the one in ``before``: other bytes, another verdict, another reason, or a new count of a fault
    ("check 2 of 3"). A seat that this process sees for the first time gets a line when it has a file. The last
    line is the summary of the check, with the count of each rule. The verdicts are public (``Verdict``), so a
    line never holds batch content.
    """
    notes = {hotkey: verdict_note(verdicts.get(hotkey)) for _, hotkey in seats}
    lines = [
        f"files: UID {uid} {notes[hotkey]}; before: {before.get(hotkey, 'not checked by this process')}"
        for uid, hotkey in seats
        if notes[hotkey] != before.get(hotkey, "no file")
    ]
    found = [verdicts[hotkey] for _, hotkey in seats if hotkey in verdicts]
    rules = Counter(verdict.rule for verdict in found if not verdict.valid)
    detail = f" ({', '.join(f'{rule} {count}' for rule, count in sorted(rules.items()))})" if rules else ""
    at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now))
    lines.append(
        f"files: check at {at}: {len(seats)} waiting seats, {len(found) - sum(rules.values())} valid, "
        f"{sum(rules.values())} not valid{detail}, {len(seats) - len(found)} with no file, {len(lines)} changed"
    )
    return lines, {**before, **notes}


class StoreAdmission:
    """The ``Admission`` seam over an object store.

    A miner has one private object and may overwrite it until the freeze.
    ``freeze`` checks the exact bytes it copies, so what is frozen is what was
    checked and what is trained. A store failure raises ``InfraError``.
    """

    def __init__(self, store: ObjectStore, params: Params, *, tokens: Tokens, grader: Grader | None, clock: Callable[[], float] = time.time) -> None:
        self.store, self.params, self.tokens, self.grader, self.clock = store, params, tokens, grader, clock
        self._seen: dict[str, tuple[tuple[str, str], Verdict]] = {}  # hotkey -> ((stored sha256, params hash), verdict)
        self._tries: dict[str, tuple[tuple[str, str], str, int]] = {}  # hotkey -> (the same tag, rule and reason, failed checks in a row)
        self._held: dict[str, tuple[tuple[str, str], Verdict, float]] = {}  # hotkey -> (the same tag, a verdict in TRANSIENT, the time of its check)

    def _read(self, hotkey: str, key: str) -> tuple[bytes, Batch | Invalid] | None:
        """The object's bytes and their check, or None when there is no object."""
        limit = self.params.max_batch_bytes
        try:
            if self.store.size(key) > limit:  # never download an oversized object
                return b"", Invalid("size", f"file must be 1..{limit} bytes")
            raw = self.store.get(key)
        except KeyError:
            return None
        return raw, _judge(raw, hotkey, self.params, self.tokens, self.grader)

    def verdicts(self, hotkeys: Sequence[str]) -> Mapping[str, Verdict]:
        """The live check of each hotkey's stored file. A file is checked again only when its bytes or the params change.

        A verdict in ``TRANSIENT`` (a grader time limit, or a fault of the grader, the tokenizer or the checker)
        depends on this machine at this moment, so it is not kept. It holds for ``FAULT_HOLD_S``: during that
        time the same bytes with the same params keep the verdict and are not read. Then the next call checks
        the file again. After ``FAULT_TRIES`` such checks in a row the verdict is kept like any other
        (``_tried``). The hold is in memory only, and a clock that goes back ends it.
        """
        out: dict[str, Verdict] = {}
        for hotkey in hotkeys:
            key = layout.miner_object(hotkey)
            try:  # an oversized object is never hashed or downloaded
                oversized = self.store.size(key) > self.params.max_batch_bytes
                tag = ("oversized" if oversized else self.store.sha256(key), self.params.params_hash)
            except KeyError:
                for memory in (self._seen, self._tries, self._held):
                    memory.pop(hotkey, None)
                continue
            kept, held = self._seen.get(hotkey), self._held.get(hotkey)
            if held is not None and held[0] == tag and 0 <= self.clock() - held[2] < FAULT_HOLD_S:
                kept = held[:2]
            elif kept is None or kept[0] != tag:
                found = self._read(hotkey, key)
                if found is None:
                    continue
                result, final = self._tried(hotkey, tag, found[1])
                now = self.clock()
                kept = (tag, _verdict(hotkey, found[0], result, int(now)))
                self._seen.pop(hotkey, None)
                self._held.pop(hotkey, None)
                if final:
                    self._seen[hotkey] = kept
                else:
                    self._held[hotkey] = (*kept, now)
            out[hotkey] = kept[1]
        return out

    def _tried(self, hotkey: str, tag: tuple[str, str], result: Batch | Invalid) -> tuple[Batch | Invalid, bool]:
        """Count a failed check with a rule in ``TRANSIENT``. Returns the result to show and whether it is final.

        The count goes up only when the failure is proven (``Invalid.proven``) and is the same as the one before:
        the same bytes, the same params, the same rule and the same reason, so the same place in the file. Any
        other failure starts the count again, at 1 when proven and at 0 when not: a fault that moves from row to
        row, or that comes with a sign of trouble on this machine, is not held against the file. At ``FAULT_TRIES``
        the verdict is final. The count is in memory only: a new process starts at zero.
        """
        if not isinstance(result, Invalid) or result.rule not in TRANSIENT:
            self._tries.pop(hotkey, None)
            return result, True
        same = (tag, f"{result.rule}: {result}")
        tries = 0
        if result.proven:
            tries = (self._tries[hotkey][2] if self._tries.get(hotkey, ())[:2] == same else 0) + 1
        self._tries[hotkey] = (*same, tries)
        if not tries:
            return result, False
        note = f"check {tries} of {FAULT_TRIES}" if tries < FAULT_TRIES else f"checked {FAULT_TRIES} times: this verdict stays until a new file is uploaded"
        return Invalid(result.rule, f"{result} ({note})"), tries >= FAULT_TRIES

    def freeze(self, reg: Registration, round_id: str) -> FrozenBatch | None:
        """Copy the miner's file for this round and check the copy. None = missing or invalid: a skip, not charged.

        It always reads the miner's file as it is now. A miner may replace the file until the round opens, so
        a copy from an earlier try of the same round id is replaced, or removed on a skip. Call it only for a
        round that has not opened yet: an open round's tickets name the frozen copy.
        """
        frozen_key = layout.frozen_batch(round_id, reg.hotkey)
        found = self._read(reg.hotkey, layout.miner_object(reg.hotkey))
        if found is None or not isinstance(found[1], Batch):
            self.store.delete(frozen_key)
            return None
        raw, batch = found
        self.store.put(frozen_key, raw)
        if self.store.sha256(frozen_key) != batch.sha256:
            self.store.delete(frozen_key)  # a replay must not find the damaged copy
            raise VerifyError(f"the frozen copy of {reg.hotkey} for round {round_id} does not hash to the checked bytes")
        return FrozenBatch(reg.hotkey, batch.sha256, frozen_key, batch.format, batch.n_prompts, batch.n_rollouts, dict(batch.train))
