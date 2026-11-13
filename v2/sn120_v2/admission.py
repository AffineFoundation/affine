"""Admission: the strict check of a miner's batch file, the live verdicts and the freeze at round open.

A batch file is one JSON object::

    {"version": "affine-batch-v1" | "affine-batch-v2" | "affine-batch-v3",
     "hotkey": "<the miner's ss58 hotkey>",
     "train": {...},                      # optional, never on v1: only keys in Recipe.miner_bounds
     "groups": [{"prompt": "<text>", "rollouts": [{"output": "<text>", "reward": <number>}, ...]}, ...]}

``check_batch`` applies every rule in a fixed order and stops at the first one
that fails. Nothing is ever repaired or clamped: one bad value makes the whole
file invalid. Every limit comes from ``Params``.

This module uses the standard library only, so a miner can run the same check
on their own machine. The tokenizer and the official grader are inputs.
"""

from __future__ import annotations

import json
import math
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

# The official grade of one rollout: (prompt, output, capped) -> reward.
# The checker sets capped: True when the output is exactly max_output_tokens long, so it stopped at the cap.
# None means the prompt is not a task of the environment, so there is no official grade.
# It raises GraderTimeLimit when the grade did not finish in time, and any other exception when grading the row
# made it fail. A grader may also have two optional methods, as EnvGrader has:
#   many(rows) -> the outcome of each (prompt, output, capped) row, in order: the reward, None, or the exception
#   probe()    -> raises InfraError unless the grader works now
Grader = Callable[[str, str, bool], float | None]
TIME_LIMIT = "grader_time_limit"  # the rule code of a row that the official grader cannot grade in time
FAULT = "grader_fault"  # the rule code of a row that makes the official grader fail, while the grader itself works
NO_SIGNAL = "no_signal"  # the rule code of a file in which no rollout can change the model
# These two verdicts depend on this machine at this moment. They hold for one call and are never kept.
TRANSIENT = (TIME_LIMIT, FAULT)
# The env's Grade.machine_cause for an expired time limit (envs.math_grader.TIME_LIMIT). This module cannot
# import the env, which needs extra packages. If the env renames the value, a time-out becomes a grader fault.
ENV_TIME_LIMIT = "time-limit"
PROBE_TASKS = 4  # the grader works when it gives reward 1 to one of the first pool tasks answered with its own answer

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
    "reward": "a reward is not a finite number",
    "cap": "the first group alone is larger than the rollout cap",
    "text": "a prompt or an output is not valid Unicode text",
    "control_token": "a prompt or an output holds a chat control token",
    "output_tokens": "an output is longer than the output token cap",
    "context_tokens": "prompt + output is longer than the context token cap",
    "answer_only": "too many rollouts have almost no reasoning before the first boxed answer",
    "reasoning_share": "too small a share of the output tokens is reasoning",
    "answer_repeat": "one final answer is in too many rollouts",
    NO_SIGNAL: "no rollout has a reward that differs from the mean reward of its training group, so nothing trains",
    TIME_LIMIT: "the official grader cannot grade a rollout inside its time limit; the file is checked again later",
    FAULT: "a rollout makes the official grader fail; the file is checked again later",
    "grader": "a reward is not the official grade",
    "pool": "too small a share of the prompts are tasks of the public pool",
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

    def probe(self) -> None:
        """Raise ``InfraError`` unless the grader works now: a pool task with its own answer must get reward 1."""
        tasks = [self.pool.task(task_id) for task_id in self.pool.task_ids[:PROBE_TASKS]]
        try:
            grades = self.grade_many([(task, _BOXED + "{" + task.answer + "}", False) for task in tasks])
            works = any(grade.machine_cause is None and grade.reward == 1.0 for grade in grades)
        except Exception as exc:
            raise InfraError(f"the official grader failed on a known-good case: {type(exc).__name__}") from exc
        if not works:
            raise InfraError("the official grader gives no known-good case its reward")


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


class Invalid(ValueError):
    """The file breaks a published rule.

    ``rule`` is a stable code. ``str()`` is the public reason: it names
    positions, counts and published limits, never batch text or a train value.
    """

    def __init__(self, rule: str, reason: str) -> None:
        super().__init__(reason)
        self.rule = rule


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


def _kept(groups: Sequence[Group], params: Params, tokens: Tokens | None) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Text, token and content rules on the kept groups.

    Return two sets of (group, rollout) places: the outputs that stopped at the cap, and the rollouts that the
    trainer drops because no output token is left to train on (see ``gpu.batch.fit_tokens``).
    """
    cap, context = params.max_output_tokens, params.max_context_tokens
    n_rollouts = answer_only = reasoning = total = 0
    answers: Counter[str] = Counter()
    capped: set[tuple[int, int]] = set()
    untrained: set[tuple[int, int]] = set()
    for i, (prompt, rollouts) in enumerate(groups):
        _text(prompt, f"groups[{i}].prompt")
        n_prompt = tokens.prompt_tokens(prompt) if tokens else 0
        for j, (output, _) in enumerate(rollouts):
            here = f"groups[{i}].rollouts[{j}]"
            _text(output, f"{here}.output")
            n_rollouts += 1
            answer = final_answer(output)
            if answer is not None:
                answers[answer] += 1
            if tokens is None:
                continue
            n = tokens.output_tokens(output)
            _need(n <= cap, "output_tokens", f"{here}.output is {n} tokens; the cap is {cap}")
            used = n_prompt + n + (n < cap)  # an output below the cap gets one end-of-sequence token; at the cap it is cut
            capped.update({(i, j)} if n == cap else ())
            untrained.update({(i, j)} if n == 0 or n_prompt >= params.recipe.train_max_context_tokens else ())
            _need(used <= context, "context_tokens", f"{here}: prompt + output is {used} tokens; the cap is {context}")
            cut = output.find(_BOXED)
            before = n if cut < 0 else tokens.output_tokens(output[:cut])  # reasoning comes before the first boxed answer
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
    return capped, untrained


def _signal(groups: Sequence[Group], group_size: int, untrained: set[tuple[int, int]]) -> None:
    """The trainer's own test, with the trainer's own arithmetic: a rollout trains only when its advantage is not zero.

    The advantage is the reward minus the mean reward of the training group (``gpu.batch.advantages``). A file
    in which no rollout trains gives no gradient. The trainer rejects it, so it must not take a slot.
    """
    for i, (_, rollouts) in enumerate(groups):
        for j, advantage in enumerate(advantages([reward for _, reward in rollouts], group_size)):
            if advantage != 0 and (i, j) not in untrained:
                return
    raise Invalid(NO_SIGNAL, "no rollout has a reward that differs from the mean reward of its training group, so the file trains nothing")


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


def _fault(grader: Grader, cause: Exception) -> None:
    """The grader failed on one row. Return when the grader still works, so the row is at fault. If not, raise ``InfraError``."""
    if not hasattr(grader, "probe"):  # no way to tell the row from the grader: stay loud
        raise InfraError(f"the official grader failed: {type(cause).__name__}") from cause
    grader.probe()


def _graded(groups: Sequence[Group], params: Params, grader: Grader, capped: set[tuple[int, int]]) -> None:
    """Every submitted reward must be the official grade. A prompt the environment does not know has no grade.

    A row that the grader cannot grade, in time or at all, is not allowed, whatever its reward. The check stops
    at the first one, so a file cannot make the checker wait at every row. That verdict is about this file
    only: if the grader fails on its known-good case too, the failure is ours and ``InfraError`` is raised.
    """
    places = [(i, j) for i, (_, rollouts) in enumerate(groups) for j in range(len(rollouts))]
    rows = [(groups[i][0], groups[i][1][j][0], (i, j) in capped) for i, j in places]
    unknown: set[int] = set()
    for (i, j), official in zip(places, _grades(grader, rows)):
        here = f"groups[{i}].rollouts[{j}]"
        if isinstance(official, GraderTimeLimit):
            raise Invalid(TIME_LIMIT, f"{here}: the official grader did not finish this row inside its time limit")
        if isinstance(official, Exception):
            _fault(grader, official)
            raise Invalid(FAULT, f"{here}: the official grader fails on this row")
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
    """
    value = _load(raw, params.max_batch_bytes)
    version, train = _header(value, params.recipe)
    groups = _groups(value, hotkey)
    kept = _cap(groups, params.max_prompts, params.max_rollouts)
    capped, untrained = _kept(kept, params, tokens)
    _signal(kept, params.recipe.for_miner(train).group_size, untrained)
    if grader is not None:
        _graded(kept, params, grader, capped)
    checked = ("format",) + (("tokens",) if tokens is not None else ()) + (("grader",) if grader is not None else ())
    return Batch(sha256_hex(bytes(raw)), len(raw), version, tuple(kept), len(groups) - len(kept), train, checked)


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


class StoreAdmission:
    """The ``Admission`` seam over an object store.

    A miner has one private object and may overwrite it until the freeze.
    ``freeze`` checks the exact bytes it copies, so what is frozen is what was
    checked and what is trained. A store failure raises ``InfraError``.
    """

    def __init__(self, store: ObjectStore, params: Params, *, tokens: Tokens, grader: Grader | None, clock: Callable[[], float] = time.time) -> None:
        self.store, self.params, self.tokens, self.grader, self.clock = store, params, tokens, grader, clock
        self._seen: dict[str, tuple[tuple[str, str], Verdict]] = {}  # hotkey -> ((stored sha256, params hash), verdict)

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

        A verdict in ``TRANSIENT`` (a grader time limit or a grader fault) depends on this machine at this moment,
        so it is not kept: the next call checks the file again.
        """
        out: dict[str, Verdict] = {}
        for hotkey in hotkeys:
            key = layout.miner_object(hotkey)
            try:  # an oversized object is never hashed or downloaded
                oversized = self.store.size(key) > self.params.max_batch_bytes
                tag = ("oversized" if oversized else self.store.sha256(key), self.params.params_hash)
            except KeyError:
                self._seen.pop(hotkey, None)
                continue
            kept = self._seen.get(hotkey)
            if kept is None or kept[0] != tag:
                found = self._read(hotkey, key)
                if found is None:
                    continue
                kept = (tag, _verdict(hotkey, found[0], found[1], int(self.clock())))
                if getattr(found[1], "rule", "") in TRANSIENT:
                    self._seen.pop(hotkey, None)
                else:
                    self._seen[hotkey] = kept
            out[hotkey] = kept[1]
        return out

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
