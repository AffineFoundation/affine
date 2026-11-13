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
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import layout
from .canonical import sha256_hex
from .errors import BatchRejected, InfraError, VerifyError
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
# It raises GraderTimeLimit when the grade did not finish in time. Any other exception is our failure: InfraError.
Grader = Callable[[str, str, bool], float | None]
TIME_LIMIT = "grader_time_limit"  # the rule code of a row that the official grader cannot grade in time
# The env's Grade.machine_cause for an expired time limit (envs.math_grader.TIME_LIMIT). This module cannot
# import the env, which needs extra packages. If the env renames the value, a time-out becomes InfraError: loud.
ENV_TIME_LIMIT = "time-limit"


class GraderTimeLimit(Exception):
    """Raised by a ``Grader``: the official grade of this row did not finish inside its time limit."""


def env_grader(pool: Any, grade: Callable[..., Any]) -> Grader:
    """The ``Grader`` of an environment, from its task pool and its ``grade(task, output, capped)`` function.

    The controller and the miner's tool both build the grader with this one call, so they cannot differ.
    The grade's ``machine_cause`` says when the machine, not the row, decided: an expired time limit
    raises ``GraderTimeLimit``, and any other cause (a dead worker) raises ``InfraError``.
    """
    def official(prompt: str, output: str, capped: bool) -> float | None:
        task = pool.task_for_prompt(prompt)
        if task is None:
            return None
        result = grade(task, output, capped)
        if result.machine_cause == ENV_TIME_LIMIT:
            raise GraderTimeLimit
        if result.machine_cause is not None:
            raise InfraError(f"the official grader failed: {result.machine_cause}")
        return float(result.reward)

    return official


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


def _kept(groups: Sequence[Group], params: Params, tokens: Tokens | None) -> set[tuple[int, int]]:
    """Text, token and content rules on the kept groups. Return the (group, rollout) places of the capped outputs."""
    cap, context = params.max_output_tokens, params.max_context_tokens
    n_rollouts = answer_only = reasoning = total = 0
    answers: Counter[str] = Counter()
    capped: set[tuple[int, int]] = set()
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
    return capped


def _graded(groups: Sequence[Group], params: Params, grader: Grader, capped: set[tuple[int, int]]) -> None:
    """Every submitted reward must be the official grade. A prompt the environment does not know has no grade.

    A row that the grader cannot grade in time is not allowed, whatever its reward. The check stops at the
    first one, so a file cannot make the checker wait at every row.
    """
    known = 0
    for i, (prompt, rollouts) in enumerate(groups):
        for j, (output, reward) in enumerate(rollouts):
            here = f"groups[{i}].rollouts[{j}]"
            try:
                official = grader(prompt, output, (i, j) in capped)
            except GraderTimeLimit:
                raise Invalid(TIME_LIMIT, f"{here}: the official grader did not finish this row inside its time limit") from None
            except InfraError:
                raise
            except Exception as exc:
                raise InfraError(f"the official grader failed: {type(exc).__name__}") from exc
            if official is None:
                break
            _need(abs(reward - official) <= params.reward_tolerance, "grader", f"{here}.reward is not the official grade")
        else:
            known += 1
    share = params.min_pool_prompt_frac
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
    capped = _kept(kept, params, tokens)
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
    return Verdict(hotkey, False, str(result), batch_sha256=file_hash, checked_at=now)


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

        A grader time limit depends on the speed of this machine, so that verdict is not kept: the next call checks again.
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
                if getattr(found[1], "rule", "") == TIME_LIMIT:
                    self._seen.pop(hotkey, None)
                else:
                    self._seen[hotkey] = kept
            out[hotkey] = kept[1]
        return out

    def freeze(self, reg: Registration, round_id: str) -> FrozenBatch | None:
        """Copy the miner's file for this round and check the copy. None = missing or invalid: a skip, not charged.

        A second call for the same seat and round returns the same frozen copy, so a replay is safe.
        """
        frozen_key = layout.frozen_batch(round_id, reg.hotkey)
        for source in (frozen_key, layout.miner_object(reg.hotkey)):
            found = self._read(reg.hotkey, source)
            if found is not None and isinstance(found[1], Batch):
                break
        else:
            return None
        raw, batch = found
        if source != frozen_key:
            self.store.put(frozen_key, raw)
        if self.store.sha256(frozen_key) != batch.sha256:
            self.store.delete(frozen_key)  # a replay must not find the damaged copy
            raise VerifyError(f"the frozen copy of {reg.hotkey} for round {round_id} does not hash to the checked bytes")
        return FrozenBatch(reg.hotkey, batch.sha256, frozen_key, batch.format, batch.n_prompts, batch.n_rollouts, dict(batch.train))
