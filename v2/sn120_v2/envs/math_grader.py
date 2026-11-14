"""The official math grader: math-verify 0.9.0 first, then the lenient rules.

One function, ``grade``, decides every rollout: the head M, every LoRA, and
every row of a miner batch. It knows nothing about which model wrote the text.

The decision has four steps, in this order:

1. A capped output (no end of sequence before the token cap) is never graded: code ``C``.
2. The strict format rule looks for the final answer: exactly one ``<think>...</think>`` block first, then
   exactly one balanced, non-empty ``\\boxed{}`` with nothing after it.
3. math-verify compares the boxed answer with the reference. If math-verify itself fails (it cannot parse the
   reference, or it raises an error or times out), the result is code ``E``: not passed, never a pass by retry.
4. If the answer is not yet correct, the lenient rules try again: P1 and P2 find a box that the strict format
   rule refused, and E1 to E7 accept common other ways to write the same answer.

``GradePool`` runs ``grade`` in worker processes with a hard time limit, so a
hung grade can never stall a round and is never counted as a pass.
"""

from __future__ import annotations

import multiprocessing
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Callable, Sequence

import sympy
from latex2sympy2_extended import latex2sympy
from math_verify import parse, verify
from math_verify.errors import TimeoutException

from ..canonical import digest, sha256_hex
from ..errors import InfraError
from ..types import CAP, ERROR, FAIL, PASS
from .math_pool import ENV_ID, POOL_SHA256, Task

GRADER_VERSION = "math-verify-0.9.0+lenient-v1"
# The exact packages the grader needs. Another version can change a verdict, so it is refused.
PINNED = {
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "sympy": "1.14.0",
    "antlr4-python3-runtime": "4.13.2",
    "mpmath": "1.3.0",
}

BOX = "\\boxed{"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
# Control tokens of the head tokenizer. One of them inside an output means the strict format rule finds no answer.
CONTROL_TOKENS = (
    "<|endoftext|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
    "<|video_pad|>", "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>",
)  # fmt: skip
MATH_OPENER = {"\\]": "\\[", "\\)": "\\(", "$$": "$$", "$": "$"}  # closing math wrapper -> its opener
# The values of ``Grade.machine_cause``. They are part of the API: do not rename them.
TIME_LIMIT = "time-limit"
WORKER_DIED = "worker-died"


@dataclass(frozen=True, slots=True)
class Grade:
    """The grade of one output.

    ``code`` is an outcome code of ``sn120_v2.types``: ``P`` pass, ``F`` fail,
    ``C`` capped, ``E`` the grader could not decide. ``reward`` is 1.0 for
    ``P`` and 0.0 for every other code. ``answer`` is the ``\\boxed{...}``
    text that was graded, or ``None``. ``rule`` says what decided:

    * ``P``: ``live`` (strict format and math-verify) or the lenient rules used, for example ``P2+E1``.
    * ``F``: ``wrong`` (strict format, wrong answer) or ``unresolved:<why>`` (no answer in the strict format).
    * ``C``: ``capped``.
    * ``E``: ``reference-unparseable``, ``grader-error``, ``grader-timeout``, ``grade-timeout`` or ``grade-worker-died``.

    ``rule`` is text for logs. Do not branch on it. To tell the two kinds of
    ``E`` apart, read ``machine_cause``:

    * ``None``: the grade is stable. The same input gives the same grade on every machine. Every ``P``, ``F`` and
      ``C`` is stable. A stable ``E`` means the reference cannot be parsed, or math-verify raised an error.
    * ``TIME_LIMIT`` (``"time-limit"``): a time limit expired, math-verify's own limit or ``GradePool.timeout_s``.
    * ``WORKER_DIED`` (``"worker-died"``): the ``GradePool`` worker process died during the job.

    A grade with a ``machine_cause`` always has code ``E``. It can differ on another machine or at another time.
    """

    code: str
    reward: float
    cap_hit: bool
    answer: str | None
    rule: str
    machine_cause: str | None = None


def _fail(code: str, rule: str, answer: str | None = None, machine_cause: str | None = None) -> Grade:
    return Grade(code, 0.0, code == CAP, answer, rule, machine_cause)


def check_runtime() -> None:
    """Raise ``InfraError`` unless the installed grader packages are exactly the pinned versions.

    A wrong version is a failure of this box, not of a miner and not of the network.
    """
    for name, version in PINNED.items():
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            found = "not installed"
        if found != version:
            raise InfraError(f"math grader needs {name}=={version}, this box has {found}")


@lru_cache(maxsize=1)
def _runtime_checked() -> bool:
    check_runtime()
    return True


def env_pin(pool_sha256: str = POOL_SHA256) -> str:
    """The value for ``Params.env_pin``: one sha256 over the pool, the grader code and the grader packages."""
    source = sha256_hex(Path(__file__).read_bytes())
    return digest(
        {"env": ENV_ID, "pool_sha256": pool_sha256, "grader": GRADER_VERSION, "grader_source_sha256": source, "packages": PINNED}
    )


# ---- step 2: where is the final answer ----


def _box_end(text: str, start: int) -> int | None:
    """Index of the brace that closes a box whose content starts at ``start``."""
    depth = 1
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return None


def _strict_box(output: str) -> tuple[str | None, str]:
    """The boxed answer under the strict format rule, and ``ok`` or the reason there is none."""
    if output.count(THINK_OPEN) != 1 or output.count(THINK_CLOSE) != 1:
        return None, "think"
    before, _, rest = output.partition(THINK_OPEN)
    if before.strip() or THINK_CLOSE not in rest:
        return None, "think"
    final = rest.partition(THINK_CLOSE)[2].strip()
    if final.count(BOX) != 1:
        return None, "n_boxed"
    begin = final.index(BOX)
    end = _box_end(final, begin + len(BOX))
    if end is None or not final[begin + len(BOX) : end].strip():
        return None, "unclosed"
    tail, prefix = final[end + 1 :].strip(), final[:begin].rstrip()
    if tail:
        if tail not in MATH_OPENER or not prefix.endswith(MATH_OPENER[tail]):
            return None, "tail"
    elif prefix.endswith(("\\[", "\\(", "$")):
        return None, "prefix"
    return final[begin : end + 1], "ok"


def _lenient_box(output: str, reason: str) -> tuple[str | None, str]:
    """A box the strict format rule refused for ``reason``, and the rule that found it (``P1`` or ``P2``)."""
    last_part = output.split(THINK_CLOSE)[-1]
    if reason == "prefix":  # P1: the single box follows a "$" that closes earlier math
        final = last_part.strip()
        begin = final.index(BOX)
        prefix = final[:begin]
        open_dollar = re.sub(r"\\\$", "", prefix).count("$") % 2 == 1
        if not open_dollar and not prefix.rstrip().endswith(("\\[", "\\(")):
            return final[begin : _box_end(final, begin + len(BOX)) + 1], "P1"
    if reason in ("tail", "n_boxed", "think", "prefix"):  # P2: the last complete box after the last </think>
        begin = last_part.rfind(BOX)
        if begin >= 0:
            end = _box_end(last_part, begin + len(BOX))
            if end is not None and last_part[begin + len(BOX) : end].strip():
                return last_part[begin : end + 1], "P2"
    return None, ""


# ---- step 3: math-verify ----


@lru_cache(maxsize=4096)
def _reference_parses(gold: str) -> bool:
    try:
        return bool(parse(BOX + gold + "}", raise_on_error=True))
    except Exception:
        return False


@lru_cache(maxsize=4096)
def _same(gold: str, box: str) -> bool | None:
    """math-verify on a reference and a boxed answer. ``None`` means math-verify raised an error."""
    try:
        reference = parse(BOX + gold + "}", raise_on_error=True)
        answer = parse(box, raise_on_error=True)
        return bool(reference) and bool(verify(reference, answer, raise_on_error=True))
    except Exception:
        return None


# ---- step 4: other ways to write the same answer ----

_TEXT_WRAPPER = re.compile(r"\\(?:text|textbf|mathrm|mbox)\s*\{([^{}]*)\}")
_LETTER_LIST = re.compile(r"^\(?([A-Z])\)?(\s*,\s*\(?([A-Z])\)?)*$")
_OPTION_BOLD = re.compile(r"\\textbf\{\(([A-E])\)\}\\?\s*(.*?)(?=\\qquad|\\textbf\{\([A-E]\)\}|\$\s*$|$)", re.S)
_OPTION_TEXT = re.compile(r"\\text\s*([A-E])\.\s*(?:\\ )*\s*(.*?)(?=\\qquad|\\text\s*[A-E]\.|\$\$|$)", re.S)
_OPTION_LABEL = re.compile(r"^\s*\\?(?:text|textbf)?\{?\(?([A-E])\)?\}?\s*(\\\s)?\s*")
_COLUMN = re.compile(r"\\begin\{[pb]matrix\}(.*?)\\end\{[pb]matrix\}", re.S)
_NUMBER = r"(-?[\w\\{}.]+)"
_BOUND = {
    "<": "(-\\infty,{})", "\\le": "(-\\infty,{}]", "\\leq": "(-\\infty,{}]",
    ">": "({},\\infty)", "\\ge": "[{},\\infty)", "\\geq": "[{},\\infty)",
}  # fmt: skip


def _plain(latex: str) -> str:
    """E1: drop text wrappers and thousands separators, and write every fraction as ``\\frac``."""
    latex = _TEXT_WRAPPER.sub(r"\1", latex)
    for _ in range(2):  # 1,000,000 needs two passes
        latex = re.sub(r"(\d),\\!(\d{3})", r"\1\2", latex)
        latex = re.sub(r"(\d)\{,\}(\d{3})", r"\1\2", latex)
    return latex.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").strip()


def _letters(latex: str) -> frozenset[str] | None:
    """E2: the set of choice letters, when the text is only a list of letters such as ``(A), (C)``."""
    text = _plain(latex).replace("\\ ", " ").strip().strip("$")
    return frozenset(re.findall(r"[A-Z]", text)) if _LETTER_LIST.match(text.replace(" ", "")) else None


def _options(problem: str) -> dict[str, str]:
    """E3: the multiple-choice options of a problem, by letter."""
    options = {match[1]: match[2].strip().rstrip("$").strip() for match in _OPTION_BOLD.finditer(problem)}
    return options or {match[1]: match[2].strip() for match in _OPTION_TEXT.finditer(problem)}


def _columns_as_tuples(latex: str) -> str:
    """E4: write each one-column matrix as a tuple."""

    def as_tuple(match: re.Match[str]) -> str:
        rows = [row.strip() for row in match[1].split("\\\\")]
        return match[0] if any("&" in row for row in rows) else "(" + ", ".join(rows) + ")"

    return _COLUMN.sub(as_tuple, latex)


def _interval(latex: str) -> str | None:
    """E5: write a one-variable inequality as an interval."""
    text = _plain(latex).replace(" ", "")
    if match := re.fullmatch(r"[a-z](?:\\neq|\\ne)" + _NUMBER, text):
        return f"(-\\infty,{match[1]})\\cup({match[1]},\\infty)"
    if match := re.fullmatch(_NUMBER + "<[a-z]<" + _NUMBER, text):
        return f"({match[1]},{match[2]})"
    # The order of the alternatives matters: v1 tries "\le" before "\leq", and the grades must stay equal to v1.
    if match := re.fullmatch(r"[a-z](<|>|\\le|\\ge|\\leq|\\geq)" + _NUMBER, text):
        return _BOUND[match[1]].format(match[2])
    return None


def _scalar_matrix(answer: str, gold: str) -> str | None:
    """E6: write ``k I`` as the diagonal matrix of the same size as the reference."""
    scalar = re.fullmatch(r"(-?\d+)\s*\\?(?:mathbf\{I\}|I)", _plain(answer).replace(" ", ""))
    matrix = re.fullmatch(r"\\begin\{pmatrix\}(.*)\\end\{pmatrix\}", gold.strip(), re.S)
    if not (scalar and matrix):
        return None
    size = len(matrix[1].split("\\\\"))
    rows = [" & ".join(scalar[1] if row == column else "0" for column in range(size)) for row in range(size)]
    return "\\begin{pmatrix} " + " \\\\ ".join(rows) + " \\end{pmatrix}"


def _complex_equal(answer: str, gold: str) -> bool:
    """E7: equal as complex numbers, with ``i`` read as the imaginary unit."""
    try:
        unit = sympy.Symbol("i")
        left = latex2sympy(_plain(answer)).subs(unit, sympy.I)
        right = latex2sympy(_plain(gold)).subs(unit, sympy.I)
        return sympy.simplify(sympy.expand(left - right)) == 0
    except Exception:
        return False


def _equal_rule(gold: str, box: str, problem: str) -> str | None:
    """The rule under which ``box`` equals ``gold``: ``""`` for math-verify, ``E1`` to ``E7``, or ``None``."""
    if _same(gold, box):
        return ""
    answer = box[len(BOX) : -1]
    if _same(_plain(gold), BOX + _plain(answer) + "}"):
        return "E1"
    gold_letters, letters = _letters(gold), _letters(answer)
    if gold_letters and letters and gold_letters == letters:
        return "E2"
    if options := _options(problem):
        choice = _plain(answer)
        if re.match(r"^\s*\\?(?:text|textbf)?\{?\(?[A-E]\)", choice):
            choice = _OPTION_LABEL.sub("", choice)
        if letters and len(letters) == 1:
            choice = options.get(next(iter(letters)), choice)
        if choice and _same(gold, BOX + choice + "}"):
            return "E3"
    if _same(_columns_as_tuples(gold), BOX + _columns_as_tuples(answer) + "}"):
        return "E4"
    interval = _interval(answer)
    if interval and _same(gold, BOX + interval + "}"):
        return "E5"
    matrix = _scalar_matrix(answer, gold)
    if matrix and _same(gold, BOX + matrix + "}"):
        return "E6"
    if "i" in gold and "i" in answer and _complex_equal(answer, gold):
        return "E7"
    return None


# ---- the grade ----


def grade_answer(gold: str, problem: str, output: str, capped: bool) -> Grade:
    """Grade ``output`` against the reference answer ``gold`` of ``problem``. See ``grade``."""
    if capped:
        return _fail(CAP, "capped")
    if threading.current_thread() is not threading.main_thread():
        # In any other thread math-verify cannot set its alarm, and every grade would look like a stable ``E``.
        raise InfraError("the math grader must run in the main thread of a process; use GradePool")
    try:
        _runtime_checked()
        box, reason = _strict_box(output)
        strict = box if not any(token in output for token in CONTROL_TOKENS) else None
        if not _reference_parses(gold):
            return _fail(ERROR, "reference-unparseable", strict)
        if strict is not None and _same(gold, strict) is None:
            return _fail(ERROR, "grader-error", strict)
        found_by = ""
        if box is None:
            box, found_by = _lenient_box(output, reason)
        equal_by = _equal_rule(gold, box, problem) if box is not None else None
        if equal_by is not None:
            return Grade(PASS, 1.0, False, box, "+".join(rule for rule in (found_by, equal_by) if rule) or "live")
        if strict is not None:
            return _fail(FAIL, "wrong", strict)
        return _fail(FAIL, "unresolved:" + ("control-token" if reason == "ok" else reason), box)
    except TimeoutException:  # not a subclass of Exception: math-verify ran out of time
        return _fail(ERROR, "grader-timeout", machine_cause=TIME_LIMIT)


def grade(task: Task, output: str, capped: bool) -> Grade:
    """Grade one output for one task.

    ``output`` is the text the model wrote: decoded with special tokens kept
    (so ``<think>`` and ``</think>`` are in it), without the final
    end-of-sequence token. ``capped`` is true when the model stopped at the
    token cap and not at an end of sequence.

    Call it in the main thread of a process (math-verify uses an alarm signal
    for its time limits), or use ``GradePool``.
    """
    return grade_answer(task.answer, task.problem, output, capped)


# ---- the supervisor ----


def _serve(link: Connection, job: Callable[..., Grade]) -> None:
    """Worker process: grade jobs until the parent closes the link."""
    while True:
        try:
            arguments = link.recv()
        except EOFError:
            return
        link.send(job(*arguments))


class _Worker:
    def __init__(self, context: multiprocessing.context.BaseContext, job: Callable[..., Grade]) -> None:
        self.link, far_end = context.Pipe()
        self.process = context.Process(target=_serve, args=(far_end, job), daemon=True)
        self.process.start()
        far_end.close()
        self.index: int | None = None  # the job in progress
        self.deadline = 0.0

    def stop(self) -> None:
        self.link.close()
        self.process.kill()  # kills exactly this process id
        self.process.join()


class GradePool:
    """Grade in ``workers`` processes, each job under a hard limit of ``timeout_s`` seconds.

    A job that passes the limit has its worker killed and gets code ``E``
    with ``machine_cause`` ``TIME_LIMIT``. A job whose worker dies gets code
    ``E`` with ``machine_cause`` ``WORKER_DIED``. Neither is retried, so
    neither can become a pass.
    A fresh worker replaces the lost one. ``job`` is the function to run; the
    default is ``grade``.
    """

    def __init__(self, workers: int = 8, timeout_s: float = 300.0, job: Callable[..., Grade] = grade) -> None:
        check_runtime()
        self.timeout_s = timeout_s
        self._job = job
        self._context = multiprocessing.get_context("spawn")
        self._workers = [_Worker(self._context, job) for _ in range(max(1, workers))]

    def grade_many(self, jobs: Sequence[tuple[Task, str, bool]]) -> list[Grade]:
        """Grade every ``(task, output, capped)`` job. The result list has the order of ``jobs``."""
        grades: list[Grade | None] = [None] * len(jobs)
        waiting = deque(range(len(jobs)))
        while True:
            for worker in list(self._workers):
                if worker.index is None and waiting:
                    worker.index = waiting.popleft()
                    worker.deadline = time.monotonic() + self.timeout_s
                    try:
                        worker.link.send(jobs[worker.index])
                    except OSError:  # the worker died while it was idle
                        grades[worker.index] = self._replace(worker, "grade-worker-died", WORKER_DIED)
            busy = [worker for worker in self._workers if worker.index is not None]
            if not busy and waiting:
                continue
            if not busy:
                return [grade for grade in grades if grade is not None]
            ready = wait([worker.link for worker in busy], max(0.0, min(w.deadline for w in busy) - time.monotonic()))
            for worker in busy:
                if worker.link in ready:
                    try:
                        grades[worker.index], worker.index = worker.link.recv(), None
                    except (EOFError, OSError):
                        grades[worker.index] = self._replace(worker, "grade-worker-died", WORKER_DIED)
                elif time.monotonic() >= worker.deadline:
                    grades[worker.index] = self._replace(worker, "grade-timeout", TIME_LIMIT)

    def _replace(self, worker: _Worker, rule: str, machine_cause: str) -> Grade:
        worker.stop()
        self._workers[self._workers.index(worker)] = _Worker(self._context, self._job)
        return _fail(ERROR, rule, machine_cause=machine_cause)

    def close(self) -> None:
        for worker in self._workers:
            worker.stop()
        self._workers = []
