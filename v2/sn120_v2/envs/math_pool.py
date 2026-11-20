"""The public math task pool and the prompt of a task. Standard library only.

The pool is one JSONL file that ships inside this package. Its bytes are
pinned by ``POOL_SHA256``; a file with other bytes is refused. Task ids are
strings: ``math:<n>`` (slice ``affine_math``) and ``dapo:<n>`` (slice
``dapo``). The two slices are one environment and are scored together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..canonical import sha256_hex
from ..errors import ContractError, VerifyError

ENV_ID = "math"
POOL_FILE = Path(__file__).with_name("math_pool_v2_1.jsonl")
POOL_SHA256 = "9c3010317c403da0f013e15d9420cab2c241d388f3325f3de8f0ad2cb5282a0a"
# v2.1 is a drop-only change: the archive keeps the exact v2.0 bytes for replay of old rounds.
POOL_V2_0_FILE = Path(__file__).parent / "archive" / "math_pool_v2.jsonl"
POOL_V2_0_SHA256 = "8d34d276bd777c2b48506687ee902041e40f68073ce71cf60f79b27471d06ebc"
POOL_V2_1_FILE, POOL_V2_1_SHA256 = POOL_FILE, POOL_SHA256

# The instruction every prompt carries. Scoring sends it as the system message.
MANDATE = (
    "Solve the math problem. Reason step by step in plain text, then end your response with the final answer "
    "in `\\boxed{}`. Emit exactly one `\\boxed{}` block and nothing after it."
)


@dataclass(frozen=True, slots=True)
class Task:
    """One public task. ``answer`` is the reference answer: the bare content of its ``\\boxed{}``."""

    task_id: str
    slice: str
    problem: str
    answer: str

    def messages(self) -> list[dict[str, str]]:
        """The scoring prompt, as chat messages: system ``MANDATE``, then the problem as the user."""
        return [{"role": "system", "content": MANDATE}, {"role": "user", "content": self.problem}]

    def batch_prompt(self) -> str:
        """The prompt of a miner batch row: one user message, ``MANDATE``, a blank line, the problem."""
        return MANDATE + "\n\n" + self.problem


class MathPool:
    """The verified pool. ``task_ids`` keeps the order of the file."""

    def __init__(self, tasks: tuple[Task, ...], sha256: str) -> None:
        self.sha256 = sha256
        self.task_ids: tuple[str, ...] = tuple(task.task_id for task in tasks)
        self._by_id = {task.task_id: task for task in tasks}
        self._by_problem = {task.problem: task for task in tasks}
        if len(self._by_id) != len(tasks) or len(self._by_problem) != len(tasks):
            raise VerifyError("the math pool repeats a task id or a problem")

    def __len__(self) -> int:
        return len(self.task_ids)

    def task(self, task_id: str) -> Task:
        try:
            return self._by_id[task_id]
        except KeyError:
            raise ContractError(f"task {task_id!r} is not in the math pool") from None

    def task_for_prompt(self, prompt: str) -> Task | None:
        """The task a miner batch prompt belongs to, or ``None`` when the prompt is not a pool problem.

        A batch prompt is the problem text, with or without ``MANDATE`` and a blank line before it.
        """
        return self._by_problem.get(prompt.removeprefix(MANDATE + "\n\n"))


def load_pool(path: Path | None = None, sha256: str | None = None) -> MathPool:
    """Read and verify the current pool, or an explicit path and SHA for an archived pool.

    ``path`` without ``sha256`` keeps the historical form: it must be a file
    with the current pool's exact bytes.
    """
    path = POOL_FILE if path is None else Path(path)
    expected = POOL_SHA256 if sha256 is None else sha256
    try:
        data = path.read_bytes()
    except OSError as error:
        raise VerifyError(f"the math pool file {path} cannot be read: {error}") from error
    found = sha256_hex(data)
    if found != expected:
        raise VerifyError(f"the math pool file {path} has sha256 {found}, expected {expected}")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    return MathPool(tuple(Task(row["id"], row["env"], row["problem"], row["answer"]) for row in rows), found)


def pool_for_draw(task_ids: tuple[str, ...], pool: str = "auto") -> MathPool:
    """The pool for a round's draw: v2.1 normally, v2.0 when its draw needs a removed task.

    ``pool`` is ``auto``, ``current`` or ``v2.0``. The last two force one of
    the two shipped, SHA-pinned files. An unknown task is refused by ``task``.
    """
    if pool not in ("auto", "current", "v2.0"):
        raise ValueError(f"unknown pool {pool!r}")
    if pool == "v2.0":
        return load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)
    current = load_pool()
    missing = set(task_ids) - set(current.task_ids)
    if pool == "auto" and missing:
        return load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)
    if missing:
        raise ContractError(f"the draw has a task that is not in the current math pool: {sorted(missing)[0]}")
    return current
