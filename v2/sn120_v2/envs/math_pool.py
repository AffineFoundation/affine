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
POOL_FILE = Path(__file__).with_name("math_pool_v2.jsonl")
POOL_SHA256 = "8d34d276bd777c2b48506687ee902041e40f68073ce71cf60f79b27471d06ebc"

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


def load_pool(path: Path = POOL_FILE, sha256: str = POOL_SHA256) -> MathPool:
    """Read and verify the pool. Raises ``VerifyError`` unless the bytes of the file hash to ``sha256``."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise VerifyError(f"the math pool file {path} cannot be read: {error}") from error
    found = sha256_hex(data)
    if found != sha256:
        raise VerifyError(f"the math pool file {path} has sha256 {found}, expected {sha256}")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    return MathPool(tuple(Task(row["id"], row["env"], row["problem"], row["answer"]) for row in rows), found)
