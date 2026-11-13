"""Seeds and the task draw. Pure functions of public inputs.

Everything here can be recomputed by anyone from the published round seed
and snapshot, with nothing but SHA-256.

``UniformSampler`` is a placeholder for the IRT sampler: it draws every task
of the pool with equal probability. It follows the same ``TaskSampler`` seam,
so the IRT sampler replaces it without a change to the controller.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .canonical import digest, sha256_hex
from .errors import ContractError
from .types import Draw, HeadVersion

SEED_BITS = 63


def make_round_seed(round_id: str, head_version_id: str, entropy: str) -> str:
    """The public seed of a round. ``entropy`` is a value the validator cannot choose, such as a block hash."""
    if not entropy:
        raise ContractError("a round seed needs entropy")
    return digest({"round_id": round_id, "head_version_id": head_version_id, "entropy": entropy})


def rollout_seed(round_seed: str, task_id: str, sample: int) -> int:
    """Seed of one rollout: sha256("<round_seed>|<task_id>|<sample>") as a 63-bit integer.

    Null M and every LoRA use this same seed for the same task and sample.
    """
    text = f"{round_seed}|{task_id}|{sample}"
    return int(sha256_hex(text.encode("utf-8")), 16) >> (256 - SEED_BITS)


def train_seed(round_seed: str, batch_sha256: str) -> int:
    """Seed of one training job. It is part of the recovery key (head, batch, recipe, seed)."""
    return int(sha256_hex(f"{round_seed}|train|{batch_sha256}".encode("utf-8")), 16) >> (256 - SEED_BITS)


def draw_rank(round_seed: str, task_id: str) -> str:
    """Sort key that puts tasks in a seed-dependent random order."""
    return sha256_hex(f"{round_seed}|draw|{task_id}".encode("utf-8"))


def uniform_draw(pool: Sequence[str], snapshot_id: str, round_seed: str, n_tasks: int) -> Draw:
    """Take the ``n_tasks`` tasks with the smallest ``draw_rank``. Each task has the same chance.

    The tasks come out in rank order, so every look is a random part of the
    draw. Each task has inclusion probability n / pool size, so its sampling
    weight is pool size / n.
    """
    if n_tasks < 1 or not pool:
        raise ContractError("a draw needs a pool and at least one task")
    chosen = sorted(pool, key=lambda task_id: draw_rank(round_seed, task_id))[: min(n_tasks, len(pool))]
    weight = len(pool) / len(chosen)
    return Draw(snapshot_id, round_seed, tuple(chosen), (weight,) * len(chosen))


class UniformSampler:
    """Placeholder ``TaskSampler``: equal-probability draw from a fixed pool.

    It keeps no table, so its snapshot never changes and ``update`` does
    nothing. The IRT sampler will learn from each round's outcomes here.
    """

    def __init__(self, pool: Sequence[str]) -> None:
        if len(set(pool)) != len(pool) or not pool:
            raise ContractError("the task pool must be non-empty with no repeated task")
        self.pool = tuple(sorted(pool))

    def snapshot_id(self) -> str:
        return "uniform-" + digest({"pool": self.pool})[:16]

    def draw(self, head: HeadVersion, round_seed: str, n_tasks: int) -> Draw:
        return uniform_draw(self.pool, self.snapshot_id(), round_seed, n_tasks)

    def update(
        self,
        head: HeadVersion,
        draw: Draw,
        outcomes: Mapping[str, Sequence[str]],
        *,
        winner_lora_hash: str = "",
        new_head_version_id: str = "",
    ) -> str:
        return self.snapshot_id()

    def public_snapshot(self) -> Mapping[str, Any]:
        return {
            "snapshot_id": self.snapshot_id(),
            "method": "uniform placeholder: every pool task has the same chance",
            "pool_size": len(self.pool),
        }
