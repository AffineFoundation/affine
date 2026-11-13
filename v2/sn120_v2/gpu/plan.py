"""Planning and pins for one box. Pure Python: no GPU and no torch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..canonical import file_sha256, manifest_digest
from ..errors import ContractError, InfraError
from ..params import Params, require_env


@dataclass(frozen=True)
class BoxConfig:
    """What a box must be told. There is no default: a missing value stops the worker."""

    work_dir: Path  # plain local disk; every weight, LoRA and job file of this box lives below it
    image_digest: str  # digest of the baked worker image; it goes into every report and chunk

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> BoxConfig:
        return cls(Path(require_env("SN120_WORK_DIR", environ)), require_env("SN120_IMAGE_DIGEST", environ))


def train_waves(n_batches: int, params: Params) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """The GPU ids of every batch of a round, in queue order, grouped in waves.

    Each LoRA gets floor(n_gpus / batches) GPUs, and at least one. The LoRAs
    of one wave hold disjoint GPUs and train at the same time. With more
    batches than GPUs, the waves run one after the other. A round above the
    batch cap is refused.
    """
    if not 1 <= n_batches <= params.batch_cap:
        raise ContractError(f"a round trains 1 to {params.batch_cap} batches, not {n_batches}")
    per_lora = params.gpus_per_lora(n_batches)
    at_once = params.n_gpus // per_lora
    waves = []
    for first in range(0, n_batches, at_once):
        slots = range(min(at_once, n_batches - first))
        waves.append(tuple(tuple(range(slot * per_lora, (slot + 1) * per_lora)) for slot in slots))
    return tuple(waves)


def code_digest() -> str:
    """Identity of the training and scoring code: the manifest digest of this package's source files."""
    return manifest_digest({path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))})


def check_pins(image_digest: str, params: Params) -> None:
    """Refuse to work from an image or code that the round did not pin.

    A wrong pin is a failure of this box, never of a miner, so it is an
    ``InfraError``: the entries go back to the queue and nothing stalls.
    """
    recipe = params.recipe
    if image_digest not in recipe.image_digests:
        raise InfraError(f"worker image {image_digest[:16]} is not in the pinned set")
    if recipe.code_digest and recipe.code_digest != code_digest():
        raise InfraError("the training code on this box differs from recipe.code_digest")


if __name__ == "__main__":
    print(code_digest())
