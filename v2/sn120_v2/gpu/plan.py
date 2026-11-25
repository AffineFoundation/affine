"""Planning and pins for one box. Pure Python: no GPU and no torch."""

from __future__ import annotations

import json
import platform
import threading
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from ..canonical import digest, file_sha256, manifest_digest
from ..errors import InfraError
from ..params import Params, require_env


@dataclass(frozen=True)
class BoxConfig:
    """What a box must be told. There is no default: a missing value stops the worker."""

    work_dir: Path  # plain local disk; every weight, LoRA and job file of this box lives below it
    image_digest: str  # digest of the baked worker image; it goes into every report and chunk

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> BoxConfig:
        return cls(Path(require_env("SN120_WORK_DIR", environ)), require_env("SN120_IMAGE_DIGEST", environ))


def code_digest() -> str:
    """Identity of the training and scoring code: the manifest digest of this package's source files."""
    return manifest_digest({path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))})


def install_digest() -> str:
    """Identity of what is installed on this box: the Python version and every installed package with its version.

    A box that was not started from a baked image uses this value as its
    image digest. Two boxes with the same install line and the same lock
    give the same value.
    """
    packages = {dist.metadata["Name"].lower(): dist.version for dist in metadata.distributions() if dist.metadata["Name"]}
    return digest({"python": platform.python_version(), "packages": dict(sorted(packages.items()))})


_LOG_LOCK = threading.Lock()


def log_phase(work_dir: Path, phase: str, **facts: Any) -> None:
    """Append one line to ``<work_dir>/logs/phases.jsonl``: what a phase did and how long it took.

    Wall times live here and in the records, never in a hashed file. No
    key, link or miner text goes into this log.
    """
    line = json.dumps({"at": round(time.time(), 3), "phase": phase, **facts}, sort_keys=True)
    with _LOG_LOCK:
        (Path(work_dir) / "logs").mkdir(parents=True, exist_ok=True)
        with (Path(work_dir) / "logs" / "phases.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


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
    print(json.dumps({"code_digest": code_digest(), "image_digest": install_digest()}))
