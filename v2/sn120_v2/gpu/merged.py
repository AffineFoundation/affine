"""Build the base a candidate will be scored and installed as.

The only rounding is the final cast of the fp32 master plus the LoRA update.
The build calls the same ``fold_kernel.replay_fold`` as the install path, with
its named kernel; it writes no second fp32 master. The result's manifest hash
is the proof that winner promotion checks before publishing a folded head.
"""

from __future__ import annotations

import dataclasses
import shutil
import threading
from pathlib import Path

from ..canonical import is_sha256, manifest_digest
from ..errors import InfraError
from ..headchain import FIXED_ORDER, KERNELS, MATMUL
from .resident import hash_tree


@dataclasses.dataclass(frozen=True)
class TestedBase:
    """A local merged base and the proof that travels in its EntryResult."""

    path: Path
    base_hash: str
    base_files: dict[str, str]
    kernel: str


_MATMUL_LOCK = threading.Lock()


def build_candidate_base(
    master_dir: Path,
    base_dir: Path,
    lora_dir: Path,
    rank: int,
    out_base: Path,
    *,
    kernel: str = FIXED_ORDER,
    device: str = "cpu",
    threads: int | None = None,
    reserved: int = 0,
) -> TestedBase:
    """Write only ``bf16(master + scale*B@A)`` (or the head's base dtype).

    ``reserved`` is the space concurrently reserved by other candidate builds
    in this process. A full base must fit in addition to those reservations.
    An existing output is never served: a failed or partial build is removed.
    """
    from .. import fold_kernel

    master_dir, base_dir, lora_dir, out_base = Path(master_dir), Path(base_dir), Path(lora_dir), Path(out_base)
    if kernel not in KERNELS:
        raise InfraError(f"the candidate cannot use unknown install kernel {kernel!r}")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    need = sum(file.stat().st_size for file in base_dir.rglob("*") if file.is_file())
    free = shutil.disk_usage(out_base.parent).free
    if free < need + reserved:
        raise InfraError(f"not enough free disk for merged candidate base: {free} bytes are free, {need} are needed and {reserved} are reserved for concurrent builds")
    shutil.rmtree(out_base, ignore_errors=True)
    try:
        lora = fold_kernel.read_pairs(lora_dir, rank)
        if kernel == MATMUL and threads:
            # headfold._fold_in_child sets this same count for the install fold.
            # Serialize the process-global setting when several trainers finish together.
            import torch

            with _MATMUL_LOCK:
                before = torch.get_num_threads()
                torch.set_num_threads(threads)
                try:
                    folded = fold_kernel.replay_fold(master_dir, base_dir, [lora], None, out_base, kernel=kernel, threads=threads)
                finally:
                    torch.set_num_threads(before)
        else:
            folded = fold_kernel.replay_fold(master_dir, base_dir, [lora], None, out_base, kernel=kernel, device=device, threads=threads)
        files = hash_tree(out_base)
        if files != folded.base_files or manifest_digest(files) != folded.base_hash:
            raise InfraError("the merged candidate files differ from their fold hashes")
        return TestedBase(out_base, folded.base_hash, files, kernel)
    except Exception:
        shutil.rmtree(out_base, ignore_errors=True)
        raise


def candidate_fold_settings() -> tuple[str, int | None]:
    """The kernel and torch thread count that the production install fold uses."""
    from ..headchain import CHAIN, mode

    if mode() != CHAIN:
        return MATMUL, None
    from ..headfold import FoldSettings

    settings = FoldSettings.from_env()
    # in_children gives every fold an equal share of this process's threads.
    import torch

    threads = settings.threads or max(1, torch.get_num_threads() // (2 if settings.twice else 1))
    return FIXED_ORDER, threads


def assert_tested_result(params, entry) -> None:
    """A scored result in merged mode must name a real kernel and base manifest."""
    enabled = params.eval_merged
    if entry.status != "scored":
        if entry.tested_base_hash or entry.tested_fold_kernel:
            raise InfraError("an unscored entry cannot carry a tested-base proof")
        return
    if not enabled:
        if entry.tested_base_hash or entry.tested_fold_kernel:
            raise InfraError("a runtime-LoRA result cannot carry a tested-base proof")
        return
    if not is_sha256(entry.tested_base_hash) or entry.tested_fold_kernel not in KERNELS:
        raise InfraError("merged evaluation needs the tested base hash and its install fold kernel")


def assert_fold_matches(params, entry, base_hash: str) -> None:
    """Fail closed before any fold that differs from the tested base is installed."""
    assert_tested_result(params, entry)
    if params.eval_merged and base_hash != entry.tested_base_hash:
        raise InfraError(f"the install fold base hash {base_hash} differs from the winner's tested base hash {entry.tested_base_hash}; nothing is installed")
