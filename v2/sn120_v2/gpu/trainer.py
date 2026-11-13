"""The ``Trainer`` seam: one bounded GRPO LoRA from one frozen batch.

This module holds every check and no torch. The GPU work runs in a child
process (``train_job.py``) that sees only the GPUs of its request. When the
child is gone, its GPUs are free: that is what ``close`` relies on.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from ..canonical import manifest_digest, sha256_hex
from ..errors import BatchRejected, InfraError, VerifyError
from ..params import Params, Recipe
from ..seams import ObjectStore
from ..types import LoRAArtifact, TrainReport, TrainRequest
from .batch import Groups, training_rows
from .plan import BoxConfig, check_pins
from .procs import Child, Process, Spawn, gpu_env, stop_all
from .resident import Resident, resident_head, upload_tree

LORA_FILES = ("adapter_config.json", "adapter_model.safetensors", "train_report.json")

# (frozen bytes, hotkey, params) -> the kept groups. The admission module owns
# the file rules: pass ``lambda raw, hotkey, params: check_batch(raw, hotkey, params).groups``.
ReadGroups = Callable[[bytes, str, Params], Groups]


class LoraTrainer:
    def __init__(self, store: ObjectStore, config: BoxConfig, read_groups: ReadGroups, spawn: Spawn = Child):
        self.store, self.config, self.read_groups, self.spawn = store, config, read_groups, spawn
        self.resident = Resident(store, config.work_dir)
        self.jobs = Path(config.work_dir) / "train"
        self._procs: list[Process] = []
        self._lock = threading.Lock()

    def train(self, request: TrainRequest) -> LoRAArtifact:
        params, head, entry = request.params, request.head, request.entry
        check_pins(self.config.image_digest, params)
        if not request.gpu_ids:
            raise InfraError("a training request needs at least one GPU")
        recipe = params.recipe.for_miner(entry.batch.train)
        if recipe.method != "grpo-lora" or recipe.dropout or recipe.kl:
            raise InfraError("this trainer runs grpo-lora with dropout 0 and KL off only")
        if recipe.group_size:
            # Not decided yet: the trainer never guesses what a miner's value means, and never ignores it.
            raise BatchRejected("train.group_size is not supported yet: each prompt's rollouts are one group")
        self.resident.prune({head.base_hash, head.base_fp32_master_digest})
        base_dir = resident_head(self.resident, head)
        rows = self._rows(request)
        job_dir = self.jobs / f"{request.round_id}-{entry.queue_pos:03d}"
        shutil.rmtree(job_dir, ignore_errors=True)
        (job_dir / "lora").mkdir(parents=True)
        try:
            job = {"base_dir": str(base_dir), "base_hash": head.base_hash, "dtype": params.dtype}
            job.update(recipe=recipe.to_dict(), seed=request.seed, rows=rows)
            (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
            started = time.monotonic()
            report = self._check(self._run(job_dir, request), recipe)
            files, sizes = upload_tree(self.store, job_dir / "lora", request.out_prefix)
            if sorted(files) != sorted(LORA_FILES):
                raise InfraError(f"the training job wrote {sorted(files)}, not the LoRA files")
            return LoRAArtifact(
                lora_hash=manifest_digest(files),
                files=files,
                sizes=sizes,
                prefix=request.out_prefix,
                head_version_id=head.version_id,
                batch_sha256=entry.batch.batch_sha256,
                recipe_id=params.recipe_id,
                rank=recipe.rank,
                alpha=recipe.alpha,
                target_modules=recipe.target_modules,
                report=TrainReport(
                    seed=request.seed,
                    steps=recipe.steps,
                    loss=report["loss"],
                    step_norm=report["step_norm"],
                    tokens=report["tokens"],
                    wall_s=time.monotonic() - started,
                    gpu_class=report["gpu_class"],
                    image_digest=self.config.image_digest,
                ),
            )
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)  # a LoRA lives in the store; the box keeps no copy

    def close(self) -> None:
        with self._lock:
            procs, self._procs = self._procs, []
        stop_all(procs)

    def _rows(self, request: TrainRequest) -> list[dict]:
        batch = request.entry.batch
        try:
            raw = self.store.get(batch.uri)
        except KeyError as exc:
            raise InfraError(f"the frozen batch {batch.uri} is missing") from exc
        if sha256_hex(raw) != batch.batch_sha256:
            raise VerifyError("the frozen batch does not hash to batch_sha256")
        try:
            groups = self.read_groups(raw, batch.hotkey, request.params)
        except ValueError as exc:
            raise BatchRejected(f"the frozen batch is not valid: {exc}") from exc
        return training_rows(groups)

    def _run(self, job_dir: Path, request: TrainRequest) -> dict:
        argv = [sys.executable, "-m", "sn120_v2.gpu.train_job", str(job_dir)]
        proc = self.spawn(argv, gpu_env(request.gpu_ids), job_dir / "job.log")
        with self._lock:
            self._procs.append(proc)
        code = proc.wait(float(request.params.lease_timeout_s))
        with self._lock:
            if proc in self._procs:
                self._procs.remove(proc)
        if code is None:
            proc.stop()
            raise InfraError("the training job ran past the lease time")
        try:
            report = json.loads((job_dir / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise InfraError(f"the training job ended with code {code} and no report") from exc
        if code != 0 or not isinstance(report, dict):
            raise InfraError(f"the training job failed with code {code}")
        return report

    @staticmethod
    def _check(report: dict, recipe: Recipe) -> dict:
        if report.get("status") == "rejected":
            raise BatchRejected(str(report.get("reason", "the batch cannot be trained")))
        numbers = [report.get(key) for key in ("loss", "step_norm")]
        if report.get("status") != "ok" or not all(type(n) in (int, float) for n in numbers):
            raise InfraError("the training job wrote a report that cannot be read")
        if not all(math.isfinite(n) for n in numbers):
            raise BatchRejected("training gave a loss or an update that is not finite")
        cap = recipe.global_norm_cap
        if cap is not None and report["step_norm"] > cap * (1 + 1e-6):
            raise InfraError("the adapter update is larger than recipe.global_norm_cap")
        return report
