"""CPU stand-ins for every seam, and a small world that runs rounds end to end.

No GPU, no network, no chain. A simulated batch is a tiny JSON file that
states its own ``quality``: how much the LoRA trained from it changes the
pass rate. The simulated evaluator gives null M and every LoRA the same
random number for the same task and sample, as the real seeds do, so the
comparison is paired.

Use ``SimWorld`` in tests of your own part, and ``python -m sn120_v2 sim``
for a full run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import layout
from .canonical import canonical_json, manifest_digest, sha256_hex
from .controller import Controller, Stack, StackContext, stored_irt_snapshot
from .errors import BatchRejected, InfraError
from .journal import JOURNAL_FILE, Journal
from .params import GenesisSpec, Params, Recipe
from .sampling import UniformSampler, rollout_seed
from .state import RoundRecord
from .store import LocalStore
from .types import (
    CAP,
    FAIL,
    NULL_MODEL,
    PASS,
    ChainSnapshot,
    EvalChunk,
    EvalRequest,
    FoldArtifact,
    FrozenBatch,
    HeadVersion,
    LoRAArtifact,
    Registration,
    TrainReport,
    TrainRequest,
    V1Import,
    Verdict,
)
from .worker import LocalRunner

SIM_IMAGE = "sim-image"
STORE_DIR = "store"


def sim_params(**changes: Any) -> Params:
    """Small parameters for fast CPU rounds."""
    base = dict(n_tasks=64, min_scored_tasks=16, batch_cap=4, looks=(0.5, 1.0), recipe=Recipe(image_digests=(SIM_IMAGE,)))
    return Params(**(base | changes))


def _unit(*parts: object) -> float:
    """A fixed number in [0, 1) for any label."""
    return int(sha256_hex("|".join(map(str, parts)).encode())[:13], 16) / 16**13


class SimAdmission:
    """Miner files in the store. A file is valid when its JSON says so."""

    def __init__(self, store: LocalStore) -> None:
        self.store = store

    def upload(self, hotkey: str, quality: float = 0.0, train: Mapping[str, Any] | None = None, valid: bool = True, cap_rate: float = 0.0) -> None:
        body = {"hotkey": hotkey, "quality": quality, "cap_rate": cap_rate, "train": dict(train or {}), "valid": valid}
        self.store.put(layout.miner_object(hotkey), canonical_json(body))

    def _read(self, hotkey: str) -> tuple[bytes, dict] | None:
        key = layout.miner_object(hotkey)
        if not self.store.exists(key):
            return None
        raw = self.store.get(key)
        return raw, json.loads(raw)

    def verdicts(self, hotkeys: Sequence[str]) -> dict[str, Verdict]:
        found = {hotkey: self._read(hotkey) for hotkey in hotkeys}
        rule = lambda body: "" if body["valid"] else "sim_invalid"
        return {h: Verdict(h, bool(f[1]["valid"]), batch_sha256=sha256_hex(f[0]), rule=rule(f[1])) for h, f in found.items() if f}

    def freeze(self, reg: Registration, round_id: str) -> FrozenBatch | None:
        found = self._read(reg.hotkey)
        if not found or not found[1]["valid"]:
            return None
        key = layout.frozen_batch(round_id, reg.hotkey)
        self.store.put(key, found[0])
        return FrozenBatch(reg.hotkey, sha256_hex(found[0]), key, "sim-batch", 1, 1, found[1]["train"])


class SimTrainer:
    """Writes a tiny fake LoRA that remembers the quality of its batch."""

    def __init__(self, store: LocalStore) -> None:
        self.store = store
        self.fail_hotkeys: set[str] = set()  # training fails for these, as an infrastructure failure
        self.reject_hotkeys: set[str] = set()  # the batch is rejected at train time
        self.closed = 0

    def train(self, request: TrainRequest) -> LoRAArtifact:
        entry, params = request.entry, request.params
        if request.head.loras:
            raise InfraError("this trainer needs a folded head")
        if entry.reg.hotkey in self.fail_hotkeys:
            raise InfraError("simulated trainer crash")
        if entry.reg.hotkey in self.reject_hotkeys:
            raise BatchRejected("simulated bad batch")
        batch = json.loads(self.store.get(entry.batch.uri))
        recipe = params.recipe.for_miner(entry.batch.train)
        blobs = {
            "adapter_config.json": canonical_json({"quality": batch["quality"], "cap_rate": batch["cap_rate"], "rank": recipe.rank}),
            "adapter_model.bin": f"{request.head.version_id}|{entry.batch.batch_sha256}|{request.seed}".encode(),
        }
        for name, blob in blobs.items():
            self.store.put(f"{request.out_prefix}/{name}", blob)
        files = {name: sha256_hex(blob) for name, blob in blobs.items()}
        return LoRAArtifact(
            lora_hash=manifest_digest(files),
            files=files,
            sizes={name: len(blob) for name, blob in blobs.items()},
            prefix=request.out_prefix,
            head_version_id=request.head.version_id,
            batch_sha256=entry.batch.batch_sha256,
            recipe_id=params.recipe_id,
            rank=recipe.rank,
            alpha=recipe.alpha,
            target_modules=recipe.target_modules,
            report=TrainReport(request.seed, recipe.steps, 0.5, 0.01, 1000, 0.1, "cpu", SIM_IMAGE),
        )

    def close(self) -> None:
        self.closed += 1


class SimEvaluator:
    """Scores null M and LoRAs with shared random numbers per task and sample."""

    def __init__(self, store: LocalStore) -> None:
        self.store = store
        self.fail_looks: set[int] = set()  # every call for these looks fails
        self.closed = 0

    def evaluate(self, request: EvalRequest) -> EvalChunk:
        if request.look in self.fail_looks:
            raise InfraError("simulated evaluator crash")
        models = {NULL_MODEL: {"quality": 0.0, "cap_rate": 0.0}}
        for lora in request.loras:
            models[lora.lora_hash] = json.loads(self.store.get(f"{lora.prefix}/adapter_config.json"))
        outcomes = {}
        for model, effect in models.items():
            strings = []
            for sample in range(request.params.samples_per_task):
                codes = []
                for task_id in request.task_ids:
                    luck = rollout_seed(request.round_seed, task_id, sample) / 2**63
                    base = 0.25 + 0.5 * _unit("difficulty", task_id)
                    capped = _unit("cap", model, task_id, sample) < effect["cap_rate"]
                    codes.append(CAP if capped else PASS if luck < base + effect["quality"] else FAIL)
                strings.append("".join(codes))
            outcomes[model] = tuple(strings)
        return EvalChunk(request.look, request.task_ids, outcomes, SIM_IMAGE)

    def close(self) -> None:
        self.closed += 1


class SimFolder:
    """Writes a fake folded base and a fake fp32 master, each in its own prefix."""

    def __init__(self, store: LocalStore) -> None:
        self.store = store
        self.fail = False

    def fold(self, head: HeadVersion, params: Params, out_prefix: str) -> FoldArtifact:
        if self.fail:
            raise InfraError("simulated fold failure")
        base, master = f"base of {head.version_id}".encode(), f"master of {head.version_id}".encode()
        self.store.put(f"{out_prefix}/base/model.bin", base)
        self.store.put(f"{out_prefix}/master/master.bin", master)
        base_files = {"model.bin": sha256_hex(base)}
        return FoldArtifact(
            head.version_id, manifest_digest(base_files), base_files, sha256_hex(b"digest of " + master),
            {"master.bin": sha256_hex(master)}, out_prefix, SIM_IMAGE,
        )


class SimPublisher:
    def __init__(self) -> None:
        self.snapshots: list[Mapping[str, Any]] = []

    def publish(self, snapshot: Mapping[str, Any]) -> None:
        self.snapshots.append(snapshot)


def stack(context: StackContext) -> Stack:
    """The CPU stack for ``python -m sn120_v2 run --stack sn120_v2.sim:stack``. Miner files come from ``SimAdmission.upload``."""
    store = LocalStore(context.state_dir / STORE_DIR)
    context.load(store)  # the journal copy in the store is the record, as with the real parts
    ticks = iter(range(1, 10**9))
    runner = LocalRunner(SimTrainer(store), SimEvaluator(store), lambda: int(time.time()), SIM_IMAGE)
    sampler = UniformSampler([f"sim:{i}" for i in range(max(400, 2 * context.params.n_tasks))])  # enough for a round of launch size
    entropy = lambda: f"sim-entropy-{next(ticks)}"
    return Stack(store, SimAdmission(store), sampler, runner, SimFolder(store), SimPublisher(), entropy=entropy, mirror=True)


@dataclass
class SimWorld:
    """A whole validator on CPU in one directory. ``reopen`` starts a fresh controller on the same journal."""

    root: Path
    params: Params
    now: int = 1_000_000
    chain: dict[int, Registration] = field(default_factory=dict)
    rounds_run: int = 0
    v1: V1Import | None = None  # state carried over from v1, used only for the first start

    def __post_init__(self) -> None:
        self.store = LocalStore(self.root / STORE_DIR)
        self.admission, self.publisher = SimAdmission(self.store), SimPublisher()
        self.trainer, self.evaluator, self.folder = SimTrainer(self.store), SimEvaluator(self.store), SimFolder(self.store)
        self.runner = LocalRunner(self.trainer, self.evaluator, self.clock, SIM_IMAGE)
        self.pool = [f"sim:{i}" for i in range(400)]
        self.reopen()
        if not self.controller.state.heads:
            base_files, master_files = {"model.bin": sha256_hex(b"sim base")}, {"master.bin": sha256_hex(b"sim master")}
            genesis = GenesisSpec("sim/model", "rev0", manifest_digest(base_files), sha256_hex(b"sim master digest"), "5SimValidator")
            self.controller.bootstrap(genesis, base_files, master_files, self.clock(), self.v1)

    def clock(self) -> int:
        self.now += 1
        return self.now

    def reopen(self) -> Controller:
        """Start a new controller from the journal, as after a crash."""
        journal = Journal(self.root / JOURNAL_FILE, "sim-controller", self.clock())
        self.sampler = UniformSampler(self.pool)
        self.controller = Controller(
            journal, self.store, self.admission, self.sampler, self.params, folder=self.folder, publisher=self.publisher
        )
        self.restored_snapshot = stored_irt_snapshot(self.store, self.controller.state)
        return self.controller

    def register(self, uid: int, quality: float | None = 0.0, block: int | None = None, **file: Any) -> Registration:
        """Put a registration on the fake chain and sync. ``quality=None`` registers with no file."""
        reg = Registration(uid, f"hk{uid}", 1000 + uid if block is None else block, self.now)
        self.chain[uid] = reg
        if quality is not None:
            self.admission.upload(reg.hotkey, quality, **file)
        self.sync()
        return reg

    def sync(self) -> None:
        owner = Registration(0, "owner", 1, 1)
        snapshot = ChainSnapshot(self.params.netuid, self.now, self.now, (owner, *self.chain.values()))
        self.controller.sync_registrations(snapshot, self.clock(), force=True)

    def round(self) -> RoundRecord | None:
        self.rounds_run += 1
        return self.controller.run_round(self.runner, self.clock, entropy=f"sim-entropy-{self.rounds_run}")
