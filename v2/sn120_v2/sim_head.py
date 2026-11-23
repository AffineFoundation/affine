"""A simulated validator whose head is real files: a tiny bf16 base, its fp32 master, and real LoRAs.

``sim.SimWorld`` writes fake bytes for a head, so nothing can fold it. This
world has the same queue, rounds and judge, and the real folders: the tests of
``chain`` mode fold, upload and rebuild for real, in a fraction of a second.

Nothing here knows how many kings a world has had, or which head it is at: a
test reads both from the journal's state, as the code does.

Only tests import this module. It is in the package, as ``sim.py`` is, because
a test module may import only the package (``tests/test_no_legacy.py``).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import save, save_file

from . import layout
from .afterking import OVERLAP, SERIAL, Overlap
from .canonical import canonical_json, manifest_digest, sha256_hex
from .controller import Controller, HeadPublish, stored_irt_snapshot
from .errors import BatchRejected, InfraError
from .fold_kernel import upcast_master
from .gpu.folder import LoraFolder
from .gpu.plan import BoxConfig
from .gpu.resident import hash_tree, upload_tree
from .headchain import CHAIN, FULL, WEIGHT_SUFFIX, ChainWriter, promote_fold
from .headfold import ChainFolder, FoldSettings, HeadKeeper, Runner, inline, last_folded
from .journal import JOURNAL_FILE, Journal
from .params import GenesisSpec
from .sampling import UniformSampler
from .sim import SIM_IMAGE, STORE_DIR, SimAdmission, SimEvaluator, SimPublisher, SimTrainer, SimWorld
from .state import APPLIED, RoundRecord, State
from .store import LocalStore
from .types import HeadManifest, LoRAArtifact, TrainReport, TrainRequest
from .worker import LocalRunner

LAYER = "model.language_model.layers.{}.mlp.{}.weight"
LORA_KEY = "base_model.model.model.language_model.layers.{}.mlp.{}.lora_{}.weight"
SHAPES = (("down_proj", 8, 24), ("up_proj", 24, 8))  # (module, rows, columns) of the weights that a LoRA changes
LAYERS = (0, 1)


def is_weight(key: str) -> bool:
    """True for a key of a folded weight file: a master part, or a weight file of a base."""
    return key.startswith("public/masters/") or (key.startswith("public/bases/") and key.endswith(WEIGHT_SUFFIX))


def tiny_base(path: Path, plain: bool = False) -> Path:
    """A base of two weight files and two small files. ``plain``: one tensor is fp32, so the base is not the cast of its master."""
    path.mkdir(parents=True)
    generator = torch.Generator().manual_seed(11)
    make = lambda *shape: torch.randn(shape, generator=generator).to(torch.bfloat16)  # noqa: E731
    for number, layer in enumerate(LAYERS, 1):
        tensors = {LAYER.format(layer, module): make(rows, columns) for module, rows, columns in SHAPES}
        tensors[f"model.language_model.extra.{layer}.weight"] = make(5, 8).float() if plain and layer else make(5, 8)
        save_file(tensors, str(path / f"model-{number:05d}-of-{len(LAYERS):05d}.safetensors"), metadata={"format": "pt"})
    (path / "config.json").write_text('{"model_type": "tiny"}')
    (path / "tokenizer.json").write_text('{"tokens": ["a", "b"]}')
    return path


def tiny_lora(seed: str, rank: int, alpha: float, extra: dict | None = None) -> dict[str, bytes]:
    """The two files of a real LoRA on the tiny base. The same seed gives the same bytes."""
    generator = torch.Generator().manual_seed(int(sha256_hex(seed.encode())[:12], 16))
    tensors = {}
    for layer in LAYERS:
        for module, rows, columns in SHAPES:
            tensors[LORA_KEY.format(layer, module, "A")] = torch.randn(rank, columns, generator=generator) * 0.05
            tensors[LORA_KEY.format(layer, module, "B")] = torch.randn(rows, rank, generator=generator) * 0.05
    config = {"r": rank, "lora_alpha": alpha} | (extra or {})
    return {"adapter_config.json": canonical_json(config), "adapter_model.safetensors": save(tensors)}


class TinyTrainer(SimTrainer):
    """The simulated trainer, with a real LoRA of the tiny base: both folders can read it."""

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
        judged = {"quality": batch["quality"], "cap_rate": batch["cap_rate"], "rank": recipe.rank}  # what the simulated evaluator reads
        blobs = tiny_lora(f"{request.head.version_id}|{entry.batch.batch_sha256}|{request.seed}", recipe.rank, recipe.alpha, judged)
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
            report=TrainReport(request.seed, recipe.steps, 0.5, 0.1, 1000, 0.1, "cpu", SIM_IMAGE),
        )


class WatchedStore(LocalStore):
    """A local store that lists each key it wrote, that signs a link as a ``file://`` address, and that a test can stop.

    ``refuse``: a write of a key for which it is true fails. ``gate``: while
    it is set and not open, the write of a folded weight file waits.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.wrote: list[str] = []
        self.refuse: Callable[[str], bool] | None = None
        self.gate: threading.Event | None = None
        self._watch = threading.Lock()

    def _note(self, key: str) -> None:
        if self.gate is not None and is_weight(key):
            assert self.gate.wait(60)
        if self.refuse is not None and self.refuse(key):
            raise InfraError(f"simulated failure of the store at {key}")
        with self._watch:
            self.wrote.append(key)

    def put(self, key: str, data: bytes) -> None:
        self._note(key)
        super().put(key, data)

    def put_file(self, key: str, path: Path) -> None:
        self._note(key)
        super().put_file(key, path)

    def copy(self, src: str, dst: str) -> None:
        self._note(dst)
        super().copy(src, dst)

    def presign_get(self, key: str, ttl_s: int) -> str:
        if not self.exists(key):
            raise KeyError(key)
        return self._path(key).as_uri()

    def weights(self) -> list[str]:
        """Every folded weight file that the store holds."""
        return [key for key in self.list("public/") if is_weight(key)]


@dataclass
class ChainWorld(SimWorld):
    """``SimWorld`` with a real tiny head. ``mode`` is ``SN120_HEAD_PUBLISH``, ``every`` is ``SN120_CHECKPOINT_EVERY``, ``order`` is ``SN120_AFTER_KING``."""

    mode: str = FULL
    every: int = 0
    order: str = SERIAL
    settings: FoldSettings = FoldSettings()
    run: Runner = inline  # the folds in this process: a child process costs seconds, and one test has it

    def __post_init__(self) -> None:
        self.store = WatchedStore(self.root / STORE_DIR)
        self.admission, self.publisher = SimAdmission(self.store), SimPublisher()
        self.trainer, self.evaluator = TinyTrainer(self.store), SimEvaluator(self.store)
        self.runner = LocalRunner(self.trainer, self.evaluator, self.clock, SIM_IMAGE)
        self.pool = [f"sim:{i}" for i in range(400)]
        self.box = BoxConfig(self.root / "work", SIM_IMAGE)
        self.uid = 2
        self.reopen()
        if not self.controller.state.heads:
            base = tiny_base(self.root / "genesis" / "base")
            master = self.root / "genesis" / "master"
            master_files = upcast_master(base, master)
            base_files = upload_tree(self.store, base, layout.public_base_prefix(manifest_digest(hash_tree(base))))[0]
            assert upload_tree(self.store, master, layout.public_master_prefix(manifest_digest(master_files)))[0] == master_files
            genesis = GenesisSpec("sim/model", "rev0", manifest_digest(base_files), manifest_digest(master_files), "5SimValidator")
            self.controller.bootstrap(genesis, base_files, master_files, self.clock(), None)

    def reopen(self) -> Controller:
        """A new controller on the same journal and the same box, with the parts that ``assembly.stack`` picks for the mode."""
        journal = Journal(self.root / JOURNAL_FILE, "sim-controller", self.clock())
        self.sampler = UniformSampler(self.pool)
        self.writer = self.keeper = self.overlap = None
        head = HeadPublish()
        if self.mode == CHAIN:
            self.writer = ChainWriter(self.store)
            self.folder = ChainFolder(self.store, self.box, self.settings, self.run)
            self.keeper = HeadKeeper(self.store, self.box.work_dir, self.writer, self.every, self.settings, self.run, self.folder.proven)
            head = HeadPublish(promote_fold, self.writer, self.keeper.prepare, self.keeper.after_round, self.keeper.checkpoint, self.folder.proven)
        else:
            self.folder = LoraFolder(self.store, self.box)
        self.controller = Controller(journal, self.store, self.admission, self.sampler, self.params, folder=self.folder, publisher=self.publisher, head=head)
        if self.order == OVERLAP and self.mode == CHAIN:  # as ``cli.cmd_run`` does for a stack with ``overlap``
            self.overlap = Overlap()
            self.controller.use_overlap(self.overlap)
        self.restored_snapshot = stored_irt_snapshot(self.store, self.controller.state)
        return self.controller

    def start(self, mode: str | None = None, every: int | None = None) -> None:
        """Stop ``run`` and start it again, with new settings if given: a new controller, then the proof of the local head."""
        self.wait()
        self.mode, self.every = mode or self.mode, self.every if every is None else every
        self.reopen()
        if self.controller.head.prepare is not None:
            self.controller.head.prepare(self.state)

    @property
    def state(self) -> State:
        return self.controller.state

    @property
    def head(self) -> HeadManifest:
        """The manifest of the last folded head."""
        return self.state.manifests[last_folded(self.state).version_id]

    def round(self) -> RoundRecord | None:
        record = super().round()
        if self.controller.head.after_round is not None:
            self.controller.head.after_round(self.state)  # as the loop of ``run`` does after each pass
        return record

    def king(self) -> HeadManifest:
        """One round with a winner and its fold. Returns the manifest of the new folded head."""
        before = len(self.state.heads)
        self.register(self.uid, 0.3)
        self.register(self.uid + 1, 0.0)
        self.uid += 2
        record = self.round()
        assert record is not None and record.outcome == APPLIED and len(self.state.heads) == before + 2, self.controller.last_error
        return self.head

    def wait(self) -> None:
        """Wait for the checkpoint thread, if one runs."""
        if self.keeper is not None:
            assert self.keeper.wait(60)

    def chain_file(self) -> dict:
        """The chain file as the store has it now."""
        return json.loads(self.store.get(layout.HEAD_CHAIN))

    def local(self, identity: str) -> Path:
        return self.box.work_dir / "cache" / identity
