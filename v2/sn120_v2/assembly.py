"""The real parts of one validator box: ``--stack sn120_v2.assembly:stack``.

One process on the GPU box holds the controller, the round runner (trainer
and evaluator), the folder and the publisher. The object store is R2. The
journal's durable copy is in the store, so the box's disk is only a cache:
a new box runs the same command and continues.

Every setting comes from the environment. A missing one is an error.

  SN120_R2_ENDPOINT, SN120_R2_BUCKET, SN120_R2_ACCESS_KEY_ID,
  SN120_R2_SECRET_ACCESS_KEY, SN120_R2_PREFIX (optional)    the store
  SN120_PUBLIC_URL          where the public zone is served; "none" before a public address exists
  SN120_V2_CHAIN_NETWORK    "finney" or a node address: the queue follows the chain and the round
                            seed uses a block hash. "none": no chain read (the smoke test)
  SN120_WORK_DIR, SN120_IMAGE_DIGEST                        the box; read by ``run`` only

The box holds no wallet. It only reads the chain.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from transformers import AutoTokenizer

from . import layout
from .admission import HFTokens, LazyPool, StoreAdmission, check_batch, pool_grader
from .canonical import file_sha256
from .chain import Reader, SubtensorClient
from .controller import Stack, StackContext, stored_irt_snapshot
from .envs.math_grader import GradePool
from .envs.math_pool import load_pool
from .errors import ConfigError, InfraError, VerifyError
from .gpu.evaluator import VllmEvaluator
from .gpu.folder import LoraFolder
from .gpu.plan import BoxConfig
from .gpu.trainer import LoraTrainer
from .irt import IRTSampler
from .params import Params, require_env
from .publisher import StorePublisher
from .r2 import R2Store
from .seams import ObjectStore
from .store import get_record
from .types import ChainSnapshot, HeadManifest
from .worker import LocalRunner

NONE = "none"
# Admission grades the rewards of miner files in its own worker processes, apart from the evaluator's pool.
# A row that is not graded inside the limit makes its file invalid for that check (rule grader_time_limit).
# A normal row takes milliseconds, and math-verify stops itself after 5 seconds per step.
ADMISSION_GRADE_WORKERS = min(8, os.cpu_count() or 1)
ADMISSION_GRADE_TIMEOUT_S = 20.0
WEIGHT_SUFFIX = ".safetensors"


class HeadTokens:
    """Token counts with the tokenizer of the current head.

    The tokenizer files are read from the public store on first use, each one
    checked against the head manifest. So the checker counts tokens with
    exactly the files that the trainer and the evaluator load.
    """

    def __init__(self, store: ObjectStore, cache: Path, journal_manifest: HeadManifest | None = None) -> None:
        self.store, self.cache, self.journal_manifest = store, cache, journal_manifest
        self._tokens: HFTokens | None = None

    def _load(self) -> HFTokens:
        if self._tokens is None:
            try:
                manifest = get_record(self.store, layout.CURRENT, HeadManifest)
            except KeyError:
                # A store that was bootstrapped when the pointer had another key has none at this key until the
                # controller's next tick writes it. The journal's manifest is what that pointer will hold.
                manifest = self.journal_manifest
                if manifest is None:
                    raise ConfigError("the store has no current head yet; run bootstrap first") from None
            base_hash = manifest.head.base_hash
            directory = self.cache / base_hash
            for name, sha in sorted(manifest.base_files.items()):
                path = directory / name
                if name.endswith(WEIGHT_SUFFIX) or (path.is_file() and file_sha256(path) == sha):
                    continue
                self.store.get_file(layout.public_base_file(base_hash, name), path)
                if file_sha256(path) != sha:
                    path.unlink()
                    raise VerifyError(f"{name} of base {base_hash[:16]} does not match the head manifest")
            self._tokens = HFTokens(AutoTokenizer.from_pretrained(str(directory), local_files_only=True))
        return self._tokens

    def prompt_tokens(self, prompt: str) -> int:
        return self._load().prompt_tokens(prompt)

    def output_tokens(self, text: str) -> int:
        return self._load().output_tokens(text)


class ChainFeed:
    """Registrations and the round's entropy from one read-only chain connection, opened on first use."""

    def __init__(self, netuid: int, network: str) -> None:
        self.netuid, self.network = netuid, network
        self._reader: Reader | None = None
        self._block = 0

    def _open(self) -> Reader:
        if self._reader is None:
            self._reader = Reader(SubtensorClient.connect(self.netuid, self.network), self.netuid)  # no wallet
        return self._reader

    def snapshot(self) -> ChainSnapshot:
        snapshot = self._open().snapshot()
        self._block = snapshot.block
        return snapshot

    def entropy(self) -> str:
        """The hash of the block of the last snapshot: a value this validator cannot choose."""
        if not self._block:
            raise InfraError("no chain snapshot yet, so there is no block hash for the round seed")
        return self._open().block_hash(self._block)


def _read_groups(raw: bytes, hotkey: str, params: Params):
    return check_batch(raw, hotkey, params).groups


def stack(context: StackContext) -> Stack:
    store = R2Store.from_env()
    state = context.load(store)  # brings the local journal up to date from the store, and pins context.params
    params = context.params
    pool = load_pool()
    tokens = HeadTokens(store, context.state_dir / "tokenizer", state.manifest)
    grades = LazyPool(lambda: GradePool(workers=ADMISSION_GRADE_WORKERS, timeout_s=ADMISSION_GRADE_TIMEOUT_S))  # starts at the first file
    # One chunk is one row for each worker: a file whose rows all hang costs one time limit, not several.
    admission = StoreAdmission(store, params, tokens=tokens, grader=pool_grader(pool, grades, chunk=ADMISSION_GRADE_WORKERS))
    sampler = IRTSampler(pool.task_ids, params, snapshot=stored_irt_snapshot(store, state))
    public_url = require_env("SN120_PUBLIC_URL")
    publisher = StorePublisher(store, "" if public_url == NONE else public_url)
    network = require_env("SN120_V2_CHAIN_NETWORK")
    feed = None if network == NONE else ChainFeed(params.netuid, network)
    chain: Callable[[], ChainSnapshot] | None = feed.snapshot if feed else None
    entropy: Callable[[], str] = feed.entropy if feed else lambda: os.urandom(16).hex()
    runner = folder = None
    if context.command == "run":
        box = BoxConfig.from_env()
        trainer = LoraTrainer(store, box, _read_groups)
        evaluator = VllmEvaluator(store, box, pool, GradePool())
        runner = LocalRunner(trainer, evaluator, lambda: int(time.time()), box.image_digest)
        folder = LoraFolder(store, box)
    return Stack(store, admission, sampler, runner, folder, publisher, entropy=entropy, chain=chain, mirror=True)
