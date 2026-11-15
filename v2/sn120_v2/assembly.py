"""The real parts of one validator box: ``--stack sn120_v2.assembly:stack``.

One process on the GPU box holds the controller, the round runner (trainer
and evaluator), the folder and the publisher. The object store is R2. The
journal's durable copy is in the store, so the box's disk is only a cache:
a new box runs the same command and continues.

Every setting comes from the environment. A missing one is an error.

  SN120_R2_ENDPOINT, SN120_R2_BUCKET, SN120_R2_ACCESS_KEY_ID,
  SN120_R2_SECRET_ACCESS_KEY, SN120_R2_PREFIX (optional)    the store
  SN120_PUBLIC_URL          where the public zone is served; "none" before a public address exists
  SN120_LINKS_URL (optional)  where the upload link files are served; not set or "none": not published
  SN120_V2_CHAIN_NETWORK    "finney" or a node address: the queue follows the chain and the round
                            seed uses a block hash. "none": no chain read (the smoke test)
  SN120_WORK_DIR, SN120_IMAGE_DIGEST                        the box; read by ``run`` only
  SN120_HEAD_PUBLISH (optional)  "full" (the default): each folded head is uploaded. "chain": the folded
                            weight files stay on the box, and the store gets the LoRAs and the head chain
                            (``headchain.py``). Then also: SN120_CHECKPOINT_EVERY, and the fold settings
                            SN120_FOLD_IMPLEMENTATION, SN120_FOLD_TWICE, SN120_FOLD_DEVICES, SN120_FOLD_THREADS
                            (``headfold.FoldSettings``); read by ``run`` and ``checkpoint`` only
  SN120_CHECK_STREAMS (optional)  how many waiting files the live check reads from the store at one time, 1 to 32.
                            Not set: 1, one file after the other. The verdicts are the same for each count.
                            Read by each command that makes the live check: ``run``, ``status``, ``publish``
                            and ``bootstrap``
  SN120_AFTER_KING (optional)  "serial" (the default): after a king, each step waits for the one before it.
                            "overlap": the fold and the check of the waiting files are before the next open; the
                            other steps run beside the fold or after the open (``afterking.py``). Only with
                            SN120_HEAD_PUBLISH=chain; read by ``run`` only

``run`` prints one line at its start for each of SN120_HEAD_PUBLISH, SN120_CHECK_STREAMS and SN120_AFTER_KING,
with the words NOT SET when the shell does not hold the variable. It prints no line for another setting.

The box holds no wallet. It only reads the chain.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import time
from pathlib import Path
from typing import Callable

from transformers import AutoTokenizer

from . import layout
from .admission import SERIAL_STREAMS, STREAMS_ENV, HFTokens, LazyPool, StoreAdmission, check_batch, check_streams, pool_grader
from .afterking import ORDER_ENV, OVERLAP, after_king
from .canonical import file_sha256
from .chain import Feed, SubtensorClient
from .controller import HeadPublish, Stack, StackContext, stored_irt_snapshot
from .envs.math_grader import GradePool
from .envs.math_pool import load_pool
from .errors import ConfigError, InfraError, VerifyError
from .gpu.evaluator import VllmEvaluator
from .gpu.folder import LoraFolder
from .gpu.plan import BoxConfig, check_pins, code_digest
from .gpu.trainer import LoraTrainer
from .headchain import CHAIN, MODE_ENV, ChainWriter, checkpoint_every, mode, promote_fold
from .headfold import ChainFolder, FoldSettings, HeadKeeper
from .irt import IRTSampler
from .params import Params, require_env
from .publisher import StorePublisher
from .r2 import R2Store
from .reuse import WinnerRollouts, store_reader
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
BOX_COMMANDS = ("run", "checkpoint")  # the commands that work on the local head of the box


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


class ChainFeed(Feed):
    """Registrations and the round's entropy from one read-only chain connection: opened on first use, and again after a failed read."""

    def __init__(self, netuid: int, network: str) -> None:
        super().__init__(netuid, lambda: SubtensorClient.connect(netuid, network))  # no wallet
        self.network = network


def _read_groups(raw: bytes, hotkey: str, params: Params):
    return check_batch(raw, hotkey, params).groups


def box_pins(image_digest: str, params: Params) -> None:
    """The check of the trainer (``check_pins``), with the pinned value and the value of this box in the text.

    No command shows the pinned digests, so the refusal of ``run`` must show them.
    """
    try:
        check_pins(image_digest, params)
    except InfraError as exc:
        recipe, code, differs = params.recipe, code_digest(), []
        if image_digest not in recipe.image_digests:
            differs.append(f"image digest: pinned {' or '.join(recipe.image_digests)}, this box has {image_digest} (the value of SN120_IMAGE_DIGEST)")
        if recipe.code_digest and recipe.code_digest != code:
            differs.append(f"code digest: pinned {recipe.code_digest}, this box has {code}")
        raise InfraError("; ".join(differs) or str(exc)) from None


def setting(name: str, meaning: str) -> str:
    """One line for the start of ``run``: what a setting does now, and whether the shell holds it."""
    value = os.environ.get(name, "").strip()
    return f"{meaning} ({name}={value})" if value else f"{meaning} ({name} is NOT SET: this is the default)"


def stack(context: StackContext) -> Stack:
    store = R2Store.from_env()
    state = context.load(store)  # brings the local journal up to date from the store, and pins context.params
    params = context.params
    pool = load_pool()
    tokens = HeadTokens(store, context.state_dir / "tokenizer", state.manifest)
    grades = LazyPool(lambda: GradePool(workers=ADMISSION_GRADE_WORKERS, timeout_s=ADMISSION_GRADE_TIMEOUT_S))  # starts at the first file
    # One chunk is one row for each worker: a file whose rows all hang costs one time limit, not several.
    streams = check_streams()
    # The winner files are read from the store. The directory only keeps their rollout keys, so a new process does not read them again.
    winners = WinnerRollouts(store_reader(store), context.state_dir / "winner_rollouts")
    admission = StoreAdmission(store, params, tokens=tokens, grader=pool_grader(pool, grades, chunk=ADMISSION_GRADE_WORKERS), streams=streams, public=winners)
    order = after_king()
    if order == OVERLAP and mode() != CHAIN:
        raise ConfigError(f"{ORDER_ENV}={OVERLAP} needs {MODE_ENV}={CHAIN}: in the mode {mode()!r} the upload of the folded head is the wait, and this order does not move it")
    shown = (
        setting(MODE_ENV, f"the head goes public in the mode {mode()!r}"),
        setting(STREAMS_ENV, "the check of the waiting files reads " + ("one file after the other" if streams == SERIAL_STREAMS else f"{streams} files at one time")),
        setting(ORDER_ENV, "after a king, " + ("only the fold is before the next open (the order 'overlap')" if order == OVERLAP else "each step waits for the one before it (the order 'serial')")),
    )
    sampler = IRTSampler(pool.task_ids, params, snapshot=stored_irt_snapshot(store, state))
    public_url = require_env("SN120_PUBLIC_URL")
    links_url = os.environ.get("SN120_LINKS_URL", "").strip()
    publisher = StorePublisher(store, "" if public_url == NONE else public_url, "" if links_url == NONE else links_url)
    network = require_env("SN120_V2_CHAIN_NETWORK")
    feed = None if network == NONE else ChainFeed(params.netuid, network)
    chain: Callable[[], ChainSnapshot] | None = feed.snapshot if feed else None
    entropy: Callable[[], str] = feed.entropy if feed else lambda: os.urandom(16).hex()
    entropy_now: Callable[[], str] | None = feed.entropy_now if feed else None
    runner = folder = pins = None
    if context.command == "run":
        box = BoxConfig.from_env()
        trainer = LoraTrainer(store, box, _read_groups)
        evaluator = VllmEvaluator(store, box, pool, GradePool())
        runner = LocalRunner(trainer, evaluator, lambda: int(time.time()), box.image_digest)
        folder = LoraFolder(store, box)
        pins = functools.partial(box_pins, box.image_digest)  # the check of the trainer, made before any work starts
    head = HeadPublish()
    if mode() == CHAIN:
        writer = ChainWriter(store)
        head = HeadPublish(promote=promote_fold, chain=writer)
        if context.command in BOX_COMMANDS:
            box, settings = BoxConfig.from_env(), FoldSettings.from_env()
            if folder is not None:
                folder = ChainFolder(store, box, settings)
                head = dataclasses.replace(head, proven=folder.proven)
            keeper = HeadKeeper(store, box.work_dir, writer, checkpoint_every(), settings, proven=head.proven)
            head = dataclasses.replace(head, prepare=keeper.prepare, after_round=keeper.after_round, checkpoint=keeper.checkpoint)
    overlap = order == OVERLAP and context.command == "run"
    return Stack(store, admission, sampler, runner, folder, publisher, entropy=entropy, chain=chain, pins=pins, mirror=True, head=head, settings=shown, overlap=overlap, entropy_now=entropy_now)
