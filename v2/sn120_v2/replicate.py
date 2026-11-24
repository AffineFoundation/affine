"""Train a LoRA from your batch file and score it against null M on your own GPUs, with the code of the validator's box.

    python -m sn120_v2.replicate compare --mining-json https://affine.io/mining.json --gpus 1
    python -m sn120_v2.replicate fetch --mining-json https://affine.io/mining.json --work /data/affine
    python -m sn120_v2.replicate run --mining-json https://affine.io/mining.json --hotkey <ss58> \
        --work /data/affine --gpus 1 batch.json

This module trains nothing and scores nothing by itself. It builds the box's
own trainer, evaluator, round runner and judging (``sn120_v2.gpu``,
``worker``, ``judging``) over a store on the local disk, and runs a round of
one batch. Where a miner's machine cannot equal the box, the run uses a copy
of the params with that one value changed, and the report names each change
under ``differences``. The document is ``v2/REPLICATE.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from transformers import AutoTokenizer

from . import judging, layout
from .admission import Grader, HFTokens, Invalid, LazyPool, Tokens, check_batch, pool_grader
from .canonical import file_sha256, manifest_digest, to_jsonable
from .envs.math_grader import GradePool
from .envs.math_pool import ENV_ID, POOL_SHA256, MathPool, load_pool, pool_for_draw
from .errors import ConfigError, InfraError, V2Error, VerifyError
from .gpu.evaluator import VllmEvaluator
from .gpu.fold_input import FoldInputs, unchanged
from .gpu.plan import BoxConfig, code_digest, install_digest
from .gpu.solo import SoloRunner
from .gpu.procs import Child, Spawn, nvidia_smi, wait_gpus_free
from .gpu.resident import Resident, hash_tree
from .gpu.trainer import LoraTrainer
from .headchain import heads_of
from .headfold import FoldSettings, LinkSource, download, read, read_json, rebuild, say
from .irt import draw_from_snapshot
from .params import Params
from .seams import Evaluator, ObjectStore, Trainer
from .store import LocalStore
from .types import CAP, NOT_RUN, PASS, Draw, Entry, FrozenBatch, HeadManifest, HeadVersion, Registration, RoundTicket
from .worker import LocalRunner

ROUND_ID, LEASE_ID, BOX_ID = "local", "local-l0", "local-box"
NO_ROUND_LIMIT_S = 30 * 86400  # stands for "no time limit of the round": no run reaches it
# The validator's file check grades with this many workers and this limit for one row (the same numbers as ``assembly.py``).
GRADE_WORKERS = min(8, os.cpu_count() or 1)
GRADE_TIMEOUT_S = 20.0
HARDWARE = ("install", "code", "gpus")  # the differences that ``--strict`` refuses


@dataclass(frozen=True)
class Contract:
    """mining.json: the params, the head with its file hashes, and where the public objects are."""

    doc: Mapping[str, Any]
    params: Params
    manifest: HeadManifest

    @classmethod
    def load(cls, source: str) -> Contract:
        doc = read_json(source, "mining.json")
        try:
            return cls(doc, Params.from_public_dict(doc["params"]), HeadManifest.from_dict(doc["head"]))
        except (KeyError, TypeError, AttributeError):
            raise ConfigError("this mining.json has no params or no head") from None

    @property
    def head(self) -> HeadVersion:
        return self.manifest.head

    def url(self, key: str) -> str:
        """The public address of an object key."""
        public = (self.doc.get("contract") or {}).get("public_url") or ""
        if not public:
            raise ConfigError(f"this mining.json names no public address; pass {key} as a file")
        return f"{public.rstrip('/')}/{key}"

    def links_url(self) -> str:
        """The public address of the file of signed links to the head's files."""
        key = (self.doc.get("contract") or {}).get("read_links") or ""
        if not key:
            raise ConfigError("this mining.json names no links file; pass --read-links or --from")
        return self.url(str(key))

    def round(self, round_id: str | None) -> Mapping[str, Any]:
        """A round that mining.json shows: the one named, or the newest (the open round, else the last closed one)."""
        rounds = [view for view in (self.doc.get("round"), *reversed(self.doc.get("history") or [])) if view and view.get("round_seed")]
        for view in rounds:
            if round_id in (None, view.get("round_id")):
                return view
        shown = ", ".join(str(view.get("round_id")) for view in rounds) or "none"
        raise ConfigError(f"mining.json does not show round {round_id or '(any)'} (it shows: {shown}); pass --round-file, or --round-seed with --irt-snapshot")


# ---- what differs from the box ----


def gpu_report(n: int | None = None) -> str:
    """What the driver says of the GPUs, one line for each, cut to the first ``n``."""
    lines = [line for line in nvidia_smi().splitlines() if line.strip()]
    return "\n".join(lines if n is None else lines[:n])


def gpu_count() -> int | None:
    """How many GPUs the driver lists, or None when it does not answer."""
    try:
        return len(gpu_report().splitlines())
    except InfraError:
        return None


def local_params(params: Params, image: str, code: str, gpus: int, round_clock: bool = False) -> tuple[Params, list[dict[str, str]]]:
    """The params of a run on this machine, and every difference between that run and a round on the box.

    Only a value that this machine cannot keep is changed. Every other value is the one of mining.json.
    """
    notes: list[dict[str, str]] = []

    def note(what: str, text: str) -> None:
        notes.append({"what": what, "text": text})

    recipe, changes = params.recipe, {}
    if image not in recipe.image_digests:
        pinned = " or ".join(recipe.image_digests) or "(none)"
        note("install", f"The install of this machine has digest {image}. The box runs the pinned install {pinned}. The box code refuses an install that is not pinned, so this run pins its own. The versions of torch, vLLM and every other package can differ from the box.")
        recipe = dataclasses.replace(recipe, image_digests=(image,))
    if recipe.code_digest and recipe.code_digest != code:
        note("code", f"sn120_v2/gpu/ of this checkout has digest {code}. The box runs the pinned code {recipe.code_digest}. This run pins its own. Check out the commit that the box runs to remove this difference.")
        recipe = dataclasses.replace(recipe, code_digest=code)
    if recipe is not params.recipe:
        changes["recipe"] = recipe
    if gpus != params.n_gpus:
        note("gpus", f"This run uses {gpus} GPU(s). The box has {params.n_gpus}. One vLLM server runs on each GPU, and task i of a chunk goes to server i mod the number of GPUs.")
        changes["n_gpus"] = gpus
    if not round_clock:
        stop = params.round_time_limit_s - params.promote_reserve_s
        note("round-clock", f"No time limit of the round applies. On the box the training ends {params.train_time_budget_s} s after the start of the round and the scoring stops at {stop} s; what is scored by then is the whole test. The limit of one reply ({params.eval_time_budget_s} s) is the same here.")
        changes.update(round_time_limit_s=NO_ROUND_LIMIT_S, lease_timeout_s=NO_ROUND_LIMIT_S, promote_reserve_s=0, train_time_budget_s=NO_ROUND_LIMIT_S - 1)
    note("one-batch", f"This is a round of one batch. The result says whether this LoRA beats null M by the win rule" + (f": delta above the tie tolerance and z strictly above {params.z_bar}, with positive standard error" if params.crown_gate == "argmax_symmetric" and params.z_bar > 0 else "") + f". In a real round the eligible LoRA with the highest delta wins. The training job has all {gpus} GPU(s) of this run; on the box a job has floor({params.n_gpus} / batches of the round) GPUs, at least 1. The LoRA table of the servers (--max-loras) is 1; on the box it is the number of batches of the round.")
    note("master", "For eval_merged=true the real fp32 master is required and is large (tens of GB): fetch downloads and verifies it, and run builds the official fp32-master + LoRA candidate. For eval_merged=false the master is not required.")
    note("free-gpus", f"Before the scoring starts, only GPUs 0 to {gpus - 1} must be free. The box waits for all of its GPUs.")
    return (dataclasses.replace(params, **changes) if changes else params), notes


# ---- the parts of the box, on a local disk ----


class BaseOnly(Resident):
    """The miner's local model cache, with no implicit public-store downloads.

    ``fetch`` puts complete, verified file sets here. A missing or invalid set
    is a local configuration error, not a non-existent master path or a search
    for a public prefix in the miner's empty local store.
    """

    merged = False

    def tree(self, prefix: str, identity: str) -> Path:
        path = self.root / identity
        if prefix == layout.public_master_prefix(identity):
            if not self.merged:
                # The box's resident_head always checks the master. Non-merged
                # local training has no merged candidate and does not need it.
                return path
            if not path.is_dir():
                raise ConfigError(
                    f"eval_merged requires the real fp32 master at {path}. "
                    "The master is large (tens of GB). Run the fetch command first; "
                    "do not substitute the bf16 base."
                )
            try:
                return super().tree(prefix, identity)
            except VerifyError:
                raise ConfigError(
                    f"the fp32 master at {path} is missing files or does not hash to "
                    f"base_fp32_master_digest {identity[:16]}. Under eval_merged the real "
                    "master is required and is large (tens of GB). Remove the invalid "
                    "directory and run the fetch command again."
                ) from None
        if prefix == layout.public_base_prefix(identity) and not path.is_dir():
            raise ConfigError(f"the base files of the head are not in {path}; run the fetch command first")
        return super().tree(prefix, identity)


def _read_groups(raw: bytes, hotkey: str, params: Params):
    return check_batch(raw, hotkey, params).groups


@dataclass
class Parts:
    store: LocalStore
    trainer: LoraTrainer
    evaluator: VllmEvaluator
    image: str
    resident: BaseOnly
    fold_input: FoldInputs | None = None


def parts(
    work: Path,
    gpus: int | None = None,
    *,
    grades: Any = None,
    train_spawn: Spawn = Child,
    serve: Mapping[str, Any] | None = None,
    pool: MathPool | None = None,
    fold_input: FoldInputs | None = None,
) -> Parts:
    """The box's trainer and evaluator, built as ``assembly.stack`` builds them, over a store in ``<work>/store``.

    ``gpus`` limits the free-GPU check to the first GPUs; None keeps the check of the box. ``grades`` is the
    evaluator's grade pool (default: the box's ``GradePool()``). ``serve`` holds more arguments of ``VllmEvaluator``.
    ``pool`` is the current pool unless a past round supplies the archived v2.0 pool.
    """
    work = Path(work).resolve()
    store, box = LocalStore(work / "store"), BoxConfig(work, install_digest())
    free = wait_gpus_free if gpus is None else functools.partial(wait_gpus_free, report=functools.partial(gpu_report, gpus))
    prepared = fold_input or unchanged
    trainer = LoraTrainer(store, box, _read_groups, spawn=train_spawn, fold_input=prepared)
    evaluator = VllmEvaluator(store, box, pool or load_pool(), grades or GradePool(), gpus_free=free, fold_input=prepared, **(serve or {}))
    resident = BaseOnly(store, work)
    resident.merged = fold_input is not None
    trainer.resident = evaluator.resident = resident
    return Parts(store, trainer, evaluator, box.image_digest, resident, fold_input)


def backend(workdir: str) -> tuple[ObjectStore, Trainer, Evaluator, str]:
    """For ``affine_miner.py reproduce --backend sn120_v2.replicate:backend``: the parts of the box, unchanged but for the master.

    That command runs with the params of mining.json as they are. So the box code refuses the run unless this
    machine has the pinned install, the pinned code and the GPUs of the params. ``run`` of this module is for every other machine.
    """
    built = parts(Path(workdir))
    return built.store, built.trainer, built.evaluator, built.image


def base_dir(resident: Resident, head: HeadVersion) -> Path:
    """The verified base files of the head in the cache. Nothing is downloaded here: ``fetch`` does that."""
    try:
        return resident.tree(layout.public_base_prefix(head.base_hash), head.base_hash)
    except VerifyError:
        raise VerifyError(f"the files in {resident.root / head.base_hash} did not hash to base_hash {head.base_hash[:16]}; the directory was deleted, as on the box. Run the fetch command again") from None


# ---- fetch ----


def place(source: Path, path: Path) -> None:
    """Put a local file at ``path``: a hard link when the file system allows it, else a copy."""
    if not source.is_file():
        raise ConfigError(f"{source} is missing")
    try:
        os.link(source.resolve(), path)
    except OSError:
        shutil.copyfile(source, path)


def base_links(contract: Contract, links: Mapping[str, Any], now: float) -> dict[str, Mapping[str, Any]]:
    """The base and, for merged evaluation, fp32-master links of the contract's head."""
    head = contract.head
    base, master = links.get("base") or {}, links.get("master") or {}
    if links.get("head_version_id") != head.version_id or base.get("id") != head.base_hash:
        raise ConfigError("mining.json and the links file name two different heads; fetch both again")
    if contract.params.eval_merged and master.get("id") != head.base_fp32_master_digest:
        raise ConfigError("eval_merged needs the current head's fp32 master in the links file; fetch both files again")
    if now >= int(links.get("expires_at") or 0):
        raise ConfigError("the links have expired; fetch the links file again")

    def files(wanted: Mapping[str, str], group: Mapping[str, Any], what: str) -> dict[str, Mapping[str, Any]]:
        found = {str(row.get("name")): row for row in group.get("files") or [] if isinstance(row, dict)}
        if set(found) != set(wanted) or any(found[name].get("sha256") != sha for name, sha in wanted.items()):
            raise VerifyError(f"the links file does not list the {what} files of the head manifest")
        return found

    found = files(contract.manifest.base_files, base, "base")
    if contract.params.eval_merged:
        found |= {f"../{head.base_fp32_master_digest}/{name}": row for name, row in files(contract.manifest.master_files, master, "fp32-master").items()}
    return found


def fetch_file_set(work: Path, identity: str, wanted: Mapping[str, str], *, source_dir: Path | None = None, links: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Fetch one whole file set without ever changing a sealed cache directory."""
    if any(not name or name.startswith("/") or ".." in name.split("/") for name in wanted):
        raise VerifyError("the head manifest has an unsafe file name")
    if not wanted:
        raise ConfigError(f"the head manifest names no files for {identity}; fetch cannot build a valid cache")
    cache = Path(work).resolve() / "cache" / identity
    if cache.is_dir():
        files = hash_tree(cache)
        if files == dict(wanted) and manifest_digest(files) == identity:
            return {"dir": str(cache), "files": len(files), "bytes": sum((cache / name).stat().st_size for name in files), "fetched": 0, "fetched_bytes": 0}
        raise ConfigError(f"{cache} exists and does not match its manifest and id; remove it and run fetch again")
    stage = Path(work).resolve() / "fetch" / identity
    fetched = fetched_bytes = 0
    for name, sha in sorted(wanted.items()):
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and file_sha256(path) == sha:
            continue
        path.unlink(missing_ok=True)
        if source_dir:
            place(Path(source_dir) / name, path)
            if file_sha256(path) != sha:
                path.unlink()
                raise VerifyError(f"{Path(source_dir) / name} is not the file of the head manifest")
        else:
            try:
                link = (links or {})[name]
            except KeyError:
                raise ConfigError(f"the links file has no download for {name} of {identity[:16]}; fetch both public files again") from None
            say(f"fetch {name} ({link['size']} bytes)")
            download(str(link["url"]), path, sha, int(link["size"]))
        fetched += 1
        fetched_bytes += path.stat().st_size
    extra = sorted(file.relative_to(stage).as_posix() for file in stage.rglob("*") if file.is_file() and file.relative_to(stage).as_posix() not in wanted)
    if extra:
        raise ConfigError(f"{stage} holds files that are not of the head: {extra[:3]}; remove them and run fetch again")
    cache.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(cache)
    return {"dir": str(cache), "files": len(wanted), "bytes": sum((cache / name).stat().st_size for name in wanted), "fetched": fetched, "fetched_bytes": fetched_bytes}


def fetch(contract: Contract, work: Path, *, source_dir: Path | None = None, master_dir: Path | None = None, links_source: str | None = None, now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Fetch the base, and the fp32 master required by a merged-candidate run, under ``<work>/cache``."""
    found = None if source_dir else read_json(links_source or contract.links_url(), "the links file")
    if found is not None and found.get("base") is None and contract.doc.get("head_chain"):
        if master_dir is not None:
            raise ConfigError("--master-from cannot replace a head-chain rebuild; use the checkpoint master in the links file")
        return fetch_by_chain(contract, Path(work).resolve(), found, now)
    if contract.params.eval_merged and source_dir and master_dir is None:
        raise ConfigError("eval_merged needs the actual fp32 master: pass --master-from alongside --from, or omit --from to use signed links")
    links = None if source_dir else base_links(contract, found, now())
    head = contract.head
    result = fetch_file_set(Path(work), head.base_hash, contract.manifest.base_files, source_dir=source_dir, links=links)
    result["base_hash"] = head.base_hash
    result["base_dir"] = result["dir"]
    result["base_files"] = result["files"]
    result["base_bytes"] = result["bytes"]
    result["base_fetched"] = result["fetched"]
    result["base_fetched_bytes"] = result["fetched_bytes"]
    if contract.params.eval_merged:
        identity = head.base_fp32_master_digest
        master_prefix = f"../{identity}/"
        master_links = {name.removeprefix(master_prefix): row for name, row in (links or {}).items() if name.startswith(master_prefix)} or None
        master = fetch_file_set(Path(work), identity, contract.manifest.master_files, source_dir=master_dir, links=master_links)
        result.update(master_digest=identity, master_hash=identity, master_dir=master["dir"], master_files=master["files"], master_bytes=master["bytes"], master_fetched=master["fetched"], master_fetched_bytes=master["fetched_bytes"])
        result["fetched"] += master["fetched"]
        result["fetched_bytes"] += master["fetched_bytes"]
    return result


def fetch_by_chain(contract: Contract, work: Path, links: Mapping[str, Any], now: Callable[[], float]) -> dict[str, Any]:
    """The base of a head whose weight files are not in the store: the newest checkpoint, and each fold after it done again here.

    ``<work>/cache`` keeps the fp32 master of the head too. The next fetch then starts from this head and replays only the new folds.
    """
    head, chain = contract.head, contract.doc["head_chain"]
    if links.get("head_version_id") != head.version_id or chain.get("head_version_id") != head.version_id:
        raise ConfigError("mining.json and the links file name two different heads; fetch both again")
    try:
        target = heads_of(chain)[-1]
        same = target["base_hash"] == head.base_hash
    except (KeyError, TypeError, IndexError):
        raise VerifyError("the head chain of mining.json is not complete") from None
    if not same:
        raise VerifyError("the head chain of mining.json does not end at the base of the head")
    say("the head is not a checkpoint: this machine makes its base from the newest checkpoint and the LoRAs after it")
    built = rebuild(chain, work, LinkSource(links, now), target["version_id"], settings=FoldSettings.from_env(box=False))
    result = {
        "base_hash": head.base_hash,
        "base_dir": built["base_dir"],
        "dir": built["base_dir"],
        "base_files": len(contract.manifest.base_files),
        "files": len(contract.manifest.base_files),
        "base_bytes": sum((Path(built["base_dir"]) / name).stat().st_size for name in contract.manifest.base_files),
        "fetched": 0,  # a chain rebuild reports downloaded bytes, not a file count
        "fetched_bytes": built["fetched_bytes"],
        "base_fetched": 0,
        "base_fetched_bytes": built["fetched_bytes"],
        "replayed_folds": built["steps"],
    }
    for directory, wanted, identity in (
        (Path(built["base_dir"]), contract.manifest.base_files, head.base_hash),
        *(([(Path(built["master_dir"]), contract.manifest.master_files, head.base_fp32_master_digest)] if contract.params.eval_merged else [])),
    ):
        actual = hash_tree(directory)
        if actual != dict(wanted) or manifest_digest(actual) != identity:
            raise VerifyError(f"the rebuilt file set at {directory} does not match its head manifest and id {identity[:16]}")
    if contract.params.eval_merged:
        identity = head.base_fp32_master_digest
        master = Path(built["master_dir"])
        result.update(master_digest=identity, master_hash=identity, master_dir=str(master), master_files=len(contract.manifest.master_files), master_bytes=sum((master / name).stat().st_size for name in contract.manifest.master_files), master_fetched=0, master_fetched_bytes=0)
    return result


# ---- the round to repeat ----


@dataclass(frozen=True)
class Round:
    """The round a run repeats."""

    shown: dict[str, Any]  # what the report shows under "round"
    draw: Draw  # the tasks the run scores, in the order of the round
    view: Mapping[str, Any]  # what is public about the real round; only the seed for a seed of your choice
    box_file: Mapping[str, Any] | None  # the public round file, when one was given


def the_round(contract: Contract, *, round_id: str | None = None, round_file: str | None = None, round_seed: str | None = None, irt_snapshot: str | None = None, first_tasks: int | None = None) -> Round:
    """The seed and the draw of the run, from a round of mining.json, from a public round file, or from a seed of your choice."""
    box_file = read_json(round_file, "the round file") if round_file else None
    if round_seed:
        if not irt_snapshot:
            raise ConfigError("--round-seed needs --irt-snapshot")
        view: Mapping[str, Any] = {"round_seed": round_seed}
    else:
        view = box_file if box_file is not None else contract.round(round_id)
    if not view.get("round_seed") or not (irt_snapshot or view.get("irt_snapshot_id")):
        raise ConfigError("the round has no seed or names no snapshot")
    snapshot = read_json(irt_snapshot or contract.url(layout.irt_snapshot(view["irt_snapshot_id"])), "the IRT snapshot")
    if view.get("irt_snapshot_id") not in (None, snapshot.get("snapshot_id")):
        raise ConfigError(f"the round names snapshot {view['irt_snapshot_id']}; the snapshot given is {snapshot.get('snapshot_id')}")
    draw = draw_from_snapshot(snapshot, str(view["round_seed"]), int(view.get("n_tasks") or contract.params.n_tasks))
    n_draw = len(draw.task_ids)
    if first_tasks is not None:
        if not 1 <= first_tasks <= n_draw:
            raise ConfigError(f"--first-tasks must be from 1 to {n_draw}")
        draw = Draw(draw.snapshot_id, draw.round_seed, draw.task_ids[:first_tasks], draw.weights[:first_tasks])
    shown = {"round_id": view.get("round_id"), "round_seed": draw.round_seed, "irt_snapshot_id": draw.snapshot_id, "n_tasks": n_draw, "scored_tasks": len(draw.task_ids)}
    return Round(shown, draw, view, box_file)


def round_notes(contract: Contract, the: Round) -> list[dict[str, str]]:
    """Differences between this run and the real round whose seed it uses."""
    notes, view, shown = [], the.view, the.shown
    head, params_hash = view.get("head_version_id"), view.get("params_hash")
    if head and head != contract.head.version_id:
        notes.append({"what": "round-head", "text": f"Round {view.get('round_id')} ran on head {head}. This run uses the head of mining.json, {contract.head.version_id}."})
    if params_hash and params_hash != contract.params.params_hash:
        notes.append({"what": "round-params", "text": f"Round {view.get('round_id')} ran with params {params_hash}. This run uses the params of mining.json, {contract.params.params_hash}."})
    if shown["scored_tasks"] != shown["n_tasks"]:
        least = contract.params.min_scored_tasks
        few = f" With fewer than {least} scored tasks no LoRA can win." if shown["scored_tasks"] < least else ""
        notes.append({"what": "first-tasks", "text": f"Only the first {shown['scored_tasks']} of the {shown['n_tasks']} tasks of the draw are scored, in the order of the round. The looks are shares of {shown['scored_tasks']}.{few}"})
    return notes


def against_box(box_file: Mapping[str, Any], head: HeadVersion, draw: Draw, null: Sequence[str]) -> dict[str, Any]:
    """This run next to the public file of the real round: the same head, the same draw, and how often null M has the same outcome.

    Null M is compared only when the head and the draw are the same and one box scored the round.
    """
    theirs, n = box_file.get("draw") or {}, len(draw.task_ids)
    same_draw = list(theirs.get("task_ids") or [])[:n] == list(draw.task_ids) and list(theirs.get("weights") or [])[:n] == list(draw.weights)
    report: dict[str, Any] = {"round_id": box_file.get("round_id"), "same_head": box_file.get("head_version_id") == head.version_id, "same_draw": same_draw}
    boxes = list((box_file.get("null_outcomes") or {}).values())
    if report["same_head"] and same_draw and null and len(boxes) == 1:
        pairs = [(a, b) for here, there in zip(null, boxes[0]) for a, b in zip(here, there) if NOT_RUN not in (a, b)]
        report["null_m"] = {
            "rollouts_compared": len(pairs),
            "same_outcome": sum(a == b for a, b in pairs),
            "passed_here": sum(a == PASS for a, _ in pairs),
            "passed_on_box": sum(b == PASS for _, b in pairs),
        }
    return report


# ---- run ----


def applied_train(params: Params, train: Mapping[str, Any]) -> dict[str, Any]:
    """The values the trainer uses for this ``train`` object. Raises ``BatchRejected``; a value is never clamped."""
    recipe = params.recipe.for_miner(train)
    return {bound.key: to_jsonable(getattr(recipe, bound.key)) for bound in params.recipe.miner_bounds}


def run_round(
    raw: bytes,
    hotkey: str,
    params: Params,
    local: Params,
    head: HeadVersion,
    draw: Draw,
    built: Parts,
    *,
    tokens: Tokens | None = None,
    grader: Grader | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """One batch as a round of one entry: the validator's file check, then the box's runner and the judging.

    ``params`` are the ones of mining.json: the file check uses them. ``local`` are the ones of this machine
    (``local_params``): the round runs with them.
    """
    batch = check_batch(raw, hotkey, params, tokens=tokens, grader=grader)
    report: dict[str, Any] = {
        "valid": True,
        "format": batch.format,
        "batch_sha256": batch.sha256,
        "n_prompts": batch.n_prompts,
        "n_rollouts": batch.n_rollouts,
        "dropped_groups": batch.dropped_groups,
        "rules_run": list(batch.checked),
        "train_applied": applied_train(params, batch.train),
    }
    key, now = layout.frozen_batch(ROUND_ID, hotkey), int(clock())
    built.store.put(key, raw)
    frozen = FrozenBatch(hotkey, batch.sha256, key, batch.format, batch.n_prompts, batch.n_rollouts, dict(batch.train))
    ticket = RoundTicket(ROUND_ID, LEASE_ID, BOX_ID, head, local, draw, (Entry(0, Registration(0, hotkey, 0), frozen),), now, now + local.lease_timeout_s)
    runner = (
        SoloRunner(built.trainer, built.evaluator, lambda: int(clock()), built.image)
        if local.eval_merged
        else LocalRunner(built.trainer, built.evaluator, lambda: int(clock()), built.image)
    )
    result = runner.run(ticket)
    decision = judging.decide([(ticket, result)])
    row, mine = decision.entries[0], result.entries[0]
    report |= {"status": row.status, "reason": row.reason or decision.reason, "eliminated_at_look": row.eliminated_at_look, "score": row.score and row.score.to_dict()}
    report |= {"null_outcomes": list(result.null_outcomes), "outcomes": list(mine.outcomes), "stopped_at_limit": result.stopped_at_limit, "phase_s": dict(result.phase_s)}
    report["kl_gate"] = {"enabled": local.kl_floor_on, "min_kl_seq": local.min_kl_seq, "min_kl_tok": local.min_kl_tok, "kl_rollouts": local.kl_rollouts, "kl_max_tokens": local.kl_max_tokens, "measured": mine.kl_seq is not None, "sample": 0, "source": "sn120_v2/gpu/kl.py"}
    report["kl_gate"] |= {name: getattr(mine, name) for name in ("kl_seq", "kl_seq_se", "kl_tok", "kl_k", "kl_tokens") if getattr(mine, name) is not None}
    if mine.kl_seq is not None:
        report["kl_gate"]["passed"] = all((
            local.min_kl_seq <= 0 or mine.kl_seq >= local.min_kl_seq,
            local.min_kl_tok <= 0 or mine.kl_tok >= local.min_kl_tok,
        ))
    report["win_vs_null"] = {"won": row.status == "winner", "status": row.status, "reason": row.reason or decision.reason, "score": row.score.to_dict() if row.score else None}
    if row.score is not None:
        n = row.score.n_tasks
        hits = judging.code_count(mine.outcomes, CAP, n)
        null_hits = judging.code_count(result.null_outcomes, CAP, n)
        rollouts = n * len(mine.outcomes)
        null_rollouts = n * len(result.null_outcomes)
        report["cap_hit"] = {"passed": judging.cap_hit_ok(hits, null_hits, rollouts, local, null_rollouts), "rate": row.score.cap_rate, "null_rate": row.score.null_cap_rate, "limit": judging.cap_limit(row.score.null_cap_rate, local), "margin": local.cap_hit_margin, "floor": local.cap_hit_floor, "hits": hits, "rollouts": rollouts}
    if mine.lora is not None:
        report["lora"] = {"lora_hash": mine.lora.lora_hash, "dir": str(built.store.root.joinpath(*mine.lora.prefix.split("/"))), "train_report": mine.lora.report.to_dict()}
    return report


def head_tokens(directory: Path) -> Tokens:
    """Token counts with the tokenizer files of the head, as the validator's file check counts them."""
    return HFTokens(AutoTokenizer.from_pretrained(str(directory), local_files_only=True))


def compare(contract: Contract, gpus: int | None, round_clock: bool = False) -> dict[str, Any]:
    """The box and this machine side by side, and every difference a run here would have. Starts nothing."""
    params, recipe = contract.params, contract.params.recipe
    image, code, found = install_digest(), code_digest(), gpu_count()
    _, notes = local_params(params, image, code, params.n_gpus if gpus is None else gpus, round_clock)
    return {
        "box": {"params_hash": params.params_hash, "recipe_id": params.recipe_id, "image_digests": list(recipe.image_digests), "code_digest": recipe.code_digest, "n_gpus": params.n_gpus},
        "here": {"image_digest": image, "code_digest": code, "gpus_found": found, "n_gpus": params.n_gpus if gpus is None else gpus},
        "differences": notes,
    }


def run(args: argparse.Namespace, contract: Contract) -> dict[str, Any]:
    """The ``run`` command: check the file, train the LoRA, score it and null M, and judge the round of one batch."""
    params, head = contract.params, contract.head
    if params.env != ENV_ID:
        raise ConfigError(f"this tool has no task pool for environment {params.env!r}")
    gpus = params.n_gpus if args.gpus is None else args.gpus
    found = gpu_count()
    if gpus < 1 or found is None or gpus > found:
        raise ConfigError(f"--gpus is {gpus} (the box has {params.n_gpus}) and this machine has {found or 'no'} GPU(s) that nvidia-smi lists; pass --gpus with a number it has")
    image, code = install_digest(), code_digest()
    local, notes = local_params(params, image, code, gpus, args.round_clock)
    refused = [note["text"] for note in notes if note["what"] in HARDWARE]
    if args.strict and refused:
        raise ConfigError("--strict: this machine does not equal the box. " + " ".join(refused))
    work = Path(args.work).resolve()
    resident = BaseOnly(LocalStore(work / "store"), work)
    resident.merged = params.eval_merged
    base = base_dir(resident, head)
    if params.eval_merged:
        if not (contract.manifest.master_files or contract.doc.get("head_chain")):
            raise ConfigError("eval_merged needs the fp32 master and its files in the head manifest; run the fetch command first")
        # Fail before admission, training or any GPU process. The merged model
        # cannot be reconstructed from the bf16 base alone.
        resident.tree(layout.public_master_prefix(head.base_fp32_master_digest), head.base_fp32_master_digest)
    the = the_round(contract, round_id=args.round, round_file=args.round_file, round_seed=args.round_seed, irt_snapshot=args.irt_snapshot, first_tasks=args.first_tasks)
    notes += round_notes(contract, the)
    pool = pool_for_draw(the.draw.task_ids, args.pool)
    if isinstance(the.view.get("pool"), Mapping):
        from .pool_catalog import public_catalog
        pool, active = public_catalog(contract, Path(args.work), the.view, pool)
        if set(the.draw.task_ids) - set(active):
            raise VerifyError("the round draw contains a task outside its rotated active set")
    if pool.sha256 != POOL_SHA256:
        notes.append({"what": "pool", "text": f"This draw needs the archived v2.0 task pool (sha256 {pool.sha256}). Its task ids and answer keys are used unchanged."})
    for note in notes:
        say(f"differs from the box [{note['what']}]: {note['text']}")
    raw = read(args.batch)
    fold_input = None
    if local.eval_merged:
        # Head records omit the model repository/revision, which are publication
        # facts, not Params. FoldInputs uses its pinned lineage guard as on prod.
        from .params import GENESIS_BASE_HASH, GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION, GenesisSpec
        genesis = GenesisSpec(GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION, GENESIS_BASE_HASH, head.base_fp32_master_digest, "local-replication")
        fold_input = FoldInputs(Path(args.work).resolve(), genesis)
    built = parts(Path(args.work), gpus, pool=pool, fold_input=fold_input)
    admission = LazyPool(lambda: GradePool(workers=GRADE_WORKERS, timeout_s=GRADE_TIMEOUT_S))
    try:
        tokens = head_tokens(base)
        grader = pool_grader(built.evaluator.pool, admission, chunk=GRADE_WORKERS)
        result = run_round(raw, args.hotkey, params, local, head, the.draw, built, tokens=tokens, grader=grader)
    finally:
        admission.close()
        closer = getattr(built.evaluator.grader, "close", None)
        if closer:
            closer()
    result |= {
        "head_version_id": head.version_id,
        "base_hash": head.base_hash,
        "round": the.shown,
        "box": {"params_hash": params.params_hash, "recipe_id": params.recipe_id, "n_gpus": params.n_gpus},
        "here": {"image_digest": image, "code_digest": code, "n_gpus": gpus},
        "differences": notes,
    }
    if the.box_file is not None:
        result["box_round"] = against_box(the.box_file, head, the.draw, result["null_outcomes"])
    return result


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m sn120_v2.replicate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)
    for name in ("compare", "fetch", "run"):
        command = commands.add_parser(name)
        command.add_argument("--mining-json", required=True, help="mining.json: a file or an http(s) address")
        if name != "fetch":
            command.add_argument("--gpus", type=int, help="GPUs to use, 0 to N-1 (default: n_gpus of the params, the count of the box)")
            command.add_argument("--round-clock", action="store_true", help="keep the time limits of a round (default: no time limit of the round)")
    rotation = commands.add_parser("rotation", help="recompute a published rotation from public artifacts")
    rotation.add_argument("--mining-json", required=True)
    rotation.add_argument("--round", required=True, help="round id, rNNNNNN")
    rotation.add_argument("--work", required=True)
    for name in ("compare", "fetch", "run"):
        command = commands.choices[name]
        if name != "compare":
            command.add_argument("--work", required=True, help="work directory on a local disk: the model cache, the local store, the logs")
    fetcher, runner = commands.choices["fetch"], commands.choices["run"]
    fetcher.add_argument("--from", dest="source_dir", help="take the base files from this directory, and download nothing")
    fetcher.add_argument("--master-from", help="with eval_merged, take the fp32-master files from this directory (default: the current head's master links)")
    fetcher.add_argument("--read-links", help="the links file: a file or an address (default: the one mining.json names)")
    runner.add_argument("batch", help="your batch file")
    runner.add_argument("--hotkey", required=True, help="the ss58 address of your hotkey")
    which = runner.add_mutually_exclusive_group()
    which.add_argument("--round", help="repeat this round of mining.json (default: the open round, else the last closed one)")
    which.add_argument("--round-file", help="the public round file of a closed round (a file or an address): its seed and snapshot, and a comparison with it")
    which.add_argument("--round-seed", help="a round seed of your choice; needs --irt-snapshot")
    runner.add_argument("--irt-snapshot", help="the IRT snapshot: a file or an address (default: the public object of the round's snapshot id)")
    runner.add_argument("--pool", choices=("auto", "current", "v2.0"), default="auto", help="task pool for the draw (default: archived v2.0 when a task left the current pool)")
    runner.add_argument("--first-tasks", type=int, help="score only the first N tasks of the draw")
    runner.add_argument("--strict", action="store_true", help="refuse unless this machine has the pinned install, the pinned code and the GPU count of the box")
    runner.add_argument("--out", help="also write the report to this file")
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        contract = Contract.load(args.mining_json)
        if args.command == "compare":
            report = compare(contract, args.gpus, args.round_clock)
        elif args.command == "rotation":
            from .pool_catalog import public_rotation_inputs
            from .pool_runtime import replay_rotation
            view = contract.round(args.round)
            if not view.get("pool"):
                raise ConfigError(f"round {args.round} has no public rotation block")
            store, _ = public_rotation_inputs(contract, Path(args.work), view)
            report = replay_rotation(store, args.round)
        elif args.command == "fetch":
            report = fetch(contract, Path(args.work), source_dir=Path(args.source_dir) if args.source_dir else None, master_dir=Path(args.master_from) if args.master_from else None, links_source=args.read_links)
        else:
            report = run(args, contract)
    except Invalid as exc:
        print(json.dumps({"valid": False, "rule": exc.rule, "reason": str(exc)}, indent=1))
        return 1
    except V2Error as exc:
        print(json.dumps({"error": type(exc).__name__, "reason": str(exc)}, indent=1))
        return 2
    text = json.dumps(report, indent=1)
    if getattr(args, "out", None):
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
