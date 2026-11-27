"""The head chain: the public record from which anyone rebuilds a head whose weight files are not in the store.

A fold makes a new fp32 master and a new bf16 base from the old master and the
stacked LoRAs. In ``chain`` mode (``MODE_ENV``) the box keeps the folded weight
files on its own disk. The store gets the LoRA of each winner, the small files
of each base, and the sha256 of every folded file (the head manifest). A head
whose full files are in the store is a **checkpoint**: a marker object says so,
and the marker is written last. The genesis head is always a checkpoint.

``chain_doc`` builds ``layout.HEAD_CHAIN`` from the journal's state::

    {"schema", "head_version_id", "recipe",
     "genesis": {"version_id", "base_hash", "base_fp32_master_digest", "checkpoint": true},
     "steps": [{"from_version_id", "stacked_version_id", "version_id", "round_id",
                "loras": [{"lora_hash", "round_id", "miner_uid", "rank", "files": {name: sha256}}],
                "kernel", "base_hash", "base_fp32_master_digest", "checkpoint"}],
     "stacked": null, or the LoRAs of a head that is not folded yet,
     "start": {"version_id", "base_hash", "base_fp32_master_digest", "checkpoint": true, "steps_to_head"}}

One step is one fold. ``from_version_id`` is the folded head before it,
``stacked_version_id`` the head with the LoRAs on it, ``version_id`` the folded
head after it. The two ids of a result pin every file: ``public/heads/<version_id>.json``
has the sha256 of each one. ``kernel`` names the arithmetic of the LoRA product;
a reader that does not know a name must stop. ``start`` is the newest checkpoint.

mining.json shows ``tail(doc)``: the same record without ``genesis`` and
``recipe``, and with only the steps after ``start``. A reader needs no more.

This module has no torch import: the publisher and the site's pull job use it.
The fold, the checkpoint upload and the rebuild are in ``headfold.py``.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable, Iterable, Mapping

from . import layout
from .canonical import manifest_digest
from .errors import ConfigError, InfraError, VerifyError
from .seams import ObjectStore
from .state import State
from .store import promote_files
from .types import FoldArtifact, HeadManifest, HeadVersion

SCHEMA = "sn120-head-chain-v1"
CHECKPOINT_SCHEMA = "sn120-head-checkpoint-v1"
FOLD_SCHEMA = "sn120-head-fold-v1"
MODE_ENV, FULL, CHAIN = "SN120_HEAD_PUBLISH", "full", "chain"
EVERY_ENV, DEFAULT_EVERY = "SN120_CHECKPOINT_EVERY", 10
WEIGHT_SUFFIX = ".safetensors"
FOLD_RECORD = "fold.json"  # the fold record in the fold's own prefix under pending/, until the controller publishes it
# The kernel of ``gpu/folder.py: read_lora``: (lora_alpha / r) * (B @ A) in float32, by torch's CPU matrix product.
# A fold that has no fold record was made by that code, so it has this name.
MATMUL = "torch-matmul-v1"
# For each element: the ranks in order 0, 1, 2, ..., each one a fused multiply-add from +0; then the scale as one fp32 multiply.
FIXED_ORDER = "fixed-order-v1"
KERNELS = (MATMUL, FIXED_ORDER)  # a kernel name is a rule; an implementation is any code that gives the same bytes
RECIPE = {
    "master_dtype": "float32",
    "delta": "(lora_alpha / r) * (B @ A) in float32, by the kernel that the step names",
    "order": "base weight files in name order; for each tensor: the master plus the delta of each LoRA of the step, in list order",
    "base_cast": "each tensor of the new master, cast to the dtype that the tensor has in the old base",
    "master_part_bytes": 4 * 1024**3,
    "small_files": "each base file that is not a weight file is copied unchanged",
}
_CHECKPOINTS = layout.head_checkpoint("x").partition("x")[0]
_IDS = ("version_id", "base_hash", "base_fp32_master_digest")


def mode(environ: Mapping[str, str] = os.environ) -> str:
    """``full``: the box uploads each folded head (the default). ``chain``: it keeps the weights local."""
    value = environ.get(MODE_ENV, "").strip() or FULL
    if value not in (FULL, CHAIN):
        raise ConfigError(f"{MODE_ENV} must be {FULL!r} or {CHAIN!r}")
    return value


def checkpoint_every(environ: Mapping[str, str] = os.environ) -> int:
    """After this many folds with no checkpoint, the box uploads one in the background. 0: never by itself."""
    text = environ.get(EVERY_ENV, "").strip()
    if not text:
        return DEFAULT_EVERY
    if not text.isdigit():
        raise ConfigError(f"{EVERY_ENV} must be a whole number, 0 or more")
    return int(text)


def small_files(files: Mapping[str, str]) -> dict[str, str]:
    """The files of a base that are not weight files: the config and the tokenizer."""
    return {name: sha for name, sha in files.items() if not name.endswith(WEIGHT_SUFFIX)}


def checkpoints(store: ObjectStore) -> set[str]:
    """The version ids that have a checkpoint marker in the store."""
    found = set()
    for key in store.list(_CHECKPOINTS.rstrip("/")):
        version_id = key[len(_CHECKPOINTS) :].removesuffix(".json")
        if "/" not in version_id and key == layout.head_checkpoint(version_id):
            found.add(version_id)
    return found


def marker(manifest: HeadManifest, base_sizes: Mapping[str, int], master_sizes: Mapping[str, int]) -> dict[str, Any]:
    """The content of a checkpoint marker: the ids of the head and the size of each file. No time: a second write gives the same bytes."""
    head = manifest.head
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version_id": head.version_id,
        "base_hash": head.base_hash,
        "base_fp32_master_digest": head.base_fp32_master_digest,
        "base_sizes": dict(base_sizes),
        "master_sizes": dict(master_sizes),
    }


def promote_fold(store: ObjectStore, fold: FoldArtifact) -> None:
    """What the controller publishes of a fold in ``chain`` mode: the small files of the new base, and the fold record.

    The record names the kernel of the fold. Without it a reader would take
    the fold for one of ``MATMUL``, so a fold without a record is refused.
    """
    promote_files(store, f"{fold.prefix}/base", layout.public_base_prefix(fold.base_hash), small_files(fold.base_files))
    try:
        data = store.get(f"{fold.prefix}/{FOLD_RECORD}")
        record = json.loads(data)
        named = (record["schema"], record["stacked_version_id"], record["base_hash"], record["base_fp32_master_digest"], bool(record["kernel"]))
    except (KeyError, ValueError, TypeError):
        raise VerifyError("the fold left no complete fold record; the head does not move") from None
    if named != (FOLD_SCHEMA, fold.parent_version_id, fold.base_hash, fold.base_fp32_master_digest, True):
        raise VerifyError("the fold record is of another fold; the head does not move")
    key = layout.head_fold(fold.parent_version_id)
    store.put(key, data)
    if store.get(key) != data:
        raise InfraError(f"{key} does not read back as written")


def _ids(head: HeadVersion) -> dict[str, str]:
    return {"version_id": head.version_id, "base_hash": head.base_hash, "base_fp32_master_digest": head.base_fp32_master_digest}


def _loras(manifest: HeadManifest) -> list[dict[str, Any]]:
    return [
        {"lora_hash": ref.lora_hash, "round_id": ref.round_id, "miner_uid": ref.miner_uid, "rank": ref.rank, "files": dict(manifest.lora_files[ref.lora_hash])}
        for ref in manifest.head.loras
    ]


def heads_of(doc: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The folded heads of a chain record, oldest first. The first one is a checkpoint: genesis, or ``start`` of a tail."""
    return [doc["genesis"] if "genesis" in doc else doc["start"], *doc["steps"]]


def with_start(doc: dict[str, Any]) -> dict[str, Any]:
    """Set ``start`` of a whole chain record: the newest checkpoint, and the count of steps from it to the head."""
    heads = heads_of(doc)
    behind = next(count for count, head in enumerate(reversed(heads)) if head["checkpoint"])
    start = heads[len(heads) - 1 - behind]
    doc["start"] = {key: start[key] for key in _IDS} | {"checkpoint": True, "steps_to_head": behind}
    return doc


def chain_doc(state: State, marked: Iterable[str], kernel_of: Callable[[str], str]) -> dict[str, Any]:
    """The chain of the state. ``marked``: the heads with a checkpoint marker. ``kernel_of(stacked id)``: the kernel of a fold."""
    marked, genesis = set(marked), state.heads[0]
    steps, last = [], genesis.version_id
    for head in state.heads[1:]:
        if not head.folded:
            continue
        stacked = state.manifests[head.parent_version_id]
        steps.append(
            {"from_version_id": last, "stacked_version_id": stacked.head.version_id, "round_id": head.round_id, "loras": _loras(stacked)}
            | {"kernel": kernel_of(stacked.head.version_id), "checkpoint": head.version_id in marked}
            | _ids(head)
        )
        last = head.version_id
    current = state.head
    doc = {
        "schema": SCHEMA,
        "head_version_id": current.version_id,
        "recipe": RECIPE,
        "genesis": _ids(genesis) | {"checkpoint": True},
        "steps": steps,
        "stacked": None if current.folded else {"version_id": current.version_id, "loras": _loras(state.manifests[current.version_id])},
    }
    if state.reverts:  # operator head reverts: the steps end at the target; the reverted-away heads are named here
        doc["reverts"] = [dict(revert) for revert in state.reverts]
    return with_start(doc)


def tail(doc: Mapping[str, Any]) -> dict[str, Any]:
    """What mining.json shows of a whole chain record: ``start`` and the steps after it. Enough to rebuild the head."""
    behind = doc["start"]["steps_to_head"]
    part = {key: value for key, value in doc.items() if key not in ("genesis", "recipe", "steps")}
    return part | {"file": layout.HEAD_CHAIN, "steps": doc["steps"][len(doc["steps"]) - behind :]}


def check_chain(doc: Mapping[str, Any]) -> None:
    """Fail closed unless ``doc`` is a chain (whole, or a tail) whose steps link up and whose LoRA files hash to their ids."""
    try:
        if doc["schema"] != SCHEMA:
            raise VerifyError(f"the head chain has the schema {doc['schema']!r}; this code reads {SCHEMA!r}")
        heads = heads_of(doc)
        if heads[0]["checkpoint"] is not True:
            raise VerifyError("the head chain does not start at a checkpoint")
        for before, step in zip(heads, heads[1:]):
            if step["from_version_id"] != before["version_id"]:
                raise VerifyError(f"the step to head {step['version_id'][:12]} does not start at the head before it")
        stacked = doc["stacked"]
        for lora in [lora for step in doc["steps"] for lora in step["loras"]] + (stacked["loras"] if stacked else []):
            if manifest_digest(lora["files"]) != lora["lora_hash"]:
                raise VerifyError(f"the files of LoRA {lora['lora_hash'][:12]} do not hash to its id")
        if doc["head_version_id"] != (stacked["version_id"] if stacked else heads[-1]["version_id"]):
            raise VerifyError("the head chain does not end at the head that it names")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise VerifyError(f"the head chain is not complete: {type(exc).__name__}: {exc}") from None


def route(doc: Mapping[str, Any], target: str, held: Callable[[Mapping[str, Any]], bool]) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    """(the head to start from, the steps to replay) for the folded head ``target``.

    The walk goes back from ``target``. It stops at the first head that
    ``held`` says is on this machine. With none, it stops at the first checkpoint.
    """
    heads = heads_of(doc)
    at = next((index for index, head in enumerate(heads) if head["version_id"] == target), None)
    if at is None:
        raise VerifyError(f"head {target[:12]} is not a folded head of this chain record")
    for want in (held, lambda head: head["checkpoint"]):
        for index in range(at, -1, -1):
            if want(heads[index]):
                return heads[index], heads[index + 1 : at + 1]
    raise VerifyError("the head chain names no checkpoint")


def file_set(store: Any, set_id: str, files: Mapping[str, str], key_of: Callable[[str, str], str], ttl_s: int) -> dict[str, Any]:
    """One file set of a links file: a signed read link, the sha256 and the size of each file. A missing file is ``KeyError``."""
    rows = []
    for name, sha256 in sorted(files.items()):
        key = key_of(set_id, name)
        rows.append({"name": name, "key": key, "sha256": sha256, "size": store.size(key), "url": store.presign_get(key, ttl_s)})
    return {"id": set_id, "bytes": sum(row["size"] for row in rows), "files": rows}


def chain_links(store: Any, doc: Mapping[str, Any], ttl_s: int, target: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Signed read links for a chain: ``checkpoints`` (the base and the master of each one) and ``chain_loras`` (each step's).

    ``store`` is an ``ObjectStore`` that can sign (``presign_get``). With
    ``target``, a folded head: only the checkpoint to start from and the LoRAs
    of the steps to that head. The file names of a checkpoint come from its
    head manifest in the store.
    """
    heads = heads_of(doc)
    if target is None:
        marked, steps, stacked = [head for head in heads if head["checkpoint"]], doc["steps"], doc["stacked"]
    else:
        start, steps = route(doc, target, lambda head: False)
        marked, stacked = [start], None
    points = []
    for head in marked:
        manifest = HeadManifest.from_json(store.get(layout.head_manifest(head["version_id"])))
        if _ids(manifest.head) != {key: head[key] for key in _IDS}:
            raise VerifyError(f"the manifest stored for head {head['version_id'][:12]} is of another head")
        points.append(
            {"version_id": head["version_id"]}
            | {"base": file_set(store, head["base_hash"], manifest.base_files, layout.public_base_file, ttl_s)}
            | {"master": file_set(store, head["base_fp32_master_digest"], manifest.master_files, layout.public_master_file, ttl_s)}
        )
    loras = [lora for step in steps for lora in step["loras"]] + (stacked["loras"] if stacked else [])
    return {
        "checkpoints": points,
        "chain_loras": [file_set(store, lora["lora_hash"], lora["files"], layout.public_lora_file, ttl_s) | {"round_id": lora["round_id"]} for lora in loras],
    }


class ChainWriter:
    """Writes ``layout.HEAD_CHAIN``. The controller calls it with each publish; the checkpoint job calls ``mark``."""

    def __init__(self, store: ObjectStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._kernels: dict[str, str] = {}
        self._doc: dict[str, Any] | None = None

    def _kernel_of(self, stacked_version_id: str) -> str:
        """The kernel that the fold record names. A fold without a record was made by ``gpu/folder.py`` as it is."""
        if stacked_version_id not in self._kernels:
            key = layout.head_fold(stacked_version_id)
            record = json.loads(self.store.get(key)) if self.store.exists(key) else {}
            self._kernels[stacked_version_id] = str(record.get("kernel") or MATMUL)
        return self._kernels[stacked_version_id]

    def doc(self, state: State) -> dict[str, Any]:
        """The whole chain record of ``state``. Nothing is written. The markers are listed each time: another process can write one."""
        marked = checkpoints(self.store)
        with self._lock:
            return chain_doc(state, marked, self._kernel_of)

    def _put(self, doc: dict[str, Any]) -> None:
        if doc != self._doc:
            self.store.put(layout.HEAD_CHAIN, (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8"))
            self._doc = doc

    def __call__(self, state: State) -> dict[str, Any]:
        """Write the chain file when it changed. Returns the part of it that mining.json shows."""
        doc = self.doc(state)
        with self._lock:
            self._put(doc)
        return tail(doc)

    def mark(self, version_id: str) -> None:
        """A checkpoint marker of this head was written. The chain file says so now, not at the next publish."""
        with self._lock:
            if self._doc is not None and any(step["version_id"] == version_id and not step["checkpoint"] for step in self._doc["steps"]):
                doc = json.loads(json.dumps(self._doc))
                for step in doc["steps"]:
                    step["checkpoint"] = step["checkpoint"] or step["version_id"] == version_id
                self._put(with_start(doc))
