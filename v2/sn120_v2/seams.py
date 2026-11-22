"""The seams: what the controller needs from the parts it does not own.

Each sibling workstream implements one or more of these protocols. The
controller, the CPU simulation and the tests depend only on this file, on
``types.py`` and on ``params.py``.

Two rules hold for every implementation:

* Raise ``InfraError`` when our infrastructure fails (timeout, crash, store
  unavailable). The controller turns it into ``no_decision``: the miner is not
  charged and nothing is promoted.
* Raise ``VerifyError`` when a hash, digest or manifest does not match. Never
  continue on unverified bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .params import Params
from .types import (
    BoxResult,
    ChainSnapshot,
    Draw,
    EvalChunk,
    EvalRequest,
    FoldArtifact,
    FrozenBatch,
    HeadVersion,
    LoRAArtifact,
    Registration,
    RoundTicket,
    TrainRequest,
    Verdict,
    WeightReceipt,
)


@runtime_checkable
class ObjectStore(Protocol):
    """A flat key/value store of bytes (R2 in production). Keys come from ``layout.py``.

    Every implementation behaves the same way, because the simulation runs
    on the local store and production runs on R2:

    * ``list(prefix)`` returns sorted full keys. It matches by path segment:
      a key matches when it equals ``prefix`` or starts with ``prefix + "/"``.
      So lease ``r000001-l1`` never matches lease ``r000001-l10``. An empty
      prefix, or one that ends in "/", is a plain string prefix.
    * ``get``, ``get_file``, ``size``, ``sha256`` and ``copy`` raise
      ``KeyError(key)`` when the source key is missing. ``exists`` returns False.
    * ``delete`` of a missing key is not an error.
    * ``put``, ``put_file`` and ``copy`` overwrite an existing key.
      ``copy(k, k)`` does nothing once the key is known to exist.
    * ``get_file`` creates parent directories and writes through a temporary
      file and a rename, so a failed download leaves no partial file.
    * Every other failure (network, permission, server error, short read,
      size mismatch after a write) raises ``InfraError``.
    """

    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def put_file(self, key: str, path: Path) -> None:
        """Upload a large file from disk."""
        ...

    def get_file(self, key: str, path: Path) -> None:
        """Download a large object to disk."""
        ...

    def exists(self, key: str) -> bool: ...

    def size(self, key: str) -> int: ...

    def sha256(self, key: str) -> str:
        """The real sha256 of the stored bytes, not a value the uploader claimed."""
        ...

    def copy(self, src: str, dst: str) -> None:
        """Copy inside the store. It must work for multi-gigabyte objects."""
        ...

    def delete(self, key: str) -> None: ...

    def list(self, prefix: str) -> list[str]: ...


@runtime_checkable
class Presigner(Protocol):
    """Time-limited links, so a box or a miner needs no store credentials."""

    def presign_get(self, key: str, ttl_s: int) -> str: ...

    def presign_put(self, key: str, ttl_s: int) -> tuple[str, dict[str, str]]:
        """Return the upload URL and the headers the uploader must send."""
        ...


@runtime_checkable
class Admission(Protocol):
    """Format and content checks on miner files."""

    def verdicts(self, hotkeys: Sequence[str]) -> Mapping[str, Verdict]:
        """The latest live check of each hotkey's stored file. A hotkey with no file may be absent."""
        ...

    def freeze(self, reg: Registration, round_id: str) -> FrozenBatch | None:
        """Copy the stored file for this round and check it again.

        Return ``None`` when the file is missing or invalid at this moment:
        the seat is skipped, not charged, and keeps its place.
        """
        ...


@runtime_checkable
class TaskSampler(Protocol):
    """The public task draw and the shared IRT table. Only the controller calls ``update``."""

    def snapshot_id(self) -> str:
        """Id of the latest published snapshot."""
        ...

    def draw(self, head: HeadVersion, round_seed: str, n_tasks: int) -> Draw:
        """A pure function of the latest snapshot and the public round seed."""
        ...

    def update(
        self,
        head: HeadVersion,
        draw: Draw,
        outcomes: Mapping[str, Sequence[str]],
        *,
        winner_lora_hash: str = "",
        new_head_version_id: str = "",
    ) -> str:
        """Fold one round's graded rollouts into the table and return the new snapshot id.

        ``outcomes`` maps ``NULL_MODEL`` and each ``lora_hash`` to outcome
        strings over ``draw.task_ids``. When a winner was applied, the two
        keyword arguments say which LoRA became which head.
        """
        ...

    def public_snapshot(self) -> Mapping[str, Any]:
        """The snapshot to publish: difficulties, abilities and sampling weights."""
        ...


@runtime_checkable
class Trainer(Protocol):
    """Runs the published recipe on one frozen batch."""

    def train(self, request: TrainRequest) -> LoRAArtifact:
        """Train one LoRA and upload its files under ``request.out_prefix``.

        Before training, check that the resident head matches ``request.head``
        (base hash and fp32 master digest), and fail closed if not. Refuse a
        non-finite loss, gradient or weight. Raise ``BatchRejected`` when the
        frozen batch itself is unusable.
        """
        ...

    def close(self) -> None:
        """Release every GPU reference. The runner calls this before any evaluation starts."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores null M and LoRAs on the same tasks, seeds and caps."""

    def evaluate(self, request: EvalRequest) -> EvalChunk:
        """Score one look. Null M is always scored, with the same settings as every LoRA."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class Folder(Protocol):
    """Merges the stacked LoRAs of a head into a new base."""

    def fold(self, head: HeadVersion, params: Params, out_prefix: str) -> FoldArtifact:
        """Write a new bf16 base and a new fp32 master under ``out_prefix``.

        Start from the fp32 master named by ``head.base_fp32_master_digest``.
        Fail closed if that master is missing or its digest differs.
        """
        ...


@runtime_checkable
class RoundRunner(Protocol):
    """One box: trains and scores its leased entries, then returns a sealed result."""

    def run(self, ticket: RoundTicket) -> BoxResult: ...


@runtime_checkable
class Publisher(Protocol):
    """Writes the miner contract and the website data from one snapshot."""

    def publish(self, snapshot: Mapping[str, Any]) -> None:
        """Write llms.txt and mining.json together, from this one snapshot."""
        ...


@runtime_checkable
class ChainReader(Protocol):
    """Read-only chain access."""

    def snapshot(self) -> ChainSnapshot:
        """Every registration of the subnet at one block. Raise ``InfraError`` on a partial read."""
        ...

    def block_hash(self, block: int) -> str: ...


@runtime_checkable
class WeightSetter(Protocol):
    """The single writer of chain weights."""

    def set_weights(self, weights: Mapping[int, float]) -> WeightReceipt:
        """Plan first, then submit. On a rate limit or a recent write, return ``deferred``: never loop."""
        ...
