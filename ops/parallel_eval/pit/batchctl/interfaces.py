"""The seams of the batch controller.

The controller owns the decisions. Everything that touches the world is
behind one of these interfaces, so each side can change without the other:

``ObjectStore``  R2. Conditional writes are part of the contract.
``Provider``     rents, lists and destroys GPU boxes.
``BoxAgent``     starts the job on a box. The job itself is not in this package.
``BatchSource``  lists public miner batches. Read-only.
``Judge``        turns the bytes of a batch result into GREEN or RED.
``Clock``        the time.

Error rule for every interface: an exception that is not one of the
"definite" classes below means "outcome unknown". The controller then reads
the world again and retries. It never guesses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .layout import is_hex64

VERDICT_GREEN = 'green'
VERDICT_RED = 'red'


# ----------------------------------------------------------------------------
# Object store
# ----------------------------------------------------------------------------

class StoreError(RuntimeError):
    """The request failed, or nobody knows if it was applied. Retry."""


class NotFound(StoreError):
    """Definite: the key does not exist."""


class PreconditionFailed(StoreError):
    """Definite: a conditional write was refused and nothing was written."""


class TooLarge(StoreError):
    """Definite: the object is larger than the caller allows."""


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    etag: str


@dataclass(frozen=True)
class Blob:
    data: bytes
    etag: str


@dataclass(frozen=True)
class Digest:
    """Size and sha256 of the bytes that the store holds, with their etag."""

    size: int
    sha256: str
    etag: str


class ObjectStore(Protocol):
    """A strongly consistent object store with conditional writes.

    ``etag`` is an opaque string that changes when the bytes of a key change.
    """

    def get(self, key: str, *, max_bytes: int) -> Blob:
        """Read a whole object. ``NotFound``; ``TooLarge`` above ``max_bytes``."""

    def head(self, key: str) -> Optional[ObjectInfo]:
        """Size and etag, or None when the key does not exist."""

    def digest(self, key: str) -> Digest:
        """Hash the bytes that the store holds now. ``NotFound``.

        The plain implementation reads the object back and hashes it. An
        implementation may use a checksum that the store itself computed over
        the full object, but never a value that an uploader only claimed.
        """

    def put(
        self, key: str, data: bytes, *, if_none_match: bool = False, if_match: Optional[str] = None,
    ) -> str:
        """Write an object and return its etag.

        ``if_none_match``: create only; ``PreconditionFailed`` if the key exists.
        ``if_match``: replace only this etag; ``PreconditionFailed`` otherwise.
        """

    def copy(self, src: str, dst: str, *, src_etag: str) -> None:
        """Server-side copy of ``src`` to ``dst``, only while ``src`` still has
        ``src_etag``. ``PreconditionFailed`` if it changed; ``NotFound``."""

    def delete(self, key: str) -> None:
        """Delete a key. A missing key is not an error."""

    def list(self, prefix: str) -> List[ObjectInfo]:
        """Every key under the prefix."""


# ----------------------------------------------------------------------------
# Provider
# ----------------------------------------------------------------------------

class ProviderError(RuntimeError):
    """The request failed, or nobody knows if it was applied."""


class RentUnavailable(ProviderError):
    """Definite: no box was created (no offer, price too high, refused)."""


@dataclass(frozen=True)
class BoxInfo:
    box_id: str
    name: str
    provider: str
    gpu: str
    gpu_count: int
    tier: str
    price_usd_h: float


class Provider(Protocol):
    name: str

    def rent(
        self, *, name: str, gpu: str, tier: str, gpu_count: int, max_price_usd_h: float,
        ttl_s: float, timeout_s: float,
    ) -> BoxInfo:
        """Rent one box with exactly this name, GPU class and tier.

        Must return or raise within ``timeout_s``. ``ttl_s`` is the time after
        which the provider itself removes the box: a backstop for the case
        that nobody destroys it. ``RentUnavailable`` means no box exists. Any
        other error means a box may exist; the controller finds it by name.
        """

    def list(self) -> List[BoxInfo]:
        """Every box of the account. The controller filters by exact name."""

    def destroy(self, box_id: str) -> None:
        """Destroy one box by its exact id. A missing box is not an error."""


# ----------------------------------------------------------------------------
# Box agent and job contract
# ----------------------------------------------------------------------------

class BoxError(RuntimeError):
    """The job could not be started, or nobody knows if it was. Retry."""


@dataclass(frozen=True)
class JobSpec:
    """Everything a box needs. A box needs nothing else from the controller.

    Contract for the job on the box:

    1. Download the state ``base_state`` from ``base_prefix``. Check that the
       sha256 of ``manifest.json`` is ``base_state`` and that every file has
       the size and sha256 that the manifest names.
    2. Refuse to train unless the fp32 master digest of that state is
       ``expected_fp32_master_digest``. There is no override.
    3. Train on the batch ``batch_ref`` and probe on the same box.
    4. Write only under ``staging_prefix``: the state files under ``obj/``,
       then ``result.json``, then ``COMPLETE.json`` as the last object.
    5. Write ``HEARTBEAT.json`` with a growing ``seq`` at least every
       ``heartbeat_every_s`` seconds.
    6. A second launch with the same ``fence`` must not start a second job.
    """

    run_id: str
    batch_id: str
    batch_ref: str
    fence: int
    attempt: int
    base_state: str
    base_seq: int
    base_prefix: str
    expected_fp32_master_digest: str
    staging_prefix: str
    heartbeat_every_s: float
    job_ttl_s: float

    def __post_init__(self) -> None:
        if not is_hex64(self.base_state):
            raise ValueError('base_state must be a sha256')
        # Fail closed: a job without the expected fp32 master digest is never
        # built, so a box can never train on an unchecked master.
        if not is_hex64(self.expected_fp32_master_digest):
            raise ValueError('expected_fp32_master_digest must be a sha256')
        if self.fence < 1 or self.attempt < 1 or self.base_seq < 0:
            raise ValueError('fence, attempt and base_seq out of range')
        if not self.staging_prefix.endswith('/%012d/' % self.fence):
            raise ValueError('staging_prefix must end with the fence')
        if not self.base_prefix.endswith('/%s/' % self.base_state):
            raise ValueError('base_prefix must end with the base state id')

    def doc(self) -> Dict[str, Any]:
        return asdict(self)


class BoxAgent(Protocol):
    def launch(self, box_id: str, provider: str, spec: JobSpec) -> None:
        """Start the job on the box. Safe to call again for the same fence."""


# ----------------------------------------------------------------------------
# Batches and verdicts
# ----------------------------------------------------------------------------

class SourceError(RuntimeError):
    """The batch listing failed. Retry."""


@dataclass(frozen=True)
class BatchRef:
    batch_id: str
    miner: str
    ref: str = ''


class BatchSource(Protocol):
    def list(self) -> Sequence[BatchRef]:
        """Public batches in arrival order. Read-only.

        A batch must stay in the listing until the controller has admitted
        it. The controller ignores a batch that it knows already.
        """


class JudgeInvalid(ValueError):
    """The bytes are not a result for this batch and this base state."""


class Judge(Protocol):
    def verdict(self, result: bytes, *, batch_id: str, base_state: str) -> str:
        """Return ``VERDICT_GREEN`` or ``VERDICT_RED``.

        Called only with bytes that were read back from R2 and match the
        size and sha256 of the package. A result that says "the batch made
        the trainer fail" must be RED here, not ``JudgeInvalid``: an invalid
        result counts as an infrastructure loss, and a miner must not be
        able to buy that credit with a bad batch.
        """


class Clock(Protocol):
    def now(self) -> float:
        """Seconds since the epoch. Must not go backwards within a process."""
