"""Error types. The type of an error decides who pays for it.

The controller maps each type to one outcome, so raise the most specific one.
"""

from __future__ import annotations


class V2Error(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(V2Error):
    """A setting or environment variable is missing or invalid. Fail closed: never fall back to a host path."""


class InfraError(V2Error):
    """Our infrastructure failed (box died, OOM, timeout, store unavailable).

    Never the miner's fault. The entry becomes ``no_decision``: not charged,
    back to the front of the queue, and nothing is promoted.
    """


class BatchRejected(V2Error):
    """A frozen batch failed a check at train time. The entry is a ``skip``: not charged, keeps its place."""


class VerifyError(V2Error):
    """A checksum, manifest or digest does not match. Fail closed: never promote, never train on it."""


class HeadMismatchError(V2Error):
    """A ticket, a result or resident weights name a head that is not the expected head.

    A result against any head but the current one can never be promoted.
    """


class LeaseError(V2Error):
    """A result names a lease that is unknown, expired or already settled."""


class ContractError(V2Error):
    """A ticket, result or record is malformed (wrong field, wrong length, wrong id)."""


class FencedError(V2Error):
    """This journal writer lost the fence: another controller took over. Stop writing."""


class JournalCorrupt(V2Error):
    """The journal fails its hash chain or cannot be parsed. Fail closed: do not start."""
