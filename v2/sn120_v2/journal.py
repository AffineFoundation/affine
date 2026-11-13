"""The journal: an append-only file that is the controller's only durable memory.

Each line is one record in canonical JSON. A record carries its own
checksum and the checksum of the record before it, so a missing, changed or
reordered line is detected. The state of the controller is always rebuilt
by replaying this file.

One writer at a time. A writer takes the next fence number when it opens the
journal. If another writer opens the journal later, the older writer's next
append fails with ``FencedError``, so two controllers can never both apply.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_json, sha256_hex
from .errors import FencedError, JournalCorrupt

LEADER = "leader"


@dataclass(frozen=True)
class JournalRecord:
    seq: int
    fence: int
    at: int
    kind: str
    data: dict[str, Any]
    prev: str  # checksum of the record before this one; "" for the first record

    def _body(self) -> dict[str, Any]:
        return {"seq": self.seq, "fence": self.fence, "at": self.at, "kind": self.kind, "data": self.data, "prev": self.prev}

    @property
    def checksum(self) -> str:
        return sha256_hex(canonical_json(self._body()))

    def line(self) -> bytes:
        """The stored line: the record plus its own checksum, so damage to any line is detected."""
        return canonical_json(self._body() | {"sum": self.checksum})


def _parse(raw: bytes) -> tuple[list[JournalRecord], int]:
    """Return the verified records and the byte length they cover. A torn last line is left out."""
    records: list[JournalRecord] = []
    covered = 0
    for line in raw.split(b"\n")[:-1]:
        try:
            fields = json.loads(line)
            claimed = fields.pop("sum")
            record = JournalRecord(**fields)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise JournalCorrupt(f"journal line {len(records)} cannot be read: {exc!r}") from exc
        last = records[-1] if records else None
        follows = record.seq == len(records) and record.prev == (last.checksum if last else "") and record.fence >= (last.fence if last else 0)
        if not follows or claimed != record.checksum or record.line() != line:
            raise JournalCorrupt(f"journal line {len(records)} is damaged or out of place")
        records.append(record)
        covered += len(line) + 1
    return records, covered


def read_journal(path: str | Path) -> list[JournalRecord]:
    """Read and verify a journal without writing. Safe while a controller is appending."""
    return _parse(Path(path).read_bytes())[0]


class Journal:
    """The single writer of one journal file."""

    def __init__(self, path: str | Path, owner: str, at: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked() as handle:
            self.records, covered = _parse(handle.read())
            handle.truncate(covered)  # drop a torn last line left by a crash
            self._size = covered
            self.fence = (self.records[-1].fence if self.records else 0) + 1
            self._write(handle, LEADER, {"owner": owner}, at)

    @contextmanager
    def _locked(self) -> Iterator[Any]:
        with open(self.path, "a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0)
                yield handle
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _write(self, handle: Any, kind: str, data: dict[str, Any], at: int) -> JournalRecord:
        prev = self.records[-1].checksum if self.records else ""
        record = JournalRecord(len(self.records), self.fence, at, kind, data, prev)
        handle.seek(0, os.SEEK_END)
        handle.write(record.line() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
        self.records.append(record)
        self._size += len(record.line()) + 1
        return record

    def append(self, kind: str, data: dict[str, Any], at: int) -> JournalRecord:
        """Write one record durably. Fails with ``FencedError`` if another writer has taken over."""
        with self._locked() as handle:
            if os.fstat(handle.fileno()).st_size != self._size:
                raise FencedError(f"journal {self.path} was written by another controller; this one must stop")
            return self._write(handle, kind, data, at)
