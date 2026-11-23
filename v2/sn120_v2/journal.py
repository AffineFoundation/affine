"""The journal: an append-only file that is the controller's only durable memory.

Each line is one record in canonical JSON. A record carries its own
checksum and the checksum of the record before it, so a missing, changed or
reordered line is detected. The state of the controller is always rebuilt
by replaying this file.

One writer at a time. A writer takes the next fence number when it opens the
journal. If another writer opens the journal later, the older writer's next
append fails with ``FencedError``, so two controllers can never both apply.

The object store holds the durable copy (``Mirror``): one object per record.
A record goes to the store first and to the local file second, so the local
file is only a cache. A machine with an empty disk gets the journal back
with ``pull``, and a controller on another machine fences this one through
the store.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import layout
from .canonical import canonical_json, sha256_hex
from .errors import FencedError, JournalCorrupt
from .seams import ObjectStore

LEADER = "leader"
JOURNAL_FILE = "journal.jsonl"  # the journal's name inside a state directory


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


@contextmanager
def _locked(path: Path) -> Iterator[Any]:
    with open(path, "a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _sync(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


class Mirror:
    """The durable copy of a journal in the object store: one object per record, never rewritten."""

    def __init__(self, store: ObjectStore) -> None:
        self.store = store

    def count(self) -> int:
        keys = sorted(self.store.list(layout.JOURNAL))
        if keys != [layout.journal_record(seq) for seq in range(len(keys))]:
            raise JournalCorrupt("the stored journal has a gap or an object that is not a record")
        return len(keys)

    def line(self, seq: int) -> bytes:
        return self.store.get(layout.journal_record(seq))

    def push(self, record: JournalRecord) -> None:
        """Store one new record. A record that is already there was written by another controller."""
        key = layout.journal_record(record.seq)
        if self.store.exists(key):
            raise FencedError(f"journal record {record.seq} is already in the store: another controller has taken over")
        self.store.put(key, record.line())

    def catch_up(self, handle: Any, backfill: bool) -> None:
        """Append to the local file every record that only the store has. ``backfill`` also stores records that only the file has."""
        records, covered = _parse(handle.read())
        stored = self.count()
        shared = min(len(records), stored)
        if shared and self.line(shared - 1) != records[shared - 1].line():
            raise JournalCorrupt("the local journal and the stored journal differ; move the local file away and start again")
        if stored > len(records):
            handle.truncate(covered)
            for seq in range(len(records), stored):
                handle.write(self.line(seq) + b"\n")
            _sync(handle)
        elif backfill:
            for record in records[stored:]:
                self.push(record)
        handle.seek(0)


def pull(store: ObjectStore, path: str | Path) -> list[JournalRecord]:
    """Bring the local copy at ``path`` up to date from the store, verify it, and return its records.

    It never writes to the store, so any machine may call it: the host that
    sets weights, a status command, or a new box before it takes over.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path) as handle:
        Mirror(store).catch_up(handle, backfill=False)
        return _parse(handle.read())[0]


class Journal:
    """The single writer of one journal file, and of its copy in the store when ``mirror`` is given."""

    def __init__(self, path: str | Path, owner: str, at: int, mirror: Mirror | None = None) -> None:
        self.path, self.mirror = Path(path), mirror
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _locked(self.path) as handle:
            if mirror is not None:
                mirror.catch_up(handle, backfill=True)
            self.records, covered = _parse(handle.read())
            handle.truncate(covered)  # drop a torn last line left by a crash
            self._size = covered
            self.fence = (self.records[-1].fence if self.records else 0) + 1
            self._write(handle, LEADER, {"owner": owner}, at)

    def _write(self, handle: Any, kind: str, data: dict[str, Any], at: int) -> JournalRecord:
        prev = self.records[-1].checksum if self.records else ""
        record = JournalRecord(len(self.records), self.fence, at, kind, data, prev)
        if self.mirror is not None:
            self.mirror.push(record)  # the store first: a record that is not durable never takes effect
        handle.seek(0, os.SEEK_END)
        handle.write(record.line() + b"\n")
        _sync(handle)
        self.records.append(record)
        self._size += len(record.line()) + 1
        return record

    def append(self, kind: str, data: dict[str, Any], at: int) -> JournalRecord:
        """Write one record durably. Fails with ``FencedError`` if another writer has taken over."""
        with _locked(self.path) as handle:
            if os.fstat(handle.fileno()).st_size != self._size:
                raise FencedError(f"journal {self.path} was written by another controller; this one must stop")
            return self._write(handle, kind, data, at)
