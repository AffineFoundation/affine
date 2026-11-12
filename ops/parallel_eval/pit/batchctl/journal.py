"""Durable append-only journal for the batch controller.

One JSON record per line. Each record carries its index, the hash of the
previous record, and its own hash. That makes a hash chain: a missing, moved
or edited record breaks the chain and the journal refuses to open.

Durability rules
----------------
* ``append`` returns only after ``fsync``. The controller acts on a record
  only after ``append`` returns. This is the write-ahead rule.
* The file is created by atomic rename, so it never exists without a valid
  first record.
* A crash can leave a torn last line. ``open`` cuts the torn tail by atomic
  rename. A complete sealed record that does not continue the chain is not a
  torn tail; it is corruption, and ``open`` fails closed.
* A failed write or fsync poisons the journal object. The process must stop
  and reopen the file, because the on-disk state is unknown.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ...durable import canonical

GENESIS_HASH = '0' * 64
RECORD_KEYS = frozenset({'i', 'p', 't', 'at', 'b', 'h'})


class JournalError(RuntimeError):
    """Base class for journal failures."""


class JournalCorrupt(JournalError):
    """The file is damaged in a way a crash cannot explain. Fail closed."""


class JournalPoisoned(JournalError):
    """A write or fsync failed. Reopen the journal before any other use."""


class JournalLocked(JournalError):
    """Another controller process holds the state root."""


class FileOps:
    """The file operations the journal uses.

    The fault harness wraps this object to cut writes and to raise crashes at
    exact points. Production uses it as is.
    """

    def exists(self, path: Path) -> bool:
        return os.path.exists(path)

    def read_bytes(self, path: Path) -> bytes:
        with open(path, 'rb') as f:
            return f.read()

    def open_append(self, path: Path) -> int:
        return os.open(path, os.O_WRONLY | os.O_APPEND)

    def write(self, fd: int, data: bytes) -> int:
        return os.write(fd, data)

    def fsync(self, fd: int) -> None:
        os.fsync(fd)

    def close(self, fd: int) -> None:
        os.close(fd)

    def write_temp(self, path: Path, data: bytes) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def replace(self, src: Path, dst: Path) -> None:
        os.replace(src, dst)

    def fsync_dir(self, path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def seal(index: int, prev: str, kind: str, at: float, body: Mapping[str, Any]) -> Dict[str, Any]:
    """Build one record and stamp it with its own hash."""
    core = {'i': index, 'p': prev, 't': kind, 'at': at, 'b': dict(body)}
    return dict(core, h=hashlib.sha256(canonical(core)).hexdigest())


def encode(record: Mapping[str, Any]) -> bytes:
    return canonical(record) + b'\n'


def _self_consistent(line: bytes) -> Optional[Dict[str, Any]]:
    """Return the record if the line is a sealed record, else None."""
    try:
        rec = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(rec, dict) or set(rec) != RECORD_KEYS:
        return None
    if not isinstance(rec['i'], int) or isinstance(rec['i'], bool):
        return None
    if not isinstance(rec['t'], str) or not isinstance(rec['b'], dict):
        return None
    if not isinstance(rec['p'], str) or not isinstance(rec['h'], str):
        return None
    if not isinstance(rec['at'], (int, float)) or isinstance(rec['at'], bool):
        return None
    core = {k: rec[k] for k in ('i', 'p', 't', 'at', 'b')}
    try:
        expect = hashlib.sha256(canonical(core)).hexdigest()
    except ValueError:
        return None
    if expect != rec['h'] or canonical(rec) != line:
        return None
    return rec


def scan(data: bytes) -> Tuple[List[Dict[str, Any]], int]:
    """Parse journal bytes.

    Returns the valid records and the byte length of the valid prefix.

    One append writes one line, and the line feed is its last byte. So a crash
    can leave only a fragment without a line feed after the valid prefix. A
    complete line after the valid prefix that is a sealed record is not a torn
    tail: a record was moved, removed or edited. ``JournalCorrupt`` is raised.
    """
    records: List[Dict[str, Any]] = []
    prev = GENESIS_HASH
    pos = 0
    while pos < len(data):
        end = data.find(b'\n', pos)
        if end < 0:
            break
        rec = _self_consistent(data[pos:end])
        if rec is None or rec['i'] != len(records) or rec['p'] != prev:
            break
        records.append(rec)
        prev = rec['h']
        pos = end + 1
    complete_lines = data[pos:].split(b'\n')[:-1]
    for line in complete_lines:
        if line and _self_consistent(line) is not None:
            raise JournalCorrupt(
                'journal has a sealed record after the valid prefix at byte %d' % pos
            )
    return records, pos


class Journal:
    """Append-only journal. One writer. Use ``create`` once, then ``open``."""

    def __init__(self, path: Path, ops: FileOps, records: List[Dict[str, Any]]):
        self.path = Path(path)
        self._ops = ops
        self._loaded = records
        self._next = len(records)
        self._last_hash = records[-1]['h'] if records else GENESIS_HASH
        self._fd: Optional[int] = None
        self._poisoned = False

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        kind: str,
        body: Mapping[str, Any],
        at: float,
        ops: Optional[FileOps] = None,
    ) -> 'Journal':
        """Create the journal with its first record. Refuses to overwrite."""
        ops = ops or FileOps()
        path = Path(path)
        if ops.exists(path):
            raise JournalError('journal already exists: %s' % path.name)
        first = seal(0, GENESIS_HASH, kind, at, body)
        tmp = path.with_name(path.name + '.create.tmp')
        ops.write_temp(tmp, encode(first))
        ops.replace(tmp, path)
        ops.fsync_dir(path.parent)
        return cls.open(path, ops=ops)

    @classmethod
    def open(cls, path: Path, *, ops: Optional[FileOps] = None) -> 'Journal':
        """Open an existing journal, cut a torn tail, and verify the chain."""
        ops = ops or FileOps()
        path = Path(path)
        data = ops.read_bytes(path)
        records, valid = scan(data)
        if not records:
            raise JournalCorrupt('journal has no valid first record')
        if valid != len(data):
            tmp = path.with_name(path.name + '.repair.tmp')
            ops.write_temp(tmp, data[:valid])
            ops.replace(tmp, path)
            ops.fsync_dir(path.parent)
        journal = cls(path, ops, records)
        journal._fd = ops.open_append(path)
        return journal

    def replay(self) -> List[Dict[str, Any]]:
        """Records that were on disk at open. Call once; the list is dropped."""
        loaded, self._loaded = self._loaded, []
        return loaded

    @property
    def count(self) -> int:
        return self._next

    @property
    def last_hash(self) -> str:
        return self._last_hash

    def append(self, kind: str, body: Mapping[str, Any], at: float) -> Dict[str, Any]:
        """Write one record and fsync. Returns the sealed record."""
        if self._poisoned or self._fd is None:
            raise JournalPoisoned('journal is not writable; reopen it')
        rec = seal(self._next, self._last_hash, kind, at, body)
        data = memoryview(encode(rec))
        try:
            while data:
                n = self._ops.write(self._fd, data)
                if n <= 0:
                    raise JournalError('short journal write')
                data = data[n:]
            self._ops.fsync(self._fd)
        except BaseException:
            self._poisoned = True
            raise
        self._next += 1
        self._last_hash = rec['h']
        return rec

    def close(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            self._ops.close(fd)


def verify_file(path: Path) -> Tuple[int, str]:
    """Check a journal file without changing it. Returns (count, last hash)."""
    data = FileOps().read_bytes(Path(path))
    records, valid = scan(data)
    if not records:
        raise JournalCorrupt('journal has no valid first record')
    if valid != len(data):
        raise JournalCorrupt('journal has a torn tail at byte %d' % valid)
    return len(records), records[-1]['h']


class RootLock:
    """Exclusive lock on the state root. One controller process per root."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: Optional[int] = None

    def acquire(self) -> 'RootLock':
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise JournalLocked('state root is locked by another process') from exc
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
