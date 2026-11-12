"""The batches of the live training queue, for the batch controller.

A batch is one training epoch ``N`` of the validator host. Its files are in
the directory ``<life_root>/epochs/<N>/``. ``LiveQueue`` is the
``BatchSource`` of the controller. It lists the epochs that the live queue
(``pit.queue_ingest``) would take now.

Read-only
---------
This module reads files and directories. It writes nothing. The caller gives
every path. No environment variable is read: in shadow mode the variables of
the process point at shadow directories. Of ``queue_ingest`` only three pure
rule functions are used. A pure function reads no variable and writes
nothing.

Rows
----
A row is what one source says about one epoch: the epoch number and the miner
(uid and hotkey). The sources are read in this order. The first row of an
epoch wins, as in the live queue:

``inbox``           each ``*.json`` ticket in each inbox directory.
``k1-turn``         ``K1-TURN.json`` in each epoch directory, by number.
``k1-turns-jsonl``  the last 200 lines of the turns log.
``pending``         each ``e*.json`` in the pending directory that is not
                    marked ``pit_claimed``.

Batches
-------
The floor is ``epoch_index`` of ``<life_root>/CURRENT.json``. The listing
fails without it: the floor is never taken as 0. A row is a batch only if
every rule below is true. A row that fails a rule is counted in ``skipped``
under the name in brackets:

* the epoch is not below the floor, or its PIT dispatch has no result yet:
  ``PIT-DISPATCHED.json`` is there and ``PIT-RESULT.json`` is not
  [below_floor];
* the epoch directory is ready by the rule of the live queue: it has a
  training job and a dispatch, no result, no rerun that runs, and no
  ``FINALIZED.json`` without ``PIT-DISPATCHED.json`` [not_ready];
* the row names a uid and a hotkey [no_miner];
* ``TRAIN-DISPATCH-FF.json`` is signed and selects one document or more,
  each with a sha256 and a key [dispatch];
* ``TRAINING-JOB.json`` names a checkpoint id [job].

Identity
--------
The id of epoch ``N`` is ``e<N>``. The miner text is the uid and the first 16
characters of the hotkey, so each source gives the same text. The reference
is the sha256 of the epoch number and the sorted document hashes. So one id
is bound to one miner and to one set of documents.

After admission
---------------
``LiveQueue.batch`` reads one admitted epoch again for the box agent. The
floor and the ready rule do not apply there. The other rules do.

Errors
------
A directory that cannot be read fails the call with ``SourceError``. One file
that cannot be read or parsed counts as a missing file: its row is skipped.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from ...durable import canonical
from .. import queue_ingest as Q
from .interfaces import BatchRef, SourceError
from .layout import is_hex64, sha256_hex

__all__ = [
    'BATCH_ID', 'REASONS', 'LiveBatch', 'LiveQueue', 'batch_id_of', 'dispatch_documents', 'epoch_of',
    'job_checkpoint',
]

# Used with ``fullmatch`` only. No leading zero, so one epoch has one id.
BATCH_ID = re.compile(r'e(0|[1-9][0-9]{0,8})')
REASONS = ('below_floor', 'not_ready', 'no_miner', 'dispatch', 'job')
UNSIGNED_SIGNER = 'SHADOW-UNSIGNED'
TURNS_LOG_LINES = 200

# ASCII digits only, and short: ``int`` must never fail on a value from a file.
_DIGITS = re.compile(r'[0-9]{1,18}')

# The three pure rules of the live queue. No other name of ``queue_ingest`` is used in this module.
_epoch_index = Q._epoch_index_of
_passes_floor = Q._above_floor_or_pit_open
_dir_ready = Q._epoch_dir_ready


def batch_id_of(epoch: int) -> str:
    return 'e%d' % epoch


def epoch_of(batch_id: str) -> Optional[int]:
    """The epoch number of a batch id. None when the text is not a batch id."""
    found = BATCH_ID.fullmatch(batch_id) if isinstance(batch_id, str) else None
    return int(found.group(1)) if found is not None else None


@dataclass(frozen=True)
class LiveBatch:
    """One epoch of the live queue that is complete enough to train."""

    epoch: int
    uid: int
    hotkey: str                 # as the source gives it: the full key or its first characters
    epoch_dir: Path
    checkpoint_id: str          # the checkpoint that the signed training job of the epoch names
    documents: Tuple[str, ...]  # sorted sha256 of the documents that the signed dispatch selects
    source: str                 # 'inbox' | 'k1-turn' | 'k1-turns-jsonl' | 'pending'

    @property
    def batch_id(self) -> str:
        return batch_id_of(self.epoch)

    @property
    def miner(self) -> str:
        return '%d:%s' % (self.uid, self.hotkey[:16])

    @property
    def ref(self) -> str:
        return sha256_hex(canonical({'epoch': self.epoch, 'documents': list(self.documents)}))

    def as_ref(self) -> BatchRef:
        return BatchRef(self.batch_id, self.miner, self.ref)


class _Row(NamedTuple):
    """What one source says about one epoch. The uid and the hotkey are not checked yet."""

    epoch: int
    uid: Any
    hotkey: Any
    source: str


def _read(path: Path) -> Optional[bytes]:
    """The bytes of one file. None when it cannot be read: it then counts as a missing file."""
    try:
        return path.read_bytes()
    except OSError:
        return None


def _parse_object(raw: Optional[bytes]) -> Optional[Dict[str, Any]]:
    """The JSON object in the bytes. None for no bytes, for bad JSON and for JSON that is not an object."""
    if raw is None:
        return None
    try:
        doc = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    return doc if isinstance(doc, dict) else None


def _read_object(path: Path) -> Optional[Dict[str, Any]]:
    return _parse_object(_read(path))


def _index(value: Any) -> Optional[int]:
    """A count from a file: an int that is not a bool and not below 0, or a text of digits. Else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and _DIGITS.fullmatch(value) is not None:
        return int(value)
    return None


def _matching(root: Path, pattern: str) -> List[Path]:
    """The entries of a directory whose names match the pattern, in the order of ``sorted(root.glob(pattern))``.

    ``glob`` gives an empty list for a directory that cannot be read. This function raises the ``OSError``,
    so an unreadable directory is never taken for an empty one.
    """
    return sorted(path for path in root.iterdir() if fnmatchcase(path.name, pattern))


def _floor(life_root: Path) -> int:
    """``epoch_index`` of ``CURRENT.json``. Fail closed: no file or no number there is an error, never floor 0."""
    current = _read_object(life_root / 'CURRENT.json')
    floor = _index(current.get('epoch_index')) if current is not None else None
    if floor is None:
        raise SourceError('CURRENT.json gives no epoch_index: the floor of the live queue is unknown')
    return floor


def _row(doc: Optional[Mapping[str, Any]], source: str, dir_name: Optional[str] = None) -> Iterator[_Row]:
    """The row of one source document, or nothing. An epoch that has no batch id gives no row."""
    if doc is None:
        return
    try:
        epoch = _epoch_index(doc, dir_name=dir_name)
    except OverflowError:
        # The rule calls ``int`` on the number. An infinite float fails there.
        return
    if epoch is not None and epoch_of(batch_id_of(epoch)) == epoch:
        yield _Row(epoch, doc.get('uid'), doc.get('hotkey') or doc.get('hotkey_prefix'), source)


def _inbox_rows(inboxes: Sequence[Path]) -> Iterator[_Row]:
    for inbox in inboxes:
        if inbox.is_dir():
            for path in _matching(inbox, '*.json'):
                yield from _row(_read_object(path), 'inbox')


def _turn_rows(epochs: Path) -> Iterator[_Row]:
    if not epochs.is_dir():
        return
    numbered = [path for path in epochs.iterdir() if _DIGITS.fullmatch(path.name) is not None]
    for path in sorted(numbered, key=lambda p: (int(p.name), p.name)):
        yield from _row(_read_object(path / 'K1-TURN.json'), 'k1-turn', path.name)


def _log_rows(turns_log: Optional[Path]) -> Iterator[_Row]:
    if turns_log is None or not turns_log.is_file():
        return
    for line in (_read(turns_log) or b'').splitlines()[-TURNS_LOG_LINES:]:
        yield from _row(_parse_object(line), 'k1-turns-jsonl')


def _pending_rows(pending: Optional[Path]) -> Iterator[_Row]:
    if pending is None or not pending.is_dir():
        return
    for path in _matching(pending, 'e*.json'):
        doc = _read_object(path)
        if doc is not None and not doc.get('pit_claimed'):
            number = path.stem[1:]
            yield from _row(doc, 'pending', number if _DIGITS.fullmatch(number) is not None else None)


def dispatch_documents(path: Path) -> Optional[Tuple[str, ...]]:
    """The sorted sha256 of the documents that a signed dispatch selects, each one time.

    None when the file is not a JSON object, has no signature, names the signer of an unsigned shadow
    dispatch, selects nothing, or selects a row without a document sha256 and a document key.
    """
    doc = _read_object(path)
    if doc is None or not doc.get('signature') or doc.get('signer') == UNSIGNED_SIGNER:
        return None
    payload = doc.get('payload')
    selected = payload.get('selected') if isinstance(payload, dict) else None
    if not isinstance(selected, list) or not selected:
        return None
    found = set()
    for row in selected:
        document = row.get('document') if isinstance(row, dict) else None
        if not isinstance(document, dict) or not is_hex64(document.get('sha256')):
            return None
        if not isinstance(document.get('key'), str) or not document['key']:
            return None
        found.add(document['sha256'])
    return tuple(sorted(found))


def _named_id(doc: Mapping[str, Any]) -> Optional[str]:
    """``checkpoint.id`` of a document if it is a text of 64 characters: the test of the live queue."""
    checkpoint = doc.get('checkpoint')
    found = checkpoint.get('id') if isinstance(checkpoint, dict) else None
    return found if isinstance(found, str) and len(found) == 64 else None


def job_checkpoint(path: Path) -> Optional[str]:
    """The checkpoint id that a training job names, or None.

    The id is the one that the live queue reads (``queue_ingest.training_job_checkpoint``). The payload is
    ``payload`` of the file, or the file itself. First the manifest of the payload: ``checkpoint.id`` of
    its ``payload``, or of the manifest itself. Then ``checkpoint.id`` of the payload. This function is
    stricter in one point: an id that is not a sha256 is no id.
    """
    job = _read_object(path)
    if job is None:
        return None
    payload = job['payload'] if isinstance(job.get('payload'), dict) else job
    found = None
    manifest = payload.get('manifest')
    if isinstance(manifest, dict):
        found = _named_id(manifest['payload'] if isinstance(manifest.get('payload'), dict) else manifest)
    if found is None:
        found = _named_id(payload)
    return found if is_hex64(found) else None


def _unreadable(exc: OSError) -> SourceError:
    return SourceError('the live queue cannot be read: %s' % type(exc).__name__)


class LiveQueue:
    """The ``BatchSource`` of the controller: the live training queue of the validator host. Read-only."""

    def __init__(self, life_root: Path, *, inboxes: Sequence[Path] = (), turns_log: Optional[Path] = None,
                 pending: Optional[Path] = None) -> None:
        if isinstance(inboxes, (str, bytes)):
            raise TypeError('inboxes is a sequence of paths, not one path')
        self.life_root = Path(life_root)
        self.inboxes = tuple(Path(inbox) for inbox in inboxes)
        self.turns_log = Path(turns_log) if turns_log is not None else None
        self.pending = Path(pending) if pending is not None else None
        # reason -> rows that the last complete scan skipped. A scan that fails leaves the counts as they are.
        self.skipped: Dict[str, int] = dict.fromkeys(REASONS, 0)

    def batches(self) -> List[LiveBatch]:
        """Every batch that the live queue would take now, by epoch. Raises ``SourceError``."""
        floor = _floor(self.life_root)
        skipped = dict.fromkeys(REASONS, 0)
        found: List[LiveBatch] = []
        try:
            for row in self._first_rows().values():
                batch = self._resolve(row, floor)
                if isinstance(batch, str):
                    skipped[batch] += 1
                else:
                    found.append(batch)
        except OSError as exc:
            raise _unreadable(exc) from exc
        self.skipped = skipped
        return sorted(found, key=lambda batch: batch.epoch)

    def list(self) -> Sequence[BatchRef]:
        return [batch.as_ref() for batch in self.batches()]

    def batch(self, batch_id: str) -> Optional[LiveBatch]:
        """An admitted batch, read again from its sources and its epoch directory. None when a part is missing.

        The floor and the ready rule do not apply: an admitted batch stays valid when the floor moves. The
        caller compares ``ref`` with the reference that the controller admitted. Raises ``SourceError``.
        """
        epoch = epoch_of(batch_id)
        if epoch is None:
            return None
        try:
            row = self._first_rows().get(epoch)
            found = self._resolve(row, None) if row is not None else None
        except OSError as exc:
            raise _unreadable(exc) from exc
        return found if isinstance(found, LiveBatch) else None

    def _first_rows(self) -> Dict[int, _Row]:
        """The first row of each epoch, over the four sources in their order."""
        first: Dict[int, _Row] = {}
        for row in chain(_inbox_rows(self.inboxes), _turn_rows(self.life_root / 'epochs'),
                         _log_rows(self.turns_log), _pending_rows(self.pending)):
            first.setdefault(row.epoch, row)
        return first

    def _resolve(self, row: _Row, floor: Optional[int]) -> Union[LiveBatch, str]:
        """The batch of a row, or the reason why the row is not a batch.

        ``floor`` is None for an admitted batch: the floor and the ready rule then do not apply.
        """
        epoch_dir = self.life_root / 'epochs' / str(row.epoch)
        present = epoch_dir.is_dir()
        if floor is not None and not _passes_floor(row.epoch, floor, epoch_dir if present else None):
            return 'below_floor'
        if not present or (floor is not None and not _dir_ready(epoch_dir)):
            return 'not_ready'
        uid = _index(row.uid)
        if uid is None or not isinstance(row.hotkey, str) or not row.hotkey:
            return 'no_miner'
        documents = dispatch_documents(epoch_dir / 'TRAIN-DISPATCH-FF.json')
        if documents is None:
            return 'dispatch'
        checkpoint_id = job_checkpoint(epoch_dir / 'TRAINING-JOB.json')
        if checkpoint_id is None:
            return 'job'
        return LiveBatch(row.epoch, uid, row.hotkey, epoch_dir, checkpoint_id, documents, row.source)
