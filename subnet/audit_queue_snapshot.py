"""Short, consistent read-only queue views; no writer reservation or decoding.

All rows are copied and the read transaction released before callers perform
signature checks, JSON parsing, storage IO, or reward computation. In rollback
journal mode this short reader can briefly delay a writer's COMMIT, but never
reserves the writer lock itself. Busy reads retry the whole read-only view.
"""
import math
import sqlite3
import time
from pathlib import Path


def queue_rows(queue, identifiers, *, complete=False, include_deadline=False, timeout_seconds=30.0):
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30:
        raise ValueError('audit read snapshot timeout')
    identifiers = list(identifiers)
    if any(type(i) is not str for i in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError('exact audit queue identifiers')
    if not identifiers:
        return {}
    path = Path(queue.path)
    original = path.stat()
    expected = (original.st_dev, original.st_ino)
    signed_inode = getattr(queue, '_queue_inode', None)
    if signed_inode is not None and expected != signed_inode:
        raise ValueError('authoritative queue inode changed')
    def check_identity():
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) != expected:
            raise ValueError('authoritative queue inode changed')
    deadline = time.monotonic() + timeout_seconds
    columns = '*' if complete else 'id,status,expires' if include_deadline else 'id,status'
    while True:
        db = None
        try:
            check_identity()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('authoritative audit read snapshot unavailable')
            db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True,
                                 timeout=min(.25, remaining), isolation_level=None)
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA query_only=ON')
            db.execute('BEGIN')
            rows = {}
            for offset in range(0, len(identifiers), 500):
                group = identifiers[offset:offset + 500]
                statement = 'SELECT ' + columns + ' FROM jobs WHERE id IN (' + ','.join('?' for _ in group) + ')'
                rows.update((r['id'], dict(r)) for r in db.execute(statement, group).fetchall())
            check_identity()
            # A read-only rollback releases the snapshot, with no write COMMIT.
            db.rollback()
            db.close()
            db = None
            check_identity()
            return rows
        except sqlite3.OperationalError as error:
            if not any(word in str(error).lower() for word in ('locked', 'busy')):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('authoritative audit read snapshot unavailable') from error
            if db is not None:
                db.close()
                db = None
            time.sleep(min(.025, remaining))
        finally:
            if db is not None:
                db.close()
