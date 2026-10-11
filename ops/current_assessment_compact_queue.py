"""Read-only byte fingerprints and exact signed-cache reuse, not trusted SQL counters.

Every invocation reads/hashes actual strings, including complete job envelopes,
reports and terminal requests. SQLite returns only small typed fingerprints and
selected scalar metadata to Python. It still scans all evidence bytes; this is
bounded-memory, incremental authentication, not O(new jobs) storage IO.
"""
import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path

from ops.current_assessment_admission_cache import (
    canonical, digest, cutoff_phase, queue_fingerprint, KEY_VERSION,
)

VERSION = 'fresh-all-sqlite-fields-raw-byte-fingerprint-v1'
POLICY = dict(version=VERSION, maximum_rows=20000,
              maximum_string_bytes=256 * 1024**2,
              maximum_row_string_bytes=512 * 1024**2, snapshot_seconds=120)
REQUIRED = {'id', 'digest', 'envelope', 'status', 'report', 'report_digest',
            'report_request', 'worker', 'token', 'role'}


def _fingerprint(value):
    if type(value) is str:
        raw = value.encode('utf-8')
        return dict(type='str', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    if value is None or type(value) in (int, bool):
        return dict(type=type(value).__name__, value=value)
    if type(value) is float and math.isfinite(value):
        return dict(type='float', value=value)
    raise ValueError('finite ordinary SQLite scalar')


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('original regular authoritative queue')
    st = path.stat()
    return st.st_dev, st.st_ino


class BoundQueue:
    def __init__(self, path, inode, bindings):
        self.path, self.inode, self.bindings = path, inode, bindings
        self.view_sha256 = digest(dict(version=VERSION, rows=bindings))

    def fetch(self, identifier):
        """Only cache misses read a full row; compare to the exact prior snapshot."""
        fingerprint = self.bindings[identifier]['fingerprint']
        if fingerprint is None:
            raise ValueError('unsupported or oversized original queue row')
        if _identity(self.path) != self.inode:
            raise ValueError('original queue inode changed')
        # Enforce original string lengths in SQL before returning full bodies.
        # A concurrent replacement cannot inflate a miss into an unbounded read.
        conditions, parameters = ['id=?'], [identifier]
        for name, value in fingerprint.items():
            column = _quote(name)
            if value['type'] == 'str':
                conditions += ["typeof(" + column + ")='text'",
                               'length(CAST(' + column + ' AS BLOB))=?']
                parameters.append(value['bytes'])
            else:
                kind = {'NoneType': 'null', 'int': 'integer', 'float': 'real'}.get(value['type'])
                if kind is None:
                    raise ValueError('original bounded SQLite scalar type')
                conditions.append("typeof(" + column + ")='" + kind + "'")
        with sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=.25) as db:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA query_only=ON')
            db.execute('BEGIN')
            if {r['name'] for r in db.execute('PRAGMA table_info(jobs)')} != set(fingerprint):
                raise ValueError('original queue schema changed after snapshot')
            row = db.execute('SELECT * FROM jobs WHERE ' + ' AND '.join(conditions), parameters).fetchone()
        if _identity(self.path) != self.inode or row is None:
            raise ValueError('original queue bytes changed or row removed')
        actual = dict(row)
        if queue_fingerprint(actual) != self.bindings[identifier]['fingerprint']:
            raise ValueError('original row bytes changed after binding snapshot')
        return actual


def scan(queue, identifiers, *, policy=POLICY):
    if policy != POLICY:
        raise ValueError('explicit ROOT-admitted bounded compact queue profile')
    # The production caller supplies a list. Bound its container and strings
    # before making copies/UTF-8 encodings; never consume an arbitrary iterator.
    if (type(identifiers) not in (list, tuple) or
            len(identifiers) > policy['maximum_rows'] or
            any(type(i) is not str or len(i) > 128 or
                len(i.encode('utf-8')) > 128 for i in identifiers) or
            len(set(identifiers)) != len(identifiers)):
        raise ValueError('bounded distinct original job identifiers')
    identifiers = list(identifiers)
    path = Path(queue.path).absolute()
    if path.resolve() != path:
        raise ValueError('queue path without symlink parents')
    inode = _identity(path)
    if getattr(queue, '_queue_inode', inode) not in (None, inode):
        raise ValueError('authoritative queue inode changed')
    deadline = time.monotonic() + policy['snapshot_seconds']

    def fp(value):
        if time.monotonic() >= deadline:
            raise TimeoutError('compact evidence snapshot budget')
        return canonical(_fingerprint(value)).decode()

    bindings = {}
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=.25) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.create_function('fresh_fp', 1, fp)
        db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        db.execute('BEGIN')
        names = [r['name'] for r in db.execute('PRAGMA table_info(jobs)')]
        if not REQUIRED <= set(names) or len(names) > 64:
            raise ValueError('original bounded Coordinator job schema')
        expressions = []
        for index, name in enumerate(names):
            column = _quote(name)
            # Refuse oversized/unsupported cells without transferring their body.
            expressions.append("CASE WHEN typeof(" + column + ") IN ('null','integer','real','text') "
                "AND (typeof(" + column + ")!='text' OR length(CAST(" + column + " AS BLOB))<="
                + str(policy['maximum_string_bytes']) + ") THEN fresh_fp(" + column + ") ELSE NULL END AS f" + str(index))
        # Existing completion filtering is repeated from the actual current report,
        # never a cache-supplied timestamp. Malformed individual reports stay rows
        # with missing completion, to be refused without suppressing peers.
        # Only these three selectors leave SQLite outside typed fingerprints.
        # Identity/lease/report bodies remain fingerprints; oversized selectors
        # cannot transfer unbounded strings even when another cell is refused.
        for name in ('id', 'status', 'digest'):
            column = _quote(name)
            expressions.append("CASE WHEN typeof(" + column + ")='text' AND "
                "length(CAST(" + column + " AS BLOB))<=128 THEN " + column
                + " ELSE NULL END AS " + column)
        # A numeric completion is the only report value transferred. Nested CASE
        # bounds report bytes before JSON parsing and excludes string/object values
        # before json_extract, preventing giant invalid completion payloads.
        # Python json.loads keeps the LAST decoded top-level duplicate key.
        # json_extract instead keeps the first, so use json_each document order.
        # Extract only a numeric atom; strings/objects never leave SQLite.
        report_guard = ("typeof(report)='text' AND length(CAST(report AS BLOB))<="
                        + str(policy['maximum_string_bytes']))
        last_key = " FROM json_each(report) WHERE key='completed_at' ORDER BY id DESC LIMIT 1)"
        report_type = ("CASE WHEN " + report_guard
            + " THEN CASE WHEN json_valid(report) THEN (SELECT type" + last_key + " END END")
        report_number = ("CASE WHEN " + report_guard
            + " THEN CASE WHEN json_valid(report) THEN "
            "(SELECT CASE WHEN type IN ('integer','real') THEN atom END"
            + last_key + " END END")
        expressions += [report_type + " AS completion_type",
                        report_number + " AS completed_at"]
        for offset in range(0, len(identifiers), 250):
            group = identifiers[offset:offset + 250]
            sql = 'SELECT ' + ','.join(expressions) + ' FROM jobs WHERE id IN (' + ','.join('?' for _ in group) + ')'
            for row in db.execute(sql, group):
                fingerprint = ({name: json.loads(row['f' + str(i)]) for i, name in enumerate(names)}
                    if all(row['f' + str(i)] is not None for i in range(len(names))) else None)
                if fingerprint is not None and sum(v.get('bytes', 0) for v in fingerprint.values()) > policy['maximum_row_string_bytes']:
                    fingerprint = None
                completion_type, completed = row['completion_type'], row['completed_at']
                if (completion_type not in ('integer', 'real') or
                        type(completed) not in (int, float) or not math.isfinite(completed)):
                    completion_type, completed = 'invalid', None
                bindings[row['id']] = dict(fingerprint=fingerprint, status=row['status'],
                    digest=row['digest'], completion_type=completion_type, completed_at=completed)
        db.rollback()
    if _identity(path) != inode:
        raise ValueError('authoritative queue replaced during snapshot')
    return BoundQueue(path, inode, bindings)


def admit_bound(cache, binding, selected_records, *, cutoff, full_row):
    """Reuse the unchanged signed entry key only with fresh actual-byte binding."""
    phase = cutoff_phase(cache._context['execution_evidence_policy'], cutoff)
    fingerprint = binding['fingerprint']
    if phase is None or fingerprint is None:
        cache.stats['uncacheable'] += 1
        return cache._original(full_row(), selected_records, cutoff)
    key = digest(dict(version=KEY_VERSION, policy_sha256=cache._policy_sha,
        context_sha256=cache._context_sha, cutoff_phase=phase, queue=fingerprint,
        selected_records_sha256=digest(selected_records)))
    try:
        cached = cache._read(key)
    except TimeoutError:
        raise
    except Exception:
        cache.stats['rejected_entries'] += 1
        return cache._original(full_row(), selected_records, cutoff)
    if cached is not None:
        cache.stats['hits'] += 1
        return cached
    cache.stats['misses'] += 1
    result = cache._original(full_row(), selected_records, cutoff)
    try:
        cache._write(key, result)
    except TimeoutError:
        raise
    except Exception:
        cache.stats['write_errors'] += 1
    return result
