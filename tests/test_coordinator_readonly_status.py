import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from subnet.distributed_roles import Coordinator


class ReadonlyStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'queue ? original.sqlite3'
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,status TEXT,report TEXT,report_digest TEXT,worker TEXT,attempt INTEGER)')
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?)',
                       ('original', 'queued', json.dumps({'audits': [], 'success': True}), 'digest', 'worker', 2))
        self.queue = Coordinator.__new__(Coordinator)
        self.queue.path = str(self.path)

    def legacy(self, identifier):
        with self.queue.transaction() as db:
            row = db.execute('SELECT status,report,report_digest,worker,attempt FROM jobs WHERE id=?', (identifier,)).fetchone()
            if not row:
                raise ValueError('unknown job')
            value = dict(row)
            value['report'] = json.loads(value['report']) if value['report'] else None
            return value

    def test_same_fields_and_report_as_original(self):
        self.assertEqual(self.queue.status('original'), self.legacy('original'))
        with sqlite3.connect(self.path) as db:
            db.execute('UPDATE jobs SET report=NULL')
        self.assertEqual(self.queue.status('original'), self.legacy('original'))

    def test_unknown_job_unchanged(self):
        with self.assertRaisesRegex(ValueError, '^unknown job$'):
            self.queue.status('absent')

    def test_concurrent_writer_reads_latest_committed_snapshot(self):
        before = self.queue.status('original')
        writer = sqlite3.connect(self.path)
        try:
            writer.execute('BEGIN IMMEDIATE')
            writer.execute("UPDATE jobs SET status='leased'")
            self.assertEqual(self.queue.status('original'), before)
            writer.commit()
            self.assertEqual(self.queue.status('original'), self.legacy('original'))
            self.assertEqual(self.queue.status('original')['status'], 'leased')
        finally:
            writer.close()

    def test_exclusive_writer_retries_then_reads(self):
        ready = threading.Event()
        def write():
            db = sqlite3.connect(self.path)
            db.execute('BEGIN EXCLUSIVE')
            ready.set()
            time.sleep(.35)
            db.rollback()
            db.close()
        thread = threading.Thread(target=write)
        thread.start()
        ready.wait(2)
        try:
            self.assertEqual(self.queue.status('original', timeout_seconds=2)['status'], 'queued')
        finally:
            thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_exclusive_writer_deadline_never_fabricates_status(self):
        writer = sqlite3.connect(self.path)
        writer.execute('BEGIN EXCLUSIVE')
        try:
            start = time.monotonic()
            with self.assertRaises(sqlite3.OperationalError):
                self.queue.status('original', timeout_seconds=.04)
            self.assertLess(time.monotonic()-start, .3)
        finally:
            writer.rollback()
            writer.close()

    def test_non_busy_error_propagates_without_retry(self):
        self.queue.path = str(Path(self.temp.name)/'absent.sqlite3')
        start = time.monotonic()
        with self.assertRaises(sqlite3.OperationalError):
            self.queue.status('original')
        self.assertLess(time.monotonic()-start, .3)
        self.assertFalse(Path(self.queue.path).exists())

    def test_read_only_one_select_and_no_writer_transaction(self):
        trace = []
        connect = sqlite3.connect
        def traced(*args, **kwargs):
            self.assertTrue(kwargs['uri'])
            self.assertTrue(args[0].endswith('?mode=ro'))
            db = connect(*args, **kwargs)
            db.set_trace_callback(trace.append)
            return db
        with patch('subnet.distributed_roles.sqlite3.connect', side_effect=traced), \
                patch.object(self.queue, 'transaction', side_effect=AssertionError('writer transaction used')):
            self.queue.status('original')
        self.assertEqual(len(trace), 1)
        self.assertTrue(trace[0].startswith('SELECT status,report,report_digest,worker,attempt'))

    def test_connection_closed_before_report_decode(self):
        connections = []
        connect = sqlite3.connect
        decode = json.loads
        def tracked(*args, **kwargs):
            db = connect(*args, **kwargs)
            connections.append(db)
            return db
        def checked_decode(value):
            for db in connections:
                with self.assertRaises(sqlite3.ProgrammingError):
                    db.execute('SELECT 1')
            return decode(value)
        with patch('subnet.distributed_roles.sqlite3.connect', side_effect=tracked), \
                patch('subnet.distributed_roles.json.loads', side_effect=checked_decode):
            self.assertTrue(self.queue.status('original')['report']['success'])

    def test_corrupt_report_still_raises(self):
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE jobs SET report='invalid json'")
        with self.assertRaises(json.JSONDecodeError):
            self.queue.status('original')

    def test_timeout_bounds(self):
        for timeout in (0, -1, 31, float('inf'), float('nan'), True, '1'):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, 'status timeout'):
                self.queue.status('original', timeout_seconds=timeout)


if __name__ == '__main__':
    unittest.main()
