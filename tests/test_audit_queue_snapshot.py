import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from subnet.audit_queue_snapshot import queue_rows


class QueueSnapshotControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'queue.sqlite3'
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,status TEXT,report TEXT,envelope TEXT)')
            db.executemany('INSERT INTO jobs VALUES(?,?,?,?)', [(str(i),'queued',None,'original') for i in range(501)])
        stat = self.path.stat()
        self.queue = SimpleNamespace(path=str(self.path), _queue_inode=(stat.st_dev,stat.st_ino))

    def test_reader_never_reserves_writer_and_releases_before_json_work(self):
        original = sqlite3.connect; reserved = []; statements = []
        def connect(*args, **kwargs):
            db = original(*args, **kwargs)
            if kwargs.get('uri'):
                def trace(sql):
                    statements.append(sql)
                    if sql.startswith('SELECT'):
                        writer = original(self.path, isolation_level=None, timeout=.1)
                        writer.execute('BEGIN IMMEDIATE')  # Succeeds during the read snapshot.
                        reserved.append(writer)
                db.set_trace_callback(trace)
            return db
        with patch('subnet.audit_queue_snapshot.sqlite3.connect',side_effect=connect):
            rows = queue_rows(self.queue,['0'],complete=True)
        self.assertEqual(rows['0']['envelope'],'original')
        self.assertIn('BEGIN',statements); self.assertNotIn('BEGIN IMMEDIATE',statements)
        self.assertNotIn('COMMIT',statements)
        # Expensive caller processing occurs now; reader has already closed so
        # the reserved writer can commit despite arbitrary subsequent work.
        writer = reserved[0]
        writer.execute("UPDATE jobs SET status='complete', report=? WHERE id='0'",(json.dumps({'completed_at':99}),))
        writer.commit(); writer.close()
        self.assertEqual(rows['0']['status'],'queued')
        self.assertEqual(queue_rows(self.queue,['0'],complete=True)['0']['status'],'complete')

    def test_chunked_view_consistent_with_concurrent_late_commit(self):
        original = sqlite3.connect
        with original(self.path) as db:db.execute('PRAGMA journal_mode=WAL')
        selects = []
        def connect(*args, **kwargs):
            db = original(*args, **kwargs)
            if kwargs.get('uri'):
                def trace(sql):
                    if sql.startswith('SELECT'):
                        selects.append(sql)
                        if len(selects)==2:
                            with original(self.path) as writer:
                                writer.execute("UPDATE jobs SET status='complete', report=? WHERE id='500'",(json.dumps({'completed_at':10}),))
                db.set_trace_callback(trace)
            return db
        with patch('subnet.audit_queue_snapshot.sqlite3.connect',side_effect=connect):
            rows = queue_rows(self.queue,[str(i)for i in range(501)],complete=True)
        self.assertEqual(rows['500']['status'],'queued')
        self.assertEqual(queue_rows(self.queue,['500'],complete=True)['500']['status'],'complete')

    def test_busy_read_retries_without_zero_or_mutating_rows(self):
        entered=threading.Event();release=threading.Event()
        def lock():
            db=sqlite3.connect(self.path,isolation_level=None)
            db.execute('BEGIN EXCLUSIVE');entered.set();release.wait(2);db.rollback();db.close()
        thread=threading.Thread(target=lock);thread.start();entered.wait(1)
        timer=threading.Timer(.3,release.set);timer.start()
        try:self.assertEqual(queue_rows(self.queue,['0'])['0']['status'],'queued')
        finally:release.set();thread.join();timer.cancel()

    def test_busy_timeout_is_not_empty_snapshot(self):
        db=sqlite3.connect(self.path,isolation_level=None);db.execute('BEGIN EXCLUSIVE')
        try:
            with self.assertRaises(TimeoutError):queue_rows(self.queue,['0'],timeout_seconds=.02)
        finally:db.rollback();db.close()

    def test_hourly_cutoff_uses_original_raw_rows_after_read_close(self):
        from subnet.continuous_audit_service import ContinuousAuditor
        with sqlite3.connect(self.path) as db:
            db.execute('ALTER TABLE jobs ADD COLUMN report_request TEXT')
            db.execute("UPDATE jobs SET status='complete',report=?,report_request='signed-original' WHERE id='0'",(json.dumps({'completed_at':99}),))
            db.execute("UPDATE jobs SET status='complete',report=? WHERE id='1'",(json.dumps({'completed_at':101}),))
        service=ContinuousAuditor.__new__(ContinuousAuditor)
        service.directory=Path(self.tmp.name)/'audits';service.directory.mkdir()
        service.queue=self.queue;self.queue.workers={}
        service.controller=SimpleNamespace(authority=SimpleNamespace(id='authority'),signed=lambda x:x)
        service.state={'jobs':{'0':{'row_sha256':'a'},'1':{'row_sha256':'b'},'2':{'row_sha256':'c'}},
                       'draws':{i:{'row':{'round':1,'committed_at':50}}for i in ('a','b','c')},
                       'populations':{'epoch':{'eligible_evidence_ids':[]}},'capture_failures':{}}
        service.records=lambda:[];service.sources={};service.execution_evidence_policy=None
        service.backend_evidence_deferral_policy=None;service.policy={};service.publish_immutable=lambda *a:None
        seen=[]
        def admit(queued,*args,**kwargs):
            seen.extend(queued)
            # No read transaction remains while auth/policy work takes place.
            with sqlite3.connect(self.path,timeout=.1)as db:db.execute("UPDATE jobs SET status='queued' WHERE id='0'")
            return {},[]
        with patch('subnet.continuous_audit_service.admit_completed_reports',side_effect=admit),patch('subnet.continuous_audit_service.authenticate',side_effect=lambda x,a:x),patch('subnet.continuous_audit_service.snapshot',return_value={'original':'snapshot'}):
            service.hourly_snapshot('epoch',1,'checkpoint',100)
        self.assertEqual([r['id']for r in seen],['0'])
        self.assertEqual(seen[0]['report_request'],'signed-original')
        self.assertEqual(seen[0]['status'],'complete')
        self.assertEqual(json.loads(seen[0]['report'])['completed_at'],99)

    def test_identity_replacement_and_duplicate_ids_rejected(self):
        self.queue._queue_inode=(0,0)
        with self.assertRaises(ValueError):queue_rows(self.queue,['0'])
        with self.assertRaises(ValueError):queue_rows(self.queue,['0','0'])

if __name__=='__main__':unittest.main()
