"""Large retained envelopes must not turn idle claims into blob-table scans."""
import sqlite3,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from subnet.distributed_roles import Coordinator,QUEUE_METADATA_INDEXES

CLAIM=("SELECT * FROM jobs WHERE role=? AND expires>? AND attempt<? AND (status='queued' OR (status='leased' AND lease<=?)) ORDER BY rowid LIMIT 1",('verify',100,3,100))
QUERIES=[("UPDATE jobs SET status='expired' WHERE status IN ('queued','leased') AND expires<=?",(100,)),("UPDATE jobs SET status='failed' WHERE status='leased' AND lease<=? AND attempt>=?",(100,3)),CLAIM,('DELETE FROM requests WHERE expires<?',(100,))]
class MetadataQueueTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'queue.sqlite3';self.queue=Coordinator(self.path,'operator',{})
 def rows(self,db):return [db.execute('SELECT * FROM '+table+' ORDER BY 1').fetchall()for table in ('jobs','requests','events')]
 def seed(self,db):
  for i in range(20):db.execute('INSERT INTO jobs(id,digest,envelope,role,expires,status,attempt,lease,report,report_digest,report_request) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(str(i),str(i),'ORIGINAL-SIGNED-JOB'+('x'*100000),'verify',500,'complete'if i%2==0 else'queued',0,50,'ORIGINAL-SIGNED-REPORT'+str(i),'originaldigest'+str(i),'ORIGINAL-SIGNED-REQUEST'+str(i)))
  db.execute("INSERT INTO requests VALUES('worker','original-nonce',500)");db.execute("INSERT INTO events(job,at,kind,detail)VALUES('1',100,'completed','original-event')");db.commit()
 def test_new_database_indexes_exact_and_claim_uses_metadata(self):
  with sqlite3.connect(self.path)as db:
   self.seed(db);self.assertEqual(dict(db.execute("SELECT name,sql FROM sqlite_master WHERE name LIKE 'affine_%'")),QUEUE_METADATA_INDEXES)
   for sql,args in QUERIES:
    plan=db.execute('EXPLAIN QUERY PLAN '+sql,args).fetchall();self.assertNotIn('SCAN jobs',str(plan));self.assertNotIn('SCAN requests',str(plan))
 def test_legacy_initialization_preserves_all_original_rows_and_inode(self):
  with sqlite3.connect(self.path)as db:
   self.seed(db)
   for name in QUEUE_METADATA_INDEXES:db.execute('DROP INDEX '+name)
   db.commit();before=self.rows(db);claim=db.execute(*CLAIM).fetchone()
  inode=self.path.stat().st_ino;Coordinator(self.path,'operator',{})
  with sqlite3.connect(self.path)as db:self.assertEqual(self.rows(db),before);self.assertEqual(db.execute(*CLAIM).fetchone(),claim)
  self.assertEqual(self.path.stat().st_ino,inode)
 def test_index_definition_collision_refuses_preserving_rows(self):
  with sqlite3.connect(self.path)as db:
   self.seed(db);name=next(iter(QUEUE_METADATA_INDEXES));db.execute('DROP INDEX '+name);db.execute('CREATE INDEX '+name+' ON jobs(role)');db.commit();before=self.rows(db)
  with self.assertRaisesRegex(ValueError,'conflicting'):Coordinator(self.path,'operator',{})
  with sqlite3.connect(self.path)as db:self.assertEqual(self.rows(db),before)
 def test_constructor_retries_real_BEGIN_lock(self):
  lock=sqlite3.connect(self.path,check_same_thread=False);lock.execute('BEGIN IMMEDIATE')
  t=threading.Thread(target=lambda:(time.sleep(.35),lock.rollback()));t.start()
  try:Coordinator(self.path,'operator',{})
  finally:t.join();lock.close()
 def test_acquisition_timeout_does_not_return_idle_or_write(self):
  lock=sqlite3.connect(self.path);lock.execute('BEGIN IMMEDIATE');now=[0]
  try:
   with self.assertRaises(TimeoutError):self.queue._begin_transaction(budget=.1,monotonic=lambda:now[0],sleep=lambda delta:now.__setitem__(0,now[0]+delta))
  finally:lock.rollback();lock.close()
 def test_body_busy_is_rolled_back_not_replayed(self):
  called=[]
  with self.assertRaises(sqlite3.OperationalError):
   with self.queue.transaction()as db:called.append(1);db.execute("INSERT INTO requests VALUES('w','n',1)");raise sqlite3.OperationalError('database is locked')
  self.assertEqual(called,[1])
  with sqlite3.connect(self.path)as db:self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0],0)
 def test_commit_busy_never_replays_transaction(self):
  reader=sqlite3.connect(self.path)
  try:
   with patch.object(self.queue,'_begin_transaction',wraps=self.queue._begin_transaction)as begin:
    with self.assertRaises(sqlite3.OperationalError):
     with self.queue.transaction()as db:
      db.execute("INSERT INTO requests VALUES('w','n',1)");reader.execute('BEGIN');reader.execute('SELECT * FROM jobs').fetchall()
    self.assertEqual(begin.call_count,1)
  finally:reader.rollback();reader.close()
  with sqlite3.connect(self.path)as db:self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0],0)
 def test_swapped_inode_refuses_before_BEGIN(self):
  self.path.rename(self.path.with_suffix('.original'));sqlite3.connect(self.path).close()
  with patch('subnet.distributed_roles.sqlite3.connect',side_effect=AssertionError('no connection')):
   with self.assertRaisesRegex(ValueError,'inode'):self.queue._begin_transaction()
if __name__=='__main__':unittest.main()
