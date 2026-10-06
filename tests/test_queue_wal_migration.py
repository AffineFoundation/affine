import sqlite3,tempfile,threading,time,unittest
from pathlib import Path
from ops.queue_wal_migration import migrate,logical_snapshot

class QueueWALMigration(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name)/'queue';db=sqlite3.connect(self.p);db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,status TEXT,job BLOB,report TEXT)');db.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,body TEXT)');db.execute('INSERT INTO jobs VALUES(?,?,?,?)',('original','leased',b'original\x00signedbytes','original signed report'));db.execute('INSERT INTO events VALUES(?,?)',(1,'originalevent'));db.commit();db.close();s=self.p.stat();self.inode=[s.st_dev,s.st_ino]
 def run_migrate(self):return migrate(self.p,self.inode,writer_handoff_guard=lambda:None)
 def test_preserves_all_original_bytes_schema_inode_full_and_idempotent(self):
  r=self.run_migrate();self.assertEqual(r['journal_mode'],'wal');self.assertTrue(r['all_records_and_schema_preserved']);self.assertEqual(r['table_rows'],{'events':1,'jobs':1});self.assertEqual(self.run_migrate()['logical_sha256'],r['logical_sha256'])
 def test_real_reader_does_not_block_single_writer_commit_enqueue_and_claim(self):
  self.run_migrate();reader=sqlite3.connect(self.p);reader.execute('BEGIN');reader.execute('SELECT * FROM jobs').fetchall();writer=sqlite3.connect(self.p,timeout=.1);writer.execute('PRAGMA synchronous=FULL');writer.execute('BEGIN IMMEDIATE');writer.execute('INSERT INTO jobs VALUES(?,?,?,?)',('next','queued',b'unchanged next envelope',None));writer.execute("UPDATE jobs SET status='leased' WHERE id='next'");writer.commit();self.assertEqual(writer.execute('PRAGMA synchronous').fetchone()[0],2);self.assertEqual(reader.execute('SELECT count(*) FROM jobs').fetchone()[0],1);reader.commit();self.assertEqual(reader.execute('SELECT count(*) FROM jobs').fetchone()[0],2);reader.close();writer.close()
 def test_multiple_readers_and_writers_preserve_unique_dispatches(self):
  self.run_migrate();barrier=threading.Barrier(5);errors=[]
  def reader():
   try:
    db=sqlite3.connect(self.p);db.execute('BEGIN');db.execute('SELECT * FROM jobs').fetchall();barrier.wait();time.sleep(.2);db.close()
   except Exception as e:errors.append(e)
  def writer(i):
   try:
    db=sqlite3.connect(self.p,timeout=2);db.execute('PRAGMA synchronous=FULL');barrier.wait();db.execute('BEGIN IMMEDIATE');db.execute('INSERT INTO events VALUES(?,?)',(i,'dispatch-'+str(i)));db.commit();db.close()
   except Exception as e:errors.append(e)
  threads=[threading.Thread(target=reader)for _ in range(2)]+[threading.Thread(target=writer,args=(i,))for i in(2,3,4)]
  for t in threads:t.start()
  for t in threads:t.join(3)
  self.assertEqual(errors,[]);db=sqlite3.connect(self.p);self.assertEqual(db.execute('SELECT count(*) FROM events').fetchone()[0],4);db.close()
 def test_existing_reader_makes_migration_defer_without_job_mutation(self):
  reader=sqlite3.connect(self.p);reader.execute('BEGIN');reader.execute('SELECT * FROM jobs').fetchall()
  with self.assertRaises(TimeoutError):migrate(self.p,self.inode,writer_handoff_guard=lambda:None,timeout_seconds=.1)
  reader.close();db=sqlite3.connect(self.p);self.assertEqual(db.execute('PRAGMA journal_mode').fetchone()[0],'delete');self.assertEqual(db.execute('SELECT job FROM jobs').fetchone()[0],b'original\x00signedbytes');db.close()
 def test_changed_inode_or_unapproved_writer_handoff_refused(self):
  with self.assertRaises(ValueError):migrate(self.p,[self.inode[0],self.inode[1]+1],writer_handoff_guard=lambda:None)
  def active():raise ValueError('preserve active API')
  with self.assertRaisesRegex(ValueError,'active API'):migrate(self.p,self.inode,writer_handoff_guard=active)
 def test_released_reader_allows_bounded_migration(self):
  ready=threading.Event()
  def reader():
   db=sqlite3.connect(self.p);db.execute('BEGIN');db.execute('SELECT * FROM jobs').fetchall();ready.set();time.sleep(.3);db.close()
  t=threading.Thread(target=reader);t.start();ready.wait();self.assertEqual(self.run_migrate()['journal_mode'],'wal');t.join()

if __name__=='__main__':unittest.main()
