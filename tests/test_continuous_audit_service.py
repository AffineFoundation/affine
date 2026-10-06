import datetime,io,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from botocore.exceptions import ClientError
from nacl.signing import SigningKey
from test_continuous_audit_policy import signed
from subnet.continuous_audit_policy import VERSION,digest
from subnet.continuous_audit_service import ContinuousAuditor,InvalidCommittedArtifact,atomic,completed_learners
class Bucket:
 def __init__(self):self.name='bucket';self.client=self;self.error=None;self.body=b'proof';self.copies=[]
 def head_object(self,**kw):
  if self.error:raise self.error
  return dict(ContentLength=5,LastModified=datetime.datetime.fromtimestamp(15,datetime.timezone.utc),ETag='original-etag')
 def copy(self,key,frozen,expected_etag):self.copies.append((key,frozen,expected_etag))
 def get_object(self,**kw):return {'Body':io.BytesIO(self.body)}
 def presign(self,key):return 'https://private/'+key
class Queue:
 workers={}
 def __init__(self):self.envelopes=[]
 def status(self,j):return {'status':'queued'}
 def enqueue(self,e):self.envelopes.append(e)
 def archive(self,*a):pass
class ServiceControls(unittest.TestCase):
 def test_failed_commit_retries_same_request_after_rollback(self):
  import sqlite3
  from subnet.continuous_audit_service import service_cycle
  path=Path(self.directory.name)/'commit.sqlite3'
  original=signed(self.key,dict(job_id='original',created_at=10,expires_at=100))
  import json
  raw=json.dumps(original,sort_keys=True)
  with sqlite3.connect(path)as db:
   db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,original TEXT)')
   db.execute('CREATE TABLE existing(id INTEGER)');db.execute('INSERT INTO existing VALUES(1)')
  reader=sqlite3.connect(path,isolation_level=None)
  reader.execute('BEGIN');reader.execute('SELECT * FROM existing').fetchall()
  attempts=[]
  class Service:
   def tick(inner):
    attempts.append(raw)
    db=sqlite3.connect(path,timeout=.05,isolation_level=None)
    try:
     db.execute('BEGIN IMMEDIATE');db.execute('INSERT OR IGNORE INTO jobs VALUES(?,?)',('original',raw));db.commit()
    finally:db.close()  # Failed commit rolls back, no body replay.
    return dict(enqueued=1)
   def reconcile_hours(inner,*args):pass
  service=Service()
  with patch('subnet.continuous_audit_service.completed_learners',return_value=[]):
   try:
    failed=service_cycle(service,self.directory.name,self.root)
    self.assertEqual(failed['status'],'retryable-queue-contention')
    self.assertNotIn('enqueued',failed)
    with sqlite3.connect(path)as db:self.assertEqual(db.execute('SELECT count(*) FROM jobs').fetchone()[0],0)
   finally:reader.rollback();reader.close()
   self.assertEqual(service_cycle(service,self.directory.name,self.root)['enqueued'],1)
   service_cycle(service,self.directory.name,self.root)
  self.assertEqual(attempts,[raw,raw,raw])
  with sqlite3.connect(path)as db:self.assertEqual(db.execute('SELECT * FROM jobs').fetchall(),[('original',raw)])
 def test_service_cycle_never_masks_integrity_or_schema_failures(self):
  import sqlite3
  from subnet.continuous_audit_service import service_cycle
  for error in(ValueError('signature mismatch'),sqlite3.OperationalError('no such table: jobs')):
   service=SimpleNamespace(tick=lambda:None)
   with patch.object(service,'tick',side_effect=error):
    with self.assertRaises(type(error)):service_cycle(service,self.directory.name,self.root)
 def test_expired_sqlite_jobs_release_capacity_without_mutating_originals(self):
  import sqlite3
  path=Path(self.directory.name)/'queue.sqlite3'
  with sqlite3.connect(path)as db:
   db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,status TEXT,expires REAL,envelope TEXT,report TEXT)')
   db.executemany('INSERT INTO jobs VALUES(?,?,?,?,?)',[
    ('expired-queued','queued',24,'original-queued',None),
    ('expired-leased','leased',25,'original-leased',None)])
  self.service.queue=SimpleNamespace(path=str(path))
  self.service.state['jobs']={i:dict(row_sha256=digest(self.row))for i in('expired-queued','expired-leased')}
  self.service.metadata.clear()  # No scientific dispatch; test capacity only.
  for group_size in(1,2):
   self.service.group_size=group_size
   result=self.service.tick(now=25)
   self.assertFalse(result.get('backpressure'))
   self.assertEqual(result['enqueued'],0)
  with sqlite3.connect(path)as db:
   self.assertEqual(db.execute('SELECT * FROM jobs ORDER BY id').fetchall(),[
    ('expired-leased','leased',25.,'original-leased',None),
    ('expired-queued','queued',24.,'original-queued',None)])
   db.execute("UPDATE jobs SET expires=26 WHERE id='expired-leased'")
  self.assertTrue(self.service.tick(now=25)['backpressure'])
 def test_malformed_deadline_is_not_silent_capacity_credit(self):
  from subnet.continuous_audit_service import inflight_status
  for value in(True,None,float('nan'),float('inf'),'25'):
   with self.assertRaises(ValueError):inflight_status(dict(status='leased',expires=value),25)
 def test_signed_closure_and_unsigned_mirror_count_once(self):
  key=SigningKey.generate();root=key.verify_key.encode().hex()
  with tempfile.TemporaryDirectory()as directory:
   p=Path(directory);record=dict(epoch='e1',completed_at=3601,round=1,checkpoint='a'*64)
   atomic(p/'e1-learner-completion.json',record);atomic(p/'e1-signed-learner-completion.json',signed(key,record))
   self.assertEqual(completed_learners(p,root),[record])
   atomic(p/'unsigned-learner-completion.json',dict(epoch='unsigned',completed_at=0))
   self.assertEqual(completed_learners(p,root),[record])
 def test_forged_signed_closure_fails_closed(self):
  key=SigningKey.generate();other=SigningKey.generate()
  with tempfile.TemporaryDirectory()as directory:
   atomic(Path(directory)/'e1-signed-learner-completion.json',signed(other,dict(epoch='e1',completed_at=3601)))
   with self.assertRaises(ValueError):completed_learners(directory,key.verify_key.encode().hex())
 def test_signed_closure_cannot_substitute_epoch(self):
  key=SigningKey.generate()
  with tempfile.TemporaryDirectory()as directory:
   atomic(Path(directory)/'e1-signed-learner-completion.json',signed(key,dict(epoch='e2',completed_at=3601)))
   with self.assertRaises(ValueError):completed_learners(directory,key.verify_key.encode().hex())
 def setUp(self):
  self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup);self.key=SigningKey.generate();self.root=self.key.verify_key.encode().hex();self.bucket=Bucket();self.queue=Queue()
  self.controller=SimpleNamespace(authority=SimpleNamespace(id=self.root),bucket=self.bucket,signed=lambda p:signed(self.key,p))
  policy=dict(version=VERSION,recent_epochs=4,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.25,zero_epoch_after=3,blacklist_after=4,blacklist_epochs=2)
  self.service=ContinuousAuditor(self.controller,self.queue,directory=self.directory.name,approved_sources={},job_metadata={},audit_policy=policy,max_inflight=1)
  import hashlib
  self.row=dict(epoch='e1',round=1,checkpoint='a'*64,miner='b'*64,env_id='math',index=0,batch_sha256='c'*64,proof_sha256=hashlib.sha256(b'proof').hexdigest(),commitment_sha256='e'*64,verifier_contract_sha256='f'*64,committed_at=20)
  artifact=dict(slot=0,batch_sha256='c'*64,sha256=self.row['proof_sha256'],size=5,key='original',frozen_key='immutable')
  self.p=dict(manifest_document=signed(self.key,dict(epoch='e1',start=10,deadline=20,source_bundle={'sha256':'9'*64})),receipts={'b'*64:{'artifacts':[artifact]}},records=[self.row])
  self.service.state['populations']['e1']=signed(self.key,self.p);self.service.sources['9'*64]={};self.service.metadata['9'*64]={}
 def test_hour_boundary_includes_all_new_completed_epochs_once(self):
  self.service.state['populations']['e2']=self.service.state['populations']['e1'];calls=[]
  def snapshot(epoch,round,checkpoint,cutoff):
   calls.append(epoch);return signed(self.key,dict(version=VERSION,epoch=epoch,round=round,checkpoint=checkpoint,cutoff=cutoff,policy=self.service.policy,population_sha256='1'*64,points={'b'*64:1}))
  completed=[dict(epoch='e1',round=1,checkpoint='a'*64,completed_at=1),dict(epoch='e2',round=2,checkpoint='a'*64,completed_at=3600),dict(epoch='old',round=0,checkpoint='a'*64,completed_at=0),dict(epoch='future',round=3,checkpoint='a'*64,completed_at=3601)]
  with patch.object(self.service,'hourly_snapshot',side_effect=snapshot),patch.object(self.service,'publish_immutable'):
   document=self.service.hourly_completed(completed,3600);self.service.hourly_completed(completed,3600)
  self.assertEqual(calls,['e1','e2']);self.assertEqual(document['payload']['points']['b'*64],2.)
 def test_penalty_hyperparameters_require_exact_operator_admission(self):
  from subnet.continuous_audit_service import admitted_service_config
  p=dict(version='continuous-audit-service-sources-v1',approved_sources={},job_metadata={},audit_policy=self.service.policy);config=dict(source_admission=signed(self.key,p),policy=self.service.policy)
  self.assertEqual(admitted_service_config(config,self.root),p)
  config['policy']={**self.service.policy,'invalid_multiplier':1.}
  with self.assertRaises(ValueError):admitted_service_config(config,self.root)
 def test_unsupported_source_is_deferred_before_proof_io(self):
  self.service.metadata.clear()
  with patch.object(self.service,'_capture',side_effect=AssertionError('unsupported source must not fetch proof')):result=self.service.tick(now=25)
  self.assertEqual(result['source_deferred'],1);self.assertEqual(result['selected'],0);self.assertEqual(self.service.state['draws'],{})
 def test_missed_hours_reconcile_original_completion_times_idempotently(self):
  self.service.state['populations']['e2']=self.service.state['populations']['e1'];self.service.state['populations']['e3']=self.service.state['populations']['e1'];calls=[]
  completed=[dict(epoch='e1',completed_at=3600),dict(epoch='e2',completed_at=3600.1),dict(epoch='e3',completed_at=10801)]
  def publish(c,cutoff):
   calls.append(cutoff);document=signed(self.key,dict(cutoff=cutoff));atomic(Path(self.directory.name)/('hourly-weights-'+str(cutoff)+'.json'),document);return document
  with patch.object(self.service,'hourly_completed',side_effect=publish):
   self.service.reconcile_hours(completed,10800);self.service.reconcile_hours(completed,10800)
  self.assertEqual(calls,[3600,7200,10800]);self.assertNotIn('14400',self.service.state['published_hours'])
 def test_failed_publication_not_marked_complete_or_backdated(self):
  completed=[dict(epoch='e1',completed_at=3601)];self.service.state['published_hours']={}
  with patch.object(self.service,'hourly_completed',side_effect=OSError('bucket unavailable')):
   with self.assertRaises(OSError):self.service.reconcile_hours(completed,7200)
  self.assertEqual(self.service.state['published_hours'],{})
  self.assertEqual(completed[0]['completed_at'],3601)
 def test_actual_conditional_copy_full_hash_before_capability(self):
  _,_,captured=self.service._capture(self.row,self.p);self.assertEqual(self.bucket.copies,[('original','immutable','original-etag')]);self.assertEqual(captured['read_url'],'https://private/immutable')
 def test_corrupt_full_bytes_cannot_claim_verified(self):
  self.bucket.body=b'wrong'
  with self.assertRaises(InvalidCommittedArtifact):self.service._capture(self.row,self.p)
 def test_deadline_overwrite_rejected(self):
  original=self.bucket.head_object;self.bucket.head_object=lambda **kw:dict(original(**kw),LastModified=datetime.datetime.fromtimestamp(20,datetime.timezone.utc))
  with self.assertRaises(InvalidCommittedArtifact):self.service._capture(self.row,self.p)
  self.assertFalse(self.bucket.copies)
 def test_infra_retries_same_draw_but_missing_is_invalid(self):
  self.bucket.error=ClientError({'Error':{'Code':'503'}},'HeadObject');first=self.service.tick(now=25);draw=dict(self.service.state['draws']);self.service.tick(now=26);self.assertEqual(draw,self.service.state['draws']);self.assertEqual(first['selected'],1);self.assertEqual(next(iter(self.service.state['capture_failures'].values()))['kind'],'infrastructure_error')
  self.bucket.error=ClientError({'Error':{'Code':'NoSuchKey'}},'HeadObject');self.service.tick(now=27);self.assertEqual(next(iter(self.service.state['capture_failures'].values()))['kind'],'confirmed_invalid_artifact');self.assertEqual(self.service.tick(now=28)['retried'],0)
 def test_restart_reuses_original_request_without_mutable_head(self):
  identity=digest(self.row);jobid='continuous-audit-'+identity[:32];self.service.state['draws'][identity]=dict(row=self.row,seed='1'*64,selected_at=25);job=dict(job_id=jobid,created_at=25,expires_at=100,manifest='immutable');envelope=signed(self.key,job);atomic(Path(self.directory.name)/(jobid+'-job.json'),envelope)
  with patch.object(self.service,'_capture',side_effect=AssertionError('must not redraw/rehead')):self.service.tick(now=26)
  self.assertEqual(self.queue.envelopes,[envelope])
 def test_actual_queued_status_enforces_backpressure(self):
  self.service.state['jobs']['job']=dict(row_sha256=digest(self.row));self.assertTrue(self.service.tick(now=25)['backpressure'])
if __name__=='__main__':unittest.main()
