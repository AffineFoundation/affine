import tempfile,threading,time,unittest,copy
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from nacl.signing import SigningKey
from test_commitment_child_queue import ChildQueueTests,sign
from subnet.continuous_audit_service import ContinuousAuditor
from subnet.continuous_audit_policy import VERSION,digest
from subnet.commitment_transport import make,VERSION2
from subnet.distributed_roles import Coordinator
class ParallelCapture(unittest.TestCase):
 def setUp(self):
  self.fixture=ChildQueueTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=self.fixture.root;self.authority=self.fixture.authority;self.owner=threading.get_ident();self.signed_threads=[]
  self.queue=Coordinator(Path(self.tmp.name)/'q.sqlite',self.authority,{},clock=time.time);self.queue.archive=lambda *a:None
  def signed(x):self.signed_threads.append(threading.get_ident());return sign(self.root,x)
  self.controller=SimpleNamespace(authority=SimpleNamespace(id=self.authority),signed=signed,bucket=object());policy=dict(version=VERSION,recent_epochs=8,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.1,zero_epoch_after=2,blacklist_after=3,blacklist_epochs=4)
  self.service=ContinuousAuditor(self.controller,self.queue,directory=Path(self.tmp.name)/'audit',approved_sources={},job_metadata={},audit_policy=policy,max_inflight=8,budget_per_tick=8,capture_workers=4)
  self.rows=[];self.capture_data={};self.completed={}
  for i in range(8):
   key=SigningKey.generate();identity=SimpleNamespace(key=key,id=key.verify_key.encode().hex());m=copy.deepcopy(self.fixture.manifest);m['submission_transport_policy']=VERSION2;m.pop('audit_frozen_receipts');m['audit_policy']={'mode':'full','version':1};batch=self.fixture.batch;env=make(identity,m,[(batch,b'ZIP actual bytes')]);child=env['payload']['batches'][0];frozen='public/'+m['epoch']+'/submissions/'+identity.id+'/'+digest(env)+'/0.zip';artifact=dict(child,frozen_key=frozen,key='private/'+str(i),read_url='https://example.r2.cloudflarestorage.com/bucket/'+frozen,etag='original',received_at=15);receipt=dict(sha256=digest(env),commitment_document=env,artifacts=[artifact]);row=dict(epoch=m['epoch'],round=1,checkpoint=m['checkpoint']['id'],miner=identity.id,env_id='math',index=17,batch_sha256=child['batch_sha256'],proof_sha256=child['sha256'],commitment_sha256=digest(env),verifier_contract_sha256='f'*64,committed_at=20);self.rows.append(row);self.capture_data[digest(row)]=(m,receipt,artifact)
  epoch=m['epoch'];self.service.state['populations'][epoch]=sign(self.root,{'manifest_document':sign(self.root,m)});self.service.dispatch_records=lambda:(self.rows,0);self.service.metadata[m['source_bundle']['sha256']]=dict(source_files={'subnet/model.py':'3'*64},runtime_versions={'torch':'pinned'})
 def test_four_parallel_captures_owner_signing_full_original_bindings_and_fresh_ttl(self):
  lock=threading.Lock();active=0;maximum=0
  def capture(row,p):
   nonlocal active,maximum
   with lock:active+=1;maximum=max(maximum,active)
   time.sleep(.025)
   with lock:active-=1;self.completed[digest(row)]=time.time()
   return self.capture_data[digest(row)]
  self.service._capture=capture;result=self.service.tick(now=1);self.assertEqual(result['enqueued'],8);self.assertEqual(maximum,4);self.assertEqual(set(self.signed_threads),{self.owner})
  for row in self.rows:
   identity=digest(row);path=self.service.directory/('continuous-audit-'+identity[:32]+'-job.json');import json;job=json.loads(path.read_text())['payload'];self.assertGreaterEqual(job['created_at'],self.completed[identity]);self.assertEqual(job['expires_at']-job['created_at'],900);self.assertEqual(job['submissions'][0]['commitment_ref']['commitment_sha256'],row['commitment_sha256'])
 def test_restart_reuses_exact_original_job_without_recapture_or_extension(self):
  row=self.rows[0];self.service.dispatch_records=lambda:([row],0);self.service._capture=lambda row,p:self.capture_data[digest(row)];self.service.tick(now=1)
  path=self.service.directory/('continuous-audit-'+digest(row)[:32]+'-job.json');original=path.read_bytes();draw=copy.deepcopy(self.service.state['draws']);self.service.state['jobs']={};self.service.persist();self.service._capture=lambda *a:(_ for _ in ()).throw(AssertionError('no recapture'))
  self.service.tick(now=100);self.assertEqual(path.read_bytes(),original);self.assertEqual(self.service.state['draws'],draw)
 def test_infra_failure_retains_selection_and_retry_no_redraw(self):
  row=self.rows[0];self.service.dispatch_records=lambda:([row],0);self.service._capture=lambda *a:(_ for _ in ()).throw(TimeoutError());self.service.tick(now=1);draw=copy.deepcopy(self.service.state['draws']);self.assertEqual(self.service.state['capture_failures'][digest(row)]['kind'],'infrastructure_error');self.service._capture=lambda row,p:self.capture_data[digest(row)];result=self.service.tick(now=2);self.assertEqual(result['selected'],0);self.assertEqual(self.service.state['draws'],draw);self.assertEqual(result['enqueued'],1)
if __name__=='__main__':unittest.main()
