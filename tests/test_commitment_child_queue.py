"""Actual SQLite/Ed25519 claim/report controls, not numerical GPU qualification."""
import base64,copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from nacl.signing import SigningKey
from subnet.distributed_roles import Coordinator,digest
from subnet.storage import canonical
from subnet.commitment_transport import make,VERSION
from subnet.backend_profiles import for_config

def sign(key,p):return dict(payload=p,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(p)).signature).decode())

class ChildQueueTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=SigningKey.generate();self.worker=SigningKey.generate();self.miner=SigningKey.generate();self.authority=self.root.verify_key.encode().hex();self.identity=SimpleNamespace(id=self.miner.verify_key.encode().hex(),key=self.miner);self.now=100.;self.nonce=0
  self.queue=Coordinator(Path(self.tmp.name)/'q.sqlite',self.authority,{self.worker.verify_key.encode().hex():['verify']},clock=lambda:self.now)
  revision,profile,policy=for_config({'model_runtime_revision':'cuda-bf16-eager-sm90-v1'})
  self.manifest=dict(epoch='nonpayable-child-controls',payable=False,checkpoint={'id':'1'*64},source_bundle={'sha256':'2'*64},max_batches=3,environments=[dict(env_id='math',indices=[17])],submission_transport_policy=VERSION,model_runtime_revision=revision,backend_profile=profile,numerical_policy=policy)
  batch=dict(env_id='math',index=17,epoch=self.manifest['epoch'],checkpoint=self.manifest['checkpoint']['id'],rollouts=[dict(tokens=[1,2])]);self.batch=batch;envelope=make(self.identity,self.manifest,[(batch,b'ZIP actual bytes')]);row=envelope['payload']['batches'][0];key='public/'+self.manifest['epoch']+'/submissions/'+self.identity.id+'/'+digest(envelope)+'/0.zip';url='https://example.r2.cloudflarestorage.com/bucket/'+key+'?X-Amz-Signature=test';artifact=dict(row,frozen_key=key,read_url=url);receipt=dict(sha256=digest(envelope),commitment_document=envelope,artifacts=[artifact]);self.manifest['audit_frozen_receipts']={self.identity.id:receipt}
  ref=dict(miner=self.identity.id,commitment_sha256=receipt['sha256'],**{k:artifact[k]for k in ('slot','env_id','index','batch_sha256','size','frozen_key')});self.obj=dict(url=url,sha256=row['sha256'],commitment_miner=self.identity.id,commitment_ref=ref)
  self.job=dict(schema=1,job_id='child-job',role='verify',created_at=90.,expires_at=200.,manifest=sign(self.root,self.manifest),source_files={'subnet/model.py':'3'*64},runtime_versions={'torch':'pinned'},submissions=[self.obj])
 def enqueue(self,job=None):return self.queue.enqueue(sign(self.root,job or self.job))
 def req(self,action,**fields):
  self.nonce+=1;return self.queue.request(sign(self.worker,dict(action=action,at=self.now,nonce=str(self.nonce).zfill(32),**fields)))
 def report(self):return dict(job_id=self.job['job_id'],job_sha256=digest(self.job),operator=self.authority,role='verify',epoch=self.manifest['epoch'],checkpoint=self.manifest['checkpoint']['id'],source_files=self.job['source_files'],runtime_versions=self.job['runtime_versions'],backend_profile=self.manifest['backend_profile'],numerical_policy=self.manifest['numerical_policy'],chain_transactions=False,success=True,completed_at=self.now,audits=[dict(epoch=self.manifest['epoch'],submission_sha256=self.obj['sha256'],accepted=[copy.deepcopy(self.batch)])])
 def test_actual_enqueue_authenticated_claim_worker_report_and_export(self):
  self.enqueue();claim=self.req('claim',role='verify')['claim'];self.assertEqual(claim['job']['payload'],self.job);self.assertTrue(self.req('report',job_id=self.job['job_id'],token=claim['token'],report=self.report())['accepted']);self.assertEqual(self.queue.status(self.job['job_id'])['status'],'complete');
  with self.queue.transaction()as db:stored=json.loads(db.execute('select report_request from jobs where id=?',(self.job['job_id'],)).fetchone()[0])
  self.assertEqual(stored['signer'],self.worker.verify_key.encode().hex())
 def test_real_worker_HTTP_claim_and_authenticated_report_admission(self):
  import threading
  from unittest.mock import patch
  from subnet.distributed_roles import CoordinatorServer
  from subnet.distributed_worker import Worker
  self.enqueue();server=CoordinatorServer(('127.0.0.1',0),self.queue,lambda value:sign(self.root,value));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   worker=Worker('http://127.0.0.1:'+str(server.server_port),bytes(self.worker),self.authority,Path(self.tmp.name)/'worker')
   with patch('subnet.distributed_worker.time.time',return_value=self.now):
    claim=worker.request('claim',role='verify')['claim'];self.assertEqual(claim['job']['payload']['submissions'],[self.obj])
    result=worker.request('report',job_id=self.job['job_id'],token=claim['token'],report=self.report());self.assertTrue(result['accepted'])
   self.assertEqual(self.queue.status(self.job['job_id'])['status'],'complete')
  finally:server.shutdown();server.server_close();thread.join()
 def test_reject_mutated_child_hash_size_slot_miner_environment_index_key(self):
  changes=[('sha256','9'*64),('ref:size',99),('ref:slot',1),('ref:slot',False),('ref:miner','9'*64),('ref:env_id','foreign'),('ref:index',18),('ref:frozen_key','public/foreign.zip'),('ref:batch_sha256','9'*64),('ref:commitment_sha256','9'*64),('url','https://foreign.example/bucket/elsewhere'),('url',self.obj['url'].replace('/0.zip','/other.zip'))]
  for name,value in changes:
   with self.subTest(name=name):
    job=copy.deepcopy(self.job)
    if name.startswith('ref:'):job['submissions'][0]['commitment_ref'][name[4:]]=value
    else:job['submissions'][0][name]=value
    with self.assertRaises(ValueError):self.enqueue(job)
 def test_reject_malformed_child_objects_before_queue_insertion(self):
  for obj in (None,'bad',dict(self.obj,url=None),dict(self.obj,unexpected=True)):
   job=copy.deepcopy(self.job);job['submissions']=[obj]
   with self.assertRaises(ValueError):self.enqueue(job)
  with self.queue.transaction()as db:self.assertEqual(db.execute('select count(*)from jobs').fetchone()[0],0)
 def test_reject_parent_payload_scope_and_foreign_artifact_inventory(self):
  for what in ('source','checkpoint','artifact','signature'):
   with self.subTest(what=what):
    m=copy.deepcopy(self.manifest);receipt=m['audit_frozen_receipts'][self.identity.id]
    if what in('source','checkpoint'):
     payload=copy.deepcopy(receipt['commitment_document']['payload']);payload[what]='8'*64;receipt['commitment_document']=sign(self.miner,payload)
    elif what=='artifact':receipt['artifacts'][0]['size']+=1
    else:receipt['commitment_document']['signature']=base64.b64encode(b'0'*64).decode()
    job=copy.deepcopy(self.job);job['manifest']=sign(self.root,m)
    with self.assertRaises((ValueError,)):self.enqueue(job)
 def test_reject_missing_metadata_and_duplicate_child(self):
  for mutate in ('missing','duplicate'):
   job=copy.deepcopy(self.job)
   if mutate=='missing':del job['submissions'][0]['commitment_ref']
   else:job['submissions'].append(copy.deepcopy(job['submissions'][0]))
   with self.assertRaises(ValueError):self.enqueue(job)
 def test_report_rejects_foreign_environment_and_index(self):
  self.enqueue();claim=self.req('claim',role='verify')['claim']
  for field,value in (('env_id','foreign'),('index',18)):
   report=self.report();report['audits'][0]['accepted'][0][field]=value
   with self.assertRaises(ValueError):self.req('report',job_id=self.job['job_id'],token=claim['token'],report=report)
 def test_reject_altered_parent_digest_even_when_reference_matches_it(self):
  m=copy.deepcopy(self.manifest);m['audit_frozen_receipts'][self.identity.id]['sha256']='7'*64
  job=copy.deepcopy(self.job);job['manifest']=sign(self.root,m);job['submissions'][0]['commitment_ref']['commitment_sha256']='7'*64
  with self.assertRaisesRegex(ValueError,'canonical parent commitment digest'):self.enqueue(job)
 def test_reject_changed_accepted_tokens_and_scope(self):
  self.enqueue();claim=self.req('claim',role='verify')['claim']
  for what in ('tokens','scope'):
   report=self.report();batch=report['audits'][0]['accepted'][0]
   if what=='tokens':batch['rollouts'][0]['tokens'][1]=99
   else:batch['epoch']='foreign'
   with self.assertRaises(ValueError):self.req('report',job_id=self.job['job_id'],token=claim['token'],report=report)
 def test_legacy_frozen_zip_remains_supported(self):
  m=copy.deepcopy(self.manifest);del m['submission_transport_policy'];m['audit_frozen_receipts']={'legacy':{'sha256':'oldZIP'}};j=dict(self.job,manifest=sign(self.root,m),submissions=[{'url':'oldcap','sha256':'oldZIP'}]);self.enqueue(j);self.assertIsNotNone(self.req('claim',role='verify')['claim'])

class V2ChildQueueTests(ChildQueueTests):
 def setUp(self):
  super().setUp()
  from subnet.commitment_transport import VERSION2
  self.manifest['submission_transport_policy']=VERSION2
  envelope=make(self.identity,self.manifest,[(self.batch,b'ZIP actual bytes')]);row=envelope['payload']['batches'][0];key='public/'+self.manifest['epoch']+'/submissions/'+self.identity.id+'/'+digest(envelope)+'/0.zip';url='https://example.r2.cloudflarestorage.com/bucket/'+key+'?X-Amz-Signature=test';artifact=dict(row,frozen_key=key,read_url=url);receipt=dict(sha256=digest(envelope),commitment_document=envelope,artifacts=[artifact]);self.manifest['audit_frozen_receipts']={self.identity.id:receipt}
  self.obj=dict(url=url,sha256=row['sha256'],commitment_miner=self.identity.id,commitment_ref=dict(miner=self.identity.id,commitment_sha256=receipt['sha256'],**{k:artifact[k]for k in ('slot','env_id','index','batch_sha256','size','frozen_key')}));self.job.update(manifest=sign(self.root,self.manifest),submissions=[self.obj])
 def test_v2_token_digest_binding_cannot_be_substituted(self):
  m=copy.deepcopy(self.manifest);m['audit_frozen_receipts'][self.identity.id]['artifacts'][0]['training_sha256']='9'*64
  with self.assertRaisesRegex(ValueError,'inventory binding'):self.enqueue(dict(self.job,manifest=sign(self.root,m)))
 def test_transport_downgrade_rejected(self):
  m=copy.deepcopy(self.manifest);m['submission_transport_policy']=VERSION
  with self.assertRaisesRegex(ValueError,'original miner commitment transport'):self.enqueue(dict(self.job,manifest=sign(self.root,m)))
 def test_unknown_transport_rejected(self):
  m=copy.deepcopy(self.manifest);m['submission_transport_policy']='unknown'
  with self.assertRaisesRegex(ValueError,'explicit child'):self.enqueue(dict(self.job,manifest=sign(self.root,m)))
