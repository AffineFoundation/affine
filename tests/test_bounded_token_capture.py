import copy,datetime,io,threading,time,unittest
from types import SimpleNamespace
from subnet.storage import Identity
from subnet.commitment_transport import make,canonical,sha,VERSION2
from subnet.training_documents import document,capture,capture_policy
POLICY=dict(version='bounded-parallel-token-capture-v1',workers=8,max_document_bytes=2000000,max_inflight_bytes=16000000,completion_order='first-completed')
class ParallelCaptureTests(unittest.TestCase):
 def gateway(self,count=12,block=False):
  now=time.time();manifest=dict(epoch='bounded-test',checkpoint={'id':'a'*64},source_bundle={'sha256':'b'*64},submission_transport_policy=VERSION2);pending={};data={};first=None
  for i in range(count):
   who=Identity(bytes([i+1])*32);batch=dict(env_id='math',index=i,samples=[{'tokens':[1,2],'reward':True},{'tokens':[1,3],'reward':False}]);env=make(who,manifest,[(batch,b'proof')]);pending[who.id]=dict(document=env,root='public/test/'+who.id,sha256=sha(canonical(env)),size=len(canonical(env)),received_at=now);key='private/bounded-test/training/'+who.id+'/0.json';data[key]=document(batch,manifest,who.id,0)
  first='private/bounded-test/training/'+sorted(pending)[0]+'/0.json';state=dict(start=now-10,deadline=now+10,commitment_capture_complete=True,commitment_binding=dict(checkpoint='a'*64,freeze_until=now+15,learner_capture_policy=copy.deepcopy(POLICY)),commitment_pending=pending,rejections={});lock=threading.Lock();release=threading.Event();progress=threading.Event();meter={'active':0,'peak':0,'calls':0};puts={}
  def get(**kw):
   with lock:meter['active']+=1;meter['calls']+=1;meter['peak']=max(meter['peak'],meter['active'])
   try:
    if block and kw['Key']==first:release.wait(4)
    return dict(Body=io.BytesIO(data[kw['Key']]),LastModified=datetime.datetime.fromtimestamp(now,datetime.timezone.utc))
   finally:
    with lock:meter['active']-=1
  def persist():
   if sum(len(v)for v in state.get('training_document_snapshots',{}).values())>=9:progress.set()
  gateway=SimpleNamespace(epochs={'bounded-test':state},bucket=SimpleNamespace(name='bucket',client=SimpleNamespace(get_object=get),put=lambda k,b:puts.setdefault(k,b)),persist=persist)
  return gateway,state,meter,release,progress,puts
 def test_first_completed_refills_and_journals_while_first_GET_blocked(self):
  g,s,m,release,progress,puts=self.gateway(block=True);errors=[]
  def run():
   try:capture(g,'bounded-test')
   except BaseException as e:errors.append(e)
  t=threading.Thread(target=run);t.start()
  try:self.assertTrue(progress.wait(2),'FIFO head-of-line regression');self.assertNotIn(sorted(s['commitment_pending'])[0],s['training_document_snapshots'])
  finally:release.set();t.join(4)
  self.assertFalse(t.is_alive());self.assertFalse(errors);self.assertEqual(len(puts),12);self.assertLessEqual(m['peak'],8);self.assertEqual(s['training_capture_runs'][0]['GET_attempts'],12);self.assertLessEqual(s['training_capture_runs'][0]['maximum_inflight'],8)
 def test_resume_does_not_reread_or_republish_successful_originals(self):
  g,s,m,*_=self.gateway();capture(g,'bounded-test');capture(g,'bounded-test');self.assertEqual(m['calls'],12);self.assertEqual(sum(len(v)for v in s['training_document_snapshots'].values()),12)
 def test_invalid_signed_policy_refuses_before_network(self):
  for key,val in [('workers',True),('workers',32),('max_inflight_bytes',32000000),('max_document_bytes',4000000),('completion_order','fifo'),('version','wrong')]:
   g,s,m,*_=self.gateway();s['commitment_binding']['learner_capture_policy'][key]=val
   with self.assertRaises(ValueError):capture(g,'bounded-test')
   self.assertEqual(m['calls'],0)
 def test_16worker_bound_requires_exact32MB(self):
  v=dict(POLICY,workers=16,max_inflight_bytes=32000000);self.assertEqual(capture_policy(v),v)
 def test_expired_cutoff_submits_no_new_GET_and_preserves_deferred(self):
  g,s,m,*_=self.gateway();s['commitment_binding']['freeze_until']=time.time()-1;capture(g,'bounded-test');self.assertEqual(m['calls'],0);self.assertEqual(sum(len(v)for v in s['training_document_deferred'].values()),12);self.assertFalse(s['rejections']);self.assertEqual(s['training_capture_runs'][0]['deferred_slots'],12)
 def test_old_unspecified_policy_preserves_no_new_run_receipt(self):
  g,s,m,*_=self.gateway(1);del s['commitment_binding']['learner_capture_policy'];capture(g,'bounded-test');self.assertNotIn('training_capture_runs',s);self.assertEqual(m['calls'],1)
 def test_signed_concurrency_sizes_real_Bucket_read_connection_pool(self):
  from unittest.mock import patch
  from botocore.config import Config
  from subnet.storage import Bucket
  b=object.__new__(Bucket);b.client=SimpleNamespace(meta=SimpleNamespace(config=Config()));b._client_options={}
  with patch('subnet.storage.boto3.client')as factory:
   b.commitment_read_client(parallel_workers=16)
   c=factory.call_args.kwargs['config'];self.assertEqual(c.max_pool_connections,16);self.assertEqual(c.read_timeout,10);self.assertEqual(c.connect_timeout,5);self.assertEqual(c.retries['total_max_attempts'],1)
  with patch('subnet.storage.boto3.client')as factory:
   with self.assertRaises(ValueError):b.commitment_read_client(parallel_workers=True)
   factory.assert_not_called()
 def test_late_uploaded_document_never_published_under_parallel_policy(self):
  g,s,m,release,progress,puts=self.gateway(1);original=g.bucket.client.get_object
  def late(**kw):
   r=original(**kw);r['LastModified']=datetime.datetime.fromtimestamp(s['deadline']+1,datetime.timezone.utc);return r
  g.bucket.client.get_object=late;capture(g,'bounded-test');self.assertFalse(puts);self.assertEqual(len(s['rejections']),1);self.assertEqual(s['training_capture_runs'][0]['structural_failures'],1)

class CaptureOpeningTests(unittest.TestCase):
 def test_first_signed_manifest_and_durable_gateway_bind_policy_copy(self):
  import json,tempfile
  from pathlib import Path
  from subnet.controller import Controller
  from subnet.storage import Gateway
  from subnet.backend_jobs import signed
  from test_real_gpu_epoch_open import MemoryBucket
  hourly=dict(version='bounded-hourly-phases-v1',mine_seconds=60,freeze_seconds=60,audit_seconds=600,train_publication_seconds=2100,weight_seconds=60,slack_seconds=120)
  with tempfile.TemporaryDirectory()as folder:
   bucket=MemoryBucket();gateway=Gateway(bucket,state_path=Path(folder)/'gateway.json',direct_r2=True)
   try:
    controller=Controller(bucket,gateway,Path(folder)/'controller');raw=copy.deepcopy(POLICY)
    manifest=controller.open('nonpayable-capture',dict(id='a'*64,files={}),[Identity().id],duration=60,source_bundle={'sha256':'b'*64},submission_transport_policy=VERSION2,training_input_policy='committed-unaudited-training-v1',hourly_execution_policy=hourly,learner_capture_policy=raw)
    raw['workers']=16
    public=signed(json.loads(bucket.objects['public/nonpayable-capture/manifest.json']),controller.authority.id)
    self.assertEqual(public,manifest);self.assertEqual(public['learner_capture_policy'],POLICY)
    self.assertEqual(gateway.epochs['nonpayable-capture']['commitment_binding']['learner_capture_policy'],POLICY)
    durable=json.loads((Path(folder)/'gateway.json').read_text())
    self.assertEqual(durable['epochs']['nonpayable-capture']['commitment_binding']['learner_capture_policy'],POLICY)
   finally:gateway.server.shutdown();gateway.server.server_close();gateway.thread.join()
 def test_invalid_or_incompatible_capture_refuses_before_gateway_side_effect(self):
  import tempfile
  from unittest.mock import Mock
  from subnet.controller import Controller
  for fields in [dict(learner_capture_policy=dict(POLICY,workers=32)),dict(learner_capture_policy=POLICY),dict(learner_capture_policy=POLICY,training_input_policy='committed-unaudited-training-v1',submission_transport_policy=VERSION2)]:
   with tempfile.TemporaryDirectory()as folder:
    gateway=Mock();controller=Controller(Mock(),gateway,folder)
    with self.assertRaises(ValueError):controller.open('nonpayable-invalid',{},[],**fields)
    gateway.open.assert_not_called()
 def test_config_contract_copies_explicit_policy_and_rejects_wrong_learner(self):
  from unittest.mock import patch
  from subnet.gpu_service import contract
  from subnet.backend_jobs import COVERED_POLICY
  row=dict(spec=dict(id='math',version='fixed-v1',num_samples=4,max_output_tokens=512),indices=[0],harness=dict(version='text-tools-v1'))
  raw=copy.deepcopy(POLICY)
  config=dict(source_bundle={},heldout=[],training_policy=COVERED_POLICY,learner_capture_policy=raw,training_input_policy='committed-unaudited-training-v1',submission_transport_policy=VERSION2,hourly_execution_policy={'placeholder':True})
  with patch('subnet.gpu_service.definitions',return_value=[row]):
   chosen=contract(config,0);raw['workers']=16;self.assertEqual(chosen['learner_capture_policy'],POLICY)
   config['learner_capture_policy']=copy.deepcopy(POLICY)
   del config['training_input_policy']
   with self.assertRaisesRegex(ValueError,'hourly unaudited'):contract(config,0)

if __name__=='__main__':unittest.main()
