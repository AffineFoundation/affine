import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import test_commitment_production as fixtures
from subnet import commitment_transport as c,selected_proof_copy as s
from subnet.remote_backend import RemoteController
from subnet.forced_sampling import assurance

class SelectedCopies(unittest.TestCase):
 def fixture(self,count=3):
  fixture=fixtures.Tests();fixture.setUp();self.addCleanup(fixture.temp.cleanup)
  for i in range(count):fixture.submit(i,i+1)
  fixture.m.update(environments=[dict(env_id='math',indices=[1,2,3])],proof_copy_policy=dict(s.POLICY),hourly_execution_policy=dict(version='bounded-hourly-phases-v1',mine_seconds=20,freeze_seconds=60,audit_seconds=600,train_publication_seconds=1200,weight_seconds=300,slack_seconds=300))
  fixture.g.epochs['e']['commitment_binding'].update(proof_copy_policy=dict(s.POLICY),freeze_until=80)
  return fixture
 def freeze(self,f):
  with patch('subnet.selected_proof_copy.time.time',return_value=30):return c.freeze(f.g,'e')
 def manifest(self,f,receipts):
  from subnet.audit_policy import allocate
  seed='d'*64;alloc=allocate({m:len(r['artifacts'])for m,r in receipts.items()},f.m['audit_policy'],seed)
  return dict(f.m,audit_seed=seed,audit_frozen_receipts=receipts,audit_policy=dict(f.m['audit_policy'],submission_counts={r['sha256']:alloc[m]for m,r in receipts.items()}))
 def test_inventory_is_complete_without_any_heavy_copy_or_read(self):
  f=self.fixture();r=self.freeze(f)
  self.assertEqual(len(r),3);self.assertEqual(f.b.copies,[]);self.assertEqual(f.b.heavy_reads,0)
  for receipt in r.values():
   self.assertEqual(receipt['proof_copy_policy'],s.POLICY)
   for child in receipt['artifacts']:self.assertEqual(child['snapshot_status'],'metadata_captured');self.assertNotIn('read_url',child);self.assertNotIn(child['frozen_key'],f.b.objects)
 def test_original_random_draw_only_and_retry_no_duplicate_copy(self):
  f=self.fixture();r=self.freeze(f);m=self.manifest(f,r);slots=s.selected_slots(m,r)
  with patch('subnet.selected_proof_copy.time.time',return_value=100):
   results=s.snapshot_selected(f.g,m,slots);self.assertEqual(len(f.b.copies),1)
   self.assertEqual(s.snapshot_selected(f.g,m,slots),results);self.assertEqual(len(f.b.copies),1)
  m['selected_proof_copy_receipts']=results
  projected=s.audit_inventory(m,f.b)
  self.assertEqual(sum('read_url'in b for r in projected.values()for b in r['artifacts']),1)
  for miner,receipt in r.items():
   for child in receipt['artifacts']:
    if child['slot']in slots[miner]:s.require_snapshot(m,miner,child)
    else:
     with self.assertRaises(ValueError):s.require_snapshot(m,miner,child)
  changed=copy.deepcopy(m);changed['audit_seed']='e'*64
  with self.assertRaises(ValueError):s.snapshot_selected(f.g,changed,s.selected_slots(changed,r))
 def test_source_overwrite_is_infrastructure_no_redraw(self):
  f=self.fixture();r=self.freeze(f);m=self.manifest(f,r);slots=s.selected_slots(m,r)
  miner=next(x for x in r if slots[x]);child=r[miner]['artifacts'][0];f.b.put(child['key'],b'late replacement')
  with patch('subnet.selected_proof_copy.time.time',return_value=100):results=s.snapshot_selected(f.g,m,slots)
  row=results[miner+'/0'];self.assertEqual(row['status'],'infrastructure_deferred');self.assertEqual(f.b.copies,[])
  m['selected_proof_copy_receipts']=results
  with self.assertRaises(ValueError):s.require_snapshot(m,miner,child)
  report=c.unchecked(m,r[miner]);s.annotate_deferrals(m,miner,report)
  self.assertIsNone(report['outcomes'][0]['valid']);self.assertFalse(report['outcomes'][0]['fully_audited'])
 def test_expired_capture_does_not_publish_complete_inventory(self):
  f=self.fixture()
  with patch('subnet.selected_proof_copy.time.time',return_value=81):
   with self.assertRaises(c.FreezeMetadataIncomplete):c.freeze(f.g,'e')
  self.assertNotIn('frozen_receipts',f.g.epochs['e']);self.assertEqual(f.b.copies,[])
 def test_real_controller_dispatches_only_copied_draw_and_preserves_population(self):
  f=self.fixture();calls=[];controller=object.__new__(RemoteController);controller.state=f.state;controller.gateway=f.g;controller.bucket=f.b;controller.signed=lambda v:v;controller.training_execution_amendment_files={}
  def run(label,role,manifest,cp,submissions,**extra):
   calls.append(submissions);audits=[]
   from subnet.distributed_roles import validate_frozen_submissions
   validate_frozen_submissions(manifest,submissions)
   for obj in submissions:
    miner=obj['commitment_miner'];child=manifest['audit_frozen_receipts'][miner]['artifacts'][0]
    audits.append(dict(epoch='e',submission_sha256=obj['sha256'],accepted=[dict(env_id='math',index=child['index'],sample_index=child['index'])],outcomes=[dict(batch=0,index=child['index'],env_id='math',valid=True,fully_audited=True)],sampling_assurance=assurance(manifest)))
   return dict(audits=audits,job_id='CONTROL-ORIGINAL',backend_profile={},execution_resources_enforced=True)
  controller.jobs=SimpleNamespace(run=run,queue=object(),verifiers=[1,2]);f.g.freeze=lambda epoch:c.freeze(f.g,epoch)
  self.freeze(f)
  with patch('subnet.selected_proof_copy.time.time',return_value=100):result,reports=controller.finalize(f.m,'/unused')
  self.assertEqual(len(calls),1);self.assertEqual(len(f.b.copies),1);self.assertEqual(len(result['receipts']),3)
  self.assertEqual(sum(result['points'].values()),1);self.assertEqual(f.b.heavy_reads,0)
 def test_control_refuses_planned_copy_or_mutated_etag_time(self):
  f=self.fixture(1);r=self.freeze(f);m=self.manifest(f,r);miner=next(iter(r));child=r[miner]['artifacts'][0]
  with self.assertRaises(ValueError):s.require_snapshot(m,miner,child)
  with patch('subnet.selected_proof_copy.time.time',return_value=100):m['selected_proof_copy_receipts']=s.snapshot_selected(f.g,m,s.selected_slots(m,r))
  for key,value in [('source_etag','new'),('sha256','f'*64),('finished_at',680),('started_at',True),('full_SHA_verified',True)]:
   bad=copy.deepcopy(m);bad['selected_proof_copy_receipts'][miner+'/0'][key]=value
   with self.assertRaises(ValueError,msg=key):s.require_snapshot(bad,miner,child)
 def test_training_coverage_keeps_original_draw_hash_with_renewable_selected_URL(self):
  from subnet.training_policy import coverage_manifest,validate_coverage
  from subnet.persistent_cpu_adamw import POLICY
  f=self.fixture(1);r=self.freeze(f);m=self.manifest(f,r)
  with patch('subnet.selected_proof_copy.time.time',return_value=100):m['selected_proof_copy_receipts']=s.snapshot_selected(f.g,m,s.selected_slots(m,r))
  m['audit_frozen_receipts']=s.audit_inventory(m,f.b);m['training_policy']=POLICY
  challenge=dict(seed=m['audit_seed'],receipts=r,generated_after_freeze_at=90)
  covered=coverage_manifest(m,r,challenge)
  self.assertEqual(s.captured_inventory(covered),r)
  child=next(iter(r.values()))['artifacts'][0]
  self.assertEqual(validate_coverage(covered,[{'sha256':child['sha256']}]),covered['training_coverage'])
 def test_history_never_routes_uncopied_slots(self):
  import json
  from subnet.controller import Controller
  from subnet.publication import history
  from subnet.storage import canonical
  f=self.fixture();r=self.freeze(f);f.m['checkpoint']['files']={};f.m['source_bundle']['key']='public/sources/source.tar.gz'
  (f.state/'e-manifest.json').write_bytes(canonical(f.m))
  controller=Controller.__new__(Controller);controller.state=f.state;controller.bucket=f.b;controller.authority=f.ids[2];f.b.get=lambda key:f.b.objects[key][0]
  result=dict(epoch_id='e',receipts=r,payable=False)
  rows=history(controller,[result])['epochs'][0]['frozen']
  self.assertTrue(all('url'not in b for row in rows.values()for b in row['artifacts']))
  m=self.manifest(f,r)
  with patch('subnet.selected_proof_copy.time.time',return_value=100):m['selected_proof_copy_receipts']=s.snapshot_selected(f.g,m,s.selected_slots(m,r))
  m['audit_frozen_receipts']=s.audit_inventory(m,f.b);(f.state/'e-audit-manifest.json').write_bytes(canonical(m))
  rows=history(controller,[result])['epochs'][0]['frozen']
  self.assertEqual(sum('url'in b for row in rows.values()for b in row['artifacts']),1)
 def test_all_selected_copy_failures_dispatch_no_science_and_no_fraud(self):
  f=self.fixture();self.freeze(f);f.b.fail_copy=lambda key:True
  controller=RemoteController.__new__(RemoteController);controller.state=f.state;controller.gateway=f.g;controller.bucket=f.b;controller.signed=lambda x:x;f.g.freeze=lambda epoch:c.freeze(f.g,epoch)
  controller.jobs=SimpleNamespace(run=lambda *a,**kw:self.fail('no unavailable proof dispatch'),queue=object(),verifiers=[1,2])
  with patch('subnet.selected_proof_copy.time.time',return_value=100):result,reports=controller.finalize(f.m,'/unused')
  self.assertEqual(sum(result['points'].values()),0);self.assertEqual(f.b.copies,[])
  self.assertTrue(all(o['valid']is None and not o['fully_audited']for report in reports.values()for o in report['outcomes']))
 def test_dedicated_transport_bounds_and_conditional_original_ETag(self):
  from unittest.mock import Mock
  from botocore.config import Config
  from subnet.storage import Bucket
  shared=Mock();shared.meta.config=Config(connect_timeout=60,read_timeout=60)
  with tempfile.TemporaryDirectory()as folder:
   path=Path(folder)/'creds';path.write_text('R2_ACCESS_KEY_ID=test\nR2_SECRET_ACCESS_KEY=test\n')
   scoped=Mock()
   with patch('subnet.storage.boto3.client',side_effect=[shared,scoped])as factory:
    bucket=Bucket(dict(bucket='same',endpoint='https://same-storage',credentials_file=str(path)));bucket.selected_proof_client(60)
   opts=factory.call_args.kwargs
   self.assertEqual(opts['config'].read_timeout,30);self.assertEqual(opts['config'].connect_timeout,5)
   self.assertEqual(opts['config'].retries['total_max_attempts'],1);self.assertEqual(shared.meta.config.read_timeout,60)
   scoped.head_object.return_value=dict(ContentLength=7,ETag='destination')
   with patch.object(bucket,'selected_proof_client',return_value=scoped),patch('subnet.selected_proof_copy.time.time',return_value=100):
    result=s.conditional_snapshot(bucket,dict(key='old',frozen_key='new',etag='original',size=7),160)
   self.assertEqual(result['status'],'copied');self.assertFalse(result['full_SHA_verified'])
   scoped.copy_object.assert_called_once_with(Bucket='same',Key='new',CopySource={'Bucket':'same','Key':'old'},CopySourceIfMatch='original');scoped.close.assert_called_once()
 def test_selected_COPY_concurrency_is_four_and_inventory_HEAD_failure_is_infra(self):
  from test_commitment_freeze_prefetch import BoundedCommitmentFreeze
  import threading
  fixture=BoundedCommitmentFreeze();gateway,ids,_=fixture.fixture(12)
  gateway.epochs['e']['commitment_binding'].update(proof_copy_policy=dict(s.POLICY),freeze_until=80)
  with patch('subnet.selected_proof_copy.time.time',return_value=30):receipts=c.freeze(gateway,'e')
  f=self.fixture();manifest=self.manifest(f,receipts);manifest['audit_policy']['submission_counts']={r['sha256']:1 for r in receipts.values()}
  barrier=threading.Barrier(4);lock=threading.Lock();counts=dict(active=0,peak=0,started=0);original=gateway.bucket.copy
  def copied(*args,**kwargs):
   with lock:counts['active']+=1;counts['started']+=1;first=counts['started']<=4;counts['peak']=max(counts['peak'],counts['active'])
   try:
    if first:barrier.wait(timeout=5)
    return original(*args,**kwargs)
   finally:
    with lock:counts['active']-=1
  gateway.bucket.copy=copied
  with patch('subnet.selected_proof_copy.time.time',return_value=100):results=s.snapshot_selected(gateway,manifest,s.selected_slots(manifest,receipts))
  self.assertEqual(counts['peak'],4);self.assertEqual(len(results),12);self.assertTrue(all(x['status']=='copied'for x in results.values()))
  f=self.fixture(1)
  with patch.object(f.b,'head_object',side_effect=RuntimeError('storage unreachable')),patch('subnet.selected_proof_copy.time.time',return_value=30):
   with self.assertRaises(c.FreezeMetadataIncomplete):c.freeze(f.g,'e')
  self.assertNotIn(f.ids[0].id,f.g.epochs['e']['rejections']);self.assertNotIn('frozen_receipts',f.g.epochs['e'])
