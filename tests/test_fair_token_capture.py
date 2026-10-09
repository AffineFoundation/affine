import copy,datetime,hashlib,importlib.util,json,os,subprocess,sys,tempfile,time,threading,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

W=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(W),str(W/'tests')]
import test_capture_journal as fixtures
from ops.trainer_lifecycle import fair_token_capture as fair
spec=importlib.util.spec_from_file_location('subnet.fair_capture_test',W/'subnet/training_documents.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def crash_before_global_checkpoint(root):
 g,*_=fixtures.disk_gateway(root);original=fixtures.CaptureJournal.commit;count=[0]
 def commit(self,miner,receipt):
  original(self,miner,receipt);count[0]+=1
  if count[0]==77:os._exit(77)
 fixtures.CaptureJournal.commit=commit
 m.capture(g,'bounded-test',ordering=fair.prepare_order(g,'bounded-test'))


class FairCapture(unittest.TestCase):
 def gateway(self,root,count=40):
  g,calls,puts,persists=fixtures.disk_gateway(root,True,count)
  s=g.epochs['bounded-test'];s['start']=time.time()-20;s['deadline']=time.time()-1;s['commitment_binding']['freeze_until']=time.time()+10
  s['commitment_binding']['learner_capture_policy']=dict(fixtures.V2,workers=16,max_inflight_bytes=32000000)
  return g,s,calls,puts,persists
 def test_exact_commitments_close_before_entropy_issued(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,*_=self.gateway(d);s['deadline']=time.time()+2
   with patch.object(fair.secrets,'token_hex')as rng:
    with self.assertRaises(ValueError):fair.prepare_order(g,'bounded-test')
    rng.assert_not_called()
   self.assertNotIn('training_document_capture_order',s)
 def test_incomplete_commitment_capture_refuses(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,*_=self.gateway(d);s['commitment_capture_complete']=False
   with self.assertRaises(ValueError):fair.prepare_order(g,'bounded-test')
 def test_order_saved_before_any_GET_and_reused_after_process_state_readback(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,calls,*_=self.gateway(d)
   order=fair.prepare_order(g,'bounded-test');self.assertFalse(calls)
   restored,*_=fixtures.disk_gateway(d)
   with patch.object(fair.secrets,'token_hex')as rng:
    self.assertEqual(fair.prepare_order(restored,'bounded-test'),order);rng.assert_not_called()
 def test_modified_commitment_inventory_refuses_seed_reuse(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,*_=self.gateway(d);fair.prepare_order(g,'bounded-test');next(iter(s['commitment_pending'].values()))['sha256']='e'*64
   with self.assertRaises(ValueError):fair.prepare_order(g,'bounded-test')
 def test_order_schema_and_seed_mutants_refuse(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,*_=self.gateway(d);order=fair.prepare_order(g,'bounded-test')
   for field,value in [('version','wrong'),('epoch','other'),('seed','z'*64),('commitments_sha256','f'*64)]:
    with self.subTest(field=field):
     s['training_document_capture_order']=dict(order,**{field:value})
     with self.assertRaises(ValueError):fair.prepare_order(g,'bounded-test')
 def test_persist_failure_retries_same_seed_before_any_GET(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,calls,*_=self.gateway(d);persist=g.persist;g.persist=lambda:(_ for _ in()).throw(OSError('disk full'))
   with self.assertRaises(OSError):fair.prepare_order(g,'bounded-test')
   original=dict(s['training_document_capture_order']);self.assertFalse(calls);g.persist=persist
   with patch.object(fair.secrets,'token_hex')as rng:
    self.assertEqual(fair.prepare_order(g,'bounded-test'),original);rng.assert_not_called()
 def test_actual_capture_all_bytes_hashes_with_sixteen_worker_bound(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,calls,puts,persists=self.gateway(d);order=fair.prepare_order(g,'bounded-test');m.capture(g,'bounded-test',ordering=order)
   self.assertEqual(len(puts),40);self.assertFalse(s['rejections']);self.assertEqual(s['training_capture_runs'][-1]['ordering'],order)
   self.assertLessEqual(s['training_capture_runs'][-1]['maximum_inflight'],16);self.assertEqual(s['training_capture_runs'][-1]['raw_document_byte_limit'],32000000)
   m.capture(g,'bounded-test',ordering=order);self.assertEqual(len(calls),40)
 def test_postcommit_order_changes_lexical_first_window(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,calls,*_=self.gateway(d)
   with patch.object(fair.secrets,'token_hex',return_value='1'*64):order=fair.prepare_order(g,'bounded-test')
   m.capture(g,'bounded-test',ordering=order)
   first16={key.split('/')[-2]for key in calls[:16]}
   self.assertNotEqual(first16,set(sorted(s['commitment_pending'])[:16]))
 def test_expired_cutoff_does_not_issue_GET_under_fair_order(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,calls,*_=self.gateway(d);s['commitment_binding']['freeze_until']=time.time()-1;order=fair.prepare_order(g,'bounded-test');m.capture(g,'bounded-test',ordering=order)
   self.assertFalse(calls);self.assertFalse(s['rejections']);self.assertEqual(sum(map(len,s['training_document_deferred'].values())),40)
 def test_late_document_still_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,calls,puts,*_=self.gateway(d,1);original=g.bucket.client.get_object
   def late(**kw):
    result=original(**kw);result['LastModified']=datetime.datetime.fromtimestamp(s['deadline']+1,datetime.timezone.utc);return result
   g.bucket.client.get_object=late;m.capture(g,'bounded-test',ordering=fair.prepare_order(g,'bounded-test'))
   self.assertFalse(puts);self.assertEqual(len(s['rejections']),1)
 def test_policy_only_doubles_bounded_IO(self):
  old={'learner_capture_policy':copy.deepcopy(fixtures.V2),'source':'unchanged','deadline':60};new=copy.deepcopy(old);new['learner_capture_policy'].update(workers=16,max_inflight_bytes=32000000,state_checkpoint_documents=128)
  self.assertEqual(fair.validate_config(old,new),{'learner_capture_policy'})
  for field,value in [('deadline',120),('source','changed')]:
   with self.assertRaises(ValueError):fair.validate_config(old,dict(new,**{field:value}))
 def test_install_requires_exact_approved_code_and_preserves_historical_epoch(self):
  path=W/'subnet/training_documents.py';module=SimpleNamespace(capture=lambda g,e:'historical');installed=fair.install(module,path,hashlib.sha256(path.read_bytes()).hexdigest(),92)
  self.assertEqual(installed(None,'epoch-91'),'historical')
  with self.assertRaises(ValueError):fair.install(module,path,'0'*64,92)
 def test_bounded_128_checkpoint_interval_and_final_flush(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,calls,puts,persists=self.gateway(d,135);s['commitment_binding']['learner_capture_policy']['state_checkpoint_documents']=128
   order=fair.prepare_order(g,'bounded-test');m.capture(g,'bounded-test',ordering=order)
   self.assertEqual(len(puts),135);self.assertEqual(s['training_capture_runs'][-1]['global_state_checkpoints'],2)
   for bad in (True,0,129):
    with self.assertRaises(ValueError):m.capture_policy(dict(s['commitment_binding']['learner_capture_policy'],state_checkpoint_documents=bad))
 def test_real_process_crash_recovers_all_77_fsynced_rows_before_global_checkpoint(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,*_=self.gateway(d,100);s['commitment_binding']['learner_capture_policy']['state_checkpoint_documents']=128;s['commitment_binding']['freeze_until']=time.time()+2
   fair.prepare_order(g,'bounded-test')
   command="import sys;sys.path.insert(0,%r);from test_fair_token_capture import crash_before_global_checkpoint;crash_before_global_checkpoint(%r)"%(str(Path(__file__).parent),d)
   child=subprocess.run([sys.executable,'-I','-B','-c',command],timeout=5);self.assertEqual(child.returncode,77)
   before=json.loads((Path(d)/'gateway.json').read_text());self.assertEqual(sum(map(len,before.get('training_document_snapshots',{}).values())),0)
   while time.time()<s['commitment_binding']['freeze_until']:time.sleep(.02)
   restored,calls,puts,*_=fixtures.disk_gateway(d);m.capture(restored,'bounded-test',ordering=fair.prepare_order(restored,'bounded-test'))
   after=restored.epochs['bounded-test'];self.assertGreaterEqual(sum(map(len,after['training_document_snapshots'].values())),77)
   self.assertTrue(all(key.startswith('public/')for key in calls));self.assertFalse(puts);self.assertFalse(after['rejections'])
 def test_prospective_opening_changes_only_capture_policy_and_preserves_historical_open(self):
  class Controller:
   def open(self,epoch,**kwargs):return dict(epoch=epoch,**kwargs)
  path=W/'subnet/training_documents.py';module=SimpleNamespace(capture=lambda g,e:None);before=copy.deepcopy(fixtures.V2)
  fair.install(module,path,hashlib.sha256(path.read_bytes()).hexdigest(),92,controller_class=Controller,previous_policy=before)
  kwargs=dict(learner_capture_policy=before,deadline=60,source='unchanged')
  self.assertEqual(Controller().open('epoch-91',**kwargs),dict(epoch='epoch-91',**kwargs))
  after=Controller().open('epoch-92',**kwargs);self.assertEqual(after['deadline'],60);self.assertEqual(after['source'],'unchanged');self.assertEqual(before,fixtures.V2)
  self.assertEqual(after['learner_capture_policy'],dict(before,workers=16,max_inflight_bytes=32000000,state_checkpoint_documents=128))
  with self.assertRaises(ValueError):Controller().open('epoch-92',learner_capture_policy=dict(before,workers=4))
 def test_parallel_parent_PUTs_finish_before_final_receipt_publication(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,*_=self.gateway(d,20);m.capture(g,'bounded-test',ordering=fair.prepare_order(g,'bounded-test'));original=g.bucket.put;release=threading.Event();progress=threading.Event();calls=[];published=[];errors=[]
   first=s['commitment_pending'][sorted(s['commitment_pending'])[0]]['root']+'/commitment.json'
   def put(key,data):
    if key==first:release.wait(3)
    original(key,data);calls.append(key)
    if len(calls)>=8:progress.set()
   g.bucket.put=put;g.bucket.json=lambda k,v:published.append(copy.deepcopy(v))
   def run():
    try:m.freeze_receipts(g,'bounded-test',publication_workers=16)
    except BaseException as e:errors.append(e)
   t=threading.Thread(target=run);t.start()
   try:self.assertTrue(progress.wait(2));self.assertFalse(published);self.assertNotIn('frozen_receipts',s)
   finally:release.set();t.join(4)
   self.assertFalse(errors);self.assertFalse(t.is_alive());self.assertEqual(len(calls),20);self.assertEqual(published,[s['frozen_receipts']])
 def test_failed_parent_PUT_publishes_no_final_receipts_and_retry_is_exact(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,*_=self.gateway(d,8);m.capture(g,'bounded-test',ordering=fair.prepare_order(g,'bounded-test'));original=g.bucket.put;published=[];g.bucket.json=lambda k,v:published.append(copy.deepcopy(v));failed=s['commitment_pending'][sorted(s['commitment_pending'])[3]]['root']+'/commitment.json'
   def put(key,data):
    if key==failed:raise OSError('transient PUT')
    original(key,data)
   g.bucket.put=put
   with self.assertRaises(OSError):m.freeze_receipts(g,'bounded-test',publication_workers=16)
   self.assertNotIn('frozen_receipts',s);self.assertFalse(published);g.bucket.put=original
   result=m.freeze_receipts(g,'bounded-test',publication_workers=16);self.assertEqual(published,[result]);self.assertEqual(len(result),8)
 def test_all_parent_commitment_hashes_validate_before_any_parallel_PUT(self):
  with tempfile.TemporaryDirectory()as d:
   g,s,*_=self.gateway(d,3);m.capture(g,'bounded-test',ordering=fair.prepare_order(g,'bounded-test'));calls=[];g.bucket.put=lambda *a:calls.append(a);s['commitment_pending'][sorted(s['commitment_pending'])[-1]]['sha256']='f'*64
   with self.assertRaises(ValueError):m.freeze_receipts(g,'bounded-test',publication_workers=16)
   self.assertFalse(calls)

if __name__=='__main__':unittest.main()
