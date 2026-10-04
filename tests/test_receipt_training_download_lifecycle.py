import base64,copy,hashlib,unittest
from nacl.signing import SigningKey
from ops.receipt_training_download_lifecycle import (canonical,sha,prepare_completion,verify_durable_archives,retirement_grant,on_authenticated_training_completion)
class CompletionLifecycle(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();source='a'*64;cp='b'*64;data=b'durable rollout artifact';digest=hashlib.sha256(data).hexdigest();self.data=data
  manifest=dict(epoch='epoch-1',source_bundle={'sha256':source},audit_frozen_receipts={'miner':dict(sha256=digest,size=len(data),frozen_key='public/epoch-1/submissions/'+'c'*64+'.zip')})
  job=dict(job_id='epoch-1-train-1',role='train',manifest=self.sign(manifest),source_files={'subnet/backend_jobs.py':'d'*64},runtime_versions={'torch':'pinned'},submissions=[dict(sha256=digest,size=len(data),verifier_receipt=self.sign({'test_fixture_only':True}))])
  self.envelope=self.sign(job);self.report=dict(success=True,job_id=job['job_id'],role='train',job_sha256=sha(job),source_files=job['source_files'],runtime_versions=job['runtime_versions'],new_checkpoint=dict(id=cp,files={'model.safetensors':'e'*64},path='/worker/jobs/epoch-1-train-1/checkpoint-final'))
  self.remote=dict(workspace='/worker',job_sha256=sha(self.envelope),report_sha256=sha(self.report),terminal_sha256='f'*64,terminal=dict(job_id=job['job_id'],phase='complete',exit_code=0))
  self.readback=dict(all_checkpoint_objects_fully_read=True,successor_checkpoint=cp,objects={'model.safetensors':dict(bytes=4,sha256='e'*64)})
  self.sources={source:job['source_files']};self.calls=[]
 def sign(self,v):return dict(payload=v,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(v)).signature).decode())
 def plan(self):return prepare_completion(self.envelope,self.report,self.remote,self.authority,self.readback,validate_receipt_report=lambda *x:self.calls.append('receipt_report_validated'),approved_execution_sources=self.sources)
 def test_normal_completion_and_protection(self):
  p=self.plan();self.assertEqual(len(p['objects']),1);self.assertEqual(self.calls,['receipt_report_validated']);self.assertEqual(p['protected_final_export'],self.report['new_checkpoint']['path'])
 def test_signed_approved_execution_source_required(self):
  self.sources={}
  with self.assertRaisesRegex(ValueError,'approved original'):self.plan()
 def test_source_runtime_report_must_match(self):
  self.report['runtime_versions']={'torch':'different'};self.remote['report_sha256']=sha(self.report)
  with self.assertRaisesRegex(ValueError,'source/runtime'):self.plan()
 def test_original_terminal_zero_required(self):
  self.remote['terminal']['exit_code']=1
  with self.assertRaisesRegex(ValueError,'completion'):self.plan()
 def test_report_appearing_before_terminal_insufficient(self):
  self.remote['terminal']['phase']='running'
  with self.assertRaisesRegex(ValueError,'completion'):self.plan()
 def test_current_output_full_readback_required(self):
  self.readback['all_checkpoint_objects_fully_read']=False
  with self.assertRaisesRegex(ValueError,'full readback'):self.plan()
 def test_original_request_signature_required(self):
  self.envelope['payload']['role']='evaluate'
  with self.assertRaises(Exception):self.plan()
 def test_full_bounded_durable_archive_before_grant(self):
  p=self.plan();records=[];r=verify_durable_archives(p,lambda key:iter([self.data[:3],self.data[3:]]),records.append);g=retirement_grant(p,r)
  self.assertEqual(records,r);self.assertTrue(g['no_model_or_optimizer_deletion']);self.assertEqual(g['plans'][0]['kind'],'submission')
 def test_mutated_durable_bytes_never_get_grant(self):
  p=self.plan();records=[]
  with self.assertRaisesRegex(ValueError,'durable archive SHA'):verify_durable_archives(p,lambda key:iter([b'fake']),records.append)
  self.assertFalse(records)
 def test_partial_durable_readbacks_refused(self):
  with self.assertRaisesRegex(ValueError,'complete durable'):retirement_grant(self.plan(),[])
 def test_lifecycle_enqueues_only_after_full_archives(self):
  events=[]
  result=on_authenticated_training_completion(self.envelope,self.report,self.remote,self.authority,self.readback,validate_receipt_report=lambda *a:events.append('authenticate'),approved_execution_sources=self.sources,read_object=lambda k:iter([self.data]),record_readback=lambda r:events.append('readback'),enqueue_authenticated_retirement=lambda g:events.append('enqueue'))
  self.assertEqual(events,['authenticate','readback','enqueue'])
if __name__=='__main__':unittest.main()
