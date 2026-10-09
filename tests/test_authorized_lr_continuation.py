"""Tiny real Adam1→2 CPU continuation; no production job or inputs issued."""
import copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'tests')]
import torch
import test_authorized_lr_evidence as baseline
from test_learning_rate_transition import Store
from subnet import persistent_training_evidence as evidence
from subnet.persistent_cpu_adamw import HYPERPARAMETERS,PersistentCPUAdamW,sha
from subnet.persistent_training_state import resource_plan,admit_resources,export_state,restore_state,HEADER_RESERVE
from subnet.learning_rate_transition import VERSION

class Continuation(unittest.TestCase):
 def setUp(self):
  self.parent_store=Store();base=baseline.AuthorizedEvidence();self.addCleanup(base.doCleanups)
  with patch.object(baseline,'Store',return_value=self.parent_store):base.setUp()
  self.base=base;self.f=base.f;self.parent=base.report['persistent_training_state']['descriptor'];self.manifest=copy.deepcopy(base.manifest)
  binding=self.manifest['trainer_state_binding'];binding.update(genesis=None,parent={'descriptor_sha256':sha(self.parent)},global_step_before=1)
  self.job=copy.deepcopy(base.job);self.job['job_id']='private-cpu-successor';self.job['manifest']=self.f.sign(self.manifest);self.job['persistent_training']['global_step_after']=2
  grant={k:v for k,v in base.grant.items()if k!='run_id'};grant.update(version=VERSION,job_id=self.job['job_id'],parent_descriptor_sha256=sha(self.parent),optimizer_step_before=1)
  self.authorization=self.f.sign(grant);self.grant=grant
  workspace=self.f.root/'continuation';workspace.mkdir();admission=admit_resources(workspace,resource_plan(self.f.inventory,bf16_export_bytes=100,transfer_bytes=HEADER_RESERVE+256,ram_reserve_bytes=0,disk_reserve_bytes=0))
  restored,_=restore_state(self.parent,sha(self.parent),self.f.cp['id'],self.f.inventory,workspace=workspace,fetch_shard=self.parent_store.fetch,resource_admission=admission)
  parameters=[('weight',torch.nn.Parameter(restored[1]['weight']['master'].to(torch.bfloat16)))]
  with patch('subnet.learning_rate_transition.time.time',return_value=1000):
   optimizer=PersistentCPUAdamW(parameters,self.f.cp['id'],restored=restored,learning_rate_authorization=self.authorization,learning_rate_authority=self.f.authority,epoch=self.manifest['epoch'],job_id=self.job['job_id'],steps=1)
  self.assertEqual(optimizer.global_step,1);optimizer.step(gradients={'weight':torch.ones(2,dtype=torch.float32)})
  store=Store();descriptor,_=export_state(optimizer,epoch=self.manifest['epoch'],inference_checkpoint=self.f.cp['id'],workspace=workspace,publish_shard=store.put,readback_shard=store.read,commit_descriptor=store.commit,resource_admission=admission,shard_bytes=HEADER_RESERVE+256)
  self.declaration=dict(base.declaration,version='unaudited-training-execution-amendment-v2-effective-lr',method='fp32-task-gradient-effective-lr-v1',job_id=self.job['job_id'],original_signed_manifest_sha256=sha(self.job['manifest']),trainer_binding_sha256=sha(binding),parent_descriptor_sha256=sha(self.parent),optimizer_step_before=1,learning_rate_authorization=self.authorization)
  self.job['unaudited_training_execution']=self.f.sign(self.declaration)
  self.report=copy.deepcopy(base.report);self.report['persistent_training_state'].update(descriptor=descriptor,descriptor_sha256=sha(descriptor));training=self.report['training'];training.update(global_step_before=1,global_step_after=2)
  update=training['updates'][0];update.update(global_optimizer_step=2,precision=copy.deepcopy(optimizer.last_update));diagnostics=training['persistent_diagnostics'];diagnostics.update(global_optimizer_step_before=1,global_optimizer_step_after=2,learning_rate_authorization_sha256=sha(self.authorization))
 def test_genuine_v3_parent_restored_v2_successor_step1_to2(self):
  result=evidence.validate_updates(self.report,self.job,self.manifest)
  self.assertEqual(result['global_step_after'],2);d=self.report['persistent_training_state']['descriptor'];self.assertEqual(d['parent_state_sha256'],sha(self.parent));self.assertEqual(d['hyperparameters']['lr'],5e-7);self.assertEqual(d['genesis_sha256'],self.parent['genesis_sha256']);self.assertEqual(HYPERPARAMETERS['lr'],1e-5)
 def test_validly_signed_foreign_parent_counter_or_rate_rejected(self):
  for field,value in(('parent_descriptor_sha256','e'*64),('optimizer_step_before',0),('effective_learning_rate',1e-5)):
   with self.subTest(field=field):
    d=dict(self.declaration,learning_rate_authorization=self.f.sign(dict(self.grant,**{field:value})));job=dict(self.job,unaudited_training_execution=self.f.sign(d))
    with self.assertRaises(ValueError):evidence.validate_updates(self.report,job,self.manifest)
 def test_successor_cannot_reuse_genesis_mode(self):
  d=dict(self.declaration,version='unaudited-training-execution-amendment-v3-effective-lr-genesis',method='fp32-task-gradient-effective-lr-genesis-v1');job=dict(self.job,unaudited_training_execution=self.f.sign(d))
  with self.assertRaises(ValueError):evidence.validate_updates(self.report,job,self.manifest)
 def test_report_cannot_change_actual_effective_rate_or_global_counter(self):
  for field,value in(('lr',1e-5),('global_optimizer_step',1)):
   r=copy.deepcopy(self.report)
   if field=='lr':r['training']['updates'][0]['hyperparameters']['lr']=value
   else:r['training']['updates'][0][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):evidence.validate_updates(r,self.job,self.manifest)
if __name__=='__main__':unittest.main()
