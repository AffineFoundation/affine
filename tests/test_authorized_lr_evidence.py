"""CPU signed-job LR attribution controls using actual tiny Adam state exports."""
import copy
import unittest
from unittest.mock import patch

import torch
import test_persistent_training_integration as fixture
from test_learning_rate_transition import Store
from subnet import persistent_training_evidence as evidence
from subnet.persistent_cpu_adamw import HYPERPARAMETERS,PersistentCPUAdamW,sha
from subnet.persistent_training_state import resource_plan,admit_resources,export_state
from subnet.learning_rate_transition import GENESIS_AUTH_VERSION,genesis_document
from ops.trainer_lifecycle import authorized_lr_evidence as helper


class AuthorizedEvidence(unittest.TestCase):
 def setUp(self):
  f=fixture.PersistentIntegrationTests();f.setUp();self.addCleanup(f.doCleanups);self.f=f
  self.legacy_report,self.legacy_job=f.report(f.job(steps=1));self.manifest=copy.deepcopy(f.manifest)
  binding=self.manifest['trainer_state_binding'];g=genesis_document(sha(f.inventory),f.cp['id'],'9'*64,5e-7)
  binding.update(genesis=g,genesis_sha256=sha(g))
  self.grant=dict(version=GENESIS_AUTH_VERSION,epoch=self.manifest['epoch'],job_id=self.legacy_job['job_id'],input_checkpoint=f.cp['id'],parent_descriptor_sha256=None,genesis_sha256=sha(g),optimizer_step_before=0,steps=1,parameters_sha256=sha(f.inventory),base_hyperparameters_sha256=sha(HYPERPARAMETERS),effective_learning_rate=5e-7,created_at=0,expires_at=2000,execution_release_sha256='4'*64,run_id=g['run_id'])
  self.authorization=f.sign(self.grant)
  parameters=[('weight',torch.nn.Parameter(torch.tensor([.02,.02],dtype=torch.bfloat16)))]
  workspace=f.root/'effective-lr';workspace.mkdir();admission=admit_resources(workspace,resource_plan(f.inventory,bf16_export_bytes=100,ram_reserve_bytes=0,disk_reserve_bytes=0))
  with patch('subnet.learning_rate_transition.time.time',return_value=1000):
   optimizer=PersistentCPUAdamW(parameters,f.cp['id'],approved_genesis=g,approved_genesis_sha256=sha(g),resource_admission=admission,learning_rate_authorization=self.authorization,learning_rate_authority=f.authority,epoch=self.manifest['epoch'],job_id=self.legacy_job['job_id'],steps=1)
  optimizer.step(gradients={'weight':torch.ones(2,dtype=torch.float32)})
  store=Store();descriptor,_=export_state(optimizer,epoch=self.manifest['epoch'],inference_checkpoint=f.cp['id'],workspace=workspace,publish_shard=store.put,readback_shard=store.read,commit_descriptor=store.commit,resource_admission=admission)
  self.job=copy.deepcopy(self.legacy_job);self.job['manifest']=f.sign(self.manifest)
  self.declaration=dict(version='unaudited-training-execution-amendment-v3-effective-lr-genesis',method='fp32-task-gradient-effective-lr-genesis-v1',job_id=self.job['job_id'],epoch=self.manifest['epoch'],original_signed_manifest_sha256=sha(self.job['manifest']),trainer_binding_sha256=sha(binding),execution_source_files=self.job['source_files'],genesis_sha256=sha(g),parent_descriptor_sha256=None,optimizer_step_before=0,steps=1,learning_rate_authorization=self.authorization,execution_release_sha256='4'*64,effective_learning_rate=5e-7)
  self.job['unaudited_training_execution']=f.sign(self.declaration)
  self.report=copy.deepcopy(self.legacy_report);self.report['persistent_training_state']['descriptor']=descriptor
  self.report['persistent_training_state']['descriptor_sha256']=sha(descriptor)
  update=self.report['training']['updates'][0];update['hyperparameters']=copy.deepcopy(descriptor['hyperparameters']);update['precision']=copy.deepcopy(optimizer.last_update)
  diagnostics=self.report['training']['persistent_diagnostics'];diagnostics['effective_hyperparameters']=copy.deepcopy(descriptor['hyperparameters']);diagnostics['learning_rate_authorization_sha256']=sha(self.authorization)
  self.original=evidence.validate_updates;self.addCleanup(setattr,evidence,'validate_updates',self.original)
  helper.install(f.authority)
 def test_valid_signed_effective_lr_accepts_original_attribution(self):
  with self.assertRaisesRegex(ValueError,'attribution'):self.original(self.report,self.job,self.manifest)
  result=evidence.validate_updates(self.report,self.job,self.manifest)
  self.assertEqual(result['distinct_tasks'],1);self.assertEqual(result['global_step_after'],1);self.assertEqual(HYPERPARAMETERS['lr'],1e-5)
 def test_rate_beta_decay_mutations_reject(self):
  for field,value in(('lr',1e-6),('betas',[.1,.999]),('weight_decay',.5)):
   with self.subTest(field=field):
    report=copy.deepcopy(self.report);report['training']['updates'][0]['hyperparameters'][field]=value
    with self.assertRaises(ValueError):evidence.validate_updates(report,self.job,self.manifest)
 def test_unsigned_outer_rejects(self):
  self.job['unaudited_training_execution'].pop('signature')
  with self.assertRaises(Exception):evidence.validate_updates(self.report,self.job,self.manifest)
 def test_signed_outer_unsigned_nested_rejects(self):
  self.declaration['learning_rate_authorization']=copy.deepcopy(self.authorization);self.declaration['learning_rate_authorization'].pop('signature');self.job['unaudited_training_execution']=self.f.sign(self.declaration)
  with self.assertRaises(Exception):evidence.validate_updates(self.report,self.job,self.manifest)
 def test_validly_signed_cross_job_and_release_reject(self):
  for field,value in(('job_id','other-job'),('execution_release_sha256','0'*64)):
   with self.subTest(field=field):
    grant=dict(self.grant,**{field:value});declaration=dict(self.declaration,learning_rate_authorization=self.f.sign(grant));job=dict(self.job,unaudited_training_execution=self.f.sign(declaration))
    with self.assertRaises(ValueError):evidence.validate_updates(self.report,job,self.manifest)
 def test_signed_changed_execution_source_rejects(self):
  declaration=copy.deepcopy(self.declaration);declaration['execution_source_files']['subnet/task_normalized_training.py']='0'*64;self.job['unaudited_training_execution']=self.f.sign(declaration)
  with self.assertRaises(ValueError):evidence.validate_updates(self.report,self.job,self.manifest)
 def test_legacy_still_uses_original_learning_rate(self):
  self.assertEqual(evidence.validate_updates(self.legacy_report,self.legacy_job,self.f.manifest)['global_step_after'],1)
  report=copy.deepcopy(self.legacy_report);report['training']['updates'][0]['hyperparameters']['lr']=5e-7
  with self.assertRaises(ValueError):evidence.validate_updates(report,self.legacy_job,self.f.manifest)
 def test_original_gradient_and_counter_checks_remain(self):
  for field,value in(('gradient_tasks',0),('global_optimizer_step',2)):
   with self.subTest(field=field):
    report=copy.deepcopy(self.report);report['training']['updates'][0][field]=value
    with self.assertRaises(ValueError):evidence.validate_updates(report,self.job,self.manifest)
 def test_diagnostics_cannot_disagree_with_authenticated_grant(self):
  self.report['training']['persistent_diagnostics']['effective_hyperparameters']['lr']=1e-5
  with self.assertRaisesRegex(ValueError,'diagnostics'):evidence.validate_updates(self.report,self.job,self.manifest)
 def test_output_descriptor_must_use_original_grant(self):
  d=self.report['persistent_training_state']['descriptor'];d['learning_rate_authorization']=self.f.sign(dict(self.grant,job_id='other'))
  with self.assertRaises(ValueError):evidence.validate_updates(self.report,self.job,self.manifest)

if __name__=='__main__':unittest.main()
