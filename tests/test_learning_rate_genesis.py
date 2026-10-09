import copy
import unittest
from unittest.mock import patch
import torch
import test_learning_rate_transition as transition
from subnet.persistent_cpu_adamw import PersistentCPUAdamW,HYPERPARAMETERS,sha
from subnet.learning_rate_transition import (GENESIS_AUTH_VERSION,GENESIS_DOCUMENT_VERSION,
    STATE_VERSION,genesis_document,validate_authorization)


class LRGenesisControls(transition.LRTransitionTests):
 def document(self,rate=1e-6,run_id='99'*32):
  return genesis_document(sha(self.inventory),'11'*32,run_id,rate)
 def initial_grant(self,document=None):
  document=document or self.document()
  return self.sign(dict(version=GENESIS_AUTH_VERSION,epoch='fresh90',job_id='freshjob90',
    input_checkpoint='11'*32,parent_descriptor_sha256=None,genesis_sha256=sha(document),
    optimizer_step_before=0,steps=1,parameters_sha256=sha(self.inventory),
    base_hyperparameters_sha256=sha(HYPERPARAMETERS),effective_learning_rate=document['initial_effective_learning_rate'],
    created_at=0,expires_at=2000,execution_release_sha256='44'*32,run_id=document['run_id']))
 def initial(self,document=None,grant=None,**kw):
  document=document or self.document()
  parameters=[('weight',torch.nn.Parameter(torch.tensor([.5,.25,-.75],dtype=torch.bfloat16)))]
  with patch('subnet.learning_rate_transition.time.time',return_value=1000):
   return PersistentCPUAdamW(parameters,'11'*32,approved_genesis=document,
     approved_genesis_sha256=sha(document),resource_admission=self.admission,
     learning_rate_authorization=grant or self.initial_grant(document),learning_rate_authority=self.authority,
     epoch=kw.pop('epoch','fresh90'),job_id=kw.pop('job_id','freshjob90'),steps=kw.pop('steps',1),**kw)
 def test_distinct_runs_have_distinct_genesis(self):
  self.assertNotEqual(sha(self.document()),sha(self.document(run_id='88'*32)))
  self.assertNotEqual(sha(self.document()),self.parent['genesis_sha256'])
 def test_first_update_zero_moments_exact_torch_all_rates(self):
  for rate in (1e-5,1e-6,5e-7):
   with self.subTest(rate=rate):
    optimizer=self.initial(self.document(rate));row=optimizer.rows['weight']
    self.assertEqual(optimizer.global_step,0);self.assertIsNone(optimizer.parent_state_sha256)
    self.assertTrue(torch.equal(row['exp_avg'],torch.zeros(3)));self.assertTrue(torch.equal(row['exp_avg_sq'],torch.zeros(3)))
    reference=torch.nn.Parameter(row['master'].clone());adam=torch.optim.AdamW([reference],lr=rate,
       betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False)
    gradient=torch.tensor([.1,-.25,.75]);reference.grad=gradient.clone();adam.step();optimizer.step(gradients={'weight':gradient})
    torch.testing.assert_close(row['master'],reference.detach(),rtol=0,atol=0)
    for slot in ('exp_avg','exp_avg_sq'):torch.testing.assert_close(row[slot],adam.state[reference][slot],rtol=0,atol=0)
 def test_initial_v2_descriptor_then_existing_parent_continuation(self):
  optimizer=self.initial();optimizer.step(gradients={'weight':torch.ones(3)})
  d,store=self.export(optimizer,'fresh90','33'*32)
  self.assertEqual(d['version'],STATE_VERSION);self.assertEqual(d['optimizer_steps'],1)
  self.assertEqual(d['hyperparameters']['lr'],1e-6);self.assertIsNone(d['parent_state_sha256'])
  next_optimizer=self.continuation(self.grant(d,epoch='epoch91',job='job91'),d,store,epoch='epoch91',job_id='job91')
  self.assertEqual(next_optimizer.global_step,1);self.assertEqual(next_optimizer.genesis_sha256,sha(self.document()))
  for slot in ('master','exp_avg','exp_avg_sq'):torch.testing.assert_close(next_optimizer.rows['weight'][slot],optimizer.rows['weight'][slot],rtol=0,atol=0)
  next_optimizer.step(gradients={'weight':torch.ones(3)});next_d,_=self.export(next_optimizer,'epoch91','55'*32)
  self.assertEqual(next_d['optimizer_steps'],2);self.assertEqual(next_d['genesis_sha256'],d['genesis_sha256'])
 def test_genesis_grant_cannot_reset_existing_parent(self):
  with self.assertRaises(ValueError):self.continuation(self.initial_grant())
 def test_existing_parent_grant_cannot_initialize_new_genesis(self):
  with self.assertRaises(ValueError):self.initial(grant=self.grant())
 def test_genesis_context_mutants_before_allocation(self):
  for field,new in [('run_id','88'*32),('effective_learning_rate',5e-7),('optimizer_step_before',1),
      ('optimizer_step_before',False),('parent_descriptor_sha256','77'*32),('parameters_sha256','55'*32),
      ('genesis_sha256','55'*32),('job_id','other'),('epoch','other'),('input_checkpoint','55'*32),
      ('expires_at',999)]:
   value=self.initial_grant()['payload'];value[field]=new
   with (self.subTest(field=field,new=new),
       patch('torch.zeros_like',side_effect=AssertionError('allocation before auth')),
       self.assertRaises(ValueError)):
    self.initial(grant=self.sign(value))
 def test_absent_initial_lr_grant_fails_before_allocation(self):
  document=self.document()
  with patch('torch.zeros_like',side_effect=AssertionError('allocation before auth')),self.assertRaises(ValueError):
   PersistentCPUAdamW(self.parameters,'11'*32,approved_genesis=document,approved_genesis_sha256=sha(document),resource_admission=self.admission)
 def test_zero_counter_binding_is_explicit_and_round_scoped(self):
  from subnet.persistent_training_protocol import opening_binding,validate_binding
  document=self.document();admission=dict(parameters=self.inventory,parameters_sha256=sha(self.inventory),source_sha256='74'*32,
      gpu_qualification_sha256='66'*32,genesis_round=91,genesis_checkpoint='11'*32,genesis_sha256=sha(document),genesis_document=document)
  config=dict(persistent_training_admission=admission,source_bundle={'sha256':'74'*32})
  status=dict(round=91,checkpoint={'id':'11'*32},trainer_state=None)
  binding=opening_binding(config,status,'fresh91');self.assertEqual(binding['global_step_before'],0)
  self.assertEqual(binding['genesis'],document)
  for key,value in [('round',92),('persistent_state_committed',True),('checkpoint',{'id':'22'*32})]:
   bad=copy.deepcopy(status);bad[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):opening_binding(config,bad,'fresh91')
 def test_new_genesis_output_requires_distinct_genesis_outer_declaration(self):
  from subnet.persistent_training_protocol import validate_output
  optimizer=self.initial();optimizer.step(gradients={'weight':torch.ones(3)});d,_=self.export(optimizer,'fresh90','33'*32)
  binding=dict(parameters=self.inventory,parameters_sha256=sha(self.inventory),input_checkpoint='11'*32,
      parent=None,genesis_sha256=sha(self.document()),global_step_before=0)
  manifest=dict(epoch='fresh90',trainer_state_binding=binding)
  value=dict(version='unaudited-training-execution-amendment-v3-effective-lr-genesis',
      method='fp32-task-gradient-effective-lr-genesis-v1',execution_release_sha256='44'*32,
      learning_rate_authorization=optimizer.learning_rate_authorization)
  job=dict(job_id='freshjob90',steps=1,manifest=self.sign(manifest),persistent_training=dict(global_step_after=1,
      output_shards={x['name']:{}for x in d['shards']}),unaudited_training_execution=self.sign(value))
  validate_output(d,job,manifest)
  value['version']='unaudited-training-execution-amendment-v2-effective-lr';job['unaudited_training_execution']=self.sign(value)
  with self.assertRaises(ValueError):validate_output(d,job,manifest)

if __name__=='__main__':unittest.main()
