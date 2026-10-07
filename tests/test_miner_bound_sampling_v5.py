import copy,unittest
from unittest.mock import patch
import torch
import test_fast_prefill_audit as fixture
from subnet import forced_sampling as f,fast_prefill_audit as fast
from subnet.sampling_uniqueness import validate_batch,content_digest
from subnet.controller import class_quotas
from subnet.audit_policy import InvalidSample

class Controls(unittest.TestCase):
 def setUp(self):
  self.runtime,self.manifest=fixture.Controls().support_runtime()
  m=self.manifest;m.update(K=2,L=2,max_batches=3)
  c=m['sampling_contract'];c.update(version=f.MINER_VERSION,max_attempts=1000)
  self.miner='d'*64;f.bind_runtime(self.runtime,m,self.miner);self.context=self.runtime.sampling_context
  self.batch=dict(epoch=m['epoch'],checkpoint=m['checkpoint']['id'],env_id='math',index=4,sample_index=4,environment_version='v1',rollouts=[dict(env_id='math',index=4,sample_index=4,environment_version='v1',seed=i,sampling=f.receipt(self.context,i),task_hash='c'*64,classification='positive'if i<2 else 'negative',turns=[dict(prompt=[1,2],output=[i+10,6])])for i in range(4)])
 def test_honest_four_distinct_attempts(self):validate_batch(self.batch,self.manifest,self.miner)
 def test_eight_distinct_attempts_follow_manifest_and_reject_padding(self):
  m=copy.deepcopy(self.manifest);m.update(K=4,L=4)
  c=f.binding(m,self.miner);b=copy.deepcopy(self.batch)
  b['rollouts']=[]
  for i in range(8):
   r=copy.deepcopy(self.batch['rollouts'][0]);r.update(seed=i,sampling=f.receipt(c,i),classification='positive'if i<4 else 'negative');r['turns'][0]['output']=[20+i,6];b['rollouts'].append(r)
  validate_batch(b,m,self.miner)
  for mutate in (lambda x:x['rollouts'].pop(),lambda x:x['rollouts'][7]['turns'][0].__setitem__('output',[20,6]),lambda x:x['rollouts'][7].update(seed=0,sampling=f.receipt(c,0)),lambda x:x['rollouts'][7].__setitem__('classification','positive')):
   bad=copy.deepcopy(b);mutate(bad)
   with self.assertRaises(ValueError):validate_batch(bad,m,self.miner)
  self.assertEqual(class_quotas(4,4,m['sampling_contract']),(4,4))
 def test_exact_nonce_boundary(self):
  for i in (0,999):self.assertEqual(f.receipt(self.context,i)['attempt'],i)
  for i in (-1,1000,True,1.0):
   with self.subTest(i=i),self.assertRaises(ValueError):f.receipt(self.context,i)
 def test_contract_exact_thousand(self):
  for i in (128,999,1001,True):
   c=copy.deepcopy(self.manifest['sampling_contract']);c['max_attempts']=i
   with self.assertRaises(ValueError):f.validate(c)
 def test_legacy_budget_cannot_increase(self):
  c=copy.deepcopy(self.manifest['sampling_contract']);c['version']=fast.SUPPORT_VERSION
  with self.assertRaises(ValueError):f.validate(c)
 def test_runtime_requires_authenticated_miner(self):
  with self.assertRaises(ValueError):f.bind_runtime(self.runtime,self.manifest)
 def test_each_binding_changes_draws(self):
  args=['math','c'*64,4,0,0,0];base=f.uniform(self.context,*args)
  for field in ('miner','epoch','checkpoint'):
   c=copy.deepcopy(self.context);c[field]='e'*64
   self.assertNotEqual(base,f.uniform(c,*args))
  for pos in (2,3,4,5):
   a=args[:];a[pos]+=1;self.assertNotEqual(base,f.uniform(self.context,*a))
 def test_copied_output_new_metadata_and_prompt_rejected(self):
  b=copy.deepcopy(self.batch);b['rollouts'][1]['turns']=[dict(prompt=[987],output=[10,6])]
  b['rollouts'][1]['filename']='different';b['rollouts'][1]['timestamp']=100
  with self.assertRaisesRegex(ValueError,'duplicate generated'):validate_batch(b,self.manifest,self.miner)
 def test_unique_tokens_reused_attempt_rejected(self):
  b=copy.deepcopy(self.batch);b['rollouts'][1]['seed']=0;b['rollouts'][1]['sampling']=f.receipt(self.context,0)
  with self.assertRaisesRegex(ValueError,'reused sampling'):validate_batch(b,self.manifest,self.miner)
 def test_other_miner_receipt_rejected(self):
  with self.assertRaisesRegex(ValueError,'receipt'):validate_batch(self.batch,self.manifest,'e'*64)
 def test_old_epoch_receipt_rejected(self):
  m=copy.deepcopy(self.manifest);m['epoch']='next'
  with self.assertRaises(ValueError):validate_batch(self.batch,m,self.miner)
 def test_quotas_and_slots_strict(self):
  for field,value in [('K',1),('L',1),('max_batches',4),('max_batches',True)]:
   m=copy.deepcopy(self.manifest);m[field]=value
   with self.assertRaises(ValueError):f.binding(m)
  self.assertEqual(class_quotas(2,2,self.manifest['sampling_contract']),(2,2))
  with self.assertRaises(ValueError):class_quotas(1,1,self.manifest['sampling_contract'])
 def test_same_task_geometry_cannot_be_relabelled(self):
  for field,value in [('index',5),('sample_index',5),('index',True),('env_id','other'),('environment_version','other'),('task_hash','x'*64)]:
   b=copy.deepcopy(self.batch);b['rollouts'][0][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):validate_batch(b,self.manifest,self.miner)
 def test_two_per_class_required(self):
  b=copy.deepcopy(self.batch);b['rollouts'][1]['classification']='negative'
  with self.assertRaisesRegex(ValueError,'manifest positive'):validate_batch(b,self.manifest,self.miner)
 def test_old_v3_batch_admission_unchanged(self):
  m=copy.deepcopy(self.manifest);m['sampling_contract']['version']=fast.SUPPORT_VERSION
  validate_batch({},m,None)
 def test_accepted_report_rechecks_content_and_attempt_quota(self):
  report=dict(sampling_miner=self.miner,sampling_assurance=f.assurance(self.manifest),accepted=[self.batch])
  f.require_report(self.manifest,report)
  duplicate=copy.deepcopy(report);duplicate['accepted'][0]['rollouts'][1]['turns'][0]['output']=[10,6]
  with self.assertRaisesRegex(ValueError,'duplicate generated'):f.require_report(self.manifest,duplicate)
  missing=copy.deepcopy(report);missing.pop('sampling_miner')
  with self.assertRaises(ValueError):f.require_report(self.manifest,missing)
 def test_v5_preserves_actual_cached_support_adjudication(self):
  runtime=self.runtime;prompt=[0,1];output=fast.cached_sample(runtime,prompt,0,0,2,'c'*64)
  roll=dict(seed=0,index=2,task_hash='c'*64)
  with patch.object(fast,'verify_intervals',side_effect=fast.NumericalAmbiguity('boundary')):
   result=fast.verify_sampling(runtime,roll,0,prompt,output,torch.zeros((len(output),7)))
  self.assertTrue(result['cached_reference_adjudication'])
  altered=output[:];altered[0]=(altered[0]+1)%6
  with patch.object(fast,'verify_intervals',side_effect=fast.NumericalAmbiguity('boundary')):
   with self.assertRaises(InvalidSample):fast.verify_sampling(runtime,roll,0,prompt,altered,torch.zeros((len(output),7)))
 def test_content_projection_ignores_metadata(self):
  r=copy.deepcopy(self.batch['rollouts'][0]);before=content_digest(r);r.update(seed=999,sampling={},classification='negative');r['turns'][0]['prompt']=[987]
  self.assertEqual(before,content_digest(r));r['turns'][0]['output'][0]+=1;self.assertNotEqual(before,content_digest(r))

class CheapAdmissionControls(unittest.TestCase):
 def setUp(self):
  from test_committed_training_inputs import LearnerAdmissionTests
  self.case=LearnerAdmissionTests();self.case.setUp();self.addCleanup(self.case.doCleanups)
  _,v5=fixture.Controls().support_runtime()
  m=self.case.manifest;m.update(K=2,L=2,max_batches=3,sampling_source_hash=f.source_hash(),sampling_contract=copy.deepcopy(v5['sampling_contract']))
  m['sampling_contract'].update(version=f.MINER_VERSION,max_attempts=1000)
  c=f.binding(m,self.case.identity)
  old=self.case.batch['rollouts'];rolls=[]
  for i in range(4):
   roll=copy.deepcopy(old[0 if i<2 else 1]);roll.update(seed=i,sampling=f.receipt(c,i),sample_index=self.case.batch['sample_index'],env_id=self.case.batch['env_id'],index=self.case.batch['index'],environment_version=self.case.batch['environment_version']);roll['turns'][0]['output']=[50+i,60];rolls.append(roll)
  self.case.batch['rollouts']=rolls;self.case.build()
 def test_four_unaudited_rows_no_model_or_grader(self):
  with patch('subnet.batches.unpack',side_effect=AssertionError('proof')),patch('subnet.model.Runtime.compute',side_effect=AssertionError('model')):
   summary,pairs=self.case.admit()
  self.assertEqual(len(pairs),2);self.assertEqual(summary['assurance'],'unaudited')
 def test_eight_unaudited_rows_form_four_disjoint_pairs(self):
  m=self.case.manifest;m.update(K=4,L=4);c=f.binding(m,self.case.identity)
  old=copy.deepcopy(self.case.batch['rollouts']);rolls=[]
  for i in range(8):
   r=copy.deepcopy(old[0 if i<4 else 2]);r.update(seed=i,sampling=f.receipt(c,i));r['turns'][0]['output']=[70+i,80];rolls.append(r)
  self.case.batch['rollouts']=rolls;self.case.build()
  with patch('subnet.batches.unpack',side_effect=AssertionError('proof')),patch('subnet.model.Runtime.compute',side_effect=AssertionError('model')):
   summary,pairs=self.case.admit()
  self.assertEqual(len(pairs),4);self.assertEqual(summary['assurance'],'unaudited')
 def test_resigned_duplicate_output_different_prompt_rejected(self):
  b=self.case.batch;b['rollouts'][1]['turns'][0]['output']=b['rollouts'][0]['turns'][0]['output'][:];b['rollouts'][1]['turns'][0]['prompt']=[888];self.case.build()
  with self.assertRaisesRegex(ValueError,'duplicate generated'):self.case.admit()
 def test_resigned_receipt_wrong_miner_rejected(self):
  self.case.batch['rollouts'][0]['sampling']=f.receipt(f.binding(self.case.manifest,'e'*64),0);self.case.build()
  with self.assertRaisesRegex(ValueError,'receipt'):self.case.admit()

class ExecuteTrainingControls(unittest.TestCase):
 setUp=CheapAdmissionControls.setUp
 def test_actual_execute_unaudited_trainer_does_not_bind_one_miner(self):
  from subnet import backend_jobs as backend
  from types import SimpleNamespace
  from contextlib import ExitStack
  from pathlib import Path
  import hashlib
  case=self.case;m=case.manifest;m['training_input_policy']='committed-unaudited-training-v1';m['training_coverage']={'seed':'f'*64};case.build()
  # The real signed document has learner_admission, never commitment_miner.
  self.assertNotIn('commitment_miner',case.obj)
  job=dict(role='train',job_id='v5-mixed-train',runtime_versions={},source_files={},training_policy=backend.COVERED_POLICY,steps=1,submissions=[case.obj])
  model=torch.nn.Linear(1,1,bias=False);runtime=SimpleNamespace(model=model,spec=SimpleNamespace(id='math'))
  def train(runtime,pairs,out,**kwargs):
   self.assertEqual(len(pairs),2);self.assertIsNone(runtime.sampling_context)
   optimizer=torch.optim.AdamW(model.parameters(),lr=.01);optimizer.zero_grad();model(torch.ones(1,1)).square().sum().backward();optimizer.step()
   target=out/'newmodel';target.mkdir();body=model.weight.detach().numpy().tobytes();(target/'model.safetensors').write_bytes(body);(target/'config.json').write_text('{}')
   return target,[dict(loss=1.,step=1)]
  class Prefetch:
   def __init__(self):self.rows=iter([(0,case.obj,case.root/'jobs'/'v5-mixed-train'/'submission-0.json')])
   def __iter__(self):return self
   def __next__(self):return next(self.rows)
   def close(self):pass
  def prefetch(*args):
   (case.root/'jobs'/'v5-mixed-train'/'submission-0.json').write_bytes(case.data)
   return Prefetch()
  with ExitStack()as st:
   st.enter_context(patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}))
   for target,value in [('subnet.backend_jobs._validate',(job,m)),('subnet.backend_jobs.version','approved'),('subnet.backend_jobs.checkpoint',case.root)]:st.enter_context(patch(target,return_value=value))
   st.enter_context(patch('subnet.backend_jobs.prefetched_training_submissions',side_effect=prefetch))
   st.enter_context(patch('subnet.backend_jobs.install_source_loader'))
   st.enter_context(patch('subnet.backend_profiles.execution_profile',return_value=('test',{},{})))
   st.enter_context(patch('subnet.task_assets.hydrate_manifest'))
   st.enter_context(patch('subnet.covered_epoch_optimizer.train_epoch',side_effect=train))
   st.enter_context(patch('subnet.committed_training_inputs.validate_native_prompt'))
   st.enter_context(patch('subnet.model.model_files',side_effect=lambda p:{name:hashlib.sha256((Path(p)/name).read_bytes()).hexdigest()for name in ('config.json','model.safetensors')}))
   st.enter_context(patch('subnet.forced_sampling.bind_runtime',side_effect=AssertionError('trainer must not bind generation draws')))
   report=backend.execute({},case.authority,case.root,runtime_factory=lambda *args:runtime)
  self.assertTrue(report['training']['weights_changed']);self.assertFalse(report['training']['trainer_verification_performed']);self.assertEqual(report['training']['input_assurance'],'unaudited')

class OwnedMiningControls(unittest.TestCase):
 setUp=Controls.setUp
 def test_cumulative_miner_collects_eight_from_manifest(self):
  from subnet.backend_jobs import mine_cumulative
  from types import SimpleNamespace
  m=copy.deepcopy(self.manifest);m.update(K=4,L=4,start=10,deadline=90)
  context=f.binding(m,self.miner)
  definition=dict(env_id='math',spec={},harness={},indices=[4])
  runtime=SimpleNamespace(spec=SimpleNamespace(version='v1'));runtime.for_environment=lambda *args:runtime
  def rollout(index,seed):
   return dict(schema=2,index=index,sample_index=index,env_id='math',environment_version='v1',task_hash='c'*64,seed=seed,sampling=f.receipt(context,seed),classification='positive'if seed<4 else 'negative',turns=[dict(prompt=[100],output=[seed+10,6])]),[]
  runtime.rollout=rollout;seen=[]
  with patch('subnet.protocol.entries',return_value=[definition]),patch('subnet.batches.pack',side_effect=lambda rows:seen.append(rows)or b'packed'):
   _,report=mine_cumulative(runtime,m,dict(miner_id=self.miner,seed_start=0,search_budget=8),lambda *args:None,clock=lambda:20)
  self.assertEqual(report['batches'],1)
  self.assertEqual(len(seen[0][0][0]['rollouts']),8)
 def test_cumulative_miner_skips_output_duplicates_with_changed_prompts(self):
  from subnet.backend_jobs import mine_cumulative
  from types import SimpleNamespace
  m=copy.deepcopy(self.manifest);m.update(start=10,deadline=90)
  definition=dict(env_id='math',spec={},harness={},indices=[4])
  runtime=SimpleNamespace(spec=SimpleNamespace(version='v1'));runtime.for_environment=lambda *args:runtime
  outputs=[10,10,11,12,12,13]
  def rollout(index,seed):
   row=dict(schema=2,index=index,sample_index=index,env_id='math',environment_version='v1',task_hash='c'*64,seed=seed,sampling=f.receipt(self.context,seed),classification='positive'if seed<3 else 'negative',turns=[dict(prompt=[100+seed],output=[outputs[seed],6])])
   return row,[]
  runtime.rollout=rollout;seen=[]
  with patch('subnet.protocol.entries',return_value=[definition]),patch('subnet.batches.pack',side_effect=lambda rows:seen.append(rows)or b'packed'):
   data,report=mine_cumulative(runtime,m,dict(miner_id=self.miner,seed_start=0,search_budget=6),lambda *args:None,clock=lambda:20)
  self.assertEqual(report['batches'],1)
  self.assertEqual([r['seed']for r in seen[0][0][0]['rollouts']],[0,2,3,5])
