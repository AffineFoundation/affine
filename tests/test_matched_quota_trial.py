import copy,unittest
from ops.matched_quota_trial import task_plan,select_matched,validate_stream,verify_generated

def stream(index=2):
 return [dict(index=index,attempt=i,status='verified',native_verified=True,sampler_verified=True,rollout=dict(index=index,seed=i,classification='positive'if i%3 else'negative',reward=float(bool(i%3)),task_hash='a'*64,turns=[dict(prompt=[1],output=[i+2])]))for i in range(16)]
class TrialTests(unittest.TestCase):
 def test_common_tasks_and_first_zip_not_cartesian(self):
  a,s=select_matched({2:stream()},dict(env_id='math'));self.assertEqual([len(a[x])for x in ('1P1N','2P2N')],[1,2]);self.assertEqual(a['1P1N'][0],a['2P2N'][0]);self.assertEqual(s[0]['first2_prefix8'],4)
 def test_censored_task_excluded_from_both(self):
  x=stream();
  for r in x[1:]:r.update(status='native_error',native_verified=False,sampler_verified=False)
  a,s=select_matched({2:x},{});self.assertEqual(a,{'1P1N':[],'2P2N':[]});self.assertFalse(s[0]['matched_included'])
 def test_unknown_is_not_positive(self):
  x=stream();x[1].update(status='numerical_unknown',sampler_verified=False);a,s=select_matched({2:x},{});self.assertEqual(a['1P1N'][0][1]['seed'],2);self.assertEqual(s[0]['unknown'],1)
 def test_all16_or_fail(self):
  for x in (stream()[:8],stream()[::-1]):
   with self.assertRaises(ValueError):validate_stream(x,2)
 def test_claims_and_bool_alias_not_verification(self):
  for k,v in [('native_verified',1),('sampler_verified',False)]:
   x=stream();x[0][k]=v
   with self.assertRaises(ValueError):validate_stream(x,2)
 def test_duplicate_seed_reward_and_task_reject(self):
  for mutation in ('seed','duplicate','reward','task'):
   x=stream()
   if mutation=='seed':x[0]['rollout']['seed']=3
   if mutation=='duplicate':x[1]['rollout']['turns']=copy.deepcopy(x[0]['rollout']['turns'])
   if mutation=='reward':x[0]['rollout']['reward']=1
   if mutation=='task':x[0]['index']=4
   with self.assertRaises(ValueError):validate_stream(x,2)
 def test_reserved_cohort_excluded_and_deterministic(self):
  a=task_plan(list(range(40)),list(range(20)),'a'*64);self.assertTrue(set(a).isdisjoint(range(20)));self.assertEqual(a,task_plan(list(reversed(range(40))),list(range(20)),'a'*64))
 def test_honest_repeated_draws_preserved_but_do_not_fill_second_quota(self):
  x=stream()
  for row in x:
   first=x[1] if row['rollout']['classification']=='positive' else x[0]
   row['rollout']['turns']=copy.deepcopy(first['rollout']['turns'])
  validate_stream(x,2)
  arms,supply=select_matched({2:x},{})
  self.assertEqual(arms,{'1P1N':[],'2P2N':[]})
  self.assertEqual(supply[0]['verified_attempts'],16)
  self.assertEqual(supply[0]['duplicate_verified_attempts'],14)
  self.assertEqual((supply[0]['positive'],supply[0]['negative']),(1,1))
  self.assertEqual(len(x),16)
 def test_duplicate_uses_first_verified_draw_and_later_distinct_pair(self):
  x=stream();x[2]['rollout']['turns']=copy.deepcopy(x[1]['rollout']['turns'])
  arms,supply=select_matched({2:x},{})
  self.assertEqual([pair[1]['seed'] for pair in arms['2P2N']],[1,4])
  self.assertEqual(supply[0]['first2_prefix8'],5)
  self.assertEqual(supply[0]['duplicate_verified_attempts'],1)
 def test_official_false_does_not_become_native_pass(self):
  class R:
   def rollout(self,i,a):return stream(i)[a]['rollout'],[]
   def verify(self,r,a):return False
  with self.assertRaises(ValueError):verify_generated(R(),2,0)

class ObjectiveTests(unittest.TestCase):
 def test_real_task_mean_grouping_keeps_task_weights_equal(self):
  from subnet.task_normalized_training import task_groups,accumulate_tasks
  import torch
  ss={i:stream(i)for i in (2,3)}
  for rows in ss.values():
   for x in rows:x['rollout']['env_id']='math'
  arms,_=select_matched(ss,dict(env_id='math'))
  for name in arms:
   pairs,tasks,groups,identities=task_groups(arms[name],1,'b'*64)
   self.assertEqual(len(tasks),2);self.assertEqual(len(pairs),2 if name=='1P1N' else 4)
   margins=[torch.tensor(.2,requires_grad=True)for _ in pairs]
   observations=accumulate_tasks(torch,lambda i:margins[i],[.2]*len(pairs),tasks,groups[0])
   for task in tasks:self.assertAlmostEqual(sum(x['gradient_weight']for x in observations if x['task_sha256']==task['task_sha256']),.5)
 def test_task_hash_switch_fails(self):
  x=stream();x[2]['rollout']['task_hash']='b'*64
  with self.assertRaises(ValueError):validate_stream(x,2)

class RetentionTests(unittest.TestCase):
 def test_real_lifecycle_retention_requires_full_ACK_and_preserves_foreign_inode(self):
  import tempfile,pathlib,json,hashlib
  from nacl.signing import SigningKey
  from subnet.backend_jobs import canonical
  from subnet.cache_lifecycle import CacheLifecycle
  from ops.matched_quota_trial import digest
  from ops.matched_quota_retention import retire_artifacts
  import base64
  key=SigningKey(bytes(32));auth=key.verify_key.encode().hex()
  def sign(v):return dict(payload=v,signer=auth,signature=base64.b64encode(key.sign(canonical(v)).signature).decode())
  with tempfile.TemporaryDirectory()as tmp:
   root=pathlib.Path(tmp);s=dict(version='matched-quota-research-v1',production_state_changes=False,workspace=str(root),phases=['generate'],task_indices=[2]);identity=digest(s);directory=root/'jobs'/identity;directory.mkdir(parents=True);p=directory/'submission-0.zip';p.write_bytes(b'original');c=CacheLifecycle(root);c.record_download(p,hashlib.sha256(p.read_bytes()).hexdigest());report=dict(scope_sha256=identity,streams={'2':[dict(index=2,attempt=0,artifact_path=str(p.relative_to(root)))]});raw=canonical(report);(root/'generation-result.json').write_bytes(raw);a=dict(version='matched-quota-full-evidence-durable-ack-v1',scope_sha256=identity,R2_full_GET_verified=True,completed_phases=['generate'],generation_result_sha256=hashlib.sha256(raw).hexdigest())
   with self.assertRaises(ValueError):retire_artifacts(sign(s),sign(dict(a,R2_full_GET_verified=False)),auth,live_phase=lambda p:False)
   with self.assertRaises(ValueError):retire_artifacts(sign(s),sign(a),auth,live_phase=lambda p:True)
   foreign=directory/'replacement';foreign.write_bytes(b'original');foreign.replace(p);self.assertEqual(retire_artifacts(sign(s),sign(a),auth,live_phase=lambda p:False),[]);self.assertTrue(p.exists())

class PolicyVerdictTests(unittest.TestCase):
 def test_native_replay_infrastructure_error_never_becomes_negative_or_fraud(self):
  from verifiers.v1.errors import TaskError
  arrays=[object()]
  class R:
   def rollout(self,i,a):return stream(i)[a]['rollout'],arrays
   def verify(self,r,a):raise TaskError('grader unavailable')
  row,retained=verify_generated(R(),2,0)
  self.assertEqual(row['status'],'native_error');self.assertIs(retained,arrays)
  self.assertFalse(row['native_verified']);self.assertFalse(row['sampler_verified'])
  x=stream();x[0]=row;arms,supply=select_matched({2:x},{})
  self.assertEqual(supply[0]['native_errors'],1);self.assertEqual(supply[0]['confirmed_invalid'],0)
  self.assertEqual(arms['1P1N'][0][2]['seed'],3)
 def test_confirmed_invalid_stream_kept_as_evidence_not_negative(self):
  from subnet.audit_policy import InvalidSample
  class R:
   def rollout(self,i,a):return stream(i)[a]['rollout'],[]
   def verify(self,r,a):raise InvalidSample('outside calibrated prescribed CDF')
  row,_=verify_generated(R(),2,0);self.assertEqual(row['status'],'confirmed_invalid');self.assertFalse(row['sampler_verified']);self.assertFalse(row['native_verified'])
  x=stream();x[0]=row;a,s=select_matched({2:x},{});self.assertEqual(s[0]['confirmed_invalid'],1);self.assertEqual(a['1P1N'][0][2]['seed'],3)


class ScopeTests(unittest.TestCase):
 def setUp(self):
  import tempfile,pathlib,hashlib
  from nacl.signing import SigningKey
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  root=pathlib.Path(self.tmp.name);model=root/'model';model.mkdir();(model/'config.json').write_bytes(b'{}')
  source=pathlib.Path(__file__).resolve().parent.parent
  self.key=SigningKey(bytes(32));self.authority=self.key.verify_key.encode().hex()
  version='forced-inverse-cdf-prefill-threeway-v4'
  self.scope=dict(version='matched-quota-research-v1',production_state_changes=False,execute_allowed=True,
   created_at=100,expires_at=200,stage='pilot32',phases=['generate'],training_steps=1,attempts=16,K=2,L=2,
   restore_concurrency=4,output_limit_bytes=1024,task_count=8,generation_task_limit=2,
   mining_indices=list(range(20)),heldout128_indices=[0],old32_indices=[1],selection_seed='a'*64,
   checkpoint='b'*64,source_sha256='c'*64,sampling_version=version,
   heldout_cohort_sha256='20a077180d7cc088669f53bd551c5bf3c4aa51ded6367463f405094ad054b153',
   source_path=str(source),source_files={'ops/run_matched_quota_trial.py':hashlib.sha256((source/'ops/run_matched_quota_trial.py').read_bytes()).hexdigest()},
   workspace=str(root/'owned'),checkpoint_path=str(model))
  self.scope['task_indices']=task_plan(list(range(20)),[0,1],'a'*64,8)
  self.scope['generation_manifest']=dict(checkpoint=dict(id='b'*64,files={'config.json':hashlib.sha256(b'{}').hexdigest()}),source_bundle=dict(sha256='c'*64),
   probability_artifact_policy={'version':'selected-token-logprobs-v1'},sampling_contract=dict(version=version,max_attempts=16),K=2,L=2)
 def check(self,s=None,phase='generate',now=150):
  import base64
  from subnet.storage import canonical
  from ops.run_matched_quota_trial import validate_scope
  s=self.scope if s is None else s
  envelope=dict(payload=s,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(s)).signature).decode())
  return validate_scope(envelope,self.authority,phase,now)
 def test_real_signed_pilot_source_and_model_inventory_cpu_preflight(self):
  from unittest.mock import patch
  with patch('subnet.gpu_runtime.GPURuntime',side_effect=AssertionError('GPU construction forbidden')):
   for lanes in (4,8):
    for version in ('forced-inverse-cdf-prefill-support-v3','forced-inverse-cdf-prefill-threeway-v4'):
     s=copy.deepcopy(self.scope);s['restore_concurrency']=lanes;s['sampling_version']=version;s['generation_manifest']['sampling_contract']['version']=version
     self.assertEqual(self.check(s)['generation_task_limit'],2)
 def test_pilot_refuses_training_even_if_added_to_capability(self):
  from unittest.mock import patch
  with patch('subnet.model.model_files',side_effect=AssertionError('reject before model inventory')):
   for phase in ('1P1N','2P2N'):
    for phases in (['generate'],['generate','1P1N','2P2N']):
     with self.subTest(phase=phase,phases=phases):
      s=copy.deepcopy(self.scope);s['phases']=phases
      with self.assertRaises(ValueError):self.check(s,phase)
 def test_malformed_stage_limits_and_expired_capability_rejected(self):
  for field,value in [('stage',None),('stage','pilot'),('generation_task_limit',True),('generation_task_limit',0),('generation_task_limit',3),('generation_task_limit',9),('task_count',True),('task_count',7),('restore_concurrency',16)]:
   with self.subTest(field=field,value=value):
    s=copy.deepcopy(self.scope);s[field]=value
    with self.assertRaises(ValueError):self.check(s)
  for now in (99,200):
   with self.assertRaises(ValueError):self.check(now=now)
 def test_unknown_or_mismatched_sampling_policy_rejected(self):
  for version in ('unknown',None):
   s=copy.deepcopy(self.scope);s['sampling_version']=version
   with self.assertRaises(ValueError):self.check(s)
  s=copy.deepcopy(self.scope);s['generation_manifest']['sampling_contract']['version']='forced-inverse-cdf-prefill-support-v3'
  with self.assertRaises(ValueError):self.check(s)
