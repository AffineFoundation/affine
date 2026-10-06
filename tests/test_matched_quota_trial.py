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
