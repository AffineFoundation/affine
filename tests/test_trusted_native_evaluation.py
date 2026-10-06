"""Genuine tiny cached/uncached generation and TOPLOC baseline, no GPU claim."""
import copy,unittest
from unittest.mock import patch
import test_forced_sampling as base
import test_fast_prefill_audit as cached
from subnet.trusted_native_evaluation import POLICY,rollout,evaluate,digest,validate_policy
class TrustedNativeControls(unittest.TestCase):
 def test_identical_generation_grade_to_real_full_proof_baseline(self):
  for version in ('uncached','cached-support-v3'):
   if version=='uncached':case=base.ForcedSamplingTests();case.setUp();runtime=case.runtime()
   else:runtime,_=cached.Controls().support_runtime()
   with self.subTest(version=version),patch('subnet.model.create_session',side_effect=lambda spec:base.Session()):
    doc,arrays=runtime.rollout(2,0);self.assertTrue(runtime.verify(doc,arrays))
   with patch.object(runtime,'compute',side_effect=AssertionError('full vocabulary allocation')),patch.object(runtime,'build_proofs',side_effect=AssertionError('TOPLOC build')),patch.object(runtime,'verify_proofs',side_effect=AssertionError('TOPLOC verify')):
    row=rollout(runtime,2,0,create_session=lambda spec:base.Session())
   self.assertEqual(row['turns'][0]['output_sha256'],digest(doc['turns'][0]['output']))
   self.assertEqual((row['reward'],row['classification'],row['task_hash']),(doc['reward'],doc['classification'],doc['task_hash']))
   self.assertFalse(row['verified']);self.assertTrue(row['native_graded']);self.assertFalse(row['proof_verification_performed'])
 def runtime(self):
  case=base.ForcedSamplingTests();case.setUp();return case.runtime()
 def test_error_is_infrastructure_not_wrong_answer(self):
  runtime=self.runtime();runtime.for_environment=lambda spec,harness:runtime
  job={'trusted_evaluation_policy':POLICY,'heldout':[dict(env_id='tiny',indices=[2],seeds=[0],harness=runtime.harness)]}
  class Broken(base.Session):
   def step(self,a):raise RuntimeError('native grader unavailable')
  with patch('subnet.protocol.entry',return_value={'indices':[1],'spec':{}}):values,errors,meta=evaluate(runtime,{},job,create_session=lambda spec:Broken())
  self.assertEqual(values,[]);self.assertEqual(len(errors),1);self.assertTrue(errors[0]['infrastructure_failure']);self.assertNotIn('reward',errors[0]);self.assertFalse(meta['miner_reward_evidence'])
 def test_heldout_overlap_refused(self):
  runtime=self.runtime();job={'trusted_evaluation_policy':POLICY,'heldout':[dict(env_id='tiny',indices=[2],seeds=[0],harness=runtime.harness)]}
  with patch('subnet.protocol.entry',return_value={'indices':[2],'spec':{}}),self.assertRaisesRegex(ValueError,'overlap'):evaluate(runtime,{},job,create_session=lambda spec:base.Session())
 def test_signed_caps_preserved_and_session_closed(self):
  runtime=self.runtime();runtime.model.config.max_position_embeddings=4
  class Count(base.Session):
   def close(self):self.closed=True
  session=Count()
  with self.assertRaisesRegex(ValueError,'context'):rollout(runtime,2,0,create_session=lambda spec:session)
  self.assertTrue(session.closed)
 def test_invalid_policy_refused(self):
  for bad in [None,{},dict(POLICY,sampling_policy='torch-new-seed'),dict(POLICY,proof_reverification=0),dict(POLICY,version='unknown')]:
   with self.subTest(bad=bad),self.assertRaises(ValueError):validate_policy(bad)
class SignedTrustedControls(unittest.TestCase):
 def setUp(self):
  from test_owned_cached_evaluation import SignedOwnedJobControls
  self.fixture=SignedOwnedJobControls();self.fixture.setUp();self.job=copy.deepcopy(self.fixture.job);self.job.pop('owned_evaluation_policy');self.job['trusted_evaluation_policy']=POLICY;self.job['heldout'][0]['harness']['version']='text-tools-long-v2'
 def test_explicit_signed_only_and_legacy_default(self):
  from subnet.backend_jobs import validate
  validate(self.fixture.sign(self.job),self.fixture.authority,now=50)
  self.job.pop('trusted_evaluation_policy');self.assertNotIn('trusted_evaluation_policy',validate(self.fixture.sign(self.job),self.fixture.authority,now=50)[0])
 def test_wrong_role_missing_source_and_mixed_policy(self):
  from subnet.backend_jobs import validate
  for name in ('role','source','mixed','changed-sampler','null'):
   job=copy.deepcopy(self.job)
   if name=='role':job['role']='mine'
   if name=='source':job['source_files'].pop('subnet/trusted_native_evaluation.py')
   if name=='mixed':job['owned_evaluation_policy']=__import__('subnet.owned_cached_evaluation',fromlist=['POLICY']).POLICY
   if name=='changed-sampler':job['trusted_evaluation_policy']['sampling_policy']='different'
   if name=='null':job['trusted_evaluation_policy']=None
   with self.subTest(name=name),self.assertRaises((ValueError,KeyError)):validate(self.fixture.sign(job),self.fixture.authority,now=50)
if __name__=='__main__':unittest.main()

class TrustedRoutingControls(unittest.TestCase):
 def test_policy_separates_queue_identity_and_preserves_legacy(self):
  from test_gpu_service import GPUFixedHeldout
  from subnet.checkpoint_evaluator import fingerprint
  case=GPUFixedHeldout();case.setUp();plan=[dict(env_id='env',indices=[2,3],seeds=[2100,3100],harness=case.row['harness'])]
  legacy=fingerprint(case.manifest,case.config,plan)
  self.assertEqual(legacy,fingerprint(case.manifest,dict(case.config),plan))
  self.assertNotEqual(legacy,fingerprint(case.manifest,dict(case.config,trusted_evaluation_policy=POLICY),plan))
 def test_report_assurance_and_infrastructure_neutrality(self):
  import tempfile
  from types import SimpleNamespace
  from unittest.mock import Mock
  from test_gpu_service import GPUFixedHeldout
  from subnet.gpu_service import evaluate
  fixture=GPUFixedHeldout();fixture.setUp();report=fixture.report()
  report['trusted_native_evaluation']={'policy':POLICY};report['source_files']['subnet/trusted_native_evaluation.py']='a'*64
  for row in report['heldout']:row.update(verified=False,native_graded=True,proof_verification_performed=False,trust_scope=POLICY['trust_scope'],sampling_policy=POLICY['sampling_policy'])
  with tempfile.TemporaryDirectory()as directory,patch('subnet.gpu_service.definitions',return_value=[fixture.row]):
   config=dict(fixture.config,evaluation_state=directory,trusted_evaluation_policy=POLICY);controller=SimpleNamespace(jobs=SimpleNamespace(run=Mock(return_value=report)))
   result=evaluate(controller,fixture.manifest,'cache','before',0,config)[0]
   self.assertFalse(result['verified']);self.assertFalse(result['proof_verification_performed']);self.assertEqual(controller.jobs.run.call_args.kwargs['trusted_evaluation_policy'],POLICY)
   report['heldout_failures']=[dict(env_id='env',index=3,seed=3100,error_type='RuntimeError',infrastructure_failure=True)];report['heldout']=report['heldout'][:1]
   result=evaluate(controller,fixture.manifest,'cache','before',0,config)[0];self.assertEqual(result['status'],'error');self.assertIsNone(result['mean_reward']);self.assertIsNone(result['uncertainty'])
   report['heldout'][0]['verified']=True
   with self.assertRaisesRegex(ValueError,'completeness'):evaluate(controller,fixture.manifest,'cache','before',0,config)
