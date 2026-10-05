import copy,unittest
from subnet import backend_profiles as p
from subnet.persistent_cpu_adamw import POLICY
class Controls(unittest.TestCase):
 def setUp(self):
  rev,profile,numeric=p.profile(p.HOPPER_FP32_REVISION)
  _,bf,np=p.profile(p.HOPPER_REVISION)
  self.m=dict(model_runtime_revision=rev,backend_profile=profile,numerical_policy=numeric,training_input_policy='committed-unaudited-training-v1',training_policy=POLICY,training_runtime=dict(version='separate-bf16-persistent-training-runtime-v1',model_runtime_revision=p.HOPPER_REVISION,backend_profile=bf,numerical_policy=np))
 def test_train_bf16_mine_and_verify_fp32_without_mutating_manifest(self):
  before=copy.deepcopy(self.m)
  self.assertEqual(p.execution_profile(self.m,'train')[1]['dtype'],'bfloat16')
  for role in ('mine','verify','evaluate'):self.assertEqual(p.execution_profile(self.m,role)[1]['dtype'],'float32')
  self.assertEqual(self.m,before)
 def test_public_runtime_factory_accepts_explicit_fp32_generation_profile(self):
  from subnet.runtime_factory import validate_backend
  self.assertEqual(validate_backend(self.m),p.HOPPER_FP32_REVISION)
 def test_historical_profile_no_override_unchanged(self):
  rev,profile,numeric=p.profile(p.HOPPER_REVISION)
  m=dict(model_runtime_revision=rev,backend_profile=profile,numerical_policy=numeric)
  self.assertEqual(p.execution_profile(m,'train'),p.resolve(m))
 def test_override_cannot_switch_dtype_math_or_old_policy(self):
  for key,value in [('training_input_policy','authenticated-verifier-compact-inputs-v2'),('training_policy','head-v1')]:
   m=copy.deepcopy(self.m);m[key]=value
   with self.assertRaisesRegex(ValueError,'separate'):p.execution_profile(m,'train')
  m=copy.deepcopy(self.m);m['training_runtime']['backend_profile']['dtype']='float32'
  with self.assertRaisesRegex(ValueError,'separate'):p.execution_profile(m,'train')
 def test_boolean_and_numeric_profile_types_are_exact(self):
  m=copy.deepcopy(self.m);m['training_runtime']['backend_profile']['deterministic_algorithms']=1
  with self.assertRaisesRegex(ValueError,'separate'):p.execution_profile(m,'train')
 def test_computation_binding_includes_signed_training_override(self):
  from subnet.training_receipts import computation_binding
  m=dict(self.m,checkpoint={'id':'a'*64,'files':{}})
  before=computation_binding(m);m['training_runtime']['version']='tampered'
  self.assertNotEqual(computation_binding(m),before)
 def test_actual_backend_signed_job_admission_rejects_runtime_mutation(self):
  from test_committed_training_inputs import LearnerAdmissionTests
  from subnet.backend_jobs import _validate
  from subnet.committed_training_inputs import coverage_manifest
  fx=LearnerAdmissionTests();fx.setUp();self.addCleanup(fx.doCleanups)
  m=dict(fx.manifest,**self.m);m=coverage_manifest(m,[fx.obj],seed='c'*64,captured_at=21)
  # Rebind the original token admission to this unchanged original checkpoint/source.
  from test_persistent_training_integration import PersistentIntegrationTests
  pf=PersistentIntegrationTests();pf.setUp();self.addCleanup(pf.doCleanups)
  # Runtime guard is evaluated before receipt details and source hashes.
  job=pf.job();job['manifest']=pf.sign(m)
  from subnet.backend_jobs import _validate
  with self.assertRaisesRegex(ValueError,'job-manifest|policy|binding|checkpoint|source|sampling'):_validate(pf.sign(job),pf.authority,now=30,resolve_source=False)
  m['training_runtime']['numerical_policy']['logprob_atol']=1
  job['manifest']=pf.sign(m)
  with self.assertRaisesRegex(ValueError,'separate'):_validate(pf.sign(job),pf.authority,now=30,resolve_source=False)
