import unittest
from subnet.runtime_factory import validate_backend,CPU_REVISION,GPU_REVISION,GPU_PROFILE,GPU_POLICY

class BackendTest(unittest.TestCase):
 def test_legacy_cpu_profile_remains_strict_and_gpu_cannot_downgrade(self):
  self.assertEqual(validate_backend({}),CPU_REVISION)
  with self.assertRaises(ValueError):validate_backend({'backend_profile':GPU_PROFILE})
  with self.assertRaises(ValueError):validate_backend({'model_runtime_revision':'unknown'})
 def test_gpu_exact_policy_rejects_loosened_tolerance_and_different_device(self):
  m=dict(model_runtime_revision=GPU_REVISION,backend_profile=GPU_PROFILE,numerical_policy=GPU_POLICY)
  self.assertEqual(validate_backend(m),GPU_REVISION)
  with self.assertRaises(ValueError):validate_backend(dict(m,numerical_policy=dict(GPU_POLICY,logprob_atol=.7)))
  with self.assertRaises(ValueError):validate_backend(dict(m,backend_profile=dict(GPU_PROFILE,tf32=True)))
  with self.assertRaises(ValueError):validate_backend(dict(m,backend_profile=dict(GPU_PROFILE,sm=[8,9])))
