import io
import unittest
import numpy as np
from subnet.artifact_budget import for_manifest,LEGACY,LONG,LONG_REVISION
from subnet.backend_profiles import profile,HOPPER_REVISION,HOPPER_FP32_REVISION
from subnet.batches import pack,unpack,bounded_tensor
from subnet.harness import normalize

class ArtifactBudgets(unittest.TestCase):
    def manifest(self):
        revision,backend,numerical=profile(HOPPER_REVISION)
        return dict(model_runtime_revision=revision,backend_profile=backend,
            numerical_policy=numerical,artifact_policy=LONG_REVISION)
    def test_long_budget_requires_approved_hardware_policy(self):
        self.assertEqual(for_manifest({}),LEGACY)
        self.assertEqual(for_manifest(self.manifest()),LONG)
        for update in ({'model_runtime_revision':'cuda-bf16-eager-sm86-v1'},
                       {'artifact_policy':'unbounded'}, {'backend_profile':{}}):
            with self.assertRaises(ValueError):for_manifest(dict(self.manifest(),**update))
    def test_large_tensor_needs_operator_budget_and_full_framing(self):
        values=np.arange(513*2,dtype=np.float32).reshape(513,2)
        data=pack([({'index':7},[[values]])],budget=LONG)
        with self.assertRaisesRegex(ValueError,'tensor header'):unpack(data)
        result=unpack(data,budget=LONG)
        np.testing.assert_array_equal(result[0][1][0][0],values)
        raw=io.BytesIO();np.save(raw,values,allow_pickle=False)
        with self.assertRaisesRegex(ValueError,'payload length'):bounded_tensor(raw.getvalue()[:-1],max_rows=2048)
        with self.assertRaises(ValueError):bounded_tensor(raw.getvalue(),max_rows=4096)
        with self.assertRaises(ValueError):unpack(data,budget=dict(LONG,tensor_rows=4096))
    def test_generation_extension_has_its_own_harness_version(self):
        with self.assertRaisesRegex(ValueError,'token budget'):normalize({'max_output_tokens':1024})
        self.assertEqual(normalize({'version':'text-tools-long-v2','max_output_tokens':2048})['max_output_tokens'],2048)
        with self.assertRaisesRegex(ValueError,'token budget'):normalize({'version':'text-tools-long-v2','max_output_tokens':2049})

    def test_fp32_hopper_long_budget_retains_all_caps_and_exact_profile(self):
        revision,backend,numerical=profile(HOPPER_FP32_REVISION)
        manifest=dict(model_runtime_revision=revision,backend_profile=backend,numerical_policy=numerical,artifact_policy=LONG_REVISION)
        self.assertEqual(for_manifest(manifest),LONG)
        for update in ({'backend_profile':dict(backend,dtype='bfloat16')},{'numerical_policy':{}},{'model_runtime_revision':'cuda-fp32-eager-sm86-v1'}):
            with self.subTest(update=update),self.assertRaises(ValueError):for_manifest(dict(manifest,**update))
        self.assertEqual(for_manifest(dict(manifest,artifact_policy=None)),LEGACY)
