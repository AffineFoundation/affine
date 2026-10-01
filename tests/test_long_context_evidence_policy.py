import copy
import unittest
from unittest.mock import patch

from ops.check_gpu_continuous_evidence import (
    LONG_CONTEXT_REVISION, LONG_CONTEXT_PROFILE, LONG_CONTEXT_NUMERICAL,
    unpack_authenticated_epoch,
)
from subnet.batches import unpack


class ArtifactPolicyTests(unittest.TestCase):
    def manifest(self):
        return dict(model_runtime_revision=LONG_CONTEXT_REVISION,
            backend_profile=copy.deepcopy(LONG_CONTEXT_PROFILE),
            numerical_policy=copy.deepcopy(LONG_CONTEXT_NUMERICAL),
            transport_policy='direct-r2-v1',
            artifact_policy=dict(compressed_bytes=250_000_000,raw_bytes=500_000_000))

    def test_historical_reader_budget_unchanged(self):
        with patch('ops.check_gpu_continuous_evidence.unpack',return_value=[]) as reader:
            unpack_authenticated_epoch(b'archive',{})
            reader.assert_called_once_with(b'archive')

    def test_exact_reviewed_profile_selects_bounded_larger_reader(self):
        with patch('ops.check_gpu_continuous_evidence.unpack',return_value=[]) as reader:
            unpack_authenticated_epoch(b'archive',self.manifest())
            reader.assert_called_once_with(b'archive',max_upload=250_000_000)

    def test_unknown_or_relaxed_policy_rejected_before_reader(self):
        cases=[]
        for field,value in [('model_runtime_revision','unknown'),('backend_profile',{}),
                ('numerical_policy',dict(LONG_CONTEXT_NUMERICAL,logprob_atol=.01)),
                ('transport_policy','unknown'),('artifact_policy',dict(compressed_bytes=500_000_000,raw_bytes=500_000_000))]:
            m=self.manifest();m[field]=value;cases.append(m)
        m=self.manifest();m['backend_profile']['tf32']=0;cases.append(m)
        for m in cases:
            with self.subTest(manifest=m),patch('ops.check_gpu_continuous_evidence.unpack') as reader:
                with self.assertRaises(ValueError):unpack_authenticated_epoch(b'archive',m)
                reader.assert_not_called()

    def test_reader_rejects_unbounded_or_boolean_limits_before_archive(self):
        for limit in [True,0,-1,250_000_001,1.0]:
            with self.subTest(limit=limit),self.assertRaisesRegex(ValueError,'budget bounds'):
                unpack(b'',max_upload=limit)


if __name__=='__main__':unittest.main()
