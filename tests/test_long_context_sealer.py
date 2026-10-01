import unittest
from pathlib import Path
from ops.seal_long_context_eog import validate_fetched
from subnet.long_context_runtime import POLICY,digest
class SealerTests(unittest.TestCase):
    def test_report_cannot_authorize_wrong_job_role(self):
        with self.assertRaisesRegex(ValueError,'model role'):validate_fetched({'role':'weight-setter'},{},Path('/unused'),b'')
    def test_empty_or_partial_verification_records_rejected(self):
        job={'role':'long-context-proof-probe','experiment':'native-public-eog-v1','policy':POLICY}
        for records in ([],[{'full_proof_verified':True}]*5):
            report={'job_hash':digest(job),'completed':True,'full_model_recompute':True,'records':records}
            with self.assertRaisesRegex(ValueError,'complete fresh'):validate_fetched(job,report,Path('/unused'),b'')
