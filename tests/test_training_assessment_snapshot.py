import copy,unittest
from ops.training_assessment_snapshot import reusable

class AssessmentCache(unittest.TestCase):
    def setUp(self):
        self.a=dict(version='hourly-current-miner-assessment-v1',cutoff=3600,evidence_cutoff=3600,assessment_stale=False,writer_policy_sha256='a'*64,miner_estimates={},evidence_hashes={'source_admission_sha256':'b'*64})
    def test_exact_original_hour_and_policy_reused_without_timestamp_change(self):
        before=copy.deepcopy(self.a)
        self.assertTrue(reusable(self.a,3600,'a'*64,'b'*64));self.assertEqual(self.a,before)
    def test_later_hour_changed_pins_stale_or_different_evidence_not_reused(self):
        for hour,writer,source in [(7200,'a'*64,'b'*64),(3600,'c'*64,'b'*64),(3600,'a'*64,'c'*64)]:
            self.assertFalse(reusable(self.a,hour,writer,source))
        for field,value in [('assessment_stale',True),('evidence_cutoff',0),('miner_estimates',None)]:
            a=copy.deepcopy(self.a);a[field]=value
            self.assertFalse(reusable(a,3600,'a'*64,'b'*64))
