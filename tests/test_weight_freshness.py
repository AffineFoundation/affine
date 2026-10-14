import base64
import json
import unittest
from nacl.signing import SigningKey
from ops.check_weight_freshness import inspect_freshness


class FreshnessTests(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()

    def signed(self, cutoff):
        p = dict(cutoff=cutoff, evidence_cutoff=cutoff, assessment_stale=False)
        raw = json.dumps(p, sort_keys=True, separators=(',', ':')).encode()
        return dict(payload=p, signer=self.authority,
                    signature=base64.b64encode(self.key.sign(raw).signature).decode())

    def check(self, evidence=36000, submitted=36000, now=39000):
        return inspect_freshness(self.signed(evidence),
            {'latest': {'window_end': submitted, 'status': 'submitted'}}, self.authority, now=now)

    def test_current_receipt_does_not_hide_stale_evidence(self):
        self.assertEqual(self.check(evidence=0)['reasons'], ['assessment-evidence-stale'])

    def test_healthy(self):
        self.assertEqual(self.check()['status'], 'healthy')

    def test_missing_submission_remains_visible(self):
        self.assertEqual(self.check(submitted=32400)['status'], 'attention')

    def test_hourly_grace_then_overdue(self):
        self.assertEqual(self.check(now=40000)['status'], 'healthy')
        self.assertEqual(self.check(now=40900)['status'], 'attention')

    def test_future_assessment_rejected(self):
        with self.assertRaises(ValueError):
            self.check(evidence=43200)

    def test_signature_tampering_rejected(self):
        doc=self.signed(36000);doc['payload']['cutoff']=32400
        with self.assertRaises(Exception):
            inspect_freshness(doc,{},self.authority,now=39000)


if __name__ == '__main__':
    unittest.main()
