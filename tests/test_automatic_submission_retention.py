import hashlib
import io
from types import SimpleNamespace
import unittest
from ops.automatic_submission_retention import archived, run

class AutomaticRetention(unittest.TestCase):
    def fixture(self, data, expected=b'archive'):
        body=io.BytesIO(data)
        client=SimpleNamespace(get_object=lambda **kw:{'Body':body})
        bucket=SimpleNamespace(client=client,name='durable')
        plan={'archive_key':'public/frozen.zip','size':len(expected),'sha256':hashlib.sha256(expected).hexdigest()}
        return bucket,plan,body

    def test_retirement_requires_complete_matching_archive(self):
        bucket,plan,body=self.fixture(b'archive')
        verified=archived(bucket,plan)
        self.assertTrue(verified['archive_verified'])
        self.assertNotIn('archive_verified',plan)
        self.assertTrue(body.closed)

    def test_corrupt_truncated_or_oversized_archive_refuses_retirement(self):
        for payload in [b'corrupt',b'arch',b'archive-too-long']:
            bucket,plan,body=self.fixture(payload)
            with self.assertRaises(ValueError):archived(bucket,plan)
            self.assertTrue(body.closed)
            self.assertNotIn('archive_verified',plan)

    def test_unbounded_retention_rate_refuses_before_read_or_network(self):
        for limit in [0,33,True,1.5]:
            with self.assertRaises(ValueError):run('absent','absent','absent','absent',per_worker=limit)
