"""A late MiMo report preserves evidence and does not stop queue polling."""
import time
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from nacl.signing import SigningKey
import requests
import test_mimo_worker as fixture
from subnet.distributed_roles import Coordinator
from subnet.distributed_worker import Worker, ExpiredCompletedLease

class MiMoExpiredReportTests(unittest.TestCase):
    def test_report_transport_expiry_retains_original_result_and_lease_identity(self):
        f=fixture.WorkerIntegrationTests();f.setUp();self.addCleanup(f.doCleanups)
        key=SigningKey.generate();kid=key.verify_key.encode().hex()
        queue=Coordinator(f.root/'expiry.db',f.authority,{kid:['verify']})
        queue.enqueue(f.envelope());counter=iter(range(100));claims=[];reports=[]
        f.factory=SimpleNamespace(close=lambda:None)
        worker=Worker('http://localhost:99',key.encode(),f.authority,f.root/'expiry-worker',mimo_factory=f.factory)
        self.addCleanup(worker.close)
        def request(action,**kw):
            if action=='report':
                reports.append(kw['report']);claims[0]['lease_until']=0
                raise requests.ConnectionError('original report ACK unavailable past lease')
            value=queue.request(fixture.sign(key,dict(action=action,at=time.time(),nonce=str(next(counter)).zfill(32),**kw)))
            if action=='claim':claims.append(value['claim'])
            return value
        worker.request=request
        with patch('subnet.backend_jobs.get_object',side_effect=f.fetch),patch('subnet.complete_group_verification.verify',side_effect=f.verify)as verify:
            with self.assertRaises(ExpiredCompletedLease):worker.once()
        self.assertEqual(len(reports),1);self.assertEqual(verify.call_count,1)
        attempt=worker.workspace/f.job['job_id']/'attempt-1'
        self.assertTrue((attempt/'pending-report.json').is_file())
        self.assertTrue((attempt/'expired-completed-lease.json').is_file())
        self.assertEqual(queue.status(f.job['job_id'])['status'],'leased')

if __name__=='__main__':unittest.main()
