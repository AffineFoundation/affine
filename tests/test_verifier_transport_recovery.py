import unittest
from unittest.mock import Mock
import requests
from subnet.distributed_worker import serve, ExpiredCompletedLease

class Stopped(Exception): pass

class VerifierTransportRecovery(unittest.TestCase):
    def test_connection_outage_reuses_worker_and_can_claim_after_recovery(self):
        worker=Mock();worker.once.side_effect=[requests.ConnectionError('forward absent'),requests.Timeout('temporary outage'),True,False]
        waits=[]
        def pause(seconds):
            waits.append(seconds)
            if len(waits)==3:raise Stopped()
        with self.assertRaises(Stopped):serve(worker,pause)
        self.assertEqual(waits,[5,5,5]);self.assertEqual(worker.once.call_count,4)

    def test_signature_or_lease_integrity_failure_is_not_a_transport_retry(self):
        worker=Mock();worker.once.side_effect=ValueError('wrong authority');pause=Mock()
        with self.assertRaisesRegex(ValueError,'wrong authority'):serve(worker,pause)
        worker.once.assert_called_once();pause.assert_not_called()

    def test_terminal_backend_with_expired_lease_returns_to_polling(self):
        worker=Mock();worker.once.side_effect=[ExpiredCompletedLease('retained'),True,False]
        waits=[]
        def pause(seconds):
            waits.append(seconds)
            if len(waits)==2:raise Stopped()
        with self.assertRaises(Stopped):serve(worker,pause)
        self.assertEqual(waits,[5,5]);self.assertEqual(worker.once.call_count,3)

if __name__=='__main__':unittest.main()
