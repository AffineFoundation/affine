import threading
import unittest
from types import SimpleNamespace
from subnet.remote_backend import dispatch_verifications

class VerificationDispatch(unittest.TestCase):
    def test_three_admitted_workers_run_concurrently_and_results_keep_order(self):
        jobs=SimpleNamespace(queue=object(),verifiers=[1,2,3]);barrier=threading.Barrier(3);active=0;peak=0;lock=threading.Lock()
        def operation(value):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            barrier.wait(timeout=3)
            with lock:active-=1
            return value*2
        self.assertEqual(dispatch_verifications(jobs,[2,1,3],operation),[4,2,6]);self.assertEqual(peak,3)
    def test_single_host_preserves_serial_recovery_order(self):
        observed=[]
        self.assertEqual(dispatch_verifications(object(),[3,1],lambda x:observed.append(x)or x),[3,1]);self.assertEqual(observed,[3,1])
    def test_empty_work_dispatches_nothing(self):
        self.assertEqual(dispatch_verifications(SimpleNamespace(queue=object(),verifiers=[1,2,3]),[],lambda _:self.fail('unexpected job')),[])
    def test_failed_audit_does_not_yield_a_partial_finalization(self):
        def operation(x):
            if x==2:raise ValueError('audit not complete')
            return x
        with self.assertRaisesRegex(ValueError,'audit not complete'):
            dispatch_verifications(SimpleNamespace(queue=object(),verifiers=[1,2,3]),[1,2,3],operation)

if __name__=='__main__':unittest.main()
