import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import requests
from subnet.persistent_training_worker import put_file

class PersistentPutRetry(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'state.bin';self.path.write_bytes(b'original immutable state')
        self.seen=[];self.closed=[]
    def response(self,status):
        from types import SimpleNamespace
        return SimpleNamespace(status_code=status,close=lambda:self.closed.append(status))
    def run_attempts(self,items):
        items=iter(items)
        def put(url,*,data,**kwargs):
            self.assertFalse(kwargs['allow_redirects']);self.seen.append((data.fileno(),data.read()))
            item=next(items)
            if isinstance(item,Exception):raise item
            return self.response(item)
        with patch('subnet.backend_jobs.r2_url',return_value='https://private.invalid/signed-secret'),patch('requests.put',side_effect=put),patch('subnet.persistent_training_worker.time.sleep'):
            return put_file('private grant',self.path)
    def test_transient_status_rewinds_same_descriptor_exact_bytes(self):
        self.run_attempts([503,429,204]);self.assertEqual(len(set(fd for fd,_ in self.seen)),1)
        self.assertEqual([raw for _,raw in self.seen],[self.path.read_bytes()]*3)
        self.assertEqual(self.closed,[503,429,204])
    def test_timeout_after_consumption_rewinds(self):
        self.run_attempts([requests.Timeout('contains private grant'),200]);self.assertEqual(self.seen[0][1],self.seen[1][1])
    def test_authorization_or_permanent_failures_never_retry_or_expose_grant(self):
        for status in (400,403,404,409,413,301):
            self.seen=[]
            with self.subTest(status=status),self.assertRaisesRegex(ValueError,'HTTP '+str(status)+' after 1')as error:self.run_attempts([status,200])
            self.assertEqual(len(self.seen),1);self.assertNotIn('private',str(error.exception))
    def test_retry_exhaustion_is_bounded(self):
        with self.assertRaisesRegex(ValueError,'HTTP 503 after 4'):self.run_attempts([503]*4)
        self.assertEqual(len(self.seen),4)
    def test_transport_exhaustion_does_not_echo_secret_exception(self):
        with self.assertRaisesRegex(ValueError,'transport exhausted')as error:self.run_attempts([requests.ConnectionError('secret')]*4)
        self.assertNotIn('secret',str(error.exception));self.assertEqual(len(self.seen),4)
    def test_symlink_refused_before_network(self):
        other=Path(self.tmp.name)/'other';other.write_bytes(b'x');self.path.unlink();self.path.symlink_to(other)
        with self.assertRaises(OSError),patch('subnet.backend_jobs.r2_url',return_value='https://private.invalid'),patch('requests.put')as put:put_file('grant',self.path)
        put.assert_not_called()
    def test_changed_file_stops_even_when_server_reports_success(self):
        def put(url,*,data,**kwargs):
            data.read();self.path.write_bytes(b'changed');return self.response(200)
        with patch('subnet.backend_jobs.r2_url',return_value='https://private.invalid'),patch('requests.put',side_effect=put),self.assertRaisesRegex(ValueError,'input changed'):put_file('grant',self.path)
        self.assertEqual(self.closed,[200])
