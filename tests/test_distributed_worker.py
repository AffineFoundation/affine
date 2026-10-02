import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace
from nacl.signing import SigningKey
from subnet.distributed_worker import Worker
from subnet.storage import canonical
from subnet.distributed_roles import digest


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        self.worker=Worker('http://127.0.0.1:19081',bytes(SigningKey.generate()),self.authority,self.folder.name)
        self.job={'job_id':'verify-job','role':'verify','manifest':{'payload':{'checkpoint':{'id':'CP'}}}}
        envelope={'signer':self.authority,'payload':self.job,'signature':base64.b64encode(self.key.sign(canonical(self.job)).signature).decode()}
        self.claim={'job':envelope,'job_sha256':digest(self.job),'token':'lease-secret','lease_until':time.time()+300,'attempt':1}

    def test_worker_rejects_plain_remote_http(self):
        with self.assertRaises(ValueError):Worker('http://remote:19080',bytes(SigningKey.generate()),self.authority,self.folder.name)

    def test_fresh_subprocess_and_exact_report_submission(self):
        report={'job_id':'verify-job','chain_transactions':False}
        self.worker.request=Mock(side_effect=[{'claim':self.claim},{'accepted':True}])
        def run(args,**kwargs):
            self.assertEqual(kwargs['env']['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
            workspace=Path(args[args.index('--workspace')+1]);out=workspace/'jobs'/'verify-job'
            out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(report))
            return SimpleNamespace(returncode=0)
        with patch('subnet.distributed_worker.subprocess.run',side_effect=run):self.assertTrue(self.worker.once())
        self.worker.request.assert_called_with('report',job_id='verify-job',token='lease-secret',report=report)
        self.assertTrue((Path(self.folder.name)/'verify-job'/'attempt-1'/'pending-report.json').is_file())

    def test_failed_job_reports_failure_not_fabricated_metrics(self):
        self.worker.request=Mock(side_effect=[{'claim':self.claim},{'status':'queued'}])
        with patch('subnet.distributed_worker.subprocess.run',return_value=SimpleNamespace(returncode=1)):
            self.assertTrue(self.worker.once())
        self.worker.request.assert_called_with('fail',job_id='verify-job',token='lease-secret')

    def test_retry_keeps_prior_workspace_and_uses_checked_shared_cache(self):
        cache=Path(self.folder.name)/'backend'/'checkpoints'/'CP';cache.mkdir(parents=True)
        prior=Path(self.folder.name)/'verify-job'/'attempt-1';prior.mkdir(parents=True);(prior/'worker.log').write_text('old')
        self.claim['attempt']=2;self.worker.request=Mock(side_effect=[{'claim':self.claim},{'status':'failed'}])
        with patch('subnet.distributed_worker.subprocess.run',return_value=SimpleNamespace(returncode=1)) as run:
            self.worker.once()
        self.assertEqual((prior/'worker.log').read_text(),'old')
        self.assertIn('--checkpoint-cache',run.call_args.args[0]);self.assertIn(str(cache),run.call_args.args[0])

if __name__=='__main__':unittest.main()
