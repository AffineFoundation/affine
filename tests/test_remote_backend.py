import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from types import SimpleNamespace
from subnet.remote_backend import RemoteJobs,RemoteController
from subnet.backend_jobs import BACKEND_PROFILE,NUMERICAL_POLICY,canonical
import hashlib

class RemoteReportBinding(unittest.TestCase):
    def setUp(self):
        self.jobs=RemoteJobs.__new__(RemoteJobs);self.jobs.controller=SimpleNamespace(authority=SimpleNamespace(id='operator'))
        self.manifest=dict(epoch='nonpayable-test',checkpoint={'id':'approved'})
        self.prior=dict(job_id='same-job',role='verify',job_sha256='digest',source_files={'a':'b'},runtime_versions={'torch':'approved'},manifest_sha256=hashlib.sha256(canonical(self.manifest)).hexdigest())
        self.report=dict(job_id='same-job',role='verify',operator='operator',job_sha256='digest',checkpoint='approved',epoch='nonpayable-test',success=True,chain_transactions=False,backend_profile=BACKEND_PROFILE,numerical_policy=NUMERICAL_POLICY,source_files={'a':'b'},runtime_versions={'torch':'approved'})
    def test_approved_report_binding(self):self.assertIs(self.jobs.checked(self.report,self.prior,self.manifest),self.report)
    def test_changed_identity_sources_or_runtime_is_rejected(self):
        for name,value in [('job_id','another-job'),('source_files',{'a':'bad'}),('runtime_versions',{'torch':'unapproved'}),('chain_transactions',True),('backend_profile',{'device':'cpu'})]:
            report=copy.deepcopy(self.report);report[name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):self.jobs.checked(report,self.prior,self.manifest)
    def test_live_prior_job_is_polled_instead_of_launched_again(self):
        with tempfile.TemporaryDirectory() as d:
            self.jobs.state=Path(d);self.jobs.workspace='/remote';record=self.jobs.state/'label.json';record.write_text(json.dumps(self.prior))
            self.jobs.remote_status=Mock(side_effect=[{'phase':'running'},{'phase':'complete'}]);self.jobs.command=Mock();self.jobs.copy_to=Mock()
            def fetch(remote,local):local.write_text(json.dumps(self.report))
            self.jobs.copy_from=fetch
            with patch('subnet.remote_backend.time.sleep'):result=self.jobs.run('label','verify',self.manifest)
            self.assertTrue(result['success']);self.jobs.command.assert_not_called();self.jobs.copy_to.assert_not_called()
    def test_status_connection_failure_never_launches_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            self.jobs.state=Path(d);(self.jobs.state/'label.json').write_text(json.dumps(self.prior));self.jobs.remote_status=Mock(side_effect=TimeoutError('SSH status unreachable'));self.jobs.copy_to=Mock()
            with self.assertRaises(TimeoutError):self.jobs.run('label','verify',self.manifest)
            self.jobs.copy_to.assert_not_called()

class DurableRemoteLiveness(unittest.TestCase):
    def test_old_pid_reuse_is_not_treated_as_live_job(self):
        from subnet.remote_runner import probe
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'runner-status';folder.mkdir();(folder/'job.json').write_text(json.dumps(dict(phase='running',runner_pid=1,runner_pid_ticks='old',child_pid=2,child_pid_ticks='old')))
            with patch('subnet.remote_runner.ticks',return_value='new'):self.assertEqual(probe(d,'job')['phase'],'failed')
    def test_live_child_survives_dead_launcher(self):
        from subnet.remote_runner import probe
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'runner-status';folder.mkdir();(folder/'job.json').write_text(json.dumps(dict(phase='running',runner_pid=1,runner_pid_ticks='old',child_pid=2,child_pid_ticks='same')))
            with patch('subnet.remote_runner.ticks',side_effect=lambda pid:'same' if pid==2 else None):self.assertEqual(probe(d,'job')['phase'],'running')
    def test_completed_report_recovers_after_lost_launcher_marker(self):
        from subnet.remote_runner import probe
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'jobs/job';folder.mkdir(parents=True);(folder/'report.json').write_text('{}');self.assertEqual(probe(d,'job')['phase'],'complete')

if __name__=='__main__':unittest.main()

class InitialManifestPublication(unittest.TestCase):
    def test_quota_is_in_first_public_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            controller=RemoteController.__new__(RemoteController);controller.state=Path(d);bucket=SimpleNamespace(json=Mock());controller.bucket=bucket;controller.signed=lambda payload:dict(payload=payload)
            def base_open(instance,*args,**kwargs):
                manifest=dict(epoch='nonpayable-test',max_batches=4)
                instance.bucket.json('public/nonpayable-test/manifest.json',instance.signed(manifest));instance.bucket.json('public/nonpayable-test/current.json',{'payload':{'epoch':'nonpayable-test'}});return manifest
            with patch('subnet.controller.Controller.open',base_open):controller.open('nonpayable-test',{},[],max_batches=3)
            manifests=[c.args[1]['payload'] for c in bucket.json.call_args_list if c.args[0].endswith('/manifest.json')]
            self.assertEqual(manifests,[dict(epoch='nonpayable-test',max_batches=3)])
