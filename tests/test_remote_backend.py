import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from types import SimpleNamespace
from subnet.remote_backend import RemoteJobs,RemoteController,role_time_budget
from nacl.signing import SigningKey
import base64
from subnet.backend_jobs import BACKEND_PROFILE,NUMERICAL_POLICY,canonical,signed
import hashlib

class RemoteReportBinding(unittest.TestCase):
    def setUp(self):
        self.key=SigningKey.generate();self.operator=self.key.verify_key.encode().hex()
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.jobs=RemoteJobs.__new__(RemoteJobs);self.jobs.controller=SimpleNamespace(authority=SimpleNamespace(id=self.operator));self.jobs.state=Path(self.folder.name)
        self.manifest=dict(epoch='nonpayable-test',checkpoint={'id':'approved'})
        job=dict(job_id='same-job',role='verify',created_at=10,expires_at=30)
        self.envelope=dict(payload=job,signer=self.operator,signature=base64.b64encode(self.key.sign(canonical(job)).signature).decode())
        digest=hashlib.sha256(canonical(job)).hexdigest()
        self.prior=dict(job_id='same-job',role='verify',job_sha256=digest,source_files={'a':'b'},runtime_versions={'torch':'approved'},manifest_sha256=hashlib.sha256(canonical(self.manifest)).hexdigest())
        self.report=dict(job_id='same-job',role='verify',operator=self.operator,job_sha256=digest,completed_at=20,checkpoint='approved',epoch='nonpayable-test',success=True,chain_transactions=False,backend_profile=BACKEND_PROFILE,numerical_policy=NUMERICAL_POLICY,source_files={'a':'b'},runtime_versions={'torch':'approved'})
        (self.jobs.state/'same-job-job.json').write_text(json.dumps(self.envelope))
    def test_approved_report_binding(self):self.assertIs(self.jobs.checked(self.report,self.prior,self.manifest),self.report)
    def test_changed_identity_sources_or_runtime_is_rejected(self):
        for name,value in [('job_id','another-job'),('source_files',{'a':'bad'}),('runtime_versions',{'torch':'unapproved'}),('chain_transactions',True),('backend_profile',{'device':'cpu'})]:
            report=copy.deepcopy(self.report);report[name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):self.jobs.checked(report,self.prior,self.manifest)
    def test_live_prior_job_is_polled_instead_of_launched_again(self):
        with tempfile.TemporaryDirectory() as d:
            self.jobs.state=Path(d);self.jobs.workspace='/remote';record=self.jobs.state/'label.json';record.write_text(json.dumps(self.prior))
            (self.jobs.state/'same-job-job.json').write_text(json.dumps(self.envelope))
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
    def test_late_prestart_and_invalid_completion_times_rejected(self):
        for completed in (9,30,31,True,None,float('nan'),float('inf')):
            report=dict(self.report,completed_at=completed)
            with self.subTest(completed=completed),self.assertRaises(ValueError):
                self.jobs.checked(report,self.prior,self.manifest)
    def test_new_signature_cannot_extend_original_job_lifetime(self):
        payload=dict(self.envelope['payload'],expires_at=100)
        changed=dict(payload=payload,signer=self.operator,signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())
        (self.jobs.state/'same-job-job.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'original signed job binding'):
            self.jobs.checked(self.report,self.prior,self.manifest)
    def test_new_evaluation_job_signs_configured_lifetime(self):
        self.jobs.config={'job_ttl_seconds_by_role':{'evaluate':7200}}
        self.jobs.metadata=dict(source_files=self.prior['source_files'],runtime_versions=self.prior['runtime_versions'])
        self.jobs.workspace='/remote';self.jobs.code='/frozen';self.jobs.python='/python'
        self.jobs.controller.signed=lambda payload:dict(payload=payload,signer=self.operator,signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())
        self.jobs.command=Mock();self.jobs.copy_to=Mock();self.jobs.remote_status=Mock(return_value={'phase':'complete'})
        def fetch(remote,local):
            path=next(self.jobs.state.glob('fresh-evaluation-*-job.json'))
            job=signed(json.loads(path.read_text()),self.operator)
            self.assertEqual(job['expires_at']-job['created_at'],7200)
            report=dict(self.report,job_id=job['job_id'],role='evaluate',job_sha256=hashlib.sha256(canonical(job)).hexdigest(),completed_at=1100)
            local.write_text(json.dumps(report))
        self.jobs.copy_from=fetch
        with patch('subnet.remote_backend.time.time',return_value=1000):
            result=self.jobs.run('fresh-evaluation','evaluate',self.manifest)
        self.assertEqual(result['completed_at'],1100)
        self.jobs.copy_to.assert_called_once()

class RoleTimeBudget(unittest.TestCase):
    def test_expanded_evaluation_does_not_extend_other_roles(self):
        config={'job_ttl_seconds_by_role':{'evaluate':10800}}
        self.assertEqual(role_time_budget(config,'evaluate'),10800)
        self.assertEqual(role_time_budget(config,'mine'),3600)
        self.assertEqual(role_time_budget({},'evaluate'),3600)
    def test_unbounded_unknown_and_noninteger_budgets_rejected(self):
        for budgets in ({'evaluate':86401},{'evaluate':59},{'evaluate':True},{'evaluate':7200.0},{'unknown':3600},3600):
            with self.subTest(budgets=budgets),self.assertRaises(ValueError):
                role_time_budget({'job_ttl_seconds_by_role':budgets},'evaluate')

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
    def test_heldout_registry_is_in_first_public_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            controller=RemoteController.__new__(RemoteController);controller.state=Path(d)
            bucket=SimpleNamespace(json=Mock());controller.bucket=bucket
            controller.signed=lambda payload:dict(payload=payload)
            rows=[dict(spec=dict(id='e',num_samples=4),indices=[0],harness=None)]
            def base_open(instance,*args,**kwargs):
                self.assertEqual(kwargs['environments'],rows)
                manifest=dict(epoch='nonpayable-test',max_batches=4)
                instance.bucket.json('public/nonpayable-test/manifest.json',instance.signed(manifest))
                return manifest
            with patch('subnet.controller.Controller.open',base_open):
                controller.open('nonpayable-test',{},[],environments=rows,heldout_indices={'e':[1]})
            self.assertEqual(bucket.json.call_args.args[1]['payload']['heldout_indices'],{'e':[1]})
            bucket.json.reset_mock()
            with self.assertRaises(ValueError):
                controller.open('nonpayable-test',{},[],environments=rows,heldout_indices={'e':[0]})
            bucket.json.assert_not_called()
    def test_quota_is_in_first_public_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            controller=RemoteController.__new__(RemoteController);controller.state=Path(d);bucket=SimpleNamespace(json=Mock());controller.bucket=bucket;controller.signed=lambda payload:dict(payload=payload)
            def base_open(instance,*args,**kwargs):
                manifest=dict(epoch='nonpayable-test',max_batches=4)
                instance.bucket.json('public/nonpayable-test/manifest.json',instance.signed(manifest));instance.bucket.json('public/nonpayable-test/current.json',{'payload':{'epoch':'nonpayable-test'}});return manifest
            with patch('subnet.controller.Controller.open',base_open):controller.open('nonpayable-test',{},[],max_batches=3)
            manifests=[c.args[1]['payload'] for c in bucket.json.call_args_list if c.args[0].endswith('/manifest.json')]
            self.assertEqual(manifests,[dict(epoch='nonpayable-test',max_batches=3)])
