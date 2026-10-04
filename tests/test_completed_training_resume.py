import base64,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from nacl.signing import SigningKey
from subnet.backend_jobs import BACKEND_PROFILE,NUMERICAL_POLICY,REVISION,FIXED_POLICY,COVERED_POLICY,canonical
from subnet.remote_backend import RemoteJobs,RemoteController


class OriginalTrainingResume(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name);self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        self.jobs=RemoteJobs.__new__(RemoteJobs);self.jobs.state=self.root
        self.jobs.controller=SimpleNamespace(authority=SimpleNamespace(id=self.authority))
        self.manifest=dict(epoch='nonpayable-original',checkpoint={'id':'input'},model_runtime_revision=REVISION,backend_profile=BACKEND_PROFILE,numerical_policy=NUMERICAL_POLICY)
        self.submissions=[{'sha256':'original','url':'expired-original-url'}]
        self.job=dict(job_id='original-job',role='train',manifest=self.sign(self.manifest),steps=3,training_policy=FIXED_POLICY,submissions=self.submissions,created_at=10,expires_at=30)
        self.prior=dict(job_id='original-job',role='train',job_sha256=hashlib.sha256(canonical(self.job)).hexdigest(),source_files={'source':'approved'},runtime_versions={'torch':'approved'},manifest_sha256=hashlib.sha256(canonical(self.manifest)).hexdigest())
        self.report=dict(job_id='original-job',role='train',operator=self.authority,job_sha256=self.prior['job_sha256'],completed_at=20,checkpoint='input',epoch=self.manifest['epoch'],success=True,chain_transactions=False,backend_profile=BACKEND_PROFILE,numerical_policy=NUMERICAL_POLICY,source_files=self.prior['source_files'],runtime_versions=self.prior['runtime_versions'])
        (self.root/'original-train.json').write_bytes(canonical(self.prior));(self.root/'original-job-job.json').write_bytes(canonical(self.sign(self.job)))
        self.jobs.remote_status=Mock(return_value={'phase':'running'})
        self.jobs.capacity=Mock(side_effect=ValueError('outputs consumed free reserve'))

    def sign(self,payload):return dict(payload=payload,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())

    def resume(self,**changes):
        args=dict(label='original-train',manifest=self.manifest,submissions=[dict(self.submissions[0],url='fresh-url')],steps=3,replay=None);args.update(changes)
        return self.jobs.training_resume(**args)

    def test_completed_original_report_recovers_with_consumed_disk_without_network_or_training(self):
        (self.root/'original-job-report.json').write_bytes(canonical(self.report))
        result=self.resume();self.assertEqual(result['original_phase'],'complete')
        self.assertFalse(result['new_training_started']);self.jobs.remote_status.assert_not_called();self.jobs.capacity.assert_not_called()

    def test_live_original_is_observed_without_new_capacity_or_launch(self):
        self.assertEqual(self.resume()['original_phase'],'running')
        self.jobs.remote_status.assert_called_once_with('original-job');self.jobs.capacity.assert_not_called()
        self.assertIsNone(self.resume(label='new-training'))

    def test_covered_original_resume_keeps_exact_policy_and_context(self):
        self.manifest=dict(self.manifest,training_policy=COVERED_POLICY,
            training_coverage={'seed':'ab'*32,'receipts_sha256':'cd'*32})
        self.job=dict(self.job,manifest=self.sign(self.manifest),training_policy=COVERED_POLICY)
        self.prior=dict(self.prior,job_sha256=hashlib.sha256(canonical(self.job)).hexdigest(),
            manifest_sha256=hashlib.sha256(canonical(self.manifest)).hexdigest())
        (self.root/'original-train.json').write_bytes(canonical(self.prior))
        (self.root/'original-job-job.json').write_bytes(canonical(self.sign(self.job)))
        result=self.resume();self.assertEqual(result['original_job_id'],'original-job')
        self.jobs.capacity.assert_not_called()
        with self.assertRaisesRegex(ValueError,'request changed'):
            self.resume(manifest=dict(self.manifest,training_policy=FIXED_POLICY))

    def test_changed_request_or_corrupted_original_signature_refuses(self):
        for changes in ({'steps':2},{'steps':True},{'submissions':[{'sha256':'changed'}]}, {'manifest':dict(self.manifest,epoch='another')},{'replay':{'new':'inputs'}}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.resume(**changes)
        envelope=self.sign(self.job);envelope['payload']['steps']=1
        (self.root/'original-job-job.json').write_bytes(canonical(envelope))
        with self.assertRaises(Exception):self.resume()
        self.jobs.capacity.assert_not_called()

    def test_original_failed_or_missing_never_relaunches(self):
        for phase in ('failed','not_launched'):
            self.jobs.remote_status=Mock(return_value={'phase':phase})
            with self.assertRaisesRegex(ValueError,'without relaunch'):self.resume()
            self.jobs.copy_to=Mock();self.jobs.command=Mock()
            with self.assertRaisesRegex(RuntimeError,'refuse automatic relaunch'):
                self.jobs.run('original-train','train',self.manifest)
            self.jobs.copy_to.assert_not_called();self.jobs.command.assert_not_called()

    def test_corrupt_completed_report_refuses(self):
        (self.root/'original-job-report.json').write_bytes(canonical(dict(self.report,checkpoint='other')))
        with self.assertRaises(ValueError):self.resume()

    def test_streaming_publication_only_requires_metadata_space(self):
        self.jobs.python='/python';self.jobs.command=Mock(return_value=json.dumps({'free_bytes':2*1024**2,'checkpoint_bytes':15*1024**3}))
        self.assertEqual(self.jobs.publication_capacity('/already-complete')['required_bytes'],1024**2)
        self.jobs.command.return_value=json.dumps({'free_bytes':0,'checkpoint_bytes':15*1024**3})
        with self.assertRaises(ValueError):self.jobs.publication_capacity('/already-complete')

    def test_controller_uses_original_request_before_new_training_reserve(self):
        state=self.root/'controller';state.mkdir();epoch=self.manifest['epoch']
        (state/(epoch+'-scores.json')).write_bytes(canonical({'receipts':{'miner':{'size':100,'sha256':'original','frozen_key':'frozen'}}}))
        controller=RemoteController.__new__(RemoteController);controller.state=state
        controller.bucket=SimpleNamespace(presign=lambda key:'new-url',json=Mock());controller.signed=lambda value:{'payload':value}
        files={'model.safetensors':'b'*64};cp={'id':hashlib.sha256(canonical(files)).hexdigest(),'files':files,'path':'/original-output'}
        controller.jobs=SimpleNamespace(training_resume=Mock(return_value={'resuming_original_training':True}),training_capacity=Mock(side_effect=AssertionError('do not allocate a second reserve')),run=Mock(return_value={'new_checkpoint':cp,'training':{'updates':[]},'job_id':'original-job'}))
        controller.publish_remote_checkpoint=Mock(return_value={k:v for k,v in cp.items() if k!='path'})
        _,metrics=controller.train(self.manifest,{'miner':{'accepted':[{}],'submission_sha256':'original'}},'/input',steps=3)
        controller.jobs.training_capacity.assert_not_called();self.assertEqual(metrics['remote_job_id'],'original-job')
        self.assertEqual(controller.jobs.run.call_args.args[:2],(epoch+'-train','train'))
        self.assertTrue(metrics['capacity_preflight']['resuming_original_training'])
        # With no original request, a failed reserve must still prevent launch.
        (state/(epoch+'-training-metrics.json')).unlink()
        controller.jobs.training_resume.return_value=None;controller.jobs.run.reset_mock()
        with self.assertRaisesRegex(AssertionError,'second reserve'):
            controller.train(self.manifest,{'miner':{'accepted':[{}],'submission_sha256':'original'}},'/input',steps=3)
        controller.jobs.run.assert_not_called()


if __name__=='__main__':unittest.main()
