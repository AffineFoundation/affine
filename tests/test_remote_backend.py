import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from types import SimpleNamespace
from subnet.remote_backend import RemoteJobs,RemoteController,role_time_budget,RemoteObservationTimeout
from nacl.signing import SigningKey
import base64
from subnet.backend_jobs import BACKEND_PROFILE,NUMERICAL_POLICY,REVISION,canonical,signed
import hashlib

class RemoteReportBinding(unittest.TestCase):
    def setUp(self):
        self.key=SigningKey.generate();self.operator=self.key.verify_key.encode().hex()
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.jobs=RemoteJobs.__new__(RemoteJobs);self.jobs.controller=SimpleNamespace(authority=SimpleNamespace(id=self.operator));self.jobs.state=Path(self.folder.name)
        self.manifest=dict(epoch='nonpayable-test',checkpoint={'id':'approved'},model_runtime_revision=REVISION,backend_profile=BACKEND_PROFILE,numerical_policy=NUMERICAL_POLICY)
        job=dict(job_id='same-job',role='verify',created_at=10,expires_at=30)
        self.envelope=dict(payload=job,signer=self.operator,signature=base64.b64encode(self.key.sign(canonical(job)).signature).decode())
        digest=hashlib.sha256(canonical(job)).hexdigest()
        self.prior=dict(job_id='same-job',role='verify',job_sha256=digest,source_files={'a':'b'},runtime_versions={'torch':'approved'},manifest_sha256=hashlib.sha256(canonical(self.manifest)).hexdigest())
        self.report=dict(job_id='same-job',role='verify',operator=self.operator,job_sha256=digest,completed_at=20,checkpoint='approved',epoch='nonpayable-test',success=True,chain_transactions=False,backend_profile=BACKEND_PROFILE,numerical_policy=NUMERICAL_POLICY,source_files={'a':'b'},runtime_versions={'torch':'approved'})
        (self.jobs.state/'same-job-job.json').write_text(json.dumps(self.envelope))
    def test_approved_report_binding(self):self.assertIs(self.jobs.checked(self.report,self.prior,self.manifest),self.report)
    def test_changed_identity_sources_or_runtime_is_rejected(self):
        for name,value in [('job_id','another-job'),('source_files',{'a':'bad'}),('runtime_versions',{'torch':'unapproved'}),('chain_transactions',True),('backend_profile',{'device':'cpu'}),('backend_profile',dict(BACKEND_PROFILE,tf32=0)),('success',1)]:
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
    def test_observation_timeout_resumes_same_original_job_without_new_signature(self):
        self.jobs.config={};self.jobs.workspace='/remote'
        (self.jobs.state/'label.json').write_text(json.dumps(self.prior))
        self.jobs.remote_status=Mock(return_value={'phase':'running'})
        self.jobs.command=Mock();self.jobs.copy_to=Mock()
        original=(self.jobs.state/'same-job-job.json').read_bytes()
        with patch('subnet.remote_backend.time.time',side_effect=[0,1801]),patch('subnet.remote_backend.time.sleep'):
            with self.assertRaises(RemoteObservationTimeout) as caught:self.jobs.run('label','verify',self.manifest)
        self.assertEqual(caught.exception.job_id,'same-job')
        self.jobs.remote_status=Mock(return_value={'phase':'complete'})
        self.jobs.copy_from=lambda remote,local:local.write_text(json.dumps(self.report))
        result=self.jobs.run('label','verify',self.manifest)
        self.assertEqual(result['job_id'],'same-job')
        self.assertEqual((self.jobs.state/'same-job-job.json').read_bytes(),original)
        self.jobs.command.assert_not_called();self.jobs.copy_to.assert_not_called()
    def test_independent_evaluation_does_not_relaunch_terminal_job(self):
        self.jobs.config={'retain_original_jobs':True};self.jobs.workspace='/remote'
        job=dict(self.envelope['payload'],role='evaluate',heldout=[{'indices':[1]}],manifest=self.envelope_manifest())
        envelope=dict(payload=job,signer=self.operator,signature=base64.b64encode(self.key.sign(canonical(job)).signature).decode())
        prior=dict(self.prior,role='evaluate',job_sha256=hashlib.sha256(canonical(job)).hexdigest())
        (self.jobs.state/'same-job-job.json').write_text(json.dumps(envelope))
        (self.jobs.state/'label.json').write_text(json.dumps(prior))
        self.jobs.remote_status=Mock(return_value={'phase':'failed'});self.jobs.copy_to=Mock()
        with self.assertRaisesRegex(RuntimeError,'refuse automatic relaunch'):
            self.jobs.run('label','evaluate',self.manifest,heldout=[{'indices':[1]}])
        self.jobs.copy_to.assert_not_called()
        with self.assertRaisesRegex(ValueError,'evaluation request changed'):
            self.jobs.run('label','evaluate',self.manifest,heldout=[{'indices':[2]}])
    def envelope_manifest(self):
        return dict(payload=self.manifest,signer=self.operator,signature=base64.b64encode(self.key.sign(canonical(self.manifest)).signature).decode())
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
    def test_new_full_coverage_training_job_binds_longer_deadline_at_creation(self):
        self.jobs.config={'job_ttl_seconds_by_role':{'train':86400,'evaluate':10800}}
        self.jobs.metadata=dict(source_files=self.prior['source_files'],runtime_versions=self.prior['runtime_versions'])
        self.jobs.workspace='/remote';self.jobs.code='/frozen';self.jobs.python='/python'
        self.jobs.controller.signed=lambda payload:dict(payload=payload,signer=self.operator,signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())
        self.jobs.command=Mock();self.jobs.copy_to=Mock();self.jobs.remote_status=Mock(return_value={'phase':'complete'})
        def fetch(remote,local):
            job=signed(json.loads(next(self.jobs.state.glob('covered-training-*-job.json')).read_text()),self.operator)
            self.assertEqual(job['expires_at']-job['created_at'],86400)
            local.write_text(json.dumps(dict(self.report,job_id=job['job_id'],role='train',job_sha256=hashlib.sha256(canonical(job)).hexdigest(),completed_at=1100)))
        self.jobs.copy_from=fetch
        with patch('subnet.remote_backend.time.time',return_value=1000):
            self.jobs.run('covered-training','train',self.manifest)
        original=next(self.jobs.state.glob('covered-training-*-job.json')).read_bytes()
        self.jobs.config['job_ttl_seconds_by_role']['train']=7200
        with patch('subnet.remote_backend.time.time',return_value=2000):
            self.jobs.run('covered-training','train',self.manifest)
        self.assertEqual(next(self.jobs.state.glob('covered-training-*-job.json')).read_bytes(),original)
        self.jobs.copy_to.assert_called_once()

class RoleTimeBudget(unittest.TestCase):
    def test_full_coverage_training_budget_preserves_other_role_deadlines(self):
        original={'verify':3600,'train':7200,'evaluate':10800,'upload':3600,'mine':3600}
        prospective=dict(original,train=86400)
        for role in original:
            self.assertEqual(role_time_budget({'job_ttl_seconds_by_role':original},role),original[role])
            self.assertEqual(role_time_budget({'job_ttl_seconds_by_role':prospective},role),86400 if role=='train' else original[role])
        self.assertEqual(original['train'],7200)
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

class SuccessorCheckpointReads(unittest.TestCase):
    def setUp(self):
        from subnet.storage import Identity
        from subnet.backend_jobs import file_map
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.state=Path(self.folder.name);self.identity=Identity()
        self.bodies={'config.json':b'{}','model.safetensors':b'actual-trained-weight-bytes'}
        self.files={n:hashlib.sha256(b).hexdigest() for n,b in self.bodies.items()};self.cp={'id':file_map(self.files),'files':self.files}
        self.controller=RemoteController.__new__(RemoteController);self.controller.state=self.state;self.controller.authority=self.identity
        self.controller.jobs=SimpleNamespace(capacity=Mock(return_value={}),run=Mock(return_value={'success':True}))
        def url(key,operation='get_object',*args):return 'https://test.r2.cloudflarestorage.com/b/'+key+'?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test'
        self.controller.bucket=SimpleNamespace(presign=Mock(side_effect=url),get=Mock(side_effect=KeyError('missing')),json=Mock())
    def response(self,url,**kwargs):
        name=url.split('?',1)[0].rsplit('/',1)[1];body=self.bodies[name]
        result=Mock();result.status_code=200;result.headers={'Content-Length':str(len(body))};result.iter_content.return_value=[body];result.__enter__=Mock(return_value=result);result.__exit__=Mock(return_value=False);return result
    def publish(self):
        # S3 missing-key shape, not an unchecked network failure.
        from botocore.exceptions import ClientError
        self.controller.bucket.get=Mock(side_effect=ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject'))
        with patch('subnet.remote_backend.requests.get',side_effect=self.response):return self.controller.publish_remote_checkpoint({'epoch':'nonpayable-source','checkpoint':self.cp},'/trainer/exact-output')
    def test_new_remote_evaluator_fetches_exact_successor_after_publication(self):
        from subnet.backend_jobs import checkpoint
        published=self.publish();self.assertEqual(set(published['read_urls']),set(self.files))
        descriptor=self.controller.bucket.json.call_args.args[1]['payload'];self.assertEqual(descriptor,self.cp);self.assertNotIn('read_urls',descriptor)
        with patch('requests.get',side_effect=self.response):target=checkpoint({'checkpoint':published},self.state/'different-evaluator')
        self.assertEqual({n:(target/n).read_bytes() for n in self.files},self.bodies)
    def test_failed_independent_hash_never_returns_checkpoint_or_signs_descriptor(self):
        self.bodies['model.safetensors']=b'changed-after-upload'
        with self.assertRaisesRegex(ValueError,'independent checkpoint integrity'):self.publish()
        self.controller.bucket.json.assert_not_called()
    def test_signed_parallel_policy_checks_all_bytes_before_descriptor(self):
        import threading
        from botocore.exceptions import ClientError
        from subnet.persistent_publication import VERSION
        barrier=threading.Barrier(2,timeout=3);lock=threading.Lock();finished=set()
        def response(url,**kwargs):
            result=self.response(url,**kwargs);name=url.split('?',1)[0].rsplit('/',1)[1]
            def chunks(size):
                self.assertEqual(size,1024**2);barrier.wait()
                self.controller.bucket.json.assert_not_called()
                yield self.bodies[name]
                with lock:finished.add(name)
            result.iter_content.side_effect=chunks
            return result
        policy=dict(version=VERSION,state_readback='local-full',checkpoint_readback_workers=2)
        self.controller.bucket.get=Mock(side_effect=ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject'))
        with patch('subnet.remote_backend.requests.get',side_effect=response):
            result=self.controller.publish_remote_checkpoint(dict(epoch='nonpayable-source',checkpoint=self.cp,persistent_publication_policy=policy),'/trainer/exact-output')
        self.assertEqual(finished,set(self.files));self.assertEqual(result['id'],self.cp['id'])
        self.controller.bucket.json.assert_called_once()
    def test_encoded_or_truncated_checkpoint_never_signs_descriptor(self):
        original=self.response
        for malformed in ('encoding','truncated'):
            with self.subTest(malformed=malformed):
                def response(url,**kwargs):
                    result=original(url,**kwargs)
                    if malformed=='encoding':result.headers['Content-Encoding']='gzip'
                    else:result.headers['Content-Length']=str(int(result.headers['Content-Length'])+1)
                    return result
                from botocore.exceptions import ClientError
                self.controller.bucket.get=Mock(side_effect=ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject'))
                with patch('subnet.remote_backend.requests.get',side_effect=response),self.assertRaises(ValueError):
                    self.controller.publish_remote_checkpoint({'epoch':'nonpayable-source','checkpoint':self.cp},'/trainer/exact-output')
                self.controller.bucket.json.assert_not_called()
    def test_cached_training_refreshes_narrow_reads_without_gpu_or_saved_metrics_mutation(self):
        from subnet.remote_backend import FULL_POLICY
        metrics=dict(source_epoch='nonpayable-source',input_checkpoint='input',training_policy=FULL_POLICY,weights_changed=True,checkpoint=self.cp['id'],new_checkpoint=self.cp,steps=1)
        path=self.state/'nonpayable-source-training-metrics.json';raw=canonical(metrics);path.write_bytes(raw)
        (self.state/'nonpayable-source-checkpoint-publication.json').write_bytes(canonical(dict(checkpoint=self.cp['id'],operator_independent_hashes=True,objects={n:{'sha256':h} for n,h in self.files.items()})))
        cp,current=self.controller.train({'epoch':'nonpayable-source','checkpoint':{'id':'input'}},{},'/unused',steps=1)
        self.assertEqual(set(cp['read_urls']),set(self.files));self.assertEqual(current['new_checkpoint'],cp);self.assertEqual(path.read_bytes(),raw)
        self.controller.jobs.run.assert_not_called();self.controller.jobs.capacity.assert_not_called()
    def test_cached_reads_reject_missing_or_mismatched_publication_receipt(self):
        from subnet.remote_backend import FULL_POLICY
        metrics=dict(source_epoch='nonpayable-source',input_checkpoint='input',training_policy=FULL_POLICY,weights_changed=True,checkpoint=self.cp['id'],new_checkpoint=self.cp,steps=1)
        (self.state/'nonpayable-source-training-metrics.json').write_bytes(canonical(metrics))
        (self.state/'nonpayable-source-checkpoint-publication.json').write_bytes(canonical(dict(checkpoint=self.cp['id'],operator_independent_hashes=True,objects={})))
        with self.assertRaisesRegex(ValueError,'publication receipt binding'):self.controller.train({'epoch':'nonpayable-source','checkpoint':{'id':'input'}},{},'/unused',steps=1)
        self.controller.bucket.presign.assert_not_called();self.controller.jobs.run.assert_not_called()
