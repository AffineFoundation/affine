import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from subnet.checkpoint_evaluator import enqueue,evaluate_one,progress,pending_pass,VERSION
from subnet.remote_backend import RemoteObservationTimeout
from test_gpu_service import GPUFixedHeldout

class IndependentEvaluator(unittest.TestCase):
    def setUp(self):
        fixture=GPUFixedHeldout();fixture.setUp()
        self.config=dict(fixture.config,evaluation_mode=VERSION)
        self.manifest=fixture.manifest;self.row=fixture.row;self.report=fixture.report()
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.state=Path(self.temp.name);self.config['evaluation_state']=str(self.state/'evaluations')
        self.controller=SimpleNamespace(state=self.state,bucket=SimpleNamespace(json=Mock()),signed=lambda v:v,jobs=SimpleNamespace(run=Mock(return_value=self.report)))
        self.patch=patch('subnet.gpu_service.definitions',return_value=[self.row]);self.patch.start();self.addCleanup(self.patch.stop)
    def queued(self):
        record=enqueue(self.controller,self.manifest,'cache','after',3,self.config)
        return self.state/'checkpoint-evaluations'/(record['evaluation_id']+'.json')
    def test_enqueue_performs_no_gpu_work_and_pins_exact_plan(self):
        path=self.queued();self.controller.jobs.run.assert_not_called()
        original=path.read_bytes();self.queued();self.assertEqual(path.read_bytes(),original)
        request=json.loads(original)['request']
        self.assertEqual(request['heldout_plan'][0]['indices'],[2,3])
        self.assertEqual(request['heldout_plan'][0]['seeds'],[2100,3100])
        self.config['heldout'][0]['seed']=999
        changed=self.queued();self.assertNotEqual(changed,path)
        self.assertEqual(path.read_bytes(),original)
    def test_actual_result_preserves_cohort_ids_and_provenance(self):
        path=self.queued();record=evaluate_one(self.controller,path)
        result=record['records'][0]
        self.assertEqual(result['remote_job_id'],'job');self.assertEqual(result['timestamp'],77)
        self.assertEqual(result['heldout_indices'],[2,3]);self.assertEqual(result['fixed_task_ids'],['2'*64,'3'*64])
        self.assertEqual(result['training_steps'],3)
        self.controller.jobs.run.assert_called_once()
        evaluate_one(self.controller,path);self.controller.jobs.run.assert_called_once()
    def test_source_url_refresh_reuses_bytes_but_source_change_does_not(self):
        self.manifest['source_bundle']=dict(sha256='a'*64,format='tar.gz',url='original-capability')
        first=self.queued()
        self.manifest['source_bundle']=dict(sha256='a'*64,format='tar.gz',url='renewed-capability')
        self.assertEqual(self.queued(),first)
        self.manifest['source_bundle']=dict(sha256='b'*64,format='tar.gz',url='renewed-capability')
        self.assertNotEqual(self.queued(),first)
    def test_observation_timeout_preserves_request_and_original_job(self):
        path=self.queued();self.controller.jobs.run.side_effect=[RemoteObservationTimeout('original','evaluate'),self.report]
        request=json.loads(path.read_text())['request']
        first=evaluate_one(self.controller,path)
        self.assertEqual(first['status'],'observing_original_job');self.assertEqual(first['remote_job_id'],'original')
        self.assertEqual(first['request'],request)
        second=evaluate_one(self.controller,path);self.assertEqual(second['status'],'complete')
        self.assertEqual(self.controller.jobs.run.call_args_list[0],self.controller.jobs.run.call_args_list[1])
    def test_failed_publication_does_not_mark_complete(self):
        path=self.queued();self.controller.bucket.json.side_effect=OSError('unavailable')
        with self.assertRaises(OSError):evaluate_one(self.controller,path)
        self.assertEqual(json.loads(path.read_text())['status'],'queued')
        self.controller.bucket.json.side_effect=None;self.assertEqual(evaluate_one(self.controller,path)['status'],'complete')
    def test_progress_does_not_relabel_old_eval_as_current(self):
        path=self.queued();evaluate_one(self.controller,path)
        (self.state/'controller.json').write_text(json.dumps(dict(checkpoint={'id':'new'},public_optimizer_steps=4)))
        value=progress(self.state)
        self.assertEqual(value['latest_training_checkpoint'],'new');self.assertEqual(value['latest_evaluated_checkpoint'],'approved')
        self.assertFalse(value['evaluation_caught_up']);self.assertEqual(value['public_optimizer_steps'],4)
    def test_fixed_32_and_200_cohorts_keep_comparison_id_across_checkpoints(self):
        for count in (32,200):
            with self.subTest(count=count):
                self.row['spec']['num_samples']=1000
                self.config['heldout'][0]['indices']=list(range(100,100+count))
                self.config['evaluation_experiment_id']='fixed-'+str(count)
                self.report['heldout']=[dict(env_id='env',index=i,seed=100+i*1000,task_hash=format(i,'064x'),verified=True,reward=0,classification='negative') for i in self.config['heldout'][0]['indices']]
                self.manifest['epoch']='nonpayable-cohort-'+str(count);self.manifest['checkpoint']={'id':'approved'}
                first=evaluate_one(self.controller,self.queued())['records'][0]
                self.manifest=dict(self.manifest,epoch='nonpayable-cohort-next-'+str(count),checkpoint={'id':'next'})
                second=evaluate_one(self.controller,self.queued())['records'][0]
                self.assertEqual(first['count'],count);self.assertEqual(second['count'],count)
                self.assertEqual(first['taskset_hash'],second['taskset_hash'])
                self.assertEqual(first['experiment_id'],second['experiment_id'])
                self.assertEqual(first['fixed_task_ids'],second['fixed_task_ids'])
                self.assertNotEqual(first['checkpoint'],second['checkpoint'])
    def test_adjacent_epochs_evaluate_checkpoint_once_without_relabeling(self):
        path=self.queued();first=evaluate_one(self.controller,path)
        self.manifest=dict(self.manifest,epoch='nonpayable-next')
        reused=enqueue(self.controller,self.manifest,'different-cache','before',999,self.config)
        self.assertEqual(reused['request']['training_steps'],3)
        self.assertEqual(reused['request']['manifest']['epoch'],'nonpayable-gpu-test')
        self.assertEqual(reused['records'],first['records'])
        self.assertEqual(len(list((self.state/'checkpoint-evaluations').glob('*.json'))),1)
        evaluate_one(self.controller,path);self.controller.jobs.run.assert_called_once()
        reference=json.loads((self.state/'checkpoint-evaluation-references'/'nonpayable-next-eval-before.json').read_text())
        self.assertTrue(reference['reuses_original_execution']);self.assertEqual(reference['original_training_steps'],3)
    def test_authenticated_historical_baseline_keeps_original_execution_labels(self):
        import base64,hashlib
        from nacl.signing import SigningKey
        from subnet.storage import canonical
        from subnet.remote_backend import RemoteJobs
        key=SigningKey.generate();operator=key.verify_key.encode().hex()
        sign=lambda value:dict(payload=value,signer=operator,signature=base64.b64encode(key.sign(canonical(value)).signature).decode())
        label=self.manifest['epoch']+'-eval-after';jobid=label+'-original'
        from subnet.gpu_service import heldout
        job=dict(job_id=jobid,role='evaluate',manifest=sign(self.manifest),heldout=heldout(self.config,self.manifest),created_at=10,expires_at=100)
        digest=hashlib.sha256(canonical(job)).hexdigest()
        prior=dict(job_id=jobid,role='evaluate',job_sha256=digest,source_files=self.report['source_files'],runtime_versions=self.report['runtime_versions'],manifest_sha256=hashlib.sha256(canonical(self.manifest)).hexdigest())
        report=dict(self.report,job_id=jobid,role='evaluate',operator=operator,job_sha256=digest,checkpoint=self.manifest['checkpoint']['id'],epoch=self.manifest['epoch'],success=True,chain_transactions=False,backend_profile=self.manifest['backend_profile'],numerical_policy=self.manifest['numerical_policy'])
        roles=self.state/'roles';roles.mkdir()
        for name,value in [(label+'.json',prior),(jobid+'-job.json',sign(job)),(jobid+'-report.json',report)]:
            (roles/name).write_text(json.dumps(value))
        folder=Path(self.config['evaluation_state']);folder.mkdir()
        (folder/(label+'-env.json')).write_text(json.dumps(dict(remote_job_id=jobid,checkpoint='approved',timestamp=77,training_steps=7)))
        jobs=RemoteJobs.__new__(RemoteJobs);jobs.state=roles;jobs.config={'retain_original_jobs':True};jobs.controller=SimpleNamespace(authority=SimpleNamespace(id=operator))
        self.controller.jobs=jobs;self.controller.authority=jobs.controller.authority
        self.manifest=dict(self.manifest,epoch='nonpayable-new-reference')
        record=enqueue(self.controller,self.manifest,'cache','before',999,self.config)
        self.assertEqual(record['request']['label'],label);self.assertEqual(record['request']['training_steps'],7)
        self.assertEqual(record['request']['manifest']['epoch'],'nonpayable-gpu-test')
        final=evaluate_one(self.controller,self.state/'checkpoint-evaluations'/(record['evaluation_id']+'.json'))
        self.assertEqual(final['records'][0]['remote_job_id'],jobid)
        self.assertEqual(final['records'][0]['training_steps'],7)
    def test_deferred_old_source_preserves_original_and_runs_newest_approved(self):
        self.manifest['source_bundle']={'sha256':'a'*64};old=self.queued();original=old.read_bytes()
        self.manifest=dict(self.manifest,epoch='nonpayable-new',checkpoint={'id':'new'},source_bundle={'sha256':'b'*64});self.queued()
        self.controller.jobs.dispatch_eligible=lambda r:r['manifest']['source_bundle']['sha256']=='b'*64
        result=pending_pass(self.controller)
        self.assertEqual(result['request']['manifest']['checkpoint']['id'],'new');self.assertEqual(old.read_bytes(),original)
        deferred=json.loads((self.state/'checkpoint-evaluation-deferrals'/old.name).read_text());self.assertFalse(deferred['remote_job_started'])
        self.controller.jobs.run.assert_called_once()
    def test_newest_first_keeps_issued_original_timeout_first(self):
        old=self.queued();record=json.loads(old.read_text());(self.state/'roles').mkdir()
        (self.state/'roles'/(record['request']['label']+'.json')).write_text('{}')
        self.manifest=dict(self.manifest,epoch='nonpayable-new',checkpoint={'id':'new'});new=self.queued()
        self.controller.jobs.dispatch_eligible=lambda r:True
        self.controller.jobs.run.side_effect=RemoteObservationTimeout('same-issued-original','evaluate')
        result=pending_pass(self.controller,dispatch_order='latest-approved-source-first-v1')
        self.assertEqual(result['remote_job_id'],'same-issued-original');self.assertEqual(self.controller.jobs.run.call_args.args[0],record['request']['label'])
        self.assertEqual(json.loads(new.read_text())['status'],'queued')
    def test_newest_first_runs_latest_preserving_old_request(self):
        old=self.queued();original=old.read_bytes();self.manifest=dict(self.manifest,epoch='nonpayable-new',checkpoint={'id':'new'});self.queued()
        result=pending_pass(self.controller,dispatch_order='latest-approved-source-first-v1')
        self.assertEqual(result['request']['manifest']['checkpoint']['id'],'new');self.assertEqual(old.read_bytes(),original)
    def test_tampered_deferred_queue_fails_closed(self):
        old=self.queued();record=json.loads(old.read_text());record['request']['training_steps']=99;old.write_text(json.dumps(record))
        self.controller.jobs.dispatch_eligible=lambda r:False;self.controller.jobs.busy=lambda:False
        self.assertIsNone(pending_pass(self.controller));self.assertFalse((self.state/'checkpoint-evaluation-deferrals'/old.name).exists())
        self.assertEqual(json.loads((self.state/'checkpoint-evaluation-faults'/old.name).read_text())['status'],'failed');self.controller.jobs.run.assert_not_called()
    def test_explicit_infrastructure_recovery_preserves_manifest_cohort_and_provenance(self):
        from subnet.storage import canonical
        import hashlib
        old=self.queued();original=old.read_bytes();r=json.loads(original)
        r['request']['label']+='-infra-cache-route-v1';r['request_sha256']=hashlib.sha256(canonical(r['request'])).hexdigest();r['evaluation_id']+='-infra-cache-route-v1'
        recovery=old.with_name(r['evaluation_id']+'.json');recovery.write_text(json.dumps(r))
        result=evaluate_one(self.controller,recovery)
        self.assertEqual(self.controller.jobs.run.call_args.args[0],r['request']['label'])
        self.assertEqual(self.controller.jobs.run.call_args.args[2],self.manifest)
        self.assertEqual(result['records'][0]['run_id'],r['request']['label']+'-env')
        self.assertEqual(result['records'][0]['heldout_indices'],[2,3]);self.assertEqual(old.read_bytes(),original)
    def test_failed_old_request_does_not_starve_new_checkpoint(self):
        from subnet.remote_backend import RemoteJobTerminalError
        old=self.queued()
        self.manifest=dict(self.manifest,epoch='nonpayable-new',checkpoint={'id':'new'})
        new=self.queued();self.controller.jobs.busy=Mock(return_value=False)
        self.controller.jobs.run.side_effect=[RemoteJobTerminalError('failed original'),self.report]
        result=pending_pass(self.controller)
        self.assertEqual(result['status'],'complete');self.assertEqual(result['request']['manifest']['checkpoint']['id'],'new')
        failure=json.loads((self.state/'checkpoint-evaluation-faults'/old.name).read_text())
        self.assertEqual(failure['status'],'failed');self.assertNotIn('records',failure)
        pending_pass(self.controller);self.assertEqual(self.controller.jobs.run.call_count,2)
    def test_corrupt_request_is_retained_and_new_checkpoint_runs_when_idle(self):
        old=self.queued();old.write_text('{broken')
        self.manifest=dict(self.manifest,epoch='nonpayable-new',checkpoint={'id':'new'});new=self.queued()
        self.controller.jobs.busy=Mock(return_value=False)
        result=pending_pass(self.controller)
        self.assertEqual(result['status'],'complete');self.assertEqual(old.read_text(),'{broken')
        self.assertEqual(json.loads((self.state/'checkpoint-evaluation-faults'/old.name).read_text())['status'],'failed')
    def test_unknown_gpu_liveness_does_not_dispatch_new_checkpoint(self):
        from subnet.remote_backend import RemoteJobTerminalError
        old=self.queued();self.manifest=dict(self.manifest,epoch='nonpayable-new',checkpoint={'id':'new'});self.queued()
        self.controller.jobs.busy=Mock(side_effect=OSError('SSH unavailable'))
        self.controller.jobs.run.side_effect=RemoteJobTerminalError('original failed but liveness unknown')
        result=pending_pass(self.controller)
        self.assertEqual(result['status'],'retry_original_request');self.assertEqual(self.controller.jobs.run.call_count,1)
    def test_tampered_queue_or_wrong_cohort_fails_before_dispatch(self):
        path=self.queued();record=json.loads(path.read_text());record['request']['heldout_plan'][0]['seeds'][0]+=1
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'queue binding'):evaluate_one(self.controller,path)
        self.controller.jobs.run.assert_not_called()

if __name__=='__main__':unittest.main()

class EpochTiming(unittest.TestCase):
    def test_empty_epoch_is_not_training_step_or_full_hourly_success(self):
        from subnet.epoch_timing import transition,completion
        active=dict(phase='collect',phase_started_at=10,started_at=0,next_steps=3,next_checkpoint={'id':'same'})
        transition(active,'after',now=20)
        result=completion(active,{'epoch':'e'},3,now=25)
        self.assertFalse(result['nonempty_training_update']);self.assertEqual(result['training_updates'],0)
        self.assertEqual(result['controller_seconds'],25);self.assertEqual(result['phase_timings'][0]['seconds'],10)
        self.assertFalse(result['chain_weights_observed']);self.assertFalse(result['full_hourly_epoch_confirmed'])

class IndependentControllerCommit(unittest.TestCase):
    def test_after_phase_commits_real_parent_without_calling_evaluator(self):
        from subnet.gpu_service import run
        with tempfile.TemporaryDirectory() as d:
            state=Path(d);epoch='nonpayable-async'
            pointer=dict(inference_checkpoint='new',optimizer_steps=4)
            active=dict(epoch=epoch,phase='after',next_checkpoint={'id':'new'},next_path='new-path',next_steps=4,next_trainer_state=pointer)
            status=dict(active=active,round=1,training_steps=3,checkpoint={'id':'old'},checkpoint_path='old-path',initial_published=True)
            for name,value in [('controller.json',status),('latest-trainer-state.json',pointer),(epoch+'-manifest.json',dict(epoch=epoch,checkpoint={'id':'old'})),(epoch+'-verified.json',{}),(epoch+'-scores.json',dict(weights={})),('finalized-reports.json',[])]:
                (state/name).write_text(json.dumps(value))
            config=dict(state=d,bucket={},remote={},source_bundle={},epoch_prefix='nonpayable-async',registration_allowlist=[],evaluation_mode=VERSION)
            controller=SimpleNamespace(train=Mock(),signed=lambda v:v)
            with patch('subnet.gpu_service.Bucket'),patch('subnet.gpu_service.Gateway'),patch('subnet.gpu_service.RemoteController',return_value=controller),patch('subnet.gpu_service.ChainAdapter'),patch('subnet.gpu_service.evaluate') as evaluate,patch('subnet.checkpoint_evaluator.enqueue') as queue,patch('subnet.gpu_service.publish_history'):
                run(config,once=True)
            evaluate.assert_not_called();controller.train.assert_not_called();queue.assert_called_once()
            final=json.loads((state/'controller.json').read_text())
            self.assertIsNone(final['active']);self.assertEqual(final['trainer_state'],pointer)
            self.assertEqual(final['checkpoint']['id'],'new');self.assertEqual(final['public_optimizer_steps'],4)
            self.assertEqual(final['round'],2)
            timing=json.loads((state/(epoch+'-controller-timing.json')).read_text())
            self.assertTrue(timing['nonempty_training_update']);self.assertFalse(timing['full_hourly_epoch_confirmed'])
    def test_wrong_durable_parent_refuses_controller_commit(self):
        from subnet.gpu_service import run
        with tempfile.TemporaryDirectory() as d:
            state=Path(d);epoch='nonpayable-wrong-parent'
            pointer=dict(inference_checkpoint='new',optimizer_steps=4)
            status=dict(active=dict(epoch=epoch,phase='after',next_checkpoint={'id':'new'},next_path='new-path',next_steps=4,next_trainer_state=pointer),round=1,training_steps=3,checkpoint={'id':'old'},checkpoint_path='old-path',initial_published=True)
            for name,value in [('controller.json',status),('latest-trainer-state.json',dict(pointer,optimizer_steps=999)),(epoch+'-manifest.json',dict(epoch=epoch,checkpoint={'id':'old'})),(epoch+'-verified.json',{}),(epoch+'-scores.json',dict(weights={})),('finalized-reports.json',[])]:
                (state/name).write_text(json.dumps(value))
            config=dict(state=d,bucket={},remote={},source_bundle={},epoch_prefix='nonpayable-async',registration_allowlist=[],evaluation_mode=VERSION)
            controller=SimpleNamespace(train=Mock(),signed=lambda v:v)
            with patch('subnet.gpu_service.Bucket'),patch('subnet.gpu_service.Gateway'),patch('subnet.gpu_service.RemoteController',return_value=controller),patch('subnet.gpu_service.ChainAdapter'),patch('subnet.checkpoint_evaluator.enqueue'),patch('subnet.gpu_service.publish_history'),patch('subnet.gpu_service.log.exception'):
                with self.assertRaisesRegex(ValueError,'journal mismatch'):run(config,once=True)
            self.assertEqual(json.loads((state/'controller.json').read_text()),status)
