import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from subnet.checkpoint_evaluator import enqueue,evaluate_one,progress,VERSION
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
        enqueue(self.controller,self.manifest,'cache','after',3,self.config)
        return self.state/'checkpoint-evaluations'/(self.manifest['epoch']+'-eval-after.json')
    def test_enqueue_performs_no_gpu_work_and_pins_exact_plan(self):
        path=self.queued();self.controller.jobs.run.assert_not_called()
        original=path.read_bytes();self.queued();self.assertEqual(path.read_bytes(),original)
        request=json.loads(original)['request']
        self.assertEqual(request['heldout_plan'][0]['indices'],[2,3])
        self.assertEqual(request['heldout_plan'][0]['seeds'],[2100,3100])
        self.config['heldout'][0]['seed']=999
        with self.assertRaisesRegex(ValueError,'immutable'):self.queued()
    def test_actual_result_preserves_cohort_ids_and_provenance(self):
        path=self.queued();record=evaluate_one(self.controller,path)
        result=record['records'][0]
        self.assertEqual(result['remote_job_id'],'job');self.assertEqual(result['timestamp'],77)
        self.assertEqual(result['heldout_indices'],[2,3]);self.assertEqual(result['fixed_task_ids'],['2'*64,'3'*64])
        self.assertEqual(result['training_steps'],3)
        self.controller.jobs.run.assert_called_once()
        evaluate_one(self.controller,path);self.controller.jobs.run.assert_called_once()
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
