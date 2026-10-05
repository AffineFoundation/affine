import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from subnet.gpu_service import heldout,evaluate,contract,run
from subnet.backend_profiles import for_config

class GPUFixedHeldout(unittest.TestCase):
    def setUp(self):
        self.row={'spec':{'id':'env','num_samples':4,'max_output_tokens':512,'version':'fixed-v1'},'indices':[0,1],'harness':{'version':'text-tools-v1'}}
        self.config={'environments':[], 'heldout':[dict(env_id='env',indices=[2,3],seed=100,harness=dict(version='text-tools-v1',policy='autoregressive',max_output_tokens=128))]}
        self.manifest={'epoch':'nonpayable-gpu-test','checkpoint':{'id':'approved'},'environments':[dict(env_id='env',**self.row)],'harness_source_hash':'pinned'}
        revision,profile,policy=for_config({})
        self.manifest.update(model_runtime_revision=revision,backend_profile=profile,numerical_policy=policy)
    def test_inactive_epoch_group_cannot_train_on_fixed_heldout(self):
        self.manifest['environments'][0]['indices']=[];self.config['heldout'][0]['indices']=[1]
        with patch('subnet.gpu_service.definitions',return_value=[self.row]),self.assertRaisesRegex(ValueError,'fixed heldout binding'):heldout(self.config,self.manifest)
    def test_prospective_rotation_covers_training_and_preserves_other_groups(self):
        other=dict(self.row,spec=dict(self.row['spec'],id='other'),indices=[0,1])
        config=dict(heldout=[],training_groups=[['env'],['other']],source_bundle={},indices_per_environment_per_epoch=1)
        with patch('subnet.gpu_service.definitions',return_value=[self.row,other]):
            rounds=[contract(config,n)['environments'] for n in range(4)]
        self.assertEqual([rounds[n][0]['indices'] for n in range(4)],[[0],[],[1],[]])
        self.assertEqual([rounds[n][1]['indices'] for n in range(4)],[[],[0],[],[1]])
        self.assertTrue(all(set(r['indices'])<={0,1} for rows in rounds for r in rows))
    def test_hopper_contract_and_artifact_policy_are_explicit(self):
        with patch('subnet.gpu_service.definitions',return_value=[self.row]):
            value=contract(dict(source_bundle={},heldout=[],model_runtime_revision='cuda-bf16-eager-sm90-v1',artifact_policy='full-vocabulary-long-v1'),0)
        self.assertEqual(value['backend_profile']['sm'],[9,0])
        self.assertEqual(value['numerical_policy']['logprob_atol'],0.00001)
        self.assertEqual(value['artifact_policy'],'full-vocabulary-long-v1')

    def test_unconfigured_rotation_retains_historical_indices(self):
        with patch('subnet.gpu_service.definitions',return_value=[self.row]):
            value=contract(dict(source_bundle={},heldout=[]),17)
        self.assertEqual(value['environments'][0]['indices'],[0,1])
    def test_signed_optimizer_transport_has_strict_bounded_admission(self):
        policy={'version':'bounded-parallel-fp32-state-v1','concurrency':4}
        with patch('subnet.gpu_service.definitions',return_value=[self.row]):
            value=contract(dict(source_bundle={},heldout=[],optimizer_state_transport=policy),0)
            self.assertEqual(value['optimizer_state_transport'],policy)
            policy['concurrency']=2
            self.assertEqual(value['optimizer_state_transport']['concurrency'],4)
            for bad in (True,0,5,1.5):
                with self.assertRaises(ValueError):
                    contract(dict(source_bundle={},heldout=[],optimizer_state_transport=dict(policy,concurrency=bad)),0)
            with self.assertRaises(ValueError):
                contract(dict(source_bundle={},heldout=[],optimizer_state_transport=dict(policy,extra=True)),0)
            self.assertNotIn('optimizer_state_transport',contract(dict(source_bundle={},heldout=[]),0))
    def test_readback_budget_requires_qualified_remote_publication(self):
        from subnet.remote_optimizer_readback import STREAM_BUDGET_VERSION
        from subnet.persistent_publication import VERSION
        budget=dict(version=STREAM_BUDGET_VERSION,concurrency=8,ram_reserve_bytes=1024**3)
        with patch('subnet.gpu_service.definitions',return_value=[self.row]):
            with self.assertRaisesRegex(ValueError,'qualified independent'):
                contract(dict(source_bundle={},heldout=[],independent_state_readback_budget=budget),0)
            policy=dict(version=VERSION,state_readback='local-full',checkpoint_readback_workers=4)
            with self.assertRaisesRegex(ValueError,'qualified independent'):
                contract(dict(source_bundle={},heldout=[],independent_state_readback_budget=budget,persistent_publication_policy=policy),0)
            self.assertNotIn('independent_state_readback_budget',contract(dict(source_bundle={},heldout=[]),0))

    def report(self):
        return dict(job_id='job',completed_at=77,runtime_versions={'torch':'approved'},source_files={n:'approved' for n in ('subnet/model.py','subnet/gpu_runtime.py','subnet/environments.py','subnet/harness.py','subnet/proofs.py')},heldout=[dict(env_id='env',index=i,seed=100+i*1000,task_hash=str(i)*64,verified=True,reward=0,classification='negative') for i in [2,3]])
    def test_evaluation_uses_actual_hashes_and_worker_completion_time(self):
        with tempfile.TemporaryDirectory() as d,patch('subnet.gpu_service.definitions',return_value=[self.row]):
            self.config['evaluation_state']=d;controller=SimpleNamespace(jobs=SimpleNamespace(run=Mock(return_value=self.report())))
            record=evaluate(controller,self.manifest,'remote','before',0,self.config)[0]
            self.assertEqual(record['timestamp'],77);self.assertEqual(record['fixed_task_ids'],['2'*64,'3'*64]);self.assertEqual(record['completed_count'],2)
    def test_prospective_eval_label_is_configured_without_relabeling_old_defaults(self):
        with tempfile.TemporaryDirectory() as d,patch('subnet.gpu_service.definitions',return_value=[self.row]):
            self.config.update(evaluation_state=d,evaluation_experiment_id='wide-fixed16-autoregressive256')
            controller=SimpleNamespace(jobs=SimpleNamespace(run=Mock(return_value=self.report())))
            record=evaluate(controller,self.manifest,'remote','before',0,self.config)[0]
            self.assertEqual(record['experiment_id'],'wide-fixed16-autoregressive256')
    def test_equal_count_wrong_seed_is_rejected(self):
        report=self.report();report['heldout'][0]['seed']+=1
        with tempfile.TemporaryDirectory() as d,patch('subnet.gpu_service.definitions',return_value=[self.row]):
            self.config['evaluation_state']=d;controller=SimpleNamespace(jobs=SimpleNamespace(run=Mock(return_value=report)))
            with self.assertRaisesRegex(ValueError,'exact plan'):evaluate(controller,self.manifest,'remote','before',0,self.config)
    def test_failure_is_not_scored_as_completed_zero(self):
        report=self.report();failed=report['heldout'].pop();report['heldout_failures']=[dict(env_id='env',index=failed['index'],seed=failed['seed'],error='failed to execute')]
        with tempfile.TemporaryDirectory() as d,patch('subnet.gpu_service.definitions',return_value=[self.row]):
            self.config['evaluation_state']=d;controller=SimpleNamespace(jobs=SimpleNamespace(run=Mock(return_value=report)))
            record=evaluate(controller,self.manifest,'remote','before',0,self.config)[0]
            self.assertEqual(record['completed_count'],1);self.assertEqual(record['requested_count'],2);self.assertEqual(record['status'],'error');self.assertIsNone(record['mean_reward'])

class GPUHistoryCompletion(unittest.TestCase):
    def test_failed_history_retains_after_phase_and_retry_does_not_train_again(self):
        with tempfile.TemporaryDirectory() as d:
            state=Path(d);epoch='nonpayable-history-retry'
            old=dict(id='old',files={});new=dict(id='new',files={})
            status=dict(active=dict(epoch=epoch,phase='after',next_checkpoint=new,next_path='new-path',next_steps=2),round=1,training_steps=1,checkpoint=old,checkpoint_path='old-path',initial_published=True)
            for name,value in [('controller.json',status),(epoch+'-manifest.json',dict(epoch=epoch,checkpoint=old)),(epoch+'-verified.json',{}),(epoch+'-scores.json',dict(weights={})),('finalized-reports.json',[])]:
                (state/name).write_text(json.dumps(value))
            config=dict(state=d,bucket={},remote={},source_bundle={},epoch_prefix='nonpayable-history',registration_allowlist=[])
            controller=SimpleNamespace(train=Mock(),signed=lambda value:value)
            with patch('subnet.gpu_service.Bucket'),patch('subnet.gpu_service.Gateway'),patch('subnet.gpu_service.RemoteController',return_value=controller),patch('subnet.gpu_service.ChainAdapter'),patch('subnet.gpu_service.evaluate') as evaluate_after,patch('subnet.gpu_service.publish_history',side_effect=KeyError('key')),patch('subnet.gpu_service.log.exception'):
                with self.assertRaises(KeyError):run(config,once=True)
                evaluate_after.assert_called_once()
            failed=json.loads((state/'controller.json').read_text())
            self.assertEqual(failed,status)
            self.assertEqual(json.loads((state/'health.json').read_text())['status'],'error_retry')
            with patch('subnet.gpu_service.Bucket'),patch('subnet.gpu_service.Gateway'),patch('subnet.gpu_service.RemoteController',return_value=controller),patch('subnet.gpu_service.ChainAdapter'),patch('subnet.gpu_service.evaluate'),patch('subnet.gpu_service.publish_history') as publish:
                run(config,once=True);publish.assert_called_once()
            final=json.loads((state/'controller.json').read_text())
            self.assertIsNone(final['active']);self.assertEqual(final['round'],2)
            self.assertEqual(final['checkpoint'],new);self.assertEqual(final['training_steps'],2)
            controller.train.assert_not_called()

if __name__=='__main__':unittest.main()
