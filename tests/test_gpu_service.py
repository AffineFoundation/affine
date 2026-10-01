import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from subnet.gpu_service import heldout,evaluate,contract

class GPUFixedHeldout(unittest.TestCase):
    def setUp(self):
        self.row={'spec':{'id':'env','num_samples':4,'max_output_tokens':512,'version':'fixed-v1'},'indices':[0,1],'harness':{'version':'text-tools-v1'}}
        self.config={'environments':[], 'heldout':[dict(env_id='env',indices=[2,3],seed=100,harness=dict(version='text-tools-v1',policy='autoregressive',max_output_tokens=128))]}
        self.manifest={'epoch':'nonpayable-gpu-test','checkpoint':{'id':'approved'},'environments':[dict(env_id='env',**self.row)],'harness_source_hash':'pinned'}
    def test_inactive_epoch_group_cannot_train_on_fixed_heldout(self):
        self.manifest['environments'][0]['indices']=[];self.config['heldout'][0]['indices']=[1]
        with patch('subnet.gpu_service.definitions',return_value=[self.row]),self.assertRaisesRegex(ValueError,'fixed heldout binding'):heldout(self.config,self.manifest)
    def report(self):
        return dict(job_id='job',completed_at=77,runtime_versions={'torch':'approved'},source_files={n:'approved' for n in ('subnet/model.py','subnet/gpu_runtime.py','subnet/environments.py','subnet/harness.py','subnet/proofs.py')},heldout=[dict(env_id='env',index=i,seed=100+i*1000,task_hash=str(i)*64,verified=True,reward=0,classification='negative') for i in [2,3]])
    def test_evaluation_uses_actual_hashes_and_worker_completion_time(self):
        with tempfile.TemporaryDirectory() as d,patch('subnet.gpu_service.definitions',return_value=[self.row]):
            self.config['evaluation_state']=d;controller=SimpleNamespace(jobs=SimpleNamespace(run=Mock(return_value=self.report())))
            record=evaluate(controller,self.manifest,'remote','before',0,self.config)[0]
            self.assertEqual(record['timestamp'],77);self.assertEqual(record['fixed_task_ids'],['2'*64,'3'*64]);self.assertEqual(record['completed_count'],2)
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

if __name__=='__main__':unittest.main()
