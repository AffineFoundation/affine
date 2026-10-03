"""Native grader faults never manufacture a negative or discard earlier samples."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from verifiers.v1.errors import TaskError
from subnet.miner import Miner

class TaskErrors(unittest.TestCase):
    def search(self,events):
        count=len(events)
        class Runtime:
            spec=SimpleNamespace(version='native-test')
            def for_environment(self,*_):return self
            def rollout(self,index,seed):
                event=events.pop(0)
                if isinstance(event,Exception):raise event
                return dict(classification=event,turns=[event]),[]
        actor=Miner.__new__(Miner);actor.manifest=dict(epoch='test',checkpoint={'id':'test'},K=1,L=1,deadline=time.time()+60)
        actor.runtime=Runtime();actor.runtimes={};actor.batches=[];actor.checkpoint='unused'
        def run():
            with patch('subnet.miner.entry',return_value={'env_id':'native','spec':{}}),patch('subnet.miner.harness_for',return_value={}),patch('subnet.miner.classification',side_effect=lambda r:r['classification']),patch('subnet.miner.pack',return_value=b'test'),patch('subnet.miner.for_manifest',return_value={}):
                return actor.search(1,seed=3,max_attempts=count,env_id='native')
        return actor,run
    def test_real_success_then_grader_error_then_real_failure_retains_pair(self):
        actor,run=self.search(['positive',TaskError('unscorable'),'negative']);run()
        self.assertEqual([r['classification'] for r in actor.batches[0][0]['rollouts']],['positive','negative'])
        self.assertEqual(actor.last_generation_error['kind'],'unscorable_native_task_error')
    def test_grader_error_cannot_count_as_negative(self):
        actor,run=self.search([TaskError('unscorable'),'positive'])
        with self.assertRaisesRegex(RuntimeError,'search budget exhausted'):run()
        self.assertEqual(actor.batches,[])
    def test_model_assertion_is_not_swallowed(self):
        actor,run=self.search([ValueError('model assertion')])
        with self.assertRaisesRegex(ValueError,'model assertion'):run()
        self.assertEqual(actor.batches,[])

if __name__=='__main__':unittest.main()
