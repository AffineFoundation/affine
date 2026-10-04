import importlib.util,json,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import subnet.miner as m
from verifiers.v1.errors import TaskError
class ProgressTests(unittest.TestCase):
 def actor(self,root,events,callback=None):
  class Runtime:
   spec=SimpleNamespace(version='test')
   sampled_ordinals=[]
   def for_environment(self,*args):return self
   def rollout(self,index,seed):
    self.sampled_ordinals.append(seed)
    event=events.pop(0)
    if callback:callback()
    if isinstance(event,Exception):raise event
    return dict(classification=event,turns=['DO_NOT_LOG_TOKENS_OR_KEYS_'+event]),[]
  a=m.Miner.__new__(m.Miner);a.manifest=dict(epoch='test-telemetry',checkpoint={'id':'approved-checkpoint'},source_bundle={'sha256':'approved-source'},K=1,L=1,deadline=time.time()+60);a.runtime=Runtime();a.runtimes={};a.batches=[];a.checkpoint='unused';a.progress=m.MinerProgress(root/'progress.json',a.manifest);return a
 def run_search(self,a,count):
  with patch.object(m,'entry',return_value={'env_id':'math','spec':{}}),patch.object(m,'harness_for',return_value={}),patch.object(m,'classification',side_effect=lambda r:r['classification']),patch.object(m,'pack',return_value=b'test'),patch.object(m,'for_manifest',return_value={}):return a.search(1,seed=3,max_attempts=count,env_id='math')
 def test_pair_with_indeterminate_counts_and_no_sensitive_payload(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=self.actor(root,['positive',TaskError('DO_NOT_LOG_ERROR_DETAIL'),'negative']);self.run_search(a,3);text=(root/'progress.json').read_text();v=json.loads(text)
   self.assertEqual(a.runtime.sampled_ordinals,[3,4,5]);self.assertEqual(v['completed_batch_count'],1);self.assertEqual(v['counts']['attempts_started'],3);self.assertEqual(v['counts']['attempts_ended'],3);self.assertEqual(v['counts']['completed_rollouts'],2);self.assertEqual(v['counts']['indeterminate'],1);self.assertEqual(v['counts']['tasks_ended'],1);self.assertGreaterEqual(v['task_elapsed_seconds'],0);self.assertGreaterEqual(v['cumulative_generation_elapsed_seconds'],0);self.assertNotIn('DO_NOT_LOG',text);self.assertEqual((root/'progress.json').stat().st_mode&0o077,0)
 def test_runtime_error_observed_without_semantic_change(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=self.actor(root,[RuntimeError('SECRET_DETAIL')]);
   with self.assertRaisesRegex(RuntimeError,'SECRET_DETAIL'):self.run_search(a,1)
   text=(root/'progress.json').read_text();v=json.loads(text);self.assertEqual(v['counts']['error'],1);self.assertEqual(v['completed_batch_count'],0);self.assertNotIn('SECRET_DETAIL',text)
 def test_completed_after_deadline_remains_unaccepted(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);now=[100.0];a=self.actor(root,['positive'],callback=lambda:now.__setitem__(0,102.0));a.manifest['deadline']=101.0
   with patch.object(m.time,'time',side_effect=lambda:now[0]),self.assertRaises(m.EpochClosed):self.run_search(a,1)
   v=json.loads((root/'progress.json').read_text());self.assertEqual(v['counts']['deadline'],1);self.assertEqual(v['counts']['completed_rollouts'],0);self.assertEqual(v['completed_batch_count'],0);self.assertEqual(a.batches,[])
 def test_no_output_path_failure_can_change_pair_acceptance(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=self.actor(root,['positive','negative']);a.progress=m.MinerProgress(Path('/dev/null/unwritable.json'),a.manifest);self.run_search(a,2);self.assertEqual(len(a.batches),1)
 def test_task_exhaustion_is_bounded_sidecar(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=self.actor(root,['negative']*16)
   with self.assertRaisesRegex(RuntimeError,'budget exhausted'):self.run_search(a,16)
   v=json.loads((root/'progress.json').read_text());self.assertEqual(v['event'],'task_exhausted');self.assertEqual(v['counts']['attempts_started'],16);self.assertEqual(v['counts']['negative'],16);self.assertEqual(v['counts']['tasks_ended'],1);self.assertLess((root/'progress.json').stat().st_size,1600)
 def test_start_timestamp_immutable_and_generation_duration_cumulative(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=self.actor(root,['positive']);start=a.progress.value['started_at']
   for ordinal,kind in [(3,'positive'),(4,'indeterminate'),(5,'error'),(6,'deadline')]:
    a.progress.record('attempt_start',env_id='math',index=1,attempt=ordinal)
    a.progress.record('attempt_end',env_id='math',index=1,attempt=ordinal,outcome=kind,elapsed=1.25)
   v=json.loads((root/'progress.json').read_text());self.assertEqual(v['started_at'],start);self.assertEqual(v['cumulative_generation_elapsed_seconds'],5.0);self.assertEqual(v['counts']['attempts_ended'],4)
   a.progress.record('attempt_end',attempt=7,outcome='error',elapsed=float('nan'))
   v=json.loads((root/'progress.json').read_text());self.assertEqual(v['cumulative_generation_elapsed_seconds'],5.0)
 def test_cli_progress_namespace_binds_epoch_and_identity(self):
  text=Path(m.__file__).with_name('cli.py').read_text();self.assertIn('progress_path=Path(a.state)/f"{manifest[\'epoch\']}-{key.id}.progress.json"',text)
if __name__=='__main__':unittest.main()
