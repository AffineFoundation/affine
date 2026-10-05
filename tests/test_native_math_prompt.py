import json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from subnet.environments import build_spec,create_session
from subnet.native_math_prompt import NativeMathPromptSession

class NativeMathPromptControls(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  root=Path(__file__).resolve().parents[1]/'subnet/vendor/legacy/rollouts/envs/affine_math_v1';sys.path.insert(0,str(root))
  from affine_math_v1.taskset import MathData,MathConfig,SYSTEM
  self.path=Path(self.temp.name)/'tasks.json';self.rows=[]
  for i in range(3):
   data=MathData(idx=i,name='math-control-'+str(i),system_prompt=SYSTEM,prompt='Compute '+str(i)+'+2.',problem='Compute '+str(i)+'+2.',answer=str(i+2),subject='Algebra',level='1')
   self.rows.append(dict(task_class='MathTask',data=data.model_dump(mode='json'),task_config=MathConfig().task.model_dump(mode='json')))
  self.path.write_text(json.dumps(self.rows));self.spec=build_spec('affine_math',{'task_snapshot':str(self.path)},num_samples=3,max_turns=1)
 def test_actual_native_messages_task_hash_and_tools_equivalence(self):
  normal=create_session(self.spec);fast=NativeMathPromptSession(self.spec)
  try:
   for i in range(3):self.assertEqual(fast.reset(i,17),normal.reset(i,17))
  finally:normal.close();fast.close()
 def test_no_runtime_setup_grader_or_model_calls_and_snapshot_loaded_once(self):
  from affine_math_v1.taskset import MathTask
  with patch.object(MathTask,'setup',side_effect=AssertionError('eligibility does not grade')):
   fast=NativeMathPromptSession(self.spec)
   try:
    with patch('subnet.native_math_prompt.json.loads',side_effect=AssertionError('no repeated parse')):
     for i in range(3):self.assertTrue(fast.reset(i,9)['task_hash'])
   finally:fast.close()
 def test_snapshot_change_after_admission_rejects_every_cached_row(self):
  fast=NativeMathPromptSession(self.spec)
  try:
   self.path.write_text(json.dumps(list(reversed(self.rows))))
   with self.assertRaisesRegex(ValueError,'snapshot changed'):fast.reset(0,0)
  finally:fast.close()
 def test_source_or_snapshot_tampering_before_admission_fails_closed(self):
  from dataclasses import replace
  with self.assertRaisesRegex(ValueError,'hash mismatch'):NativeMathPromptSession(replace(self.spec,source_hash='0'*64))
  self.rows[0]['data']['prompt']='foreign task';self.path.write_text(json.dumps(self.rows))
  with self.assertRaisesRegex(ValueError,'hash mismatch'):NativeMathPromptSession(self.spec)
 def test_fresh_authenticated_loader_replaces_preloaded_prompt_module(self):
  import subprocess
  from test_persistent_publication_bootstrap import SCRIPT
  script=SCRIPT.replace('persistent_publication','native_math_prompt')
  for mode in ('reload','runtime-preload','undeclared','bad-hash'):
   with self.subTest(mode=mode):
    result=subprocess.run([sys.executable,'-B','-c',script,mode],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,0,result.stderr)
 def test_wrong_environment_multi_turn_index_taskclass_and_closed_session_reject(self):
  from dataclasses import replace
  for spec in [replace(self.spec,id='affine_when2call'),replace(self.spec,max_turns=2)]:
   with self.assertRaisesRegex(ValueError,'one-turn'):NativeMathPromptSession(spec)
  fast=NativeMathPromptSession(self.spec)
  try:
   for index in [True,-1,3]:
    with self.assertRaisesRegex(ValueError,'index'):fast.reset(index,0)
   fast.rows[0]['task_class']='ArbitraryTask'
   with self.assertRaisesRegex(ValueError,'task class'):fast.reset(0,0)
  finally:fast.close()
  with self.assertRaisesRegex(ValueError,'closed'):fast.reset(0,0)
if __name__=='__main__':unittest.main()
