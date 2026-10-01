import unittest,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from subnet.native_prolog_session import NativePrologSession,VERSION
from subnet.native_prolog_actor import REVISION,BASE,SHIM_SHA
class Tests(unittest.TestCase):
 def spec(self):
  return SimpleNamespace(id='affine_prolog',adapter='prime_v1',num_samples=1,max_turns=2,success_reward=1.,config={'prolog_session_revision':VERSION,'prolog_runtime':{'revision':REVISION,'base_image':BASE,'shim_sha256':SHIM_SHA,'image':'sha256:'+'a'*64},'prolog_source_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path('subnet/native_prolog_actor.py'),Path('subnet/native_prolog_session.py')]}})
 def test_source_substitution_rejected(self):
  spec=self.spec();spec.config['prolog_source_files']['subnet/native_prolog_session.py']='f'*64
  with self.assertRaisesRegex(ValueError,'source pin'):NativePrologSession(spec)
 def test_wrong_original_kind_rejected(self):
  with patch('subnet.environments._taskset',return_value=[SimpleNamespace(data=SimpleNamespace(kind='sudoku'))]):
   with self.assertRaisesRegex(ValueError,'task selection'):NativePrologSession(self.spec())
 def session(self):
  session=object.__new__(NativePrologSession);session.actor=Mock();session.done=False;session.turns=0;session.task=object();session.spec=SimpleNamespace(max_turns=2,success_reward=1.)
  session.actor.shell.return_value={'exit_code':0,'stdout':'public','stderr':''};return session
 def test_unknown_tool_rejected_and_owned_actor_closed(self):
  s=self.session();actor=s.actor
  with self.assertRaises(ValueError):s.step({'tool_calls':[{'name':'foreign','arguments':{'command':'x'}}]})
  actor.close.assert_called_once();self.assertIsNone(s.actor)
 def test_native_terminal_grade_not_called_during_tool_turn(self):
  s=self.session()
  async def grade(task,actor):return {'reward':1.,'original_grader_info':{'private':'never expose'}}
  with patch('subnet.native_prolog_session.grade_original',side_effect=grade)as grading:
   step=s.step({'tool_calls':[{'name':'bash','arguments':{'command':'x'}}]});self.assertFalse(step['done']);grading.assert_not_called()
   terminal=s.step({'text':'Done'});self.assertEqual(terminal['classification'],'positive');self.assertNotIn('private',json.dumps(terminal));grading.assert_called_once()
  with self.assertRaises(ValueError):s.step({'text':'again'})
 def test_nonfinite_native_reward_rejected(self):
  s=self.session();actor=s.actor
  async def grade(task,actor):return {'reward':float('nan')}
  with patch('subnet.native_prolog_session.grade_original',side_effect=grade):
   with self.assertRaises(ValueError):s.step({'text':'Done'})
  actor.close.assert_called_once()
if __name__=='__main__':unittest.main()
