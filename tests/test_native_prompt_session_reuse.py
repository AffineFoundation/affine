import copy,unittest
from unittest.mock import patch
from types import SimpleNamespace
from subnet.committed_training_inputs import validate_native_prompt
class NativeReuse(unittest.TestCase):
 def setUp(self):
  self.runtime=SimpleNamespace(tokenizer=object(),model=SimpleNamespace(config=SimpleNamespace(vocab_size=10)))
  spec={'id':'affine_math','max_turns':1,'max_output_tokens':64,'source_hash':'a'*64,'config':{'seed':7}}
  self.definition={'spec':spec};rollout={'index':0,'task_hash':'task0','env_seed':7,'turns':[{'prompt':[1,2],'output':[3]}]};self.pairs=[(self.definition,rollout,copy.deepcopy(rollout))];self.sessions=[]
 def create(self,spec):
  s=SimpleNamespace(resets=[],closed=False)
  def reset(index,seed):s.resets.append((index,seed));return {'task_hash':'task'+str(index),'messages':[],'tools':[]}
  s.reset=reset;s.close=lambda:setattr(s,'closed',True);self.sessions.append(s);return s
 def run_pairs(self,pairs):
  with patch('subnet.native_math_prompt.NativeMathPromptSession',side_effect=self.create),patch('subnet.protocol.harness_for',return_value={'max_output_tokens':64}),patch('subnet.harness.render',return_value=[1,2]):validate_native_prompt(self.runtime,pairs,{'checkpoint':{'id':'cp'}},prompt_only=True)
 def test_historical_unpinned_path_retains_original_session(self):
  with patch('subnet.environments.create_session',side_effect=self.create),patch('subnet.native_math_prompt.NativeMathPromptSession',side_effect=AssertionError('historical contract must not substitute adapter')),patch('subnet.protocol.harness_for',return_value={'max_output_tokens':64}),patch('subnet.harness.render',return_value=[1,2]):
   validate_native_prompt(self.runtime,self.pairs,{})
  self.assertEqual(len(self.sessions),1)
 def test_same_exact_spec_constructed_once_each_task_reset(self):
  second=copy.deepcopy(self.pairs[0])
  for rollout in second[1:]:rollout.update(index=1,task_hash='task1')
  self.run_pairs(self.pairs+[second]);self.assertEqual(len(self.sessions),1);self.assertEqual(self.sessions[0].resets,[(0,7),(1,7)]);self.assertTrue(self.sessions[0].closed)
 def test_different_source_spec_not_reused(self):
  second=copy.deepcopy(self.pairs[0]);second[0]['spec']['source_hash']='b'*64;self.run_pairs(self.pairs+[second]);self.assertEqual(len(self.sessions),2);self.assertTrue(all(s.closed for s in self.sessions))
 def test_changed_prompt_rejected_and_session_closed(self):
  self.pairs[0][1]['turns'][0]['prompt']=[9]
  with self.assertRaisesRegex(ValueError,'eligibility'):self.run_pairs(self.pairs)
  self.assertTrue(self.sessions[0].closed)
 def test_cross_task_negative_rejected_without_model_or_grading(self):
  self.pairs[0][2]['task_hash']='foreign'
  with self.assertRaisesRegex(ValueError,'eligibility'):self.run_pairs(self.pairs)
  self.assertTrue(self.sessions[0].closed)
 def test_reset_failure_closes_already_constructed_session(self):
  def create(spec):s=self.create(spec);s.reset=lambda *a:(_ for _ in ()).throw(RuntimeError('native failure'));return s
  with patch('subnet.native_math_prompt.NativeMathPromptSession',side_effect=create):
   with self.assertRaises(RuntimeError):validate_native_prompt(self.runtime,self.pairs,{},prompt_only=True)
  self.assertTrue(self.sessions[0].closed)
if __name__=='__main__':unittest.main()
