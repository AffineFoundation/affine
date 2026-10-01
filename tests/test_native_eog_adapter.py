import unittest
from types import SimpleNamespace
from unittest.mock import patch
from mcp.server.fastmcp.exceptions import ToolError
from subnet.native_eog_adapter import NativeEOGAdapter,VERSION
from subnet.native_eog_split import sha

class Actor:
 def __init__(self,task,reward=1,corrupt=False,error=None):self.task=task;self.reward=reward;self.corrupt=corrupt;self.error=error;self.events=[]
 def reset(self):return self.task
 def call(self,name,args):
  if self.error:raise self.error
  value='unchanged native observation';self.events.append(dict(name=name,arguments=args,observation=value));return value
 def finish(self):return dict(sealed=True,reward=self.reward,task_id=self.task['task_id'],public_descriptor_sha256=sha(self.task),transcript_sha256='00'*32 if self.corrupt else sha(self.events))
 def close(self):pass

class EOGCommonControls(unittest.TestCase):
 def setup_adapter(self,**options):
  # Native descriptor/privacy validation is exercised by broker tests. This
  # tests the common terminal/history boundary without container fixtures.
  task=dict(task_id='original54',messages=[dict(role='user',content='original instruction')],tools=[])
  spec=SimpleNamespace(adapter='native_eog_broker',version=VERSION,config=dict(dependency_scope='controlled-cohost-public-actor-private-grader',public_tasks=[task]),num_samples=1,max_turns=1,success_reward=1)
  with patch('subnet.native_eog_adapter.validate_public',side_effect=lambda x:x):
   actor=Actor(task,**options);adapter=NativeEOGAdapter(spec,lambda i,s,h:actor);adapter.reset(0,0)
  return adapter
 def action(self):return dict(text='',tool_calls=[dict(name='original_tool',arguments={'public':'argument'})])
 def test_native_terminal_reward_and_exact_observation(self):
  a=self.setup_adapter();r=a.step(self.action());self.assertEqual(r['reward'],1);self.assertEqual(r['classification'],'positive');self.assertEqual(r['observations'][0]['content'],'unchanged native observation')
  with self.assertRaises(ValueError):a.step(self.action())
 def test_forged_terminal_history_rejected(self):
  with self.assertRaises(ValueError):self.setup_adapter(corrupt=True).step(self.action())
 def test_boolean_and_nonfinite_reward_rejected(self):
  for value in [True,float('nan'),float('inf')]:
   with self.subTest(value=value),self.assertRaises(ValueError):self.setup_adapter(reward=value).step(self.action())
 def test_native_toolerror_is_observation_but_infra_failure_propagates(self):
  r=self.setup_adapter(error=ToolError('original MCP unknown tool')).step(self.action());self.assertEqual(r['observations'][0]['content'],'original MCP unknown tool')
  with self.assertRaises(RuntimeError):self.setup_adapter(error=RuntimeError('transport failed')).step(self.action())
