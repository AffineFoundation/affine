import hashlib
from types import SimpleNamespace
import unittest
from subnet.native_agent_adapter import NativeAgentAdapter, VERSION
from test_native_agent_isolation import descriptor

def spec():
    desc=descriptor();desc['public_files']['instruction.md']=hashlib.sha256(b'Public instruction').hexdigest()
    return SimpleNamespace(version=VERSION,adapter='native_agent_controlled',num_samples=1,max_turns=4,success_reward=1.,config=dict(dependency_scope='immutable-controlled-images-not-full-upstream-closure',native_tasks=[dict(descriptor=desc,instruction='Public instruction')]))

class Session:
    def __init__(self,*args):self.calls=[]
    def start(self):return dict(messages=[dict(role='user',content='Public instruction')],tools=[])
    def call(self,name,args):self.calls.append((name,args));return 'Unknown tool: missing' if name=='missing' else dict(actual_native_result=1)
    def grade(self):return dict(grade=dict(reward=int(any(n=='submit' for n,a in self.calls))))
    def close(self):pass

class NativeAdapterControls(unittest.TestCase):
    def test_native_error_string_observation_is_not_quoted_or_fatal(self):
        adapter=NativeAgentAdapter(spec(),Session);adapter.reset(0,0)
        result=adapter.step(dict(text='',tool_calls=[dict(name='missing',arguments={})]))
        self.assertEqual(result['observations'][0]['content'],'Unknown tool: missing')
        self.assertFalse(result['done']);self.assertEqual(adapter.step(dict(text='done'))['classification'],'negative')

    def test_original_terminal_grade_and_reset_task_identity_bound(self):
        adapter=NativeAgentAdapter(spec(),Session);initial=adapter.reset(0,1)
        adapter.step(dict(tool_calls=[dict(name='submit',arguments={})]));result=adapter.step(dict(text='done'))
        self.assertEqual(result['reward'],1.);self.assertEqual(result['classification'],'positive')
        other=NativeAgentAdapter(spec(),Session);self.assertNotEqual(initial['task_hash'],other.reset(0,2)['task_hash'])
        with self.assertRaises(ValueError):adapter.step(dict(text='done'))

    def test_unknown_scope_bad_public_pin_and_duplicate_task_fail_before_runtime(self):
        value=spec();value.config['native_tasks'][0]['instruction']='changed'
        with self.assertRaises(ValueError):NativeAgentAdapter(value,Session)
        value=spec();value.config['native_tasks']*=2;value.num_samples=2
        with self.assertRaisesRegex(ValueError,'duplicate'):NativeAgentAdapter(value,Session)
        value=spec();value.config['dependency_scope']='full-upstream'
        with self.assertRaises(ValueError):NativeAgentAdapter(value,Session)

if __name__=='__main__':unittest.main()
