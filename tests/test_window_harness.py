import json
import unittest
import copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from subnet.harness import normalize,render


class CaptureTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        self.messages=messages
        return list(json.dumps(messages).encode())


class WindowHarnessTests(unittest.TestCase):
    def test_original_task_and_latest_complete_observation_survive(self):
        messages=[dict(role='system',content='rules'),dict(role='user',content='task'),
                  dict(role='assistant',content='old action'),dict(role='user',content='old observation'),
                  dict(role='assistant',content='latest action'),dict(role='user',content='latest observation')]
        original=json.dumps(messages);tokenizer=CaptureTokenizer()
        render(tokenizer,messages,config={'version':'text-tools-window-v1'})
        self.assertEqual([v['content']for v in tokenizer.messages],['rules','task','latest action','latest observation'])
        self.assertEqual(json.dumps(messages),original)

    def test_individual_large_observation_is_preserved_without_silent_truncation(self):
        tokenizer=CaptureTokenizer();large='x'*10000
        render(tokenizer,[dict(role='user',content='task'),dict(role='assistant',content='action'),dict(role='user',content=large)],config={'version':'text-tools-window-v1'})
        self.assertEqual(tokenizer.messages[-1]['content'],large)

    def test_policy_parameters_cannot_silently_change_legacy_harness(self):
        with self.assertRaisesRegex(ValueError,'versioned'):
            normalize({'version':'text-tools-v1','history_window_messages':2})
        for invalid in [0,33,True,2.5]:
            with self.assertRaisesRegex(ValueError,'bounds'):
                normalize({'version':'text-tools-window-v1','history_window_messages':invalid})

    def test_early_turns_keep_every_message_without_duplicate_prefix(self):
        messages=[dict(role='system',content='rules'),dict(role='user',content='task')]
        tokenizer=CaptureTokenizer();render(tokenizer,messages,config={'version':'text-tools-window-v1'})
        self.assertEqual(tokenizer.messages,messages)

        messages+= [dict(role='assistant',content='action'),dict(role='user',content='observation')]
        render(tokenizer,messages,config={'version':'text-tools-window-v1'})
        self.assertEqual(tokenizer.messages,messages)

    def test_discarded_old_observation_still_requires_native_replay(self):
        from subnet.model import Runtime
        from subnet.harness import observations
        class NativeSession:
            def reset(self,index,seed):
                self.turn=0
                return dict(messages=[dict(role='system',content='rules'),dict(role='user',content='task')],tools=[],task_hash='original')
            def step(self,action):
                self.turn+=1;done=self.turn==3
                return dict(observations=[dict(role='user',content=f'native observation {self.turn}')],
                            done=done,reward=float(done),classification='positive' if done else 'negative')
            def close(self):pass
        runtime=Runtime.__new__(Runtime);runtime.spec=SimpleNamespace(id='native',version='v1',config={},max_turns=3,max_output_tokens=8)
        runtime.legacy=False;runtime.harness=normalize({'version':'text-tools-window-v1'});runtime.tokenizer=CaptureTokenizer()
        runtime.tokenizer.decode=lambda output,**kwargs:'A'
        runtime.model=SimpleNamespace(config=SimpleNamespace(vocab_size=256,max_position_embeddings=8192))
        runtime.compute=lambda prompt,output:([],np.zeros((1,256),dtype=np.float32))
        runtime.verify_proofs=lambda *args,**kwargs:[SimpleNamespace(exp_mismatches=0,mant_err_mean=0,mant_err_median=0)]*2
        native=NativeSession();messages=native.reset(0,0)['messages'];turns=[]
        for _ in range(3):
            prompt=runtime.prompt(messages);result=native.step({'text':'A'})
            turns.append(dict(prompt=prompt,output=[65],text='A',proofs=['framing tested separately'],**result))
            messages+= [dict(role='assistant',content='A')]+observations(result['observations'],runtime.harness)
        rollout=dict(schema=2,env_id='native',environment_version='v1',index=0,sample_index=0,env_seed=0,
                     task_hash='original',reward=1.,turns=turns)
        arrays=[np.zeros((1,256),dtype=np.float32)]*3
        with patch('subnet.model.create_session',side_effect=lambda _:NativeSession()),patch('subnet.model.validate_framing'):
            self.assertTrue(runtime.verify(rollout,arrays))
            forged=copy.deepcopy(rollout);forged['turns'][0]['observations'][0]['content']='forged old observation'
            with self.assertRaisesRegex(ValueError,'environment replay'):runtime.verify(forged,arrays)


if __name__=='__main__':unittest.main()
