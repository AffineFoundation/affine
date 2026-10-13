"""Channel adversaries exercise the shared miner/native/audit grading helper."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import math
import base64

import pytest

from subnet.mimo_answer_projection import POLICY, POLICY_KEY, project, grade
from subnet.complete_group_distribution import MIMO_EOS
from subnet.complete_group_runtime import outcome


class Tokenizer:
    eos_token_id=248044
    chat_template='explicit CPU fixture template'
    all_special_ids = [248045, *MIMO_EOS]
    def __len__(self):return 248320
    def apply_chat_template(self,*args,**kwargs):return [1,2]
    def convert_tokens_to_ids(self, text):
        return {'<think>':248068,'</think>':248069}[text]
    def encode(self, text):
        # Explicit fixture tokenizer: markers are one token, ordinary text chars.
        result=[]
        while text:
            marker=next((m for m in ('<think>','</think>') if text.startswith(m)),None)
            if marker:
                result.append(self.convert_tokens_to_ids(marker));text=text[len(marker):]
            else:
                result.append(ord(text[0]));text=text[1:]
        return result+[MIMO_EOS[0]]
    def decode(self, tokens, skip_special_tokens=False):
        mapping={248068:'<think>',248069:'</think>',248045:'<|im_start|>',
                 248046:'<|im_end|>',248044:'<|endoftext|>'}
        return ''.join(mapping[t] if t in mapping else chr(t) for t in tokens
                       if not(skip_special_tokens and t in self.all_special_ids))


def setup():
    from test_complete_group_mimo import mimo_manifest
    m=mimo_manifest();m[POLICY_KEY]=deepcopy(POLICY)
    return m,m['environments'][0]['harness'],Tokenizer()


class Native:
    def __init__(self):self.actions=[]
    def step(self, action):
        self.actions.append(action)
        good=action['text']==r'\boxed{1}'
        return dict(classification='positive' if good else 'negative',reward=float(good),
                    done=True,observations=[])


@pytest.mark.parametrize('reasoning,answer,expected',[
    (r'Correct: \boxed{1}',r'\boxed{0}','negative'),
    (r'Wrong: \boxed{0}',r'\boxed{1}','positive'),
    (r'Answer: \boxed{(3,-1)}',r'\boxed{(3,-1)}','negative'),
])
def test_only_final_answer_is_native_authority(reasoning,answer,expected):
    m,h,t=setup();output=t.encode('<think>'+reasoning+'</think>'+answer)
    original=list(output);session=Native()
    native,kind,reward=grade(session,m,h,t,output,'eos')
    assert kind==expected and reward==float(expected=='positive')
    assert session.actions[0]['text']==answer
    assert output==original  # Full thought+final trajectory is never rewritten.


@pytest.mark.parametrize('final',[
    r'\boxed{1}',r'Answer: \boxed{1}',r'\[\boxed{1}\]',
    r'\(\boxed{1}\)',r'$\boxed{1}$',r'$$\boxed{1}$$',
])
def test_complete_paired_wrappers_are_accepted(final):
    m,h,t=setup();output=t.encode('<think>analysis</think>'+final)
    native,kind,reward=grade(Native(),m,h,t,output,'eos')
    assert kind=='positive' and reward==1


@pytest.mark.parametrize('text',[
    r'<think>\boxed{1}',r'<think>\boxed{1}</think>',r'\boxed{1}',
    r'</think><think>bad\boxed{1}',r'<think>a<think>b</think>\boxed{1}',
    r'<think>a</think></think>\boxed{1}',r'prefix<think>a</think>\boxed{1}',
    r'<think>a</think>\boxed{1}\boxed{0}',r'<think>a</think>\boxed{1} unfinished',
    r'<think>a</think>\boxed{}',r'<think>a</think>\boxed{1',
    r'<think>a</think>\boxed{1}\]',r'<think>a</think>\[\boxed{1}',
])
def test_missing_or_malformed_final_is_unresolved_not_negative(text):
    m,h,t=setup();native,kind,reward=grade(Native(),m,h,t,t.encode(text),'eos')
    assert kind=='neutral' and reward==0


def test_budget_cap_control_tokens_and_forged_marker_text_are_unresolved():
    m,h,t=setup();tokens=t.encode(r'<think>x</think>\boxed{1}')
    assert not project(m,h,t,tokens,'budget')['complete']
    assert not project(m,h,t,tokens[:-1],'eos')['complete']
    assert not project(m,h,t,[248045]+tokens,'eos')['complete']
    # A marker spelling split into ordinary tokens is not a second channel.
    tokens=t.encode('<think>x</think>')[:-1]+list(map(ord,r'<think></think>\boxed{1}'))+[MIMO_EOS[0]]
    assert not project(m,h,t,tokens,'eos')['complete']


def test_no_thinking_mode_and_legacy_manifests_are_separate():
    m,h,t=setup();h['enable_thinking']=False
    assert grade(Native(),m,h,t,t.encode(r'\boxed{1}'),'eos')[1]=='positive'
    assert grade(Native(),m,h,t,t.encode(r'<think>x</think>\boxed{1}'),'eos')[1]=='neutral'
    # Without the new signed rule, two boxes retain the old unresolved result.
    m.pop(POLICY_KEY)
    assert grade(Native(),m,h,t,t.encode(r'<think>\boxed{1}</think>\boxed{1}'),'eos')[1]=='neutral'


def test_native_infrastructure_error_is_not_converted_to_negative():
    m,h,t=setup()
    with pytest.raises(OSError):
        grade(SimpleNamespace(step=lambda a:(_ for _ in ()).throw(OSError('grader down'))),
              m,h,t,t.encode(r'<think>x</think>\boxed{1}'),'eos')


def test_wrong_tokenizer_binding_fails_closed():
    m,h,t=setup();t.convert_tokens_to_ids=lambda _:1
    with pytest.raises(ValueError,match='tokenizer'):
        project(m,h,t,[1,MIMO_EOS[0]],'eos')


def test_native_batch_and_audit_use_same_final_channel_and_keep_all_tokens():
    """Independent admission and audit cannot use a correct analysis as a label."""
    from test_complete_group_mimo import toy_runtime
    from subnet import forced_sampling
    from subnet.complete_group_sampling import digest
    from subnet.complete_group_mining import _rollout_from_runtime,BATCH_VERSION
    from subnet.complete_group_native import grade_batch
    from subnet.complete_group_verification import native_projection
    m,h,t=setup();h['max_context_tokens']=512;h['max_output_tokens']=128
    m['group_transport_policy']['max_context_tokens']=512
    m['environments'][0]['spec']['max_output_tokens']=128
    m['sampling_contract']['calibration']['harness_sha256']=digest(h)
    r=toy_runtime(m,'a'*64);r.tokenizer=t;r.model.config.text_config.max_position_embeddings=512
    r.prompt=lambda *args:[1,2]
    class Session(Native):
        def reset(self,*args):return dict(messages=[dict(role='user',content='fixture')],task_hash='b'*64)
        def close(self):pass
    rows=[]
    for nonce in range(8):
        output=t.encode(r'<think>\boxed{'+str(nonce%2)+r'}</think>\[\boxed{'+str(1-nonce%2)+r'}\]')
        native,kind,reward=grade(Session(),m,h,t,output,'eos')
        r.complete_group_sample_evidence=dict(version='sampler-selected-behavior-logprobs-v1',attempt=nonce,
            turn=0,binding_sha256=digest(r.sampling_context),prompt_sha256=digest([1,2]),output_sha256=digest(output),
            behavior_selected_logprobs=[-1.]*len(output),stop_reason='eos')
        proofs=[base64.b64encode((40001).to_bytes(2,'big')+bytes(256)).decode()]*(1+math.ceil(len(output)/16))
        turn=dict(prompt=[1,2],output=output,text=t.decode(output,skip_special_tokens=True),proofs=proofs,
            observations=[],done=True,reward=reward,classification=kind)
        row=dict(schema=2,env_id='math',environment_version='taskset-1',index=0,sample_index=0,seed=nonce,
            env_seed=0,task_hash='b'*64,reward=reward,classification=kind,turns=[turn],
            sampling=forced_sampling.receipt(r.sampling_context,nonce))
        rows.append(_rollout_from_runtime(r,row,nonce))
    batch=dict(schema=BATCH_VERSION,epoch=m['epoch'],checkpoint=m['checkpoint']['id'],miner='a'*64,
        env_id='math',environment_version='taskset-1',index=0,sample_index=0,task_hash='b'*64,
        group_contract_sha256=digest(m['group_contract']),sampling_context_sha256=digest(r.sampling_context),rollouts=rows)
    with (patch('subnet.complete_group_native.create_session',side_effect=lambda _:Session()),
          patch('subnet.complete_group_verification.create_session',side_effect=lambda _:Session()),
          patch('subnet.complete_group_native.EnvironmentSpec.from_dict',side_effect=lambda v:SimpleNamespace(**v)),
          patch('subnet.mimo_runtime.validate_runtime_spec',return_value=SimpleNamespace(
              chat_template_sha256=hashlib.sha256(t.chat_template.encode()).hexdigest(),enable_thinking=True))):
        receipt=grade_batch(batch,m,t,miner='a'*64)
        assert [x['classification']for x in receipt['results']]==['positive','negative']*4
        for row,result in zip(rows,receipt['results'],strict=True):
            audited=native_projection(r,row)
            assert audited['classification']==result['classification']
            assert result['output']==row['turns'][0]['output']==audited['output']
            assert 248068 in result['output'] and 248069 in result['output']
