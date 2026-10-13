"""Actual signed evaluation/queue path, CPU scripted logits/native fixtures.

Shared top-k/top-p sampler and final-channel grader are real. This suite is not
a claim of MiMo hardware or learning qualification.
"""
import base64
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from nacl.signing import SigningKey
from test_complete_group_mimo import mimo_manifest
from test_mimo_answer_projection import Tokenizer, Native
from subnet import mimo_evaluation as evaluation, mimo_worker
from subnet.mimo_runtime import MimoRuntimeSpec, MODEL_ID, VERSION, EOS_TOKEN_IDS
from subnet.mimo_answer_projection import POLICY
from subnet.storage import canonical
from subnet.complete_group_sampling import digest


def sign(key, value):
    return dict(payload=deepcopy(value),signer=key.verify_key.encode().hex(),
        signature=base64.b64encode(key.sign(canonical(value)).signature).decode())


def manifest():
    m=mimo_manifest();m.update(epoch='nonpayable-eval-fixture',model_runtime_revision=VERSION,payable=False)
    m['checkpoint']['id']=digest(m['checkpoint']['files'])
    h=m['environments'][0]['harness'];h.update(max_output_tokens=64,max_context_tokens=128)
    m['environments'][0]['spec']['max_output_tokens']=64
    m['group_transport_policy']['max_context_tokens']=128
    m['sampling_contract']['calibration'].update(checkpoint=m['checkpoint']['id'],
        model_runtime_revision=VERSION,harness_sha256=digest(h))
    m['mimo_runtime']=MimoRuntimeSpec(VERSION,MODEL_ID,{'versions':{},'sources':{}},
        hashlib.sha256(Tokenizer.chat_template.encode()).hexdigest(),True,128,EOS_TOKEN_IDS).to_dict()
    m['mimo_answer_projection']=deepcopy(POLICY)
    m['evaluation_sets']=[dict(name='fixed4',env_id='math',indices=[4,5,6,7],seed=318,repeats=1)]
    return m


class Session(Native):
    opened=[]
    def reset(self,index,seed):
        self.index=index;self.opened.append(index)
        return dict(task_hash=digest(['task',index]),messages=[dict(role='user',content=str(index))],tools=[])
    def step(self,action):
        if self.index==7:raise OSError('simulated native service failure; no fabricated reward')
        return super().step(action)
    def close(self):pass


class Runtime:
    def __init__(self,m):
        self.checkpoint_id=m['checkpoint']['id'];self.spec=SimpleNamespace(to_dict=lambda:m['mimo_runtime'])
        self.max_context_tokens=128;self.vocab_size=248320;self.eos_token_ids=EOS_TOKEN_IDS
        self.model=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(
            vocab_size=self.vocab_size,max_position_embeddings=self.max_context_tokens)))
        self.tokenizer=Tokenizer();self.calls=[];self.cache_resets=[]
    def guard(self):pass
    def prepare_prompt(self,messages,tools):return [1,int(messages[0]['content'])]
    def next_token_logits(self,inputs,past_key_values=None,*,use_cache=True):
        assert not torch.is_grad_enabled() and use_cache
        if past_key_values is None:
            index=inputs[-1];position=0;self.cache_resets.append(index)
        else:index,position=past_key_values
        text=(r'<think>wrong \boxed{0}</think>\boxed{1}' if index in (4,7)
              else r'<think>right \boxed{1}</think>\boxed{0}')
        sequence=self.tokenizer.encode(text) if index!=6 else [ord('x')]*64
        token=sequence[position]
        logits=torch.full((self.vocab_size,),-80.);logits[token]=80.
        self.calls.append((index,position))
        return logits,(index,position+1)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def setUp(self):
        self.m=manifest();self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);Session.opened=[]
        self.native=patch('subnet.environments.create_session',side_effect=lambda _:Session())
        self.spec=patch('subnet.environments.EnvironmentSpec.from_dict',side_effect=lambda v:SimpleNamespace(**v))
        self.native.start();self.spec.start();self.addCleanup(self.native.stop);self.addCleanup(self.spec.stop)
    def job(self):
        return dict(evaluation=dict(version=evaluation.VERSION,run_id=self.m['run_id'],set_name='fixed4'))
    def execute(self):
        return evaluation.evaluate(self.m,evaluation.validate_job(self.job(),self.m),Runtime(self.m))
    def test_all_attempts_scored_shared_final_grade_cap_and_infra_distinct(self):
        result=self.execute()
        self.assertEqual([r['classification'] for r in result['rows']],['positive','negative','neutral','indeterminate'])
        self.assertEqual(Session.opened,[4,5,6,7]);self.assertEqual(result['requested_count'],4)
        self.assertEqual((result['count'],result['successes'],result['error_count']),(3,1,1))
        self.assertIsNone(result['mean_reward']);self.assertEqual(result['rows'][2]['output_length'],64)
        self.assertEqual(result['rows'][2]['stop_reason'],'budget')
        self.assertIsNone(result['rows'][3]['reward'])
        self.assertNotIn('simulated',str(result))
        self.assertTrue(all('proofs' not in r for r in result['rows']))
        evaluation.validate_result(result,self.job(),self.m)
    def test_all_correct_suite_is_retained_without_group_minimum(self):
        self.m['evaluation_sets'][0]['indices']=[4];self.m['evaluation_sets'][0]['repeats']=2
        runtime=Runtime(self.m)
        result=evaluation.evaluate(self.m,evaluation.validate_job(self.job(),self.m),runtime)
        self.assertEqual(result['successes'],2);self.assertEqual(result['mean_reward'],1.)
        self.assertEqual(runtime.cache_resets,[4,4])
        self.assertEqual(len(runtime.calls),2*result['rows'][0]['output_length'])
    def test_padded_model_head_never_expands_signed_tokenizer_support(self):
        self.m['evaluation_sets'][0]['indices']=[4]
        runtime=Runtime(self.m)
        runtime.vocab_size+=32
        runtime.model.config.text_config.vocab_size=runtime.vocab_size
        original=runtime.next_token_logits
        def padded(*args,**kwargs):
            logits,cache=original(*args,**kwargs)
            logits[len(runtime.tokenizer):]=1000.  # Would win without slicing.
            return logits,cache
        runtime.next_token_logits=padded
        result=evaluation.evaluate(self.m,evaluation.validate_job(self.job(),self.m),runtime)
        self.assertEqual(result['successes'],1)
        self.assertTrue(all(t<len(runtime.tokenizer) for t in result['rows'][0]['output']))
        evaluation.validate_result(result,self.job(),self.m)
    def test_changed_support_and_undersized_head_reject_before_generation(self):
        for changed_support in (True,False):
            runtime=Runtime(self.m);m=deepcopy(self.m)
            if changed_support:m['group_transport_policy']['vocab_size']+=1
            else:runtime.model.config.text_config.vocab_size-=1
            with self.subTest(support=changed_support),self.assertRaises(ValueError):
                evaluation.evaluate(m,evaluation.validate_job(self.job(),m),runtime)
            self.assertEqual(runtime.calls,[])
    def test_split_horizon_and_unknown_suite_fail_before_model(self):
        for field,value in [('indices',[0]),('indices',[4,4]),('indices',[True]),('indices',[8]),('seed',True),('repeats',0)]:
            m=deepcopy(self.m);m['evaluation_sets'][0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):evaluation.validate_sets(m)
        job=self.job();job['evaluation']['set_name']='invented'
        with self.assertRaises(ValueError):evaluation.validate_job(job,self.m)
        job=self.job();job['capability']={}
        with self.assertRaises(ValueError):evaluation.validate_job(job,self.m)
        m=deepcopy(self.m);m['environments'][0]['harness']['top_k']=0
        with self.assertRaises(ValueError):evaluation.validate_sets(m)
    def test_eval_rng_common_across_checkpoints_but_suite_changes_are_distinct(self):
        a=evaluation.validate_sets(self.m)['fixed4']['dataset_id']
        m=deepcopy(self.m);m['checkpoint']['id']='b'*64;m['epoch']='different-epoch'
        m['sampling_contract']['calibration']['checkpoint']='b'*64
        b=evaluation.validate_sets(m)['fixed4']['dataset_id']
        self.assertEqual(a,b);self.assertEqual(evaluation.draw(a,318,4,0,0),evaluation.draw(b,318,4,0,0))
        m['evaluation_sets'][0]['seed']=319
        self.assertNotEqual(a,evaluation.validate_sets(m)['fixed4']['dataset_id'])
    def test_report_reordering_dropped_tasks_cap_as_negative_and_run_substitution_reject(self):
        result=self.execute()
        for mutation in ('drop','reorder','capnegative','run','reward','checkpoint'):
            other=deepcopy(result)
            if mutation=='drop':other['rows'].pop()
            elif mutation=='reorder':other['rows'].reverse()
            elif mutation=='capnegative':other['rows'][2]['classification']='negative'
            elif mutation=='run':other['run_id']='other'
            elif mutation=='reward':other['mean_reward']=1.
            else:other['checkpoint']='c'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):evaluation.validate_result(other,self.job(),self.m)
    def test_lease_loss_aborts_without_grading_fabricated_negatives(self):
        calls=0
        def guard():
            nonlocal calls
            calls+=1
            if calls>3:raise ValueError('lost original lease')
        with self.assertRaisesRegex(ValueError,'lost original lease'):
            evaluation.evaluate(self.m,evaluation.validate_job(self.job(),self.m),Runtime(self.m),guard=guard)
        self.assertEqual(Session.opened,[4])
    def test_real_backend_and_queue_dispatch_never_enter_legacy_eval_or_miner(self):
        from subnet import backend_jobs
        from subnet.distributed_roles import Coordinator
        from subnet.distributed_worker import Worker
        key=SigningKey.generate();worker_key=SigningKey.generate();authority=key.verify_key.encode().hex()
        job=dict(schema=1,worker_policy=mimo_worker.VERSION,job_id='mimo-eval-one',role='evaluate',
            created_at=time.time()-5,expires_at=time.time()+180,manifest=sign(key,self.m),
            source_files=mimo_worker.source_inventory(),runtime_versions={p:version(p) for p in ('torch','transformers','toploc')},
            **self.job())
        runtime=Runtime(self.m)
        class Factory:
            @contextmanager
            def runtime_for_job(self,*args,**kwargs):yield SimpleNamespace(backend=runtime)
            def close(self):pass
        queue=Coordinator(self.root/'queue.db',authority,{worker_key.verify_key.encode().hex():['evaluate']})
        queue.enqueue(sign(key,job));counter=iter(range(100))
        worker=Worker('http://localhost:99',worker_key.encode(),authority,self.root/'worker',role='evaluate',mimo_factory=Factory())
        worker.request=lambda action,**kw:queue.request(sign(worker_key,dict(action=action,at=time.time(),nonce=str(next(counter)).zfill(32),**kw)))
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}),patch('subnet.evaluation.evaluate',side_effect=AssertionError('legacy')),patch('subnet.miner.Miner',side_effect=AssertionError('mining')),patch('subnet.distributed_worker.subprocess.run',side_effect=AssertionError('per-job process')):
            self.assertTrue(worker.once())
        result=queue.status(job['job_id'])
        self.assertEqual(result['status'],'complete');self.assertEqual(result['report']['evaluation']['error_count'],1)
        self.assertEqual(result['report']['evaluation']['successes'],1)
        self.assertFalse(result['report']['chain_transactions']);worker.close()


if __name__=='__main__':unittest.main()
