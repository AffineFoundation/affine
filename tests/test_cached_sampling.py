"""Real tiny CPU-model cache controls; these do not qualify GPU proof replay."""
import functools
import unittest
from types import SimpleNamespace
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from subnet.cached_sampling import sample
from subnet import harness
from subnet.gpu_runtime import GPURuntime

class CachedSampling(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        config=Qwen2Config(vocab_size=127,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
        config._attn_implementation='eager';self.model=Qwen2ForCausalLM(config).eval()
        self.prompt=[3,4,5,6];self.settings=dict(seed=19,max_output_tokens=8,temperature=.8,top_p=.9)
        self.tokenizer=SimpleNamespace(eos_token_id=None)
    def spy(self):
        calls=[];forward=self.model.forward
        @functools.wraps(forward)
        def recorded(ids,**kwargs):
            calls.append((ids.shape[1],kwargs.get('past_key_values') is not None,
                          kwargs.get('logits_to_keep'),kwargs['use_cache']))
            return forward(ids,**kwargs)
        self.model.forward=recorded;return calls
    def test_prefill_once_and_cache_reset_between_calls(self):
        calls=self.spy();output,_=sample(self.model,self.prompt,**self.settings)
        self.assertEqual(calls,[(4,False,1,True)]+[(1,True,1,True)]*7)
        calls.clear();second,_=sample(self.model,self.prompt,**self.settings)
        self.assertEqual(output,second);self.assertEqual(calls[0],(4,False,1,True))
        self.assertFalse(torch.cuda.is_initialized())
    def test_eos_stops_and_invalid_context_architecture_mode_refuse(self):
        first=sample(self.model,self.prompt,**{**self.settings,'max_output_tokens':1})[0][0]
        self.assertEqual(sample(self.model,self.prompt,eos_token_id=first,**self.settings)[0],[first])
        for prompt,extra in [([True],{}),([127],{}),([],{}),(self.prompt,{'max_output_tokens':61}),(self.prompt,{'mode':'implicit'})]:
            with self.assertRaises(ValueError):sample(self.model,prompt,**{**self.settings,**extra})
        self.model.config.model_type='unsupported'
        with self.assertRaises(ValueError):sample(self.model,self.prompt,**self.settings)
    def test_legacy_harness_stays_uncached_and_only_signed_new_version_opts_in(self):
        config=dict(version='text-tools-long-v2',policy='autoregressive',max_output_tokens=8,temperature=.8,top_p=.9)
        calls=self.spy();harness.sample(self.model,self.tokenizer,self.prompt,19,config)
        self.assertEqual(calls,[(n,False,None,False) for n in range(4,12)])
        calls.clear();new=dict(config,version='text-tools-long-kv-v3')
        harness.sample(self.model,self.tokenizer,self.prompt,19,new)
        self.assertEqual(calls,[(4,False,1,True)]+[(1,True,1,True)]*7)
        for extra in [dict(policy='candidates',candidates=['a','b']),dict(sampling_mode='legacy'),dict(generation_kv_cache=False),dict(turn_overrides={'0':{'policy':'candidates','candidates':['a','b']}})]:
            with self.assertRaises(ValueError):harness.normalize(dict(new,**extra))
    def test_GPU_runtime_opt_in_dispatch_on_real_CPU_model(self):
        config=dict(version='text-tools-long-kv-v3',policy='autoregressive',max_output_tokens=8,temperature=.8,top_p=.9)
        actor=SimpleNamespace(model=self.model,tokenizer=self.tokenizer,harness=config)
        output=GPURuntime.sample(actor,self.prompt,19,[],0)
        self.assertEqual(output,sample(self.model,self.prompt,**self.settings)[0])
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
