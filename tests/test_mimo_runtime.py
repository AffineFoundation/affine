"""Real tiny HF hybrid controls and checkpoint-lifetime regressions."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from subnet.mimo_runtime import (
    CheckpointCache, CheckpointGuard, MimoRuntime, MimoRuntimePool,
    PersistentMimoRuntimeFactory, digest, file_hash, make_runtime_spec, validate_runtime_spec,
)


def tiny_checkpoint(directory):
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5ForConditionalGeneration, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    torch.manual_seed(9)
    text = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=["linear_attention"]*3+["full_attention"],
        max_position_embeddings=128, rope_parameters={"rope_type":"default", "rope_theta":10000,
        "partial_rotary_factor":1.0, "mrope_section":[2,2,4]}, eos_token_id=2, pad_token_id=0)
    config = Qwen3_5Config(text_config=text, vision_config={"depth":1,"hidden_size":32,
        "intermediate_size":48,"num_heads":2,"patch_size":2,"spatial_merge_size":1,
        "temporal_patch_size":1,"out_hidden_size":32,"num_position_embeddings":16})
    config._attn_implementation = "eager"
    model = Qwen3_5ForConditionalGeneration(config).eval()
    model.generation_config.eos_token_id = [2, 3]
    model.save_pretrained(directory, safe_serialization=True)
    inner = Tokenizer(WordLevel({"<pad>":0,"<unk>":1,"<eos>":2,"<eot>":3,
        "user":4,"assistant":5,"hello":6,"think":7}, unk_token="<unk>"))
    inner.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner, eos_token="<eos>",
        unk_token="<unk>", pad_token="<pad>")
    tokenizer.chat_template = "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}{% if enable_thinking %}think {% endif %}{% if add_generation_prompt %}assistant{% endif %}"
    tokenizer.save_pretrained(directory)
    files = {p.name:file_hash(p) for p in Path(directory).iterdir()}
    descriptor = {"id":digest(files),"files":files}
    spec = make_runtime_spec(chat_template=tokenizer.chat_template, enable_thinking=True,
        max_context_tokens=128, scope="cpu_fixture", eos_token_ids=[2,3])
    return descriptor, spec


def dummy_checkpoint(tag):
    raw = {"config.json":json.dumps({"tag":tag}).encode(), "model.safetensors":tag.encode()}
    files = {k:hashlib.sha256(v).hexdigest() for k,v in raw.items()}
    def fetch(name, destination):
        destination.write_bytes(raw[name])
    return {"id":digest(files),"files":files}, fetch, sum(map(len,raw.values()))


class HybridRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)/"checkpoint"
        cls.path.mkdir()
        cls.descriptor, cls.spec = tiny_checkpoint(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def load(self, trainable=False):
        return MimoRuntime.load(self.path, self.descriptor, self.spec,
            device="cpu", trainable=trainable, allow_fixture=True)

    def test_hybrid_cached_prefill_and_teacher_forcing_agree(self):
        runtime = self.load()
        self.addCleanup(runtime.close)
        prompt, output = [4,6,5], [10,11,2]
        cache = None
        cached = []
        for index in range(len(output)):
            logits, cache = runtime.next_token_logits(prompt if index == 0 else [output[index-1]], cache)
            cached.append(logits)
        with torch.inference_mode():
            full = runtime.model(torch.tensor([prompt+output]), use_cache=False).logits[0, len(prompt)-1:-1].float()
        torch.testing.assert_close(torch.stack(cached), full, atol=2e-6, rtol=2e-5)
        self.assertEqual(runtime.check_recurrent_state(cache), 3)
        self.assertEqual(runtime.eos_token_ids, (2,3))
        self.assertEqual(runtime.prepare_prompt([{"role":"user","content":"hello"}]), [4,6,7,5])

    def test_training_score_and_gradients_match_full_model(self):
        runtime = self.load(trainable=True)
        self.addCleanup(runtime.close)
        prompt, output = [4,6,5], [10,11,2]
        score = runtime.response_log_probs(prompt, output, gradients=True, token_chunk_size=1)
        (-score.mean()).backward()
        gradients = {n:p.grad.clone() for n,p in runtime.model.named_parameters() if p.requires_grad}
        self.assertTrue(all(p.grad is None for n,p in runtime.model.named_parameters() if n.startswith("model.visual.")))
        runtime.model.zero_grad(set_to_none=True)
        logits = runtime.model(torch.tensor([prompt+output]), use_cache=False).logits[0,len(prompt)-1:-1]
        expected = torch.log_softmax(logits.float(),-1).gather(-1,torch.tensor(output)[:,None]).squeeze(-1)
        torch.testing.assert_close(score, expected, atol=1e-6, rtol=1e-5)
        (-expected.mean()).backward()
        for name,p in runtime.model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(gradients[name],p.grad,atol=2e-6,rtol=5e-4)
        self.assertFalse(runtime.score(prompt,output,gradients=False).requires_grad)

    def test_toploc_compute_and_model_only_export(self):
        runtime = self.load()
        self.addCleanup(runtime.close)
        acts, probs = runtime.compute([4,6,5],[10,11,2])
        self.assertEqual([a.shape[0] for a in acts], [3,1,1,1])
        self.assertEqual(probs.shape, (3,64))
        from toploc import build_proofs_base64
        from subnet.proofs import verify_mapped_proofs
        import toploc.poly as poly
        from toploc.C.csrc.utils import get_fp_parts
        with patch.object(poly,"get_fp_parts",lambda tensor:get_fp_parts(tensor,num_threads=2)):
            proofs=build_proofs_base64(acts,decode_batching_size=16,topk=16)
            results=verify_mapped_proofs(acts,proofs,decode_batching_size=16,topk=16)
        self.assertTrue(all(r.exp_mismatches==0 and r.mant_err_mean==0 for r in results))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"model"
            descriptor = runtime.save_model(output)
            CheckpointGuard(output,descriptor)()
            self.assertFalse(any("optimizer" in name for name in descriptor["files"]))
            self.assertEqual(json.loads((output/"generation_config.json").read_text())["eos_token_id"], [2,3])

    def test_real_grpo_step_then_publication_advances_without_reload(self):
        from subnet.mimo_grpo.config import TrainerConfig
        from subnet.mimo_grpo.data import InputBinding,TaskGroup,Trajectory
        from subnet.mimo_grpo.optimizer import CPUAdamW
        from subnet.mimo_grpo.training import train_update
        runtime=self.load(trainable=True)
        self.addCleanup(runtime.close)
        config=TrainerConfig(group_size=8)
        optimizer=CPUAdamW(runtime.model.named_parameters(),config,input_checkpoint=runtime.checkpoint_id)
        rows=tuple(Trajectory(i,(10+i,2),"correct"if i%2==0 else"wrong","eos")for i in range(8))
        groups=(TaskGroup("math:1",(4,6,5),rows,"b"*64),)
        before=id(runtime.model)
        binding=InputBinding("tiny-control","epoch1",runtime.checkpoint_id,"a"*64,0)
        report=train_update(runtime,optimizer,groups,binding,config=config,guard=runtime.guard)
        self.assertEqual(optimizer.global_step,1)
        with tempfile.TemporaryDirectory()as directory:
            descriptor=runtime.save_model(Path(directory)/"model")
            runtime.acknowledge_model_publication(descriptor["id"],descriptor["files"])
            self.assertEqual(runtime.checkpoint_id,descriptor["id"])
            self.assertEqual(id(runtime.model),before)
            runtime.guard()

    def test_reject_profile_and_runtime_mutations(self):
        with self.assertRaisesRegex(ValueError,"CPU fixtures"):
            validate_runtime_spec(self.spec)
        altered = self.spec.to_dict()
        altered["top_k"] = 0
        with self.assertRaisesRegex(ValueError,"sampler"):
            validate_runtime_spec(altered,allow_fixture=True)
        runtime = self.load()
        self.addCleanup(runtime.close)
        runtime.tokenizer.chat_template += "extra"
        with self.assertRaisesRegex(ValueError,"template"):
            runtime.guard()

    def test_unsupported_source_never_loads_model(self):
        altered = self.spec.to_dict()
        altered["runtime_fingerprint"] = {"versions":{},"sources":{}}
        with self.assertRaisesRegex(ValueError,"runtime differs"):
            MimoRuntime.load(self.path,self.descriptor,altered,device="cpu",allow_fixture=True)

    def test_runtime_fingerprint_binds_toploc_compiled_extensions(self):
        fingerprint = self.spec.runtime_fingerprint
        self.assertEqual(fingerprint['versions']['toploc'], '0.1.6')
        for name in ('toploc.poly', 'toploc.C.csrc.poly', 'toploc.C.csrc.ndd', 'toploc.C.csrc.utils'):
            self.assertEqual(len(fingerprint['sources'][name]), 64)


class CacheLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = CheckpointCache(Path(self.temp.name)/"cache",max_bytes=10000,
            max_checkpoints=1,disk_reserve_bytes=0)
        self.addCleanup(self.cache.close)

    def test_checkpoint_is_not_deleted_after_job(self):
        descriptor, fetch, size = dummy_checkpoint("first")
        calls=[]
        def download(name,path):
            calls.append(name);fetch(name,path)
        with self.cache.acquire(descriptor,fetch=download,expected_bytes=size) as path:
            first=path
        self.assertTrue(first.exists())
        with self.cache.acquire(descriptor,fetch=download,expected_bytes=size) as second:
            self.assertEqual(first,second)
        self.assertEqual(len(calls),2)

    def test_only_unused_old_checkpoint_is_evicted(self):
        one,fetch1,size1=dummy_checkpoint("first")
        two,fetch2,size2=dummy_checkpoint("second")
        with self.cache.acquire(one,fetch=fetch1,expected_bytes=size1) as first:
            with self.assertRaisesRegex(ValueError,"in-flight"):
                with self.cache.acquire(two,fetch=fetch2,expected_bytes=size2):pass
            self.assertTrue(first.exists())
        with self.cache.acquire(two,fetch=fetch2,expected_bytes=size2) as second:
            self.assertTrue(second.exists())
            self.assertFalse(first.exists())

    def test_bad_download_is_removed_and_source_mutation_detected(self):
        descriptor,fetch,size=dummy_checkpoint("first")
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            with self.cache.acquire(descriptor,fetch=lambda n,p:p.write_bytes(b"x"),expected_bytes=size):pass
        self.assertEqual(list(self.cache.root.glob(".download-*")),[])
        with self.cache.acquire(descriptor,fetch=fetch,expected_bytes=size) as path:
            (path/"config.json").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError,"changed"):
            with self.cache.acquire(descriptor):pass

    def test_cache_recovery_reauthenticates_and_exclusive_owner(self):
        descriptor,fetch,size=dummy_checkpoint("first")
        with self.cache.acquire(descriptor,fetch=fetch,expected_bytes=size):pass
        with self.assertRaisesRegex(ValueError,"owned"):
            CheckpointCache(self.cache.root,max_bytes=10000,disk_reserve_bytes=0)
        self.cache.close()
        recovered=CheckpointCache(self.cache.root,max_bytes=10000,disk_reserve_bytes=0)
        self.addCleanup(recovered.close)
        with recovered.acquire(descriptor) as path:
            self.assertTrue(path.exists())

    def test_restarted_cache_reconciles_lower_limits_without_evicting_current(self):
        self.cache.close()
        previous = CheckpointCache(self.cache.root, max_bytes=10000,
            max_checkpoints=2, disk_reserve_bytes=0)
        one, fetch1, size1 = dummy_checkpoint("old")
        two, fetch2, size2 = dummy_checkpoint("current")
        with previous.acquire(one, fetch=fetch1, expected_bytes=size1) as old:
            pass
        with previous.acquire(two, fetch=fetch2, expected_bytes=size2) as current:
            pass
        previous.close()
        current_bytes = sum(p.stat().st_size for p in current.iterdir())
        recovered = CheckpointCache(self.cache.root, max_bytes=current_bytes,
            max_checkpoints=1, disk_reserve_bytes=0)
        self.addCleanup(recovered.close)
        with recovered.acquire(two) as same:
            self.assertEqual(same, current)
            self.assertTrue(current.exists())
            self.assertFalse(old.exists())

    def test_persistent_runtime_reuses_model_and_releases_old_generation(self):
        one,fetch1,size1=dummy_checkpoint("first")
        two,fetch2,size2=dummy_checkpoint("second")
        from subnet.mimo_runtime import MimoRuntimeSpec,VERSION,MODEL_ID
        spec=MimoRuntimeSpec(VERSION,MODEL_ID,{"versions":{},"sources":{}},"a"*64,
            True,128,(2,3),scope="cpu_fixture",parameter_dtype="float32")
        loaded=[]
        class DummyRuntime:
            def __init__(self,*args,**kwargs):self.closed=False;loaded.append(self)
            def guard(self):self.assert_open()
            def assert_open(self):assert not self.closed
            def close(self):self.closed=True
        pool=MimoRuntimePool(self.cache,loader=DummyRuntime)
        self.addCleanup(pool.close)
        with pool.acquire(one,spec,device="cpu",fetch=fetch1,expected_bytes=size1,allow_fixture=True) as first:
            self.assertFalse(first.closed)
        with pool.acquire(one,spec,device="cpu",allow_fixture=True) as again:
            self.assertIs(first,again)
        with pool.acquire(two,spec,device="cpu",fetch=fetch2,expected_bytes=size2,allow_fixture=True):
            self.assertTrue(first.closed)
        self.assertEqual(len(loaded),2)

    def test_worker_factory_reuses_jobs_and_retires_checkpoint_only_after_lease(self):
        from subnet.mimo_runtime import MimoRuntimeSpec,VERSION,MODEL_ID
        from types import SimpleNamespace
        first,fetch1,size1=dummy_checkpoint("factory-first")
        second,fetch2,size2=dummy_checkpoint("factory-second")
        spec=MimoRuntimeSpec(VERSION,MODEL_ID,{"versions":{},"sources":{}},"a"*64,
            True,128,(2,3),scope="cpu_fixture",parameter_dtype="float32")
        loads=[];fetches=[]
        class DummyRuntime:
            def __init__(self,path,descriptor,spec,**kwargs):
                self.checkpoint_id=descriptor['id'];self.closed=False;loads.append(self)
            def guard(self):assert not self.closed
            def close(self):self.closed=True
        def fetch(manifest,name,path):
            fetches.append((manifest['checkpoint']['id'],name))
            (fetch1 if manifest['checkpoint']==first else fetch2)(name,path)
        factory=PersistentMimoRuntimeFactory(Path(self.temp.name)/'factory',max_bytes=10000,
            max_checkpoints=1,max_checkpoint_bytes=1000,disk_reserve_bytes=0,
            device='cpu',allow_fixture=True,loader=DummyRuntime,fetch_file=fetch,
            facade_builder=lambda *a,backend,miner,**k:SimpleNamespace(backend=backend,miner=miner))
        self.addCleanup(factory.close)
        def manifest(descriptor):return dict(checkpoint=descriptor,mimo_runtime=spec.to_dict())
        with factory.runtime_for_job('unused',manifest(first),{}, {},miner='one')as one:
            self.assertEqual(one.miner,'one')
            with self.assertRaisesRegex(ValueError,'in flight'):
                with factory.runtime_for_job('unused',manifest(first),{}, {},miner='same-checkpoint'):pass
            with self.assertRaisesRegex(ValueError,'in flight'):
                with factory.runtime_for_job('unused',manifest(second),{}, {},miner='two'):pass
            self.assertFalse(one.backend.closed)
        with patch('subnet.mimo_runtime.file_hash',side_effect=AssertionError('no per-job rehash')):
            with factory.runtime_for_job('removed-staging-dir',manifest(first),{}, {},miner='other')as again:
                self.assertIs(again.backend,one.backend)
                self.assertEqual(again.miner,'other')
        self.assertEqual(len(loads),1);self.assertEqual(len(fetches),2)
        with factory.runtime_for_job('unused',manifest(second),{}, {},miner='two')as two:
            self.assertTrue(one.backend.closed)
            self.assertFalse((factory.cache.root/first['id']).exists())
            self.assertTrue((factory.cache.root/second['id']).exists())
            self.assertNotEqual(id(two.backend),id(one.backend))
        self.assertEqual(len(loads),2);self.assertEqual(len(fetches),4)

    def test_verification_entrypoint_holds_worker_lease_through_all_checks(self):
        from contextlib import contextmanager
        import test_complete_group_mimo as fixtures
        from subnet.complete_group_codec import encode,limits
        from subnet.complete_group_verification import verify
        fixture=fixtures.AuditPathTests('test_full_canonical_group_audit_and_report_binding')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        original=fixture.r.compute
        state={'active':False,'jobs':0,'calls':0}
        def checked_compute(*args):
            self.assertTrue(state['active']);state['calls']+=1;return original(*args)
        fixture.r.compute=checked_compute
        class Worker:
            @contextmanager
            def runtime_for_job(self,*args,**kwargs):
                state['jobs']+=1;state['active']=True
                try:yield fixture.r
                finally:state['active']=False
        worker=Worker();data=encode([(fixture.batch,fixture.arrays)],limits(fixture.m))
        for _ in range(2):
            result=verify(data,fixture.m,'unused',miner=fixture.miner,runtime_factory=worker)
            self.assertEqual(result['accepted'],[fixture.batch])
            self.assertFalse(state['active'])
        self.assertEqual(state['jobs'],2);self.assertEqual(state['calls'],16)


if __name__ == "__main__":
    unittest.main()
