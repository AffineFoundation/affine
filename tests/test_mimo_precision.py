import unittest
import os
from unittest.mock import patch
from types import SimpleNamespace as NS
from dataclasses import replace
from subnet.mimo_precision import PrecisionProfile, PrecisionMismatch, observed, guard, configure_before_model_load, guarded_forward


class TorchFixture:
    def __init__(self):
        self.backends = NS(cuda=NS(matmul=NS(allow_tf32=False,
            allow_bf16_reduced_precision_reduction=True,
            allow_fp16_reduced_precision_reduction=True)), cudnn=NS(allow_tf32=True))
        self.precision = "highest"
        self.deterministic = False
        self.autocast = dict(cpu=False, cuda=False)
    def get_float32_matmul_precision(self): return self.precision
    def set_float32_matmul_precision(self, value):
        self.precision = value
        self.backends.cuda.matmul.allow_tf32 = value != "highest"
    def are_deterministic_algorithms_enabled(self): return self.deterministic
    def use_deterministic_algorithms(self, value): self.deterministic = value
    def is_autocast_enabled(self, device): return self.autocast[device]


class Tests(unittest.TestCase):
    def setUp(self):
        scope=patch.dict(os.environ,{"CUBLAS_WORKSPACE_CONFIG":":4096:8"});scope.start();self.addCleanup(scope.stop)
    def test_workspace_missing_or_changed_blocks_before_mutation_or_execution(self):
        for value in (None,":16:8",""):
            with patch.dict(os.environ,clear=True):
                if value is not None:os.environ["CUBLAS_WORKSPACE_CONFIG"]=value
                t=TorchFixture();t.precision="medium";executed=[]
                with self.assertRaises(PrecisionMismatch):configure_before_model_load(t)
                self.assertEqual(t.precision,"medium")
                with self.assertRaises(PrecisionMismatch):guarded_forward(t,lambda:executed.append(True))
                self.assertEqual(executed,[])
    def test_baseline(self):
        t = TorchFixture(); guard(t); self.assertEqual(observed(t), PrecisionProfile())
    def test_configure_before_load_restores_ambient_flags(self):
        t = TorchFixture(); t.set_float32_matmul_precision("medium"); t.deterministic = True
        configure_before_model_load(t); guard(t)
    def test_each_changed_flag_rejected(self):
        changes = [lambda t:setattr(t.backends.cuda.matmul,"allow_tf32",True),
            lambda t:setattr(t,"precision","medium"),
            lambda t:setattr(t.backends.cudnn,"allow_tf32",False),
            lambda t:setattr(t.backends.cuda.matmul,"allow_bf16_reduced_precision_reduction",False),
            lambda t:setattr(t.backends.cuda.matmul,"allow_fp16_reduced_precision_reduction",False),
            lambda t:setattr(t,"deterministic",True)]
        for change in changes:
            with self.subTest(change=change):
                t=TorchFixture(); change(t)
                with self.assertRaises(PrecisionMismatch):guard(t)
    def test_autocast_each_device_rejected_at_load_and_forward(self):
        for device in ("cpu","cuda"):
            t=TorchFixture(); t.autocast[device]=True
            with self.assertRaises(PrecisionMismatch):configure_before_model_load(t)
            with self.assertRaises(PrecisionMismatch):guarded_forward(t,lambda:None)
    def test_guard_never_rewrites_bad_flags(self):
        t=TorchFixture(); t.set_float32_matmul_precision("medium")
        with self.assertRaises(PrecisionMismatch):guard(t)
        self.assertEqual(t.precision,"medium")
    def test_bad_profile_rejected_before_mutation(self):
        t=TorchFixture()
        with self.assertRaises(PrecisionMismatch):configure_before_model_load(t,replace(PrecisionProfile(),matmul_tf32=True))
        self.assertEqual(observed(t),PrecisionProfile())
    def test_guard_blocks_execution(self):
        t=TorchFixture(); t.deterministic=True; executed=[]
        with self.assertRaises(PrecisionMismatch):guarded_forward(t,lambda:executed.append(True))
        self.assertEqual(executed,[])
    def test_postcondition_detects_flag_mutation(self):
        t=TorchFixture()
        with self.assertRaises(PrecisionMismatch):guarded_forward(t,lambda:setattr(t,"deterministic",True))
    def test_errors_propagate(self):
        t=TorchFixture()
        def bad():raise RuntimeError("unsupported deterministic operation")
        with self.assertRaisesRegex(RuntimeError,"unsupported deterministic"):guarded_forward(t,bad)


class RuntimeSchemaTests(unittest.TestCase):
    def test_original_v1_serialization_and_digest_are_unchanged(self):
        from subnet.mimo_runtime import MimoRuntimeSpec, VERSION, MODEL_ID, EOS_TOKEN_IDS, validate_runtime_spec, digest
        spec=MimoRuntimeSpec(VERSION,MODEL_ID,{"versions":{},"sources":{}},"a"*64,True,64,EOS_TOKEN_IDS)
        original=spec.to_dict()
        self.assertNotIn("precision_profile",original)
        self.assertEqual(validate_runtime_spec(original).to_dict(),original)
        self.assertEqual(spec.sha256,digest(original))
    def test_v2_requires_exact_profile_and_changes_bound_digest(self):
        from subnet.mimo_runtime import MimoRuntimeSpec, VERSION, PRECISION_VERSION, MODEL_ID, EOS_TOKEN_IDS, validate_runtime_spec
        from subnet.mimo_precision import profile_dict
        old=MimoRuntimeSpec(VERSION,MODEL_ID,{"versions":{},"sources":{}},"a"*64,True,64,EOS_TOKEN_IDS)
        new=replace(old,version=PRECISION_VERSION,precision_profile=profile_dict())
        self.assertNotEqual(old.sha256,new.sha256)
        self.assertEqual(validate_runtime_spec(new).to_dict(),new.to_dict())
        broken=new.to_dict();del broken["precision_profile"]
        with self.assertRaises(ValueError):validate_runtime_spec(broken)
        broken=new.to_dict();broken["precision_profile"]["matmul_tf32"]=0
        with self.assertRaises(ValueError):validate_runtime_spec(broken)
        broken=old.to_dict();broken["precision_profile"]=profile_dict()
        with self.assertRaises(ValueError):validate_runtime_spec(broken)


class RealHybridPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path
        from test_mimo_runtime import tiny_checkpoint
        from subnet.mimo_runtime import PRECISION_VERSION
        from subnet.mimo_precision import profile_dict
        import torch
        torch.set_num_threads(2)
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)/"checkpoint";cls.path.mkdir()
        cls.descriptor,spec=tiny_checkpoint(cls.path)
        cls.spec=replace(spec,version=PRECISION_VERSION,precision_profile=profile_dict())
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def setUp(self):
        import torch
        scope=patch.dict(os.environ,{"CUBLAS_WORKSPACE_CONFIG":":4096:8"});scope.start();self.addCleanup(scope.stop)
        self.old=observed(torch)
    def tearDown(self):
        import torch
        p=self.old;torch.set_float32_matmul_precision(p.float32_precision)
        torch.backends.cuda.matmul.allow_tf32=p.matmul_tf32
        torch.backends.cudnn.allow_tf32=p.cudnn_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=p.bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=p.fp16_reduced_precision_reduction
        torch.use_deterministic_algorithms(p.deterministic_algorithms)
    def load(self,trainable=False):
        from subnet.mimo_runtime import MimoRuntime
        r=MimoRuntime.load(self.path,self.descriptor,self.spec,device="cpu",trainable=trainable,allow_fixture=True)
        self.addCleanup(r.close);return r
    def test_real_cached_prefill_training_and_provenance(self):
        import torch,os
        from unittest.mock import patch
        from subnet.mimo_precision import validate_provenance
        r=self.load(trainable=True);p,o=[4,6,5],[10,11,2]
        logits,cache=r.next_token_logits(p)
        self.assertEqual(logits.shape,(64,));self.assertEqual(r.check_recurrent_state(cache),3)
        acts,probs=r.compute(p,o);self.assertEqual(probs.shape,(3,64))
        score=r.response_log_probs(p,o,gradients=True,token_chunk_size=1)
        (-score.mean()).backward()
        self.assertTrue(any(x.grad is not None for x in r.model.parameters()))
        with patch.dict(os.environ,{"CUBLAS_WORKSPACE_CONFIG":":4096:8"}):
            provenance=r.execution_provenance()
            validate_provenance(provenance,self.spec.precision_profile,device="cpu")
        self.assertEqual(provenance["scope"],"worker-process-observation")
    def test_changed_flags_block_each_public_forward_before_compute(self):
        import torch
        from unittest.mock import patch
        r=self.load()
        for fn in [lambda:r.next_token_logits([4,6,5]),lambda:r.compute([4,6,5],[10,2]),
                   lambda:r.response_log_probs([4,6,5],[10,2],gradients=False)]:
            torch.backends.cuda.matmul.allow_tf32=True
            with patch.object(r.model,"forward",side_effect=AssertionError("must not execute")):
                with self.assertRaises(PrecisionMismatch):fn()
            configure_before_model_load(torch)
    def test_ambient_autocast_cannot_load_or_infer(self):
        import torch
        r=self.load()
        with torch.autocast(device_type="cpu"):
            with self.assertRaises(PrecisionMismatch):self.load()
            with self.assertRaises(PrecisionMismatch):r.compute([4,6,5],[10,2])
    def test_backward_recompute_guard_catches_changed_flags(self):
        import torch
        r=self.load(trainable=True)
        score=r.response_log_probs([4,6,5],[10,11,2],gradients=True,token_chunk_size=1)
        torch.backends.cuda.matmul.allow_tf32=True
        with self.assertRaises(PrecisionMismatch):(-score.mean()).backward()
    def test_provenance_rejects_claimed_profile_and_unbounded_metadata(self):
        import os
        from copy import deepcopy
        from unittest.mock import patch
        from subnet.mimo_precision import validate_provenance
        r=self.load()
        with patch.dict(os.environ,{"CUBLAS_WORKSPACE_CONFIG":":4096:8"}):base=r.execution_provenance()
        for mutate in [lambda p:p.update(precision_profile_sha256="a"*64),
                       lambda p:p["observed_precision"].update(matmul_tf32=True),
                       lambda p:p["autocast"].update(cpu=0),
                       lambda p:p["build"].update(torch="a"*1000),
                       lambda p:p["launch"].update(cpu_threads=True)]:
            p=deepcopy(base);mutate(p)
            with self.assertRaises(ValueError):validate_provenance(p,self.spec.precision_profile,device="cpu")
    def test_cuda_provenance_normalizes_device_and_observes_driver(self):
        import torch,os,subprocess
        from unittest.mock import patch
        from subnet.mimo_precision import provenance,validate_provenance
        configure_before_model_load(torch)
        props=NS(name="NVIDIA H200",total_memory=140*1024**3,uuid="91272d85-4ced-010b-a415-7222d2d64269")
        with patch.dict(os.environ,{"CUBLAS_WORKSPACE_CONFIG":":4096:8"}), \
             patch.object(torch.cuda,"get_device_properties",return_value=props), \
             patch.object(torch.cuda,"get_device_capability",return_value=(9,0)), \
             patch('subnet.mimo_precision.subprocess.run',return_value=subprocess.CompletedProcess([],0,stdout="580.126.16\n")) as query:
            row=provenance(torch,self.spec.precision_profile,device="cuda:0")
        self.assertEqual(row['device'],'cuda');self.assertEqual(row['hardware']['driver'],'580.126.16')
        validate_provenance(row,self.spec.precision_profile,device="cuda")
        self.assertEqual(query.call_args.args[0][-1],'--id=GPU-91272d85-4ced-010b-a415-7222d2d64269')
        row['hardware']['total_memory']=True
        with self.assertRaises(ValueError):validate_provenance(row,self.spec.precision_profile,device="cuda")


if __name__ == "__main__":unittest.main()



