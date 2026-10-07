"""Actual frozen f213 source and CPU tensors; no model downloads/GPU/replay jobs."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
import uuid
from unittest.mock import patch
import torch
from ops.f213_sampling_diagnostics import install, PrivateReceiptSink, SOURCE_SHA256

FROZEN = Path(os.environ.get('F213_DIAGNOSTIC_TEST_SOURCE', str(Path(__file__).parent / 'fixtures/f213_sampling_diagnostic_source')))


def load_actual():
    name = '_actual_f213_' + uuid.uuid4().hex
    package = types.ModuleType(name)
    package.__path__ = [str(FROZEN / 'subnet')]
    sys.modules[name] = package
    path = FROZEN / 'subnet/fast_prefill_audit.py'
    pins=json.loads((Path(__file__).parent/'fixtures/f213_sampling_diagnostic_source/source-pins.json').read_bytes())
    for member, expected in pins.items():
        assert hashlib.sha256((FROZEN/'subnet'/member).read_bytes()).hexdigest()==expected
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA256
    spec = importlib.util.spec_from_file_location(name + '.fast_prefill_audit', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CPUCachedModel(torch.nn.Module):
    def __init__(self, failure=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.calls = 0
        self.failure = failure

    def forward(self, ids, past_key_values=None, use_cache=False):
        self.calls += 1
        if self.failure == self.calls:
            raise RuntimeError('bounded CPU fixture unavailable')
        # Actual frozen pick sees these logits for each original public draw.
        logits = torch.tensor([[[3., 2., 1., 0.]]]).expand(1, ids.shape[1], 4)
        return types.SimpleNamespace(logits=logits, past_key_values=self.calls)


class FrozenDiagnostics(unittest.TestCase):
    def setUp(self):
        self.fast = load_actual()
        self.records = []
        self.runtime = types.SimpleNamespace(
            spec=types.SimpleNamespace(id='math'),
            tokenizer=types.SimpleNamespace(eos_token_id=3),
            harness={'max_output_tokens':3, 'temperature':1., 'top_p':1.},
            model=CPUCachedModel(),
            sampling_context={'epoch':'CPU-review', 'checkpoint':'a'*64,
                              'contract':{'version':self.fast.SUPPORT_VERSION, 'max_attempts':16}},
            fast_sampling_calibration={'cdf_abs_error':1e-3})
        self.rollout = {'task_hash':'c'*64, 'index':2, 'seed':0}
        self.prompt = [0, 1]

    def enable(self, sink=None):
        undo = install(self.fast, sink or self.records.append, enabled=True,
                       original_job_sha256='d'*64)
        self.addCleanup(undo)

    def call(self, output, logprobs):
        return self.fast.verify_sampling(self.runtime, self.rollout, 0,
                                         self.prompt, output, logprobs)

    def draws(self, n):
        forced = __import__(self.fast.__package__ + '.forced_sampling', fromlist=['uniform'])
        return [forced.uniform(self.runtime.sampling_context, 'math', 'c'*64, 2, 0, 0, i)
                for i in range(n)]

    def genuine_output(self, n=3):
        forced = __import__(self.fast.__package__ + '.forced_sampling', fromlist=['pick'])
        logits = torch.tensor([3.,2.,1.,0.])
        return [forced.pick(logits, u, 1., 1.) for u in self.draws(n)]

    def interval_logits(self, outputs, expanded=False):
        # Build actual normalized distributions with claimed token0's right
        # boundary just below the original public draw. No verify method mocks.
        values=[]
        for i, u in enumerate(self.draws(len(outputs))):
            if expanded:
                q = u-5e-4
            else:
                q = u+5e-4
            values.append([q, (1-q)*.5, (1-q)*.3, (1-q)*.2])
        return torch.log(torch.tensor(values, dtype=torch.float32))

    def test_default_off_does_not_read_source_or_patch(self):
        original = self.fast.verify_sampling
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('disabled read')):
            undo = install(self.fast, self.records.append)
        undo()
        self.assertIs(self.fast.verify_sampling, original)
        self.assertEqual(self.records, [])

    def test_loaded_bytecode_difference_refused_before_activation(self):
        self.fast.verify_intervals = lambda *a, **k: {'all_intervals_verified':True}
        with self.assertRaisesRegex(ValueError, 'loaded frozen sampling code differs'):
            self.enable()
        self.assertNotIn('_f213_sampling_diagnostic_hook', self.fast.__dict__)

    def test_inside_near_boundary_is_pass_without_fallback(self):
        # Direct actual frozen interval implementation, instrumented within real
        # verify_sampling using original public uniforms.
        output=[0]*3
        probs=self.interval_logits(output)
        original=self.call(output, probs)
        self.enable()
        actual=self.call(output, probs)
        self.assertEqual(actual, original)
        r=self.records[-1]
        self.assertEqual(r['prefill_positions_checked'], 3)
        self.assertEqual(r['exact_interval_pass_positions'], 3)
        self.assertEqual(r['cached_replay_invocations'], 0)
        self.assertEqual(r['sampling_outcome'], 'pass')

    def test_genuine_expanded_only_ambiguity_cached_invalid_preserved(self):
        output=[0]*3
        probs=self.interval_logits(output, expanded=True)
        try:
            original=self.call(output, probs)
            original_type=None
        except Exception as exc:
            original_type=type(exc)
        self.runtime.model.calls=0
        self.enable()
        if original_type:
            with self.assertRaises(original_type):self.call(output, probs)
        else:
            self.assertEqual(self.call(output, probs), original)
        r=self.records[-1]
        self.assertEqual(r['expanded_only_ambiguity_positions'], 3)
        self.assertEqual(r['fallback_reason'], 'NumericalAmbiguity')
        self.assertEqual(r['cached_replay_invocations'], 1)
        self.assertGreater(r['cached_replay_tokens_compared'], 0)
        self.assertEqual(r['cached_replay_forwards_started'], self.runtime.model.calls)
        self.assertGreaterEqual(r['cached_replay_seconds'], 0)

    def test_unsupported_honest_cached_pass_and_partial_forgery_invalid(self):
        output=self.genuine_output()
        # Full length may contain EOS; original requires no early EOS. Find a
        # genuine original attempt whose first3 draws are not EOS.
        while 3 in output:
            self.rollout['seed'] += 1
            forced=__import__(self.fast.__package__+'.forced_sampling',fromlist=['pick','uniform'])
            output=[forced.pick(torch.tensor([3.,2.,1.,0.]),forced.uniform(self.runtime.sampling_context,'math','c'*64,2,self.rollout['seed'],0,i),1.,1.)for i in range(3)]
        self.runtime.harness['top_p']=.9
        forced=__import__(self.fast.__package__+'.forced_sampling',fromlist=['pick','uniform'])
        output=[forced.pick(torch.tensor([3.,2.,1.,0.]),forced.uniform(self.runtime.sampling_context,'math','c'*64,2,self.rollout['seed'],0,i),1.,.9)for i in range(3)]
        probs=torch.full((3,4),-30.)
        for i,t in enumerate(output):probs[i,(t+1)%4]=0.
        original=self.call(output,probs)
        self.runtime.model.calls=0
        self.enable()
        self.assertEqual(self.call(output,probs),original)
        r=self.records[-1]
        self.assertEqual(r['unsupported_token_positions'],3)
        self.assertEqual(r['cached_replay_tokens_compared'],3)
        self.assertEqual(r['cached_replay_tokens_matched'],3)
        self.assertEqual(r['cached_replay_outcome'],'pass')
        forged=output.copy();forged[1]=(forged[1]+1)%3
        for i,t in enumerate(forged):probs[i].fill_(-30.);probs[i,(t+1)%4]=0.
        with self.assertRaises(self.fast.InvalidSample):self.call(forged,probs)
        r=self.records[-1]
        self.assertEqual(r['cached_replay_tokens_compared'],2)
        self.assertEqual(r['cached_replay_tokens_matched'],1)
        self.assertEqual(r['cached_replay_first_mismatch_position'],1)
        self.assertEqual(r['cached_replay_outcome'],'InvalidSample')
        self.assertEqual(r['sampling_outcome'],'InvalidSample')

    def test_cached_unavailable_remains_ambiguity_and_emits_partial_counts(self):
        self.runtime.model=CPUCachedModel(failure=1)
        self.enable()
        with self.assertRaises(self.fast.NumericalAmbiguity):
            self.call([0]*3,self.interval_logits([0]*3,expanded=True))
        r=self.records[-1]
        self.assertEqual(r['cached_replay_invocations'],1)
        self.assertEqual(r['cached_replay_forwards_started'],1)
        self.assertEqual(r['cached_replay_forwards_completed'],0)
        self.assertEqual(r['cached_replay_tokens_compared'],0)
        self.assertEqual(r['cached_replay_outcome'],'RuntimeError')
        self.assertEqual(r['sampling_outcome'],'NumericalAmbiguity')

    def test_outside_expanded_immediate_invalid_has_no_cached_replay(self):
        self.enable()
        probs=torch.log(torch.tensor([[1e-9,.5,.3,.2]]).expand(3,4))
        with self.assertRaises(self.fast.InvalidSample):self.call([0]*3,probs)
        r=self.records[-1]
        self.assertEqual(r['outside_expanded_positions'],3)
        self.assertEqual(r['cached_replay_invocations'],0)
        self.assertEqual(r['sampling_outcome'],'InvalidSample')

    def test_old_contract_support_does_not_gain_replay(self):
        self.runtime.sampling_context['contract']['version']=self.fast.VERSION
        self.runtime.harness['top_p']=.9
        self.enable()
        probs=torch.log(torch.tensor([[.999999,.000001,1e-20,1e-20]]).expand(3,4))
        with self.assertRaises(self.fast.SupportMismatch):self.call([1]*3,probs)
        self.assertEqual(self.records[-1]['cached_replay_invocations'],0)
        self.assertEqual(self.records[-1]['fallback_reason'],'SupportMismatch')

    def test_bad_framing_emits_without_calling_interval_or_model(self):
        self.enable()
        with self.assertRaises(self.fast.InvalidSample):self.call([0],torch.zeros(1,4))
        r=self.records[-1]
        self.assertEqual(r['prefill_positions_checked'],0)
        self.assertEqual(r['cached_replay_invocations'],0)
        self.assertEqual(r['sampling_outcome'],'InvalidSample')

    def test_receipt_sink_failure_does_not_change_sampler_verdict(self):
        output=[0]*3;probs=self.interval_logits(output);expected=self.call(output,probs)
        def broken(record):raise OSError('fixture sink failure')
        self.enable(broken)
        with self.assertLogs('ops.f213_sampling_diagnostics',level='WARNING'):
            self.assertEqual(self.call(output,probs),expected)

    def test_private_sink_exclusive_small_files_and_mode_controls(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d).resolve();path.chmod(0o700);sink=PrivateReceiptSink(path)
            self.enable(sink)
            self.call([0]*3,self.interval_logits([0]*3))
            files=list(path.iterdir());self.assertEqual(len(files),1)
            self.assertEqual(stat.S_IMODE(files[0].stat().st_mode),0o600)
            self.assertEqual(files[0].stat().st_nlink,1)
            record=json.loads(files[0].read_bytes())
            self.assertNotIn('logprobs',record)
            self.assertEqual(record['original_job_sha256'],'d'*64)
            path.chmod(0o755)
            with self.assertRaises(ValueError):PrivateReceiptSink(path)

    def test_binding_capture_failure_keeps_original_stop_verdict(self):
        self.rollout = {}
        with self.assertRaises(self.fast.InvalidSample):self.call([0],torch.zeros(1,4))
        self.enable()
        with self.assertRaises(self.fast.InvalidSample):self.call([0],torch.zeros(1,4))
        self.assertEqual(self.records[-1]['sampling_outcome'],'InvalidSample')
        self.assertEqual(self.records[-1]['binding_capture_error_type'],'KeyError')

    def test_private_sink_rejects_replaced_directory_and_existing_target(self):
        with tempfile.TemporaryDirectory()as parent:
            path=Path(parent).resolve()/'owned';path.mkdir(mode=0o700)
            sink=PrivateReceiptSink(path);original=path.with_name('preserved')
            path.rename(original);path.mkdir(mode=0o700)
            with self.assertRaisesRegex(ValueError,'namespace changed'):sink({'bounded':1})
            self.assertEqual(list(path.iterdir()),[])
            self.assertEqual(list(original.iterdir()),[])
        with tempfile.TemporaryDirectory()as d:
            path=Path(d).resolve();path.chmod(0o700);sink=PrivateReceiptSink(path)
            with patch('ops.f213_sampling_diagnostics.time.time_ns',return_value=1):
                sink({'bounded':1})
                with self.assertRaises(FileExistsError):sink({'bounded':1})
            self.assertEqual(len(list(path.iterdir())),1)

    def test_actual_Runtime_verify_relative_import_hits_overlay_after_LP_TOPLOC(self):
        import ast, math
        import numpy as np
        path=FROZEN/'subnet/model.py'
        tree=ast.parse(path.read_bytes())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Runtime')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='verify')
        class Session:
            def reset(self,index,seed):return {'task_hash':'c'*64,'messages':[]}
            def step(self,action):return {'done':True,'reward':1.,'observations':[], 'classification':'positive'}
            def close(self):pass
        namespace={'__package__':self.fast.__package__,'math':math,
                   'create_session':lambda spec:Session(),
                   'validate_framing':lambda proofs,count:None,
                   'policy':types.SimpleNamespace(action=lambda text,harness:{'text':text},observations=lambda o,h:[])}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
        output=[0]*3;probs=self.interval_logits(output).numpy()
        self.runtime.spec=types.SimpleNamespace(id='math',version='CPU-test',config={},max_turns=1,max_output_tokens=3)
        self.runtime.model.config=types.SimpleNamespace(vocab_size=4,max_position_embeddings=8192)
        self.runtime.prompt=lambda messages,tools:self.prompt
        self.runtime.tokenizer.decode=lambda output,skip_special_tokens:'CPU fixture'
        self.runtime.sampling_receipt=lambda attempt:{'sampling':{'attempt':attempt}}
        self.runtime.compute=lambda prompt,output:([],probs)
        self.runtime.verify_proofs=lambda *a,**k:[types.SimpleNamespace(exp_mismatches=0,mant_err_mean=0,mant_err_median=0)]*2
        self.runtime.legacy=False
        self.runtime.fast_sampling_calibration.update(logprob_atol=1e-5,toploc_exp_mismatches=0,toploc_mant_err_mean=0,toploc_mant_err_median=0)
        rollout={**self.rollout,'reward':1.,'classification':'positive','sampling':{'attempt':0},
                 'turns':[{'prompt':self.prompt,'output':output,'text':'CPU fixture','done':True,
                           'reward':1.,'classification':'positive','observations':[],'proofs':[]}]}
        self.assertTrue(namespace['verify'](self.runtime,rollout,[probs.copy()]))
        self.enable()
        self.assertTrue(namespace['verify'](self.runtime,rollout,[probs.copy()]))
        self.assertEqual(len(self.records),1)
        self.assertEqual(self.records[-1]['exact_interval_pass_positions'],3)
        # Mandatory original LP comparison still runs before the diagnostic
        # sampling route: a changed LP claim fails and produces no sampler record.
        wrong=probs.copy();wrong[0,0]+=1.
        with self.assertRaises(self.fast.InvalidSample):namespace['verify'](self.runtime,rollout,[wrong])
        self.assertEqual(len(self.records),1)
