"""Real canonical dispatch/codec controls; model and native data are CPU fixtures."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from test_complete_group_fixtures import fixture, FixtureRuntime, digest
from test_complete_group_transport import FakeBucket, FakeHTTP
from subnet import forced_sampling as forced, harness
from subnet.batch_contract import for_manifest, REQUIRED_CLIENT_FEATURES
from subnet.complete_group_distribution import (distribution, pick, MIMO_VERSION, MIMO_HARNESS,
                                               MIMO_TRANSPORT, MIMO_EOS, verify_intervals)
from subnet.complete_group_sampling import prepare_draws, cached_sample
from subnet.complete_group_verification import exact_replay, sampling_check
from subnet.complete_group_runtime import outcome, completed_answer
from subnet.complete_group_transport import CompleteGroupTransport, capability
from subnet.complete_group_admission import TaskAuthority, structural_batch
from subnet.complete_group_codec import decode, limits
from subnet.miner import Miner
from subnet.storage import Identity
from subnet.audit_policy import InvalidSample
from subnet.fast_prefill_audit import NumericalAmbiguity, SupportMismatch
from subnet.complete_group_mining import _rollout_from_runtime, validate_group, BATCH_VERSION


def mimo_manifest(n=8):
    m = fixture(n)
    m['checkpoint']['files'] = {'config.json':'c'*64,'model.safetensors':'d'*64}
    m.update(epoch='private-mimo-fixture-n'+str(n), run_id='private-mimo-new-run', max_batches=9,
             mimo_runtime={'version':'qwen3_5-mimo-v1','fixture_only':True})
    m['sampling_contract']['version'] = MIMO_VERSION
    m['group_transport_policy'] = dict(version=MIMO_TRANSPORT, max_context_tokens=64,
                                     eos_token_ids=list(MIMO_EOS), vocab_size=248320)
    h = dict(version=MIMO_HARNESS, policy='autoregressive', max_output_tokens=12,
             max_context_tokens=64, temperature=1., top_p=.95, top_k=20, enable_thinking=True)
    m['environments'][0]['harness'] = harness.normalize(h)
    m['sampling_contract']['calibration']['harness_sha256'] = digest(h)
    return m


class Tokenizer:
    eos_token_id = 248044
    def __len__(self): return 248320


class ToyModel(torch.nn.Module):
    """Two-position deterministic logits, not a MiMo hardware qualification."""
    def __init__(self, stop=248046):
        super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=248320, max_position_embeddings=64))
        self.stop = stop
    def forward(self, inputs, past_key_values=None, use_cache=True):
        logits = torch.full((1, 1, 248320), -50.)
        if past_key_values is None:
            logits[0, 0, :24] = torch.arange(24).float() / 10
        else:
            logits[0, 0, self.stop] = 50.
        return SimpleNamespace(logits=logits, past_key_values=1)


def toy_runtime(m, miner, stop=248046):
    d = m['environments'][0]
    return SimpleNamespace(model=ToyModel(stop), tokenizer=Tokenizer(), eos_token_ids=MIMO_EOS,
        spec=SimpleNamespace(**d['spec']), env_config=d['spec'], harness=d['harness'],
        sampling_context=forced.binding(m, miner), fast_sampling_calibration=m['sampling_contract']['calibration'])


class MimoCollectionRuntime(FixtureRuntime):
    def __init__(self, manifest, miner, outcomes=None):
        super().__init__(manifest, miner, outcomes)
        self.tokenizer = Tokenizer(); self.eos_token_ids = MIMO_EOS
        self.model = ToyModel()
    def rollout(self, index, nonce):
        row, arrays = super().rollout(index, nonce)
        turn = row['turns'][0]
        if turn['output'][-1] == 15:
            turn['output'][-1] = MIMO_EOS[nonce % 2]
        self.complete_group_sample_evidence['output_sha256'] = digest(turn['output'])
        return row, arrays


class DistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def test_top_k_precedes_top_p_with_stable_ties(self):
        values = torch.zeros(30)
        p = distribution(values, 1., .95, 20)
        # Exactly equal logits select low token IDs before nucleus filtering.
        self.assertTrue(torch.equal(torch.nonzero(p).flatten(), torch.arange(19)))
        self.assertAlmostEqual(float(p.sum()), 1.)
        self.assertTrue(torch.equal(p, distribution(values.unsqueeze(0), 1., .95, 20)[0]))
    def test_sampling_temperature_changes_actual_selected_probability(self):
        logits = torch.tensor([0., 1., 2.])
        self.assertFalse(torch.equal(distribution(logits, 1., 1., 0), distribution(logits, .5, 1., 0)))
    def test_boundary_zero_mass_and_nonfinite_do_not_pass(self):
        p = distribution(torch.tensor([4., 2., 1.]), 1., 1., 1)
        self.assertEqual(pick(p, 0.), 0); self.assertEqual(pick(p, 1.-2**-53), 0)
        with self.assertRaises(RuntimeError): distribution(torch.tensor([float('nan'), 0.]), 1., 1., 0)
        with self.assertRaises(SupportMismatch):
            verify_intervals(torch.tensor([[4., 2., 1.]]), [1], [.5],
                             {'temperature':1.,'top_p':1.,'top_k':1}, 1e-4)
    def test_both_stop_tokens_terminate_and_capture_actual_lp(self):
        m=mimo_manifest(); miner='a'*64
        for stop in MIMO_EOS:
            r=toy_runtime(m, miner, stop)
            output=cached_sample(r,[1,2],0,0,0,'b'*64)
            self.assertEqual(output[-1], stop);self.assertEqual(len(output),2)
            self.assertEqual(r.complete_group_sample_evidence['stop_reason'],'eos')
            self.assertEqual(len(r.complete_group_sample_evidence['behavior_selected_logprobs']),2)
            exact_replay(r,[1,2],output,0,0,'b'*64)
            changed=output.copy();changed[0]=(changed[0]+1)%24
            with self.assertRaises(InvalidSample): exact_replay(r,[1,2],changed,0,0,'b'*64)
    def test_fast_boundary_or_support_falls_back_to_exact_not_automatic_pass(self):
        m=mimo_manifest();r=toy_runtime(m,'a'*64)
        output=cached_sample(r,[1,2],0,0,0,'b'*64)
        row=dict(seed=0,index=0,task_hash='b'*64)
        with patch('subnet.complete_group_verification.verify_intervals',side_effect=NumericalAmbiguity('fixture-boundary')):
            result=sampling_check(r,row,[1,2],output,np.zeros((2,248320),np.float32))
            self.assertTrue(result['cached_reference_adjudication'])
            output[0]=(output[0]+1)%24
            with self.assertRaises(InvalidSample):sampling_check(r,row,[1,2],output,np.zeros((2,248320),np.float32))


class BindingTests(unittest.TestCase):
    def test_exact_nonce_bounds_and_every_signed_identity_changes_draw(self):
        m=mimo_manifest();context=forced.binding(m,'a'*64)
        reference=prepare_draws(context,'math','b'*64,0,0,0,[1,2])(0)
        for nonce in (-1,8,True,1.0):
            with self.assertRaises(ValueError):prepare_draws(context,'math','b'*64,0,nonce,0,[1,2])
        for field,value in [('run_id','another-run'),('epoch','next-epoch'),('source_bundle',{'sha256':'e'*64}),
                            ('mimo_runtime',{'version':'other-runtime'}),
                            ('checkpoint',dict(m['checkpoint'],id='c'*64))]:
            other=deepcopy(m);other[field]=value
            self.assertNotEqual(reference,prepare_draws(forced.binding(other,'a'*64),'math','b'*64,0,0,0,[1,2])(0))
        self.assertNotEqual(reference,prepare_draws(forced.binding(m,'c'*64),'math','b'*64,0,0,0,[1,2])(0))
        self.assertNotEqual(reference,prepare_draws(context,'math','b'*64,0,0,0,[1,3])(0))
        self.assertNotEqual(reference,prepare_draws(context,'math','c'*64,0,0,0,[1,2])(0))
    def test_mimo_read_url_refresh_preserves_draws_but_files_do_not(self):
        m=mimo_manifest()
        def draw(value):
            return prepare_draws(forced.binding(value,'a'*64),'math','b'*64,0,0,0,[1,2])(0)
        expected=draw(m)
        renewed=deepcopy(m);renewed['checkpoint']['read_urls']={'config.json':'https://fixture.invalid/refreshed'}
        self.assertEqual(draw(renewed),expected)
        renewed['checkpoint']['files']=dict(renewed['checkpoint']['files'],newfile='1'*64)
        self.assertNotEqual(draw(renewed),expected)
        legacy=fixture();renewed=deepcopy(legacy)
        renewed['checkpoint']['read_urls']={'config.json':'https://fixture.invalid/refreshed'}
        self.assertNotEqual(forced.binding(legacy,'a'*64),forced.binding(renewed,'a'*64))
    def test_final_channel_policy_is_bound_and_cannot_be_reconfigured(self):
        from subnet.mimo_answer_projection import POLICY, POLICY_KEY
        m=mimo_manifest();old=forced.binding(m,'a'*64)
        m[POLICY_KEY]=deepcopy(POLICY)
        context=forced.binding(m,'a'*64)
        self.assertNotEqual(prepare_draws(old,'math','b'*64,0,0,0,[1,2])(0),
                            prepare_draws(context,'math','b'*64,0,0,0,[1,2])(0))
        context[POLICY_KEY]['thinking_open_token_id']+=1
        with self.assertRaises(ValueError):prepare_draws(context,'math','b'*64,0,0,0,[1,2])
    def test_unknown_or_legacy_aliases_cannot_change_group_population(self):
        for field,value in [('K',4),('max_attempts',1000),('nonce_start',1)]:
            m=mimo_manifest();m[field]=value
            with self.assertRaises(ValueError):for_manifest(m,supported_features=REQUIRED_CLIENT_FEATURES)
        m=mimo_manifest();m['sampling_contract']['max_attempts']=1000
        with self.assertRaises(ValueError):forced.binding(m,'a'*64)
        m=mimo_manifest();m['group_transport_policy']['eos_token_ids'].reverse()
        with self.assertRaises(ValueError):forced.binding(m,'a'*64)
    def test_changed_sampler_or_unknown_harness_setting_rejected(self):
        for field,value in [('top_k',19),('temperature',.6),('top_p',1.),('enable_thinking',1),('salt','x')]:
            m=mimo_manifest();m['environments'][0]['harness'][field]=value
            with self.assertRaises(ValueError):forced.binding(m,'a'*64)
    def test_legacy_uniform_remains_byte_identical(self):
        import hashlib
        context={'contract':{'version':forced.VERSION,'max_attempts':8},'epoch':'old','checkpoint':'a'*64}
        message=dict(context,environment='math',task_hash='b'*64,index=1,attempt=2,turn=0,position=3)
        expected=(int.from_bytes(hashlib.sha256(forced.canonical(message)).digest()[:8],'big')>>11)/2**53
        self.assertEqual(forced.uniform(context,'math','b'*64,1,2,0,3),expected)


class MinerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.identity=Identity(bytes(range(32)));self.bucket=FakeBucket();self.http=FakeHTTP(self.bucket)
    def actor(self,m):
        m['capabilities']={self.identity.id:capability(self.bucket,m,self.identity.id,registered_miners={self.identity.id})}
        with patch('subnet.model.check_runtime_profile'):
            actor=Miner(self.identity,m,'CPU-FIXTURE',complete_group_transport=CompleteGroupTransport(http=self.http))
        self.addCleanup(actor._group_journal.close);return actor
    def test_n4_n8_n16_complete_no_replacement_and_duplicate_output_accepted(self):
        for n in (4,8,16):
            m=mimo_manifest(n);a=self.actor(m);r=MimoCollectionRuntime(m,self.identity.id)
            with patch('subnet.miner.make_runtime',return_value=r):b=a.search(0,max_attempts=32)
            self.assertEqual(r.calls,[(0,i) for i in range(n)])
            self.assertEqual(a.search(0),b);self.assertEqual(len(r.calls),n)
            self.assertEqual([row['seed'] for row in b['rollouts']],list(range(n)))
            restored,arrays=decode(a._group_prepared[0][1],limits(m))[0]
            self.assertEqual(restored,b);self.assertEqual(len(arrays),n)
            self.assertEqual(a.upload(),200)
    def test_below_n_budget_rejects_before_model_creation(self):
        a=self.actor(mimo_manifest())
        for value in (7,False,'32',1.0):
            with patch('subnet.miner.make_runtime') as load:
                with self.assertRaises(ValueError):a.search(0,max_attempts=value)
                load.assert_not_called()
    def test_missing_reordered_reused_or_changed_attempt_rejects(self):
        m=mimo_manifest();a=self.actor(m);r=MimoCollectionRuntime(m,self.identity.id)
        with patch('subnet.miner.make_runtime',return_value=r):b=a.search(0)
        authority=TaskAuthority(m['epoch'],m['checkpoint']['id'],'math','taskset-1',0,
            b['task_hash'],(1,2),248044,248320,MIMO_EOS)
        structural_batch(b,m,self.identity.id,authority)
        variants=[]
        x=deepcopy(b);x['rollouts'].pop();variants.append(x)
        x=deepcopy(b);x['rollouts'][2]=deepcopy(x['rollouts'][1]);variants.append(x)
        x=deepcopy(b);x['rollouts'][1],x['rollouts'][2]=x['rollouts'][2],x['rollouts'][1];variants.append(x)
        x=deepcopy(b);x['rollouts'][2]['turns'][0]['output'][0]+=1;variants.append(x)
        x=deepcopy(b);x['rollouts'][2]['sampling']['attempt']=999;variants.append(x)
        for changed in variants:
            with self.assertRaises(ValueError):structural_batch(changed,m,self.identity.id,authority)


class OutcomesTests(unittest.TestCase):
    def test_caps_and_unfinished_eos_never_fill_wrong_quota(self):
        wrong=dict(classification='negative',reward=0.,done=True)
        good=dict(classification='positive',reward=1.,done=True)
        self.assertEqual(outcome(wrong,'\\boxed{2}','eos'),('negative',0.))
        self.assertEqual(outcome(good,'\\boxed{2}','eos'),('positive',1.))
        for text in ('unfinished', '\\boxed{', '\\boxed{}', '\\boxed{2} trailing', '\\boxed{1} \\boxed{2}'):
            self.assertEqual(outcome(wrong,text,'eos'),('neutral',0.))
        self.assertEqual(outcome(good,'\\boxed{2}','budget'),('neutral',0.))
    def test_malformed_native_grade_is_not_a_failure(self):
        with self.assertRaises(ValueError):outcome({'classification':'error','reward':0,'done':True},'\\boxed{2}','eos')
        with self.assertRaises(ValueError):outcome({'classification':'negative','reward':0,'done':False},'\\boxed{2}','eos')


class AuditPathTests(unittest.TestCase):
    """Real draws, selected arrays and TOPLOC; native answers are an explicit fixture."""
    def setUp(self):
        torch.set_num_threads(2)
        self.m = mimo_manifest(); self.miner = 'a'*64
        self.r = toy_runtime(self.m, self.miner)
        self.r.prompt = lambda messages, tools=(): [1,2]
        self.r.tokenizer.decode = lambda output, skip_special_tokens=True: '\\boxed{'+str(output[0])+'}'
        # Use actual proof construction/verification on deterministic activations.
        from toploc import build_proofs_base64
        from subnet.proofs import verify_mapped_proofs
        self.r.build_proofs = build_proofs_base64
        self.r.verify_proofs = lambda acts, proofs, **kwargs: verify_mapped_proofs(acts, proofs, num_threads=2, **kwargs)
        import toploc.poly as toploc_poly
        from toploc.C.csrc.utils import get_fp_parts
        self.original_parts = toploc_poly.get_fp_parts
        toploc_poly.get_fp_parts = lambda tensor:get_fp_parts(tensor, num_threads=2)
        self.addCleanup(lambda:setattr(toploc_poly,'get_fp_parts',self.original_parts))
        def compute(prompt, output):
            first=self.r.model(torch.tensor([prompt])).logits[0,-1]
            second=self.r.model(torch.tensor([[output[0]]]),past_key_values=1).logits[0,-1]
            full=torch.log_softmax(torch.stack([first,second]),-1).numpy()
            ids=prompt+output
            hidden=(torch.arange(len(ids)*256).reshape(len(ids),256)%257+1).to(torch.bfloat16)
            return [hidden[:len(prompt)]]+[hidden[i:i+1] for i in range(len(prompt),len(ids))],full
        self.r.compute=compute
        owner=self
        class Session:
            def reset(self,index,seed):return {'task_hash':'b'*64,'messages':[{'role':'user','content':'fixture'}]}
            def step(self,action):
                token=int(action['text'].split('{')[1].split('}')[0]);good=token%2==0
                return {'classification':'positive' if good else 'negative','reward':float(good),'done':True,'observations':[]}
            def close(self):pass
        self.sessions=patch('subnet.complete_group_verification.create_session',side_effect=lambda spec:Session())
        self.sessions.start();self.addCleanup(self.sessions.stop)
        self.rows=[];self.arrays=[]
        for nonce in range(8):
            output=cached_sample(self.r,[1,2],nonce,0,0,'b'*64)
            text=self.r.tokenizer.decode(output);kind='positive' if output[0]%2==0 else 'negative';reward=float(kind=='positive')
            acts,full=compute([1,2],output)
            turn=dict(prompt=[1,2],output=output,text=text,proofs=build_proofs_base64(acts,decode_batching_size=16,topk=128),
                observations=[],done=True,reward=reward,classification=kind)
            row=dict(schema=2,env_id='math',environment_version='taskset-1',index=0,sample_index=0,seed=nonce,env_seed=0,
                task_hash='b'*64,reward=reward,classification=kind,turns=[turn],sampling=forced.receipt(self.r.sampling_context,nonce))
            self.rows.append(_rollout_from_runtime(self.r,row,nonce))
            self.arrays.append([full[np.arange(2),output].reshape(-1,1).astype(np.float32)])
        self.batch=dict(schema=BATCH_VERSION,epoch=self.m['epoch'],checkpoint=self.m['checkpoint']['id'],miner=self.miner,
            env_id='math',environment_version='taskset-1',index=0,sample_index=0,task_hash='b'*64,
            group_contract_sha256=digest(self.m['group_contract']),sampling_context_sha256=digest(self.r.sampling_context),rollouts=self.rows)

    def test_full_canonical_group_audit_and_report_binding(self):
        from subnet.complete_group_codec import encode
        from subnet.complete_group_verification import verify
        self.assertTrue(any(r['classification']=='positive' for r in self.rows))
        self.assertTrue(any(r['classification']=='negative' for r in self.rows))
        data=encode([(self.batch,self.arrays)],limits(self.m))
        result=verify(data,self.m,'toy',miner=self.miner,runtime_factory=lambda *a,**k:self.r)
        self.assertEqual(len(result['outcomes'][0]['rollouts']),8)
        self.assertEqual(result['accepted'],[self.batch])
        forced.require_report(self.m,result)
        with self.assertRaises(ValueError):verify(data,self.m,'toy',miner=None,runtime_factory=lambda *a,**k:self.r)

    def test_fabricated_token_with_new_valid_toploc_and_logprobs_still_fails_draw(self):
        from subnet.complete_group_verification import verify_rollout
        row=deepcopy(self.rows[0]);turn=row['turns'][0]
        turn['output'][0]=(turn['output'][0]+2)%24
        turn['text']=self.r.tokenizer.decode(turn['output'])
        acts,full=self.r.compute(turn['prompt'],turn['output'])
        turn['proofs']=self.r.build_proofs(acts,decode_batching_size=16,topk=128)
        arrays=[full[np.arange(2),turn['output']].reshape(-1,1).astype(np.float32)]
        with self.assertRaises(InvalidSample):verify_rollout(self.r,row,arrays)

    def test_native_grader_infrastructure_never_becomes_accepted_negative(self):
        from subnet.complete_group_verification import verify_rollout
        with patch('subnet.complete_group_verification.create_session',side_effect=RuntimeError('native infra fixture')):
            with self.assertRaisesRegex(RuntimeError,'native infra'):verify_rollout(self.r,self.rows[0],self.arrays[0])


if __name__ == '__main__': unittest.main()
