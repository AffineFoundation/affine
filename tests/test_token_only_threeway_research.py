import copy
import unittest
from unittest.mock import patch
import test_forced_sampling as fixture
import test_fast_prefill_audit as fast_fixture
from subnet import token_only_threeway_research as tokens
from subnet import forced_sampling
from subnet.audit_policy import InvalidSample


class TokenOnlyThreewayControls(unittest.TestCase):
    def setUp(self):
        self.runtime, self.manifest = fast_fixture.Controls().support_runtime()
        self.manifest['token_only_verification_policy'] = dict(tokens.POLICY)
        with patch('subnet.model.create_session', return_value=fixture.Session()):
            self.original, self.arrays = self.runtime.rollout(2, 0)
        self.document = tokens.document(self.original)

    def check(self, value=None, manifest=None, eligible=None):
        with patch('subnet.environments.create_session', return_value=fixture.Session()):
            return tokens.verify(self.runtime, manifest or self.manifest,
                self.document if value is None else value, eligible_indices={2} if eligible is None else eligible)

    def test_real_honest_tokens_need_no_probability_or_proof_upload(self):
        with patch.object(self.runtime, 'compute', side_effect=AssertionError('no submitted-probability recomputation')), \
             patch.object(self.runtime, 'verify_proofs', side_effect=AssertionError('no TOPLOC claimed')):
            result = self.check()
        self.assertEqual(result['sampler_check'], 'calibrated-interior-prefill-threeway-no-cached-fallback')
        self.assertFalse(result['historical_execution_proven'])
        self.assertFalse(result['TOPLOC_verified'])

    def test_normal_path_one_prefill_no_cached_regeneration(self):
        before = self.runtime.model.calls
        with patch('subnet.fast_prefill_audit.verify_cached_reference', side_effect=AssertionError('normal prefill must not regenerate')):
            self.check()
        self.assertEqual(self.runtime.model.calls-before,1)

    def test_original_v4_draws_remain_v4_in_token_only_comparison(self):
        from test_compact_threeway_sampling import CompactThreeway
        self.runtime, self.manifest = CompactThreeway().runtime()
        self.manifest['token_only_verification_policy'] = dict(tokens.POLICY)
        with patch('subnet.model.create_session', return_value=fixture.Session()):
            rollout, _ = self.runtime.rollout(2, 0)
        self.document = tokens.document(rollout)
        before = self.runtime.model.calls
        with patch('subnet.fast_prefill_audit.verify_cached_reference', side_effect=AssertionError('no fallback')):
            self.assertTrue(self.check()['valid'])
        self.assertEqual(self.runtime.model.calls-before, 1)
        retagged = copy.deepcopy(self.manifest)
        from subnet.fast_prefill_audit import SUPPORT_VERSION
        contract = retagged['sampling_contract']
        contract.update(version=SUPPORT_VERSION, verification='prefill-cdf-calibrated', support_adjudication='exact-cached-replay-v1')
        contract.pop('uncertainty_adjudication')
        with self.assertRaises(InvalidSample):
            self.check(manifest=retagged)

    def test_numerical_ambiguity_is_not_converted_to_success_or_fraud(self):
        from subnet.fast_prefill_audit import NumericalAmbiguity
        with patch('subnet.threeway_prefill_research.verify_sampling',side_effect=NumericalAmbiguity('calibrated boundary')):
            with self.assertRaises(NumericalAmbiguity):self.check()

    def test_eighty_offpolicy_token_swaps_rejected(self):
        for number in range(80):
            bad=copy.deepcopy(self.document)
            output=bad['turns'][0]['output']
            position=number%len(output)
            output[position]=(output[position]+1+(number//len(output))%6)%7
            bad['turns'][0]['text']=self.runtime.tokenizer.decode(output)
            acts, probabilities=self.runtime.compute(bad['turns'][0]['prompt'],output)
            proofs=self.runtime.build_proofs(acts,decode_batching_size=16,topk=128)
            evidence=self.runtime.verify_proofs(acts,proofs,decode_batching_size=16,topk=128)
            self.assertTrue(evidence)
            self.assertTrue(all(r.exp_mismatches==0 and r.mant_err_mean==0 and r.mant_err_median==0 for r in evidence))
            with self.subTest(number=number),self.assertRaises(InvalidSample):self.check(bad)

    def test_rebound_approved_seed_grinding_rejects_replayed_sequence(self):
        from subnet.fast_prefill_audit import cached_sample
        rejected=0
        for seed in range(1,16):
            if cached_sample(self.runtime,[0,1],seed,0,2,'c'*64)==self.document['turns'][0]['output']:continue
            bad=copy.deepcopy(self.document);bad['seed']=seed
            bad['sampling']=forced_sampling.receipt(self.runtime.sampling_context,seed)
            with self.assertRaises(InvalidSample):self.check(bad)
            rejected+=1
        self.assertGreater(rejected,0)

    def test_default_off_and_unknown_or_null_policy(self):
        for value in (None, {}, {'version': 'unknown'},
                      {**tokens.POLICY,'historical_execution_proof':0},
                      {**tokens.POLICY,'historical_execution_proof':0.0}):
            m = copy.deepcopy(self.manifest); m['token_only_verification_policy'] = value
            with self.assertRaises(ValueError): self.check(manifest=m)
        m = copy.deepcopy(self.manifest); m.pop('token_only_verification_policy')
        with self.assertRaises(ValueError): self.check(manifest=m)

    def test_fresh_genuine_TOPLOC_cannot_legalize_selected_forged_tokens(self):
        bad = copy.deepcopy(self.original); turn = bad['turns'][0]
        turn['output'][0] = (turn['output'][0]+1) % 7
        turn['text'] = self.runtime.tokenizer.decode(turn['output'])
        acts, _ = self.runtime.compute(turn['prompt'], turn['output'])
        turn['proofs'] = self.runtime.build_proofs(acts, decode_batching_size=16, topk=128)
        self.assertTrue(turn['proofs'])
        result = fixture.Session().step({'text': turn['text']})
        for key in ('reward', 'classification'): bad[key] = turn[key] = result[key]
        with self.assertRaises(InvalidSample): self.check(tokens.document(bad))

    def test_changed_valid_attempt_with_fresh_receipt_cannot_reuse_tokens(self):
        bad = copy.deepcopy(self.document)
        # Find a prescribed alternative; changing metadata alone must not pass.
        from subnet.fast_prefill_audit import cached_sample
        for seed in range(1, 16):
            expected = cached_sample(self.runtime, [0,1], seed, 0, 2, 'c'*64)
            if expected != bad['turns'][0]['output']: break
        else: self.fail('fixture needs distinct approved attempt')
        bad['seed'] = seed; bad['sampling'] = forced_sampling.receipt(self.runtime.sampling_context, seed)
        with self.assertRaises(InvalidSample): self.check(bad)

    def test_checkpoint_draw_binding_and_task_selection(self):
        for field in ('checkpoint', 'sampling_source_hash'):
            m = copy.deepcopy(self.manifest)
            if field == 'checkpoint': m[field]['id'] = 'f'*64
            else: m[field] = 'f'*64
            with self.assertRaises((InvalidSample, ValueError)): self.check(manifest=m)
        with self.assertRaises(InvalidSample): self.check(eligible={3})
        for field, value in [('index',3),('task_hash','f'*64),('seed',16),('env_seed',1)]:
            bad = copy.deepcopy(self.document); bad[field] = value
            with self.assertRaises(InvalidSample): self.check(bad)

    def test_prompt_reward_stopping_and_transport_tampering(self):
        for field, value in [('prompt',[1,0]),('reward',float('nan')),('text','forged'),('done',False),('proofs',[])]:
            bad = copy.deepcopy(self.document); bad['turns'][0][field] = value
            with self.assertRaises(InvalidSample): self.check(bad)
        bad = copy.deepcopy(self.document); bad['turns'][0]['output']=[True]
        with self.assertRaises(InvalidSample): self.check(bad)

    def test_genuine_honest_positive_negative_pair(self):
        pair = {}
        with patch('subnet.model.create_session', return_value=fixture.Session()):
            for seed in range(16):
                rollout, arrays = self.runtime.rollout(2, seed)
                pair.setdefault(rollout['classification'], tokens.document(rollout))
                if set(pair) == {'positive', 'negative'}: break
        self.assertEqual(set(pair), {'positive', 'negative'})
        with patch('subnet.environments.create_session', return_value=fixture.Session()):
            results=tokens.verify_pair(self.runtime,{**self.manifest,'K':1,'L':1},
                [pair['positive'],pair['negative']],eligible_indices={2})
        self.assertEqual(len(results),2)
        self.assertTrue(all(r['valid'] for r in results))

    def test_pair_duplicate_and_quota_rejection(self):
        m = {**self.manifest, 'K':1, 'L':1}
        with self.assertRaises(InvalidSample): tokens.verify_pair(self.runtime,m,[self.document]*2,eligible_indices={2})
        with self.assertRaises(InvalidSample): tokens.verify_pair(self.runtime,m,[self.document],eligible_indices={2})

    def test_session_cleanup_even_failed_exact_sampler(self):
        session = fixture.Session()
        with patch('subnet.environments.create_session',return_value=session),patch.object(session,'close') as close:
            bad=copy.deepcopy(self.document);bad['turns'][0]['output'][0]=(bad['turns'][0]['output'][0]+1)%7
            with self.assertRaises(InvalidSample):tokens.verify(self.runtime,self.manifest,bad,eligible_indices={2})
            close.assert_called_once()

    def mixed_pair(self):
        pair = {}
        with patch('subnet.model.create_session', return_value=fixture.Session()):
            for seed in range(16):
                rollout, _ = self.runtime.rollout(2, seed)
                pair.setdefault(rollout['classification'], tokens.document(rollout))
                if set(pair) == {'positive', 'negative'}: break
        self.assertEqual(set(pair), {'positive', 'negative'})
        return [pair['positive'], pair['negative']]

    def test_ambiguous_first_rollout_cannot_hide_second_offpolicy_tokens(self):
        from subnet import threeway_prefill_research as sampler
        from subnet.fast_prefill_audit import NumericalAmbiguity
        pair=self.mixed_pair();first_seed=pair[0]['seed'];original=sampler.verify_sampling
        bad=pair[1];turn=bad['turns'][0]
        turn['output'][0]=(turn['output'][0]+1)%7
        turn['text']=self.runtime.tokenizer.decode(turn['output'])
        def check(runtime,rollout,*args):
            if rollout['seed']==first_seed:raise NumericalAmbiguity('first honest boundary')
            return original(runtime,rollout,*args)
        with patch('subnet.environments.create_session',side_effect=lambda _:fixture.Session()),patch.object(sampler,'verify_sampling',side_effect=check):
            with self.assertRaises(InvalidSample):tokens.verify_pair(self.runtime,{**self.manifest,'K':1,'L':1},pair,eligible_indices={2})

    def test_ambiguous_sampler_cannot_hide_same_rollout_false_reward(self):
        from subnet.fast_prefill_audit import NumericalAmbiguity
        bad=copy.deepcopy(self.document);bad['turns'][0]['reward']=float('nan')
        with patch('subnet.threeway_prefill_research.verify_sampling',side_effect=NumericalAmbiguity('honest boundary')):
            with self.assertRaises(InvalidSample):self.check(bad)

    def test_honest_pair_with_one_ambiguity_stays_unknown_after_all_checks(self):
        from subnet import threeway_prefill_research as sampler
        from subnet.fast_prefill_audit import NumericalAmbiguity
        pair=self.mixed_pair();first_seed=pair[0]['seed'];original=sampler.verify_sampling;calls=[]
        def check(runtime,rollout,*args):
            calls.append(rollout['seed'])
            if rollout['seed']==first_seed:raise NumericalAmbiguity('honest boundary')
            return original(runtime,rollout,*args)
        with patch('subnet.environments.create_session',side_effect=lambda _:fixture.Session()),patch.object(sampler,'verify_sampling',side_effect=check):
            with self.assertRaises(NumericalAmbiguity):tokens.verify_pair(self.runtime,{**self.manifest,'K':1,'L':1},pair,eligible_indices={2})
        self.assertEqual(calls,[r['seed'] for r in pair])
