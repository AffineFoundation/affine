import copy
import types
import unittest
from unittest.mock import patch

import torch
from subnet import forced_sampling as f
from subnet.audit_policy import InvalidSample
from subnet.model import Runtime


class TinyModel(torch.nn.Module):
    """Real causal computations and TOPLOC evidence without a GPU rental."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(81)
            self.embedding = torch.nn.Embedding(8, 128)
            self.head = torch.nn.Linear(128, 8)
        self.config = types.SimpleNamespace(vocab_size=8, max_position_embeddings=8192)

    def forward(self, ids, output_hidden_states=False, use_cache=False):
        assert not use_cache
        hidden = self.embedding(ids).cumsum(1)
        return types.SimpleNamespace(logits=self.head(hidden), hidden_states=[hidden])


class Session:
    def reset(self, index, seed):
        return {'task_hash': 'c' * 64, 'messages': [{'role': 'user', 'content': 'task'}]}

    def step(self, action):
        reward = float('1' in action['text'])
        return {'done': True, 'reward': reward, 'classification': 'positive' if reward else 'negative', 'observations': []}

    def close(self):
        pass


class ForcedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'epoch': 'test-forced-1', 'checkpoint': {'id': 'a' * 64},
                         'sampling_contract': f.new_contract({'version': f.VERSION, 'max_attempts': 16}),
                         'sampling_source_hash': f.source_hash()}
        self.manifest['sampling_contract']['randomness'] = 'b' * 64

    def runtime(self):
        from toploc import build_proofs_base64
        from subnet.proofs import verify_mapped_proofs
        from toploc.C.csrc.utils import get_fp_parts
        import toploc.poly as poly
        torch.set_num_threads(1)
        poly.get_fp_parts = lambda tensor: get_fp_parts(tensor, num_threads=1)
        runtime = Runtime.__new__(Runtime)
        runtime.model = TinyModel().eval()
        runtime.tokenizer = types.SimpleNamespace(eos_token_id=7, decode=lambda ids, **kw: ' '.join(map(str, ids)))
        runtime.spec = types.SimpleNamespace(id='tiny', version='v1', max_turns=1, max_output_tokens=4, config={})
        runtime.legacy = False
        runtime.harness = {'version': 'text-tools-v1', 'policy': 'autoregressive', 'max_output_tokens': 4, 'temperature': .7, 'top_p': 1.}
        runtime.prompt = lambda messages, tools=(): [0, 1]
        runtime.build_proofs = build_proofs_base64
        runtime.verify_proofs = lambda acts, proofs, **kw: verify_mapped_proofs(acts, proofs, num_threads=1, **kw)
        return f.bind_runtime(runtime, self.manifest)

    def test_real_honest_rollout_and_independent_toploc_replay(self):
        with patch('subnet.model.create_session', return_value=Session()):
            rollout, arrays = self.runtime().rollout(2, 0)
            self.assertTrue(self.runtime().verify(rollout, arrays))

    def test_credited_reports_require_exact_sampling_binding(self):
        report = {'sampling_assurance': f.assurance(self.manifest),
                  'accepted': [{'rollouts': [{'seed': 0, 'sampling': f.receipt(f.binding(self.manifest), 0)}]}]}
        f.require_report(self.manifest, report)
        for field in ('sampling_assurance', 'accepted'):
            bad = copy.deepcopy(report)
            if field == 'sampling_assurance':
                bad[field]['binding_sha256'] = '0' * 64
            else:
                bad[field][0]['rollouts'][0]['seed'] = 1
            with self.assertRaises(ValueError):
                f.require_report(self.manifest, bad)
        with self.assertRaises(ValueError):
            f.require_report(self.manifest, {'accepted': []})
        f.require_report({'epoch': 'legacy'}, {'accepted': []})

    def test_synthesized_tokens_with_fresh_genuine_proofs_rejected(self):
        with patch('subnet.model.create_session', return_value=Session()):
            miner = self.runtime()
            rollout, arrays = miner.rollout(2, 0)
            turn = rollout['turns'][0]
            # The attacker chooses different tokens and legitimately computes
            # ALL probabilities and ALL TOPLOC fingerprints on those new tokens.
            chosen = list(turn['output'])
            chosen[0] = (chosen[0] + 1) % 7
            turn['output'] = chosen
            turn['text'] = miner.tokenizer.decode(chosen)
            acts, arrays[0] = miner.compute(turn['prompt'], chosen)
            turn['proofs'] = miner.build_proofs(acts, decode_batching_size=16, topk=128)
            result = Session().step({'text': turn['text']})
            for key in ('reward', 'classification'):
                turn[key] = rollout[key] = result[key]
            # Demonstrate this exact attack passes the old computation check.
            legacy = self.runtime()
            legacy.sampling_context = None
            self.assertTrue(legacy.verify(rollout, arrays))
            with self.assertRaisesRegex(InvalidSample, 'sampling replay mismatch'):
                self.runtime().verify(rollout, arrays)

    def test_changed_attempt_and_missing_receipt_rejected(self):
        with patch('subnet.model.create_session', return_value=Session()):
            rollout, arrays = self.runtime().rollout(2, 0)
            for seed in (-1, 16, True, '0', None):
                changed = copy.deepcopy(rollout)
                changed['seed'] = seed
                with self.subTest(seed=seed), self.assertRaises(InvalidSample):
                    self.runtime().verify(changed, arrays)
            changed = copy.deepcopy(rollout)
            changed.pop('sampling')
            with self.assertRaisesRegex(InvalidSample, 'binding'):
                self.runtime().verify(changed, arrays)

    def test_valid_changed_attempt_with_rebound_receipt_cannot_reuse_tokens(self):
        with patch('subnet.model.create_session', return_value=Session()):
            runtime = self.runtime()
            rollout, arrays = runtime.rollout(2, 0)
            different = next(attempt for attempt in range(1,16)
                             if runtime.rollout(2, attempt)[0]['turns'][0]['output'] != rollout['turns'][0]['output'])
            rollout['seed'] = different
            rollout['sampling'] = f.receipt(runtime.sampling_context, different)
            with self.assertRaisesRegex(InvalidSample, 'sampling replay mismatch'):
                self.runtime().verify(rollout, arrays)

    def test_all_draw_inputs_bind_and_identity_is_excluded(self):
        context = f.binding(self.manifest)
        args = [context, 'tiny', 'c'*64, 2, 0, 0, 0]
        base = f.uniform(*args)
        for slot, replacement in [(1,'other'),(2,'d'*64),(3,3),(4,1),(5,1),(6,1)]:
            changed = list(args); changed[slot] = replacement
            self.assertNotEqual(base, f.uniform(*changed))
        for key in ('epoch', 'checkpoint'):
            changed=copy.deepcopy(args);changed[0][key]='different'
            self.assertNotEqual(base,f.uniform(*changed))
        self.assertNotIn('miner', context)
        self.assertEqual(base, f.uniform(*args))

    def test_contract_source_and_configuration_downgrades_fail(self):
        for field, value in [('version','other'), ('verification','ratio'), ('generation','cached'), ('max_attempts',True),('randomness','x'*64)]:
            changed=copy.deepcopy(self.manifest);changed['sampling_contract'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):f.binding(changed)
        changed=copy.deepcopy(self.manifest);changed['sampling_source_hash']='0'*64
        with self.assertRaisesRegex(ValueError,'source'):f.binding(changed)
        changed=copy.deepcopy(self.manifest);changed.pop('sampling_contract')
        with self.assertRaisesRegex(ValueError,'missing'):f.binding(changed)
        for changes in [dict(policy='candidates',candidates=['a','b']),dict(version='text-tools-long-kv-v3'),dict(turn_overrides={'0':{'temperature':1.}})]:
            with self.assertRaises(ValueError):f.validate_harness({**self.runtime().harness,**changes})

    def test_inverse_cdf_edges_and_top_p_zero_mass(self):
        logits=torch.zeros(4)
        self.assertEqual(f.pick(logits,0.,1.,1.),0)
        self.assertEqual(f.pick(logits,.25,1.,1.),1)
        self.assertEqual(f.pick(logits,1.-2**-53,1.,1.),3)
        for u in [0.,.2,.9,1.-2**-53]:
            self.assertIn(f.pick(logits,u,1.,.5),(0,1))

    def test_context_does_not_alias_mutable_manifest(self):
        context=f.binding(self.manifest)
        self.manifest['sampling_contract']['randomness']='0'*64
        self.assertEqual(context['contract']['randomness'],'b'*64)

    def test_legacy_contract_is_not_relabelled_sampling_verified(self):
        self.assertIsNone(f.binding({'epoch':'legacy'}))
        self.assertFalse(f.assurance({'epoch':'legacy'})['sampling_required'])


if __name__ == '__main__':
    unittest.main()
