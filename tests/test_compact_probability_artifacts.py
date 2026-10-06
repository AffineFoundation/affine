import copy
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_forced_sampling as tiny
import test_fast_prefill_audit as prefill
from subnet import forced_sampling as sampling
from subnet import probability_artifacts as artifacts
from subnet.audit_policy import InvalidSample
from subnet.batches import pack, unpack

POLICY = {'version': artifacts.VERSION}


class CompactProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tiny.ForcedSamplingTests()
        self.fixture.setUp()
        self.fixture.manifest['probability_artifact_policy'] = dict(POLICY)

    def runtime(self):
        return artifacts.bind_runtime(self.fixture.runtime(), self.fixture.manifest)

    def rollout(self, runtime=None):
        with patch('subnet.model.create_session', return_value=tiny.Session()):
            return (runtime or self.runtime()).rollout(2, 0)

    def verify(self, rollout, arrays, runtime=None):
        with patch('subnet.model.create_session', return_value=tiny.Session()):
            return (runtime or self.runtime()).verify(rollout, arrays)

    def test_real_toploc_compact_generation_and_serialized_reverification(self):
        rollout, arrays = self.rollout()
        self.assertEqual(arrays[0].shape, (len(rollout['turns'][0]['output']), 1))
        batch = dict(rollouts=[rollout])
        data = pack([(batch, [arrays])], stable=True)
        decoded = unpack(data)[0]
        self.assertTrue(self.verify(decoded[0]['rollouts'][0], decoded[1][0]))

    def test_identical_full_model_sampler_and_toploc_outputs_under_both_formats(self):
        compact_rollout, selected = self.rollout()
        self.fixture.manifest.pop('probability_artifact_policy')
        legacy_rollout, full = self.rollout()
        self.assertEqual(compact_rollout, legacy_rollout)
        np.testing.assert_array_equal(selected[0], artifacts.selected_values(full[0], legacy_rollout['turns'][0]['output']))

    def test_forged_selected_probability_every_position_rejected(self):
        rollout, arrays = self.rollout()
        for position in range(len(arrays[0])):
            bad = [arrays[0].copy()]; bad[0][position, 0] -= .1
            with self.subTest(position=position), self.assertRaisesRegex(InvalidSample, 'probabilities'):
                self.verify(rollout, bad)

    def test_shape_dtype_missing_and_nonfinite_states_rejected(self):
        rollout, arrays = self.rollout(); a = arrays[0]
        for bad in (None, a[:, 0], a[:-1], np.repeat(a, 2, axis=1), a.astype(np.float64), np.full_like(a, np.nan)):
            with self.subTest(shape=getattr(bad, 'shape', None)), self.assertRaises(InvalidSample):
                self.verify(rollout, [bad])
        with self.assertRaises(InvalidSample):
            self.verify(rollout, [])

    def test_missing_full_reference_does_not_become_selected_only_verification(self):
        runtime = self.runtime(); rollout, arrays = self.rollout(runtime)
        acts, _ = runtime.compute(rollout['turns'][0]['prompt'], rollout['turns'][0]['output'])
        with patch.object(runtime, 'compute', return_value=(acts, None)), self.assertRaises(InvalidSample):
            self.verify(rollout, arrays, runtime)

    def test_missing_or_forged_toploc_proof_rejected(self):
        rollout, arrays = self.rollout()
        for proofs in ([], ['invalid-base64']):
            bad = copy.deepcopy(rollout); bad['turns'][0]['proofs'] = proofs
            with self.assertRaises(InvalidSample):
                self.verify(bad, arrays)

    def test_seed_change_even_with_fresh_receipt_rejected(self):
        runtime = self.runtime(); rollout, arrays = self.rollout(runtime)
        with patch('subnet.model.create_session', return_value=tiny.Session()):
            different = next(i for i in range(1, 16) if runtime.rollout(2, i)[0]['turns'][0]['output'] != rollout['turns'][0]['output'])
        rollout['seed'] = different; rollout['sampling'] = sampling.receipt(runtime.sampling_context, different)
        with self.assertRaisesRegex(InvalidSample, 'sampling replay mismatch'):
            self.verify(rollout, arrays, runtime)

    def test_forged_tokens_fresh_genuine_toploc_and_selected_probabilities_rejected(self):
        runtime = self.runtime(); rollout, arrays = self.rollout(runtime); turn = rollout['turns'][0]
        turn['output'][0] = (turn['output'][0] + 1) % 7
        turn['text'] = runtime.tokenizer.decode(turn['output'])
        acts, full = runtime.compute(turn['prompt'], turn['output'])
        arrays[0] = artifacts.encode(full, turn['output'], POLICY)
        turn['proofs'] = runtime.build_proofs(acts, decode_batching_size=16, topk=128)
        result = tiny.Session().step({'text': turn['text']})
        for key in ('reward', 'classification'): turn[key] = rollout[key] = result[key]
        with self.assertRaisesRegex(InvalidSample, 'sampling replay mismatch'):
            self.verify(rollout, arrays, runtime)

    def test_context_task_hash_checkpoint_and_sampling_receipt_remain_bound(self):
        rollout, arrays = self.rollout()
        for change in ('context', 'task', 'receipt', 'checkpoint'):
            bad = copy.deepcopy(rollout); runtime = self.runtime()
            if change == 'context': bad['turns'][0]['prompt'][0] = 1
            elif change == 'task': bad['task_hash'] = '0' * 64
            elif change == 'receipt': bad.pop('sampling')
            else:
                manifest = copy.deepcopy(self.fixture.manifest); manifest['checkpoint']['id'] = '0' * 64
                sampling.bind_runtime(runtime, manifest)
            with self.subTest(change=change), self.assertRaises(InvalidSample):
                self.verify(bad, arrays, runtime)

    def test_forced_sampler_cannot_be_disabled_to_accept_compact_claims(self):
        runtime = self.runtime(); rollout, arrays = self.rollout(runtime); runtime.sampling_context = None
        with self.assertRaisesRegex(InvalidSample, 'require authenticated forced sampling'):
            self.verify(rollout, arrays, runtime)
        with self.assertRaises(ValueError):
            artifacts.bind_runtime(runtime, {'probability_artifact_policy': POLICY})

    def test_legacy_full_vocab_default_preserved_and_cross_contract_shapes_rejected(self):
        self.fixture.manifest.pop('probability_artifact_policy'); legacy = self.runtime()
        rollout, full = self.rollout(legacy); self.assertGreater(full[0].shape[1], 1)
        self.assertTrue(self.verify(rollout, full, legacy))
        selected = [artifacts.encode(full[0], rollout['turns'][0]['output'], POLICY)]
        with self.assertRaises(InvalidSample): self.verify(rollout, selected, legacy)
        sampling.bind_runtime(legacy, dict(self.fixture.manifest, probability_artifact_policy=POLICY))
        artifacts.bind_runtime(legacy, dict(self.fixture.manifest, probability_artifact_policy=POLICY))
        with self.assertRaises(InvalidSample): self.verify(rollout, full, legacy)
        self.assertIs(artifacts.encode(full[0], rollout['turns'][0]['output'], None), full[0])

    def test_legacy_still_detects_forged_unselected_vocab_probability(self):
        self.fixture.manifest.pop('probability_artifact_policy'); runtime = self.runtime()
        rollout, arrays = self.rollout(runtime); token = rollout['turns'][0]['output'][0]
        arrays[0][0, (token + 1) % arrays[0].shape[1]] -= .1
        with self.assertRaisesRegex(InvalidSample, 'probabilities'): self.verify(rollout, arrays, runtime)

    def test_fast_prefill_compact_uses_full_distribution_for_every_token(self):
        fixture = prefill.Controls(); miner, manifest = fixture.support_runtime()
        manifest['probability_artifact_policy'] = dict(POLICY); sampling.bind_runtime(miner, manifest)
        artifacts.bind_runtime(miner, manifest)
        verifier, _ = fixture.support_runtime(); sampling.bind_runtime(verifier, manifest)
        artifacts.bind_runtime(verifier, manifest)
        rollout, arrays = self.rollout(miner)
        from subnet.fast_prefill_audit import verify_intervals
        calls = []
        def intervals(full, tokens, draws, *args):
            calls.append((tuple(full.shape), list(tokens), list(draws)))
            return verify_intervals(full, tokens, draws, *args)
        with patch('subnet.fast_prefill_audit.verify_intervals', side_effect=intervals), patch.object(verifier, 'sample_output', side_effect=AssertionError('prefill should use prescribed intervals')):
            self.assertTrue(self.verify(rollout, arrays, verifier))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], (len(rollout['turns'][0]['output']), 8))
        self.assertEqual(len(calls[0][2]), calls[0][0][0])

    def test_unknown_policy_and_null_policy_fail_closed(self):
        for value in (None, {}, {'version': 'full-or-selected-auto'}, {'version': artifacts.VERSION, 'allow_missing': True}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                artifacts.for_manifest({'probability_artifact_policy': value})
        self.assertIsNone(artifacts.for_manifest({}))

    def test_config_contract_copies_explicit_policy_and_rejects_automatic_formats(self):
        from subnet.gpu_service import contract
        row = dict(spec=dict(id='math', version='fixed-v1', num_samples=4, max_output_tokens=512), indices=[0], harness=dict(version='text-tools-v1'))
        raw = dict(POLICY)
        with patch('subnet.gpu_service.definitions', return_value=[row]):
            chosen = contract(dict(source_bundle={}, heldout=[], probability_artifact_policy=raw), 0)
            default = contract(dict(source_bundle={}, heldout=[]), 0)
            with self.assertRaises(ValueError):
                contract(dict(source_bundle={}, heldout=[], probability_artifact_policy=None), 0)
        raw['version'] = 'changed-after-contract'
        self.assertEqual(chosen['probability_artifact_policy'], POLICY)
        self.assertNotIn('probability_artifact_policy', default)

    def test_real_serialization_size_comparison_keeps_token_and_proof_metadata(self):
        rng = np.random.default_rng(81); full = rng.normal(-10, 2, (64, 4096)).astype(np.float32)
        tokens = [int(x) for x in rng.integers(0, 4096, 64)]
        batch = dict(rollouts=[dict(turns=[dict(output=tokens, proofs=['genuine-proof-framing-is-tested-separately'])])])
        legacy = pack([(batch, [[full]])], stable=True, compression_level=1)
        compact = pack([(batch, [[artifacts.encode(full, tokens, POLICY)]])], stable=True, compression_level=1)
        self.assertEqual(unpack(legacy)[0][0], unpack(compact)[0][0])
        self.assertGreater(len(legacy) / len(compact), 1000)
        self.assertEqual(unpack(compact)[0][1][0][0].nbytes, 64 * 4)
        print('CPU serialized64x4096 comparison: legacy=%d compact=%d ratio=%.1f' % (len(legacy), len(compact), len(legacy)/len(compact)))

    def test_real_epoch_signature_binds_policy_and_training_amendment_cannot_change_it(self):
        from test_real_gpu_epoch_open import MemoryBucket
        from subnet.controller import Controller
        from subnet.storage import Gateway, Identity
        from subnet.backend_jobs import signed
        from subnet.training_receipts import computation_binding
        with tempfile.TemporaryDirectory() as folder:
            bucket = MemoryBucket(); gateway = Gateway(bucket, state_path=Path(folder)/'gateway.json', direct_r2=True)
            try:
                controller = Controller(bucket, gateway, Path(folder)/'controller'); identity = Identity()
                m = controller.open('nonpayable-compact-proof-control', dict(id='a'*64, files={}), [identity.id], source_bundle={'sha256':'b'*64}, sampling_policy={'version':sampling.VERSION,'max_attempts':16}, harness={'version':'text-tools-v1','policy':'autoregressive','max_output_tokens':4}, probability_artifact_policy=POLICY)
                actual = signed(json.loads(bucket.objects['public/nonpayable-compact-proof-control/manifest.json']), controller.authority.id)
                self.assertEqual(actual['probability_artifact_policy'], POLICY)
                changed = dict(actual); changed.pop('probability_artifact_policy')
                self.assertNotEqual(computation_binding(actual), computation_binding(changed))
            finally:
                gateway.server.shutdown(); gateway.server.server_close(); gateway.thread.join()

if __name__ == '__main__': unittest.main()
