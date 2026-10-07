import copy
from types import SimpleNamespace
import unittest

from ops.paired_quota_batch_adapter import CurrentBatchAdapter
from ops.paired_quota_common_stream import collect, select_nested, selected_artifact_refs
from subnet import forced_sampling, harness
from subnet.audit_policy import InvalidSample
from subnet.fast_prefill_audit import NumericalAmbiguity


class CommonStreamControls(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(epoch='common-research-40', checkpoint={'id': 'a' * 64}, K=1, L=1,
            harness_source_hash=harness.source_hash(), sampling_source_hash=forced_sampling.source_hash(),
            sampling_contract=dict(version=forced_sampling.VERSION, randomness='b' * 64,
                max_attempts=16, verification='exact-token-replay', generation='uncached-eager-inverse-cdf'),
            environments=[dict(env_id='math', indices=[10], spec=dict(id='math', version='v1',
                num_samples=32, max_turns=4, max_output_tokens=64, config={'seed': 7}),
                harness=dict(harness.DEFAULT))])
        self.adapter = CurrentBatchAdapter.from_manifest(self.manifest, 'math', 10,
                        task_sha256='c' * 64, decode=lambda output: ''.join(chr(i) for i in output))
        self.generated = []; self.archived = []; self.saved = []
        self.runtime = SimpleNamespace(sampling_context=self.adapter.sampling_context,
            harness=dict(harness.DEFAULT), spec=SimpleNamespace(id='math', version='v1'),
            rollout=self.rollout, verify=lambda r, a: True)

    def rollout(self, index, attempt):
        self.generated.append(attempt)
        label = 'positive' if attempt % 2 == 0 else 'negative'
        text = chr(65 + attempt)
        r = dict(schema=2, env_id='math', environment_version='v1', index=index,
            sample_index=index, seed=attempt, env_seed=7, task_hash='c' * 64,
            classification=label, reward=int(label == 'positive'),
            sampling=forced_sampling.receipt(self.adapter.sampling_context, attempt),
            turns=[dict(prompt=[10, 20], output=[ord(text)], text=text,
                observations=[dict(role='user', content='feedback')], done=True,
                reward=1, classification=label, proofs=['ORIGINAL-PROOF'])])
        return r, [['ORIGINAL-ARRAY', attempt]]

    def archive(self, rollout, arrays):
        self.archived.append((copy.deepcopy(rollout), copy.deepcopy(arrays)))
        return dict(sha256='f' * 64, size=10, key=f"attempt-{rollout['seed']}")

    def persist(self, row):
        self.saved.append(row)
        return True

    def collect(self, **kwargs):
        args = dict(admit_runtime=lambda r, t: True, persist_artifact=self.archive,
                    persist_attempt=self.persist, enabled=True)
        args.update(kwargs)
        return collect(self.runtime, self.adapter, **args)

    def test_all_attempts_preserved_even_after_both_quotas_complete(self):
        rows = self.collect()
        self.assertEqual(self.generated, list(range(16)))
        self.assertEqual(len(self.archived), 16); self.assertEqual(len(self.saved), 16)
        self.assertEqual(self.archived[0][0]['turns'][0]['proofs'], ['ORIGINAL-PROOF'])
        self.assertEqual(self.archived[0][1], [['ORIGINAL-ARRAY', 0]])
        selected = select_nested(self.adapter, 'miner', rows)
        self.assertEqual(selected['supply']['first_K1L1_completion'], 2)
        self.assertEqual(selected['supply']['first_K2L2_completion'], 4)
        members = lambda arm: {r['content_id'] for p in arm['pairs'] for r in (p['positive'], p['negative'])}
        self.assertTrue(members(selected['arms']['K1L1']) < members(selected['arms']['K2L2']))
        self.assertEqual(selected['arms']['K2L2']['pair_weight_within_task'], .5)
        self.assertEqual(self.manifest['K'], 1)

    def test_unfulfilled_quota_keeps_task_and_excludes_both_matched_arms(self):
        original = self.rollout
        def only_positive(index, attempt):
            rollout, arrays = original(index, attempt)
            rollout['classification'] = 'positive'
            return rollout, arrays
        self.runtime.rollout = only_positive
        result = select_nested(self.adapter, 'miner', self.collect())
        self.assertFalse(result['supply']['matched_included'])
        self.assertEqual(result['supply']['attempted'], 16)
        self.assertEqual(result['arms'], {})
        self.assertEqual(len(self.saved), 16)

    def test_k1_available_but_not_k2_does_not_change_matched_task_population(self):
        original = self.rollout
        def one_failure(index, attempt):
            r, a = original(index, attempt)
            r['classification'] = 'negative' if attempt == 1 else 'positive'
            return r, a
        self.runtime.rollout = one_failure
        result = select_nested(self.adapter, 'miner', self.collect())
        self.assertTrue(result['supply']['K1L1_possible'])
        self.assertFalse(result['supply']['matched_included'])
        self.assertEqual(result['arms'], {})

    def test_unknown_invalid_and_infra_preserved_without_usable_negatives(self):
        original = self.rollout
        def generate(index, attempt):
            if attempt == 0: raise RuntimeError('native unavailable')
            return original(index, attempt)
        def verify(r, a):
            if r['seed'] == 1: raise NumericalAmbiguity('boundary')
            if r['seed'] == 2: raise InvalidSample('token mismatch')
            if r['seed'] == 3: raise RuntimeError('transport')
            return True
        self.runtime.rollout = generate; self.runtime.verify = verify
        rows = self.collect()
        self.assertEqual([r['status'] for r in rows[:4]],
                         ['infrastructure_error', 'numerical_unknown', 'confirmed_invalid', 'infrastructure_error'])
        self.assertIsNone(rows[0]['artifact'])
        self.assertTrue(all(r['normalized'] is None for r in rows[:4]))
        self.assertTrue(all(r['artifact'] is not None for r in rows[1:]))
        self.assertEqual(len(self.saved), 16)
        selected = select_nested(self.adapter, 'miner', rows)
        self.assertEqual(selected['supply']['unavailable']['infrastructure_error'], 2)
        self.assertEqual(selected['supply']['first_K2L2_completion'], 8)

    def test_repeated_contents_across_draws_do_not_fill_extra_quota(self):
        original = self.rollout
        def duplicates(index, attempt):
            r, a = original(index, attempt)
            r['turns'][0].update(output=[65 if attempt % 2 == 0 else 66],
                                 text='A' if attempt % 2 == 0 else 'B')
            return r, a
        self.runtime.rollout = duplicates
        result = select_nested(self.adapter, 'miner', self.collect())
        self.assertEqual(result['supply']['duplicate_verified_content'], 14)
        self.assertTrue(result['supply']['K1L1_possible'])
        self.assertFalse(result['supply']['matched_included'])

    def test_missing_reordered_and_recontextualized_stream_refused(self):
        rows = self.collect()
        mutations = [rows[:-1], rows[::-1], copy.deepcopy(rows), copy.deepcopy(rows)]
        mutations[2][0]['sampling_context_sha256'] = '0' * 64
        mutations[3][1]['attempt'] = True
        for changed in mutations:
            with self.assertRaises(ValueError):
                select_nested(self.adapter, 'miner', changed)

    def test_no_seed_retry_after_artifact_or_metadata_sink_failure(self):
        with self.assertRaises(ValueError):
            self.collect(persist_attempt=lambda r: False)
        self.assertEqual(self.generated, [0])
        self.generated.clear()
        with self.assertRaises(ValueError):
            self.collect(persist_artifact=lambda r, a: {})
        self.assertEqual(self.generated, [0])

    def test_default_off_or_unadmitted_runtime_never_generates(self):
        for args in (dict(enabled=False), dict(admit_runtime=lambda r, t: False), dict(budget=17)):
            with self.assertRaises(ValueError):
                self.collect(**args)
        self.assertEqual(self.generated, [])
        self.runtime.sampling_context = copy.deepcopy(self.runtime.sampling_context)
        self.runtime.sampling_context['contract']['randomness'] = '0' * 64
        with self.assertRaises(ValueError):
            self.collect()
        self.assertEqual(self.generated, [])

    def test_selected_members_keep_original_proof_artifact_bindings(self):
        rows = self.collect(); result = select_nested(self.adapter, 'miner', rows)
        refs = selected_artifact_refs(self.adapter, result['arms']['K2L2'], rows)
        self.assertEqual(len(refs), 2)
        attempts = {v['attempt'] for pair in refs for v in pair.values()}
        self.assertEqual(attempts, {0, 1, 2, 3})
        self.assertEqual({v['artifact']['key'] for pair in refs for v in pair.values()},
                         {'attempt-0', 'attempt-1', 'attempt-2', 'attempt-3'})
        with self.assertRaisesRegex(ValueError, 'missing original'):
            selected_artifact_refs(self.adapter, result['arms']['K2L2'], rows[1:])

    def test_false_verdict_not_truthy_acceptance(self):
        self.runtime.verify = lambda r, a: {'success': True}
        result = select_nested(self.adapter, 'miner', self.collect())
        self.assertEqual(result['arms'], {})
        self.assertEqual(result['supply']['unavailable']['infrastructure_error'], 16)


if __name__ == '__main__':
    unittest.main()
