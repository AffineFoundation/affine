import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest

from nacl.signing import SigningKey
from nacl.exceptions import BadSignatureError
from dashboard.training_diagnostic_projection import canonical, digest, project, rows


class TrainingProjectionTests(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()
        profile = dict(max_output_tokens=2048, max_context_tokens=8192,
            temperature=.7, top_p=1., source_files_sha256='e' * 64,
            runtime_versions={'torch': 'test-pinned'})
        self.value = dict(version='fresh-training-evaluation-v1', series='training-fresh',
            label='Training set — fresh generations', generation_origin='fresh-policy-generation',
            run_id='train128-v1', training_run_id='fp32-conservative-v1', env_id='affine_math',
            epoch='nonpayable-live-reward-math-v1--100-158', phase='latest',
            cohort_sha256='a' * 64, profile_sha256=digest(profile), seeds_sha256='b' * 64,
            checkpoint='c' * 64, optimizer_step=68, generated_at=1000., profile=profile,
            comparability_passed=True,
            requested=128, completed=128, solved=81, wrong=40, unresolved=7, errors=0,
            cap_hits=7, terminal_eos=121, mean_output_tokens=301.5, median_output_tokens=256.,
            exposure=dict(scope='current-training-run-through-checkpoint', trained_tasks=90,
                untrained_tasks=38, unknown_tasks=0, task_exposures=731),
            exposure_groups=dict(trained=dict(count=90,solved=60,wrong=25,unresolved=5,errors=0),
                untrained=dict(count=38,solved=21,wrong=15,unresolved=2,errors=0),
                unknown=dict(count=0,solved=0,wrong=0,unresolved=0,errors=0)),
            exposure_subsets_descriptive=True, fixed_initial_trained=None,
            durable_result=dict(key='private/diagnostic/result.json', sha256='d' * 64, full_get_verified=True))

    def signed(self, value=None):
        value = copy.deepcopy(self.value if value is None else value)
        return dict(payload=value, signer=self.authority,
            signature=base64.b64encode(self.key.sign(canonical(value)).signature).decode())

    def test_exact_complete_fresh_score_and_no_private_projection(self):
        self.value['private_debug'] = 'SECRET PRIVATE PATH OR TOKEN'
        row = project(self.signed(), self.authority)
        self.assertEqual(row['mean_reward'], 81 / 128)
        self.assertEqual(row['evaluation_split'], 'train')
        self.assertEqual(row['completed_count'], 128)
        self.assertEqual(row['actual_trained_task_count'], 90)
        public = json.dumps(row)
        self.assertNotIn('SECRET', public)
        self.assertNotIn('private/diagnostic', public)
        self.assertFalse(row['proof_verification_performed'])

    def genesis(self):
        value = copy.deepcopy(self.value)
        value.update(phase='baseline', optimizer_step=0,
            reference_kind='actual-run-genesis-comparison',
            actual_run_genesis_checkpoint=value['checkpoint'],
            genesis_admission_sha256='f' * 64, reference_is_original_upstream=False)
        value['exposure'].update(trained_tasks=0, untrained_tasks=128,
            unknown_tasks=0, task_exposures=0)
        value['exposure_groups'] = dict(
            trained=dict(count=0, solved=0, wrong=0, unresolved=0, errors=0),
            untrained=dict(count=128, solved=81, wrong=40, unresolved=7, errors=0),
            unknown=dict(count=0, solved=0, wrong=0, unresolved=0, errors=0))
        return value

    def test_reference_roles_preserve_current_run_exposure_meaning(self):
        value = self.genesis()
        row = project(self.signed(value), self.authority)
        self.assertEqual(row['training_steps'], 0)
        self.assertEqual(row['actual_trained_task_count'], 0)
        self.assertEqual(row['actual_training_exposures'], 0)
        self.assertIn('of this run; earlier training retained', row['reference_label'])
        self.assertNotIn('optimizer_step', row)
        upstream = copy.deepcopy(value)
        upstream.update(reference_kind='original-upstream-comparison',
            checkpoint='a' * 64, reference_is_original_upstream=True)
        row = project(self.signed(upstream), self.authority)
        self.assertIn('not actual run genesis', row['reference_label'])
        self.assertEqual(row['actual_run_genesis_checkpoint'], value['checkpoint'])

    def test_genesis_requires_correct_identity_and_no_new_exposures(self):
        for change in ({'optimizer_step': 68}, {'checkpoint': 'a' * 64},
                       {'phase': 'latest'}, {'reference_is_original_upstream': True}):
            value = self.genesis(); value.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                project(self.signed(value), self.authority)
        for mode in ('trained', 'unknown', 'exposure_count'):
            value = self.genesis()
            if mode == 'exposure_count':
                value['exposure']['task_exposures'] = 1
            else:
                value['exposure'].update(trained_tasks=128 if mode == 'trained' else 0,
                    untrained_tasks=0, unknown_tasks=128 if mode == 'unknown' else 0,
                    task_exposures=128 if mode == 'trained' else 0)
                value['exposure_groups'][mode] = value['exposure_groups']['untrained']
                value['exposure_groups']['untrained'] = dict(
                    count=0, solved=0, wrong=0, unresolved=0, errors=0)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'zero new training exposure'):
                project(self.signed(value), self.authority)

    def test_infra_errors_remain_unavailable_not_zero_or_reduced_denominator(self):
        self.value.update(completed=127, errors=1, wrong=39, terminal_eos=120)
        self.value['exposure_groups']['untrained'].update(wrong=14,errors=1)
        row = project(self.signed(), self.authority)
        self.assertEqual((row['status'], row['count'], row['error_count']), ('partial', 128, 1))
        self.assertIsNone(row['mean_reward'])

    def test_missing_exposure_is_unknown_not_zero(self):
        self.value['exposure'].update(trained_tasks=None, untrained_tasks=None,
            task_exposures=None, unknown_tasks=128)
        self.value['exposure_groups'] = dict(trained=dict(count=0,solved=0,wrong=0,unresolved=0,errors=0),
            untrained=dict(count=0,solved=0,wrong=0,unresolved=0,errors=0),
            unknown=dict(count=128,solved=81,wrong=40,unresolved=7,errors=0))
        self.assertIsNone(project(self.signed(), self.authority)['actual_trained_task_count'])
        self.value['exposure']['unknown_tasks'] = 0
        with self.assertRaises(ValueError): project(self.signed(), self.authority)

    def test_actual_prompt_mismatch_is_unavailable_even_without_infrastructure_errors(self):
        self.value['comparability_passed'] = False
        row = project(self.signed(), self.authority)
        self.assertEqual(row['status'], 'partial')
        self.assertIsNone(row['mean_reward'])
        del self.value['comparability_passed']
        with self.assertRaises(ValueError): project(self.signed(), self.authority)

    def test_invalid_counts_runtime_durability_and_origin_refused(self):
        mutations = [lambda v: v.update(completed=127), lambda v: v.update(solved=129),
            lambda v: v.update(errors=True), lambda v: v.update(cap_hits=129),
            lambda v: v.update(generation_origin='submitted-miner-labels'),
            lambda v: v['profile'].update(max_output_tokens=1024),
            lambda v: v['durable_result'].update(full_get_verified=False),
            lambda v: v['durable_result'].update(key='https://private.example/?secret=x'),
            lambda v: v.update(generated_at=float('nan'))]
        for change in mutations:
            value = copy.deepcopy(self.value); change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                project(self.signed(value), self.authority)

    def test_exposure_groups_cannot_misstate_seen_tasks_or_discard_errors(self):
        row = project(self.signed(), self.authority)
        self.assertEqual(row['exposure_groups']['trained']['solved'], 60)
        self.assertTrue(row['exposure_subsets_descriptive'])
        self.value['exposure_groups']['trained']['solved'] += 1
        with self.assertRaises(ValueError): project(self.signed(), self.authority)

    def test_tampered_signature_and_foreign_authority_rejected(self):
        document = self.signed(); document['payload']['solved'] = 100
        with self.assertRaises(BadSignatureError): project(document, self.authority)
        with self.assertRaises(ValueError): project(self.signed(), 'f' * 64)

    def test_record_identity_separates_cohort_seed_checkpoint_and_run(self):
        original = project(self.signed(), self.authority)['run_id']
        for field in ('checkpoint', 'cohort_sha256', 'seeds_sha256', 'training_run_id'):
            value = copy.deepcopy(self.value); value[field] = 'f' * 64
            self.assertNotEqual(project(self.signed(value), self.authority)['run_id'], original)

    def test_directory_ignores_bad_or_symlink_records_and_conflicting_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'good.signed.json').write_bytes(canonical(self.signed()))
            (root/'bad.signed.json').write_text('{}')
            (root/'link.signed.json').symlink_to(root/'good.signed.json')
            self.assertEqual(len(rows(root, self.authority)), 1)
            changed = copy.deepcopy(self.value); changed.update(solved=80, wrong=41)
            changed['exposure_groups']['trained'].update(solved=59,wrong=26)
            (root/'conflict.signed.json').write_bytes(canonical(self.signed(changed)))
            self.assertEqual(rows(root, self.authority), [])


if __name__ == '__main__': unittest.main()
