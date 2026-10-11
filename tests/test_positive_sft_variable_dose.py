"""CPU admission/lifecycle controls, not GPU or production-release qualification."""
import base64
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'tests')]
from nacl.signing import SigningKey
from botocore.exceptions import ClientError
from subnet.storage import canonical
from subnet import positive_only_training as kernel
from test_positive_only_training import pairs_for_tasks, EOS, SEED, OBJECTIVE
from ops import positive_sft_variable_dose as f

CAP = dict(target_tasks_per_step=64, max_tasks=512, max_steps=8)


def select(pairs, **kw):
    options = dict(seed=SEED, eos_token_id=EOS, required_pairs_per_task=4, capacity=CAP)
    options.update(kw)
    return f.selection(pairs, **options)


class Bucket:
    def __init__(self):
        self.objects = {}; self.puts = 0; self.fail_put = False; self.missing_client_error = False
        self.before_get = None

    def get_bounded(self, key, *, limit):
        if self.before_get:
            self.before_get()
        if key not in self.objects and self.missing_client_error:
            raise ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
        data = self.objects[key]
        if len(data) != limit or not 0 < limit <= 2_000_000:
            raise ValueError('immutable learner capture byte size')
        return data

    def put(self, key, data):
        if self.fail_put:
            raise TimeoutError('injected publication failure')
        self.objects[key] = data; self.puts += 1


class SelectionTests(unittest.TestCase):
    def test_every_population_partitions_once(self):
        for n in range(1, 513):
            tasks = [dict(task_sha256=f'{i+1:064x}', positive_indices=[i]) for i in range(n)]
            k = (n+63)//64
            groups = kernel.fixed_task_minibatches(tasks, k)
            self.assertEqual([i for g in groups for i in g], list(range(n)))
            self.assertTrue(all(1 <= len(g) <= 64 for g in groups))
            self.assertLessEqual(max(map(len, groups))-min(map(len, groups)), 1)

    def test_selection_matches_scientific_candidate_at_boundaries(self):
        for n in (1, 2, 63, 64, 65, 127, 128, 129, 511, 512):
            rows = pairs_for_tasks(n)
            original = copy.deepcopy(rows)
            proposed = select(rows)
            _, tasks, groups, expected = kernel.select_positive_tasks(rows, (n+63)//64, SEED, EOS, 4)
            self.assertEqual((proposed['tasks'], proposed['groups'], proposed['selection']),
                             (tasks, groups, expected))
            self.assertEqual(proposed['steps'], (n+63)//64)
            self.assertEqual(proposed['positive_count'], 4*n)
            self.assertEqual(rows, original)

    def test_eos_exclusions_keep_partial_task_and_remove_zero_positive_task(self):
        pairs = pairs_for_tasks(3)
        for i in (0, 1, 4, 5, 6, 7):
            pairs[i][1]['turns'][0]['output'][-1] = 27
        proposed = select(pairs)
        _, tasks, groups, details = kernel.select_positive_tasks(pairs, 1, SEED, EOS, 4)
        self.assertEqual(proposed['selection'], details)
        self.assertEqual(proposed['tasks'], tasks)
        self.assertEqual(proposed['groups'], groups)
        self.assertEqual(proposed['task_count'], 2)
        self.assertEqual(proposed['positive_count'], 6)
        self.assertEqual(len(details['excluded_tasks']), 1)

    def test_zero_population_and_zero_eos_are_explicit_not_fake_steps(self):
        for pairs in ([], pairs_for_tasks(2)):
            for _, positive, _ in pairs:
                positive['turns'][0]['output'][-1] = 27
            result = select(pairs)
            self.assertEqual((result['task_count'], result['positive_count'], result['steps']), (0, 0, 0))
            self.assertEqual(result['groups'], [])
            self.assertEqual(result['disposition'], 'no_update')

    def test_invalid_or_duplicate_or_incomplete_inputs_do_not_become_no_data(self):
        rows = pairs_for_tasks(1)
        for bad in (rows[:-1], rows+[copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError): select(bad)
        bad = copy.deepcopy(rows); bad[1] = (bad[1][0], copy.deepcopy(bad[0][1]), bad[1][2])
        with self.assertRaises(ValueError): select(bad)
        bad = copy.deepcopy(rows); bad[0][1]['turns'][0]['output'] = [True, EOS]
        with self.assertRaises(ValueError): select(bad)
        with self.assertRaises(ValueError): select(rows, eos_token_id=True)
        with self.assertRaises(ValueError): select(rows, capacity=dict(CAP, max_steps=1))
        with self.assertRaises(ValueError): select(rows, capacity=dict(CAP, max_tasks=True))
        with self.assertRaises(ValueError): select(pairs_for_tasks(513))


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        self.key = SigningKey.generate(); self.authority = self.key.verify_key.encode().hex()
        self.pointer = dict(namespace='private/trainer-state/test/job',
            descriptor_key='private/trainer-state/test/job/authority-state.json',
            publication_sha256='1'*64, descriptor_sha256='2'*64, genesis_sha256='3'*64,
            optimizer_steps=8, inference_checkpoint='4'*64, parameters_sha256='5'*64)
        # Non-canonical whitespace must survive the no-update path byte for byte.
        self.raw_pointer = json.dumps(self.pointer, indent=2).encode()+b'\n'
        (self.state/'latest-trainer-state.json').write_bytes(self.raw_pointer)
        self.manifest = dict(epoch='test-146', checkpoint=dict(id='4'*64, files=[]), K=4, L=4,
            training_objective=OBJECTIVE, training_input_policy='committed-unaudited-training-v1',
            trainer_state_binding=dict(epoch='test-146', input_checkpoint='4'*64,
                source_sha256='6'*64, parent=copy.deepcopy(self.pointer), genesis=None,
                global_step_before=8))
        self.status = dict(round=146, active=dict(epoch='test-146', phase='train'),
            checkpoint=copy.deepcopy(self.manifest['checkpoint']), checkpoint_path='/model/original',
            trainer_state=copy.deepcopy(self.pointer), persistent_state_committed=True, training_steps=8)
        self.scope = dict(version=f.RELEASE_VERSION, manifest_sha256=f.sha(self.sign(self.manifest)),
            training_source_sha256='6'*64, objective=OBJECTIVE, seed=SEED, eos_token_id=EOS,
            required_pairs_per_task=4, capacity=CAP)
        self.bucket = Bucket()
        self.controller = SimpleNamespace(state=self.state, authority=SimpleNamespace(id=self.authority),
            signed=self.sign, bucket=self.bucket)

    def sign(self, payload):
        return dict(payload=copy.deepcopy(payload), signer=self.authority,
            signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())

    def inputs(self, pairs):
        admission = dict(version=f.ADMISSION_VERSION, epoch='test-146', checkpoint='4'*64,
            manifest_sha256=f.sha(self.sign(self.manifest)), admitted_pairs_sha256=f.sha(pairs),
            pair_sha256=[f.pair_identity(p) for p in pairs], complete=True,
            pending_input_count=0, infrastructure_error_count=0,
            native_admission_sha256='7'*64, sampling_assurance='unaudited',
            proof_verification_performed=False)
        return [self.sign(self.manifest), self.sign(admission), self.sign(self.scope)]

    def close(self, pairs=None, inputs=None, status=None):
        pairs = [] if pairs is None else pairs
        return f.close_no_update(self.controller, pairs, *(inputs or self.inputs(pairs)),
                                 status or self.status)

    def test_nonempty_step_decision_is_authority_bound_and_no_dispatch_occurs(self):
        pairs = pairs_for_tasks(65)
        route = f.route_proposal(pairs, *self.inputs(pairs), self.authority)
        args = f.dispatch_arguments(route)
        self.assertEqual(args['steps'], 2)
        self.assertEqual((args['expected_global_step_before'], args['expected_global_step_after']), (8, 10))
        self.assertTrue(args['requires_actual_step_execution_release'])
        self.assertTrue(args['requires_actual_step_learning_rate_authorization'])
        self.assertFalse((self.state/'roles').exists())
        with self.assertRaisesRegex(ValueError, 'zero eligible'):
            self.close(pairs)

    def test_no_update_preserves_exact_pointer_checkpoint_and_counters(self):
        before = copy.deepcopy(self.status)
        checkpoint, metrics = self.close()
        self.assertEqual(checkpoint, before['checkpoint'])
        self.assertEqual(metrics['trainer_state'], self.pointer)
        self.assertEqual(metrics['steps'], 0)
        receipt = metrics['native_no_update']
        self.assertFalse(receipt['training_succeeded'])
        self.assertFalse(receipt['training_dispatched'])
        self.assertIsNone(receipt['training_report'])
        self.assertIsNone(receipt['losses'])
        self.assertFalse(receipt['reward_policy_changed'])
        self.assertNotIn('payable', receipt)
        self.assertEqual(receipt['training_steps_before'], receipt['training_steps_after'])
        self.assertEqual(self.status, before)
        self.assertEqual((self.state/'latest-trainer-state.json').read_bytes(), self.raw_pointer)
        completion = f.completion_fields(self.controller, 'test-146')
        self.assertEqual(completion['optimizer_updates'], 0)
        self.assertEqual(completion['status'], 'closed_positive_sft_no_update')
        with self.assertRaisesRegex(ValueError, 'no optimizer dispatch'):
            f.dispatch_arguments(f.route_proposal([], *self.inputs([]), self.authority))

    def test_read_capability_refresh_does_not_replace_checkpoint_descriptor(self):
        self.status['checkpoint']['read_capability_revision'] = 'hydrated-original-id'
        before = copy.deepcopy(self.status['checkpoint'])
        cp, metrics = self.close()
        self.assertEqual(cp, before)
        self.assertEqual(metrics['native_no_update']['checkpoint_descriptor_sha256'], f.sha(before))
        self.assertEqual((self.state/'latest-trainer-state.json').read_bytes(), self.raw_pointer)

    def test_all_eos_excluded_closes_with_distinct_truthful_reason(self):
        pairs = pairs_for_tasks(2)
        for _, positive, _ in pairs:
            positive['turns'][0]['output'][-1] = 27
        _, metrics = self.close(pairs)
        self.assertEqual(metrics['native_no_update']['reason'], 'no-terminal-eos-positives')
        route = json.loads((self.state/'test-146-positive-sft-fallback/route.ROOT-SIGNED.json').read_bytes())
        self.assertEqual(len(route['payload']['decision']['selection']['excluded']), 8)

    def test_publication_failure_then_retry_reuses_original_receipt(self):
        self.bucket.fail_put = True
        with self.assertRaises(TimeoutError): self.close()
        path = self.state/'test-146-positive-sft-fallback/no-update.ROOT-SIGNED.json'
        original = path.read_bytes()
        self.bucket.fail_put = False
        result = self.close()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.close(), result)
        self.assertEqual(self.bucket.puts, 1)
        self.assertEqual((self.state/'latest-trainer-state.json').read_bytes(), self.raw_pointer)

    def test_r2_notfound_is_supported_but_transport_error_is_not_emptiness(self):
        self.bucket.missing_client_error = True
        self.close()
        self.assertEqual(self.bucket.puts, 1)
        with patch.object(self.bucket, 'get_bounded', side_effect=TimeoutError('transport')):
            with self.assertRaises(TimeoutError): self.close()
        self.assertEqual(self.bucket.puts, 1)

    def test_incomplete_input_and_infrastructure_failure_never_close_no_update(self):
        for field, value in (('complete', False), ('pending_input_count', 1),
                             ('infrastructure_error_count', 1), ('pending_input_count', False)):
            inputs = self.inputs([]); inputs[1]['payload'][field] = value
            inputs[1] = self.sign(inputs[1]['payload'])
            with self.assertRaisesRegex(ValueError, 'incomplete or failed'): self.close(inputs=inputs)
        self.assertFalse((self.state/'test-146-positive-sft-fallback').exists())

    def test_original_content_source_epoch_eos_scope_cannot_be_relabelled(self):
        pairs = pairs_for_tasks(1); inputs = self.inputs(pairs)
        pairs[0][0]['spec']['changed'] = True
        with self.assertRaisesRegex(ValueError, 'complete frozen pair'): self.close(pairs, inputs)
        for component, field, value in ((0, 'epoch', 'other'), (2, 'eos_token_id', EOS+1),
                                         (1, 'checkpoint', '8'*64)):
            inputs = self.inputs([]); inputs[component]['payload'][field] = value
            with self.assertRaises(Exception): self.close(inputs=inputs)
        inputs = self.inputs([])
        inputs[2]['payload']['training_source_sha256'] = '8'*64
        inputs[2] = self.sign(inputs[2]['payload'])
        with self.assertRaisesRegex(ValueError, 'training source binding'): self.close(inputs=inputs)

    def test_issued_job_and_changed_parent_never_hidden_by_no_update(self):
        roles = self.state/'roles'; roles.mkdir()
        for filename in ('test-146-train.json', 'test-146-train-recovery-job.json'):
            marker = roles/filename; marker.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'issued training'): self.close()
            marker.unlink()
        pointer = copy.deepcopy(self.pointer); pointer['optimizer_steps'] += 1
        (self.state/'latest-trainer-state.json').write_bytes(canonical(pointer))
        with self.assertRaisesRegex(ValueError, 'current optimizer pointer'): self.close()
        self.assertFalse((self.state/'test-146-positive-sft-fallback').exists())

    def test_durable_collision_is_not_overwritten(self):
        key = 'public/test-146/positive-sft-no-update.json'
        self.bucket.objects[key] = b'wrong previous bytes'
        with self.assertRaises(ValueError): self.close()
        self.assertEqual(self.bucket.puts, 0)
        self.assertEqual(self.bucket.objects[key], b'wrong previous bytes')

    def test_parent_change_during_publication_is_detected(self):
        called = [False]
        def mutate():
            if called[0]: return
            called[0] = True
            pointer = copy.deepcopy(self.pointer); pointer['descriptor_sha256'] = '9'*64
            (self.state/'latest-trainer-state.json').write_bytes(canonical(pointer))
        self.bucket.before_get = mutate
        with self.assertRaisesRegex(ValueError, 'current optimizer pointer'): self.close()
        self.assertFalse((self.state/'test-146-signed-learner-completion.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
