import base64
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from nacl.signing import SigningKey
from dashboard import training_evidence_projection as p


class TrainingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name)
        (self.state/'roles').mkdir()
        self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()
        self.epoch = 'nonpayable-live-reward-math-v1--1700000000-121'
        self.input_cp, self.output_cp = '11'*32, '22'*32
        self.manifest = {'epoch': self.epoch, 'checkpoint': {'id': self.input_cp},
                         'deadline': 1700000100, 'source_bundle': {'sha256': '33'*32}}
        self.completion = {'epoch': self.epoch, 'round': 121, 'checkpoint': self.input_cp,
                           'next_checkpoint': self.output_cp, 'completed_at': 1700000300}
        rollout = {'index': 7, 'env_id': 'affine_math', 'task_hash': '44'*32,
            'classification': 'positive', 'reward': 1., 'sampling': {'attempt': 0, 'version': 'sampler-v1'},
            'turns': [{'prompt': [1,2], 'output': [3,4], 'text': 'A solution.', 'done': True,
                      'classification': 'positive', 'reward': 1.,
                      'proofs': [{'url': 'https://secret/upload?token=PRIVATE'}],
                      'full_vocab_logprobs': [[-1.]*10], 'path': '/secret/cache'}]}
        negative = copy.deepcopy(rollout)
        negative['classification'] = 'negative'
        negative['reward'] = 0.
        negative['turns'][0].update(output=[5,6], text='Wrong answer.', classification='negative', reward=0.)
        self.batch = {'epoch': self.epoch, 'checkpoint': self.input_cp, 'env_id': 'affine_math',
                      'index': 7, 'rollouts': [rollout, negative]}
        self.admission = {'version': 'committed-unaudited-training-v1', 'epoch': self.epoch,
            'checkpoint': self.input_cp, 'assurance': 'unaudited', 'batch_sha256': p.digest(self.batch),
            'document_sha256': '55'*32, 'document_size': 100, 'miner_identity': '66'*32, 'slot': 0}
        ad = self.signed(self.admission)
        self.job_id = self.epoch+'-train-recovery-abcdef12'
        self.job = {'job_id': self.job_id, 'role': 'train', 'manifest': self.signed(self.manifest),
            'created_at': 1700000110, 'source_files': {'source.py': '77'*32}, 'runtime_versions': {'torch': '2.14'},
            'submissions': [{'learner_admission': ad, 'sha256': '55'*32, 'size': 100, 'url': 'PRIVATE'}]}
        self.task = p.digest({'env_id': 'affine_math', 'index': 7, 'task_hash': '44'*32})
        self.updates = [{'loss': .8, 'preference_loss': .69, 'positive_nll': .11, 'positive_nll_weight': 1.,
            'gradient_norm_before_clip': 2., 'gradient_accumulation_dtype': 'torch.float32',
            'hyperparameters': {'lr': 5e-7, 'max_grad_norm': 1., 'secret': 'PRIVATE'},
            'precision': {'bf16_changed_elements': 1, 'master_changed_elements': 9,
                          'parameters': [{'name': 'model.layer.weight', 'elements': 10,
                                          'master_delta_l2': .001, 'private_path': '/secret'}]},
            'pairs': [{'task_sha256': self.task, 'pair_sha256': p.digest(dict(env_id='affine_math', index=7, positive=rollout, negative=negative)), 'gradient_weight': 1.}]}]
        self.diagnostics = {'complete': True, 'global_optimizer_step_before': 30,
            'global_optimizer_step_after': 31, 'phase_seconds': {'gradient_seconds': 10., '/secret': 123}}
        self.report = {'success': True, 'role': 'train', 'job_id': self.job_id,
            'job_sha256': p.digest(self.job), 'epoch': self.epoch, 'checkpoint': self.input_cp,
            'new_checkpoint': {'id': self.output_cp, 'private_path': '/secret'},
            'source_files': self.job['source_files'], 'runtime_versions': self.job['runtime_versions'],
            'completed_at': 1700000200,
            'training': {'updates': self.updates, 'persistent_diagnostics': self.diagnostics},
            'training_admissions': [dict(self.admission, learner_admission_sha256=p.digest(ad), claimed_batch=self.batch)]}
        self.metrics = {'source_epoch': self.epoch, 'input_checkpoint': self.input_cp, 'checkpoint': self.output_cp,
            'state_authority_committed': True, 'remote_job_id': self.job_id, 'original_job_sha256': p.digest(self.job),
            'updates': self.updates, 'persistent_diagnostics': self.diagnostics, 'steps': 1,
            'weights_changed': True, 'input_assurance': 'unaudited', 'checkpoint_path': '/secret',
            'new_checkpoint': {'read_urls': {'x': 'PRIVATE'}}}
        self.record = {'manifest_envelope': self.signed(self.manifest),
            'completion_envelope': self.signed(self.completion), 'training_envelope': self.signed(self.metrics)}
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def signed(self, value):
        return {'payload': copy.deepcopy(value), 'signer': self.authority,
                'signature': base64.b64encode(self.key.sign(p.canonical(value)).signature).decode()}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(p.canonical(value))

    def save(self):
        self.write(self.state/'roles'/(self.job_id+'-job.json'), self.signed(self.job))
        self.write(self.state/'roles'/(self.job_id+'-report.json'), self.report)

    def project(self):
        return p.project_epoch(self.state, self.epoch, finalized_record=self.record, authority=self.authority)

    def test_recovery_job_used_instead_of_stale_role_pointer(self):
        self.write(self.state/'roles'/(self.epoch+'-train.json'), {'job_id': self.epoch+'-train-failed-original'})
        output = self.project()
        self.assertEqual(output['training']['provenance']['job_id'], self.job_id)
        self.assertEqual(output['training_inputs']['payload']['rollout_count'], 2)

    def test_allowlist_preserves_tokens_and_metrics_without_secrets(self):
        output = self.project()
        encoded = json.dumps(output)
        for secret in ['PRIVATE', '/secret', 'full_vocab', 'proofs', 'read_urls']:
            if secret == 'full_vocab':
                self.assertNotIn('full_vocab_logprobs', encoded)
            else:
                self.assertNotIn(secret, encoded)
        batch = output['training_inputs']['payload']['batches'][0]
        self.assertEqual(batch['rollouts'][0]['turns'][0]['output'], [3,4])
        update = output['training']['payload']['updates'][0]
        self.assertEqual(update['derived_clip_scale_upper_bound'], .5)
        self.assertEqual(update['gradient_accumulation_dtype'], 'torch.float32')
        self.assertEqual(update['precision']['parameters'][0]['elements'], 10)

    def test_no_signed_training_is_explicitly_unavailable(self):
        del self.record['training_envelope']
        output = self.project()
        self.assertEqual(output['training']['status'], 'unavailable')
        self.assertNotIn('payload', output['training_inputs'])

    def test_loader_only_requests_signed_training_object(self):
        envelope = self.record.pop('training_envelope')
        seen = []
        def load(key):
            seen.append(key)
            return p.canonical(envelope)
        output = p.project_epoch(self.state, self.epoch, finalized_record=self.record,
                                 authority=self.authority, input_loader=load)
        self.assertEqual(seen, ['public/'+self.epoch+'/training.json'])
        self.assertEqual(output['training']['status'], 'available')

    def test_unfinalized_or_future_completion_rejects(self):
        self.completion['completed_at'] = time.time()+100
        self.record['completion_envelope'] = self.signed(self.completion)
        with self.assertRaises(ValueError): self.project()

    def test_pre14_epoch_rejects(self):
        with self.assertRaises(ValueError):
            p.project_epoch(self.state, self.epoch.replace('-121', '-13'),
                            finalized_record=self.record, authority=self.authority)

    def test_completion_output_cannot_be_swapped(self):
        self.completion['next_checkpoint'] = '99'*32
        self.record['completion_envelope'] = self.signed(self.completion)
        with self.assertRaises(ValueError): self.project()

    def test_unsigned_or_wrong_authority_training_rejects(self):
        self.record['training_envelope']['signer'] = 'ab'*32
        with self.assertRaises(ValueError): self.project()

    def test_report_scalar_mutation_rejects(self):
        self.report['training'] = copy.deepcopy(self.report['training'])
        self.report['training']['updates'][0]['loss'] = 3.
        self.save()
        with self.assertRaises(ValueError): self.project()

    def test_manufactured_rollout_mutation_rejects(self):
        self.report['training_admissions'] = copy.deepcopy(self.report['training_admissions'])
        self.report['training_admissions'][0]['claimed_batch']['rollouts'][0]['turns'][0]['output'][0] = 99
        self.save()
        with self.assertRaises(ValueError): self.project()

    def test_repeated_report_admission_rejects(self):
        self.report['training_admissions'] *= 2
        self.save()
        with self.assertRaises(ValueError): self.project()

    def test_actual_gradient_task_missing_from_inputs_rejects(self):
        self.updates[0]['pairs'][0]['task_sha256'] = 'ab'*32
        self.record['training_envelope'] = self.signed(self.metrics)
        self.save()
        with self.assertRaises(ValueError): self.project()

    def test_pair_content_hash_cannot_be_invented(self):
        self.updates[0]['pairs'][0]['pair_sha256'] = 'ab'*32
        self.record['training_envelope'] = self.signed(self.metrics)
        self.save()
        with self.assertRaises(ValueError): self.project()

    def test_repeated_update_exposures_are_not_unique_pairs(self):
        self.updates.append(copy.deepcopy(self.updates[0]))
        self.record['training_envelope'] = self.signed(self.metrics)
        self.save()
        inputs = self.project()['training_inputs']['payload']
        self.assertEqual(inputs['actual_gradient_pair_exposure_count'], 2)
        self.assertEqual(inputs['actual_unique_gradient_pair_count'], 1)
        self.assertEqual(inputs['rollout_count'], 2)

    def test_diagnostic_arrays_and_weighted_components_retained(self):
        self.diagnostics['training_pair_margin_before'] = [.1, .2]
        self.diagnostics['phase_seconds']['CPU_optimizer_by_step'] = [2.5]
        self.diagnostics['positive_nll_components'] = {'weighted_before': {'loss': .8, 'positive_nll': .1},
            'before': [{'pair_index': 0, 'positive_mean_logprob': -.1, 'secret': 'PRIVATE'}]}
        self.record['training_envelope'] = self.signed(self.metrics)
        self.save()
        data = self.project()['log_metrics']['payload']['structured_training_diagnostics']
        self.assertEqual(data['training_pair_margin_before'], [.1, .2])
        self.assertEqual(data['phase_seconds']['CPU_optimizer_by_step'], [2.5])
        self.assertEqual(data['positive_nll_components']['weighted_before']['loss'], .8)
        self.assertNotIn('PRIVATE', json.dumps(data))

    def test_missing_report_preserves_signed_scalar_metrics(self):
        (self.state/'roles'/(self.job_id+'-report.json')).unlink()
        output = self.project()
        self.assertEqual(output['training']['status'], 'available')
        self.assertEqual(output['log_metrics']['status'], 'available')
        self.assertEqual(output['training_inputs']['reason'], 'authoritative_job_or_report_not_retained_locally')

    def test_original_recovery_alias_is_authenticated_by_job_not_filename(self):
        self.job_id = 'E121-local-state-recovery-deadbeef'
        self.job['job_id'] = self.report['job_id'] = self.metrics['remote_job_id'] = self.job_id
        self.report['job_sha256'] = self.metrics['original_job_sha256'] = p.digest(self.job)
        self.record['training_envelope'] = self.signed(self.metrics)
        self.save()
        self.assertEqual(self.project()['training']['provenance']['job_id'], self.job_id)

    def test_path_traversal_job_rejects(self):
        self.metrics['remote_job_id'] = '../private'
        self.record['training_envelope'] = self.signed(self.metrics)
        with self.assertRaises(ValueError): self.project()

    def test_unused_rollout_in_trained_task_is_not_published_as_trained(self):
        extra = copy.deepcopy(self.batch['rollouts'][0])
        extra['turns'][0]['output'] = [7,8]
        self.batch['rollouts'].append(extra)
        self.admission['batch_sha256'] = p.digest(self.batch)
        envelope = self.signed(self.admission)
        self.job['submissions'][0]['learner_admission'] = envelope
        self.report['training_admissions'] = [dict(self.admission,
            learner_admission_sha256=p.digest(envelope), claimed_batch=self.batch)]
        self.report['job_sha256'] = self.metrics['original_job_sha256'] = p.digest(self.job)
        self.record['training_envelope'] = self.signed(self.metrics)
        self.save()
        inputs = self.project()['training_inputs']['payload']
        self.assertEqual(inputs['rollout_count'], 2)
        batch = inputs['batches'][0]
        self.assertEqual(batch['admitted_rollout_count'], 3)
        self.assertEqual([r['admitted_rollout_index'] for r in batch['rollouts']], [0,1])
        self.assertEqual(len(batch['actual_gradient_pairs']), 1)

    def test_empty_closure_observation_does_not_become_authenticated_training(self):
        self.record.pop('training_envelope')
        self.completion['next_checkpoint'] = self.input_cp
        self.record['completion_envelope'] = self.signed(self.completion)
        self.write(self.state/(self.epoch+'-empty-closed.json'),
            {'epoch': self.epoch, 'checkpoint': self.input_cp, 'status': 'closed_no_eligible_batches'})
        result = self.project()['training']
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['reason'], 'empty_closure_without_signed_training_record')
        self.assertFalse(result['provenance']['empty_closure_authenticated'])
        self.assertNotIn('payload', result)

    def test_same_pair_in_two_admissions_rejects_ambiguous_attribution(self):
        second = dict(self.admission, slot=1, document_sha256='aa'*32)
        envelope = self.signed(second)
        self.job['submissions'].append({'learner_admission': envelope,
            'sha256': second['document_sha256'], 'size': second['document_size']})
        self.report['training_admissions'].append(dict(second,
            learner_admission_sha256=p.digest(envelope), claimed_batch=self.batch))
        self.report['job_sha256'] = self.metrics['original_job_sha256'] = p.digest(self.job)
        self.record['training_envelope'] = self.signed(self.metrics)
        self.save()
        with self.assertRaisesRegex(ValueError, 'ambiguous_duplicate_pair_attribution'):
            self.project()

    def test_signed_historical_population_recovers_exclusions(self):
        population = {'epoch': self.epoch, 'checkpoint': self.input_cp,
            'committed_count': 5, 'eligible_count': 3, 'training_count': 1,
            'exclusions': [{'document_sha256': 'ab'*32, 'reason': 'duplicate_task'}]}
        self.record['population_envelope'] = self.signed(population)
        result = self.project()['exclusions']
        self.assertEqual(result['payload']['legacy_reward_exclusions'][0]['reason'], 'duplicate_task')
        self.assertFalse(result['payload']['native_grading_evidence_available'])
        self.assertEqual(result['payload']['population_counts']['eligible_count'], 3)
        population['epoch'] = self.epoch+'wrong'
        self.record['population_envelope'] = self.signed(population)
        with self.assertRaisesRegex(ValueError, 'population_epoch_binding'): self.project()

    def test_legacy_native_context_authenticates_exclusions_and_grades(self):
        population = {'population': {'exclusions': [{'reason': 'duplicate_task', 'document_sha256': 'cc'*32}]}}
        path = self.state/(self.epoch+'-learner-population.json')
        self.write(path, population)
        context = self.signed({'original_population_file_sha256': p.hashlib.sha256(path.read_bytes()).hexdigest()})
        grade = self.signed({'rows': [{'status': 'rejected_native_labels', 'pair_sha256': 'aa'*32,
            'grades': [{'reason': 'incomplete_answer', 'complete_answer': False, 'claim': 'negative', 'secret': 'PRIVATE'}]}],
            'document_decisions': [{'accepted': False, 'batch_sha256': 'bb'*32}]})
        subset = self.signed({'accepted_count': 1, 'excluded_count': 1})
        self.manifest['native_training_eligibility_receipt'] = {
            'context_sha256': p.digest(context), 'grades_sha256': p.digest(grade), 'subset_sha256': p.digest(subset)}
        self.job['manifest'] = self.signed(self.manifest)
        self.report['job_sha256'] = self.metrics['original_job_sha256'] = p.digest(self.job)
        self.record['training_envelope'] = self.signed(self.metrics)
        directory = self.state/'native-outcome-eligibility'/self.epoch
        for name, document in [('context', context), ('grades', grade), ('subset', subset)]:
            self.write(directory/(name+'.ROOT-SIGNED.json'), document)
        self.save()
        result = self.project()['exclusions']['payload']
        self.assertEqual(result['native_checked_count'], 2)
        self.assertEqual(result['native_rejected_pairs'][0]['grades'][0]['reason'], 'incomplete_answer')
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.write(directory/'grades.ROOT-SIGNED.json', self.signed({'rows': []}))
        with self.assertRaisesRegex(ValueError, 'legacy_native_context_grade_subset_binding'): self.project()

    def log_record(self, raw):
        return {'raw': raw, 'receipt': {'epoch': self.epoch, 'job_id': self.job_id,
            'raw_sha256': p.hashlib.sha256(raw).hexdigest(), 'raw_size': len(raw),
            'job_raw_sha256': p.hashlib.sha256((self.state/'roles'/(self.job_id+'-job.json')).read_bytes()).hexdigest(),
            'original_job_bytes_match': True}}

    def test_runtime_log_allowlist_preserves_events_not_raw_secrets(self):
        raw = b'SECRET_URL=https://private/?key=PRIVATE\n' + p.canonical({
            'version': 'reference-boundary-CUDA-cache-admission-v1', 'allocated_bytes': 123,
            'private_key': 'PRIVATE', 'device': 'https://private'}) + b'\nTraceback secret /private/path\n'
        self.record['trainer_log_record'] = self.log_record(raw)
        output = self.project()['log_metrics']['payload']['runtime_log']
        self.assertEqual(output['payload']['structured_events'][0]['allocated_bytes'], 123)
        self.assertEqual(output['payload']['omitted_line_count'], 2)
        self.assertFalse(output['payload']['raw_log_signed_by_training_authority'])
        self.assertNotIn('PRIVATE', json.dumps(output))
        self.assertNotIn('/private', json.dumps(output))

    def test_runtime_log_wrong_original_job_or_bytes_rejects(self):
        self.record['trainer_log_record'] = self.log_record(b'{}')
        self.record['trainer_log_record']['receipt']['job_raw_sha256'] = 'ab'*32
        with self.assertRaisesRegex(ValueError, 'retained_runtime_log_identity'): self.project()

    def test_exact_historical_signed_empty_closure_schema(self):
        self.completion['next_checkpoint'] = self.input_cp
        self.record['completion_envelope'] = self.signed(self.completion)
        self.record['training_envelope'] = self.signed({'checkpoint': self.input_cp,
            'status': 'closed_no_eligible_batches'})
        result = self.project()
        self.assertEqual(result['training']['payload']['status'], 'closed_no_eligible_batches')
        self.assertFalse(result['training']['provenance']['training_receipt_contains_epoch'])
        self.assertNotIn('steps', result['training']['payload'])
        self.assertEqual(result['training_inputs']['reason'], 'no_training_job_for_signed_empty_closure')
        self.record['training_envelope'] = self.signed({'checkpoint': self.output_cp,
            'status': 'closed_no_eligible_batches'})
        with self.assertRaises((ValueError, KeyError)): self.project()

    def test_missing_historical_dtype_is_explicit_without_inference(self):
        self.updates[0].pop('gradient_accumulation_dtype')
        self.record['training_envelope'] = self.signed(self.metrics)
        self.save()
        update = self.project()['training']['payload']['updates'][0]
        self.assertNotIn('gradient_accumulation_dtype', update)
        self.assertEqual(update['metric_availability']['gradient_accumulation_dtype'], 'not_retained')
        self.assertEqual(update['metric_availability']['learning_rate'], 'recorded')
        self.assertEqual(update['hyperparameters']['lr'], 5e-7)

    def test_log_progress_keeps_numbers_and_completion_without_suffix(self):
        raw = (b'Loading weights: 100%|######| 339/339 [00:03<00:00, 12.3it/s, SECRET=/private/path]\n'
               b'Writing model shards: 25%|##| 1/4 [00:13<00:40, 1.0s/it]\n'
               b'Writing model shards: 100%|####| 4/4 [01:29<00:00, 1.0s/it]\n\n')
        self.record['trainer_log_record'] = self.log_record(raw)
        payload = self.project()['log_metrics']['payload']['runtime_log']['payload']
        events = payload['structured_events']
        self.assertEqual([row['elapsed_seconds'] for row in events], [3,13,89])
        self.assertEqual([row['progress_reports_completion'] for row in events], [True,False,True])
        self.assertEqual(payload['line_classification_counts']['progress_observations'], 3)
        self.assertEqual(payload['line_classification_counts']['blank'], 1)
        self.assertEqual(payload['line_classification_counts']['recognized_warning_markers'], 0)
        self.assertEqual(payload['omitted_line_count'], 1)
        self.assertNotIn('SECRET', json.dumps(payload))
        self.assertNotIn('/private', json.dumps(payload))

    def test_unknown_log_lines_do_not_become_zero_warning_claim(self):
        raw = b'WARNING token=SECRET url=https://private\nunknown diagnostic PRIVATE\n'
        self.record['trainer_log_record'] = self.log_record(raw)
        payload = self.project()['log_metrics']['payload']['runtime_log']['payload']
        self.assertEqual(payload['line_classification_counts']['recognized_warning_markers'], 1)
        self.assertEqual(payload['line_classification_counts']['unrecognized_text'], 1)
        self.assertEqual(payload['warning_detection_scope'], 'recognized_warning_markers_only')
        self.assertNotIn('SECRET', json.dumps(payload))
        self.assertNotIn('PRIVATE', json.dumps(payload))
        self.assertEqual(payload['structured_events'][0]['marker'], 'WARNING')

    def test_invalid_progress_counts_and_timer_remain_unclassified(self):
        for line in ('Writing model shards: 101%|##| 4/4 [00:02<00:00, ignored]',
                     'Loading weights: 100%|##| 340/339 [00:02<00:00, ignored]',
                     'Loading weights: 100%|##| 339/339 [00:99<00:00, ignored]'):
            self.assertIsNone(p.progress_projection(line))

    def test_native_pool_pins_legacy_reward_exclusions(self):
        population = {'population': {'exclusions': [{'reason': 'duplicate_task', 'document_sha256': 'cc'*32}]}}
        population_path = self.state/(self.epoch+'-learner-population.json')
        self.write(population_path, population)
        pool = self.signed({'original_population_file_sha256': p.hashlib.sha256(population_path.read_bytes()).hexdigest()})
        result = self.signed({'native_waves': [], 'checked_count': 1, 'accepted_count': 1, 'sampling_assurance': 'unaudited'})
        self.manifest['native_training_eligibility_receipt'] = {'pool_sha256': p.digest(pool), 'result_sha256': p.digest(result)}
        self.job['manifest'] = self.signed(self.manifest)
        self.report['job_sha256'] = self.metrics['original_job_sha256'] = p.digest(self.job)
        self.record['training_envelope'] = self.signed(self.metrics)
        directory = self.state/'native-outcome-eligibility'/self.epoch
        self.write(directory/'pool.ROOT-SIGNED.json', pool)
        self.write(directory/'result.ROOT-SIGNED.json', result)
        self.save()
        output = self.project()['exclusions']
        self.assertTrue(output['payload']['legacy_reward_exclusions_are_not_training_exclusions'])
        self.assertEqual(output['payload']['legacy_reward_exclusions'][0]['reason'], 'duplicate_task')
        population_path.write_text('{}')
        with self.assertRaises(ValueError): self.project()

    def selection_fixture(self):
        extra = copy.deepcopy(self.job['submissions'][0])
        body = dict(self.admission, document_sha256='aa'*32)
        extra.update(sha256=body['document_sha256'], learner_admission=self.signed(body))
        selected = p.submission_inventory(self.job['submissions'], self.epoch, self.input_cp, self.authority)
        eligible = p.submission_inventory(self.job['submissions']+[extra], self.epoch, self.input_cp, self.authority)
        population = dict(eligible_inventory=eligible, training_selection=dict(version='bounded-postfreeze-learner-selection-v1',
            cap=1, eligible_count=2, training_count=1, unselected_count=1,
            eligible_inventory_sha256=p.digest(eligible), selected_inventory_sha256=p.digest(selected)),
            committed_inventory=[dict(training_documents=[dict(sha256=r['sha256'],size=r['size']) for r in eligible]
                +[dict(sha256='bb'*32,size=100)])],
            exclusions=[dict(document_sha256='bb'*32,reason='duplicate_task')])
        return population, extra

    def test_retained_legacy_cap_enumerates_unselected_and_actual_training(self):
        population, _ = self.selection_fixture()
        population.update(epoch=self.epoch, checkpoint=self.input_cp)
        self.record['population_envelope'] = self.signed(population)
        result = self.project()['exclusions']['payload']['training_selection']
        rows = result['decisions']['payload']['rows']
        self.assertEqual([r['status'] for r in rows], ['actually_trained','eligible_but_not_selected'])
        self.assertEqual(rows[1]['reason'], 'bounded_selection_cap')
        self.assertEqual(result['legacy_selection']['payload']['unselected_count'], 1)
        outside = result['captured_outside_selection_inventory']['payload']['rows'][0]
        self.assertEqual(outside['recorded_population_reason'], 'duplicate_task')
        self.assertEqual(outside['training_rejection_reason'], 'not_retained')
        self.assertNotIn('PRIVATE',json.dumps(result))

    def test_selection_does_not_invent_cap_reason_without_inventory_binding(self):
        population, _ = self.selection_fixture()
        population['training_selection']['selected_inventory_sha256'] = 'cc'*32
        result = p.training_selection_projection(population,self.manifest,self.authority,
            submissions=self.job['submissions'],trained_documents={'55'*32})
        self.assertEqual(result['decisions']['payload']['rows'][1]['reason'],'not_retained')

    def test_selected_unused_is_separate_from_actually_trained(self):
        population, _ = self.selection_fixture()
        result = p.training_selection_projection(population,self.manifest,self.authority,
            submissions=self.job['submissions'],trained_documents=set())
        self.assertEqual(result['decisions']['payload']['rows'][0]['status'],'selected_but_not_used_by_gradient')
        self.assertEqual(result['decisions']['payload']['actually_trained_count'],0)

    def test_native_full_pool_ignores_legacy_reward_selection_count(self):
        population, extra = self.selection_fixture()
        population['eligible_inventory'] = []
        population['training_selection']['training_count'] = 0
        result = p.training_selection_projection(population,self.manifest,self.authority,
            submissions=self.job['submissions'],trained_documents={'55'*32},
            native_pool=self.job['submissions']+[extra],native_draw={'cap':1},
            native_result={'complete':True,'completion_reason':'selection_complete',
                           'accepted_count':1,'checked_count':1,'accepted_submissions':self.job['submissions']})
        self.assertTrue(result['legacy_selection_is_not_native_final_selection'])
        self.assertEqual(result['decisions']['payload']['eligible_count'],2)
        self.assertEqual(result['decisions']['payload']['selected_count'],1)
        self.assertEqual(result['native_selection']['payload']['cap'],1)
        self.assertEqual(result['decisions']['payload']['rows'][1]['reason'],'not_retained')

    def test_native_rejection_reason_joins_by_original_pair_and_admission(self):
        population, extra = self.selection_fixture()
        grade={'document_decisions':[{'accepted':False,'document_sha256':extra['sha256'],
            'learner_admission_sha256':p.digest(extra['learner_admission']),'pair_sha256':['ee'*32]}],
            'rows':[{'pair_sha256':'ee'*32,'grades':[{'reason':'incomplete_answer'}]}]}
        result=p.training_selection_projection(population,self.manifest,self.authority,
            submissions=self.job['submissions'],trained_documents={'55'*32},native_pool=self.job['submissions']+[extra],
            native_result={'accepted_submissions':self.job['submissions']},grade_documents=[grade])
        row=result['decisions']['payload']['rows'][1]
        self.assertEqual(row['status'],'native_rejected');self.assertEqual(row['native_reasons'],['incomplete_answer'])
        grade['document_decisions'][0]['learner_admission_sha256']='cc'*32
        with self.assertRaisesRegex(ValueError,'checked_inventory_binding'):
            p.training_selection_projection(population,self.manifest,self.authority,
                submissions=self.job['submissions'],trained_documents={'55'*32},native_pool=self.job['submissions']+[extra],
                native_result={},grade_documents=[grade])

    def test_changed_native_accepted_set_refuses(self):
        population, extra = self.selection_fixture()
        with self.assertRaisesRegex(ValueError,'actual_training_selection_binding'):
            p.training_selection_projection(population,self.manifest,self.authority,
                submissions=self.job['submissions'],trained_documents={'55'*32},native_pool=self.job['submissions']+[extra],
                native_result={'accepted_submissions':[extra]})

    def test_absent_old_native_pool_keeps_context_and_marks_decisions_missing(self):
        population,_=self.selection_fixture()
        result=p.training_selection_projection(population,self.manifest,self.authority,
            submissions=self.job['submissions'],trained_documents={'55'*32},native_result={'accepted_count':1})
        self.assertEqual(result['decisions']['status'],'unavailable')
        self.assertEqual(result['native_selection']['payload']['draw_status'],'not_retained')

    def test_selection_hash_types_and_duplicate_identities_refuse(self):
        population,_=self.selection_fixture()
        row=population['eligible_inventory'][0]
        for field in ('sha256','learner_admission_sha256'):
            with self.assertRaises(ValueError):p.inventory_projection([dict(row,**{field:[]})])
        with self.assertRaises(ValueError):p.inventory_projection([row,row])

    def native_selection_fixture(self):
        objects=[]
        for identity,task in [('55'*32,7),('aa'*32,7),('bb'*32,8)]:
            admission=dict(self.admission,document_sha256=identity,miner_identity=self.authority)
            commitment=dict(epoch=self.epoch,checkpoint=self.input_cp,miner=self.authority,
                batches=[dict(slot=0,training_sha256=identity,training_size=100,
                              batch_sha256=admission['batch_sha256'],env_id='affine_math',index=task)])
            admission['original_commitment']=self.signed(commitment)
            objects.append(dict(sha256=identity,size=100,learner_admission=self.signed(admission),url='PRIVATE'))
        return dict(submissions=objects[:1],trained_documents={'55'*32},native_pool=objects,
            native_result=dict(complete=True,completion_reason='selection_complete',accepted_count=1,
                checked_count=1,accepted_submissions=objects[:1]),
            native_draw=dict(cap=1,blacklist_selection=dict(exclusions=[],excluded_miners=[])),
            grade_documents=[dict(document_decisions=[dict(accepted=True,document_sha256='55'*32,
                learner_admission_sha256=p.digest(objects[0]['learner_admission']))])])

    def test_native_unchecked_reasons_reconstructed_from_committed_tasks(self):
        kwargs=self.native_selection_fixture()
        result=p.training_selection_projection({},self.manifest,self.authority,**kwargs)
        rows=result['decisions']['payload']['rows']
        self.assertEqual([r['reason'] for r in rows],[None,'another_task_representative_selected','native_task_cap_reached'])
        self.assertEqual(rows[1]['task'],dict(env_id='affine_math',index=7))
        self.assertEqual(rows[2]['reason_basis'],'reconstructed_from_signed_selection_and_task_commitments')
        self.assertEqual(rows[2]['status'],'eligible_but_not_selected')
        self.assertNotIn('PRIVATE',json.dumps(result))

    def test_native_skip_reasons_require_draw_counts_and_no_blacklist(self):
        for change in ('missing_draw','unbound_checked_count','blacklist'):
            kwargs=self.native_selection_fixture()
            if change=='missing_draw':kwargs['native_draw']=None
            elif change=='unbound_checked_count':kwargs['native_result']['checked_count']=2
            else:kwargs['native_draw']['blacklist_selection']['excluded_miners']=[self.authority]
            result=p.training_selection_projection({},self.manifest,self.authority,**kwargs)
            self.assertEqual([r['reason'] for r in result['decisions']['payload']['rows'][1:]],['not_retained','not_retained'])
        kwargs=self.native_selection_fixture();kwargs['native_result'].update(complete=False,completion_reason='native_budget_exhausted')
        result=p.training_selection_projection({},self.manifest,self.authority,**kwargs)
        self.assertEqual(result['decisions']['payload']['rows'][2]['reason'],'not_retained')

    def test_native_task_commitment_cannot_relabel_a_submission(self):
        kwargs=self.native_selection_fixture();obj=kwargs['native_pool'][1]
        admission=copy.deepcopy(obj['learner_admission']['payload']);commitment=copy.deepcopy(admission['original_commitment']['payload'])
        commitment['batches'][0]['training_sha256']='cc'*32
        admission['original_commitment']=self.signed(commitment);obj['learner_admission']=self.signed(admission)
        with self.assertRaisesRegex(ValueError,'native_task_commitment_batch_identity'):
            p.training_selection_projection({},self.manifest,self.authority,**kwargs)

    def test_accepted_native_decision_requires_matching_admission(self):
        kwargs=self.native_selection_fixture()
        kwargs['grade_documents'][0]['document_decisions'][0]['learner_admission_sha256']='ff'*32
        with self.assertRaisesRegex(ValueError,'native_checked_inventory_binding'):
            p.training_selection_projection({},self.manifest,self.authority,**kwargs)


if __name__ == '__main__':
    unittest.main()
