import copy
import unittest

from ops.paired_quota_batch_adapter import CurrentBatchAdapter, CumulativeTaskSlot
from ops.paired_quota_qualification import identities
from subnet import forced_sampling, harness


class CurrentWireAdapterControls(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(epoch='research-40', checkpoint={'id': 'a' * 64}, K=1, L=1,
            harness_source_hash=harness.source_hash(), sampling_source_hash=forced_sampling.source_hash(),
            sampling_contract=dict(version=forced_sampling.VERSION, randomness='b' * 64,
                max_attempts=8, verification='exact-token-replay', generation='uncached-eager-inverse-cdf'),
            environments=[dict(env_id='math', indices=[10, 11], spec=dict(id='math', version='v1', num_samples=32,
                max_turns=4, max_output_tokens=64, config={'seed': 7}), harness=dict(harness.DEFAULT))])
        self.decode = lambda output: ''.join(chr(i) for i in output)
        self.adapter = CurrentBatchAdapter.from_manifest(self.manifest, 'math', 10,
                                                         task_sha256='c' * 64, decode=self.decode)
        self.rolls = [self.roll(i, 'positive' if i < 2 else 'negative') for i in range(4)]

    def roll(self, seed, label):
        context = forced_sampling.binding(self.manifest)
        text = str(seed)
        return dict(schema=2, env_id='math', environment_version='v1', index=10, sample_index=10,
                    seed=seed, env_seed=7, task_hash='c' * 64, classification=label,
                    reward=1 if label == 'positive' else 0,
                    sampling=forced_sampling.receipt(context, seed),
                    turns=[dict(prompt=[10, 20], output=[ord(text)], text=text,
                        observations=[dict(role='user', content='feedback')], done=True,
                        reward=1, classification=label, proofs=['untrusted packing'])])

    def batch(self, rolls):
        return dict(schema=2, epoch=self.manifest['epoch'], checkpoint='a' * 64,
                    env_id='math', environment_version='v1', index=10, sample_index=10,
                    rollouts=copy.deepcopy(rolls))

    def test_actual_schema_two_maps_draw_tokens_actions_observations(self):
        rows = self.adapter.normalize(self.batch(self.rolls))
        self.assertEqual(rows[0]['attempt'], self.rolls[0]['seed'])
        self.assertEqual(rows[0]['turns'][0]['actions'], [{'text': '0', 'tool_calls': []}])
        self.assertEqual(rows[0]['turns'][0]['observations'], [{'role': 'user', 'content': 'feedback'}])
        slot = CumulativeTaskSlot(self.adapter, 'miner-A')
        slot.add_revision(self.batch(self.rolls))
        self.assertEqual(len(slot.select(quota=2)['pairs']), 2)
        self.assertEqual(len(slot.select()['pairs']), 1)
        self.assertEqual(self.manifest['K'], 1)

    def test_cumulative_partial_append_then_repack_redelivery(self):
        slot = CumulativeTaskSlot(self.adapter, 'miner-A')
        first = slot.add_revision(self.batch(self.rolls[:1]))
        with self.assertRaisesRegex(ValueError, 'quota'):
            slot.select(quota=2)
        full = slot.add_revision(self.batch(self.rolls))
        repacked = self.batch(self.rolls[::-1] + self.rolls)
        repacked.update(filename='new', proof_encoding='other', updated_at=-1)
        for row in repacked['rollouts']:
            row['turns'][0]['proofs'] = {'different': 'bytes'}
            row['turns'][0]['observations'][0]['proof'] = 'ignored wrapper'
            row.update(uid=85, miner='different claimed identity', reward=444)
        repeated = slot.add_revision(repacked)
        self.assertNotEqual(first['revision_id'], full['revision_id'])
        self.assertEqual(full['revision_id'], repeated['revision_id'])
        self.assertTrue(repeated['redelivery'])
        self.assertEqual(repeated['stored_revision_count'], 2)
        self.assertEqual(slot.select(quota=2)['contribution_units'], 1)

    def test_revisions_cannot_remove_or_rewrite_attempt(self):
        for mutation in ('remove', 'tokens', 'label'):
            slot = CumulativeTaskSlot(self.adapter, 'miner-A')
            before = slot.add_revision(self.batch(self.rolls))
            changed = self.batch(self.rolls)
            if mutation == 'remove':
                changed['rollouts'].pop()
            elif mutation == 'tokens':
                turn = changed['rollouts'][0]['turns'][0]
                turn.update(output=[ord('x')], text='x')
            else:
                changed['rollouts'][0]['classification'] = 'negative'
            with self.assertRaisesRegex(ValueError, 'cumulative revision'):
                slot.add_revision(changed)
            self.assertEqual(slot.add_revision(self.batch(self.rolls))['revision_id'], before['revision_id'])
            self.assertEqual(len(slot.select(quota=2)['pairs']), 2)

    def test_repeated_draw_cannot_become_second_member(self):
        rows = copy.deepcopy(self.rolls)
        rows[1].update(seed=0, sampling=rows[0]['sampling'])
        with self.assertRaisesRegex(ValueError, 'conflicting cumulative attempt'):
            CumulativeTaskSlot(self.adapter, 'miner-A').add_revision(self.batch(rows))

    def test_same_content_new_draw_not_fraud_and_not_quota(self):
        rows = copy.deepcopy(self.rolls)
        rows[1]['turns'] = copy.deepcopy(rows[0]['turns'])
        slot = CumulativeTaskSlot(self.adapter, 'miner-A')
        result = slot.add_revision(self.batch(rows))
        self.assertEqual(result['unique_attempts'], 4)
        self.assertEqual(result['unique_contents'], 3)
        with self.assertRaisesRegex(ValueError, 'quota'):
            slot.select(quota=2)
        self.assertEqual(slot.select()['duplicate_content_count'], 1)

    def test_old_checkpoint_epoch_and_different_task_refused(self):
        for key, value in [('checkpoint', 'f' * 64), ('epoch', 'old'), ('index', 11), ('sample_index', True)]:
            batch = self.batch(self.rolls); batch[key] = value
            with self.assertRaisesRegex(ValueError, 'batch binding'):
                self.adapter.normalize(batch)

    def test_receipt_is_recomputed_from_approved_context(self):
        for mutation in ('context', 'seed', 'env_seed', 'task_hash'):
            batch = self.batch(self.rolls)
            row = batch['rollouts'][0]
            if mutation == 'context':
                row['sampling']['binding_sha256'] = 'f' * 64
            elif mutation == 'seed':
                row['seed'] = 99
            elif mutation == 'env_seed':
                row['env_seed'] = 8
            else:
                row['task_hash'] = 'f' * 64
            with self.assertRaises(ValueError):
                self.adapter.normalize(batch)

    def test_decode_and_observation_wire_validation(self):
        for mutation in ('text', 'observations', 'done', 'output'):
            batch = self.batch(self.rolls)
            turn = batch['rollouts'][0]['turns'][0]
            turn[mutation] = {'text': 'fabricated', 'observations': [{'role': 'invalid', 'content': 'x'}],
                              'done': False, 'output': [True]}[mutation]
            with self.assertRaises(ValueError):
                self.adapter.normalize(batch)

    def test_multi_turn_tool_action_and_every_observation_bound(self):
        batch = self.batch(self.rolls[:1])
        turn = batch['rollouts'][0]['turns'][0]
        text = '{"tool_call":{"name":"search","arguments":{"q":"x"}}}'
        turn.update(output=[ord(c) for c in text], text=text, done=False,
                    observations=[dict(role='tool', content='result')])
        final = copy.deepcopy(turn); final.update(output=[ord('A')], text='A', done=True)
        batch['rollouts'][0]['turns'].append(final)
        original = self.adapter.normalize(batch)[0]
        self.assertEqual(original['turns'][0]['actions'][0]['tool_calls'][0]['name'], 'search')
        changed = copy.deepcopy(batch)
        changed['rollouts'][0]['turns'][1]['observations'][0]['content'] = 'different'
        self.assertNotEqual(identities(self.adapter.task, original)[1],
                            identities(self.adapter.task, self.adapter.normalize(changed)[0])[1])

    def test_cross_uid_copy_same_content_separate_slot(self):
        batch = self.batch(self.rolls)
        rows = self.adapter.normalize(batch)
        copied = copy.deepcopy(batch); copied['miner'] = 'miner-B'
        self.assertEqual([identities(self.adapter.task, r) for r in rows],
                         [identities(self.adapter.task, r) for r in self.adapter.normalize(copied)])
        slots = [CumulativeTaskSlot(self.adapter, miner) for miner in ('miner-A', 'miner-B')]
        for slot in slots:
            slot.add_revision(batch)
        self.assertNotEqual(slots[0].select(quota=2)['slot_id'], slots[1].select(quota=2)['slot_id'])

    def test_observation_only_rewrite_cannot_fill_extra_quota(self):
        rows = copy.deepcopy(self.rolls)
        rows[1]['turns'] = copy.deepcopy(rows[0]['turns'])
        rows[1]['turns'][0]['observations'][0]['content'] = 'invented feedback'
        with self.assertRaisesRegex(ValueError, 'conflicting cumulative token trace'):
            CumulativeTaskSlot(self.adapter, 'miner-A').add_revision(self.batch(rows))

    def test_indexed_harness_is_resolved_and_pinned(self):
        manifest = copy.deepcopy(self.manifest)
        configs = {str(i): dict(harness.DEFAULT, temperature=.6 if i == 10 else .9) for i in (10, 11)}
        indexed = dict(version='indexed-harness-v1', by_index=configs)
        manifest['environments'][0]['harness'] = indexed
        manifest['sample_harness_registry'] = {'math': dict(indices=[10, 11], harness=indexed)}
        a = CurrentBatchAdapter.from_manifest(manifest, 'math', 10, task_sha256='c' * 64, decode=self.decode)
        b = CurrentBatchAdapter.from_manifest(manifest, 'math', 11, task_sha256='c' * 64, decode=self.decode)
        self.assertNotEqual(a.task.harness_sha256, b.task.harness_sha256)
        # Draw binding is shared; per-index harness semantics are independently pinned.
        self.assertEqual(a.task.sampling_context_sha256, b.task.sampling_context_sha256)
        self.assertEqual(len(a.normalize(self.batch(self.rolls))), 4)

    def test_heldout_unapproved_index_and_source_pins_refused(self):
        for mutation in ('heldout', 'index', 'source'):
            manifest = copy.deepcopy(self.manifest); index = 10
            if mutation == 'heldout': manifest['heldout_indices'] = {'math': [10]}
            elif mutation == 'index': index = 12
            else: manifest['sampling_source_hash'] = 'f' * 64
            with self.assertRaises(ValueError):
                CurrentBatchAdapter.from_manifest(manifest, 'math', index,
                                                   task_sha256='c' * 64, decode=self.decode)


if __name__ == '__main__':
    unittest.main()
