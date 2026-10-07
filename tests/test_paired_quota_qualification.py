import copy
import unittest

from ops.paired_quota_qualification import ApprovedTask, identities, select_pairs


class PairedQuotaControls(unittest.TestCase):
    def setUp(self):
        self.task = ApprovedTask('epoch-40', 'checkpoint-A', 'a' * 64, 'math', 10,
                                 'b' * 64, 'c' * 64, 'd' * 64, tuple(range(10)))
        self.rows = [self.row(i, 'positive' if i < 2 else 'negative') for i in range(4)]

    def row(self, attempt, label):
        return dict(self.task.task_binding(), epoch=self.task.epoch,
                    harness_sha256=self.task.harness_sha256,
                    sampling_context_sha256=self.task.sampling_context_sha256,
                    attempt=attempt, classification=label,
                    turns=[dict(prompt=[10, 20], output=[100 + attempt],
                                actions=[], observations=['feedback'])])

    def test_four_distinct_nonoverlap_task_normalized(self):
        result = select_pairs(self.task, 'miner-A', self.rows, quota=2)
        ids = [r['content_id'] for p in result['pairs'] for r in (p['positive'], p['negative'])]
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(result['pair_weight_within_task'], .5)
        self.assertEqual(result['contribution_units'], 1)

    def test_default_stays_one_pair(self):
        self.assertEqual(len(select_pairs(self.task, 'miner-A', self.rows)['pairs']), 1)

    def test_repacking_order_and_redelivery_are_same_revision(self):
        original = select_pairs(self.task, 'miner-A', self.rows, quota=2)
        repacked = copy.deepcopy(self.rows[::-1])
        for row in repacked:
            row.update(filename='new-name', proof={'different': 'encoding'},
                       upload_timestamp=99, claimed_reward=999, miner='other-label')
            row['turns'][0]['proof'] = ['untrusted wrapper']
        changed = select_pairs(self.task, 'miner-A', repacked + repacked, quota=2)
        self.assertEqual(original['revision_id'], changed['revision_id'])
        self.assertEqual(original['pairs'], changed['pairs'])

    def test_same_attempt_different_content_cannot_fill_quota(self):
        rows = copy.deepcopy(self.rows)
        rows[1]['attempt'] = rows[0]['attempt']
        with self.assertRaisesRegex(ValueError, 'conflicting prescribed attempt'):
            select_pairs(self.task, 'miner-A', rows, quota=2)

    def test_same_attempt_different_label_refused(self):
        rows = copy.deepcopy(self.rows)
        duplicate = copy.deepcopy(rows[0])
        duplicate['classification'] = 'negative'
        with self.assertRaisesRegex(ValueError, 'conflicting prescribed attempt'):
            select_pairs(self.task, 'miner-A', rows + [duplicate], quota=2)

    def test_repeated_output_new_attempt_does_not_fill_quota(self):
        rows = copy.deepcopy(self.rows)
        rows[1]['turns'] = copy.deepcopy(rows[0]['turns'])
        self.assertNotEqual(identities(self.task, rows[0])[0], identities(self.task, rows[1])[0])
        self.assertEqual(identities(self.task, rows[0])[1], identities(self.task, rows[1])[1])
        with self.assertRaisesRegex(ValueError, 'quota not met'):
            select_pairs(self.task, 'miner-A', rows, quota=2)
        result = select_pairs(self.task, 'miner-A', rows, quota=1)
        self.assertEqual(result['duplicate_content_count'], 1)

    def test_relabel_does_not_change_content_or_create_negative(self):
        positive = self.rows[0]
        relabeled = copy.deepcopy(positive)
        relabeled.update(attempt=5, classification='negative', claimed_reward=-1)
        self.assertEqual(identities(self.task, positive)[1], identities(self.task, relabeled)[1])
        with self.assertRaisesRegex(ValueError, 'conflicting claimed content classification'):
            select_pairs(self.task, 'miner-A', self.rows + [relabeled], quota=2)

    def test_cross_uid_copy_keeps_content_and_execution(self):
        copied = copy.deepcopy(self.rows[0])
        copied.update(miner='miner-B', uid=85)
        self.assertEqual(identities(self.task, self.rows[0]), identities(self.task, copied))
        self.assertNotEqual(self.task.slot_id('miner-A'), self.task.slot_id('miner-B'))

    def test_old_checkpoint_and_wrong_epoch_refused(self):
        for field, value in [('checkpoint', 'checkpoint-old'), ('epoch', 'epoch-old')]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.assertRaisesRegex(ValueError, 'binding'):
                select_pairs(self.task, 'miner-A', rows, quota=2)

    def test_changed_trace_is_content_change(self):
        for field in ('prompt', 'output', 'actions', 'observations'):
            row = copy.deepcopy(self.rows[0])
            row['turns'][0][field].append(44)
            self.assertNotEqual(identities(self.task, self.rows[0])[1], identities(self.task, row)[1])

    def test_complete_revision_repeat_attempt_does_not_add_reward(self):
        result = select_pairs(self.task, 'miner-A', self.rows + self.rows, quota=2)
        self.assertEqual(result['contribution_units'], 1)
        self.assertEqual(result['revision_id'], select_pairs(self.task, 'miner-A', self.rows, quota=2)['revision_id'])

    def test_context_harness_and_attempt_require_approval(self):
        for field, value in [('sampling_context_sha256', 'f' * 64),
                             ('harness_sha256', 'f' * 64), ('attempt', 99), ('attempt', True)]:
            row = copy.deepcopy(self.rows[0]); row[field] = value
            with self.assertRaises(ValueError):
                identities(self.task, row)

    def test_missing_trace_and_noncanonical_tokens_refused(self):
        for field, value in [('observations', None), ('actions', None), ('output', [True]),
                             ('prompt', []), ('output', [200000])]:
            row = copy.deepcopy(self.rows[0]); row['turns'][0][field] = value
            with self.assertRaises(ValueError):
                identities(self.task, row)

    def test_missing_opposite_class_cannot_reuse_one_member(self):
        with self.assertRaisesRegex(ValueError, 'quota not met'):
            select_pairs(self.task, 'miner-A', [self.rows[0], self.rows[1], self.rows[2]], quota=2)

    def test_noncanonical_trace_and_size_budget(self):
        for value in ([{1: "ambiguous key"}], [float('nan')], [float('inf')], ['x' * 1048576]):
            row = copy.deepcopy(self.rows[0]); row['turns'][0]['observations'] = value
            with self.assertRaises(ValueError):
                identities(self.task, row)
        row = copy.deepcopy(self.rows[0]); row['turns'][0]['output'] = [1] * 2049
        with self.assertRaises(ValueError):
            identities(self.task, row)
        row = copy.deepcopy(self.rows[0]); row['turns'][0]['prompt'] = [1] * 8192
        with self.assertRaises(ValueError):
            identities(self.task, row)

    def test_quota_and_task_identity_strict(self):
        for quota in (True, 0, 3):
            with self.assertRaises(ValueError):
                select_pairs(self.task, 'miner-A', self.rows, quota=quota)
        with self.assertRaises(ValueError):
            ApprovedTask('e', 'cp', 'bad', 'math', 1, 'b' * 64, 'c' * 64, 'd' * 64, (0,))


if __name__ == '__main__':
    unittest.main()
