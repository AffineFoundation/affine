"""GRPO pack authenticity gate: verified-or-nothing."""
import unittest

from subnet.mimo_training_authenticity import (
    ASSURANCE_VERIFIED, VERSION, batch_key, build_train_pack_payload,
    denylist_keys_from_document, filter_train_pack, outcomes_from_ledger,
    train_eligible_outcome)


def row(miner='a' * 64, batch='b' * 64, index=0):
    return dict(miner=miner, env_id='affine_math', index=index, batch_sha256=batch,
                document=dict(sha256='c' * 64, key='k'), native_grades={'sampling_verified': False},
                assurance='natively-graded-sampling-unaudited')


class TrainingAuthenticityTests(unittest.TestCase):
    def test_only_verified_valid_is_train_eligible(self):
        self.assertTrue(train_eligible_outcome('verified_valid'))
        for outcome in ('confirmed_invalid', 'pending', 'unaudited', None, 'structural_reject'):
            self.assertFalse(train_eligible_outcome(outcome))

    def test_filter_strips_fail_and_unverified(self):
        m, good, bad, pending = 'a' * 64, '1' * 64, '2' * 64, '3' * 64
        selected = [row(m, good, 0), row(m, bad, 1), row(m, pending, 2)]
        outcomes = {
            batch_key(m, good): 'verified_valid',
            batch_key(m, bad): 'confirmed_invalid',
            # pending missing from ledger => unverified
        }
        eligible, excluded = filter_train_pack(selected, outcomes)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0]['batch_sha256'], good)
        self.assertEqual(eligible[0]['assurance'], ASSURANCE_VERIFIED)
        self.assertEqual(eligible[0]['sampling_audit_outcome'], 'verified_valid')
        self.assertEqual({r['reason'] for r in excluded},
                         {'sampling-verify-fail', 'sampling-unverified'})

    def test_denylist_strips_even_without_ledger_row(self):
        m, batch = 'a' * 64, 'd' * 64
        selected = [row(m, batch)]
        key = batch_key(m, batch)
        eligible, excluded = filter_train_pack(selected, {}, denylist_keys={key})
        self.assertEqual(eligible, [])
        self.assertEqual(excluded[0]['reason'], 'sampling-verify-denylist')

    def test_train_pack_payload_requires_audit_flag(self):
        m, batch = 'a' * 64, 'e' * 64
        sealed = dict(version='mimo-complete-group-coordinator-v1', epoch='e1', run_id='r1',
                      checkpoint='f' * 64, manifest_sha256='0' * 64, population_sha256='1' * 64,
                      selection={'seed': 's', 'max_tasks': 32}, selected=[row(m, batch)],
                      excluded=[], training_requires_sampling_audit=False,
                      selected_task_count=1, selected_rollout_count=8)
        pack = build_train_pack_payload(sealed, {batch_key(m, batch): 'verified_valid'})
        self.assertTrue(pack['training_requires_sampling_audit'])
        self.assertEqual(pack['authenticity_gate'], VERSION)
        self.assertEqual(pack['selected_task_count'], 1)
        self.assertEqual(pack['selected'][0]['assurance'], ASSURANCE_VERIFIED)

    def test_unverified_pack_is_empty(self):
        sealed = dict(version='mimo-complete-group-coordinator-v1', epoch='e1', run_id='r1',
                      checkpoint='f' * 64, manifest_sha256='0' * 64, selected=[row()],
                      excluded=[], selected_task_count=1, selected_rollout_count=8)
        pack = build_train_pack_payload(sealed, {})
        self.assertEqual(pack['selected'], [])
        self.assertEqual(pack['sampling_verify_stripped'], 1)
        self.assertEqual(pack['selected_task_count'], 0)

    def test_ledger_and_denylist_parsers(self):
        m, batch = 'a' * 64, 'b' * 64
        outcomes = outcomes_from_ledger({'rows': [dict(miner=m, batch_sha256=batch,
                                                       outcome='verified_valid')]})
        self.assertEqual(outcomes[batch_key(m, batch)], 'verified_valid')
        keys = denylist_keys_from_document({'rows': [dict(miner=m, batch=batch)]})
        self.assertEqual(keys, {batch_key(m, batch)})


if __name__ == '__main__':
    unittest.main()
