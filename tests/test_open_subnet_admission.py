"""Open admission uses authenticated chain snapshots at real epoch boundaries."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from subnet.gpu_service import admitted_registrations, registration_policy, run


class ReachedOpening(Exception):
    pass


class OpenSubnetAdmission(unittest.TestCase):
    def setUp(self):
        self.first = {'owned': {'uid': 131, 'public_key': 'owned-public'}}
        self.second = dict(self.first, external={'uid': 85, 'public_key': 'external-public'})

    def test_all_authenticated_subnet_identities_are_eligible(self):
        self.assertEqual(admitted_registrations(
            {'registration_policy': 'all_activated_subnet'}, self.second), self.second)

    def test_open_policy_has_no_private_approval_list(self):
        for values in ([], ['owned-public']):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'must omit'):
                registration_policy({'registration_policy': 'all_activated_subnet',
                                     'registration_allowlist': values})

    def test_legacy_filter_remains_explicit(self):
        self.assertEqual(admitted_registrations(
            {'registration_allowlist': ['owned-public']}, self.second), self.first)
        self.assertEqual(admitted_registrations({'registration_allowlist': []}, self.second), {})

    def test_missing_or_malformed_legacy_list_never_opens_admission(self):
        for values in (None, 'owned-public', [123], ['owned-public', 'owned-public']):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'allowlist required'):
                admitted_registrations({'registration_allowlist': values}, self.second)

    def test_unknown_policy_fails_before_side_effects(self):
        for value in ('all', 'public', True, None, [], {}):
            with self.subTest(value=value), patch('subnet.gpu_service.Bucket') as bucket, \
                    patch('subnet.gpu_service.Path') as path, self.assertRaisesRegex(ValueError, 'unknown registration'):
                run({'registration_policy': value})
            bucket.assert_not_called()
            path.assert_not_called()

    def test_selection_does_not_modify_authenticated_snapshot(self):
        selected = admitted_registrations({'registration_policy': 'all_activated_subnet'}, self.second)
        selected.pop('external')
        self.assertIn('external', self.second)

    def fixtures(self, directory, chain, openings, stack):
        state = Path(directory)
        checkpoint = {'id': 'fixture-checkpoint', 'files': {}}
        config = dict(state=str(state), bucket={}, remote={}, initial_checkpoint=checkpoint,
                      initial_checkpoint_path='/fixture/checkpoint', owned_miner_dispatch=False,
                      registration_policy='all_activated_subnet')
        status = dict(active=None, round=0, training_steps=1, checkpoint=checkpoint,
                      checkpoint_path='/fixture/checkpoint', initial_published=True)
        (state/'controller.json').write_text(json.dumps(status))

        def opening(epoch, checkpoint, identities, **kwargs):
            openings.append(dict(epoch=epoch, identities=dict(identities)))
            raise ReachedOpening()

        stack.enter_context(patch('subnet.gpu_service.Bucket'))
        stack.enter_context(patch('subnet.gpu_service.Gateway', return_value=SimpleNamespace(epochs={})))
        stack.enter_context(patch('subnet.gpu_service.RemoteController', return_value=SimpleNamespace(open=opening)))
        stack.enter_context(patch('subnet.gpu_service.ChainAdapter', return_value=chain))
        stack.enter_context(patch('subnet.gpu_service.contract', return_value={}))
        stack.enter_context(patch('subnet.gpu_service.log.exception'))
        return state, config

    def test_real_run_discovers_new_participant_at_next_opening(self):
        # Exercise run's actual saved registration snapshots. Stop at opening;
        # these pure fixtures run no model, storage API or blockchain operation.
        chain = SimpleNamespace(registrations=Mock(side_effect=[self.first, self.second]))
        openings = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            state, config = self.fixtures(directory, chain, openings, stack)
            with self.assertRaises(ReachedOpening):
                run(config, once=True)
            original = json.loads((state/(openings[0]['epoch']+'-registrations.json')).read_text())
            self.assertEqual(openings[0]['identities'], {'owned-public': 'owned'})
            status = json.loads((state/'controller.json').read_text())
            status.update(active=None, round=1)
            (state/'controller.json').write_text(json.dumps(status))
            with self.assertRaises(ReachedOpening):
                run(config, once=True)
            self.assertEqual(openings[1]['identities'],
                             {'owned-public': 'owned', 'external-public': 'external'})
            self.assertEqual(chain.registrations.call_count, 2)
            self.assertEqual(json.loads((state/(openings[0]['epoch']+'-registrations.json')).read_text()), original)

    def test_resuming_existing_epoch_keeps_its_saved_participants(self):
        chain = SimpleNamespace(registrations=Mock(side_effect=AssertionError('unexpected fresh snapshot')))
        openings = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            state, config = self.fixtures(directory, chain, openings, stack)
            status = json.loads((state/'controller.json').read_text())
            status['active'] = dict(epoch='nonpayable-existing-fixture', registrations=self.first,
                                    identities={'owned-public': 'owned'}, phase='opening')
            (state/'controller.json').write_text(json.dumps(status))
            with self.assertRaises(ReachedOpening):
                run(config, once=True)
            self.assertEqual(openings[0]['identities'], {'owned-public': 'owned'})
            chain.registrations.assert_not_called()


if __name__ == '__main__':
    unittest.main()
