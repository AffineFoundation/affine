"""Portable signed-cache controls for the hourly writer's no-eviction wrapper."""
import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nacl.signing import SigningKey
from ops import current_assessment_admission_cache as admission
from ops import current_assessment_cache as bridge
from ops import current_assessment_cache_storage as storage
from ops import no_eviction_cache as candidate
import test_current_assessment_admission_cache as fixture


class NoEvictionControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'cache'
        self.path.mkdir(mode=0o700)
        self.x, self.q, self.records, self.context = fixture.fixture()
        self.key = SigningKey.generate()
        policy = fixture.make_policy(self.path, self.x.authority, self.key, self.x.root)['payload']
        policy['maximum_entries'] = 3
        self.policy = fixture.signed(self.x.authority, policy)

    def open(self, cls=storage.RetainingAdmissionCache, policy=None):
        handle = cls(self.policy if policy is None else policy, authority=self.x.root,
                     context=self.context, signing_key=self.key)
        self.addCleanup(handle.close)
        return handle

    def admit(self, handle, nonce, queue=None):
        queue = dict(self.q, attempt=nonce) if queue is None else queue
        expected = fixture.output(lambda: fixture.baseline(
            queue, self.records, self.context, self.x.root, 30))
        actual = fixture.output(lambda: handle.admit(queue, self.records, cutoff=30))
        self.assertEqual(actual, expected)
        return actual

    def fill(self):
        with self.open(admission.AdmissionCache) as handle:
            for nonce in range(3):
                self.admit(handle, nonce)
        self.assertEqual(len(list(self.path.glob('*.json'))), 3)

    def bytes(self):
        return {path.name: path.read_bytes() for path in self.path.glob('*.json')}

    def test_full_cache_preserves_unread_entries_and_scientific_results(self):
        self.fill()
        before = self.bytes()
        handle = self.open()
        with candidate.protected(handle):
            self.admit(handle, 10)
            for nonce in range(3):
                self.admit(handle, nonce)
            self.assertEqual(handle.stats['misses'], 1)
            self.assertEqual(handle.stats['hits'], 3)
            self.assertEqual(handle.stats['writes'], 0)
            self.assertEqual(handle.storage_stats['evictions'], 0)
            self.assertEqual(handle.storage_stats['inventories'], 1)
        self.assertEqual(before, self.bytes())
        self.assertNotIn('_retire_one', handle.__dict__)

    def test_original_churn_reproduces(self):
        self.fill()
        with self.open() as handle:
            for nonce in (10, 0, 1, 2):
                self.admit(handle, nonce)
            self.assertEqual(handle.stats['misses'], 4)
            self.assertEqual(handle.stats['hits'], 0)
            self.assertEqual(handle.storage_stats['evictions'], 3)

    def test_not_full_writes_keep_original_signed_cache_and_restart_hit(self):
        with candidate.protected(self.open()) as handle:
            self.admit(handle, 0)
            self.admit(handle, 0)
            self.assertEqual(handle.stats['writes'], 1)
            self.assertEqual(handle.stats['hits'], 1)
        with candidate.protected(self.open()) as cold:
            self.admit(cold, 0)
            self.assertEqual(cold.stats['hits'], 1)

    def test_tampered_signed_entry_falls_back_and_is_not_repaired(self):
        self.fill()
        path = next(self.path.glob('*.json'))
        document = json.loads(path.read_bytes())
        document['signature'] = 'A' * 88
        path.write_bytes(admission.canonical(document))
        before = path.read_bytes()
        with candidate.protected(self.open()) as handle:
            for nonce in range(3):
                self.admit(handle, nonce)
            self.assertEqual(handle.stats['rejected_entries'], 1)
            self.assertEqual(handle.stats['baseline_calls'], 1)
        self.assertEqual(before, path.read_bytes())

    def test_invalid_report_preserves_original_rejection_and_no_credit(self):
        self.fill()
        with candidate.protected(self.open()) as handle:
            result = self.admit(handle, 0, queue=dict(self.q, token='substitution'))
            self.assertEqual(result[0], 'error')
            self.assertEqual(handle.stats['writes'], 0)

    def test_timeout_propagates_and_instance_method_restored(self):
        handle = self.open()
        with self.assertRaises(TimeoutError):
            with candidate.protected(handle), patch.object(
                    admission, 'admit_completed_reports', side_effect=TimeoutError('deadline')):
                handle.admit(dict(self.q, attempt=90), self.records, cutoff=30)
        self.assertNotIn('_retire_one', handle.__dict__)
        self.assertIsNone(handle._held_lock)

    def test_missing_optional_cache_passthrough(self):
        with candidate.protected(nullcontext(None)) as handle:
            self.assertIsNone(handle)

    def test_all_original_scientific_outcomes_preserved(self):
        with candidate.protected(self.open()) as handle:
            for outcome in ('verified_valid', 'confirmed_invalid', 'structural_invalid',
                            'numerical_ambiguous', 'infrastructure_error'):
                with self.subTest(outcome=outcome):
                    queue = copy.deepcopy(self.q)
                    def mutate(report):
                        report['audits'][0]['outcomes'][0].update(
                            fully_audited=outcome != 'structural_invalid',
                            valid=True if outcome == 'verified_valid' else
                            False if outcome in ('confirmed_invalid', 'structural_invalid') else None,
                            failure_kind=outcome)
                    fixture.resign_report(self.x, queue, mutate)
                    self.assertEqual(self.admit(handle, 0, queue=queue)[0], 'return')

    def test_actual_factory_and_loader_preserve_writes_hits_and_fallback(self):
        policy_path = Path(self.temp.name) / 'policy'
        policy_path.write_bytes(admission.canonical(self.policy))
        policy_path.chmod(0o600)
        seed = Path(self.temp.name) / 'seed'
        seed.write_text(self.key.encode().hex())
        seed.chmod(0o600)
        scope = dict(version=bridge.VERSION, policy_path=str(policy_path),
                     policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                     seed_path=str(seed))
        statistics = {}
        original_factory = bridge.factory(scope, self.x.root)
        def loader(path, *, admission_cache_factory, extra):
            self.assertEqual(path, 'unchanged-path')
            self.assertEqual(extra, 37)
            with admission_cache_factory(authority=self.x.root, context=self.context) as handle:
                if handle is None:
                    return 'original-no-cache'
                self.admit(handle, 0)
                self.admit(handle, 0)
                return 'same-output'
        wrapped = candidate.wrap_loader(loader, statistics=statistics)
        def invoke():
            return wrapped('unchanged-path', admission_cache_factory=original_factory, extra=37)
        self.assertEqual(invoke(), 'same-output')
        self.assertEqual(statistics['stats']['writes'], 1)
        self.assertEqual(statistics['stats']['hits'], 1)
        seed.unlink()
        self.assertEqual(invoke(), 'same-output')
        self.assertEqual(statistics['stats']['hits'], 2)
        policy_path.write_text('{}')
        self.assertEqual(invoke(), 'original-no-cache')

    def test_refuses_unpinned_factory(self):
        with self.assertRaises(ValueError):
            candidate.wrap_factory(lambda **kwargs: nullcontext(None))

    def test_instance_only_restore_leaves_class_and_other_manager_unchanged(self):
        cls = storage.RetainingAdmissionCache
        original = cls._retire_one
        first, second = self.open(), self.open()
        with candidate.protected(first):
            self.assertIs(cls._retire_one, original)
            self.assertIs(second._retire_one.__func__, original)
            self.assertIsNot(first._retire_one.__func__, original)
        self.assertIs(first._retire_one.__func__, original)
        self.assertIs(cls._retire_one, original)
        self.assertIsNone(first._held_lock)

    def test_source_pin_refusal_closes_manager(self):
        handle = self.open()
        with patch.object(candidate, 'STORAGE_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                with candidate.protected(handle):
                    self.fail('changed source must not be admitted')
        self.assertNotIn('_retire_one', handle.__dict__)
        self.assertIsNone(handle._held_lock)

    def test_free_space_floor_unchanged(self):
        with candidate.protected(self.open()) as handle, patch.object(
                admission.os, 'fstatvfs', return_value=SimpleNamespace(f_bavail=0, f_frsize=4096)):
            self.admit(handle, 0)
            self.assertEqual(handle.stats['writes'], 0)
            self.assertEqual(handle.stats['baseline_calls'], 1)
            self.assertEqual(handle.storage_stats['evictions'], 0)

    def test_byte_quota_preserves_full_cache(self):
        # Signed policy plus an ordinary owned file reaches the byte quota while
        # leaving room under the entry quota, exercising the other capacity guard.
        payload = copy.deepcopy(self.policy['payload'])
        payload.update(maximum_entries=100, maximum_directory_bytes=1024 * 1024)
        self.policy = fixture.signed(self.x.authority, payload)
        self.fill()
        occupied = self.path / 'occupied'
        occupied.write_bytes(b'x' * (1024 * 1024))
        occupied.chmod(0o600)
        before = self.bytes()
        with candidate.protected(self.open()) as handle:
            self.admit(handle, 9)
            for nonce in range(3):
                self.admit(handle, nonce)
            self.assertEqual(handle.stats['writes'], 0)
            self.assertEqual(handle.stats['hits'], 3)
            self.assertEqual(handle.storage_stats['evictions'], 0)
        self.assertEqual(before, self.bytes())
        self.assertEqual(occupied.stat().st_size, 1024 * 1024)


if __name__ == '__main__':
    unittest.main(verbosity=2)
