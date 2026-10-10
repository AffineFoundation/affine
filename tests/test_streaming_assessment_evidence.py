"""Streaming queue digest parity with original signed evidence and temporary SQLite."""
import copy
import hashlib
import random
import sqlite3
import unittest
from unittest.mock import patch

from ops import current_assessment_evidence as evidence
from subnet.continuous_audit_policy import digest
from subnet.storage import canonical
import test_current_assessment_evidence as fixtures


class DigestControls(unittest.TestCase):
    def test_nested_queue_bytes_match_original_canonical_encoding(self):
        rows = {'job-z': {'report': {'text': 'é\n"\\\ud800', 'values': [None, True, False]},
                          'created': -0.0, 'finished': 1e-12},
                'job-a': {'unused': 2**100, 'nested': [{'b': 2, 'a': 1}]}}
        self.assertEqual(b''.join(evidence.canonical_fragments(rows)), canonical(rows))
        self.assertEqual(evidence.queue_view_digest(rows, enabled=True), digest(rows))
        self.assertEqual(evidence.queue_view_digest(rows), digest(rows))

    def test_seeded_queue_variations_match_canonical_digest(self):
        rng = random.Random(493)
        for _ in range(100):
            rows = {str(job): {'report': [rng.randrange(-10000, 10000), rng.random(),
                                         rng.choice([None, True, False, 'Ω', '\n', '\\'])],
                               'status': rng.choice(['queued', 'complete', 'failed'])}
                    for job in rng.sample(range(30), rng.randrange(20))}
            self.assertEqual(evidence.queue_view_digest(rows, enabled=True), digest(rows))

    def test_invalid_values_raise_without_returning_a_digest(self):
        cycle = []; cycle.append(cycle)
        for value in (float('nan'), float('inf'), -float('inf'), object(), cycle):
            rows = {'a-valid': {'report': 'already serialized'}, 'z-invalid': value}
            errors = []
            for function in (digest, lambda rows: evidence.queue_view_digest(rows, enabled=True)):
                with self.subTest(value=type(value).__name__):
                    try:
                        function(rows)
                    except (ValueError, TypeError) as error:
                        errors.append((type(error), str(error)))
                    else:
                        self.fail('invalid complete input must not produce a digest')
            self.assertEqual(errors[0], errors[1])

    def test_late_error_never_finalizes_partial_hash(self):
        original = hashlib.sha256
        updates, finalized = [], []
        class Hash:
            def __init__(self): self.inner = original()
            def update(self, fragment): updates.append(fragment); self.inner.update(fragment)
            def hexdigest(self): finalized.append(True); return self.inner.hexdigest()
        with patch.object(evidence.hashlib, 'sha256', Hash):
            with self.assertRaises(ValueError):
                evidence.queue_view_digest({'a': 'valid prefix', 'z': float('nan')}, enabled=True)
        self.assertTrue(updates)
        self.assertEqual(finalized, [])


class ReaderControls(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.EvidenceControls()
        self.fx.setUp(); self.addCleanup(self.fx.doCleanups)

    def equivalent(self):
        paths = [self.fx.cfg, self.fx.directory/'audit-state.json',
                 self.fx.root/'roles/verifier-queue.sqlite3']
        before = [path.read_bytes() for path in paths]
        with patch.object(evidence, 'queue_view_digest', side_effect=lambda rows, **_: digest(rows)):
            original = self.fx.load()
        streamed = self.fx.load()
        self.assertEqual(canonical(original), canonical(streamed))
        self.assertEqual(before, [path.read_bytes() for path in paths])
        return streamed

    def test_full_signed_evidence_result_and_original_files_are_identical(self):
        result = self.equivalent()
        self.assertEqual(result['refused'], [])
        self.assertEqual(len(result['snapshots']), 1)

    def test_all_outcome_labels_remain_identical(self):
        for outcome in ('verified_valid', 'confirmed_invalid', 'numerical_ambiguous', 'infrastructure_error'):
            self.fx.queue = self.fx.make_queue(outcome); self.fx.save()
            with self.subTest(outcome=outcome): self.equivalent()

    def test_cutoff_pending_missing_and_bad_signature_order_is_unchanged(self):
        fx = self.fx
        bad = copy.deepcopy(fx.pop); bad['signature'] = 'A'*88
        fx.state['populations']['broken'] = bad
        late = fx.make_queue('confirmed_invalid', 'job-late', 31)
        pending = fx.make_queue('verified_valid', 'job-pending', 21); pending['status'] = 'queued'
        for name in ('job-late', 'job-pending', 'job-missing'):
            fx.state['jobs'][name] = dict(fx.state['jobs']['job-1'])
        fx.save([fx.queue, late, pending])
        result = self.equivalent()
        self.assertEqual([row['identifier'] for row in result['excluded']], ['job-late', 'job-pending'])
        self.assertEqual([row['identifier'] for row in result['refused']], ['broken', 'job-missing'])

    def test_single_complete_queue_read_is_hashed_after_sqlite_read_release(self):
        original_read, original_hash = evidence.queue_rows, evidence.queue_view_digest
        reads, hashes = [], []
        def read(*args, **kwargs):
            self.assertEqual(kwargs, {'complete': True})
            result = original_read(*args, **kwargs); reads.append(result); return result
        def hashed(rows, *, enabled=False):
            self.assertTrue(enabled); self.assertIs(rows, reads[0]); hashes.append(rows)
            # A new writer lock can be acquired: the snapshot read is already closed.
            db = sqlite3.connect(self.fx.root/'roles/verifier-queue.sqlite3', timeout=.1, isolation_level=None)
            try: db.execute('BEGIN IMMEDIATE'); db.rollback()
            finally: db.close()
            return original_hash(rows, enabled=enabled)
        with patch.object(evidence, 'queue_rows', side_effect=read), \
             patch.object(evidence, 'queue_view_digest', side_effect=hashed):
            result = self.fx.load()
        self.assertEqual(len(reads), 1); self.assertEqual(len(hashes), 1)
        self.assertEqual(result['evidence_hashes']['original_queue_view_sha256'], digest(reads[0]))

    def test_columns_unused_by_admission_still_change_queue_hash(self):
        before = self.equivalent()
        db = sqlite3.connect(self.fx.root/'roles/verifier-queue.sqlite3')
        try:
            db.execute('ALTER TABLE jobs ADD COLUMN original_extra TEXT')
            db.execute('UPDATE jobs SET original_extra=?', ('Ω\n"\\',))
            db.commit()
        finally: db.close()
        after = self.equivalent()
        key = 'original_queue_view_sha256'
        self.assertNotEqual(before['evidence_hashes'][key], after['evidence_hashes'][key])
        before['evidence_hashes'][key] = after['evidence_hashes'][key]
        self.assertEqual(canonical(before), canonical(after))

    def test_late_serialization_failure_never_returns_partial_evidence(self):
        original_read = evidence.queue_rows
        def read(*args, **kwargs):
            rows = original_read(*args, **kwargs); rows['z-invalid'] = {'payload': float('nan')}
            return rows
        errors = []
        for streamed in (False, True):
            with patch.object(evidence, 'queue_rows', side_effect=read):
                try:
                    if streamed:
                        self.fx.load()
                    else:
                        with patch.object(evidence, 'queue_view_digest', side_effect=lambda rows, **_: digest(rows)):
                            self.fx.load()
                except ValueError as error:
                    errors.append((type(error), str(error)))
                else:
                    self.fail('complete evidence cannot contain a partial queue hash')
        self.assertEqual(errors[0], errors[1])


if __name__ == '__main__': unittest.main()
