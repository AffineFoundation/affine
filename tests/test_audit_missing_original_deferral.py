import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nacl.signing import SigningKey
from ops.audit_missing_original_deferral import Registry, ScopedDeferrals, VERSION, digest, install
from subnet.distributed_roles import authenticate
from subnet import continuous_audit_service as service
from subnet import numerical_resolution
from test_continuous_audit_policy import signed
from test_continuous_audit_service import ServiceControls


class MissingOriginalControls(unittest.TestCase):
    def setUp(self):
        self.fixture = ServiceControls(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.subject = self.fixture.service
        self.root = self.fixture.root
        self.key = self.fixture.key
        self.identifier = 'continuous-audit-original'
        self.job_sha = '7' * 64
        row_id = digest(self.fixture.row)
        self.subject.state['draws'] = {row_id: dict(row=self.fixture.row)}
        self.subject.state['jobs'] = {self.identifier: dict(row_sha256=row_id, job_sha256=self.job_sha)}
        p = dict(self.fixture.p, eligible_evidence_ids=[row_id])
        self.subject.state['populations']['e1'] = signed(self.key, p)
        self.previous = signed(self.key, dict(evidence_refusals=[dict(kind='job', identifier=self.identifier,
                                                                     reason='original queue job absent')]))
        self.value = dict(version=VERSION, created_at=100, previous_assessment_sha256=digest(self.previous),
                          jobs={self.identifier: dict(job_sha256=self.job_sha,
                                                     original_file_sha256='8' * 64, row_sha256s=[row_id])},
                          original_signatures_verified=True, score_evidence=False, penalty_evidence=False)
        self.original_apply = numerical_resolution.apply
        self.addCleanup(setattr, numerical_resolution, 'apply', self.original_apply)

    def registry(self, value=None, previous=None):
        return Registry(signed(self.key, self.value if value is None else value),
                        self.previous if previous is None else previous,
                        self.root, authenticate, Path(self.fixture.directory.name))

    def install(self, actuals=None, admission=None, reviewed_apply=None):
        class Subject(type(self.subject)):
            pass
        self.subject.__class__ = Subject
        self.subject.publish_immutable = lambda *args: None
        # Original snapshot globals are captured while these exact IO boundaries
        # are replaced; report admission itself remains a separate strict call.
        with patch.object(service, 'queue_rows', return_value=actuals or {}), \
                patch.object(service, 'admit_completed_reports', side_effect=admission or (lambda *a, **k: ({}, []))):
            return install(SimpleNamespace(ContinuousAuditor=Subject), self.registry(), numerical_resolution,
                           reviewed_apply or self.original_apply)

    def snapshot(self):
        return self.subject.hourly_snapshot('e1', 1, 'a' * 64, 3600)['payload']

    def test_acknowledged_absence_is_explicit_and_does_not_invent_audit_evidence(self):
        self.install()
        result = self.snapshot()
        unavailable = result['unavailable_original_evidence']
        self.assertEqual(unavailable['unavailable_job_count'], 1)
        self.assertFalse(unavailable['score_evidence'])
        self.assertFalse(unavailable['penalty_evidence'])
        self.assertFalse(unavailable['current_original_files_authenticated'])
        self.assertEqual(unavailable['jobs'][0]['job_sha256'], self.job_sha)
        self.assertEqual(result.get('numerical_resolution_observations', []), [])

    def test_immutable_snapshot_restart_reuses_original_document(self):
        self.install()
        first = self.snapshot()
        self.assertEqual(first, self.snapshot())

    def test_new_unknown_missing_job_remains_fail_closed(self):
        original = self.subject.state['jobs'].pop(self.identifier)
        self.subject.state['jobs']['new-epoch-88-job'] = original
        self.install()
        with self.assertRaisesRegex(ValueError, 'unacknowledged missing'):
            self.snapshot()

    def test_acknowledged_job_cannot_change_journal_digest(self):
        self.subject.state['jobs'][self.identifier]['job_sha256'] = '9' * 64
        self.install()
        with self.assertRaisesRegex(ValueError, 'journal binding'):
            self.snapshot()

    def test_acknowledged_job_cannot_change_selected_rows(self):
        self.value['jobs'][self.identifier]['row_sha256s'] = ['9' * 64]
        self.install()
        with self.assertRaisesRegex(ValueError, 'journal binding'):
            self.snapshot()

    def test_present_tampered_report_still_reaches_strict_admission(self):
        actual = dict(status='complete', report=dict(completed_at=30), marker='tampered-signature')
        def strict(rows, *args, **kwargs):
            self.assertEqual(rows, [actual])
            raise ValueError('original report signature mismatch')
        self.install({self.identifier: actual}, strict)
        with self.assertRaisesRegex(ValueError, 'original report signature mismatch'):
            self.snapshot()

    def test_present_known_original_is_not_silently_quarantined(self):
        actual = dict(status='complete', report=dict(completed_at=30))
        calls = []
        def admit(rows, *args, **kwargs):
            calls.append(rows)
            return {}, []
        self.install({self.identifier: actual}, admit)
        result = self.snapshot()
        self.assertEqual(calls, [[actual]])
        self.assertEqual(result['unavailable_original_evidence']['unavailable_job_count'], 0)

    def test_present_current_epoch_job_is_unaffected(self):
        new = self.subject.state['jobs'].pop(self.identifier)
        self.subject.state['jobs']['fresh-epoch-88'] = new
        actual = dict(status='complete', report=dict(completed_at=30))
        self.install({'fresh-epoch-88': actual})
        self.assertEqual(self.snapshot()['unavailable_original_evidence']['unavailable_job_count'], 0)

    def test_previous_signed_inventory_cannot_expand(self):
        value = copy.deepcopy(self.value)
        value['jobs']['new'] = value['jobs'][self.identifier]
        with self.assertRaisesRegex(ValueError, 'previously acknowledged'):
            self.registry(value)

    def test_wrong_assessment_signature_rejected(self):
        other = SigningKey.generate()
        previous = signed(other, self.previous['payload'])
        with self.assertRaises(ValueError):
            self.registry(previous=previous)

    def test_score_or_penalty_permission_refused(self):
        for field in ('score_evidence', 'penalty_evidence'):
            value = copy.deepcopy(self.value)
            value[field] = True
            with self.assertRaisesRegex(ValueError, 'exact ROOT'):
                self.registry(value)

    def test_numeric_resolver_can_only_defer_exact_missing_original(self):
        values = ScopedDeferrals([dict(job_sha256=self.job_sha)])
        issue = dict(kind='numerical_resolution', status='unresolved', original_job_sha256=self.job_sha)
        values.append(issue)
        self.assertEqual(values, [issue])
        with self.assertRaisesRegex(ValueError, 'acknowledged absent'):
            values.append(dict(issue, original_job_sha256='0' * 64))

    def test_numeric_resolution_outside_snapshot_retains_strict_behavior(self):
        self.install()
        # The old strict call is preserved; no global missing-admission bypass.
        with self.assertRaises(ValueError):
            numerical_resolution.apply([], [], {}, authority=self.root, cutoff=3600,
                                       policy_document={'bogus': True}, expected_policy_sha256='0' * 64)

    def test_scoped_numeric_deferral_is_in_signed_snapshot(self):
        def reviewed(observations, *args, unavailable_execution_deferrals, **kwargs):
            unavailable_execution_deferrals.append(dict(kind='numerical_resolution', status='unresolved',
                                                       original_job_sha256=self.job_sha))
            return observations
        self.install(reviewed_apply=reviewed)
        result = self.snapshot()['unavailable_original_evidence']
        self.assertEqual(result['numerical_resolutions'][0]['original_job_sha256'], self.job_sha)

    def test_unrelated_numeric_missing_admission_still_rejected(self):
        def reviewed(observations, *args, unavailable_execution_deferrals, **kwargs):
            unavailable_execution_deferrals.append(dict(kind='numerical_resolution', status='unresolved',
                                                       original_job_sha256='0' * 64))
            return observations
        self.install(reviewed_apply=reviewed)
        with self.assertRaisesRegex(ValueError, 'acknowledged absent'):
            self.snapshot()


if __name__ == '__main__':
    unittest.main()
