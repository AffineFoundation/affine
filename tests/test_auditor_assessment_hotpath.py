import contextlib
from types import SimpleNamespace
import unittest
from ops.auditor_assessment_hotpath_service import install


class AuditorScopeControls(unittest.TestCase):
    def test_guards_results_and_exception_scope_preserved(self):
        state = {'depth': 0, 'calls': 0, 'scopes': 0}
        @contextlib.contextmanager
        def scope():
            state['depth'] += 1
            state['scopes'] += 1
            try:
                yield
            finally:
                state['depth'] -= 1
        class Auditor:
            def hourly_snapshot(self, value, *, fail=False):
                if state['depth'] != 1:
                    raise AssertionError('missing scoped authenticated memo')
                state['calls'] += 1
                if fail:
                    raise ValueError('present report signature invalid')
                return value
        Auditor.hourly_snapshot._acknowledged_missing_originals_v1 = True
        service = SimpleNamespace(ContinuousAuditor=Auditor)
        old_policy = SimpleNamespace(valid_digest=lambda value: False)
        new_policy = SimpleNamespace(valid_digest=lambda value: value == 'a' * 64)
        install(service, old_policy, new_policy, SimpleNamespace(authenticated_reference_cache=scope))
        value = {'signed_original': 1}
        self.assertIs(Auditor().hourly_snapshot(value), value)
        with self.assertRaisesRegex(ValueError, 'present report signature invalid'):
            Auditor().hourly_snapshot(value, fail=True)
        self.assertEqual(state, {'depth': 0, 'calls': 2, 'scopes': 2})
        self.assertIs(old_policy.valid_digest, new_policy.valid_digest)
        with self.assertRaisesRegex(ValueError, 'already installed'):
            install(service, old_policy, new_policy, SimpleNamespace(authenticated_reference_cache=scope))

    def test_missing_original_guard_required(self):
        class Auditor:
            def hourly_snapshot(self):
                return None
        with self.assertRaisesRegex(ValueError, 'missing-evidence guard'):
            install(SimpleNamespace(ContinuousAuditor=Auditor), None, None, None)

if __name__ == '__main__':
    unittest.main()
