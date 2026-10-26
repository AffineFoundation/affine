"""CPU-only controls; load only pinned pure legacy function ASTs, never models."""
import ast
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
import unittest
from contextlib import contextmanager
from research.group_contract_manifest import COMPLETE_GROUP_SAMPLING_VERSION, GROUP_VERSION, REQUIRED_CLIENT_FEATURES, CompleteGroupContract, LegacyQuotaContract, ValidationHooks, for_manifest
HERE = Path(__file__).parent
FIXTURES = HERE / 'fixtures'
ROOT = HERE.parents[2]

@contextmanager
def _pinned_hooks():
    """Execute exact reviewed legacy functions without importing GPU runtimes.

    This is a local composed-source compatibility test, not a substitute legacy
    implementation. Archived sources are required; missing inputs fail clearly.
    """
    provenance = json.loads((FIXTURES / 'provenance.json').read_text())
    source = ROOT / provenance['legacy_reference_root']
    package = '_manifest_test_pinned_legacy'
    modules = {}

    def load(name, functions, constants, extra=None):
        path = source / (name + '.py')
        assert path.is_file(), f'Pinned compatibility source required: {path}'
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == provenance['legacy_source_sha256'][path.name]
        parsed = ast.parse(raw, filename=str(path))
        selected = []
        for node in parsed.body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                selected.append(node)
            elif isinstance(node, ast.Assign) and any((isinstance(t, ast.Name) and t.id in constants for t in node.targets)):
                selected.append(node)
        assert {n.name for n in selected if isinstance(n, ast.FunctionDef)} == set(functions)
        module = ModuleType(package + '.' + name)
        module.__package__ = package
        module.__dict__.update(extra or {})
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), module.__dict__)
        modules[module.__name__] = module
        return module
    fast = load('fast_prefill_audit', ['calibration'], ['VERSION', 'SUPPORT_VERSION', 'CALIBRATION'], {'math': math})
    forced = load('forced_sampling', ['validate'], ['VERSION', 'MINER_VERSION', 'FIELDS'])
    quota = load('batch_quotas', ['configured_quotas'], ['MAX_SAMPLES_PER_BATCH'])
    controller = load('controller', ['class_quotas'], [])
    previous = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    try:
        yield ValidationHooks(quota.configured_quotas, controller.class_quotas, forced.validate, fast.calibration)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

def legacy():
    return json.loads((FIXTURES / 'legacy_epoch171.json').read_text())

def complete(n=8):
    m = legacy()
    del m['K'], m['L']
    m['group_contract'] = dict(version=GROUP_VERSION, group_size=n, min_success=1, min_completed_failure=1, outcome_policy='completed-eos-math-unresolved-masked-v1')
    m['sampling_contract']['version'] = COMPLETE_GROUP_SAMPLING_VERSION
    del m['sampling_contract']['max_attempts']
    m['required_client_features'] = sorted(REQUIRED_CLIENT_FEATURES)
    return m

class ManifestResolverTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._hook_context = _pinned_hooks()
        cls.hooks = cls._hook_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._hook_context.__exit__(None, None, None)

    def test_current_legacy_signed_wire_unchanged(self):
        m = legacy()
        before = deepcopy(m)
        result = for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)
        assert isinstance(result, LegacyQuotaContract)
        assert (result.success_quota, result.failure_quota, result.required_rollouts, result.max_batches, result.attempt_count) == (4, 4, 8, 9, 1000)
        assert result.allowed_nonces == range(1000)
        assert 'samples_per_batch' not in m and m == before

    def test_one_size_derives_exact_nonces_without_balancing_quota(self):
        for n in [4, 8, 16]:
            with self.subTest(n=n):
                m = complete(n)
                before = deepcopy(m)
                c = for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)
                assert isinstance(c, CompleteGroupContract)
                assert (c.group_size, c.required_rollouts, c.min_success, c.min_completed_failure) == (n, n, 1, 1)
                assert list(c.required_nonces) == list(range(n)) and c.max_batches == 9
                assert m == before

    def test_outcome_minima_are_independent(self):
        m = complete(8)
        m['group_contract'].update(min_success=2, min_completed_failure=3)
        c = for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)
        assert (c.min_success, c.min_completed_failure, c.group_size) == (2, 3, 8)

    def test_bad_group_types_bounds_and_versions(self):
        for field, value in [('group_size', True), ('group_size', 8.0), ('group_size', '8'), ('group_size', None), ('group_size', 1), ('group_size', 129), ('min_success', False), ('min_success', 0), ('min_completed_failure', 0), ('min_completed_failure', 1.0), ('min_success', 8), ('version', 'complete-canonical-group-v999'), ('outcome_policy', 'all-zero-is-wrong')]:
            with self.subTest(field=field, value=value):
                m = complete()
                m['group_contract'][field] = value
                with self.assertRaises(ValueError):
                    for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)

    def test_even_agreeing_extra_bounds_and_legacy_fields_rejected(self):
        for scope, field, value in [('root', 'K', 4), ('root', 'L', 4), ('root', 'samples_per_batch', 8), ('root', 'max_attempts', 8), ('root', 'nonce_count', 8), ('root', 'nonce_range', [0, 7]), ('root', 'group_size', 8), ('group', 'max_attempts', 8), ('group', 'nonce_count', 8), ('sampling', 'max_attempts', 8), ('sampling', 'max_attempts', 1000), ('sampling', 'group_size', 8), ('sampling', 'nonce_start', 0)]:
            with self.subTest(scope=scope, field=field, value=value):
                m = complete()
                target = m if scope == 'root' else m['group_contract' if scope == 'group' else 'sampling_contract']
                target[field] = value
                with self.assertRaises(ValueError):
                    for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)

    def test_group_sampler_version_must_match(self):
        for version in ['unknown-v999', 'forced-inverse-cdf-prefill-miner-bound-v5', None, True]:
            with self.subTest(version=version):
                m = complete()
                m['sampling_contract']['version'] = version
                with self.assertRaises(ValueError):
                    for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)

    def test_new_sampler_without_group_and_partial_group_fail(self):
        for mutate in (lambda m: m.pop('group_contract'), lambda m: m['group_contract'].pop('min_success'), lambda m: m.__setitem__('group_contract', None)):
            m = complete()
            mutate(m)
            with self.assertRaises(ValueError):
                for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)

    def test_features_required_with_exact_types(self):
        for features in [None, 'complete-canonical-groups-v1', [], [True], ['x', 'x']]:
            with self.subTest(features=features):
                m = complete()
                m['required_client_features'] = features
                with self.assertRaises(ValueError):
                    for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)

    def test_missing_and_unsupported_client_features(self):
        m = complete()
        m['required_client_features'].pop()
        with self.assertRaisesRegex(ValueError, 'mandatory'):
            for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)
        m = complete()
        m['required_client_features'].append('future-feature-v9')
        with self.assertRaisesRegex(ValueError, 'upgrade'):
            for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)
        with self.assertRaisesRegex(ValueError, 'upgrade'):
            for_manifest(complete(), validators=self.hooks, supported_features=())
        with self.assertRaisesRegex(ValueError, 'upgrade'):
            for_manifest(complete(), validators=self.hooks)
        for malformed in (None, True, 'claimed-features'):
            with self.subTest(supported_features=malformed), self.assertRaises(ValueError):
                for_manifest(complete(), validators=self.hooks, supported_features=malformed)

    def test_non_group_contract_does_not_require_new_features(self):
        assert for_manifest(legacy(), validators=self.hooks).required_rollouts == 8

    def test_new_batch_count_is_independent_but_bounded(self):
        for value in [True, 0, 257, '9', 9.0, None]:
            with self.subTest(value=value):
                m = complete()
                m['max_batches'] = value
                with self.assertRaises(ValueError):
                    for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)

    def test_legacy_optional_consistent_size_projection(self):
        m = legacy()
        m['samples_per_batch'] = 8
        assert for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES).required_rollouts == 8
        m['samples_per_batch'] = 16
        with self.assertRaises(ValueError):
            for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)

    def test_legacy_v5_guards_preserved_and_no_partial_new_mode(self):
        for field, value in [('K', 3), ('K', True), ('L', 1), ('L', 4.0), ('max_batches', 0), ('required_client_features', []), ('group_contract', None)]:
            with self.subTest(field=field, value=value):
                m = legacy()
                m[field] = value
                with self.assertRaises(ValueError):
                    for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)

    def test_legacy_v5_original_validator_rejects_new_sampler(self):
        with self.assertRaises(ValueError):
            self.hooks.validate_sampling(complete()['sampling_contract'])
        m = legacy()
        m['sampling_contract']['max_attempts'] = 8
        with self.assertRaises(ValueError):
            for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)
        m = legacy()
        m['sampling_contract']['version'] = 'unrecognized-legacy-version'
        with self.assertRaises(ValueError):
            for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)

    def test_older_unequal_quotas_still_allowed_by_original_rule(self):
        m = legacy()
        m['K'], m['L'] = (1, 3)
        s = m['sampling_contract']
        s['version'] = 'forced-inverse-cdf-replay-v1'
        s['max_attempts'] = 64
        s['verification'] = 'exact-token-replay'
        s['generation'] = 'uncached-eager-inverse-cdf'
        del s['calibration'], s['support_adjudication']
        r = for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)
        assert (r.success_quota, r.failure_quota, r.attempt_count) == (1, 3, 64)

    def test_legacy_without_forced_sampler(self):
        m = {'epoch': 'synthetic-historical', 'K': 1, 'L': 2}
        r = for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)
        assert (r.success_quota, r.failure_quota, r.attempt_count, r.max_batches) == (1, 2, None, 4)
        m['sampling_source_hash'] = '1' * 64
        with self.assertRaises(ValueError):
            for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)

    def test_no_full_manifest_or_sampling_authentication_claim(self):
        with self.assertRaisesRegex(ValueError, 'payload'):
            for_manifest({'payload': legacy(), 'signature': 'not-authenticated'}, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)
        assert for_manifest(complete(), validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES).kind == 'complete-group'

    def test_new_branch_does_not_invoke_legacy_quota_or_sampler_checks(self):

        def fail(*args):
            raise AssertionError('new branch reached legacy validation')
        scoped = ValidationHooks(fail, fail, fail, self.hooks.validate_calibration)
        assert for_manifest(complete(), validators=scoped, supported_features=REQUIRED_CLIENT_FEATURES).required_rollouts == 8

    def test_preserves_current_calibration_bounds(self):
        m = complete()
        m['sampling_contract']['calibration']['cdf_abs_error'] = 100
        with self.assertRaises(ValueError):
            for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)
        m = complete()
        m['sampling_contract']['randomness'] = 'G' * 64
        with self.assertRaises(ValueError):
            for_manifest(m, validators=self.hooks, supported_features=REQUIRED_CLIENT_FEATURES)

    def test_dispatch_has_no_network_model_or_process_dependencies(self):
        path = HERE.parent / 'resolver.py'
        tree = ast.parse(path.read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert set(imports) <= {'collections.abc', 'copy', 'dataclasses', 'typing'}
        assert not any((isinstance(node, ast.Import) for node in ast.walk(tree)))
if __name__ == '__main__':
    unittest.main()
