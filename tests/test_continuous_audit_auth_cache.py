"""Synthetic original queue admission controls using actual Ed25519 verification.

No original admission is bypassed or mocked. These bounded fixtures are not live
reports and establish neither scientific GPU validity nor production speed.
"""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nacl.signing import SigningKey
from subnet import continuous_audit_policy as candidate

# Load the same public module in a separate namespace, then replace only its
# admission function with the frozen pre-optimization function. All real
# cryptographic, cohort and native-evidence checks remain in both arms.
spec = importlib.util.spec_from_file_location('subnet._uncached_admission_baseline',
    ROOT / 'subnet/continuous_audit_policy.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
baseline_path = ROOT / 'tests/fixtures/continuous_audit_admission_uncached.py'
exec(compile(baseline_path.read_text(), str(baseline_path), 'exec'), vars(original))

def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def signed(key, payload):
    return dict(signer=key.verify_key.encode().hex(), payload=copy.deepcopy(payload),
        signature=base64.b64encode(key.sign(original.canonical(payload)).signature).decode())


class Fixture:
    def __init__(self, count=4):
        self.root = SigningKey(bytes([19]) * 32)
        self.worker = SigningKey(bytes([27]) * 32)
        self.authority = self.root.verify_key.encode().hex()
        self.identity = self.worker.verify_key.encode().hex()
        self.workers = {self.identity: ['verify']}
        self.manifest = dict(epoch='synthetic-e1', checkpoint={'id': sha('checkpoint')},
            sampling_contract={'version': 'synthetic'}, sampling_source_hash=sha('sampler'),
            model_runtime_revision='synthetic-runtime', backend_profile={'version': 'synthetic'},
            numerical_policy={'version': 'synthetic'}, source_bundle={'sha256': sha('source')})
        self.rows = [dict(epoch=self.manifest['epoch'], round=1, checkpoint=self.manifest['checkpoint']['id'],
            miner=sha('miner' + str(i)), env_id='math', index=i, batch_sha256=sha('batch' + str(i)),
            proof_sha256=sha('proof' + str(i)), commitment_sha256=sha('commit' + str(i)),
            verifier_contract_sha256=original.verifier_contract(self.manifest), committed_at=10)
            for i in range(count)]
        self.job = dict(role='verify', job_id='synthetic-job', manifest=signed(self.root, self.manifest),
            source_files={'model.py': sha('model-source')}, runtime_versions={'torch': 'pinned'},
            submissions=[dict(sha256=r['proof_sha256'], commitment_ref={k: r[k] for k in
                ('miner', 'batch_sha256', 'commitment_sha256')}) for r in self.rows])
        outcomes = [dict(valid=True, fully_audited=True),
            dict(valid=False, fully_audited=True, failure_kind='confirmed_invalid', reason='InvalidSample: TOPLOC'),
            dict(valid=None, failure_kind='numerical_ambiguous'),
            dict(valid=None, failure_kind='infrastructure_error')]
        self.report = dict(success=True, role='verify', job_id='synthetic-job', operator=self.authority,
            epoch=self.manifest['epoch'], checkpoint=self.manifest['checkpoint']['id'],
            source_files=copy.deepcopy(self.job['source_files']), runtime_versions=copy.deepcopy(self.job['runtime_versions']),
            backend_profile=self.manifest['backend_profile'], numerical_policy=self.manifest['numerical_policy'],
            execution_resources_enforced=True, completed_at=20,
            audits=[dict(submission_sha256=r['proof_sha256'], epoch=r['epoch'], outcomes=[outcomes[i % 4]])
                    for i, r in enumerate(self.rows)])
        self.sources = {sha('source'): copy.deepcopy(self.job['source_files'])}

    def queue(self):
        report = dict(copy.deepcopy(self.report), job_sha256=original.digest(self.job))
        request = signed(self.worker, dict(action='report', job_id=self.job['job_id'], token='synthetic-lease', report=report))
        return dict(status='complete', role='verify', worker=self.identity, id=self.job['job_id'],
            digest=original.digest(self.job), envelope=signed(self.root, self.job), report=report,
            report_request=request, report_digest=original.digest(report), token='synthetic-lease')

    def resign_report(self, queue):
        queue['report_digest'] = original.digest(queue['report'])
        queue['report_request'] = signed(self.worker, dict(action='report', job_id=queue['id'],
            token='synthetic-lease', report=queue['report']))


def call(module, fixture, queues, *, defer=False, **kwargs):
    queues, records, workers, sources, kwargs = copy.deepcopy((queues, fixture.rows, fixture.workers, fixture.sources, kwargs))
    trace = []
    authenticate = module.authenticate
    def authenticated(document, identity):
        # Instrumentation delegates every original verification, including failure.
        trace.append((identity, original.digest(document)))
        return authenticate(document, identity)
    module.authenticate = authenticated
    try:
        function = module.admit_queue_reports_with_deferrals if defer else module.admit_queue_reports
        try:
            output = ('success', function(queues, records, fixture.authority, workers, sources, **kwargs))
        except Exception as error:
            # The candidate is imported under a private alias solely to compare
            # both modules in one process; deployed import names are identical.
            origin = type(error).__module__
            if origin == candidate.__name__:
                origin = original.__name__
            output = ('error', origin, type(error).__qualname__, str(error))
    finally:
        module.authenticate = authenticate
    return output, trace


class AuthenticationCacheControls(unittest.TestCase):
    def compare(self, f, queues=None, *, error=None, defer=False, **kwargs):
        queues = [f.queue()] if queues is None else queues
        before = copy.deepcopy((queues, f.rows, f.sources))
        a = call(original, f, queues, defer=defer, **kwargs)
        b = call(candidate, f, queues, defer=defer, **kwargs)
        self.assertEqual(original.canonical(a), original.canonical(b))
        self.assertEqual((queues, f.rows, f.sources), before)
        if error is None:
            self.assertEqual(a[0][0], 'success', a)
        else:
            self.assertEqual(a[0][0], 'error', a)
            self.assertIn(error, a[0][-1])
        return a

    def test_actual_signatures_full_group_outputs(self):
        for n in (1, 4):
            with self.subTest(children=n):
                result, trace = self.compare(Fixture(n))
                self.assertEqual(len(trace), 3)
                admission = next(iter(result[1].values()))
                self.assertEqual(len(admission['observations']), n)
                self.assertEqual(len(admission['native_observations']), n)

    def test_dict_json_and_mixed_encodings(self):
        for fields in (('envelope', 'report', 'report_request'), ('report_request',), ('envelope', 'report')):
            f = Fixture()
            q = f.queue()
            for key in fields:
                q[key] = json.dumps(q[key])
            with self.subTest(fields=fields):
                self.compare(f, [q])

    def test_all_three_signature_tampering_paths(self):
        for target in ('job', 'manifest', 'request'):
            f = Fixture()
            q = f.queue()
            if target == 'job':
                q['envelope']['signature'] = base64.b64encode(bytes(64)).decode()
            elif target == 'manifest':
                f.job['manifest']['signature'] = base64.b64encode(bytes(64)).decode()
                q = f.queue()
            else:
                q['report_request']['signature'] = base64.b64encode(bytes(64)).decode()
            with self.subTest(target=target):
                self.compare(f, [q], error='Signature was forged or corrupt')

    def test_queue_journal_and_lease_tampering(self):
        for key, value, message in [('status', 'claimed', 'actually completed'),
            ('role', 'train', 'actually completed'), ('worker', sha('other'), 'admitted actual worker'),
            ('id', 'other-job', 'original queue request digest'), ('digest', sha('other'), 'original queue request digest'),
            ('report_digest', sha('other'), 'original worker terminal report request'),
            ('token', 'other-token', 'original worker terminal report request')]:
            f = Fixture()
            q = f.queue()
            q[key] = value
            with self.subTest(key=key):
                self.compare(f, [q], error=message)

    def test_original_report_identity_fields(self):
        for key, value in [('success', False), ('role', 'train'), ('job_id', 'other'),
            ('job_sha256', sha('other')), ('operator', sha('other')), ('epoch', 'other'), ('checkpoint', sha('other'))]:
            f = Fixture()
            q = f.queue()
            q['report'][key] = value
            f.resign_report(q)
            with self.subTest(key=key):
                self.compare(f, [q], error='executed audit identity/checkpoint')

    def test_original_request_identity_and_payload(self):
        for key, value in [('action', 'claim'), ('job_id', 'other'), ('token', 'other'), ('report', {})]:
            f = Fixture()
            q = f.queue()
            payload = dict(q['report_request']['payload'], **{key: value})
            q['report_request'] = signed(f.worker, payload)
            with self.subTest(key=key):
                self.compare(f, [q], error='original worker terminal report request')

    def test_original_source_and_runtime_guards(self):
        f = Fixture()
        f.sources = {sha('source'): {'model.py': sha('wrong-source')}}
        self.compare(f, error='admitted executed source pins')
        for field in ('source_files', 'runtime_versions', 'backend_profile', 'numerical_policy'):
            f = Fixture()
            q = f.queue()
            q['report'][field] = {'wrong': 'value'}
            f.resign_report(q)
            with self.subTest(field=field):
                self.compare(f, [q], error='actual runtime/profile/numerical/source evidence')

    def test_group_last_child_failure_is_atomic(self):
        for kind in ('proof', 'epoch', 'outcome-count'):
            f = Fixture()
            q = f.queue()
            audit = q['report']['audits'][-1]
            if kind == 'proof': audit['submission_sha256'] = sha('bad')
            elif kind == 'epoch': audit['epoch'] = 'other'
            else: audit['outcomes'] = []
            f.resign_report(q)
            with self.subTest(kind=kind):
                self.compare(f, [q], error='single committed batch outcome' if kind == 'outcome-count' else 'original audited proof digest')

    def test_full_child_count_and_row_binding(self):
        f = Fixture()
        q = f.queue()
        q['report']['audits'].pop()
        f.resign_report(q)
        self.compare(f, [q], error='original full child report population')
        f = Fixture()
        f.job['submissions'][-1]['sha256'] = sha('wrong-proof')
        self.compare(f, error='audit immutable committed child/execution cohort')

    def test_typed_backend_deferral_preserves_valid_peer(self):
        f = Fixture()
        declined = f.queue()
        declined['report']['execution_resources_enforced'] = False
        f.resign_report(declined)
        self.compare(f, [declined], error='authenticated backend evidence not prospectively admitted')
        result, _ = self.compare(f, [declined, f.queue()], defer=True)
        self.assertEqual(len(result[1][0]), 1)
        self.assertEqual(len(result[1][1]), 1)
        self.assertEqual(result[1][1][0]['job_sha256'], declined['digest'])

    def test_non_backend_failures_never_become_deferrals(self):
        f = Fixture()
        q = f.queue()
        q['report_digest'] = sha('wrong')
        self.compare(f, [q], defer=True, error='original worker terminal report request')

    def test_execution_policy_cutoff_and_exact_scope(self):
        f = Fixture()
        q = f.queue()
        q['report']['execution_resources_enforced'] = False
        f.resign_report(q)
        ep = dict(version='explicit-backend-execution-evidence-v2', effective_cutoff=10,
            sources={sha('source'): dict(backend='standard-backend-no-os-resource-enforcement-v1',
                backend_module_sha256=sha('backend'), model_runtime_revision=f.manifest['model_runtime_revision'],
                backend_profile=f.manifest['backend_profile'], numerical_policy=f.manifest['numerical_policy'],
                runtime_versions=f.job['runtime_versions'], execution_resources_enforced=False, effective_cutoff=30)})
        f.job['source_files']['subnet/backend_jobs.py'] = sha('backend')
        f.sources[sha('source')] = copy.deepcopy(f.job['source_files'])
        f.report['source_files'] = copy.deepcopy(f.job['source_files'])
        f.report['execution_resources_enforced'] = False
        q = f.queue()
        self.compare(f, [q], execution_evidence_policy=ep, cutoff=30)
        self.compare(f, [q], execution_evidence_policy=ep, cutoff=29, error='not prospectively admitted')
        self.compare(f, [q], execution_evidence_policy=ep, cutoff=9, error='cutoff not reached')

    def test_retired_worker_only_exact_original_report(self):
        f = Fixture()
        q = f.queue()
        f.workers = {}
        historical = dict(version='exact-retired-verifier-reports-v1',
            reports={f.identity: {q['digest']: q['report_digest']}})
        self.compare(f, [q], historical_report_admission=historical)
        historical['reports'][f.identity][q['digest']] = sha('wrong')
        self.compare(f, [q], historical_report_admission=historical, error='admitted actual worker')

    def test_repeated_jobs_no_cross_invocation_or_global_cache(self):
        f = Fixture()
        self.compare(f, [f.queue(), f.queue()])
        old = f.queue()
        f.job['job_id'] = 'new-same-python-object'
        f.report['job_id'] = 'new-same-python-object'
        current = f.queue()
        self.compare(f, [old, current])
        current['digest'] = old['digest']
        self.compare(f, [current], error='original queue request digest')

    def test_short_circuit_errors_before_first_hash_are_preserved(self):
        for target in ('missing-role', 'wrong-role', 'missing-id', 'non-json-report-request'):
            f = Fixture()
            if target == 'missing-role': f.job.pop('role')
            elif target == 'wrong-role': f.job['role'] = 'train'
            elif target == 'missing-id': f.job.pop('job_id')
            # Build independently when job_id itself is deliberately absent.
            if target == 'missing-id':
                f.job['job_id'] = 'synthetic-job'
                q = f.queue()
                del q['envelope']['payload']['job_id']
                q['envelope'] = signed(f.root, q['envelope']['payload'])
            else:
                q = f.queue()
            if target == 'non-json-report-request': q['report_request'] = '{bad'
            with self.subTest(target=target):
                expected = call(original, f, [q])
                self.assertEqual(expected[0][0], 'error')
                self.compare(f, [q], error=expected[0][-1])

    def test_digest_reduction_is_only_invocation_local(self):
        f = Fixture()
        queue = f.queue()
        counts = []
        for module in (original, candidate):
            original_digest = module.digest
            seen = {'job': 0, 'report': 0}
            def counted(value):
                if value == queue['envelope']['payload']: seen['job'] += 1
                if value == queue['report']: seen['report'] += 1
                return original_digest(value)
            module.digest = counted
            try:
                module.admit_queue_reports([copy.deepcopy(queue)], f.rows, f.authority, f.workers, f.sources)
            finally:
                module.digest = original_digest
            counts.append(seen)
        self.assertEqual(counts, [{'job': 7, 'report': 2}, {'job': 1, 'report': 1}])


if __name__ == '__main__':
    unittest.main(verbosity=2)
