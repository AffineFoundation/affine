from types import SimpleNamespace
import unittest
from ops.toploc_reference_adjudication import instrument, inputs, canonical, digest
import base64
import hashlib
from nacl.signing import SigningKey


class ReferenceDiagnosticsTests(unittest.TestCase):
    def test_metrics_wrapper_preserves_native_result_and_arguments(self):
        native = [SimpleNamespace(exp_mismatches=0, mant_err_mean=.25, mant_err_median=0)]
        calls = []
        def verify(acts, proofs, **kwargs):
            calls.append((acts, proofs, kwargs)); return native
        runtime = SimpleNamespace(verify_proofs=verify)
        diagnostics = []
        instrument(runtime, diagnostics)
        acts = object()
        result = runtime.verify_proofs(acts, ['proof'], decode_batching_size=16, topk=128)
        self.assertIs(result, native)
        self.assertEqual(calls, [(acts, ['proof'], {'decode_batching_size':16, 'topk':128})])
        self.assertEqual(diagnostics[0]['segments'][0]['mant_err_mean'], .25)

    def test_segment_count_disagreement_is_preserved(self):
        native = [SimpleNamespace(exp_mismatches=0, mant_err_mean=0, mant_err_median=0)]
        runtime = SimpleNamespace(verify_proofs=lambda *a, **k: native)
        diagnostics = []
        instrument(runtime, diagnostics)
        self.assertIs(runtime.verify_proofs([], []), native)
        self.assertEqual(diagnostics[0]['expected_segments'], 0)
        self.assertIsNone(diagnostics[0]['segments'][0]['proof_sha256'])

    def test_metrics_wrapper_does_not_hide_native_failure(self):
        def verify(*args, **kwargs): raise RuntimeError('native failure')
        runtime = SimpleNamespace(verify_proofs=verify)
        instrument(runtime, [])
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            runtime.verify_proofs([], [])

    def test_wrong_original_signature_refused_before_artifact_use(self):
        with self.assertRaisesRegex(ValueError, 'expected original signer'):
            inputs({'signer':'wrong'}, {}, 'a'*64, 'b'*64, 0, b'artifact')


class OriginalBindingTests(unittest.TestCase):
    def fixture(self):
        authority, worker = SigningKey.generate(), SigningKey.generate()
        def signed(payload, key):
            return dict(payload=payload, signer=key.verify_key.encode().hex(),
                        signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())
        artifact = b'immutable artifact'
        artsha = hashlib.sha256(artifact).hexdigest()
        manifest = dict(epoch='epoch', checkpoint={'id':'checkpoint'}, backend_profile={}, numerical_policy={})
        job = dict(job_id='original', manifest=signed(manifest, authority), source_files={}, runtime_versions={}, submissions=[{'sha256':artsha}])
        report = dict(success=True, role='verify', job_id='original', job_sha256=digest(job), epoch='epoch',
                      checkpoint='checkpoint', source_files={}, runtime_versions={}, backend_profile={}, numerical_policy={},
                      audits=[dict(submission_sha256=artsha, epoch='epoch', outcomes=[{'valid':False}])])
        request = dict(action='report', job_id='original', report=report)
        return signed(job, authority), signed(request, worker), authority.verify_key.encode().hex(), worker.verify_key.encode().hex(), artifact

    def test_immutable_original_bindings_accepted(self):
        job, request, authority, worker, artifact = self.fixture()
        self.assertEqual(inputs(job, request, authority, worker, 0, artifact)[3]['outcomes'], [{'valid':False}])

    def test_mutated_committed_artifact_refused(self):
        job, request, authority, worker, artifact = self.fixture()
        with self.assertRaisesRegex(ValueError, 'immutable original artifact digest'):
            inputs(job, request, authority, worker, 0, artifact+b'changed')

    def test_bad_child_refused(self):
        job, request, authority, worker, artifact = self.fixture()
        with self.assertRaisesRegex(ValueError, 'original child index'):
            inputs(job, request, authority, worker, 1, artifact)


class ClassificationTests(unittest.TestCase):
    def test_infrastructure_never_becomes_invalid(self):
        from ops.toploc_reference_adjudication import reference_check
        class Invalid(Exception): pass
        class Ambiguous(Exception): pass
        def crash(*args): raise RuntimeError('allocation failure')
        result = reference_check(SimpleNamespace(verify=crash), {}, [], Invalid, Ambiguous)
        self.assertIsNone(result['reference_valid'])
        self.assertEqual(result['classification'], 'research_infrastructure_error')

    def test_numeric_ambiguity_never_becomes_invalid(self):
        from ops.toploc_reference_adjudication import reference_check
        class Invalid(Exception): pass
        class Ambiguous(Exception): pass
        def crash(*args): raise Ambiguous('boundary')
        result = reference_check(SimpleNamespace(verify=crash), {}, [], Invalid, Ambiguous)
        self.assertIsNone(result['reference_valid'])
        self.assertEqual(result['classification'], 'numerical_ambiguous')

    def test_full_commit_tuple_and_selection_required(self):
        from ops.toploc_reference_adjudication import committed_batch
        batch = dict(epoch='e', checkpoint='c', env_id='math', index=4, sample_index=4)
        manifest = dict(epoch='e', checkpoint={'id':'c'})
        obj = dict(commitment_miner='m', commitment_ref=dict(miner='m',slot=0,env_id='math',index=4,batch_sha256=digest(batch)))
        audit = dict(selected_batches=[0], outcomes=[dict(batch=0,fully_audited=True)])
        self.assertTrue(committed_batch(manifest,obj,audit,batch))
        with self.assertRaisesRegex(ValueError,'full committed task tuple'):
            committed_batch(manifest,obj,audit,dict(batch,index=5))
        with self.assertRaisesRegex(ValueError,'selected fully audited child'):
            committed_batch(manifest,obj,dict(audit,selected_batches=[]),batch)
