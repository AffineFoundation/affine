"""Crypto/archive controls only; synthetic fixtures are not model qualification."""
import base64
import copy
import hashlib
import io
import json
import tarfile
import unittest
import zipfile
from nacl.signing import SigningKey
from subnet.numerical_resolution import canonical, digest, reference


def signed(key, payload):
    return dict(signer=key.verify_key.encode().hex(), payload=payload,
                signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())


class NineReferenceArchiveControls(unittest.TestCase):
    def setUp(self):
        self.root, self.worker = SigningKey.generate(), SigningKey.generate()
        self.authority = self.root.verify_key.encode().hex()
        self.contents = {}
        cases, terminals = [], []
        cp, source = '1' * 64, '2' * 64
        manifest = signed(self.root, dict(epoch='fixture-e30', checkpoint=dict(id=cp), source_bundle=dict(sha256=source)))
        for i in range(9):
            batch = dict(epoch='fixture-e30', checkpoint=cp, env_id='math', index=i, sample_index=i)
            ref = dict(miner='3' * 64, slot=i, env_id='math', index=i, batch_sha256=digest(batch))
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as z:
                z.writestr('manifest.json', canonical([dict(batch=batch)]))
            artifact = stream.getvalue(); artifact_sha = hashlib.sha256(artifact).hexdigest()
            job = dict(job_id='fixture-' + str(i), manifest=manifest,
                       submissions=[dict(sha256=artifact_sha, commitment_ref=ref)])
            envelope = signed(self.root, job)
            report = dict(job_id=job['job_id'], job_sha256=digest(job), audits=[dict(
                submission_sha256=artifact_sha, selected_batches=[0], outcomes=[dict(batch=0, fully_audited=True)])])
            request = signed(self.worker, dict(action='report', job_id=job['job_id'], report=report))
            prefix = 'inputs/case-' + str(i) + '/'
            data = dict(job=canonical(envelope), report_request=canonical(request), artifact=artifact)
            for name, raw in data.items():
                self.contents[prefix + ('artifact.zip' if name == 'artifact' else name + '.json')] = raw
            case = dict(original_job_sha256=digest(job), worker=self.worker.verify_key.encode().hex(), child=0,
                file_sha256={k: hashlib.sha256(v).hexdigest() for k, v in data.items()},
                report_request_sha256=digest(request), report_sha256=digest(report),
                commitment_ref=ref, selected_batches=[0])
            cases.append(case)
            result = dict(original_job_sha256=digest(job), original_report_request_sha256=digest(request),
                artifact_sha256=artifact_sha, checkpoint=cp, source_bundle_sha256=source)
            raw = canonical(result); self.contents['original-output/case-' + str(i) + '-research.json'] = raw
            terminals.append(dict(case=i, returncode=0, timed_out=False, original_job_sha256=digest(job),
                                  output_sha256=hashlib.sha256(raw).hexdigest()))
        self.contents.update({'run_nine_CP20_references.py': b'reviewed runner fixture',
            'supervisor_nine_CP20_1650.py': b'reviewed supervisor fixture',
            'toploc_reference_adjudication.py': b'reviewed diagnostic fixture',
            'original-supervisor-intent.json': b'{}', 'original-supervisor-child.json': b'{}',
            'original-output/original-terminal.json': canonical(dict(all_nine_completed=True, completed_cases=9, cases=terminals))})
        qualification = signed(self.root, dict(full_readback_verified=True, scientific_qualification_passed=True,
            control_count=20, honest_VALID=4, production_changes=False))
        scope = dict(version='nine-CP20-original-reference-root-scope-v1', execute_allowed=True,
            production_mutations=False, production_queue_used=False, optimizer_updates=0, cases=cases,
            per_case_seconds=150, total_seconds=1500, supervisor_wall_seconds=1650,
            CUBLAS_WORKSPACE_CONFIG=':4096:8', qualification_full_ACK=qualification,
            qualification_result_sha256='4' * 64, checkpoint=cp, source_sha256=source)
        for filename, field in [('run_nine_CP20_references.py', 'runner_sha256'),
                ('supervisor_nine_CP20_1650.py', 'supervisor_sha256'),
                ('toploc_reference_adjudication.py', 'diagnostic_sha256')]:
            scope[field] = hashlib.sha256(self.contents[filename]).hexdigest()
        raw_scope = canonical(signed(self.root, scope)); self.scope_sha = hashlib.sha256(raw_scope).hexdigest()
        self.contents['scope.ROOT-SIGNED.json'] = raw_scope
        self.contents['original-supervisor-terminal.json'] = canonical(dict(scope_sha256=self.scope_sha,
            exit_code=0, timed_out=False, production_mutations=False))

    def archive(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, raw in self.contents.items():
                member = tarfile.TarInfo(name); member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        raw = stream.getvalue()
        ack = signed(self.root, dict(version='research-original-archive-full-readback-ack-v1',
            full_readback_verified=True, production_changes=False, sha256=hashlib.sha256(raw).hexdigest(),
            scope_file_sha256=self.scope_sha, reference_execution_completed=True, reference_case_count=9))
        return ack, raw

    def test_all_nine_authenticated_bindings(self):
        ack, raw = self.archive()
        self.assertEqual(len(reference(ack, raw, self.authority)[2]), 9)

    def test_truncated_or_unsigned_archive_refused(self):
        ack, raw = self.archive()
        for document, data in [(ack, raw[:-1]), (dict(ack, signature='AAAA'), raw)]:
            with self.assertRaises(Exception):
                reference(document, data, self.authority)

    def test_inner_job_worker_scope_and_result_substitutions_refused(self):
        original = copy.deepcopy(self.contents)
        for name in ['inputs/case-0/job.json', 'inputs/case-0/report_request.json',
                     'scope.ROOT-SIGNED.json', 'original-output/case-0-research.json']:
            self.contents = copy.deepcopy(original)
            doc = json.loads(self.contents[name])
            (doc['payload'] if 'payload' in doc else doc)['substitution'] = True
            self.contents[name] = canonical(doc)
            ack, raw = self.archive()
            with self.subTest(name=name), self.assertRaises(Exception):
                reference(ack, raw, self.authority)
