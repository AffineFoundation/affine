"""Synthetic read-only independent ledger projection controls."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ops.compact_training_audit_reader import read_admissions
from subnet import compact_training_inputs as compact,training_receipts as v1
from subnet.backend_jobs import SOURCE_FILES,COVERED_POLICY
from training_receipt_fixtures import sign,transport_fixture
from test_compact_training_inputs import make_setup


class CompactAuditReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.s=make_setup(self.root)
        self.manifest=self.s['manifest'];self.authority=self.s['authority']
        self.job=dict(schema=1,job_id='future-audit-train',role='train',created_at=22,expires_at=100,
            manifest=sign(self.s['key'],self.manifest),source_files={n:'a'*64 for n in set(SOURCE_FILES)|{
                'subnet/compact_training_inputs.py','subnet/training_receipts.py'}},
            runtime_versions=dict(torch='synthetic',transformers='synthetic',toploc='synthetic'),
            submissions=[self.s['obj']],steps=3,training_policy=COVERED_POLICY,training_input_policy=compact.VERSION)
        summary,_=compact.admitted_submission(self.s['path'],self.s['obj'],self.manifest,self.authority)
        self.report=dict(job_id=self.job['job_id'],job_sha256=v1.sha(self.job),operator=self.authority,
            role='train',epoch=self.manifest['epoch'],checkpoint=self.manifest['checkpoint']['id'],
            source_files=self.job['source_files'],runtime_versions=self.job['runtime_versions'],success=True,chain_transactions=False,
            audits=[],training_admissions=[summary],training=dict(training_input_policy=compact.VERSION,
                trainer_verification_performed=False,all_pairs_authenticated_verifier_receipts=True))
        self.rows={self.s['row']['id']:self.s['row']};self.workers={self.s['row']['worker']:['verify']}

    def read(self):
        return read_admissions(sign(self.s['key'],self.job),sign(self.s['key'],self.report),self.authority,
            source_files=self.job['source_files'],completed_rows=self.rows,workers=self.workers)

    def test_compact_transport_projects_exact_original_zip_and_authenticated_lineage(self):
        row=self.read()[0]
        self.assertEqual(row['submission_sha256'],self.s['fx']['frozen']['sha256'])
        self.assertEqual(row['submission_size'],self.s['fx']['frozen']['size'])
        self.assertEqual(row['training_transport_sha256'],self.s['obj']['sha256'])
        self.assertNotEqual(row['submission_sha256'],row['training_transport_sha256'])
        self.assertEqual(row['original_verifier_receipt_sha256'],v1.sha(self.s['fx']['receipt']))

    def test_raw_report_claiming_matching_job_sha_is_not_authenticity(self):
        with self.assertRaisesRegex(ValueError,'authority attestation'):
            read_admissions(sign(self.s['key'],self.job),self.report,self.authority,
                source_files=self.job['source_files'],completed_rows=self.rows,workers=self.workers)
        attestation=dict(version='training-report-audit-attestation-v1',training_job_sha256=v1.sha(self.job),
            training_report_sha256=v1.sha(self.report))
        result=read_admissions(sign(self.s['key'],self.job),self.report,self.authority,
            source_files=self.job['source_files'],completed_rows=self.rows,workers=self.workers,
            report_attestation=sign(self.s['key'],attestation))
        self.assertEqual(result[0]['submission_sha256'],self.s['fx']['frozen']['sha256'])
        for field in ('training_job_sha256','training_report_sha256'):
            bad=dict(attestation);bad[field]='0'*64
            with self.subTest(field=field),self.assertRaises(ValueError):
                read_admissions(sign(self.s['key'],self.job),self.report,self.authority,
                    source_files=self.job['source_files'],completed_rows=self.rows,workers=self.workers,
                    report_attestation=sign(self.s['key'],bad))


    def test_original_complete_registered_evidence_cannot_be_replaced(self):
        for mutation in ('status','report-digest','worker','roster'):
            rows=copy.deepcopy(self.rows);workers=copy.deepcopy(self.workers)
            if mutation=='status':next(iter(rows.values()))['status']='leased'
            elif mutation=='report-digest':next(iter(rows.values()))['report_digest']='0'*64
            elif mutation=='worker':next(iter(rows.values()))['worker']='0'*64
            else:workers={}
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                read_admissions(sign(self.s['key'],self.job),sign(self.s['key'],self.report),self.authority,
                    source_files=self.job['source_files'],completed_rows=rows,workers=workers)

    def test_altered_original_report_provenance_refused_even_with_new_operator_wrapper(self):
        payload=copy.deepcopy(self.s['obj']['verifier_receipt']['payload'])
        payload['original_verifier_receipt']['payload']['original_report_sha256']='0'*64
        payload['original_verifier_receipt']=sign(self.s['key'],payload['original_verifier_receipt']['payload'])
        self.job['submissions'][0]['verifier_receipt']=sign(self.s['key'],payload)
        self.report['job_sha256']=v1.sha(self.job)
        with self.assertRaises(ValueError):self.read()

    def test_source_inventory_cannot_be_inferred_from_report_or_mutable_worktree(self):
        with self.assertRaises(ValueError):
            read_admissions(sign(self.s['key'],self.job),sign(self.s['key'],self.report),self.authority,
                source_files={'subnet/compact_training_inputs.py':'0'*64},completed_rows=self.rows,workers=self.workers)

    def test_v1_reader_never_imports_compact_and_keeps_original_transport_identity(self):
        fx=transport_fixture(self.s['key']);self.manifest=fx['manifest']
        self.job.update(manifest=sign(self.s['key'],self.manifest),submissions=[fx['submission']],training_input_policy=v1.VERSION)
        self.job['source_files'].pop('subnet/compact_training_inputs.py')
        path=self.root/'original.zip';path.write_bytes(fx['data'])
        summary,_=v1.admitted_submission(path,fx['submission'],self.manifest,self.authority)
        self.report.update(job_sha256=v1.sha(self.job),source_files=self.job['source_files'],training_admissions=[summary],
            training=dict(training_input_policy=v1.VERSION,trainer_verification_performed=False,all_pairs_authenticated_verifier_receipts=True))
        request=fx['worker_request'];job=fx['verify_job'];self.workers={request['signer']:['verify']}
        import json
        self.rows={job['payload']['job_id']:dict(id=job['payload']['job_id'],status='complete',role='verify',
            envelope=json.dumps(job),digest=v1.sha(job['payload']),report=json.dumps(request['payload']['report']),
            report_digest=v1.sha(request['payload']['report']),report_request=json.dumps(request),worker=request['signer'])}
        with patch('subnet.compact_training_inputs.validate_report',side_effect=AssertionError('v1 imports compact')):
            row=self.read()[0]
        self.assertEqual(row['submission_sha256'],row['training_transport_sha256'])
