import base64
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from nacl.exceptions import BadSignatureError
from subnet.storage import Identity
from subnet.backend_jobs import canonical, file_map
from ops.check_gpu_continuous_evidence import checked_job, check_source_bundle


class ContinuousJobEvidenceTests(unittest.TestCase):
    def source_fixture(self, names=('subnet/model.py',)):
        stream=io.BytesIO();body=b'approved source'
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            for name in names:
                entry=tarfile.TarInfo(name);entry.size=len(body);archive.addfile(entry,io.BytesIO(body))
        raw=stream.getvalue();descriptor=dict(size=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        return raw,descriptor,{'subnet/model.py':hashlib.sha256(body).hexdigest()}

    def test_published_worker_source_bytes(self):
        raw,descriptor,expected=self.source_fixture()
        self.assertEqual(check_source_bundle(raw,descriptor,expected)['source_files'],1)

    def test_wrong_worker_source_or_missing_file(self):
        raw,descriptor,expected=self.source_fixture()
        with self.assertRaisesRegex(ValueError,'inventory'):
            check_source_bundle(raw,descriptor,{'subnet/model.py':'d'*64})
        with self.assertRaisesRegex(ValueError,'inventory'):
            check_source_bundle(raw,descriptor,dict(expected,**{'subnet/missing.py':'d'*64}))
        with self.assertRaisesRegex(ValueError,'archive bytes'):
            check_source_bundle(raw+b'corrupt',descriptor,expected)

    def test_duplicate_worker_source_entry(self):
        raw,descriptor,expected=self.source_fixture(('subnet/model.py','./subnet/model.py'))
        with self.assertRaisesRegex(ValueError,'source entry'):check_source_bundle(raw,descriptor,expected)

    def fixture(self, root):
        identity = Identity(bytes(range(32)))
        def sign(payload):
            return dict(payload=payload, signer=identity.id,
                        signature=base64.b64encode(identity.key.sign(canonical(payload)).signature).decode())
        files = {'config.json':'a'*64,'model.safetensors':'b'*64}
        manifest = dict(epoch='nonpayable-test', checkpoint=dict(id=file_map(files),files=files),
                        backend_profile={'device':'cuda'}, numerical_policy={'atol':0})
        job = dict(job_id='test',role='train',created_at=1,expires_at=3,manifest=sign(manifest),
                   source_files={'subnet/model.py':'c'*64},runtime_versions={'torch':'pinned'})
        report = dict(job_id='test',role='train',operator=identity.id,checkpoint=manifest['checkpoint']['id'],
                      epoch=manifest['epoch'],source_files=job['source_files'],runtime_versions=job['runtime_versions'],
                      backend_profile=manifest['backend_profile'],numerical_policy=manifest['numerical_policy'],
                      job_sha256=hashlib.sha256(canonical(job)).hexdigest(),success=True,
                      chain_transactions=False,completed_at=2)
        (root/'roles').mkdir(); (root/'roles/test-job.json').write_text(json.dumps(sign(job)))
        (root/'roles/test-report.json').write_text(json.dumps(report))
        return identity.id

    def test_valid_signed_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); authority=self.fixture(root)
            self.assertEqual(checked_job(root,'test',authority)[2]['role'],'train')

    def test_changed_computation_report(self):
        for field,value in [('checkpoint','d'*64),('source_files',{}),('chain_transactions',True),
                            ('runtime_versions',{'torch':'unapproved'})]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); authority=self.fixture(root); path=root/'roles/test-report.json'
                report=json.loads(path.read_text()); report[field]=value; path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'binding'): checked_job(root,'test',authority)

    def test_unsigned_manifest_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); authority=self.fixture(root); path=root/'roles/test-job.json'
            job=json.loads(path.read_text()); job['payload']['manifest']['payload']['epoch']='changed'
            path.write_text(json.dumps(job))
            with self.assertRaises(BadSignatureError): checked_job(root,'test',authority)

    def test_expired_execution_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); authority=self.fixture(root); path=root/'roles/test-report.json'
            report=json.loads(path.read_text()); report['completed_at']=3; path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'expiry'): checked_job(root,'test',authority)

class AbortedTrainingEvidence(unittest.TestCase):
    def fixture(self):
        manifest={'epoch':'nonpayable-wide','checkpoint':{'id':'approved'}}
        status=dict(status='aborted_evaluation',epoch=manifest['epoch'],checkpoint='approved',next_checkpoint='approved',optimizer_ran=False,steps=0,payable=False,chain_transactions=False,fully_audited_batches=3,failed_jobs=['nonpayable-wide-eval-before-failed'])
        return manifest,status
    def test_untrained_abort_does_not_claim_changed_checkpoint(self):
        from ops.check_gpu_continuous_evidence import check_aborted_training
        manifest,status=self.fixture();check_aborted_training(status,manifest,3)
        for key,value in [('steps',1),('steps',False),('next_checkpoint','changed'),('optimizer_ran',True),('fully_audited_batches',4),('failed_jobs',[])]:
            changed=dict(status);changed[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):check_aborted_training(changed,manifest,3)

    def test_training_admission_abort_requires_no_launch_and_distinct_job_role(self):
        from ops.check_gpu_continuous_evidence import check_aborted_training
        manifest,status=self.fixture()
        status.update(status='aborted_training_admission',failed_jobs=['nonpayable-wide-train-rejected'],
            before_evaluation_complete=True,remote_model_or_optimizer_launch=False,
            admission_failure='missing_signed_fixed_reference_training_policy')
        check_aborted_training(status,manifest,3)
        for key,value in [('remote_model_or_optimizer_launch',True),('before_evaluation_complete',False),
                          ('failed_jobs',['nonpayable-wide-eval-before-failed']),('admission_failure','unknown')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                check_aborted_training(dict(status,**{key:value}),manifest,3)
