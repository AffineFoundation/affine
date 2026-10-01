import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nacl.exceptions import BadSignatureError
from subnet.storage import Identity
from subnet.backend_jobs import canonical, file_map
from ops.check_gpu_continuous_evidence import checked_job


class ContinuousJobEvidenceTests(unittest.TestCase):
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
