import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from subnet.backend_jobs import canonical, file_map
from subnet.storage import Identity
from ops.check_gpu_role_evidence import inspect


class GPUControlEvidence(unittest.TestCase):
    def fixture(self, root):
        identity=Identity(bytes(range(32)));(root/'authority.seed').write_text(identity.key.encode().hex())
        def sign(payload):
            import base64
            return dict(payload=payload,signer=identity.id,signature=base64.b64encode(identity.key.sign(canonical(payload)).signature).decode())
        cps=[]
        for char in ('a','b'):
            files={'config.json':char*64,'model.safetensors':char*64};cps.append(dict(files=files,id=file_map(files)))
        reports=[];objects={};audits=[]
        for i in range(2):
            body=('frozen-'+str(i)).encode();digest=hashlib.sha256(body).hexdigest();objects['frozen'+str(i)]=body
            receipt=dict(frozen_key='frozen'+str(i),sha256=digest,size=len(body),received_at=15)
            (root/f'epoch{i}-freeze-policy.json').write_text(json.dumps(dict(payable=False,receipt=receipt)))
            audits.append(dict(submission_sha256=digest,accepted=[{}],outcomes=[dict(valid=True,fully_audited=True)]))
        rows=[('mine',0),('verify',0),('train',0),('evaluate',1),('upload',1),('mine',1),('verify',1)]
        for index,(role,epoch) in enumerate(rows):
            manifest=dict(epoch='epoch'+str(epoch),checkpoint=cps[epoch],start=10,deadline=20,payable=False,backend_profile={},numerical_policy={})
            job=dict(job_id=str(index),role=role,created_at=10,expires_at=25,manifest=sign(manifest),source_files={},runtime_versions={},submissions=[dict(sha256=audits[epoch]['submission_sha256'])])
            report=dict(job_id=str(index),role=role,operator=identity.id,checkpoint=cps[epoch]['id'],epoch=manifest['epoch'],source_files={},runtime_versions={},backend_profile={},numerical_policy={},job_sha256=hashlib.sha256(canonical(job)).hexdigest(),success=True,chain_transactions=False,completed_at=16)
            if role=='mine':report.update(submission_sha256=audits[epoch]['submission_sha256'],submission_size=len(objects['frozen'+str(epoch)]),batches=1)
            if role in ('verify','train'):report['audits']=[audits[epoch]]
            if role=='train':report.update(new_checkpoint=cps[1],training={'steps':1})
            (root/f'{index}-job.json').write_text(json.dumps(sign(job)))
            (root/f'{index}-report.json').write_text(json.dumps(report))
        (root/'complete.json').write_text(json.dumps(dict(success=True,chain_transactions=False,epochs=['epoch0','epoch1'],new_checkpoint=cps[1],full_model_finetune=False)))
        (root/'root-r2-checkpoint-independent-check.json').write_text(json.dumps(dict(checkpoint=cps[1]['id'],published_file_hashes_verified=True,files={n:dict(sha256=s,bytes=1) for n,s in cps[1]['files'].items()})))
        class Bucket:
            def get(self,key):return objects[key]
        return Bucket(),objects

    def test_honest_bound_control(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bucket,_=self.fixture(root)
            result=inspect(root,bucket)
            self.assertTrue(result['success']);self.assertFalse(result['continuous_controller_finalization_proven'])

    def test_corrupt_published_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bucket,objects=self.fixture(root);objects['frozen1']=b'corrupt'
            with self.assertRaisesRegex(ValueError,'frozen R2 bytes'):inspect(root,bucket)

    def test_changed_report_profile(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bucket,_=self.fixture(root);path=root/'1-report.json'
            r=json.loads(path.read_text());r['backend_profile']={'device':'wrong'};path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'binding mismatch'):inspect(root,bucket)

    def test_unsigned_job_change(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bucket,_=self.fixture(root);path=root/'1-job.json'
            r=json.loads(path.read_text());r['payload']['role']='mine';path.write_text(json.dumps(r))
            from nacl.exceptions import BadSignatureError
            with self.assertRaises(BadSignatureError):inspect(root,bucket)
