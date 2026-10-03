import base64
import copy
import json
import tempfile
import unittest
import os
from unittest.mock import patch
from pathlib import Path
from nacl.signing import SigningKey
from ops.submission_retention import completed_replicas,remove_verified_replica,canonical,digest

def sign(key,value):
    return dict(signer=key.verify_key.encode().hex(),payload=value,
        signature=base64.b64encode(key.sign(canonical(value)).signature).decode())

class SubmissionRetention(unittest.TestCase):
    def setUp(self):
        # The unprivileged test process cannot inspect root-owned /proc FDs.
        # Use its real readable PID namespace; production scans all processes.
        self.scanner=patch('ops.submission_retention.process_directories',return_value=[Path('/proc',str(os.getpid()))])
        self.scanner.start();self.addCleanup(self.scanner.stop)
    def fixture(self,workspace):
        op=SigningKey.generate();worker=SigningKey.generate();authority=op.verify_key.encode().hex();who=worker.verify_key.encode().hex()
        import hashlib
        hashed=hashlib.sha256(b'archive').hexdigest();source='a'*64
        receipt=dict(sha256=hashed,size=7,frozen_key='public/nonpayable-test/submissions/'+'b'*64+'.zip')
        manifest=dict(epoch='nonpayable-test',payable=False,checkpoint={'id':'model'},source_bundle={'sha256':source},
            backend_profile={'profile':'test'},numerical_policy={'tolerance':0},audit_frozen_receipts={'miner':receipt})
        job=dict(job_id='verify-1',role='verify',created_at=10,expires_at=20,manifest=sign(op,manifest),source_files={'subnet/storage.py':'c'*64},runtime_versions={'torch':'pinned'},submissions=[{'sha256':hashed,'url':'private capability'}])
        report=dict(job_id=job['job_id'],job_sha256=digest(job),operator=authority,role='verify',epoch=manifest['epoch'],checkpoint='model',source_files=job['source_files'],runtime_versions=job['runtime_versions'],backend_profile=manifest['backend_profile'],numerical_policy=manifest['numerical_policy'],chain_transactions=False,success=True,completed_at=15,audits=[dict(epoch=manifest['epoch'],submission_sha256=hashed,accepted=[])])
        request=sign(worker,dict(action='report',job_id=job['job_id'],token='winning lease',report=report))
        row=dict(id=job['job_id'],status='complete',role='verify',attempt=1,worker=who,token='winning lease',digest=digest(job),envelope=json.dumps(sign(op,job)),report=json.dumps(report),report_digest=digest(report),report_request=json.dumps(request))
        return row,authority,{who:str(workspace)},{source:job['source_files']},report

    def plan(self,root):
        row,authority,workspaces,sources,report=self.fixture(root)
        return completed_replicas(row,authority,workspaces,sources,now=16)[0],report

    def local(self,root):
        plan,report=self.plan(root);path=Path(plan['path']);path.parent.mkdir(parents=True)
        path.write_bytes(b'archive');Path(plan['report_path']).write_text(json.dumps(report));plan['archive_verified']=True
        return plan,path

    def test_exact_archived_replica_removed_but_report_and_model_stay(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);plan,path=self.local(root);model=root/'checkpoints/model';model.mkdir(parents=True);(model/'weights').write_bytes(b'weights')
            result=remove_verified_replica(plan)
            self.assertTrue(result['removed']);self.assertFalse(path.exists());self.assertTrue(Path(plan['report_path']).exists());self.assertEqual((model/'weights').read_bytes(),b'weights')
            self.assertTrue(remove_verified_replica(plan)['already_absent'])

    def test_unfinished_or_unsigned_or_wrong_worker_report_cannot_plan(self):
        with tempfile.TemporaryDirectory() as root:
            row,authority,workspaces,sources,_=self.fixture(Path(root))
            for field,value in [('status','leased'),('worker','unapproved'),('digest','changed'),('token','another lease')]:
                bad=dict(row,**{field:value})
                with self.assertRaises(ValueError):completed_replicas(bad,authority,workspaces,sources,now=16)
            bad=dict(row);envelope=json.loads(bad['report_request']);envelope['payload']['report']['success']=False;bad['report_request']=json.dumps(envelope)
            with self.assertRaises(Exception):completed_replicas(bad,authority,workspaces,sources,now=16)

    def test_unapproved_source_cannot_plan(self):
        with tempfile.TemporaryDirectory() as root:
            row,authority,workspaces,sources,_=self.fixture(Path(root))
            with self.assertRaises(ValueError):completed_replicas(row,authority,workspaces,{},now=16)

    def test_missing_archive_check_or_changed_local_bytes_cannot_delete(self):
        with tempfile.TemporaryDirectory() as root:
            plan,path=self.local(Path(root))
            with self.assertRaises(ValueError):remove_verified_replica(dict(plan,archive_verified=False))
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash changed'):remove_verified_replica(plan)
            self.assertTrue(path.exists())

    def test_symlink_or_non_download_target_cannot_delete(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);plan,path=self.local(root);outside=root/'outside';outside.write_bytes(b'archive');path.unlink();path.symlink_to(outside)
            with self.assertRaises(ValueError):remove_verified_replica(plan)
            self.assertEqual(outside.read_bytes(),b'archive')
            with self.assertRaises(ValueError):remove_verified_replica(dict(plan,path=str(outside)))

    def test_open_file_cannot_delete(self):
        with tempfile.TemporaryDirectory() as root:
            plan,path=self.local(Path(root))
            with path.open('rb'):
                with self.assertRaisesRegex(ValueError,'still open'):remove_verified_replica(plan)
            self.assertTrue(path.exists())

    def test_report_changed_cannot_delete(self):
        with tempfile.TemporaryDirectory() as root:
            plan,path=self.local(Path(root));Path(plan['report_path']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'original local report'):remove_verified_replica(plan)
            self.assertTrue(path.exists())

if __name__=='__main__':unittest.main()
