import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from nacl.signing import SigningKey
from ops.adopt_online_verifier_cache_catalog import VERSION, apply
from subnet.cache_lifecycle import CacheLifecycle, snapshot
from subnet.storage import canonical

class OnlineCatalogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.parent = Path(self.directory.name)
        self.root = self.parent / 'backend'
        self.root.mkdir()
        self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()
        self.cp = self.model(b'old')
        self.current = self.model(b'current')
        self.wrapper = self.parent / 'wrapper.py'; self.wrapper.write_text('pinned leased worker')
        self.worker = self.parent / 'worker.py'; self.worker.write_text('worker lease')
        self.lifecycle = self.parent / 'lifecycle.py'; self.lifecycle.write_text('inherited lease')
        files = {name:dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in [('wrapper',self.wrapper),('worker',self.worker),('lifecycle',self.lifecycle)]}
        self.value = dict(revision=VERSION,managed_checkpoint_leases_required=True,created_at=10,expires_at=20,
            expected_wrapper=dict(pid=987654,start_ticks=100,operator_files=files),
            roots=[dict(root=str(self.root),keep=[self.current['id']],checkpoints=[self.cp,self.current])])
        self.processes = {987654:dict(start_ticks=100,state='S',parent=1,arguments=['python',str(self.wrapper),'--workspace',str(self.parent)])}
    def sign(self,value):
        return dict(payload=value,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(value)).signature).decode())
    def model(self,raw):
        data={'config.json':b'{}','model.safetensors':raw}
        files={name:hashlib.sha256(content).hexdigest() for name,content in data.items()}
        cp=hashlib.sha256(canonical(files)).hexdigest(); directory=self.root/'checkpoints'/cp;directory.mkdir(parents=True)
        for name,content in data.items():(directory/name).write_bytes(content)
        return dict(id=cp,files=files,stats={name:snapshot(directory/name) for name in files},durability_ack=dict(signed_manifest_verified=True,independent_durable_hashes=True))
    def run_apply(self):return apply(self.sign(self.value),self.authority,now=15,observed=self.processes)[0]
    def test_online_removes_exact_old_model_keeps_current_and_diagnostics(self):
        report=self.root/'jobs'/'old'/'report.json';report.parent.mkdir(parents=True);report.write_text('keep')
        result=self.run_apply();self.assertEqual(result['removed'],[self.cp['id']]);self.assertTrue(report.exists());self.assertTrue((self.root/'checkpoints'/self.current['id']).exists())
    def test_active_checkpoint_lease_is_skipped_without_worker_pause(self):
        with CacheLifecycle(self.root).lease_checkpoint(self.cp['id']):
            result=self.run_apply();self.assertEqual(result['removed'],[]);self.assertEqual(result['skipped'][0]['reason'],'leased')
    def test_changed_member_cannot_be_adopted(self):
        (self.root/'checkpoints'/self.cp['id']/'model.safetensors').write_bytes(b'changed')
        self.assertEqual(self.run_apply()['removed'],[])
    def test_symlink_or_hardlink_cannot_be_adopted(self):
        member=self.root/'checkpoints'/self.cp['id']/'model.safetensors'
        os.link(member,self.parent/'extra-link');self.assertEqual(self.run_apply()['removed'],[])
    def test_new_unlisted_receipted_checkpoint_is_never_removed(self):
        newer=self.model(b'newer');cache=CacheLifecycle(self.root)
        with cache.lease_checkpoint(newer['id']):cache.record_checkpoint(newer['id'],newer['files'])
        self.assertEqual(self.run_apply()['removed'],[self.cp['id']]);self.assertTrue((self.root/'checkpoints'/newer['id']).exists())
    def test_wrapper_pid_reuse_operator_change_and_legacy_reader_fail_closed(self):
        self.processes[987654]['start_ticks']=101
        with self.assertRaises(ValueError):self.run_apply()
        self.processes[987654]['start_ticks']=100;self.worker.write_text('changed')
        with self.assertRaises(ValueError):self.run_apply()
        self.worker.write_text('worker lease')
        self.processes[987655]=dict(start_ticks=1,state='S',parent=1,arguments=['python','--workspace',str(self.parent)])
        with self.assertRaises(ValueError):self.run_apply()
    def test_real_wrapper_child_allowed_and_lease_still_protects_inputs(self):
        self.processes[987655]=dict(start_ticks=1,state='R',parent=987654,arguments=['python','--workspace',str(self.root)])
        with CacheLifecycle(self.root).lease_checkpoint(self.cp['id']):self.assertEqual(self.run_apply()['removed'],[])
    def test_old_contract_expiry_incomplete_durability_signature_rejected(self):
        for field,bad in [('revision','owned-verifier-cache-catalog-online-wrapper-v2'),('created_at',16),('expires_at',14)]:
            old=self.value[field];self.value[field]=bad
            with self.assertRaises(ValueError):self.run_apply()
            self.value[field]=old
        self.cp['durability_ack']['independent_durable_hashes']=False
        with self.assertRaises(ValueError):self.run_apply()
        self.cp['durability_ack']['independent_durable_hashes']=True
        envelope=self.sign(self.value);envelope['payload']['roots'][0]['keep']=[]
        with self.assertRaises(Exception):apply(envelope,self.authority,now=15,observed=self.processes)
    def download(self):
        job=dict(job_id='historic',role='verify',manifest=self.sign(dict(checkpoint=dict(id=self.cp['id'],files=self.cp['files']))),submissions=[dict(sha256=hashlib.sha256(b'rollouts').hexdigest())])
        jobsha=hashlib.sha256(canonical(job)).hexdigest();report=dict(job_id='historic',job_sha256=jobsha,success=True)
        folder=self.root/'jobs'/'historic';folder.mkdir(parents=True);path=folder/'submission-0.zip';path.write_bytes(b'rollouts');(folder/'report.json').write_bytes(canonical(report))
        row=dict(original_signed_job=self.sign(job),coordinator_ack=dict(accepted=True,job_sha256=jobsha,report_sha256=hashlib.sha256(canonical(report)).hexdigest()),files={'submission-0.zip':dict(sha256=job['submissions'][0]['sha256'],stat=snapshot(path))})
        self.value['roots'][0]['completed_downloads']=[row];return row,path
    def test_exact_accepted_historical_inputs_retire_report_stays(self):
        row,path=self.download();result=self.run_apply();self.assertEqual(result['retired_downloads'],['jobs/historic/submission-0.zip']);self.assertFalse(path.exists());self.assertTrue((path.parent/'report.json').exists())
    def test_missing_ack_changed_input_or_active_checkpoint_keeps_download(self):
        row,path=self.download();row['coordinator_ack']['accepted']=False;self.assertEqual(self.run_apply()['retired_downloads'],[]);self.assertTrue(path.exists())
        row['coordinator_ack']['accepted']=True
        with CacheLifecycle(self.root).lease_checkpoint(self.cp['id']):self.assertEqual(self.run_apply()['retired_downloads'],[])
        path.write_bytes(b'changed');self.assertEqual(self.run_apply()['retired_downloads'],[]);self.assertTrue(path.exists())
    def test_unlisted_download_receipt_is_retained(self):
        row,path=self.download();other=path.parent/'submission-1.zip';other.write_bytes(b'other');CacheLifecycle(self.root).record_download(other,hashlib.sha256(b'other').hexdigest())
        self.run_apply();self.assertTrue(other.exists())

if __name__=='__main__':unittest.main()
