"""Remote authority-last admission controls with real Ed25519/readback receipts.

Scientific descriptor validation is mocked at its already-tested boundary.
"""
import copy
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from nacl.signing import SigningKey
from subnet import remote_optimizer_readback as r
from subnet.remote_state_commit import independently_commit_remote

class RemoteAdmission(unittest.TestCase):
    def setUp(self):
        self.root=SigningKey.generate();self.reader=SigningKey.generate()
        self.authority=bytes(self.root.verify_key).hex();self.identity=bytes(self.reader.verify_key).hex()
        self.objects=[dict(name='state-%06d.safetensors'%i,size=31,
            sha256=hashlib.sha256(bytes([i])*31).hexdigest())for i in range(23)]
        self.descriptor=dict(shards=self.objects)
        self.report={'original':'report'}
        self.manifest={'trainer_state_binding':{'source_sha256':'a'*64}}
        self.job=dict(job_id='original',manifest=r.sign(self.manifest,self.root),
            persistent_training={'output_namespace':'private/trainer-state/epoch/job'})
        self.job_env=r.sign(self.job,self.root)
        self.host=dict(provider_UUID='reader',ssh_host_key_sha256='b'*64,evidence_sha256='c'*64)
        self.trainer=dict(self.host,provider_UUID='trainer')
        self.storage=dict(storage_origin='https://test.r2.cloudflarestorage.com',
            storage_bucket='test-bucket',storage_addressing='path')
        self.binding=dict(purpose='production-training-state',provenance=dict(
            signed_job_envelope_sha256=r.sha(self.job_env),original_report_sha256=r.sha(self.report),
            signed_manifest_envelope_sha256=r.sha(self.job['manifest'])),
            job_id='original',job_sha256=r.sha(self.job),source_sha256='a'*64,
            namespace=self.job['persistent_training']['output_namespace'],
            descriptor_sha256=r.sha(self.descriptor),reader_host_record_sha256=r.sha(self.host),
            trainer_host_record_sha256=r.sha(self.trainer),**self.storage)
        payload=dict(version=r.VERSION,**self.binding,reader_identity=self.identity,
            created_at=100,expires_at=300,max_wall_seconds=200,objects=self.objects,
            capabilities={o['name']:self.storage['storage_origin']+'/test-bucket/'+self.binding['namespace']+'/'+o['name']+'?signed=1'for o in self.objects})
        self.request=r.sign(payload,self.root);self.request_bytes=r.canonical(self.request)
        def chunks(url):yield bytes([int(url.rsplit('/',1)[1][6:12])])*31
        self.receipt=r.execute(self.request,self.authority,self.reader,
            approved_binding=self.binding,approved_objects=self.objects,
            qualified_reader=self.identity,read_chunks=chunks,clock=lambda:150)
        filehash=hashlib.sha256(self.request_bytes).hexdigest()
        self.launch=r.sign(dict(version='independent-state-readback-launch-v1',
            request_sha256=filehash,module_sha256=hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest(),
            created_at=100,expires_at=300,max_wall_seconds=200),self.root)
        self.child=dict(pid=123,ticks='456')
        self.terminal=dict(actual_child_wait_completed=True,exit_code=0,timed_out=False,
            request_sha256=filehash,authority_publication_written=False,gpu_imports_requested=False,
            pid=123,ticks='456',started_at=149,finished_at=151)
        self.bucket=Mock();self.controller=SimpleNamespace(bucket=self.bucket,
            authority=SimpleNamespace(id=self.authority),signed=lambda v:r.sign(v,self.root))

    def run_commit(self):
        def read(bucket,key):
            if key.endswith('staged-state.json'):return self.descriptor
            return self.bucket.json.call_args.args[1]
        with patch('subnet.persistent_training_protocol.validate_report',return_value=self.descriptor), \
             patch('subnet.persistent_training_protocol.read_json',side_effect=read), \
             patch('subnet.persistent_training_protocol._publish_verified_descriptor')as publish:
            self.publish=publish
            result=independently_commit_remote(self.controller,self.report,self.job_env,
                self.request_bytes,self.receipt,self.launch,self.terminal,
                qualified_reader=self.identity,reader_host=self.host,trainer_host=self.trainer,
                storage_binding=self.storage,original_child=self.child,now=152)
            return result

    def test_full_request_and_actual_wait_persist_evidence_before_authority(self):
        self.run_commit();self.bucket.json.assert_called_once();self.publish.assert_called_once()
        self.assertIn('/independent-readbacks/',self.bucket.json.call_args.args[0])

    def test_receipt_without_original_success_never_publishes(self):
        self.terminal['exit_code']=1
        with self.assertRaises(ValueError):self.run_commit()
        self.bucket.json.assert_not_called();self.publish.assert_not_called()

    def test_same_actual_machine_different_host_record_is_rejected(self):
        self.trainer['provider_UUID']='reader'
        with self.assertRaises(ValueError):self.run_commit()
        self.bucket.json.assert_not_called();self.publish.assert_not_called()

    def test_original_job_substitution_is_rejected(self):
        self.job['job_id']='substituted';self.job_env=r.sign(self.job,self.root)
        with self.assertRaises(ValueError):self.run_commit()
        self.bucket.json.assert_not_called();self.publish.assert_not_called()

    def test_request_file_substitution_is_rejected(self):
        self.request_bytes+=b'\n'
        with self.assertRaises(ValueError):self.run_commit()
        self.bucket.json.assert_not_called();self.publish.assert_not_called()

    def test_original_pid_reuse_is_rejected(self):
        self.child['ticks']='999'
        with self.assertRaises(ValueError):self.run_commit()
        self.bucket.json.assert_not_called();self.publish.assert_not_called()

    def test_qualification_cannot_commit_production_authority(self):
        p=copy.deepcopy(self.receipt['payload']);p['purpose']='cpu-transport-qualification'
        self.receipt=r.sign(p,self.reader)
        with self.assertRaises(ValueError):self.run_commit()
        self.bucket.json.assert_not_called();self.publish.assert_not_called()
