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
from subnet.remote_state_commit import independently_commit_remote, readback_objects

class RemoteAdmission(unittest.TestCase):
    def setUp(self):
        self.root=SigningKey.generate();self.reader=SigningKey.generate()
        self.authority=bytes(self.root.verify_key).hex();self.identity=bytes(self.reader.verify_key).hex()
        self.objects=[dict(name='state-%06d.safetensors'%i,size=31,
            sha256=hashlib.sha256(bytes([i])*31).hexdigest())for i in range(23)]
        self.descriptor=dict(shards=[dict(o,tensors=[dict(parameter='actual.weight',slot='master',start=0,count=7,key='tensor-%08d'%i)])for i,o in enumerate(self.objects)])
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

    def run_commit(self,now=152):
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
                storage_binding=self.storage,original_child=self.child,now=now)
            return result

    def test_full_request_and_actual_wait_persist_evidence_before_authority(self):
        self.run_commit();self.bucket.json.assert_called_once();self.publish.assert_called_once()
        self.assertIn('/independent-readbacks/',self.bucket.json.call_args.args[0])

    def test_tensor_metadata_remains_in_full_descriptor_commitment(self):
        self.assertEqual(readback_objects(self.descriptor),self.objects)
        self.assertTrue(all('tensors' in s for s in self.descriptor['shards']))
        self.run_commit()
        self.assertEqual(self.publish.call_args.args[1],self.descriptor)
        self.assertEqual(self.receipt['payload']['descriptor_sha256'],r.sha(self.descriptor))
        changed=copy.deepcopy(self.descriptor);changed['shards'][0]['tensors'][0]['slot']='exp_avg'
        self.assertNotEqual(r.sha(changed),r.sha(self.descriptor))

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

    def test_late_observation_preserves_original_completion_and_actual_clock(self):
        self.run_commit(now=500)
        payload=self.bucket.json.call_args.args[1]['payload']
        self.assertEqual(payload['observed_at'],500)
        self.assertEqual(payload['receipt_validation_time'],150)
        self.assertEqual(payload['request']['payload']['expires_at'],300)
        self.publish.assert_called_once()
    def test_terminal_after_expiry_or_after_observation_rejected(self):
        for finish in (300,501):
            self.terminal['finished_at']=finish
            with self.assertRaises(ValueError):self.run_commit(now=500)
            self.bucket.json.assert_not_called();self.publish.assert_not_called()
    def test_completed_receipt_outside_ttl_is_not_historical_adoption(self):
        self.receipt=r.sign(dict(self.receipt['payload'],completed_at=301),self.reader)
        with self.assertRaises(ValueError):self.run_commit(now=500)
        self.bucket.json.assert_not_called();self.publish.assert_not_called()

    def use_stream_budget(self, value):
        self.manifest['independent_state_readback_budget']=value
        self.job['manifest']=r.sign(self.manifest,self.root)
        self.job_env=r.sign(self.job,self.root)
        self.binding.update(job_sha256=r.sha(self.job),provenance=dict(
            signed_job_envelope_sha256=r.sha(self.job_env),original_report_sha256=r.sha(self.report),
            signed_manifest_envelope_sha256=r.sha(self.job['manifest'])))
        payload=dict(self.request['payload'],**self.binding,stream_budget=value)
        self.request=r.sign(payload,self.root);self.request_bytes=r.canonical(self.request)
        filehash=hashlib.sha256(self.request_bytes).hexdigest()
        self.launch=r.sign(dict(self.launch['payload'],request_sha256=filehash),self.root)
        self.terminal['request_sha256']=filehash

    def test_eight_streams_read_all_original_objects_and_commit_with_matching_budget(self):
        import threading
        budget=dict(version=r.STREAM_BUDGET_VERSION,concurrency=8,ram_reserve_bytes=1024**3)
        self.use_stream_budget(budget)
        lock=threading.Lock();barrier=threading.Barrier(8,timeout=5)
        active=maximum=0;complete=set()
        def chunks(url):
            nonlocal active,maximum
            index=int(url.rsplit('/',1)[1][6:12])
            with lock:active+=1;maximum=max(active,maximum)
            try:
                if index<8:barrier.wait()
                yield bytes([index])*31
            finally:
                with lock:active-=1;complete.add(index)
        with patch.object(r,'available_ram_bytes',return_value=2*1024**3):
            self.receipt=r.execute(self.request,self.authority,self.reader,
                approved_binding=self.binding,approved_objects=self.objects,
                qualified_reader=self.identity,read_chunks=chunks,clock=lambda:150)
        self.assertEqual(maximum,8);self.assertEqual(complete,set(range(23)))
        self.assertEqual(self.receipt['payload']['concurrency'],8)
        self.assertEqual(self.receipt['payload']['objects'],self.objects)
        self.run_commit();self.publish.assert_called_once()

    def test_historical_request_keeps_exact_four_stream_receipt(self):
        self.assertNotIn('stream_budget',self.request['payload'])
        self.assertEqual(self.receipt['payload']['concurrency'],4)
        self.run_commit()

    def test_new_budget_requires_ram_admission_before_any_object_is_read(self):
        self.use_stream_budget(dict(version=r.STREAM_BUDGET_VERSION,concurrency=8,ram_reserve_bytes=1024**3))
        chunks=Mock()
        with patch.object(r,'available_ram_bytes',return_value=1024**3),self.assertRaisesRegex(ValueError,'RAM budget'):
            r.execute(self.request,self.authority,self.reader,approved_binding=self.binding,
                approved_objects=self.objects,qualified_reader=self.identity,read_chunks=chunks,clock=lambda:150)
        chunks.assert_not_called()

    def test_stream_budget_types_bounds_and_unknown_fields_rejected(self):
        good=dict(version=r.STREAM_BUDGET_VERSION,concurrency=8,ram_reserve_bytes=1024**3)
        for change in ({'concurrency':True},{'concurrency':8.0},{'concurrency':3},
                {'concurrency':9},{'ram_reserve_bytes':True},{'ram_reserve_bytes':0},
                {'ram_reserve_bytes':65*1024**3},{'version':'unknown'},{'extra':1}):
            with self.subTest(change=change),self.assertRaises(ValueError):r.stream_budget(dict(good,**change))

    def test_eight_stream_request_cannot_be_added_to_historical_manifest(self):
        payload=dict(self.request['payload'],stream_budget=dict(version=r.STREAM_BUDGET_VERSION,concurrency=8,ram_reserve_bytes=1024**3))
        self.request=r.sign(payload,self.root);self.request_bytes=r.canonical(self.request)
        with self.assertRaisesRegex(ValueError,'original manifest'):
            self.run_commit()
        self.bucket.json.assert_not_called()

    def test_four_stream_receipt_cannot_be_relabelled_as_eight(self):
        self.use_stream_budget(dict(version=r.STREAM_BUDGET_VERSION,concurrency=8,ram_reserve_bytes=1024**3))
        self.receipt=r.sign(dict(self.receipt['payload'],request_sha256=r.sha(self.request['payload']),
            signed_request_sha256=r.sha(self.request),job_sha256=r.sha(self.job),
            provenance=self.binding['provenance']),self.reader)
        with self.assertRaisesRegex(ValueError,'request-bound receipt'):self.run_commit()
        self.bucket.json.assert_not_called();self.publish.assert_not_called()

    def test_eight_stream_corruption_cannot_produce_success_receipt(self):
        self.use_stream_budget(dict(version=r.STREAM_BUDGET_VERSION,concurrency=8,ram_reserve_bytes=1024**3))
        def chunks(url):
            index=int(url.rsplit('/',1)[1][6:12]);yield b'x'*31 if index==22 else bytes([index])*31
        with patch.object(r,'available_ram_bytes',return_value=2*1024**3),self.assertRaisesRegex(ValueError,'integrity'):
            r.execute(self.request,self.authority,self.reader,approved_binding=self.binding,
                approved_objects=self.objects,qualified_reader=self.identity,read_chunks=chunks,clock=lambda:150)
