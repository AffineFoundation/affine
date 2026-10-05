import base64, copy, unittest
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.current_parent_optimizer_cache import fetcher, VERSION
from subnet.persistent_cpu_adamw import sha
from subnet.persistent_training_state import restore_state
from test_parallel_persistent_state import ParallelState

class CacheControl(ParallelState):
    def setup_cache(self):
        import torch
        from subnet.persistent_cpu_adamw import PersistentCPUAdamW, genesis, parameter_inventory
        self.params=[('w',torch.nn.Parameter(torch.arange(1,31,dtype=torch.bfloat16)))]
        _,self.inventory=parameter_inventory(self.params)
        from subnet.persistent_training_state import resource_plan,admit_resources
        self.admission=admit_resources(self.root,resource_plan(self.inventory,bf16_export_bytes=100,
            transfer_bytes=self.cap,disk_reserve_bytes=0,ram_reserve_bytes=0,concurrency=4))
        self.optimizer=PersistentCPUAdamW(self.params,'11'*32,approved_genesis=genesis(self.inventory,'11'*32),
            approved_genesis_sha256=sha(genesis(self.inventory,'11'*32)),resource_admission=self.admission)
        self.params[0][1].grad=torch.ones_like(self.params[0][1]);self.optimizer.step()
        from subnet.persistent_training_state import resource_plan,admit_resources
        self.admission=admit_resources(self.root,resource_plan(self.inventory,bf16_export_bytes=100,
            transfer_bytes=self.cap,disk_reserve_bytes=0,ram_reserve_bytes=0,concurrency=4))
        descriptor,_=self.export();cache=self.root/'cache';cache.mkdir()
        for name,data in self.objects.items():(cache/name).write_bytes(data)
        key=SigningKey.generate();self.authority=key.verify_key.encode().hex()
        policy=dict(version=VERSION,descriptor_sha256=sha(descriptor),checkpoint='22'*32,
            genesis_sha256=descriptor['genesis_sha256'],optimizer_steps=1,parameters_sha256=sha(self.inventory),
            authority_committed=True,independent_durable_readback_verified=True,
            retained_cache_bytes=sum(s['size'] for s in descriptor['shards']))
        def sign(p):return dict(payload=p,signer=self.authority,signature=base64.b64encode(key.sign(canonical(p)).signature).decode())
        from subnet import remote_optimizer_readback as r
        reader=SigningKey.generate();identity=reader.verify_key.encode().hex()
        objects=[{k:row[k]for k in ('name','size','sha256')}for row in descriptor['shards']]
        self.assertEqual(len(objects),23)
        binding=dict(purpose='production-training-state',provenance=dict(
            signed_job_envelope_sha256='aa'*32,original_report_sha256='bb'*32,signed_manifest_envelope_sha256='cc'*32),
            job_id='original',job_sha256='dd'*32,source_sha256='ee'*32,namespace='private/state/original',
            descriptor_sha256=sha(descriptor),reader_host_record_sha256='ab'*32,trainer_host_record_sha256='bc'*32,
            storage_origin='https://test.r2.cloudflarestorage.com',storage_bucket='test-bucket',storage_addressing='path')
        request=r.sign(dict(version=r.VERSION,**binding,reader_identity=identity,created_at=100,expires_at=300,
            max_wall_seconds=200,objects=objects,capabilities={o['name']:binding['storage_origin']+'/test-bucket/'+binding['namespace']+'/'+o['name']+'?signed=1'for o in objects}),key)
        receipt=r.execute(request,self.authority,reader,approved_binding=binding,approved_objects=objects,
            qualified_reader=identity,read_chunks=lambda url:iter([self.objects[url.split('?')[0].rsplit('/',1)[1]]]),clock=lambda:150)
        from subnet.persistent_training_protocol import PUBLICATION_VERSION
        adoption=sign(dict(version=PUBLICATION_VERSION,namespace=binding['namespace'],job_id=binding['job_id'],
            job_sha256=binding['job_sha256'],descriptor_sha256=sha(descriptor),descriptor=descriptor))
        policy.update(job_sha256=binding['job_sha256'],source_sha256=binding['source_sha256'],adoption=adoption,
            readback_request=request,readback_receipt=receipt,adoption_sha256=sha(adoption),readback_request_sha256=sha(request),
            readback_receipt_sha256=sha(receipt),approved_readback_binding=binding,approved_readback_objects=objects,
            qualified_reader=identity,original_evidence_observed_at=152)
        admission=dict(self.admission,current_parent_cache_bytes=policy['retained_cache_bytes'],
            cache_additional_disk_required_bytes=self.admission['additional_disk_required_bytes']+policy['retained_cache_bytes'])
        self.fields=dict(document=sign(policy),authority=self.authority,checkpoint='22'*32,inventory=self.inventory,resource_admission=admission)
        self.sign=sign;self.policy=policy;self.remote=[]
        def remote(name,path):self.remote.append(name);path.write_bytes(self.objects[name])
        self.remote_fetch=remote;return descriptor,cache
    def test_exact_local_state_and_counter(self):
        descriptor,cache=self.setup_cache();fetch,stats=fetcher(cache,descriptor,self.remote_fetch,**self.fields)
        restored,_=restore_state(descriptor,sha(descriptor),'22'*32,self.inventory,workspace=self.root,
            fetch_shard=fetch,resource_admission=self.admission)
        import torch
        for slot in ('master','exp_avg','exp_avg_sq'):
            self.assertTrue(torch.equal(restored[1]['w'][slot],self.optimizer.rows['w'][slot]))
        self.assertEqual(restored[1]['w']['step'],1);self.assertFalse(self.remote)
        self.assertEqual(stats['local_verified_bytes'],self.policy['retained_cache_bytes'])
    def test_partial_and_corrupt_exact_remote_fallback(self):
        descriptor,cache=self.setup_cache();names=list(self.objects)
        (cache/names[0]).unlink();(cache/names[1]).write_bytes(b'bad')
        fetch,stats=fetcher(cache,descriptor,self.remote_fetch,**self.fields)
        for name in names:fetch(name,self.root/'download');self.assertEqual((self.root/'download').read_bytes(),self.objects[name])
        self.assertEqual(self.remote,names[:2]);self.assertGreater(stats['remote_requested_bytes'],0)
    def test_wrong_parent_and_uncommitted_rejected(self):
        descriptor,cache=self.setup_cache()
        for field,value in [('optimizer_steps',2),('checkpoint','33'*32),('source_sha256','ff'*32),('qualified_reader','ff'*32)]:
            fields=copy.deepcopy(self.fields);policy=dict(self.policy,**{field:value});fields['document']=self.sign(policy)
            with self.assertRaises(ValueError):fetcher(cache,descriptor,self.remote_fetch,**fields)
    def test_missing_extra_disk_admission_rejected(self):
        descriptor,cache=self.setup_cache();self.fields['resource_admission']=self.admission
        with self.assertRaises(ValueError):fetcher(cache,descriptor,self.remote_fetch,**self.fields)

class PendingControl(CacheControl):
    def test_pending_export_interruption_not_current(self):
        from subnet.current_parent_optimizer_cache import pending_retention
        descriptor,cache=self.setup_cache();total=sum(s['size']for s in descriptor['shards'])
        policy=dict(version='retain-pending-parent-cache-v1',job_sha256='ab'*32,current_bytes=total,
            pending_bytes=total,fresh_disk_bytes=100,total_disk_required_bytes=total*2+100)
        pending,wrap,retained=pending_retention(cache,self.sign(policy),self.authority,
            current_bytes=total,pending_bytes=total,fresh_disk_bytes=100)
        name=descriptor['shards'][0]['name'];source=cache/name
        def fail(name,path):raise ConnectionError('interrupted actual publication')
        with self.assertRaises(ConnectionError):wrap(fail)(name,source)
        self.assertTrue((pending/name).exists());self.assertFalse((cache/'current').exists())
        self.assertEqual(retained[name]['sha256'],descriptor['shards'][0]['sha256'])
        with self.assertRaises(FileExistsError):pending_retention(cache,self.sign(policy),self.authority,
            current_bytes=total,pending_bytes=total,fresh_disk_bytes=100)
    def test_pending_disk_underestimate_rejected(self):
        from subnet.current_parent_optimizer_cache import pending_retention
        _,cache=self.setup_cache()
        policy=dict(version='retain-pending-parent-cache-v1',job_sha256='ab'*32,current_bytes=90,
            pending_bytes=90,fresh_disk_bytes=100,total_disk_required_bytes=190)
        with self.assertRaisesRegex(ValueError,'filesystem reserve'):
            pending_retention(cache,self.sign(policy),self.authority,current_bytes=90,pending_bytes=90,fresh_disk_bytes=100)
    def test_wrong_original_adoption_digest_rejected(self):
        from subnet.current_parent_optimizer_cache import promote_pending
        _,cache=self.setup_cache()
        policy=dict(version='promote-current-parent-cache-v1',adoption_sha256='00'*32)
        with self.assertRaisesRegex(ValueError,'original evidence digest'):
            promote_pending(cache,signed_policy=self.sign(policy),authority=self.authority,
                adoption=self.sign({}),readback_request={},readback_receipt={},now=1)

class PromotionControl(CacheControl):
    def prepare_promotion(self):
        from subnet.current_parent_optimizer_cache import pending_retention
        descriptor,cache=self.setup_cache();total=self.policy['retained_cache_bytes']
        retention=dict(version='retain-pending-parent-cache-v1',job_sha256=self.policy['job_sha256'],
            current_bytes=total,pending_bytes=total,fresh_disk_bytes=100,total_disk_required_bytes=2*total+100)
        pending,wrap,_=pending_retention(cache,self.sign(retention),self.authority,
            current_bytes=total,pending_bytes=total,fresh_disk_bytes=100)
        for row in descriptor['shards']:wrap(lambda name,path:None)(row['name'],cache/row['name'])
        policy=dict(self.policy,version='promote-current-parent-cache-v1',inventory=self.inventory)
        return descriptor,cache,pending,policy
    def promote(self,pending,policy):
        from subnet.current_parent_optimizer_cache import promote_pending
        return promote_pending(pending,signed_policy=self.sign(policy),authority=self.authority,
            adoption=policy['adoption'],readback_request=policy['readback_request'],readback_receipt=policy['readback_receipt'],now=152)
    def test_actual_23_shard_reader_receipt_promotes(self):
        descriptor,cache,pending,policy=self.prepare_promotion();current=self.promote(pending,policy)
        self.assertEqual(current,cache/'current');self.assertFalse(pending.exists())
        self.assertEqual({p.name:p.read_bytes()for p in current.iterdir()},self.objects)
    def test_partial_corrupt_and_collision_fail_closed(self):
        _,cache,pending,policy=self.prepare_promotion();name=next(iter(self.objects));path=pending/name
        original=path.read_bytes();path.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'full hash'):self.promote(pending,policy)
        path.unlink()
        with self.assertRaisesRegex(ValueError,'partial'):self.promote(pending,policy)
        path.write_bytes(original);(cache/'current').mkdir()
        with self.assertRaisesRegex(ValueError,'collision'):self.promote(pending,policy)
    def test_wrong_reader_source_model_genesis_counter(self):
        _,_,pending,policy=self.prepare_promotion()
        for field,value in [('qualified_reader','ff'*32),('source_sha256','ff'*32),('checkpoint','ff'*32),
                            ('genesis_sha256','ff'*32),('optimizer_steps',2)]:
            with self.subTest(field=field),self.assertRaises(Exception):self.promote(pending,dict(policy,**{field:value}))
    def test_real_parallel_export_wrapper_preserves_descriptor_exactly(self):
        from subnet.current_parent_optimizer_cache import pending_retention
        descriptor,cache=self.setup_cache();total=self.policy['retained_cache_bytes']
        retention=dict(version='retain-pending-parent-cache-v1',job_sha256=self.policy['job_sha256'],
            current_bytes=total,pending_bytes=total,fresh_disk_bytes=100,total_disk_required_bytes=2*total+100)
        pending,wrap,retained=pending_retention(cache,self.sign(retention),self.authority,
            current_bytes=total,pending_bytes=total,fresh_disk_bytes=100)
        second,evidence=self.export(publish_shard=wrap(self.publish))
        self.assertEqual(second,descriptor);self.assertEqual(len(retained),23)
        self.assertEqual({p.name:p.read_bytes()for p in pending.iterdir()},self.objects)
        self.assertTrue(evidence['descriptor_committed_last'])
