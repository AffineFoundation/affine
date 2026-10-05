import base64, copy, unittest
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.current_parent_optimizer_cache import fetcher, VERSION
from subnet.persistent_cpu_adamw import sha
from subnet.persistent_training_state import restore_state
from test_parallel_persistent_state import ParallelState

class CacheControl(ParallelState):
    def setup_cache(self):
        descriptor,_=self.export();cache=self.root/'cache';cache.mkdir()
        for name,data in self.objects.items():(cache/name).write_bytes(data)
        key=SigningKey.generate();self.authority=key.verify_key.encode().hex()
        policy=dict(version=VERSION,descriptor_sha256=sha(descriptor),checkpoint='22'*32,
            genesis_sha256=descriptor['genesis_sha256'],optimizer_steps=1,parameters_sha256=sha(self.inventory),
            authority_committed=True,independent_durable_readback_verified=True,
            retained_cache_bytes=sum(s['size'] for s in descriptor['shards']))
        def sign(p):return dict(payload=p,signer=self.authority,signature=base64.b64encode(key.sign(canonical(p)).signature).decode())
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
        for field,value in [('optimizer_steps',2),('checkpoint','33'*32),('authority_committed',False),('independent_durable_readback_verified',False)]:
            fields=copy.deepcopy(self.fields);policy=dict(self.policy,**{field:value});fields['document']=self.sign(policy)
            with self.assertRaises(ValueError):fetcher(cache,descriptor,self.remote_fetch,**fields)
    def test_missing_extra_disk_admission_rejected(self):
        descriptor,cache=self.setup_cache();self.fields['resource_admission']=self.admission
        with self.assertRaises(ValueError):fetcher(cache,descriptor,self.remote_fetch,**self.fields)
