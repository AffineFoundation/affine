import copy,json,os,unittest
from unittest.mock import patch
import torch
import test_optimizer_state_cache as fixtures
from subnet.optimizer_state_cache import StateCache,STAT_VERSION,STAT_VALIDATION,sha
from subnet.persistent_training_state import restore_state,admit_resources
from subnet.storage import canonical

class StatValidationControls(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.StateCacheControls();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture
        self.f.manifest['optimizer_state_local_cache'].update(version=STAT_VERSION,validation=STAT_VALIDATION)
        self.f.job['manifest']=self.f.sign(self.f.manifest)
    def test_real_restore_uses_prior_full_sha_fd_and_matches_optimizer_slots(self):
        f=self.f;descriptor=f.promoted();warm=f.root/'warm';warm.mkdir()
        with StateCache(f.root,f.job,f.manifest,f.authority)as cache:
            with patch('subnet.optimizer_state_cache.hash_file',side_effect=AssertionError('redundant prepare hash')):
                self.assertGreater(cache.prepare_parent(descriptor,'aa'*32),0)
            with patch('subnet.persistent_training_state._hash_file',side_effect=AssertionError('redundant restore hash')):
                restored,receipts=restore_state(descriptor,sha(descriptor),'22'*32,f.inventory,workspace=warm,fetch_shard=lambda n,p:cache.fetch(n,p,lambda *a: self.fail('network')),resource_admission=admit_resources(warm,f.plan),owned_cache=cache)
            self.assertTrue(all(not r['current_hash_performed'] for r in receipts))
            self.assertTrue(all(r['verification']=='prior-full-SHA-unchanged-owned-fd' for r in receipts))
            for slot in ('master','exp_avg','exp_avg_sq'):
                self.assertTrue(torch.equal(restored[1]['w'][slot],f.optimizer.rows['w'][slot]))
            from subnet.persistent_cpu_adamw import PersistentCPUAdamW
            left=[('w',torch.nn.Parameter(f.params[0][1].detach().clone()))]
            right=[('w',torch.nn.Parameter(f.params[0][1].detach().clone()))]
            baseline=(descriptor,copy.deepcopy(f.optimizer.rows),sha(descriptor))
            a=PersistentCPUAdamW(left,'22'*32,restored=baseline,resource_admission=f.admission)
            b=PersistentCPUAdamW(right,'22'*32,restored=restored,resource_admission=f.admission)
            left[0][1].grad=torch.ones_like(left[0][1]);right[0][1].grad=torch.ones_like(right[0][1])
            a.step();b.step();self.assertTrue(torch.equal(left[0][1],right[0][1]))
            for slot in ('master','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(a.rows['w'][slot],b.rows['w'][slot]))
    def test_missing_stamp_performs_full_hash(self):
        f=self.f;descriptor=f.promoted();p=f.root/'.optimizer-state-cache/current.json';v=json.loads(p.read_bytes());v.pop('promotion_verification_sha256');p.write_bytes(canonical(v))
        from subnet.optimizer_state_cache import hash_file
        with StateCache(f.root,f.job,f.manifest,f.authority)as cache,patch('subnet.optimizer_state_cache.hash_file',wraps=hash_file)as hashed:
            cache.prepare_parent(descriptor,'aa'*32);self.assertEqual(hashed.call_count,len(descriptor['shards']))
    def test_changed_same_size_file_stat_refuses_before_hash_skip(self):
        f=self.f;descriptor=f.promoted();p=next((f.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors'));data=p.read_bytes();p.write_bytes(data)
        with StateCache(f.root,f.job,f.manifest,f.authority)as cache:
            self.assertEqual(cache.prepare_parent(descriptor,'aa'*32),0)
            self.assertIsNone(cache.current)
    def test_unreleased_cache_lease_prevents_second_consumer(self):
        f=self.f;f.promoted()
        with StateCache(f.root,f.job,f.manifest,f.authority):
            with self.assertRaises(BlockingIOError):
                with StateCache(f.root,f.job,f.manifest,f.authority):pass
    def test_renamed_fd_path_replacement_and_forged_journal_refused(self):
        f=self.f;descriptor=f.promoted();shard=descriptor['shards'][0];p=f.root/'restored.safetensors'
        with StateCache(f.root,f.job,f.manifest,f.authority)as cache:
            cache.prepare_parent(descriptor,'aa'*32);receipt=cache.fetch(shard['name'],p,lambda *a:self.fail('network'))
            try:
                self.assertTrue(receipt.validate(p,shard,cache).startswith('/proc/self/fd/'))
                receipt.journal.write_bytes(b'{}')
                with self.assertRaises(ValueError):receipt.validate(p,shard,cache)
                p.unlink();p.write_bytes(f.objects[shard['name']])
                with self.assertRaises(ValueError):receipt.validate(p,shard,cache)
            finally:receipt.close()
    def test_closed_lease_refuses_receipt(self):
        f=self.f;descriptor=f.promoted();shard=descriptor['shards'][0];p=f.root/'restored.safetensors'
        with StateCache(f.root,f.job,f.manifest,f.authority)as cache:
            cache.prepare_parent(descriptor,'aa'*32);receipt=cache.fetch(shard['name'],p,lambda *a:self.fail('network'))
        try:
            with self.assertRaises(ValueError):receipt.validate(p,shard,cache)
        finally:receipt.close()
    def test_untyped_receipt_never_skips_digest(self):
        f=self.f;descriptor=f.promoted();warm=f.root/'warm';warm.mkdir()
        def fetch(n,p):p.write_bytes(f.objects[n]);return {'sha256':descriptor['shards'][0]['sha256']}
        with StateCache(f.root,f.job,f.manifest,f.authority)as cache,self.assertRaisesRegex(ValueError,'typed owned'):
            restore_state(descriptor,sha(descriptor),'22'*32,f.inventory,workspace=warm,fetch_shard=fetch,resource_admission=admit_resources(warm,f.plan),owned_cache=cache)

    def test_changed_descriptor_slot_shape_is_rejected_before_cache_read(self):
        f=self.f;descriptor=f.promoted();changed=copy.deepcopy(descriptor);changed['shards'][0]['tensors'][0]['count']+=1
        with self.assertRaises(ValueError):
            restore_state(changed,sha(descriptor),'22'*32,f.inventory,workspace=f.out,fetch_shard=lambda *a:self.fail('fetch before metadata binding'),resource_admission=f.admission)
    def test_file_changes_during_materialization_refuse_and_close_owned_fd(self):
        f=self.f;descriptor=f.promoted();warm=f.root/'warm';warm.mkdir();receipts=[]
        import subnet.persistent_training_state as module
        original=module.finite
        def fetch(n,p):
            receipt=cache.fetch(n,p,lambda *a:self.fail('network'));receipts.append(receipt);return receipt
        def mutate(torch,value,**kw):
            original(torch,value,**kw);os.chmod(receipts[-1].path,0o640)
        with StateCache(f.root,f.job,f.manifest,f.authority)as cache:
            cache.prepare_parent(descriptor,'aa'*32)
            with patch.object(module,'finite',side_effect=mutate),self.assertRaises(ValueError):
                restore_state(descriptor,sha(descriptor),'22'*32,f.inventory,workspace=warm,fetch_shard=fetch,resource_admission=admit_resources(warm,f.plan),owned_cache=cache)
        self.assertTrue(receipts);self.assertTrue(all(r.fd is None for r in receipts))
    def test_explicit_policy_requires_exact_version_and_validation(self):
        from subnet.optimizer_state_cache import policy
        from subnet.persistent_training_protocol import optimizer_cache_policy
        self.assertEqual(policy(self.f.manifest),optimizer_cache_policy(self.f.manifest))
        for updates in ({'version':'unknown'},{'validation':'trust-path'},{'max_checkpoint_bytes':True}):
            manifest=copy.deepcopy(self.f.manifest);manifest['optimizer_state_local_cache'].update(updates)
            with self.assertRaises(ValueError):policy(manifest)
            with self.assertRaises(ValueError):optimizer_cache_policy(manifest)
    def test_hardlink_and_symlink_cannot_admit_owned_file(self):
        f=self.f;descriptor=f.promoted();p=next((f.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors'));os.link(p,f.root/'foreign-link')
        with StateCache(f.root,f.job,f.manifest,f.authority)as cache:
            with self.assertRaises(ValueError):cache.prepare_parent(descriptor,'aa'*32)
    def test_prior_root_ack_signature_and_catalogue_are_required(self):
        f=self.f;descriptor=f.promoted();p=f.root/'.optimizer-state-cache/current.json';v=json.loads(p.read_bytes());v['ROOT_ack']['signature']='AAAA';p.write_bytes(canonical(v))
        with StateCache(f.root,f.job,f.manifest,f.authority)as cache:
            with self.assertRaises(ValueError):cache.prepare_parent(descriptor,'aa'*32)
