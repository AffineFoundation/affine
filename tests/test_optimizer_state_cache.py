import base64,copy,hashlib,json,tempfile,unittest
from pathlib import Path
import torch
from nacl.signing import SigningKey
from subnet.optimizer_state_cache import StateCache,VERSION,policy,promote,sha
from subnet.storage import canonical
from subnet.persistent_cpu_adamw import PersistentCPUAdamW,genesis,parameter_inventory
from subnet.persistent_training_state import resource_plan,admit_resources,export_state,restore_state,HEADER_RESERVE

class StateCacheControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        self.manifest=dict(source_bundle=dict(sha256='aa'*32),optimizer_state_local_cache=dict(version=VERSION,max_checkpoint_bytes=10*1024**2),persistent_publication_policy=dict(state_readback='qualified-remote-full'))
        self.job=dict(job_id='original',role='train',manifest=self.sign(self.manifest),persistent_training=dict(output_shards={'state-000000.safetensors':{}}))
        self.params=[('w',torch.nn.Parameter(torch.arange(1,5,dtype=torch.bfloat16)))];_,self.inventory=parameter_inventory(self.params)
        self.cap=HEADER_RESERVE+1024;self.plan=resource_plan(self.inventory,bf16_export_bytes=100,transfer_bytes=self.cap,disk_reserve_bytes=0,ram_reserve_bytes=0,concurrency=1);self.admission=admit_resources(self.root,self.plan)
        initial=genesis(self.inventory,'11'*32);self.optimizer=PersistentCPUAdamW(self.params,'11'*32,approved_genesis=initial,approved_genesis_sha256=sha(initial),resource_admission=self.admission)
        self.params[0][1].grad=torch.ones_like(self.params[0][1]);self.optimizer.step();self.objects={};self.out=self.root/'jobs'/'original';self.out.mkdir(parents=True)
    def sign(self,v):return dict(payload=v,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(v)).signature).decode())
    def publish(self,name,path):self.objects[name]=path.read_bytes()
    def readback(self,name):yield self.objects[name]
    def candidate(self):
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            cache.begin_candidate()
            descriptor,evidence=export_state(self.optimizer,epoch='control',inference_checkpoint='22'*32,workspace=self.out,publish_shard=self.publish,readback_shard=self.readback,commit_descriptor=lambda d:dict(descriptor_sha256=sha(d),durable_readback_verified=True,authority_committed=False),resource_admission=self.admission,shard_bytes=self.cap,retain_shard=cache.retain,readback_mode='upload-only-independent-full-v1')
            self.assertFalse(evidence['shards'][0]['local_shard_retired']);cache.finish(descriptor)
        state=dict(descriptor=descriptor,descriptor_sha256=sha(descriptor),namespace='private/state/control')
        report=dict(success=True,new_checkpoint=dict(id='22'*32),persistent_training_state=state);(self.out/'report.json').write_bytes(canonical(report));(self.root/'original.json').write_bytes(canonical(self.sign(self.job)))
        ack=dict(version='durable-original-trainer-cache-ACK-v1',job_id='original',job_sha256=sha(self.job),report_sha256=sha(report),new_checkpoint=report['new_checkpoint'],authority_state_committed=True,trainer_state=dict(descriptor_sha256=sha(descriptor),optimizer_steps=descriptor['optimizer_steps'],namespace=state['namespace']))
        self.descriptor=descriptor;self.ack=self.sign(ack);return descriptor
    def promoted(self):
        descriptor=self.candidate()
        # Independent reader authenticates all actual uploaded bytes before
        # the operator signs the durability/lineage acknowledgement.
        for row in descriptor['shards']:
            self.assertEqual(hashlib.sha256(self.objects[row['name']]).hexdigest(),row['sha256']);self.assertEqual(len(self.objects[row['name']]),row['size'])
        result=promote(self.ack,self.authority,self.root);self.assertTrue(result['promoted']);return descriptor
    def test_default_off_explicit_independent_policy_and_caps(self):
        self.assertIsNone(policy({}));v=copy.deepcopy(self.manifest);v['optimizer_state_local_cache']['max_checkpoint_bytes']=0
        with self.assertRaises(ValueError):policy(v)
        v=copy.deepcopy(self.manifest);v['persistent_publication_policy']['state_readback']='local-full'
        with self.assertRaises(ValueError):policy(v)
    def test_unpromoted_candidate_never_used_even_after_successful_upload(self):
        descriptor=self.candidate()
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            with self.assertRaisesRegex(ValueError,'recover completion'):cache.prepare_parent(descriptor,'aa'*32)
        self.assertFalse((self.root/'.optimizer-state-cache/current.json').exists())
    def test_real_optimizer_restore_from_promoted_cache_matches_cold_and_consumes_bytes(self):
        descriptor=self.promoted();cold=self.root/'cold';cold.mkdir();cold_admission=admit_resources(cold,self.plan)
        def download(name,path):path.write_bytes(self.objects[name])
        original,evidence=restore_state(descriptor,sha(descriptor),'22'*32,self.inventory,workspace=cold,fetch_shard=download,resource_admission=cold_admission)
        warm=self.root/'warm';warm.mkdir();warm_admission=admit_resources(warm,self.plan)
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            self.assertGreater(cache.prepare_parent(descriptor,'aa'*32),0)
            def unexpected(*args):raise AssertionError('no parent R2 download should be required')
            restored,evidence=restore_state(descriptor,sha(descriptor),'22'*32,self.inventory,workspace=warm,fetch_shard=lambda n,p:cache.fetch(n,p,unexpected),resource_admission=warm_admission)
        for name in original[1]:
            for slot in ('master','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(original[1][name][slot],restored[1][name][slot]))
            self.assertEqual(original[1][name]['step'],restored[1][name]['step'])
        self.assertFalse(list((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors')))
        # Exact restored AdamW state produces the same next real optimizer step.
        initial=[('w',torch.nn.Parameter(self.params[0][1].detach().clone()))];other=[('w',torch.nn.Parameter(self.params[0][1].detach().clone()))]
        left=PersistentCPUAdamW(initial,'22'*32,restored=original,resource_admission=cold_admission);right=PersistentCPUAdamW(other,'22'*32,restored=restored,resource_admission=warm_admission)
        initial[0][1].grad=torch.ones_like(initial[0][1]);other[0][1].grad=torch.ones_like(other[0][1]);left.step();right.step();self.assertTrue(torch.equal(initial[0][1],other[0][1]))
    def test_corrupt_or_missing_owned_cache_falls_back_without_optimizer_reset(self):
        descriptor=self.promoted();directory=self.root/'.optimizer-state-cache/candidate-original';file=next(directory.glob('*.safetensors'));file.write_bytes(b'corrupt')
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            self.assertEqual(cache.prepare_parent(descriptor,'aa'*32),0);calls=[];target=self.out/'fallback';cache.fetch(descriptor['shards'][0]['name'],target,lambda n,p:(calls.append(n),p.write_bytes(self.objects[n])));self.assertEqual(calls,[descriptor['shards'][0]['name']]);self.assertEqual(target.read_bytes(),self.objects[calls[0]])
    def test_wrong_parent_or_source_uses_cold_transport(self):
        descriptor=self.promoted()
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:self.assertEqual(cache.prepare_parent(descriptor,'bb'*32),0)
        self.assertFalse(list((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors')))
    def test_wrong_or_uncommitted_ROOT_ACK_cannot_promote(self):
        self.candidate();value=copy.deepcopy(self.ack['payload']);value['authority_state_committed']=False
        with self.assertRaises(ValueError):promote(self.sign(value),self.authority,self.root)
        value=copy.deepcopy(self.ack['payload']);value['trainer_state']['optimizer_steps']+=1
        with self.assertRaises(ValueError):promote(self.sign(value),self.authority,self.root)
        self.assertFalse((self.root/'.optimizer-state-cache/current.json').exists())
    def test_live_cache_lease_symlinks_replaced_inodes_are_protected(self):
        descriptor=self.promoted()
        with StateCache(self.root,self.job,self.manifest,self.authority):
            with self.assertRaises(BlockingIOError):
                with StateCache(self.root,self.job,self.manifest,self.authority):pass
        file=next((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors'));file.unlink();file.symlink_to(self.out/'report.json')
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            with self.assertRaises(ValueError):cache.prepare_parent(descriptor,'aa'*32)
        self.assertTrue((self.out/'report.json').exists())
    def test_retained_state_plus_transfer_export_and_reserve_budget_is_enforced(self):
        descriptor=self.promoted()
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            verified=cache.prepare_parent(descriptor,'aa'*32);desired=self.plan['cpu_state_bytes']+1024**2;required=self.plan['additional_disk_required_bytes']+desired
            with self.assertRaises(ValueError):cache.admit(self.plan,reclaimable_parent_bytes=verified,disk_available=required-verified-1)
            budget=cache.admit(self.plan,reclaimable_parent_bytes=verified,disk_available=required-verified);self.assertEqual(budget['ordinary_disk_required_bytes'],self.plan['additional_disk_required_bytes'])
    def test_changed_candidate_or_incomplete_inventory_cannot_promote(self):
        self.candidate();file=next((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors'));file.write_bytes(b'bad')
        with self.assertRaises(ValueError):promote(self.ack,self.authority,self.root)
        self.assertFalse((self.root/'.optimizer-state-cache/current.json').exists())
    def test_missing_promoted_shard_uses_cold_transport_and_keeps_durable_objects(self):
        descriptor=self.promoted();next((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors')).unlink()
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:self.assertEqual(cache.prepare_parent(descriptor,'aa'*32),0)
        self.assertEqual(set(self.objects),{s['name']for s in descriptor['shards']})
    def test_unauthenticated_reclaim_credit_and_tampered_source_are_rejected(self):
        descriptor=self.promoted()
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            with self.assertRaises(ValueError):cache.admit(self.plan,reclaimable_parent_bytes=10**12)
        marker=self.root/'.optimizer-state-cache/current.json';value=json.loads(marker.read_bytes());value['source_sha256']='bb'*32;marker.write_bytes(canonical(value))
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            with self.assertRaisesRegex(ValueError,'signed source'):cache.prepare_parent(descriptor,'bb'*32)
    def test_new_original_job_retires_abandoned_unpromoted_candidate_before_capacity(self):
        descriptor=self.candidate();next_job=dict(self.job,job_id='next-original')
        descriptor=copy.deepcopy(descriptor);descriptor['optimizer_steps']+=1
        with StateCache(self.root,next_job,self.manifest,self.authority)as cache:
            self.assertEqual(cache.prepare_parent(descriptor,'aa'*32),0)
            self.assertFalse(list((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors')))
            self.assertFalse((self.root/'.optimizer-state-cache/pending.json').exists())
        self.assertEqual(set(self.objects),{s['name']for s in descriptor['shards']})
    def test_stale_parent_cannot_discard_newer_committed_state(self):
        descriptor=self.promoted();old=copy.deepcopy(descriptor);old['optimizer_steps']-=1
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            with self.assertRaisesRegex(ValueError,'lineage backward'):cache.prepare_parent(old,'aa'*32)
        self.assertTrue(list((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors')))
if __name__=='__main__':unittest.main()
