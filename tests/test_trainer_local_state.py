"""Real Adam-state round trip with no optimizer network transport."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from test_optimizer_state_cache import StateCacheControls
from subnet.optimizer_state_cache import StateCache, promote, sha
from subnet.persistent_training_state import export_state, restore_state, admit_resources
from subnet.persistent_publication import LOCAL_POLICY, VERSION, complete, export_policy
from subnet.storage import canonical


class LocalStateControls(unittest.TestCase):
    setUp = StateCacheControls.setUp
    sign = StateCacheControls.sign

    def local_manifest(self):
        self.manifest.update(optimizer_state_export_policy=LOCAL_POLICY,
            persistent_publication_policy=dict(version=VERSION,state_readback='trainer-local',checkpoint_readback_workers=4))
        self.job['manifest']=self.sign(self.manifest)

    def candidate(self):
        self.local_manifest()
        no_network=Mock(side_effect=AssertionError('optimizer network transport forbidden'))
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            cache.begin_candidate()
            descriptor,evidence=export_state(self.optimizer,epoch='control',inference_checkpoint='22'*32,
                workspace=self.out,publish_shard=no_network,readback_shard=no_network,
                commit_descriptor=lambda d:dict(descriptor_sha256=sha(d),local_sha_verified=True,
                    durable_readback_verified=False,authority_committed=False),resource_admission=self.admission,
                shard_bytes=self.cap,retain_shard=cache.retain,readback_mode=LOCAL_POLICY)
            candidate=cache.finish(descriptor)
        no_network.assert_not_called()
        state=dict(descriptor=descriptor,descriptor_sha256=sha(descriptor),namespace='private/state/control',
                   publication_evidence=evidence,local_optimizer_cache_candidate=candidate)
        report=dict(success=True,new_checkpoint=dict(id='22'*32),persistent_training_state=state)
        (self.out/'report.json').write_bytes(canonical(report));(self.root/'original.json').write_bytes(canonical(self.sign(self.job)))
        ack=dict(version='durable-original-trainer-cache-ACK-v1',job_id='original',job_sha256=sha(self.job),
            report_sha256=sha(report),new_checkpoint=report['new_checkpoint'],authority_state_committed=True,
            trainer_state=dict(descriptor_sha256=sha(descriptor),optimizer_steps=descriptor['optimizer_steps'],namespace=state['namespace']))
        self.report=report;self.ack=self.sign(ack)
        return descriptor,evidence

    def test_local_save_has_zero_optimizer_network_io(self):
        descriptor,evidence=self.candidate()
        self.assertFalse(evidence['optimizer_state_uploaded'])
        self.assertFalse(evidence['independent_full_readback_required'])
        self.assertTrue(all(r['upload_completed']is False and r['local_sha_verified']is True for r in evidence['shards']))
        self.assertTrue(list((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors')))

    def test_local_promotion_and_restore_preserve_real_adam_state_across_source_upgrade(self):
        descriptor,_=self.candidate();promote(self.ack,self.authority,self.root)
        out=self.root/'next';out.mkdir();admission=admit_resources(out,self.plan)
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            self.assertGreater(cache.prepare_parent(descriptor,'bb'*32),0)
            no_network=Mock(side_effect=AssertionError('missing local state must not fetch'))
            state,_=restore_state(descriptor,sha(descriptor),'22'*32,self.inventory,workspace=out,
                fetch_shard=lambda n,p:cache.fetch(n,p,no_network),resource_admission=admission)
            no_network.assert_not_called()
        import torch
        for slot in ('master','exp_avg','exp_avg_sq'):
            self.assertTrue(torch.equal(state[1]['w'][slot],self.optimizer.rows['w'][slot]))
        self.assertEqual(state[1]['w']['step'],self.optimizer.global_step)

    def test_corrupt_local_parent_is_not_deleted_or_silently_reset(self):
        descriptor,_=self.candidate();promote(self.ack,self.authority,self.root)
        p=next((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors'));p.write_bytes(b'corrupt')
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            with self.assertRaisesRegex(ValueError,'required local optimizer'):cache.prepare_parent(descriptor,'aa'*32)
        self.assertTrue(p.exists());self.assertTrue((self.root/'.optimizer-state-cache/current.json').exists())

    def test_local_state_requires_retention_and_explicit_policy(self):
        self.local_manifest();self.assertEqual(export_policy(self.manifest),LOCAL_POLICY)
        for change in ({'optimizer_state_local_cache':None},
                       {'persistent_publication_policy':dict(version=VERSION,state_readback='qualified-remote-full',checkpoint_readback_workers=4)}):
            with self.assertRaises(ValueError):export_policy(dict(self.manifest,**change))

    def test_model_only_publication_never_calls_independent_optimizer_reader(self):
        self.local_manifest();controller=SimpleNamespace(publish_remote_checkpoint=Mock(return_value={'id':'22'*32}),
            independent_state_reader=Mock(side_effect=AssertionError('no optimizer reader')))
        report=dict(new_checkpoint=dict(id='22'*32,files={},path='/original'),persistent_training_state=dict(descriptor={}))
        job=dict(persistent_training=dict(output_namespace='private/state/control'))
        with patch('subnet.persistent_training_protocol.validate_report'), \
             patch('subnet.persistent_training_protocol._publish_verified_descriptor',return_value={'optimizer_steps':1})as commit, \
             patch('subnet.remote_state_commit.independently_commit_remote',side_effect=AssertionError('no remote state')):
            _,_,timing=complete(controller,report,job,self.manifest,'/original')
        self.assertFalse(timing['optimizer_state_uploaded']);self.assertTrue(commit.call_args.kwargs['local_only'])
        controller.publish_remote_checkpoint.assert_called_once();controller.independent_state_reader.assert_not_called()


class LocalOwnedParentControls(unittest.TestCase):
    sign=LocalStateControls.sign
    local_manifest=LocalStateControls.local_manifest
    candidate=LocalStateControls.candidate
    def setUp(self):
        StateCacheControls.setUp(self)
        from subnet.optimizer_state_cache import STAT_VERSION,STAT_VALIDATION
        self.manifest['optimizer_state_local_cache'].update(version=STAT_VERSION,validation=STAT_VALIDATION)
    def test_separate_volume_checks_model_disk_and_optimizer_capacity_independently(self):
        descriptor,_=self.candidate();promote(self.ack,self.authority,self.root)
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            parent=cache.prepare_parent(descriptor,'bb'*32)
            model_disk=self.plan['additional_disk_required_bytes']
            desired=self.plan['cpu_state_bytes']+1024**2
            needed=desired+self.plan['bounded_inflight_transfer_bytes']+self.plan['disk_reserve_bytes']
            def stat_device(path,*args,**kwargs):return SimpleNamespace(st_dev=2 if path==cache.root else 1)
            with patch.object(cache,'next_volume',return_value=None),patch('pathlib.Path.stat',stat_device),patch('subnet.optimizer_state_cache.shutil.disk_usage',return_value=SimpleNamespace(free=needed)):
                result=cache.admit(self.plan,reclaimable_parent_bytes=parent,disk_available=model_disk)
                self.assertEqual(result['reclaimable_verified_parent_bytes'],0)
                with self.assertRaisesRegex(ValueError,'separate local optimizer'):
                    cache.admit(self.plan,reclaimable_parent_bytes=parent,disk_available=model_disk-1)
            with patch.object(cache,'next_volume',return_value=None),patch('pathlib.Path.stat',stat_device),patch('subnet.optimizer_state_cache.shutil.disk_usage',return_value=SimpleNamespace(free=needed-1)):
                with self.assertRaisesRegex(ValueError,'separate local optimizer'):
                    cache.admit(self.plan,reclaimable_parent_bytes=parent,disk_available=model_disk)

    def test_owned_restore_preserves_parent_for_retry_without_hash_copy_or_network(self):
        descriptor,_=self.candidate();promote(self.ack,self.authority,self.root)
        parent=self.root/'.optimizer-state-cache/candidate-original'
        original={p.name:p.read_bytes()for p in parent.glob('*.safetensors')}
        for attempt in range(2):
            out=self.root/('retry-'+str(attempt));out.mkdir();admission=admit_resources(out,self.plan)
            with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
                cache.prepare_parent(descriptor,'bb'*32)
                no_network=Mock(side_effect=AssertionError('no parent network'))
                with patch('subnet.persistent_training_state._hash_file',side_effect=AssertionError('no repeated parent hash')):
                    state,evidence=restore_state(descriptor,sha(descriptor),'22'*32,self.inventory,
                        workspace=out,fetch_shard=lambda n,p:cache.fetch(n,p,no_network),
                        resource_admission=admission,owned_cache=cache)
                no_network.assert_not_called()
            self.assertTrue(all(r['local_shard_retired']is False for r in evidence))
            self.assertEqual({p.name:p.read_bytes()for p in parent.glob('*.safetensors')},original)
            import torch
            self.assertTrue(torch.equal(state[1]['w']['master'],self.optimizer.rows['w']['master']))
    def test_retained_parent_bytes_are_not_credited_as_free_disk(self):
        descriptor,_=self.candidate();promote(self.ack,self.authority,self.root)
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            parent_bytes=cache.prepare_parent(descriptor,'bb'*32)
            required=self.plan['cpu_state_bytes']+1024**2+self.plan['additional_disk_required_bytes']
            with self.assertRaisesRegex(ValueError,'disk budget'):
                cache.admit(self.plan,reclaimable_parent_bytes=parent_bytes,disk_available=required-1)
            result=cache.admit(self.plan,reclaimable_parent_bytes=parent_bytes,disk_available=required)
            self.assertEqual(result['reclaimable_verified_parent_bytes'],0)


if __name__=='__main__':unittest.main()
