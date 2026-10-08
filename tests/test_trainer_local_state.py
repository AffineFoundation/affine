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


if __name__=='__main__':unittest.main()
