import base64,copy,json,os,unittest,hashlib
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from test_trainer_local_state import LocalStateControls
from subnet.optimizer_state_cache import promote,sha,STAT_VERSION,STAT_VALIDATION
from subnet.storage import canonical
from ops.trainer_local_cache_volume import authenticated_current
from subnet.optimizer_state_cache import StateCache,VOLUME_VERSION,volume_policy

class LocalVolumeControls(unittest.TestCase):
    def setUp(self):
        self.fixture=LocalStateControls();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.fixture.manifest['optimizer_state_local_cache'].update(version=STAT_VERSION,validation=STAT_VALIDATION)
        self.fixture.candidate();self.fixture.report['job_id']='original'
        (self.fixture.out/'report.json').write_bytes(canonical(self.fixture.report))
        value=copy.deepcopy(self.fixture.ack['payload']);value['report_sha256']=sha(self.fixture.report)
        self.fixture.ack=self.fixture.sign(value);promote(self.fixture.ack,self.fixture.authority,self.fixture.root)
        self.root=self.fixture.root/'.optimizer-state-cache'
        for p in [self.fixture.out/'report.json',self.fixture.root/'original.json']:p.chmod(0o600)
    def test_authenticates_exact_local_lineage_without_state_hash_pass(self):
        current,descriptor=authenticated_current(self.fixture.root,self.root,self.fixture.authority)
        self.assertEqual(descriptor['optimizer_steps'],1)
        self.assertEqual(set(current['files']),{r['name']for r in descriptor['shards']})
    def test_stat_promotion_marker_is_checked_even_with_legacy_candidate_version(self):
        p=self.root/'current.json';current=json.loads(p.read_bytes())
        self.assertNotEqual(current['version'],STAT_VERSION)
        current['promotion_verification_sha256']='00'*32;p.write_bytes(canonical(current))
        with self.assertRaisesRegex(ValueError,'original optimizer promotion verification'):
            authenticated_current(self.fixture.root,self.root,self.fixture.authority)
    def test_modified_parent_inode_is_rejected(self):
        current,_=authenticated_current(self.fixture.root,self.root,self.fixture.authority)
        p=self.root/('candidate-'+current['job_id'])/next(iter(current['files']))
        p.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'unchanged acknowledged'):
            authenticated_current(self.fixture.root,self.root,self.fixture.authority)
    def enable_volume(self):
        volume=Path('/dev/shm')/('affine-optimizer-'+hashlib.sha256(str(self.fixture.root).encode()).hexdigest()[:20])
        volume.mkdir(mode=0o700);self.addCleanup(volume.rmdir)
        marker=self.root/'local-volume.json'
        marker.write_bytes(canonical(dict(version=VOLUME_VERSION,workspace=str(self.fixture.root),memory_root=str(volume))))
        marker.chmod(0o600);return volume
    def test_unowned_volume_substitution_rejected(self):
        self.enable_volume();p=self.root/'local-volume.json';value=json.loads(p.read_bytes());value['memory_root']='/tmp/external';p.write_bytes(canonical(value))
        with self.assertRaisesRegex(ValueError,'exact owned trainer'):
            volume_policy(self.fixture.root,self.root)
    def test_current_disk_parent_selects_memory_candidate_and_bounds_ram(self):
        volume=self.enable_volume();descriptor=self.fixture.report['persistent_training_state']['descriptor']
        with StateCache(self.fixture.root,self.fixture.job,self.fixture.manifest,self.fixture.authority)as cache:
            parent=cache.prepare_parent(descriptor,'bb'*32);self.assertEqual(cache.next_volume(),volume)
            required=self.fixture.plan['cpu_state_bytes']+1024**2+self.fixture.plan['cpu_additional_ram_required_bytes']
            with patch('subnet.persistent_training_state.available_ram_bytes',return_value=required):
                result=cache.admit(self.fixture.plan,reclaimable_parent_bytes=parent,disk_available=self.fixture.plan['additional_disk_required_bytes'])
                self.assertEqual(result['optimizer_volume'],'memory')
            with patch('subnet.persistent_training_state.available_ram_bytes',return_value=required-1):
                with self.assertRaisesRegex(ValueError,'alternating optimizer memory'):
                    cache.admit(self.fixture.plan,reclaimable_parent_bytes=parent,disk_available=self.fixture.plan['additional_disk_required_bytes'])
            parent_dir=cache.directory(cache.current['job_id']);original=Path.stat
            def device(path,*a,**kw):return SimpleNamespace(st_dev=volume.stat().st_dev)if path==parent_dir else original(path,*a,**kw)
            with patch('pathlib.Path.stat',device):self.assertIsNone(cache.next_volume())

    def test_direct_memory_candidate_exports_and_retires_without_mount(self):
        from subnet.persistent_training_state import export_state
        from subnet.persistent_publication import LOCAL_POLICY
        from unittest.mock import Mock
        volume=self.enable_volume()
        job=copy.deepcopy(self.fixture.job);job['job_id']='memory-next'
        descriptor=self.fixture.report['persistent_training_state']['descriptor']
        no_network=Mock(side_effect=AssertionError('optimizer transport forbidden'))
        with StateCache(self.fixture.root,job,self.fixture.manifest,self.fixture.authority)as cache:
            cache.prepare_parent(descriptor,'bb'*32)
            with patch('subnet.optimizer_state_cache.subprocess.run',side_effect=AssertionError('mount forbidden')):
                cache.begin_candidate()
                destination=cache.directory(job['job_id'])
                self.assertEqual(destination,volume/'candidate-memory-next')
                result,evidence=export_state(self.fixture.optimizer,epoch='control',inference_checkpoint='22'*32,
                    workspace=destination,publish_shard=no_network,readback_shard=no_network,
                    commit_descriptor=lambda d:dict(descriptor_sha256=sha(d),local_sha_verified=True,
                        durable_readback_verified=False,authority_committed=False),resource_admission=self.fixture.admission,
                    shard_bytes=self.fixture.cap,retain_shard=cache.retain,readback_mode=LOCAL_POLICY)
                cache.finish(result)
                candidate=cache.candidate
                self.assertTrue(candidate['files'])
                cache.discard(candidate,'test-owned-retirement')
                self.assertFalse(destination.exists())
            no_network.assert_not_called()
            self.assertTrue(cache.directory(cache.current['job_id']).exists())

    def test_memory_candidate_rejects_ambiguous_disk_directory(self):
        from subnet.optimizer_state_cache import candidate_directory
        volume=self.enable_volume();memory=volume/'candidate-ambiguous';memory.mkdir(mode=0o700)
        self.addCleanup(memory.rmdir)
        disk=self.root/'candidate-ambiguous';disk.mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            candidate_directory(self.fixture.root,self.root,'ambiguous')

    def test_current_direct_memory_parent_authenticates_and_restores_real_adam(self):
        import shutil,torch
        from subnet.cache_lifecycle import snapshot
        from subnet.optimizer_state_cache import verification_body
        from subnet.persistent_training_state import restore_state,admit_resources
        from unittest.mock import Mock
        volume=self.enable_volume();marker=self.root/'current.json';current=json.loads(marker.read_bytes())
        disk=self.root/('candidate-'+current['job_id']);memory=volume/disk.name
        shutil.copytree(disk,memory);shutil.rmtree(disk)
        self.addCleanup(shutil.rmtree,memory)
        for name,row in current['files'].items():row['stat']=snapshot(memory/name)
        current['promotion_verification_sha256']=sha(verification_body(current));marker.write_bytes(canonical(current))
        _,descriptor=authenticated_current(self.fixture.root,self.root,self.fixture.authority)
        out=self.fixture.root/'restore-from-memory';out.mkdir()
        no_network=Mock(side_effect=AssertionError('optimizer transport forbidden'))
        with StateCache(self.fixture.root,self.fixture.job,self.fixture.manifest,self.fixture.authority)as cache:
            cache.prepare_parent(descriptor,'bb'*32);self.assertIsNone(cache.next_volume())
            state,_=restore_state(descriptor,sha(descriptor),'22'*32,self.fixture.inventory,
                workspace=out,fetch_shard=lambda n,p:cache.fetch(n,p,no_network),
                resource_admission=admit_resources(out,self.fixture.plan),owned_cache=cache)
        for slot in ('master','exp_avg','exp_avg_sq'):
            self.assertTrue(torch.equal(state[1]['w'][slot],self.fixture.optimizer.rows['w'][slot]))
        no_network.assert_not_called()

    def test_low_headroom_advises_only_sealed_parent_then_remeasures(self):
        self.enable_volume();descriptor=self.fixture.report['persistent_training_state']['descriptor']
        with StateCache(self.fixture.root,self.fixture.job,self.fixture.manifest,self.fixture.authority)as cache:
            parent=cache.prepare_parent(descriptor,'bb'*32)
            required=self.fixture.plan['cpu_state_bytes']+1024**2+self.fixture.plan['cpu_additional_ram_required_bytes']
            with patch('subnet.persistent_training_state.available_ram_bytes',side_effect=[required-1,required]),patch('subnet.optimizer_state_cache.os.posix_fadvise')as advise:
                result=cache.admit(self.fixture.plan,reclaimable_parent_bytes=parent,disk_available=self.fixture.plan['additional_disk_required_bytes'])
                self.assertEqual(advise.call_count,len(cache.current['files']))
                self.assertEqual(result['optimizer_volume'],'memory')
            from subnet.cache_lifecycle import snapshot
            for name,row in cache.current['files'].items():
                self.assertEqual(snapshot(cache.directory(cache.current['job_id'])/name),row['stat'])
