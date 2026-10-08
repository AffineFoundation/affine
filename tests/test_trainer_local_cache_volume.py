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
