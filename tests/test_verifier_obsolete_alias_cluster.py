import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from ops.verifier_obsolete_alias_cluster import retire_obsolete_alias_cluster
from ops.verifier_redundant_cache_lifecycle import assert_unreferenced

class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);files={'config.json':b'{}','model.safetensors':b'W'*8192};self.meta={n:dict(size=len(v),sha256=hashlib.sha256(v).hexdigest())for n,v in files.items()};cp=hashlib.sha256(json.dumps({n:v['sha256']for n,v in self.meta.items()},sort_keys=True,separators=(',',':')).encode()).hexdigest();self.paths=[self.root/'a/checkpoints'/cp,self.root/'b/checkpoints'/cp]
        for p in self.paths:p.mkdir(parents=True)
        for n,v in files.items():(self.paths[0]/n).write_bytes(v);os.link(self.paths[0]/n,self.paths[1]/n)
        self.proc=self.root/'proc';self.proc.mkdir();self.plan=dict(checkpoint=cp,files=self.meta,directories=[str(p)for p in self.paths],archive_verified=True,descriptor_authenticated=True,reference_retirements_verified=True,protected_checkpoints=['a'*64],active_checkpoints=[],worker_mapped_checkpoints=[],operation_directory=str(self.root/'operation'))
    def operation(self,apply=False):
        with patch('ops.verifier_obsolete_alias_cluster.subprocess.check_output',return_value=''),patch('ops.verifier_obsolete_alias_cluster.assert_unreferenced',side_effect=lambda roots:assert_unreferenced(roots,self.proc)):
            return retire_obsolete_alias_cluster(self.plan,apply=apply)
    def test_full_two_link_cluster_reclaim_counts_once_then_actual_apply(self):
        before=sum((self.paths[0]/n).stat().st_blocks*512 for n in self.meta);r=self.operation();self.assertEqual(r['estimated_reclaim_bytes'],before);self.assertTrue(all(p.exists()for p in self.paths));r=self.operation(True);self.assertTrue(r['completed']);self.assertFalse(any(p.exists()for p in self.paths));self.assertTrue((self.root/'operation/operation-completed.private.json').exists())
    def test_current_pending_and_worker_mapping_refuse(self):
        for field in('protected_checkpoints','active_checkpoints','worker_mapped_checkpoints'):
            old=self.plan[field];self.plan[field]=[self.plan['checkpoint']]
            with self.assertRaisesRegex(ValueError,'protected'):self.operation()
            self.plan[field]=old
    def test_unknown_third_link_and_altered_bytes_refuse(self):
        extra=self.root/'unknown';os.link(self.paths[0]/'model.safetensors',extra)
        with self.assertRaisesRegex(ValueError,'two-link'):self.operation()
        extra.unlink();(self.paths[0]/'model.safetensors').write_bytes(b'X'*8192)
        with self.assertRaisesRegex(ValueError,'byte hash'):self.operation()
    def test_directory_symlink_and_extra_membership_refuse(self):
        for n in self.meta:(self.paths[1]/n).unlink()
        self.paths[1].rmdir();self.paths[1].symlink_to(self.paths[0],target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'canonical'):self.operation()
    def test_second_rename_failure_is_uncertain_and_never_repeats(self):
        original=Path.rename
        def rename(p,dest):
            if p==self.paths[1]:raise OSError('fixture second alias rename')
            return original(p,dest)
        with patch.object(Path,'rename',rename):
            with self.assertRaises(OSError):self.operation(True)
        self.assertTrue((self.root/'operation/operation-start.private.json').exists());self.assertTrue((self.root/'operation/operation-uncertain.private.json').exists());self.assertFalse(self.paths[0].exists());self.assertTrue(self.paths[1].exists());self.assertTrue(list(self.paths[0].parent.glob('*.obsolete-cluster-*')))
    def test_missing_archive_reference_approval_refuses(self):
        self.plan['reference_retirements_verified']=False
        with self.assertRaises(ValueError):self.operation()
if __name__=='__main__':unittest.main()
