import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ops.verifier_cache_capacity_plan import canonical_digest,plan_reuse
from ops.verifier_redundant_cache_lifecycle import retire_duplicate,assert_unreferenced

class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.keeper=self.root/'keeper';self.alias=self.root/'alias';self.target=self.root/'replica'
        for p in (self.keeper,self.alias,self.target):p.mkdir()
        self.files={}
        for n,b in [('config.json',b'{}'),('model.safetensors',b'W'*8192)]:
            (self.keeper/n).write_bytes(b);os.link(self.keeper/n,self.alias/n);(self.target/n).write_bytes(b)
            self.files[n]=dict(size=len(b),sha256=hashlib.sha256(b).hexdigest())
        self.cp=canonical_digest({n:m['sha256'] for n,m in self.files.items()})
        self.protected=[self.cp,'a'*64]
        self.proof=plan_reuse(self.cp,self.files,self.keeper,self.target,self.protected,free_bytes=0,artifact_bytes=0,reserve_bytes=0)
        self.plan=dict(kind='explicit-verifier-redundancy-operation-v1',archive_readback_verified=True,root_approval_verified=True,queue_references_verified=True,
            original_worker=dict(pid=999999999,ticks='0'),checkpoint=self.cp,files=self.files,keeper=str(self.keeper),keeper_alias=str(self.alias),replica=str(self.target),
            protected_checkpoints=self.protected,artifact_bytes=0,reserve_bytes=0,prior_proof=self.proof,operation_directory=str(self.root/'operation'))
        # Real scanner, empty simulated proc directory; GPU CLI fixture only.
        self.proc=self.root/'proc';self.proc.mkdir()
    def run_plan(self,apply=False):
        with patch('ops.verifier_redundant_cache_lifecycle.subprocess.check_output',return_value=''),patch('ops.verifier_redundant_cache_lifecycle.assert_unreferenced',side_effect=lambda roots:assert_unreferenced(roots,self.proc)):
            return retire_duplicate(self.plan,apply=apply)
    def test_review_then_real_bounded_retirement_preserves_protections_and_aliases(self):
        self.assertTrue(self.run_plan()['review_only']);self.assertTrue(self.target.exists())
        r=self.run_plan(True);self.assertTrue(r['completed']);self.assertFalse(self.target.exists())
        self.assertEqual(r['protected_checkpoints'],self.protected)
        for n in self.files:self.assertEqual((self.keeper/n).stat().st_ino,(self.alias/n).stat().st_ino)
        self.assertTrue((self.root/'operation/operation-completed.private.json').exists())
    def test_started_operation_never_repeats(self):
        (self.root/'operation').mkdir()
        with self.assertRaisesRegex(ValueError,'prior operation'):self.run_plan(True)
        self.assertTrue(self.target.exists())
    def test_changed_proof_and_unknown_keeper_alias_refuse(self):
        self.plan['prior_proof']['replica_files']['model.safetensors']['inode']+=1
        with self.assertRaisesRegex(ValueError,'inode proof'):self.run_plan()
        self.plan['prior_proof']=plan_reuse(self.cp,self.files,self.keeper,self.target,self.protected,free_bytes=0,artifact_bytes=0,reserve_bytes=0)
        os.link(self.keeper/'model.safetensors',self.root/'third-link')
        with self.assertRaisesRegex(ValueError,'two-alias'):self.run_plan()
    def test_keeper_alias_directory_symlink_refuses(self):
        for n in self.files:(self.alias/n).unlink()
        self.alias.rmdir();self.alias.symlink_to(self.keeper,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'canonical keeper alias'):self.run_plan()
        self.assertTrue(self.target.exists())
    def test_rename_failure_records_uncertainty_before_retry(self):
        with patch.object(Path,'rename',side_effect=OSError('fixture rename refused')):
            with self.assertRaises(OSError):self.run_plan(True)
        self.assertTrue(self.target.exists())
        self.assertTrue((self.root/'operation/operation-start.private.json').exists())
        self.assertTrue((self.root/'operation/operation-uncertain.private.json').exists())
        with self.assertRaisesRegex(ValueError,'prior operation'):self.run_plan(True)
    def test_missing_approval_and_gpu_busy_refuse(self):
        self.plan['root_approval_verified']=False
        with self.assertRaises(ValueError):self.run_plan()
        self.plan['root_approval_verified']=True
        with patch('ops.verifier_redundant_cache_lifecycle.subprocess.check_output',return_value='123'):
            with self.assertRaisesRegex(ValueError,'idle GPU'):retire_duplicate(self.plan)
    def test_actual_scanner_fd_cwd_and_maps_refuse(self):
        p=self.proc/'123';p.mkdir();(p/'fd').mkdir();(p/'stat').write_text('123 (x) S '+'0 '*18+'1 0')
        (p/'cwd').symlink_to(self.root);(p/'maps').write_text('')
        (p/'fd/3').symlink_to(self.target/'model.safetensors')
        with self.assertRaisesRegex(ValueError,'process reference'):assert_unreferenced([self.target],self.proc)
        (p/'fd/3').unlink();(p/'cwd').unlink();(p/'cwd').symlink_to(self.target)
        with self.assertRaisesRegex(ValueError,'process reference'):assert_unreferenced([self.target],self.proc)
        (p/'cwd').unlink();(p/'cwd').symlink_to(self.root)
        (p/'maps').write_text('0-1 r--p 0 00:01 1 '+str(self.target/'model.safetensors'))
        with self.assertRaisesRegex(ValueError,'memory mapped'):assert_unreferenced([self.target],self.proc)
    def test_uncertain_operation_journal_prevents_blind_repeat(self):
        real_unlink=Path.unlink
        def broken(p,*args,**kwargs):
            if p.name=='model.safetensors':raise OSError('fixture interruption')
            return real_unlink(p,*args,**kwargs)
        with patch.object(Path,'unlink',broken):
            with self.assertRaises(OSError):self.run_plan(True)
        self.assertTrue((self.root/'operation/operation-uncertain.private.json').exists())
        self.assertTrue(list(self.root.glob('replica.redundant-retired-*')))

if __name__=='__main__':unittest.main()
