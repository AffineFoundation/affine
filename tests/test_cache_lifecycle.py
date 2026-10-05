import hashlib
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from subnet.cache_lifecycle import CacheLifecycle


def hold(root,ready,release):
    with CacheLifecycle(root).lease_checkpoint('old'):
        ready.set();release.wait(5)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cache=CacheLifecycle(self.root)
        self.files={'model.safetensors':hashlib.sha256(b'model').hexdigest()}
    def checkpoint(self,cp='old'):
        directory=self.root/'checkpoints'/cp;directory.mkdir(parents=True)
        (directory/'model.safetensors').write_bytes(b'model')
        with self.cache.lease_checkpoint(cp):self.cache.record_checkpoint(cp,self.files)
        return directory
    def test_retire_obsolete_no_rehash_and_preserve_current(self):
        old=self.checkpoint();current=self.checkpoint('new')
        self.assertEqual(self.cache.evict_checkpoints(exclude=['new'],keep=0),['old'])
        self.assertFalse(old.exists());self.assertTrue(current.exists())
    def test_active_multiprocess_lease_prevents_deletion(self):
        old=self.checkpoint();ready=multiprocessing.Event();release=multiprocessing.Event()
        process=multiprocessing.Process(target=hold,args=(self.root,ready,release));process.start()
        self.addCleanup(lambda:process.is_alive() and process.terminate())
        self.assertTrue(ready.wait(3));self.assertEqual(self.cache.evict_checkpoints(keep=0),[])
        release.set();process.join(3);self.assertEqual(self.cache.evict_checkpoints(keep=0),['old'])
    def test_backend_child_inherits_lease_after_parent_releases_fd(self):
        self.checkpoint()
        with self.cache.lease_checkpoint('old') as fd:
            child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.3)'],pass_fds=(fd,))
        self.assertEqual(self.cache.evict_checkpoints(keep=0),[])
        child.wait();self.assertEqual(self.cache.evict_checkpoints(keep=0),['old'])
    def test_changed_or_extra_files_are_not_deleted(self):
        old=self.checkpoint();(old/'model.safetensors').write_bytes(b'new bytes')
        self.assertEqual(self.cache.evict_checkpoints(keep=0),[])
        self.assertTrue(old.exists())
        with self.cache.lease_checkpoint('old'):self.cache.record_checkpoint('old',self.files)
        (old/'extra').write_bytes(b'x');self.assertEqual(self.cache.evict_checkpoints(keep=0),[])
    def test_symlink_hardlink_and_external_paths_fail_closed(self):
        old=self.checkpoint();weight=old/'model.safetensors';outside=self.root/'outside';outside.write_bytes(b'model')
        weight.unlink();weight.symlink_to(outside)
        self.assertEqual(self.cache.evict_checkpoints(keep=0),[]);self.assertTrue(outside.exists())
        with self.assertRaises(ValueError):self.cache.record_checkpoint('old',self.files)
        weight.unlink();os.link(outside,weight)
        with self.assertRaises(ValueError):self.cache.record_checkpoint('old',self.files)
        with self.assertRaises(ValueError):self.cache.adopt_checkpoint('x',outside,self.files,{'sha':'verified'})
    def test_partial_only_receipted_members_can_retire(self):
        directory=self.root/'checkpoints'/'old';directory.mkdir(parents=True)
        (directory/'model.safetensors').write_bytes(b'model')
        inventory=dict(self.files,**{'config.json':hashlib.sha256(b'{}').hexdigest()})
        self.cache.record_checkpoint_member('old','model.safetensors',inventory,self.files['model.safetensors'])
        self.assertEqual(self.cache.evict_checkpoints(keep=0),['old'])
    def test_download_cleanup_retains_reports_and_requires_unchanged_snapshot(self):
        out=self.root/'jobs'/'job';out.mkdir(parents=True);download=out/'submission-0.zip';download.write_bytes(b'x')
        report=out/'report.json';report.write_text('report');self.cache.record_download(download,'a'*64)
        download.write_bytes(b'changed');self.assertEqual(self.cache.retire_downloads('job'),[])
        self.cache.record_download(download,'b'*64)
        self.assertEqual(self.cache.retire_downloads('job'),['jobs/job/submission-0.zip']);self.assertTrue(report.exists())
    def test_export_requires_durable_ack_and_explicit_owned_path(self):
        out=self.root/'jobs'/'job'/'checkpoint-persistent-final';out.mkdir(parents=True);(out/'model.safetensors').write_bytes(b'model')
        with self.assertRaises(ValueError):self.cache.adopt_checkpoint('export',out,self.files,None)
        with self.cache.lease_checkpoint('export'):self.cache.adopt_checkpoint('export',out,self.files,{'checkpoint':'export','ack':'authenticated'})
        self.assertEqual(self.cache.evict_checkpoints(keep=0),['export']);self.assertFalse(out.exists())
    def test_unmanaged_checkpoint_never_deleted(self):
        old=self.root/'checkpoints'/'old';old.mkdir(parents=True);(old/'model.safetensors').write_bytes(b'model')
        self.assertEqual(self.cache.evict_checkpoints(keep=0),[]);self.assertTrue(old.exists())
    def test_capacity_only_evicts_until_requested_headroom(self):
        self.checkpoint();self.assertEqual(self.cache.evict_checkpoints(keep=0,required_free_bytes=1),[])

if __name__=='__main__':unittest.main()
