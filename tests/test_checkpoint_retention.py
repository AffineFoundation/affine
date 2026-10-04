import hashlib
import json
import mmap
import ctypes
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ops.checkpoint_retention import digest,remove_checkpoint_replica


class CheckpointRetirement(unittest.TestCase):
    def setUp(self):
        self.scanner=patch('ops.checkpoint_retention.processes',return_value=[Path('/proc',str(os.getpid()))])
        self.gpu=patch('ops.checkpoint_retention.gpu_processes',return_value=[])
        self.scanner.start();self.gpu.start();self.addCleanup(self.scanner.stop);self.addCleanup(self.gpu.stop)

    def fixture(self, directory):
        contents={'config.json':b'{}','model.safetensors':b'checkpoint'}
        hashes={n:hashlib.sha256(v).hexdigest() for n,v in contents.items()};checkpoint=digest(hashes)
        root=Path(directory)/'checkpoints'/checkpoint;root.mkdir(parents=True)
        for name,value in contents.items():(root/name).write_bytes(value)
        plan=dict(checkpoint=checkpoint,directory=str(root),files={n:dict(sha256=hashes[n],size=len(v)) for n,v in contents.items()},
            archive_verified=True,descriptor_authenticated=True,protected_checkpoints=['d'*64],active_checkpoints=[])
        return plan,root

    def test_exact_archived_copy_removed_with_protected_files_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,root=self.fixture(directory);protected=root.parent/'protected';protected.mkdir();(protected/'weights').write_bytes(b'retain')
            result=remove_checkpoint_replica(plan)
            self.assertTrue(result['removed']);self.assertFalse(root.exists());self.assertEqual(result['bytes'],12)
            self.assertEqual((protected/'weights').read_bytes(),b'retain');self.assertTrue(remove_checkpoint_replica(plan)['already_absent'])

    def test_current_referenced_or_unarchived_checkpoints_cannot_retire(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,root=self.fixture(directory)
            for update in ({'protected_checkpoints':[plan['checkpoint']]},{'active_checkpoints':[plan['checkpoint']]},{'archive_verified':False},{'descriptor_authenticated':False}):
                with self.assertRaises(ValueError):remove_checkpoint_replica(dict(plan,**update))
                self.assertTrue(root.exists())

    def test_changed_bytes_or_unlisted_file_cannot_retire(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,root=self.fixture(directory);path=root/'model.safetensors';path.write_bytes(b'bad-weights')
            with self.assertRaises(ValueError):remove_checkpoint_replica(plan)
            path.write_bytes(b'checkpoint');(root/'unlisted').write_bytes(b'keep')
            with self.assertRaisesRegex(ValueError,'membership'):remove_checkpoint_replica(plan)
            self.assertTrue(root.exists())

    def test_job_export_or_symlink_cannot_retire(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,root=self.fixture(directory)
            with self.assertRaisesRegex(ValueError,'cache path'):remove_checkpoint_replica(dict(plan,directory=str(Path(directory)/'jobs/checkpoint-step-1')))
            original=root/'model.safetensors';original.unlink();target=Path(directory)/'other';target.write_bytes(b'checkpoint');original.symlink_to(target)
            with self.assertRaises(ValueError):remove_checkpoint_replica(plan)
            self.assertTrue(target.exists())

    def test_gpu_occupancy_cannot_retire(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,root=self.fixture(directory)
            with patch('ops.checkpoint_retention.gpu_processes',return_value=['123']):
                with self.assertRaisesRegex(ValueError,'idle GPU'):remove_checkpoint_replica(plan)
            self.assertTrue(root.exists())

    def test_unsharded_model_bound_still_requires_exact_local_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,root=self.fixture(directory)
            plan['files']['model.safetensors']['size']=15_231_272_152
            with self.assertRaisesRegex(ValueError,'local object type or size'):
                remove_checkpoint_replica(plan)
            self.assertTrue(root.exists())
            plan['files']['model.safetensors']['size']=32*1024**3+1
            with self.assertRaisesRegex(ValueError,'metadata'):
                remove_checkpoint_replica(plan)
            plan['files']['model.safetensors']['size']=10
            plan['files']['config.json']['size']=5*1024**3+1
            with self.assertRaisesRegex(ValueError,'metadata'):
                remove_checkpoint_replica(plan)

    def test_missing_protection_or_hardlinked_weights_cannot_retire(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,root=self.fixture(directory)
            with self.assertRaisesRegex(ValueError,'protection'):remove_checkpoint_replica(dict(plan,protected_checkpoints=[]))
            os.link(root/'model.safetensors',Path(directory)/'another-reference')
            with self.assertRaisesRegex(ValueError,'type or size'):remove_checkpoint_replica(plan)
            self.assertTrue(root.exists())

    def test_open_file_and_closed_descriptor_memory_map_cannot_retire(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,root=self.fixture(directory);path=root/'model.safetensors'
            with path.open('rb') as f:
                with self.assertRaisesRegex(ValueError,'still open'):remove_checkpoint_replica(plan)
                # Python's mmap retains a duplicate FD. Use an actual libc
                # mapping so closing this descriptor leaves only /proc/maps.
                libc=ctypes.CDLL(None,use_errno=True)
                libc.mmap.restype=ctypes.c_void_p
                libc.mmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_long]
                libc.munmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
                mapped=libc.mmap(None,10,mmap.PROT_READ,mmap.MAP_PRIVATE,f.fileno(),0)
                self.assertNotEqual(mapped,ctypes.c_void_p(-1).value)
            try:
                with self.assertRaisesRegex(ValueError,'memory mapped'):remove_checkpoint_replica(plan)
            finally:self.assertEqual(libc.munmap(mapped,10),0)
            self.assertTrue(root.exists())

if __name__=='__main__':unittest.main()
