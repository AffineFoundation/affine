import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from nacl.signing import SigningKey
from subnet.distributed_worker import Worker
from subnet.cache_lifecycle import CacheLifecycle

class IdleDisposal(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.worker=Worker('http://127.0.0.1:19081',bytes(SigningKey.generate()),'a'*64,self.root);self.worker.request=Mock(return_value={'claim':None});self.backend=self.root/'backend';self.lifecycle=CacheLifecycle(self.backend);self.cache=self.backend/'checkpoints/CP';self.cache.mkdir(parents=True);self.files={'model.safetensors':hashlib.sha256(b'weights').hexdigest()};(self.cache/'model.safetensors').write_bytes(b'weights');self.lifecycle.record_checkpoint('CP',self.files);self.report=self.backend/'jobs/old-job/report.json';self.report.parent.mkdir(parents=True);self.report.write_text('retained original');self.log=self.root/'old-job/attempt-1/worker.log';self.log.parent.mkdir(parents=True);self.log.write_text('retained log')
 def test_finished_owned_model_deleted_without_a_future_job(self):
  with patch('subnet.distributed_worker.subprocess.run',side_effect=AssertionError('no GPU job')):self.assertFalse(self.worker.once())
  self.worker.request.assert_called_once_with('claim',role='verify');self.assertFalse(self.cache.exists());self.assertFalse((self.lifecycle.meta/'CP.json').exists());self.assertEqual(self.report.read_text(),'retained original');self.assertEqual(self.log.read_text(),'retained log')
 def test_other_active_lease_defers_then_next_idle_poll_disposes(self):
  with self.lifecycle.lease_checkpoint('CP'):
   self.assertFalse(self.worker.once());self.assertTrue(self.cache.exists())
  self.assertFalse(self.worker.once());self.assertFalse(self.cache.exists())
 def test_changed_inode_untracked_cache_and_external_mapping_preserved(self):
  (self.cache/'model.safetensors').write_bytes(b'changed inode snapshot');foreign=self.backend/'checkpoints/FOREIGN';foreign.mkdir();(foreign/'model.safetensors').write_bytes(b'unknown');external=self.root/'external';external.mkdir();(external/'model.safetensors').write_bytes(b'external');self.worker.checkpoint_caches={'EXTERNAL':str(external)};self.assertFalse(self.worker.once());self.assertTrue(self.cache.exists());self.assertEqual((foreign/'model.safetensors').read_bytes(),b'unknown');self.assertEqual((external/'model.safetensors').read_bytes(),b'external')
 def test_shared_hardlink_or_symlink_cache_never_deleted(self):
  import os
  os.link(self.cache/'model.safetensors',self.root/'protected-alias');self.assertFalse(self.worker.once());self.assertTrue(self.cache.exists());self.assertTrue((self.root/'protected-alias').exists())
 def test_storage_error_is_deferred_and_worker_can_poll_again(self):
  with patch.object(CacheLifecycle,'evict_checkpoints',side_effect=OSError('storage unavailable')),self.assertLogs(level='WARNING'):self.assertFalse(self.worker.once())
  self.assertTrue(self.cache.exists());self.assertFalse(self.worker.once());self.assertFalse(self.cache.exists())
 def test_transport_error_before_claim_response_does_not_sweep(self):
  self.worker.request.side_effect=ConnectionError('original transport unavailable')
  with self.assertRaises(ConnectionError):self.worker.once()
  self.assertTrue(self.cache.exists())
if __name__=='__main__':unittest.main()
