import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from ops.probe_numina_model_search import hydrate_checkpoint
from subnet.storage import canonical
class CheckpointDownloadTests(unittest.TestCase):
 def plan(self,directory):
  files={n:hashlib.sha256(b'{}').hexdigest() for n in ('config.json','model.safetensors')}
  return dict(checkpoint={'files':files,'id':hashlib.sha256(canonical(files)).hexdigest()},checkpoint_path=str(directory),checkpoint_downloads={n:{'size':2,'url':'https://test.r2.cloudflarestorage.com/a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=not-real'} for n in files})
 def test_exact_download_and_verified_cache(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.plan(Path(d)/'checkpoint')
   def get(url,digest,target,limit):self.assertEqual(limit,2);target.write_bytes(b'{}')
   with patch('subnet.backend_jobs.get_object',side_effect=get) as call:
    hydrate_checkpoint(p);hydrate_checkpoint(p);self.assertEqual(call.call_count,2)
 def test_existing_corruption_not_replaced(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.plan(Path(d));(Path(d)/'config.json').write_bytes(b'xx')
   with patch('subnet.backend_jobs.get_object') as call:
    with self.assertRaises(ValueError):hydrate_checkpoint(p)
    call.assert_not_called()
 def test_unapproved_host_no_download(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.plan(Path(d));p['checkpoint_downloads']['config.json']['url']='https://example.com/config.json'
   with patch('subnet.backend_jobs.get_object') as call:
    with self.assertRaises(ValueError):hydrate_checkpoint(p)
    call.assert_not_called()
 def test_symlink_and_extra_file_fail(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.plan(Path(d));(Path(d)/'extra').write_bytes(b'x')
   with self.assertRaises(ValueError):hydrate_checkpoint(p)
   (Path(d)/'extra').unlink();(Path(d)/'config.json').symlink_to(Path(d)/'outside')
   with self.assertRaises(ValueError):hydrate_checkpoint(p)
if __name__=='__main__':unittest.main()
