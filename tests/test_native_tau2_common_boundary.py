import hashlib,unittest
from ops.finalize_native_tau2_common_boundary import select
class BoundaryTests(unittest.TestCase):
 def setUp(self):
  raw=b'final';self.pre={'zip_sha256':hashlib.sha256(raw).hexdigest(),'zip_size':5,'r2_etag':'etag','manifest_sha256':'a'*64,'registered_uid':131};self.snapshot={'data':raw,'etag':'etag','completed_at':9.}
 def test_final_get_after_deadline_selects_admitted_body(self):
  self.assertTrue(select(self.snapshot,self.pre,10.,11.)['final_selection_after_deadline'])
 def test_early_snapshot_not_final(self):
  with self.assertRaises(ValueError):select(self.snapshot,self.pre,10.,9.)
 def test_late_upload_rejected(self):
  self.snapshot['completed_at']=10.
  with self.assertRaises(ValueError):select(self.snapshot,self.pre,10.,11.)
 def test_changed_cumulative_body_rejected(self):
  self.snapshot['data']=b'forge'
  with self.assertRaises(ValueError):select(self.snapshot,self.pre,10.,11.)
if __name__=='__main__':unittest.main()
