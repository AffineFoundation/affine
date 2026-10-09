from pathlib import Path
import hashlib,json,os,sys,tempfile,unittest
from unittest.mock import patch
from ops import current_assessment_evidence as r
class ReaderControls(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def write(self,raw=b'{"ok":true}'):
  p=self.root/'input.json';p.write_bytes(raw);return p
 def test_exact_data_and_sha(self):
  p=self.write();self.assertEqual(r._read(p),({'ok':True},hashlib.sha256(p.read_bytes()).hexdigest()))
 def test_default_small_bound_remains(self):
  p=self.write();self.assertRaises(ValueError,r._read,p,maximum=3)
 def test_oversized_sparse_registry_refused_before_allocation(self):
  p=self.write()
  with p.open('r+b')as f:f.truncate(r.AUDIT_STATE_MAX_BYTES+1)
  self.assertRaisesRegex(ValueError,'bounded',r._read,p,maximum=r.AUDIT_STATE_MAX_BYTES)
 def test_symlink_and_fifo_refused(self):
  p=self.write();q=self.root/'link';q.symlink_to(p)
  self.assertRaises(OSError,r._read,q)
  f=self.root/'fifo';os.mkfifo(f);self.assertRaisesRegex(ValueError,'regular',r._read,f)
 def test_concurrent_inode_write_refused(self):
  p=self.write();original=r.os.fstat;calls=[]
  def change(fd):
   calls.append(fd)
   if len(calls)==2:p.write_bytes(b'{"ok":false}')
   return original(fd)
  with patch.object(r.os,'fstat',change):self.assertRaisesRegex(ValueError,'changed',r._read,p)
 def test_atomic_replacement_keeps_exact_original_snapshot(self):
  p=self.write();original=r.os.fstat;calls=[]
  def replace(fd):
   result=original(fd);calls.append(fd)
   if len(calls)==1:
    q=self.root/'new';q.write_bytes(b'{"ok":false}');q.replace(p)
   return result
  with patch.object(r.os,'fstat',replace):self.assertEqual(r._read(p)[0],{'ok':True})
 def test_real_registry_size_above_previous_cap(self):
  p=self.write()
  with p.open('ab')as f:
   for _ in range(257):f.write(b' '*1024**2)
  self.assertRaisesRegex(ValueError,'bounded',r._read,p)
  self.assertEqual(r._read(p,maximum=r.AUDIT_STATE_MAX_BYTES)[0],{'ok':True})
 def test_only_registry_call_uses_growth_budget(self):
  source=Path(r.__file__).read_text();self.assertEqual(source.count('maximum=AUDIT_STATE_MAX_BYTES'),1)
  self.assertIn("_read(directory / 'audit-state.json', maximum=AUDIT_STATE_MAX_BYTES)",source)
if __name__=='__main__':unittest.main()
