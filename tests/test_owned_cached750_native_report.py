"""Native backend permission compatibility without widening private inputs."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from ops.owned_cached750_group_operator import native_report_json,private_json
from ops.owned_cached750_group_ack_relay import READ_BODY
class ReportTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.jobs=self.root/'jobs';self.jobs.mkdir(mode=0o755);self.directory=self.jobs/'original-g0';self.directory.mkdir(mode=0o700);self.p=self.directory/'report.json';self.p.write_text('{"success":true}');self.p.chmod(0o644)
 def test_backend644_original_unchanged_and_strict_private_refuses(self):
  before=self.p.stat();self.assertEqual(native_report_json(self.p,self.root),{'success':True});self.assertEqual(before,self.p.stat())
  with self.assertRaises(ValueError):private_json(self.p)
 def test_remote_observer_uses_identical_reader(self):
  ns={};exec(READ_BODY.split("root=pathlib.Path(PLAN")[0],ns);self.assertEqual(ns['native_report_json'](self.p,self.root),{'success':True})
 def test_report600_still_accepted(self):self.p.chmod(0o600);self.assertEqual(native_report_json(self.p,self.root),{'success':True})
 def test_public_job_directory_refused(self):
  self.directory.chmod(0o755)
  with self.assertRaises(ValueError):native_report_json(self.p,self.root)
 def test_group_writable_parent_refused(self):
  self.jobs.chmod(0o775)
  with self.assertRaises(ValueError):native_report_json(self.p,self.root)
 def test_public_root_refused(self):
  self.root.chmod(0o755)
  with self.assertRaises(ValueError):native_report_json(self.p,self.root)
 def test_hardlink_report_refused(self):
  os.link(self.p,self.directory/'alias')
  with self.assertRaises(ValueError):native_report_json(self.p,self.root)
 def test_symlink_report_or_jobdir_refused(self):
  self.p.unlink();self.p.symlink_to('/dev/null')
  with self.assertRaises(OSError):native_report_json(self.p,self.root)
  self.p.unlink();self.directory.rmdir();self.directory.symlink_to(self.root,target_is_directory=True)
  with self.assertRaises(OSError):native_report_json(self.p,self.root)
 def test_writable_or_executable_report_refused(self):
  for mode in(0o664,0o666,0o744):
   self.p.chmod(mode)
   with self.assertRaises(ValueError):native_report_json(self.p,self.root)
 def test_foreign_owner_refused(self):
  with patch('ops.owned_cached750_group_operator.os.geteuid',return_value=os.geteuid()+1):
   with self.assertRaises(ValueError):native_report_json(self.p,self.root)
 def test_oversize_and_wrong_report_route_refused(self):
  with self.p.open('wb')as f:f.truncate(8*1024**2+1)
  with self.assertRaises(ValueError):native_report_json(self.p,self.root)
  with self.assertRaises(ValueError):native_report_json(self.directory/'scope.json',self.root)
if __name__=='__main__':unittest.main()
