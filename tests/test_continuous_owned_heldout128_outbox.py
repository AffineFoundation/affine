import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from ops import continuous_owned_heldout128_outbox as io
from ops.owned_cached_group_operator import private_json
class OutboxTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.p=self.root/'outbox.json';io.save_outbox(self.p,{'checkpoints':{},'history':'preserved'})
 def test_growth_past8MiB_reopen_preserves_all_original_history(self):
  state=io.read_outbox(self.p);state['checkpoints']['original-checkpoint']={'phase':'dispatch_attempted','original_job_id':'never-repeat','history':'x'*(9*1024**2)};io.save_outbox(self.p,state);self.assertGreater(self.p.stat().st_size,8*1024**2);self.assertEqual(io.read_outbox(self.p),state)
  with self.assertRaises(ValueError):private_json(self.p)
  state['checkpoints']['successor']={'phase':'prepared','original_job_id':'distinct-original'};io.save_outbox(self.p,state);self.assertEqual(io.read_outbox(self.p),state);self.assertEqual(self.p.stat().st_mode&0o777,0o600)
 def test_over64MiB_read_refused_before_payload_read(self):
  with self.p.open('wb')as f:f.truncate(io.OUTBOX_MAX_BYTES+1)
  with patch.object(io.os,'fdopen',side_effect=AssertionError('no payload stream opened')):
   with self.assertRaises(ValueError):io.read_outbox(self.p)
 def test_save_capacity_refusal_preserves_original_bytes_no_temp(self):
  old=self.p.read_bytes()
  with patch.object(io,'OUTBOX_MAX_BYTES',1024),patch.object(io.tempfile,'mkstemp',side_effect=AssertionError('no temp created')):
   with self.assertRaises(ValueError):io.save_outbox(self.p,{'history':'x'*1024})
  self.assertEqual(self.p.read_bytes(),old)
 def test_symlink_hardlink_public_mode_and_owner_refused(self):
  self.p.chmod(0o644)
  with self.assertRaises(ValueError):io.read_outbox(self.p)
  self.p.chmod(0o600);os.link(self.p,self.root/'alias')
  with self.assertRaises(ValueError):io.read_outbox(self.p)
  (self.root/'alias').unlink()
  with patch.object(io.os,'geteuid',return_value=os.geteuid()+1):
   with self.assertRaises(ValueError):io.read_outbox(self.p)
  self.p.unlink();self.p.symlink_to('/dev/null')
  with self.assertRaises(OSError):io.read_outbox(self.p)
  with self.assertRaises(OSError):io.save_outbox(self.p,{})
 def test_changed_during_read_refused(self):
  duplicate=os.dup
  def mutate(fd):self.p.write_text('{"changed":true}');return duplicate(fd)
  with patch.object(io.os,'dup',side_effect=mutate):
   with self.assertRaises(ValueError):io.read_outbox(self.p)
 def test_wrong_route_or_public_parent_refused(self):
  with self.assertRaises(ValueError):io.read_outbox(self.root/'signed-job.json')
  self.root.chmod(0o755)
  with self.assertRaises(ValueError):io.read_outbox(self.p)
 def test_save_rejects_existing_hostile_file_preserving_it(self):
  self.p.chmod(0o644);before=self.p.read_bytes()
  with self.assertRaises(ValueError):io.save_outbox(self.p,{'new':True})
  self.assertEqual(self.p.read_bytes(),before);self.assertEqual(self.p.stat().st_mode&0o777,0o644)
if __name__=='__main__':unittest.main()
