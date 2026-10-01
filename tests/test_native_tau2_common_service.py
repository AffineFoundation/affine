import json,pathlib,tempfile,unittest
from unittest.mock import patch,Mock
from nacl.signing import SigningKey
from subnet import native_tau2_common_service as s

class CommonServiceTests(unittest.TestCase):
 def test_disk_admission_before_allocation(self):
  with patch.object(s.shutil,'disk_usage',return_value=Mock(free=100)):
   with self.assertRaisesRegex(RuntimeError,'waiting-disk-capacity'):s.disk_guard('.')
 def test_full_policy_does_not_train_auxiliary(self):
  self.assertFalse(s.TRAIN_POLICY['auxiliary_tokens_in_loss']);self.assertTrue(s.TRAIN_POLICY['full_model_finetune']);self.assertEqual(s.TRAIN_POLICY['steps'],1);self.assertEqual(s.TRAIN_POLICY['reference'],s.REFERENCE)
 def test_remote_signature_precedes_source_access(self):
  key=SigningKey.generate()
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'job';p.write_bytes(s.canonical(s.sign({'source_files':{'missing.py':'x'}},key)))
   with patch.object(s,'source_guard') as guard:
    with self.assertRaises(ValueError):s.remote_job(p)
    guard.assert_not_called()
 def test_remote_payable_rejected_before_execution(self):
  key=SigningKey.generate()
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'job';p.write_bytes(s.canonical(s.sign({'source_files':{},'payable':True,'chain_transactions':False},key)))
   with patch.object(s,'AUTHORITY',key.verify_key.encode().hex()),patch('subnet.long_context_runtime.AUTHORITY',key.verify_key.encode().hex()),patch.object(s,'source_guard'):
    with self.assertRaisesRegex(ValueError,'nonpayable'):s.remote_job(p)
 def test_source_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);(root/'real.py').write_text('pass');(root/'link.py').symlink_to(root/'real.py')
   with patch.object(s,'ROOT',root):
    with self.assertRaises(ValueError):s.source_guard({'link.py':s.file_sha(root/'real.py')})
 def test_private_writes(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'record';s.write(p,{'private':True});self.assertEqual(p.stat().st_mode&0o777,0o600)
 def test_remote_existing_terminal_is_not_resubmitted(self):
  with tempfile.TemporaryDirectory() as d:
   c=object.__new__(s.Coordinator);c.ssh=lambda x:['ssh',x];c.key=SigningKey.generate();job={'scope':'one'}
   s.write(pathlib.Path(d)/'train-remote-process.json',{'pid':7,'start_ticks':'9','job_sha256':s.digest(job),'remote':'/root/test','label':'train'})
   with patch.object(s.subprocess,'check_output',return_value='EXIT\n0'),patch.object(c,'scp') as scp:
    c.remote(pathlib.Path(d),'/root/test',job,'train');scp.assert_not_called()
 def test_remote_missing_is_not_retry_permission(self):
  with tempfile.TemporaryDirectory() as d:
   c=object.__new__(s.Coordinator);c.ssh=lambda x:['ssh',x];c.key=SigningKey.generate();job={'scope':'one'}
   s.write(pathlib.Path(d)/'train-remote-process.json',{'pid':7,'start_ticks':'9','job_sha256':s.digest(job),'remote':'/root/test','label':'train'})
   with patch.object(s.subprocess,'check_output',return_value='MISSING'),patch.object(c,'scp') as scp:
    with self.assertRaisesRegex(RuntimeError,'no retry'):c.remote(pathlib.Path(d),'/root/test',job,'train')
    scp.assert_not_called()
if __name__=='__main__':unittest.main()
