import unittest,tempfile,pathlib,json,hashlib,base64
from nacl.signing import SigningKey
from ops import resume_existing_gpu_epoch as m
class Controls(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.t.name);self.state=self.root/'state';(self.state/'roles').mkdir(parents=True);self.cwd=self.root/'sealed';self.cwd.mkdir();(self.cwd/'worker.py').write_text('sealed');self.key=SigningKey.generate();self.authority=bytes(self.key.verify_key).hex();self.epoch='E10';self.source='a'*64;self.config={'state':str(self.state),'source_bundle':{'sha256':self.source}};self.cp=self.root/'config.json';self.write(self.cp,self.config);self.ch=hashlib.sha256(self.cp.read_bytes()).hexdigest();self.status={'active':{'epoch':'E10','phase':'train'},'initial_published':True,'checkpoint':{'id':'c'}};self.manifest={'epoch':'E10','checkpoint':{'id':'c'},'source_bundle':{'sha256':self.source}};self.job={'role':'train','job_id':'E10-train','manifest':self.sign(self.manifest),'source_files':{'worker.py':hashlib.sha256(b'sealed').hexdigest()}};self.record={'role':'train','epoch':'E10','job_id':'E10-train','job_sha256':m.digest(self.job),'manifest_sha256':m.digest(self.manifest),'source_files':self.job['source_files'],'checkpoint':'c'};self.save()
 def tearDown(self):self.t.cleanup()
 def write(self,p,d):p.write_text(json.dumps(d))
 def sign(self,p):return {'payload':p,'signature':base64.b64encode(self.key.sign(m.canonical(p)).signature).decode(),'signer':self.authority}
 def save(self):self.write(self.state/'controller.json',self.status);self.write(self.state/'roles/E10-train.json',self.record);self.write(self.state/'roles/E10-train-job.json',self.sign(self.job))
 def guard(self):return m.guard(self.cp,self.ch,self.epoch,self.authority,self.source,self.cwd)
 def test_same_original_train(self):self.assertEqual(self.guard()['job_id'],'E10-train')
 def test_same_original_after(self):self.status['active']['phase']='after';self.save();self.assertEqual(self.guard()['phase'],'after')
 def test_completed_no_new_epoch(self):self.status['active']=None;self.save();self.assertIsNone(self.guard())
 def test_config_drift(self):self.cp.write_text('{}');self.assertRaises(ValueError,self.guard)
 def test_different_epoch(self):self.status['active']['epoch']='E11';self.save();self.assertRaises(ValueError,self.guard)
 def test_opening_rejected(self):self.status['active']['phase']='opening';self.save();self.assertRaises(ValueError,self.guard)
 def test_missing_original_job(self):(self.state/'roles/E10-train-job.json').unlink();self.assertRaises(ValueError,self.guard)
 def test_job_hash_drift(self):self.record['job_sha256']='b'*64;self.save();self.assertRaises(ValueError,self.guard)
 def test_checkpoint_drift(self):self.status['checkpoint']['id']='other';self.save();self.assertRaises(ValueError,self.guard)
 def test_source_file_drift(self):(self.cwd/'worker.py').write_text('changed');self.assertRaises(ValueError,self.guard)
 def test_signature_forged(self):j=self.sign(self.job);j['signer']='f'*64;self.write(self.state/'roles/E10-train-job.json',j);self.assertRaises(ValueError,self.guard)
if __name__=='__main__':unittest.main()
