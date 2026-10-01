"""Synthetic authority/transport controls; no native/model qualification."""
import importlib,io,json,copy,zipfile,unittest
import numpy as np
from subnet import native_tau2_common_artifacts as a
class TestRoleArtifacts(unittest.TestCase):
 def setUp(self):
  f=importlib.import_module('test_native_tau2_common_search_contract').ContractTests();f.setUp();self.f=f
  self.epoch=f.sign(f.manifest);self.records,self.audit,self.report=f.sample();self.arrays={}
  for i,r in enumerate(self.records):
   buf=io.BytesIO();np.save(buf,np.zeros((len(r['payload']['output']),100),dtype=np.float32),allow_pickle=False);data=buf.getvalue();name=f'role-{i}.npy';self.arrays[name]=data;r['payload']['probabilities_file']=name;r['payload']['probabilities_sha256']=a.sha(data);self.records[i]=f.sign(r['payload']);self.report['role_checks'][i]['signed_receipt_sha256']=a.digest(self.records[i])
  self.audit=f.audit(self.records,self.report)
 def pack(self):return a.pack_sample(self.epoch,self.records,self.audit,self.report,self.arrays,self.f.authority,self.f.user)
 def rewrite(self,raw,edit):
  with zipfile.ZipFile(io.BytesIO(raw)) as archive:files={n:archive.read(n) for n in archive.namelist()}
  edit(files);out=io.BytesIO()
  with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
   for n,v in files.items():archive.writestr(n,v)
  return out.getvalue()
 def test_roundtrip_preserves_full_arrays_auxiliary_masks(self):
  raw,descriptor=self.pack();value=a.unpack_sample(raw,self.epoch,self.f.authority,self.f.user)
  self.assertEqual(value['arrays'],self.arrays);self.assertFalse(value['fresh_model_or_native_verification_performed_here']);self.assertEqual(value['view']['training_view'][0]['loss_mask'],[False]);self.assertEqual(descriptor['zip_sha256'],a.sha(raw))
 def test_wrong_report_signer_mask_and_array_rejected(self):
  self.report['reward']=0
  with self.assertRaises(ValueError):self.pack()
  self.setUp();self.arrays['role-0.npy']=self.arrays['role-0.npy']+b'junk'
  with self.assertRaises(ValueError):self.pack()
  self.setUp();self.records[0]['payload']['loss_mask']=[True];self.records[0]=self.f.sign(self.records[0]['payload']);self.report['role_checks'][0]['signed_receipt_sha256']=a.digest(self.records[0]);self.audit=self.f.audit(self.records,self.report)
  with self.assertRaises(ValueError):self.pack()
 def test_extra_private_grader_or_path_and_missing_array_rejected(self):
  raw,_=self.pack()
  for name in ('private-database.json','../role-0.npy'):
   bad=self.rewrite(raw,lambda files:files.update({name:b'{}'}))
   with self.assertRaises(ValueError):a.unpack_sample(bad,self.epoch,self.f.authority,self.f.user)
  bad=self.rewrite(raw,lambda files:files.pop('role-0.npy'))
  with self.assertRaises(ValueError):a.unpack_sample(bad,self.epoch,self.f.authority,self.f.user)
 def test_changed_file_byte_and_current_epoch_binding(self):
  raw,_=self.pack();bad=self.rewrite(raw,lambda files:files.update({'role-1.npy':files['role-1.npy']+b'x'}))
  with self.assertRaises(ValueError):a.unpack_sample(bad,self.epoch,self.f.authority,self.f.user)
  changed=copy.deepcopy(self.f.manifest);changed['epoch']='new'
  with self.assertRaises(ValueError):a.unpack_sample(raw,self.f.sign(changed),self.f.authority,self.f.user)
 def window(self):
  self.f.manifest['submission_window']={'opens_at':10.,'deadline':20.,'registered_uid':131};self.epoch=self.f.sign(self.f.manifest)
 def test_old_controls_cannot_earn_future_window(self):
  raw,_=self.pack();self.window();receipt=self.f.sign({'version':a.FREEZE_VERSION,'manifest_sha256':a.digest(self.f.manifest),'registered_uid':131,'received_at':16.,'zip_sha256':a.sha(raw),'zip_size':len(raw),'private':True,'payable':False,'chain_transactions':False})
  a.validate_freeze(receipt,raw,self.epoch,self.f.authority,131,16.)
  with self.assertRaises(ValueError):a.verify_role_window(self.records,self.epoch,self.f.authority)
  with self.assertRaises(ValueError):a.validate_freeze(receipt,raw,self.epoch,self.f.authority,131,20.)
  with self.assertRaises(ValueError):a.validate_freeze(receipt,raw+b'x',self.epoch,self.f.authority,131,16.)
 def test_signed_role_times_inside_new_window(self):
  self.window();records=[]
  for i,r in enumerate(self.records):
   value=copy.deepcopy(r['payload']);value.update(manifest_sha256=a.digest(self.f.manifest),created_at=12.+i,completed_at=13.+i);records.append(self.f.sign(value))
  self.assertTrue(a.verify_role_window(records,self.epoch,self.f.authority))
  records[0]['payload']['created_at']=9.;records[0]=self.f.sign(records[0]['payload'])
  with self.assertRaises(ValueError):a.verify_role_window(records,self.epoch,self.f.authority)
