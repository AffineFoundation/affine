import base64,copy,json,tempfile,unittest
from pathlib import Path
from nacl.signing import SigningKey
from ops.trainer_lifecycle.prospective_collection_timing import validate_issued_window
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
class IssuedWindow(unittest.TestCase):
 def setUp(self):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name);self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();self.epoch='nonpayable-test-93';self.manifest=dict(epoch=self.epoch,start=100,deadline=1299,checkpoint=dict(id='1'*64));self.signed=dict(payload=self.manifest,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(self.manifest)).signature).decode());self.path=self.root/(self.epoch+'-first-signed-manifest.json');self.path.write_bytes(canonical(self.signed));self.gateway={'epochs':{self.epoch:dict(start=100,deadline=1299)}};self.gpath=self.root/'gateway.json';self.gpath.write_bytes(canonical(self.gateway))
 def check(self,manifest=None,**changes):
  kwargs=dict(prospective=True,state_root=self.root,epoch=self.epoch,authority=self.authority);kwargs.update(changes);return validate_issued_window(self.manifest if manifest is None else manifest,1200,**kwargs)
 def test_exact_duration_retains_original_readonly_admission(self):
  self.path.unlink();self.gpath.unlink();self.check(dict(self.manifest,deadline=1300),authority=None)
 def test_authenticated_one_second_gap_preserves_exact_bytes(self):
  before=(self.path.read_bytes(),self.gpath.read_bytes(),canonical(self.manifest));self.check();self.assertEqual(before,(self.path.read_bytes(),self.gpath.read_bytes(),canonical(self.manifest)))
 def test_1198_and_1201_are_rejected(self):
  for deadline in (1298,1301):
   with self.subTest(deadline=deadline),self.assertRaises(ValueError):self.check(dict(self.manifest,deadline=deadline))
 def test_historical_one_second_gap_is_not_newly_admitted(self):
  with self.assertRaises(ValueError):self.check(prospective=False)
 def test_unsigned_manifest_cannot_establish_exception(self):
  self.path.write_bytes(canonical(self.manifest))
  with self.assertRaises(ValueError):self.check()
 def test_local_payload_tamper_is_rejected(self):
  with self.assertRaises(ValueError):self.check(dict(self.manifest,checkpoint=dict(id='2'*64)))
 def test_modified_signed_payload_is_rejected(self):
  self.signed=copy.deepcopy(self.signed);self.signed['payload']['checkpoint']['id']='2'*64;self.path.write_bytes(canonical(self.signed))
  with self.assertRaises(ValueError):self.check(self.signed['payload'])
 def test_bad_signature_is_rejected(self):
  self.signed['signature']=base64.b64encode(b'\0'*64).decode();self.path.write_bytes(canonical(self.signed))
  with self.assertRaises(ValueError):self.check()
 def test_wrong_authority_is_rejected(self):
  with self.assertRaises(ValueError):self.check(authority=SigningKey.generate().verify_key.encode().hex())
 def test_wrong_or_missing_gateway_timing_is_rejected(self):
  for record in (None,dict(start=101,deadline=1300),dict(start=100,deadline=1300),dict(start=100.0,deadline=1299)):
   self.gpath.write_bytes(canonical(dict(epochs={self.epoch:record})))
   with self.subTest(record=record),self.assertRaises(ValueError):self.check()
 def test_wrong_epoch_is_rejected(self):
  other='nonpayable-test-94';(self.root/(other+'-first-signed-manifest.json')).write_bytes(self.path.read_bytes())
  with self.assertRaises(ValueError):self.check(epoch=other)
 def test_non_integer_manifest_timestamps_are_rejected(self):
  for field,value in [('start',100.0),('start',True),('deadline',1299.0)]:
   with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.check(dict(self.manifest,**{field:value}))
if __name__=='__main__':unittest.main()
