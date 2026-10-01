import base64,hashlib,io,tempfile,pathlib,unittest
from unittest.mock import Mock
from nacl.signing import SigningKey
from subnet.storage import canonical
from ops.rehydrate_native_tau2_cache import rehydrate

class RehydrateCacheTests(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();self.raw=b'private-zip-fixture';h=hashlib.sha256(self.raw).hexdigest()
  self.value={'version':'owned-duplicate-cumulative-cache-storage-v1','epoch':'nonpayable-native-tau2-mixed-1790869388','local_cache_name':'cumulative-1.zip','private_rehydration_key':'private/native-tau2-common-live/nonpayable-native-tau2-mixed-1790869388/final-frozen-'+h+'.zip','sha256':h,'size':len(self.raw),'exact_r2_readback_verified':True,'native_or_model_admission_claimed_by_storage':False,'active_role_arrays_touched':False,'payable':False,'chain_transactions':False}
  self.bucket=Mock();self.bucket.name='private';self.bucket.client.get_object.return_value={'Body':io.BytesIO(self.raw)}
 def signed(self):return {'payload':self.value,'signer':self.authority,'signature':base64.b64encode(self.key.sign(canonical(self.value)).signature).decode()}
 def test_exact_private_bytes_mode600_and_no_native_claim(self):
  with tempfile.TemporaryDirectory() as d:
   path=pathlib.Path(d)/'copy.zip';result=rehydrate(self.signed(),self.bucket,path,self.authority,reserve_bytes=0);self.assertEqual(path.read_bytes(),self.raw);self.assertEqual(path.stat().st_mode&0o777,0o600);self.assertFalse(result['model_or_native_verification_performed'])
 def test_forged_authority_rejected_before_object_read(self):
  with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):rehydrate(self.signed(),self.bucket,pathlib.Path(d)/'copy.zip','a'*64,reserve_bytes=0)
  self.bucket.client.get_object.assert_not_called()
 def test_wrong_length_or_hash_preserves_no_output_or_partial(self):
  for raw in (self.raw[:-1],self.raw+b'x',b'x'*len(self.raw)):
   self.bucket.client.get_object.return_value={'Body':io.BytesIO(raw)}
   with tempfile.TemporaryDirectory() as d:
    with self.assertRaises(ValueError):rehydrate(self.signed(),self.bucket,pathlib.Path(d)/'copy.zip',self.authority,reserve_bytes=0)
    self.assertEqual(list(pathlib.Path(d).iterdir()),[])
 def test_signed_unscoped_key_or_boolean_size_rejected_before_read(self):
  for field,value in [('private_rehydration_key','public/other'),('size',True)]:
   self.setUp();self.value[field]=value
   with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):rehydrate(self.signed(),self.bucket,pathlib.Path(d)/'copy.zip',self.authority,reserve_bytes=0)
   self.bucket.client.get_object.assert_not_called()
 def test_existing_output_never_overwritten(self):
  with tempfile.TemporaryDirectory() as d:
   path=pathlib.Path(d)/'copy.zip';path.write_bytes(b'existing')
   with self.assertRaises(ValueError):rehydrate(self.signed(),self.bucket,path,self.authority,reserve_bytes=0)
   self.assertEqual(path.read_bytes(),b'existing');self.bucket.client.get_object.assert_not_called()
 def test_tampered_signed_payload_rejected_before_object_read(self):
  from nacl.exceptions import BadSignatureError
  envelope=self.signed();self.value['size']+=1
  with tempfile.TemporaryDirectory() as d,self.assertRaises(BadSignatureError):rehydrate(envelope,self.bucket,pathlib.Path(d)/'copy.zip',self.authority,reserve_bytes=0)
  self.bucket.client.get_object.assert_not_called()
 def test_capacity_guard_precedes_remote_read(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d,patch('ops.rehydrate_native_tau2_cache.shutil.disk_usage',return_value=Mock(free=0)),self.assertRaisesRegex(RuntimeError,'disk-capacity'):rehydrate(self.signed(),self.bucket,pathlib.Path(d)/'copy.zip',self.authority)
  self.bucket.client.get_object.assert_not_called()
