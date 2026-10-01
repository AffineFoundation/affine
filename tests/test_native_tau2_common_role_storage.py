import hashlib,io,unittest
from unittest.mock import Mock
from subnet.native_tau2_common_role_storage import exact_read
class StoredRoleTests(unittest.TestCase):
 def setUp(self):
  self.raw=b'private-array';self.row={'key':'private/native-tau2-common-live/epoch/roles/role-0.npy','size':len(self.raw),'sha256':hashlib.sha256(self.raw).hexdigest()};self.bucket=Mock(name='bucket');self.bucket.name='dedicated';self.bucket.client.get_object.return_value={'ContentLength':len(self.raw),'Body':io.BytesIO(self.raw)}
 def test_exact_byte_read_is_storage_only(self):self.assertEqual(exact_read(self.bucket,self.row),self.raw)
 def test_corrupt_bytes_reject(self):
  self.bucket.client.get_object.return_value['Body']=io.BytesIO(b'x'*len(self.raw))
  with self.assertRaises(ValueError):exact_read(self.bucket,self.row)
 def test_size_reject_before_read(self):
  self.bucket.client.get_object.return_value['ContentLength']=999
  with self.assertRaises(ValueError):exact_read(self.bucket,self.row)
 def test_other_bucket_prefix_reject(self):
  self.row['key']='public/secret'
  with self.assertRaises(ValueError):exact_read(self.bucket,self.row)
  self.bucket.client.get_object.assert_not_called()
 def test_unbounded_header_reject(self):
  self.row['size']=2**40
  with self.assertRaises(ValueError):exact_read(self.bucket,self.row)
  self.bucket.client.get_object.assert_not_called()
if __name__=='__main__':unittest.main()

class OffloadTests(unittest.TestCase):
 def run_case(self,corrupt):
  import json,pathlib,tempfile
  from nacl.signing import SigningKey
  from unittest.mock import patch
  from subnet.native_tau2_common_service import sign
  from subnet import native_tau2_common_role_storage as s
  key=SigningKey.generate();authority=key.verify_key.encode().hex();raw=b'array-data';manifest={'epoch':'new'};epoch=sign(manifest,key)
  with tempfile.TemporaryDirectory() as d:
   out=pathlib.Path(d);roles=out/'roles';roles.mkdir();path=roles/'role-0.npy';path.write_bytes(raw);receipt=sign({'manifest_sha256':s.digest(manifest),'probabilities_file':path.name,'probabilities_sha256':hashlib.sha256(raw).hexdigest()},key);bucket=Mock();bucket.name='dedicated';bucket.client.get_object.return_value={'ContentLength':len(raw),'Body':io.BytesIO(b'x'*len(raw) if corrupt else raw)}
   with patch.object(s,'AUTHORITY',authority),patch('subnet.long_context_runtime.AUTHORITY',authority):
    if corrupt:
     with self.assertRaises(ValueError):s.offload(bucket,path,'private/native-tau2-common-live/new/role-0.npy',receipt,key,epoch)
     self.assertTrue(path.exists());self.assertFalse((out/'signed-private-role-storage.json').exists())
    else:
     s.offload(bucket,path,'private/native-tau2-common-live/new/role-0.npy',receipt,key,epoch);self.assertFalse(path.exists());record=s.authenticate(json.loads((out/'signed-private-role-storage.json').read_bytes()),authority);self.assertFalse(record['model_or_native_admission_claimed']);self.assertTrue(record['uploaded_exact_bytes_verified'])
 def test_failed_readback_preserves_original_array(self):self.run_case(True)
 def test_signed_unadmitted_inventory_precedes_cache_removal(self):self.run_case(False)
