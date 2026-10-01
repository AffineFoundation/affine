import hashlib,unittest
from unittest.mock import patch
from ops.check_gpu_continuous_evidence import read_source_bundle
class Bucket:
 def get(self,key):return b'old'
class Response:
 status_code=200
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def iter_content(self,size):yield self.body
class Tests(unittest.TestCase):
 def descriptor(self):return {'key':'old-key','size':3,'sha256':hashlib.sha256(b'new').hexdigest(),'url':'https://test.r2.cloudflarestorage.com/b/source?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=not-real'}
 def test_signed_url_recovers_stale_key_with_exact_bytes(self):
  r=Response();r.body=b'new'
  with patch('requests.get',return_value=r) as call:
   self.assertEqual(read_source_bundle(Bucket(),self.descriptor()),(b'new','signed-url-stale-key'))
   self.assertFalse(call.call_args.kwargs['allow_redirects'])
 def test_changed_or_oversized_signed_url_body_fails(self):
  for body in [b'bad',b'long']:
   r=Response();r.body=body
   with patch('requests.get',return_value=r),self.assertRaises(ValueError):read_source_bundle(Bucket(),self.descriptor())
 def test_unapproved_url_cannot_override_object_hash(self):
  d=self.descriptor();d['url']='https://example.com/source'
  with patch('requests.get') as call:
   with self.assertRaises(ValueError):read_source_bundle(Bucket(),d)
   call.assert_not_called()
 def test_valid_key_does_not_need_expiring_capability(self):
  d=self.descriptor();d['sha256']=hashlib.sha256(b'old').hexdigest();d.pop('url')
  with patch('requests.get') as call:self.assertEqual(read_source_bundle(Bucket(),d),(b'old','object-key'));call.assert_not_called()
if __name__=='__main__':unittest.main()
