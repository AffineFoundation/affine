import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from subnet.publication import history,publish_source_bundle

class Bucket:
 def __init__(self):self.data={}
 def presign(self,key):return 'https://bucket.r2.cloudflarestorage.com/'+key
 def get(self,key):return self.data[key]
 def put(self,key,value):self.data[key]=value

class PublicationTest(unittest.TestCase):
 def test_frozen_audit_routes_never_expose_staging_and_bind_source(self):
  with tempfile.TemporaryDirectory() as p:
   state=Path(p);bucket=Bucket();controller=SimpleNamespace(state=state,bucket=bucket,authority=SimpleNamespace(id='a'*64))
   (state/'e-manifest.json').write_text(json.dumps(dict(deadline=123,checkpoint=dict(id='b'*64,files={'config.json':'c'*64}))))
   archive=state/'source.tar.gz';archive.write_bytes(b'reviewed source')
   descriptor=publish_source_bundle(controller,archive)
   result=dict(epoch_id='e',receipts={'d'*64:dict(frozen_key='public/e/submissions/d.zip',sha256='e'*64,size=12)},payable=False)
   report=history(controller,[result],descriptor)
   self.assertEqual(report['epochs'][0]['source_bundle']['sha256'],descriptor['sha256'])
   self.assertIn('/public/e/submissions/',report['epochs'][0]['frozen']['d'*64]['url'])
   result['receipts']['d'*64]['frozen_key']='private/e/staging/d.zip'
   with self.assertRaises(ValueError):history(controller,[result])
 def test_source_archive_is_immutable(self):
  with tempfile.TemporaryDirectory() as p:
   archive=Path(p)/'source.tar.gz';archive.write_bytes(b'source');bucket=Bucket();c=SimpleNamespace(bucket=bucket)
   first=publish_source_bundle(c,archive);self.assertEqual(first,publish_source_bundle(c,archive))
   bucket.data[first['key']]=b'corrupt'
   with self.assertRaises(ValueError):publish_source_bundle(c,archive)
