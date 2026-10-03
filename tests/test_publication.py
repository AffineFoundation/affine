import base64,copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from subnet.publication import history,publish_source_bundle
from subnet.storage import Identity,canonical

class Bucket:
 def __init__(self):self.data={}
 def presign(self,key):return 'https://bucket.r2.cloudflarestorage.com/'+key
 def get(self,key):return self.data[key]
 def put(self,key,value):self.data[key]=value

class PublicationTest(unittest.TestCase):
 def test_bootstrap_descriptor_after_training_survives_nonempty_history(self):
  with tempfile.TemporaryDirectory() as p:
   state=Path(p);bucket=Bucket();authority=Identity(bytes(32));controller=SimpleNamespace(state=state,bucket=bucket,authority=authority)
   archive=state/'source.tar.gz';archive.write_bytes(b'exact frozen bootstrap source')
   published=publish_source_bundle(controller,archive)
   descriptor={k:v for k,v in published.items() if k!='key'}
   descriptor.update(url=bucket.presign(published['key']),read_url=bucket.presign(published['key']),format='affine-source-v1')
   # Actual bootstrap publication uses this private, digest-bound namespace.
   private_key=f"private/source-bundles/{published['sha256']}.tar.gz"
   bucket.data[private_key]=bucket.data.pop(published['key'])
   descriptor['url']=descriptor['read_url']=bucket.presign(private_key)
   original=copy.deepcopy(descriptor)
   manifest=dict(deadline=123,checkpoint=dict(id='b'*64,files={'config.json':'c'*64}),source_bundle=descriptor)
   (state/'e-manifest.json').write_text(json.dumps(manifest))
   trained='f'*64;payload=dict(id=trained,files={'model.safetensors':'e'*64})
   bucket.data[f'public/checkpoints/{trained}/authorities/{authority.id}/checkpoint.json']=canonical(dict(payload=payload,signer=authority.id,signature=base64.b64encode(authority.key.sign(canonical(payload)).signature).decode()))
   (state/'e-training-metrics.json').write_text(json.dumps(dict(checkpoint=trained,new_checkpoint=payload)))
   result=dict(epoch_id='e',receipts={},payable=False)
   for _ in range(2):
    report=history(controller,[result])
    row=report['epochs'][0]
    self.assertEqual(row['source_bundle']['read_url'],bucket.presign(published['key']))
    self.assertEqual(row['source_bundle']['url'],row['source_bundle']['read_url'])
    self.assertEqual(row['source_bundle']['sha256'],published['sha256'])
    self.assertEqual(row['source_bundle']['binding'],'epoch-signed')
    self.assertEqual(row['trained_checkpoint']['id'],trained)
    self.assertEqual(descriptor,original)
    self.assertEqual(json.loads((state/'e-manifest.json').read_text()),manifest)
    self.assertEqual(bucket.data[published['key']],bucket.data[private_key])
   # Refuse to renew a route when its content-addressed archive is corrupted.
   bucket.data[published['key']]=b'corrupt'
   with self.assertRaisesRegex(ValueError,'source archive content mismatch'):history(controller,[result])

 def test_bootstrap_source_metadata_is_checked_before_route_publication(self):
  with tempfile.TemporaryDirectory() as p:
   state=Path(p);bucket=Bucket();c=SimpleNamespace(state=state,bucket=bucket,authority=SimpleNamespace(id='a'*64))
   for source in [dict(sha256='../private',size=10),dict(sha256='b'*64,size=True),dict(sha256='b'*64,size=-1)]:
    (state/'e-manifest.json').write_text(json.dumps(dict(deadline=1,checkpoint=dict(id='c'*64,files={}),source_bundle=source)))
    with self.assertRaises(ValueError):history(c,[dict(epoch_id='e',receipts={})])

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

 def test_reviewed_reconstruction_cannot_replace_epoch_source_or_corrupt_bytes(self):
  with tempfile.TemporaryDirectory() as p:
   state=Path(p);bucket=Bucket();c=SimpleNamespace(state=state,bucket=bucket,authority=SimpleNamespace(id='a'*64))
   archive=state/'reviewed.tar.gz';archive.write_bytes(b'complete reviewed static sources')
   reconstruction=dict(publish_source_bundle(c,archive),binding='reviewed-reconstruction-not-original-epoch-archive')
   original=dict(key='public/sources/original/source.tar.gz',sha256='b'*64,size=10)
   (state/'e-manifest.json').write_text(json.dumps(dict(deadline=1,checkpoint=dict(id='c'*64,files={}),source_bundle=original)))
   ledger=[dict(epoch_id='e',receipts={})]
   report=history(c,ledger,source_reconstructions=[reconstruction])
   self.assertEqual(report['epochs'][0]['source_bundle']['sha256'],original['sha256'])
   self.assertEqual(report['epochs'][0]['source_bundle']['binding'],'epoch-signed')
   self.assertEqual(report['source_reconstruction_supplements'][0]['binding'],reconstruction['binding'])
   bucket.data[reconstruction['key']]=b'corrupt'
   with self.assertRaisesRegex(ValueError,'content mismatch'):history(c,ledger,source_reconstructions=[reconstruction])
   with self.assertRaisesRegex(ValueError,'provenance'):history(c,ledger,source_reconstructions=[dict(reconstruction,binding='epoch-signed')])
