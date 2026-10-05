"""Real >100MB ZIP proves owned upload honors ONLY approved manifest budgets."""
import io,json,tempfile,time,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from nacl.signing import SigningKey
from training_receipt_fixtures import transport_fixture
from subnet.backend_jobs import owned_commitment_upload
from subnet.batches import unpack
from subnet.artifact_budget import LEGACY,LONG,LONG_REVISION
from subnet.commitment_transport import VERSION,validate
from subnet.storage import canonical,Identity

class OwnedUploadBudgetTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  f=transport_fixture(SigningKey.generate());cls.original_manifest=f['manifest'];cls.batch=unpack(f['data'])[0][0]
  # Two actual float32 NPY frames (160*200000*4=128MB raw) in a supported
  # uncompressed ZIP. No mocked decoder/size and no executable/padding entries.
  tensor=np.zeros((80,200000),dtype=np.float32);buf=io.BytesIO();np.save(buf,tensor,allow_pickle=False);npy=buf.getvalue()
  out=io.BytesIO()
  with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED)as z:
   z.writestr('0-0-0.npy',npy);z.writestr('0-1-0.npy',npy)
   z.writestr('manifest.json',canonical([dict(batch=cls.batch,arrays=[['0-0-0.npy'],['0-1-0.npy']])]))
  cls.large=out.getvalue();assert LEGACY['compressed_bytes']<len(cls.large)<LONG['compressed_bytes']
 @classmethod
 def tearDownClass(cls):del cls.large
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.identity=Identity();self.seed=Path(self.temp.name)/'miner.seed';self.seed.write_text(self.identity.key.encode().hex());self.seed.chmod(0o600)
  self.manifest=dict(self.original_manifest,source_bundle={'sha256':'b'*64},deadline=time.time()+60,submission_transport_policy=VERSION,max_batches=1)
  self.job=dict(miner_id=self.identity.id,miner_identity_file=str(self.seed),capability=dict(put_url='SMALL',batch_put_urls=['PAIR0'],headers={}))
 def test_real_oversized_pair_rejected_legacy_but_uploaded_with_signed_long_budget(self):
  with self.assertRaisesRegex(ValueError,'compressed upload budget'):unpack(self.large)
  m=dict(self.manifest,artifact_policy=LONG_REVISION)
  from subnet.backend_profiles import for_config
  revision,profile,policy=for_config({'model_runtime_revision':'cuda-bf16-eager-sm90-v1'})
  m.update(model_runtime_revision=revision,backend_profile=profile,numerical_policy=policy)
  objects={}
  def put(url,**kw):objects[url]=kw['data'];return SimpleNamespace(status_code=200)
  with patch('requests.put',side_effect=put):owned_commitment_upload(self.job,m)(self.large,60)
  self.assertEqual(set(objects),{'SMALL','PAIR0'})
  records=unpack(objects['PAIR0'],budget=LONG);self.assertEqual(records[0][0],self.batch)
  self.assertEqual(records[0][1][0][0].shape,(80,200000));self.assertTrue(np.all(records[0][1][1][0]==0))
  claims=validate(objects['SMALL'],m['epoch'],self.identity.id,1)['payload']
  self.assertEqual(claims['batches'][0]['size'],len(objects['PAIR0']))
 def test_same_real_zip_cannot_expand_legacy_or_unapproved_manifest(self):
  for policy in (None,'unsigned-bigger-budget'):
   m=dict(self.manifest)
   if policy is not None:m['artifact_policy']=policy
   with patch('requests.put')as put:
    with self.assertRaises(ValueError):owned_commitment_upload(self.job,m)(self.large,60)
    put.assert_not_called()
