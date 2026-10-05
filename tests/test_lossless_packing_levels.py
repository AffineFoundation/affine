import hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from subnet.batches import pack,unpack,UploadBudgetExceeded,compression_for_manifest
from subnet.artifact_budget import LEGACY
from subnet.commitment_transport import check_prepared_cumulative
from subnet.storage import canonical
class Controls(unittest.TestCase):
 def setUp(self):
  self.batch=dict(env_id='math',index=7);self.arrays=[[np.random.default_rng(7).normal(size=(32,1000)).astype(np.float32)],[np.arange(32000,dtype=np.float32).reshape(32,1000)]]
 def test_default_exact_historical_bytes_all_levels_exact_array_and_metadata(self):
  baseline=pack([(self.batch,self.arrays)],stable=True)
  historical=io.BytesIO();refs=[]
  with zipfile.ZipFile(historical,'w',compression=zipfile.ZIP_DEFLATED)as z:
   def write(name,data):
    info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o600<<16;z.writestr(info,data)
   for ri,turns in enumerate(self.arrays):
    row=[]
    for ti,a in enumerate(turns):
     name=f'0-{ri}-{ti}.npy';buf=io.BytesIO();np.save(buf,a,allow_pickle=False);write(name,buf.getvalue());row.append(name)
    refs.append(row)
   write('manifest.json',canonical([dict(batch=self.batch,arrays=refs)]))
  self.assertEqual(baseline,historical.getvalue(),'exact previous default writer bytes')
  self.assertEqual(baseline,pack([(self.batch,self.arrays)],stable=True,compression_level=6))
  with zipfile.ZipFile(io.BytesIO(baseline))as z:original={e.filename:hashlib.sha256(z.read(e.filename)).hexdigest()for e in z.infolist()}
  for level in (0,1,3,6,9):
   data=pack([(self.batch,self.arrays)],stable=True,compression_level=level)
   self.assertEqual(data,pack([(self.batch,self.arrays)],stable=True,compression_level=level))
   with zipfile.ZipFile(io.BytesIO(data))as z:
    self.assertEqual({e.filename:hashlib.sha256(z.read(e.filename)).hexdigest()for e in z.infolist()},original)
    self.assertTrue(all(e.compress_type==zipfile.ZIP_DEFLATED for e in z.infolist()))
   decoded=unpack(data);self.assertEqual(decoded[0][0],self.batch)
   for original_rows,decoded_rows in zip(self.arrays,decoded[0][1]):
    for a,b in zip(original_rows,decoded_rows):self.assertTrue(np.array_equal(a,b));self.assertEqual(a.tobytes(),b.tobytes())
   check_prepared_cumulative([(self.batch,data)],{},3)
 def test_invalid_levels_fail_before_allocating_and_historical_caps_not_loosened(self):
  for level in (True,-1,10,1.0,'1',None):
   with patch('subnet.batches.io.BytesIO',side_effect=AssertionError('no allocation')),self.assertRaisesRegex(ValueError,'compression level'):pack([],compression_level=level)
  large=np.ones((512,60000),dtype=np.float32)
  with self.assertRaises(UploadBudgetExceeded):pack([(self.batch,[[large]])],stable=True,compression_level=0)
  with self.assertRaises(UploadBudgetExceeded):pack([(self.batch,[[large]*5])],stable=True,compression_level=1)
 def test_signed_manifest_selects_one_pass_level_and_invalid_policy_never_packs(self):
  from subnet.commitment_transport import pair_artifact
  self.assertEqual(compression_for_manifest({}),6)
  manifest={'artifact_compression_policy':{'version':'lossless-deflate-v1','level':1}}
  self.assertEqual(pair_artifact(self.batch,self.arrays,manifest),pack([(self.batch,self.arrays)],stable=True,compression_level=1))
  for policy in (None,{}, {'version':'other','level':1},{'version':'lossless-deflate-v1','level':True},{'version':'lossless-deflate-v1','level':10},{'version':'lossless-deflate-v1','level':1,'extra':True}):
   with patch('subnet.batches.pack',side_effect=AssertionError('no invalid allocation')),self.assertRaises(ValueError):pair_artifact(self.batch,self.arrays,{'artifact_compression_policy':policy})
 def test_actual_epoch_signature_binds_prospective_policy_and_config_copy(self):
  from test_real_gpu_epoch_open import MemoryBucket
  from subnet.controller import Controller
  from subnet.storage import Gateway,Identity
  from subnet.backend_jobs import signed
  from subnet.gpu_service import contract
  raw=dict(version='lossless-deflate-v1',level=1)
  row=dict(spec=dict(id='math',version='fixed-v1',num_samples=4,max_output_tokens=512),indices=[0],harness=dict(version='text-tools-v1'))
  with patch('subnet.gpu_service.definitions',return_value=[row]):
   chosen=contract(dict(source_bundle={},heldout=[],artifact_compression_policy=raw),0)
  raw['level']=6;self.assertEqual(chosen['artifact_compression_policy']['level'],1)
  with tempfile.TemporaryDirectory()as folder:
   bucket=MemoryBucket();gateway=Gateway(bucket,state_path=Path(folder)/'gateway.json',direct_r2=True)
   try:
    controller=Controller(bucket,gateway,Path(folder)/'controller');identity=Identity()
    manifest=controller.open('nonpayable-compression-control',dict(id='a'*64,files={}),[identity.id],source_bundle={'sha256':'b'*64},submission_transport_policy='small-commitment-pairs-v1',artifact_compression_policy=chosen['artifact_compression_policy'])
    actual=signed(json.loads(bucket.objects['public/nonpayable-compression-control/manifest.json']),controller.authority.id)
    self.assertEqual(actual['artifact_compression_policy'],dict(version='lossless-deflate-v1',level=1));self.assertEqual(actual,manifest)
   finally:gateway.server.shutdown();gateway.server.server_close();gateway.thread.join()
