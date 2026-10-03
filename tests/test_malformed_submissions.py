"""Reject miner framing without suppressing operator/runtime failures."""
import copy,io,json,struct,unittest,zipfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from subnet.backend_jobs import audit
from subnet.batches import pack,submission_records
from subnet.artifact_budget import LEGACY
from subnet.scoring import score

def archive(manifest=None,extra=None):
 stream=io.BytesIO()
 with zipfile.ZipFile(stream,'w') as z:
  if manifest is not None:z.writestr('manifest.json',manifest)
  for name,body in (extra or {}).items():z.writestr(name,body)
 return stream.getvalue()
class MalformedSubmissions(unittest.TestCase):
 def setUp(self):
  self.manifest=dict(epoch='nonpayable-test',checkpoint={'id':'approved'},K=1,L=1,max_batches=1,environment={'id':'env'},harness={},indices=[0])
  self.batch=dict(schema=2,epoch='nonpayable-test',checkpoint='approved',env_id='env',environment_version='v1',index=0,sample_index=0,rollouts=[dict(index=0,env_id='env',classification=c,reward=r,turns=[{'output':[i]}]) for c,r,i in [('positive',1,1),('negative',0,2)]])
  self.arrays=[[np.zeros((1,2),dtype=np.float32)],[np.zeros((1,2),dtype=np.float32)]]
  self.runtime=SimpleNamespace(spec=SimpleNamespace(version='v1'),verify=lambda *args:True)
  self.runtime.for_environment=lambda *args:self.runtime
 def test_malformed_archives_zero_credit_and_never_touch_runtime(self):
  cases=[b'not zip',archive(),archive('{'),archive('{}'),archive('[null]'),archive('[{"batch":{}}]'),archive('[{"batch":{},"arrays":[["missing.npy"]]}]'),archive('[{"batch":{},"arrays":[["bad.npy"]]}]',{'bad.npy':b'not npy'}),pack([({},[]),({},[])])]
  for data in cases:
   with self.subTest(size=len(data)),patch.object(self.runtime,'for_environment',side_effect=AssertionError('runtime touched')):
    report,pairs=audit(data,self.manifest,self.runtime);self.assertEqual(pairs,[]);self.assertEqual(report['accepted'],[]);self.assertTrue(report['submission_rejected']);self.assertEqual(score({'malformed':report})['points'],{'malformed':0});self.assertFalse(score({'malformed':report})['provisional'])
 def test_encryption_unknown_compression_crc_and_truncation(self):
  original=archive('[{"batch":{},"arrays":[]}]');central=original.index(b'PK\x01\x02')
  encrypted=bytearray(original);struct.pack_into('<H',encrypted,central+8,1)
  compression=bytearray(original);struct.pack_into('<H',compression,central+10,99)
  crc=bytearray(original);crc[original.index(b'batch')]=ord('B')
  for label,data in [('encrypted',bytes(encrypted)),('compression',bytes(compression)),('CRC',bytes(crc)),('truncated',original[:-8]),('UTF8',archive(b'\xff'))]:
   with self.subTest(label=label),patch.object(self.runtime,'for_environment',side_effect=AssertionError('runtime touched')):
    report,pairs=audit(data,self.manifest,self.runtime);self.assertTrue(report['submission_rejected']);self.assertEqual(pairs,[])
 def test_valid_other_submission_keeps_credit_after_bad_submission(self):
  rejected,_=audit(b'bad zip',self.manifest,self.runtime);valid,pairs=audit(pack([(self.batch,self.arrays)]),self.manifest,self.runtime)
  self.assertEqual(len(pairs),1);scores=score({'bad':rejected,'good':valid});self.assertEqual(scores['points'],{'bad':0,'good':1});self.assertEqual(scores['weights'],{'bad':0.,'good':1.})
 def test_valid_framing_nonobject_batch_is_rejected(self):
  report,pairs=audit(pack([(None,[])]),self.manifest,self.runtime);self.assertEqual(pairs,[]);self.assertFalse(report['outcomes'][0]['valid'])
 def test_wrong_operator_policy_and_manifest_still_raise(self):
  for update in ({'artifact_policy':'unapproved'},{'max_batches':True},{'environments':[]}):
   with self.subTest(update=update),self.assertRaises(ValueError):audit(b'bad zip',dict(self.manifest,**update),self.runtime)
 def test_runtime_and_infrastructure_failures_are_not_bad_submissions(self):
  valid=pack([(self.batch,self.arrays)])
  for error in (ValueError('approved runtime mismatch'),RuntimeError('model unavailable'),OSError('disk unavailable'),MemoryError('allocation unavailable')):
   with self.subTest(error=type(error).__name__),patch.object(self.runtime,'for_environment',side_effect=error),self.assertRaises(type(error)):audit(valid,self.manifest,self.runtime)
  for error in (OSError('decoder backend unavailable'),RuntimeError('unexpected decoder failure')):
   with patch('subnet.batches.unpack',side_effect=error),self.assertRaises(type(error)):submission_records(valid,budget=LEGACY,max_batches=1)
 def test_bad_policy_refused_before_decoder(self):
  with patch('subnet.batches.unpack',side_effect=AssertionError('decoder touched')):
   for budget,quota in ((dict(LEGACY,tensor_rows=100),1),(LEGACY,False)):
    with self.assertRaises(ValueError):submission_records(b'bad',budget=budget,max_batches=quota)
if __name__=='__main__':unittest.main()
