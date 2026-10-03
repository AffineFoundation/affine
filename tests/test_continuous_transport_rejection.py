import copy,hashlib,io,tarfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from subnet.backend_jobs import audit
from subnet.batches import pack
from subnet.scoring import score
from subnet.artifact_budget import LONG,LONG_REVISION
from subnet.backend_profiles import profile,HOPPER_REVISION
from ops.check_gpu_continuous_evidence import (check_frozen_submission_audit,unpack_authenticated_epoch,TRANSPORT_REJECTION_SOURCES)

class TransportEvidence(unittest.TestCase):
 def setUp(self):
  revision,backend,numerical=profile(HOPPER_REVISION)
  self.manifest=dict(epoch='nonpayable-hopper',model_runtime_revision=revision,backend_profile=backend,numerical_policy=numerical,artifact_policy=LONG_REVISION,checkpoint={'id':'approved'},K=1,L=1,max_batches=1,environment={'id':'env'},harness={},indices=[0])
  self.body=b'bad ZIP';self.receipt={'size':len(self.body),'sha256':hashlib.sha256(self.body).hexdigest()}
  class Sentinel:
   def for_environment(self,*args):raise AssertionError('model touched')
  self.rejected=audit(self.body,self.manifest,Sentinel())[0]
  from ops.archive_harness_identity import MODULES
  self.sources={};stream=io.BytesIO()
  with tarfile.open(fileobj=stream,mode='w:gz') as archive:
   for name in (*MODULES,*TRANSPORT_REJECTION_SOURCES):
    body=Path(name).read_bytes();self.sources[name]=hashlib.sha256(body).hexdigest();member=tarfile.TarInfo(name);member.size=len(body);archive.addfile(member,io.BytesIO(body))
  self.archive=stream.getvalue();self.descriptor={'size':len(self.archive),'sha256':hashlib.sha256(self.archive).hexdigest()};self.source=(self.archive,self.descriptor,self.sources)
 def check(self,report=None,body=None,receipt=None,source=None,manifest=None):
  return check_frozen_submission_audit(self.body if body is None else body,self.manifest if manifest is None else manifest,self.rejected if report is None else report,self.receipt if receipt is None else receipt,rejection_source=self.source if source is None else source)
 def test_exact_rejection_uses_archived_reviewed_decoder_and_zero_credit(self):
  self.assertEqual(self.check(),[]);self.assertEqual(score({'bad':self.rejected})['total'],0)
  self.assertFalse(score({'bad':self.rejected})['provisional'])
 def test_forged_integrity_report_or_source_refused(self):
  for field,value in [('epoch','other'),('submission_sha256','0'*64),('accepted',[{}]),('outcomes',[]),('rejection_stage','inference'),('training_eligibility','unchecked')]:
   bad=copy.deepcopy(self.rejected);bad[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):self.check(report=bad)
  with self.assertRaises(ValueError):self.check(body=self.body+b'x')
  with self.assertRaises(ValueError):self.check(receipt=dict(self.receipt,size=100))
  with self.assertRaises(ValueError):self.check(source=(self.archive,self.descriptor,dict(self.sources,**{'subnet/batches.py':'0'*64})))
  with self.assertRaises(ValueError):self.check(source=(self.archive+b'x',self.descriptor,self.sources))
  with self.assertRaises(ValueError):check_frozen_submission_audit(self.body,self.manifest,self.rejected,self.receipt)
  with patch('subnet.batches.__file__',str(Path('subnet/artifact_budget.py').resolve())),self.assertRaisesRegex(ValueError,'local rejection decoder'):self.check()
 def test_valid_zip_cannot_be_claimed_as_transport_rejection(self):
  body=pack([({'index':0},[])],budget=LONG);receipt={'size':len(body),'sha256':hashlib.sha256(body).hexdigest()};bad=dict(self.rejected,submission_sha256=receipt['sha256'])
  with self.assertRaisesRegex(ValueError,'valid transport falsely'):self.check(report=bad,body=body,receipt=receipt)
 def test_normal_hopper_long_arrays_and_accepted_index_binding(self):
  batch={'index':0};body=pack([(batch,[[np.zeros((513,2),dtype=np.float32)]])],budget=LONG);receipt={'size':len(body),'sha256':hashlib.sha256(body).hexdigest()};report={'epoch':self.manifest['epoch'],'submission_sha256':receipt['sha256'],'outcomes':[{'batch':0,'valid':True,'fully_audited':True}],'accepted':[batch]}
  self.assertEqual(check_frozen_submission_audit(body,self.manifest,report,receipt),[batch]);self.assertEqual(unpack_authenticated_epoch(body,self.manifest)[0][1][0][0].shape,(513,2))
  for number in (True,1,-1):
   changed=copy.deepcopy(report);changed['outcomes'][0]['batch']=number
   with self.assertRaises(ValueError):check_frozen_submission_audit(body,self.manifest,changed,receipt)
  duplicate=copy.deepcopy(report);duplicate['outcomes']*=2;duplicate['accepted']*=2
  with self.assertRaises(ValueError):check_frozen_submission_audit(body,self.manifest,duplicate,receipt)
 def test_policy_and_infrastructure_refusals_are_not_rejections(self):
  for change in ({'artifact_policy':'arbitrary'},{'backend_profile':{}},{'numerical_policy':{}}):
   with self.subTest(change=change),self.assertRaises(ValueError):self.check(manifest=dict(self.manifest,**change))
  for error in (OSError('storage failure'),RuntimeError('decoder environment failure')):
   with patch('subnet.batches.submission_records',side_effect=error),self.assertRaises(type(error)):self.check()
 def test_invalid_duplicate_miner_claim_does_not_dilute_valid_credit(self):
  valid={'accepted':[{'env_id':'env','index':0,'checkpoint':'approved'}],'outcomes':[{'valid':True,'fully_audited':True}]}
  self.check();points=score({'good':valid,'malformed':self.rejected});self.assertEqual(points['points'],{'good':1,'malformed':0});self.assertEqual(points['weights']['good'],1.)
if __name__=='__main__':unittest.main()
