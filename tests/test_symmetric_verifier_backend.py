"""CPU integration through original backend branches/source-aware admission, no GPU."""
from pathlib import Path
import copy,hashlib,json,sys,unittest,types,base64
from unittest.mock import patch
import numpy as np
from nacl.signing import SigningKey
D=Path(__file__).resolve().parents[1];sys.path[:0]=[str(D),str(D/'tests')]
from subnet import forced_sampling as f,fast_prefill_audit as fast,symmetric_verifier_policy as policy
from subnet import backend_jobs
from subnet.source_sampling_admission import SamplingAdmission
from subnet.batches import pack
from subnet.audit_policy import InvalidSample
import test_fast_prefill_audit as fixture

class BackendTests(unittest.TestCase):
 def setUp(self):
  self.r,self.m=fixture.Controls().support_runtime();self.m.update(K=4,L=4,max_batches=9,source_bundle={'sha256':'e'*64},environment={'id':'env'},harness={},indices=[0]);self.m['sampling_contract'].update(version=f.MINER_VERSION,max_attempts=1000);self.m[policy.FIELD]=policy.current_policy();self.identity='d'*64;f.bind_runtime(self.r,self.m,self.identity)
  self.r.for_environment=lambda spec,harness:self.r;self.r.spec=types.SimpleNamespace(id='env',version='v1')
  self.batch=dict(schema=2,epoch=self.m['epoch'],checkpoint=self.m['checkpoint']['id'],env_id='env',environment_version='v1',index=0,sample_index=0,rollouts=[dict(index=0,sample_index=0,env_id='env',environment_version='v1',task_hash='c'*64,seed=i,sampling=f.receipt(self.r.sampling_context,i),classification='positive'if i<4 else'negative',reward=1 if i<4 else 0,turns=[{'output':[i+1,6]}])for i in range(8)])
  self.arrays=[[np.zeros((2,16),dtype=np.float32)]for _ in range(8)]
  self.mode='calibrated-interior'
  def verify(roll,arr):
   self.r.sampling_verification_events.append(dict(rollout_sha256=policy.digest(roll),turn=0,positions=2,seconds=.1,adjudication=self.mode))
   if self.mode=='exact-cached-unavailable':raise fast.NumericalAmbiguity('unavailable')
   if self.mode=='exact-cached-mismatch':raise InvalidSample('mismatch')
   return True
  self.r.verify=verify
 def audit(self):return backend_jobs._audit(pack([(self.batch,self.arrays)]),self.m,self.r,commitment_miner=self.identity)
 def test_backend_pass_retains_all_eight_events(self):
  report,pairs=self.audit();self.assertEqual(len(pairs),4);self.assertEqual(len(report['outcomes'][0]['sampling_verification_events']),8);policy.require_report(self.m,report)
 def test_backend_unavailable_unknown_not_confirmed_invalid(self):
  self.mode='exact-cached-unavailable';report,pairs=self.audit();o=report['outcomes'][0];self.assertIs(o['valid'],None);self.assertFalse(o['fully_audited']);self.assertEqual(o['failure_kind'],'numerical_ambiguous');self.assertFalse(report['accepted']);policy.require_report(self.m,report)
 def test_backend_exact_mismatch_is_confirmed_invalid(self):
  self.mode='exact-cached-mismatch';report,pairs=self.audit();self.assertEqual(report['outcomes'][0]['failure_kind'],'confirmed_invalid');self.assertFalse(report['accepted']);policy.require_report(self.m,report)
 def test_source_admission_enforces_events(self):
  report,pairs=self.audit();key=SigningKey.generate();authority=key.verify_key.encode().hex();sign=lambda p:dict(payload=p,signer=authority,signature=base64.b64encode(key.sign(f.canonical(p)).signature).decode())
  files={str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(D/'subnet').glob('*.py')}
  # A complete local source closure makes this admission control portable.
  row=dict(runtime_files=files,runtime_versions={'torch':'CPU-control-only'},sampling_versions=[f.MINER_VERSION]);gate=SamplingAdmission(sign(dict(version='source-specific-sampling-api-admission-v1',sources={'e'*64:row})),authority,{'e'*64:D})
  # Exact source/calibration validation needs the same fixture harness available
  # in environments; spec structure itself is checked by ordinary backend tests.
  self.m['environments']=[dict(harness=self.r.harness)]
  report['sampling_miner']=self.identity
  job=dict(manifest=sign(self.m),source_files=files,runtime_versions=row['runtime_versions'],submissions=[dict(sha256=report['submission_sha256'],commitment_miner=self.identity)])
  gate.check(job);self.assertTrue(gate.report(job,dict(audits=[report])))
  bad=copy.deepcopy(report);bad['outcomes'][0]['sampling_verification_events'].pop()
  with self.assertRaises(ValueError):gate.report(job,dict(audits=[bad]))
if __name__=='__main__':unittest.main()
