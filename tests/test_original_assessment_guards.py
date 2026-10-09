"""Use actual frozen CPU assessment admission, without production credentials."""
import base64,copy,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet import learner_blacklist_selection as original
from subnet.storage import canonical
from ops.trainer_lifecycle.opening_assessment_ordering import install

class OriginalGuards(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
  self.manifest=dict(checkpoint=dict(id='b'*64),source_bundle=dict(sha256='c'*64))
 def sign(self,p):return dict(payload=p,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(p)).signature).decode())
 def document(self,cutoff=3600,stale=False):
  a=self.sign(dict(version='hourly-current-miner-assessment-v1',assessment_stale=stale,writer_policy_sha256='d'*64,cutoff=cutoff,evidence_cutoff=cutoff,miner_estimates={}))
  return self.sign(dict(version=original.VERSION,checkpoint='b'*64,source_sha256='c'*64,target_round=93,maximum_age_seconds=3600,assessment_document=a,writer_policy_sha256='d'*64,audit_policy=dict(version='continuous-probabilistic-audit-v3',recent_epochs=16,decay=.9,prior_alpha=1.,prior_beta=1.,invalid_multiplier=.5,zero_epoch_after=2,blacklist_after=3,blacklist_epochs=4)))
 def run_original(self,document,at):
  def prepare(c,cfg,status,contract):
   result=original.admit(document,self.manifest,self.authority,at=at,round_number=status['round'])
   return dict(contract,checked=result)
  s=types.SimpleNamespace(prepare_opening=prepare);c=types.SimpleNamespace(before_open=lambda *a:a[-1]);install(s,c,earliest_round=93)
  return c.before_open(None,{},dict(round=93),s.prepare_opening(None,{},dict(round=93),{}))
 def test_exact_3600_boundary_preserved(self):
  d=self.document();self.assertEqual(self.run_original(d,7200)['checked']['assessment_cutoff'],3600)
  with self.assertRaisesRegex(ValueError,'fresh original'):self.run_original(d,7200.001)
 def test_future_or_declared_stale_rejected(self):
  for d,at in [(self.document(cutoff=7200),7199),(self.document(stale=True),3601)]:
   with self.subTest(at=at),self.assertRaisesRegex(ValueError,'fresh original'):self.run_original(d,at)
 def test_unsigned_or_changed_authenticated_evidence_rejected(self):
  for nested in (False,True):
   d=self.document()
   if nested:
    d['payload']['assessment_document']['payload']['cutoff']=0;d=self.sign(d['payload'])
   else:d['payload']['maximum_age_seconds']=7200
   with self.subTest(nested=nested),self.assertRaisesRegex(ValueError,'signature'):self.run_original(d,3601)

if __name__=='__main__':unittest.main()
