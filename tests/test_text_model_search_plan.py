import unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.storage import canonical
import base64
import hashlib
from ops.probe_text_model_search import approved,source_membership
from pathlib import Path
class SearchApprovalControls(unittest.TestCase):
 def signed(self,**changes):
  k=SigningKey.generate();p=dict(schema=1,payable=False,chain_transactions=False,search_budget=32,indices=[0,1],**changes)
  return dict(payload=p,signer=k.verify_key.encode().hex(),signature=base64.b64encode(k.sign(canonical(p)).signature).decode()),k.verify_key.encode().hex()
 def test_wrong_authority_rejected_before_model(self):
  d,k=self.signed()
  with self.assertRaises(ValueError):approved(d,'00'*32)
 def test_heldout_indices_rejected(self):
  d,k=self.signed();d['payload']['indices']=[16];key=SigningKey.generate();d.update(signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(d['payload'])).signature).decode())
  with self.assertRaises(ValueError):approved(d,d['signer'])
 def test_changed_signed_payload_rejected(self):
  d,k=self.signed();d['payload']['search_budget']=33
  with self.assertRaises(Exception):approved(d,k)
 def test_source_membership_rejects_unlisted_shadow(self):
  actual={str(p):'00'*32 for p in Path('subnet').glob('*.py')}
  source_membership(actual)
  with self.assertRaises(ValueError):source_membership({**actual,'subnet/injected.py':'00'*32})
  with self.assertRaises(ValueError):source_membership({})
  actual.pop(next(iter(actual)))
  with self.assertRaises(ValueError):source_membership(actual)
 def valid_text_plan(self,source):
  from subnet.backend_jobs import BACKEND_PROFILE,NUMERICAL_POLICY
  return self.signed(environment=dict(id=source,adapter='prime_v1'),
   harness=dict(policy='autoregressive'),backend_profile=BACKEND_PROFILE,
   numerical_policy=NUMERICAL_POLICY,
   source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('subnet').glob('*.py')},
   probe_sha256=hashlib.sha256(Path('ops/probe_text_model_search.py').read_bytes()).hexdigest())
 def test_original_science_and_trivia_abstain_approved_without_model_execution(self):
  for source in ['affine_science','affine_trivia_abstain']:
   d,k=self.valid_text_plan(source)
   with self.subTest(source=source):self.assertEqual(approved(d,k)['environment']['id'],source)
 def test_curated_override_and_unknown_source_rejected(self):
  for source,harness in [('affine_science',dict(policy='autoregressive',turn_overrides={'0':dict(policy='candidates',candidates=['a','b'])})),('unknown',dict(policy='autoregressive'))]:
   d,k=self.valid_text_plan(source);key=SigningKey.generate();d['payload']['harness']=harness
   d.update(signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(d['payload'])).signature).decode())
   with self.subTest(source=source),self.assertRaisesRegex(ValueError,'unrestricted sampling'):approved(d,d['signer'])
