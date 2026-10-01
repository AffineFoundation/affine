import unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.storage import canonical
import base64
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
