import base64,hashlib,unittest
from pathlib import Path
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.backend_jobs import BACKEND_PROFILE,NUMERICAL_POLICY
from ops.probe_pydantic_model_search import approved

class StructuredSearchApproval(unittest.TestCase):
 def plan(self,**changes):
  key=SigningKey.generate()
  value=dict(schema=1,payable=False,chain_transactions=False,search_budget=8,indices=[0,1],environment=dict(id='affine_pydantic',adapter='prime_v1'),harness=dict(policy='autoregressive'),backend_profile=BACKEND_PROFILE,numerical_policy=NUMERICAL_POLICY,source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('subnet').glob('*.py')},probe_sha256=hashlib.sha256(Path('ops/probe_pydantic_model_search.py').read_bytes()).hexdigest())
  value.update(changes)
  return dict(payload=value,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(value)).signature).decode())
 def test_actual_signed_source_inventory_passes_without_launching_model(self):
  envelope=self.plan();self.assertEqual(approved(envelope,envelope['signer']),envelope['payload'])
 def test_candidate_policy_and_changed_source_rejected(self):
  for changes in [dict(harness=dict(policy='candidates')),dict(probe_sha256='00'*32)]:
   envelope=self.plan(**changes)
   with self.assertRaises(ValueError):approved(envelope,envelope['signer'])
 def test_heldout_and_incorrect_authority_rejected(self):
  envelope=self.plan(indices=[16])
  with self.assertRaises(ValueError):approved(envelope,envelope['signer'])
  envelope=self.plan()
  with self.assertRaises(ValueError):approved(envelope,'00'*32)
