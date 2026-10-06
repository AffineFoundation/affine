"""Source routing controls; tiny CPU protocol fixtures are not GPU qualification."""
import hashlib,json,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
from test_verifier_capacity_admission import VerifierCapacityAdmission
from ops.verifier_capacity_admission import budget,CapacityDeferred

class CapacitySelectedSource(unittest.TestCase):
 def setUp(self):
  self.fx=VerifierCapacityAdmission();self.fx.setUp();self.addCleanup(self.fx.doCleanups)
  self.source=self.fx.fx.root/'selected-source';subnet=self.source/'subnet';subnet.mkdir(parents=True)
  files={
   '__init__.py':'',
   'distributed_roles.py':'''import base64,json\nfrom nacl.signing import VerifyKey\ndef authenticate(v,a):\n if v['signer']!=a:raise ValueError('authority')\n VerifyKey(bytes.fromhex(a)).verify(json.dumps(v['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode(),base64.b64decode(v['signature']))\n return v['payload']\ndef validate_frozen_submissions(m,objs):\n if m['submission_transport_policy']!='tiny-selected-source-v2' or any(o['sha256']!='b'*64 or o['commitment_ref']['size']!=2902 for o in objs):raise ValueError('selected protocol refusal')\n''',
   'cache_lifecycle.py':'def snapshot(path):raise AssertionError("budget must not access model")\n',
   'artifact_budget.py':'def for_manifest(m):return {"compressed_bytes":100_000_000}\n',
   'training_receipts.py':'import hashlib,json\ndef sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()\n'
  }
  for name,text in files.items():(subnet/name).write_text(text)
  self.fx.job['source_files']={str(p.relative_to(self.source)):hashlib.sha256(p.read_bytes()).hexdigest()for p in subnet.glob('*.py')}
  m=dict(self.fx.job['manifest']['payload'],submission_transport_policy='tiny-selected-source-v2');self.fx.job['manifest']=self.fx.sign(m);self.fx.job['submissions'][0]['commitment_ref']={'size':2902}
 def call(self):return budget(self.fx.job,self.fx.policy,self.fx.authority,protocol_source=self.source)
 def test_selected_protocol_ignores_incompatible_parent_modules(self):
  with patch('subnet.distributed_roles.validate_frozen_submissions',side_effect=ValueError('historical parent refuses new transport')):
   result=self.call()
  self.assertEqual(result[3:],(2902,2902))
 def test_forged_child_refused_by_selected_protocol(self):
  self.fx.job['submissions'][0]['sha256']='c'*64
  with self.assertRaisesRegex(ValueError,'protocol refused'):self.call()
 def test_changed_runtime_and_extra_module_refused_before_child(self):
  with patch('subprocess.run',side_effect=AssertionError('must not execute changed source')):
   (self.source/'subnet/distributed_roles.py').write_text('raise RuntimeError("changed")')
   with self.assertRaisesRegex(ValueError,'source changed'):self.call()
  (self.source/'subnet/new_unpinned.py').write_text('')
  with self.assertRaisesRegex(ValueError,'inventory'):self.call()
 def test_symlink_source_and_member_refused(self):
  alias=self.source.parent/'alias';alias.symlink_to(self.source)
  with self.assertRaisesRegex(ValueError,'exact registry'):budget(self.fx.job,self.fx.policy,self.fx.authority,protocol_source=alias)
  p=self.source/'subnet/artifact_budget.py';target=self.source.parent/'outside';target.write_bytes(p.read_bytes());p.unlink();p.symlink_to(target)
  with self.assertRaisesRegex(ValueError,'source changed'):self.call()
 def test_cpu_timeout_deferred_without_model_or_cache_mutation(self):
  with patch('subprocess.run',side_effect=subprocess.TimeoutExpired('cpu',30)):
   with self.assertRaises(CapacityDeferred):self.call()
  self.assertFalse((self.fx.root/'checkpoints').exists())
 def test_wrong_manifest_signature_refused(self):
  self.fx.job['manifest']['payload']['checkpoint']['id']='f'*64
  with self.assertRaisesRegex(ValueError,'protocol refused'):self.call()
 def test_wait_uses_registry_selected_source_instead_of_parent_protocol(self):
  import threading
  from types import SimpleNamespace
  from subnet.distributed_worker import Worker
  from subnet.storage import canonical
  policy=self.fx.fx.root/'policy.json';policy.write_bytes(canonical(self.fx.policy));attempt=self.fx.fx.root/'attempt';attempt.mkdir()
  worker=SimpleNamespace(capacity_policy_path=policy,authority=self.fx.authority)
  with patch('ops.verifier_capacity_admission.admit',return_value={'status':'admitted'})as admit,patch('ops.verifier_capacity_admission.budget',return_value=({'poll_seconds':1},None,None,1,1))as selected:
   Worker.wait_for_capacity(worker,self.fx.job,{'lease_until':100,'job_sha256':'f'*64,'attempt':1},threading.Event(),self.fx.cache,None,attempt,protocol_source=self.source,clock=lambda:10)
  self.assertEqual(admit.call_args.kwargs['protocol_source'],self.source)
  self.assertEqual(selected.call_args.kwargs['protocol_source'],self.source)
 def test_historical_default_budget_remains_unchanged(self):
  self.fx.job['manifest']=self.fx.sign(dict(self.fx.job['manifest']['payload'],submission_transport_policy=None))
  self.assertEqual(budget(self.fx.job,self.fx.policy,self.fx.authority)[3:],(100_000_000,100_000_000))

if __name__=='__main__':unittest.main()
