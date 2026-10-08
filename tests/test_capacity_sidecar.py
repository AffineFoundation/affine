import importlib.util,sys,tempfile,hashlib,json,unittest
from pathlib import Path
from nacl.signing import SigningKey
from unittest.mock import patch
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ops.verifier_capacity_admission',P/'ops/verifier_capacity_admission.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class SidecarGrant(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);(self.root/'subnet').mkdir();(self.root/'subnet/a.py').write_text('x=1\n');self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();self.bundle='a'*64;self.files={'subnet/a.py':hashlib.sha256((self.root/'subnet/a.py').read_bytes()).hexdigest()};self.job={'source_files':self.files,'manifest':self.sign({'source_bundle':{'sha256':self.bundle}})};self.policy={}
 def sign(self,v):
  import base64
  raw=json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode();return {'payload':v,'signer':self.authority,'signature':base64.b64encode(self.key.sign(raw).signature).decode()}
 def sidecar(self):
  p=self.root/'subnet/source_sampling_admission.py';p.write_text('CPU_ONLY=True\n');self.policy={'runtime_sidecars':{self.bundle:{'subnet/source_sampling_admission.py':hashlib.sha256(p.read_bytes()).hexdigest()}}};return p
 def call(self,policy=None):return m.validate_runtime_inventory(self.job,self.sign(self.policy if policy is None else policy),self.authority,self.root)
 def test_old_exact_runtime_unchanged(self):self.call()
 def test_ungranted_sidecar_rejected(self):self.sidecar();self.assertRaises(ValueError,self.call,{})
 def test_explicit_exact_source_sidecar_accepted(self):self.sidecar();self.call()
 def test_wrong_bundle_rejected(self):self.sidecar();self.job['manifest']=self.sign({'source_bundle':{'sha256':'b'*64}});self.assertRaises(ValueError,self.call)
 def test_changed_sidecar_rejected(self):self.sidecar().write_text('changed');self.assertRaises(ValueError,self.call)
 def test_extra_module_rejected(self):self.sidecar();(self.root/'subnet/evil.py').write_text('x=2');self.assertRaises(ValueError,self.call)
 def test_missing_runtime_rejected(self):self.sidecar();(self.root/'subnet/a.py').unlink();self.assertRaises(ValueError,self.call)
 def test_sidecar_symlink_rejected(self):p=self.sidecar();p.unlink();q=self.root/'outside';q.write_text('CPU_ONLY=True\n');p.symlink_to(q);self.assertRaises(ValueError,self.call)
 def test_unsigned_policy_rejected(self):self.sidecar();e=self.sign(self.policy);e['signature']='AAAA';self.assertRaises(Exception,m.validate_runtime_inventory,self.job,e,self.authority,self.root)
 def test_unknown_sidecar_name_rejected(self):self.policy={'runtime_sidecars':{self.bundle:{'subnet/evil.py':'b'*64}}};self.assertRaises(ValueError,self.call)
 def test_runtime_cannot_be_disguised_as_sidecar(self):self.sidecar();self.job['source_files']['subnet/source_sampling_admission.py']='c'*64;self.assertRaises(ValueError,self.call)
 def backend(self):
  # The real isolated CPU guard imports its pinned protocol dependencies.
  # Supply those files in this fixture rather than relying on parent imports.
  for name in ('distributed_roles.py','cache_lifecycle.py','artifact_budget.py','storage.py'):
   data=(P/'subnet'/name).read_bytes();(self.root/'subnet'/name).write_bytes(data)
   self.files['subnet/'+name]=hashlib.sha256(data).hexdigest()
  spec=importlib.util.spec_from_file_location('candidate_backend',P/'ops/capacity_bounded_verifier_backend.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
  from types import SimpleNamespace
  envelope=self.sign(dict(self.job,role='verify'))
  jobpath=self.root/'job.json';jobpath.write_text(json.dumps(envelope));policy=self.root/'policy.json';policy.write_text(json.dumps(self.sign(self.policy)));called=[]
  fake=SimpleNamespace(main=lambda:called.append('backend'))
  import subnet
  with patch.object(sys,'argv',['backend',str(jobpath),'--capacity-policy',str(policy),'--authority',self.authority,'--workspace',str(self.root/'workspace')]),patch.object(Path,'cwd',return_value=self.root),patch.object(b,'isolated_transport_admission',return_value=object()),patch.object(b,'bind_transport'),patch.object(subnet,'backend_jobs',fake,create=True),patch.dict(sys.modules,{'subnet.backend_jobs':fake}):
   b.main()
  return called
 def test_capacity_backend_honors_signed_sidecar_without_model(self):self.sidecar();self.assertEqual(self.backend(),['backend'])
 def test_capacity_backend_rejects_changed_sidecar_before_backend(self):self.sidecar().write_text('evil');self.assertRaises(ValueError,self.backend)
 def test_capacity_backend_uses_same_guard(self):
  self.sidecar();(self.root/'subnet/unapproved.py').write_text('x=2')
  self.assertRaises(ValueError,self.backend)
if __name__=='__main__':unittest.main()
