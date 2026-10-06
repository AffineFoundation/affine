import json,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from test_verifier_capacity_admission import VerifierCapacityAdmission
from ops.capacity_bounded_verifier_backend import isolated_transport_admission,bind_transport

class BootstrapIsolation(unittest.TestCase):
 def setUp(self):
  self.f=VerifierCapacityAdmission();self.f.setUp();self.addCleanup(self.f.doCleanups)
 def test_real_subprocess_validates_authentic_budget_without_parent_module_imports(self):
  source=Path(__file__).resolve().parents[1];helper=source/'ops/verifier_capacity_admission.py';before=set(sys.modules)
  admission=isolated_transport_admission(self.f.job,self.f.policy,self.f.authority,source,helper)
  self.assertEqual(admission.budget()[2],{'model.safetensors':len(self.f.bytes)})
  self.assertEqual(admission.input_limits(),[100000000]);self.assertEqual(set(sys.modules),before)
 def test_signature_tamper_refused_before_original_download(self):
  policy=dict(self.f.policy,signature='A'*88)
  with self.assertRaisesRegex(ValueError,'isolated transport capacity validation refused'):isolated_transport_admission(self.f.job,policy,self.f.authority,Path(__file__).resolve().parents[1],Path(__file__).resolve().parents[1]/'ops/verifier_capacity_admission.py')
 def test_malformed_results_no_fallback_or_download(self):
  source=Path(__file__).resolve().parents[1]
  for value in({'sizes':{'foreign':5},'limits':[5]},{'sizes':{'model.safetensors':True},'limits':[5]},{'sizes':{'model.safetensors':5},'limits':[]},{'sizes':{'model.safetensors':5},'limits':[0]}):
   with patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(value).encode())):
    with self.assertRaisesRegex(ValueError,'exact isolated capacity result'):isolated_transport_admission(self.f.job,self.f.policy,self.f.authority,source,source/'ops/verifier_capacity_admission.py')
 def test_actual_fresh_loader_accepts_isolated_admission_not_inprocess_helper(self):
  source=Path(__file__).resolve().parents[1];helper=source/'ops/verifier_capacity_admission.py';raw=json.dumps(dict(job=self.f.job,policy=self.f.policy,authority=self.f.authority))
  code='''import sys,json
from pathlib import Path
source=Path(sys.argv[1]);sys.path.insert(0,str(source))
from ops.capacity_bounded_verifier_backend import isolated_transport_admission
v=json.load(sys.stdin);isolated_transport_admission(v['job'],v['policy'],v['authority'],source,source/'ops/verifier_capacity_admission.py')
assert not any(n.startswith('subnet.')for n in sys.modules)
from subnet import backend_jobs
backend_jobs.install_source_loader(source)
assert 'torch'not in sys.modules
print('ACTUAL_FRESH_LOADER_PASS')
'''
  r=subprocess.run([sys.executable,'-I','-B','-c',code,str(source)],input=raw,text=True,capture_output=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr);self.assertIn('ACTUAL_FRESH_LOADER_PASS',r.stdout)
 def test_actual_bootstrap_main_installs_fresh_backend_loader_before_any_model(self):
  import hashlib,tempfile
  source=Path(__file__).resolve().parents[1];job=json.loads(json.dumps(self.f.job));job['source_files']={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(source/'subnet').glob('*.py')};job['manifest']['payload']['checkpoint']['read_urls']={'model.safetensors':'model-url'};job['manifest']=self.f.sign(job['manifest']['payload']);job['submissions'][0]['url']='input-url'
  with tempfile.TemporaryDirectory()as d:
   d=Path(d);(d/'job.json').write_text(json.dumps(self.f.sign(job)));(d/'policy.json').write_text(json.dumps(self.f.policy))
   code="""import sys,importlib.util
from pathlib import Path
source=Path(sys.argv[1]);sys.path.insert(0,str(source))
from subnet import backend_jobs
from ops import capacity_bounded_verifier_backend as bootstrap
# Stop at the real unchanged fresh-source loader, before model/import/download.
def cpu_execute(*args):
 backend_jobs.install_source_loader(source)
 assert 'torch'not in sys.modules
 assert not any(n in sys.modules for n in ('subnet.cache_lifecycle','subnet.distributed_roles','subnet.forced_sampling'))
 print('BOOTSTRAP_MAIN_FRESH_PREFLIGHT_PASS')
 raise SystemExit(0)
backend_jobs.execute=cpu_execute
sys.argv=[bootstrap.__file__,sys.argv[2],'--authority',sys.argv[3],'--workspace',sys.argv[4],'--capacity-policy',sys.argv[5]]
bootstrap.main()
"""
   r=subprocess.run([sys.executable,'-I','-B','-c',code,str(source),str(d/'job.json'),self.f.authority,str(d/'workspace'),str(d/'policy.json')],text=True,capture_output=True,timeout=30,cwd=source);self.assertEqual(r.returncode,0,r.stderr);self.assertIn('BOOTSTRAP_MAIN_FRESH_PREFLIGHT_PASS',r.stdout);self.assertFalse((d/'workspace').exists())
 def test_foreign_transport_mapping_still_refused(self):
  source=Path(__file__).resolve().parents[1];job=json.loads(json.dumps(self.f.job));cp=job['manifest']['payload']['checkpoint'];cp['read_urls']={'model.safetensors':'model-url'};job['manifest']=self.f.sign(job['manifest']['payload']);job['submissions'][0]['url']='input-url';a=isolated_transport_admission(job,self.f.policy,self.f.authority,source,source/'ops/verifier_capacity_admission.py');calls=[];backend=SimpleNamespace(get_object=lambda *args,**kwargs:calls.append(args));bind_transport(backend,job,self.f.authority,self.f.root,self.f.policy,a)
  with self.assertRaises(ValueError):backend.get_object('model-url',self.f.sha,self.f.root/'foreign',10**10)
  self.assertFalse(calls)
if __name__=='__main__':unittest.main()
