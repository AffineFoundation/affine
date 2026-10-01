import base64,hashlib,importlib.util,json,tempfile,unittest,subprocess,sys
from pathlib import Path
from nacl.signing import SigningKey
spec=importlib.util.spec_from_file_location('rcore_resource_guard',Path(__file__).resolve().parents[1]/'ops/run_rcore_model_resources.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
class ResourceAdmission(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.key=SigningKey.generate()
  for name in ['source','dependencies','resources','operator']:(self.root/name).mkdir()
  (self.root/'worker.py').write_text('pass\n');(self.root/'operator/model-plan.json').write_text('{}')
  self.payload=dict(revision='original-rcore-public-arithmetic-model-resources-v2',remote=str(self.root),model_execution=True,optimizer_ran=False,chain_transactions=False,provider_namespace_controlled=True,full_transitive_closure_claimed=False,files={str(p.relative_to(self.root)):dict(size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in self.root.rglob('*') if p.is_file()},resource_environment=dict(PATH_prefix=str(self.root/'resources/bin'),NLTK_DATA=str(self.root/'resources/nltk-data'),XDG_CACHE_HOME=str(self.root/'private-cache'),XDG_DATA_HOME=str(self.root/'private-data'),MPLCONFIGDIR=str(self.root/'private-cache/matplotlib'),PYTHONPATH=[str(self.root/'dependencies'),str(self.root/'source')]))
 def admit(self):
  document=dict(payload=self.payload,signer=self.key.verify_key.encode().hex(),signature=base64.b64encode(self.key.sign(guard.canonical(self.payload)).signature).decode());
  input_path=self.root/'approval.json';input_path.write_text(json.dumps(document));code="import importlib.util,json,sys;from pathlib import Path;s=importlib.util.spec_from_file_location('guard',sys.argv[1]);g=importlib.util.module_from_spec(s);s.loader.exec_module(g);d=json.loads(Path(sys.argv[2]).read_text());g.admit(d,d['signer'],sys.argv[3])";result=subprocess.run([sys.executable,'-I','-B','-c',code,str(Path(guard.__file__).resolve()),str(input_path),str(self.root)],capture_output=True,text=True)
  if result.returncode:raise ValueError(result.stderr)
  return self.payload
 def test_private_cache_can_change_outside_exact_inventory(self):
  (self.root/'private-cache').mkdir();(self.root/'private-cache/font.json').write_text('mutable');self.assertEqual(self.admit(),self.payload)
 def test_extra_dependency_refused(self):
  (self.root/'dependencies/shadow.py').write_text('pass');self.assertRaisesRegex(ValueError,'membership',self.admit)
 def test_changed_plan_refused(self):
  (self.root/'operator/model-plan.json').write_text('{"changed":true}');self.assertRaisesRegex(ValueError,'bytes',self.admit)
 def test_mutable_cache_inside_inventory_refused(self):
  self.payload['resource_environment']['MPLCONFIGDIR']=str(self.root/'resources/cache');self.assertRaisesRegex(ValueError,'environment',self.admit)
 def test_symlink_dependency_refused(self):
  (self.root/'dependencies/alias.py').symlink_to(self.root/'worker.py');self.assertRaisesRegex(ValueError,'symlink',self.admit)
 def test_nonisolated_or_bytecode_enabled_entrypoint_refused_before_profile(self):
  for flags in [[],['-I'],['-B']]:
   result=subprocess.run([sys.executable,*flags,str(Path(guard.__file__).resolve()),'--profile','/nonexistent','--authority','00','--root','/nonexistent'],capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertIn('fresh isolated no-bytecode worker',result.stderr);self.assertNotIn('FileNotFoundError',result.stderr)
if __name__=='__main__':unittest.main()
