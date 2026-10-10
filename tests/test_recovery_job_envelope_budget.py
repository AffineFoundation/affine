import base64,copy,hashlib,json,tempfile,unittest,sys,os,subprocess
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from nacl.exceptions import BadSignatureError
from subnet import backend_jobs as backend

class RecoveryEnvelopeBudgetControls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'job.json';self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
  old={'epoch':'original','source_bundle':{'sha256':'11'*32}};self.original={'role':'train','training_policy':backend.PERSISTENT_POLICY,'training_input_policy':'committed-unaudited-training-v1','manifest':self.sign(old)}
  self.declaration={'version':'terminal-parent-restore-pre-update-recovery-v2','epoch':'original','original_signed_job':self.sign(self.original),'original_job_sha256':hashlib.sha256(backend.canonical(self.original)).hexdigest(),'original_input_source_sha256':'11'*32,'replacement_execution_source_sha256':'22'*32}
  self.manifest={'epoch':'original','source_bundle':{'sha256':'22'*32},'training_startup_recovery':self.sign(self.declaration)};self.job={'role':'train','training_policy':backend.PERSISTENT_POLICY,'training_input_policy':'committed-unaudited-training-v1','manifest':self.sign(self.manifest),'padding':'x'*4_100_000}
 def sign(self,v):return {'payload':copy.deepcopy(v),'signer':self.authority,'signature':base64.b64encode(self.key.sign(backend.canonical(v)).signature).decode()}
 def write(self,v=None,size=None):
  data=backend.canonical(self.sign(self.job)if v is None else v)
  if size is not None:data+=b' '*(size-len(data))
  self.path.write_bytes(data);return data
 def test_exact_supported_root_authenticated_large_recoveries(self):
  for version in backend.LARGE_RECOVERY_VERSIONS:
   self.declaration['version']=version;self.manifest['training_startup_recovery']=self.sign(self.declaration);self.job['manifest']=self.sign(self.manifest)
   self.write();self.assertEqual(backend.load_job_envelope(self.path,self.authority)['payload'],self.job)
 def test_postupdate_large_entrypoint_and_upload_rejection(self):
  self.assertIn('terminal-post-update-uncommitted-recovery-v1',backend.LARGE_RECOVERY_VERSIONS)
  self.declaration['version']='terminal-post-update-uncommitted-recovery-v1'
  self.manifest['training_startup_recovery']=self.sign(self.declaration);self.job['manifest']=self.sign(self.manifest)
  self.write(size=5_100_000)
  args=['backend',str(self.path),'--authority',self.authority,'--workspace',str(self.tmp.name)]
  with patch.object(sys,'argv',args),patch.object(backend,'execute',return_value={'job_id':'fresh','role':'train','checkpoint':'22'*32})as execute,patch('builtins.print'):
   backend.main()
  self.assertEqual(execute.call_args.args[0]['payload'],self.job)
  self.job['role']='upload';self.write()
  with patch.object(sys,'argv',args),patch.object(backend,'execute')as execute,self.assertRaises(ValueError):backend.main()
  execute.assert_not_called()
 def test_actual_dash_m_postupdate_51MB_loader_before_cpu_dispatch(self):
  self.declaration['version']='terminal-post-update-uncommitted-recovery-v1'
  self.manifest['training_startup_recovery']=self.sign(self.declaration);self.job['manifest']=self.sign(self.manifest)
  self.write(size=5_100_000)
  root=Path(self.tmp.name);result=root/'CPU-result.json'
  probe="""import sys,os,json
from pathlib import Path
def trace(frame,event,arg):
 if event=='call' and frame.f_code.co_name=='main' and frame.f_globals.get('__name__')=='__main__' and frame.f_code.co_filename.endswith('/subnet/backend_jobs.py'):
  g=frame.f_globals;sys.settrace(None)
  def execute(envelope,*args):
   assert envelope['payload']['manifest']['payload']['training_startup_recovery']['payload']['version']=='terminal-post-update-uncommitted-recovery-v1'
   Path(os.environ['CPU_RESULT']).write_text(json.dumps({'CPU_stub':True,'torch_imported':'torch' in sys.modules,'canonical_is_main':sys.modules['subnet.backend_jobs'] is sys.modules['__main__']}))
   return {'job_id':'fresh','role':'train','checkpoint':'22'*32}
  g['execute']=execute
 return trace
sys.settrace(trace)
"""
  (root/'sitecustomize.py').write_text(probe)
  env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(Path(__file__).resolve().parents[1]),CPU_RESULT=str(result))
  args=[sys.executable,'-m','subnet.backend_jobs',str(self.path),'--authority',self.authority,'--workspace',str(root/'workspace')]
  q=subprocess.run(args,env=env,capture_output=True,text=True,timeout=30)
  self.assertEqual(q.returncode,0,q.stderr);v=json.loads(result.read_text());self.assertTrue(v['CPU_stub']);self.assertTrue(v['canonical_is_main']);self.assertFalse(v['torch_imported'])
  result.unlink();self.job['role']='upload';self.write();q=subprocess.run(args,env=env,capture_output=True,text=True,timeout=30)
  self.assertNotEqual(q.returncode,0);self.assertFalse(result.exists());self.assertIn('job envelope size budget',q.stderr)
 def test_absolute32MB_cap_read_before_parse_and_historical_limits_unchanged(self):
  self.write(size=8_000_000);self.assertEqual(backend.load_job_envelope(self.path,self.authority)['payload'],self.job)
  self.write(size=8_000_001)
  with self.assertRaises(ValueError):backend.load_job_envelope(self.path,self.authority)
  self.path.write_bytes(b'x'*32_000_001)
  with patch.object(backend.json,'loads')as parse,self.assertRaisesRegex(ValueError,'absolute size'):backend.load_job_envelope(self.path,self.authority)
  parse.assert_not_called()
  small={'historical':'unchanged'};self.write(small,size=4_000_000);self.assertEqual(backend.load_job_envelope(self.path,self.authority),small)
  self.write(small,size=4_000_001)
  with self.assertRaises(ValueError):backend.load_job_envelope(self.path,self.authority)
 def test_normal_role_unknown_null_and_malformed_scope_never_get_exception(self):
  for change in ['role','unknown','null','no_scope','epoch','source','original_digest']:
   job=copy.deepcopy(self.job);manifest=copy.deepcopy(self.manifest);declaration=copy.deepcopy(self.declaration)
   if change=='role':job['role']='mine'
   if change=='unknown':declaration['version']='terminal-training-startup-recovery-v1'
   if change=='null':manifest['training_startup_recovery']=None
   elif change=='no_scope':manifest.pop('training_startup_recovery')
   else:
    if change=='epoch':declaration['epoch']='foreign'
    if change=='source':declaration['replacement_execution_source_sha256']='33'*32
    if change=='original_digest':declaration['original_job_sha256']='00'*32
    manifest['training_startup_recovery']=self.sign(declaration)
   job['manifest']=self.sign(manifest);self.write(self.sign(job))
   with self.subTest(change=change),self.assertRaises(ValueError):backend.load_job_envelope(self.path,self.authority)
  self.path.write_bytes(backend.canonical({'payload':None}))
  with self.assertRaisesRegex(ValueError,'object'):backend.load_job_envelope(self.path,self.authority)
  self.path.write_bytes(b'null')
  with self.assertRaisesRegex(ValueError,'object'):backend.load_job_envelope(self.path,self.authority)
 def test_foreign_and_mutated_signatures_rejected_before_dispatch(self):
  value=self.sign(self.job);value['payload']['padding']='y'+value['payload']['padding'][1:];self.write(value)
  with self.assertRaises(BadSignatureError):backend.load_job_envelope(self.path,self.authority)
  self.write()
  with self.assertRaisesRegex(ValueError,'authority'):backend.load_job_envelope(self.path,SigningKey.generate().verify_key.encode().hex())
  manifest=copy.deepcopy(self.manifest);manifest['training_startup_recovery']['payload']['version']='terminal-parent-restore-pre-update-bootstrap-recovery-v3';job=copy.deepcopy(self.job);job['manifest']=self.sign(manifest);self.write(self.sign(job))
  with self.assertRaises(BadSignatureError):backend.load_job_envelope(self.path,self.authority)
 def test_main_dispatch_entry_uses_guard_before_execute_model_or_network(self):
  self.write();report={'job_id':'replacement','role':'train','checkpoint':'22'*32}
  args=['backend',str(self.path),'--authority',self.authority,'--workspace',str(self.tmp.name)]
  with patch.object(sys,'argv',args),patch.object(backend,'execute',return_value=report)as execute,patch('builtins.print'):
   backend.main()
  self.assertEqual(execute.call_args.args[0]['payload'],self.job)
  self.path.write_bytes(b'null')
  with patch.object(sys,'argv',args),patch.object(backend,'execute')as execute,self.assertRaises(ValueError):backend.main()
  execute.assert_not_called()
if __name__=='__main__':unittest.main()
