import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path

PROBE='''import sys,os,json,hashlib,time
from pathlib import Path

def trace(frame,event,arg):
 if event=='call'and frame.f_code.co_name=='main'and frame.f_globals.get('__name__')=='__main__'and frame.f_code.co_filename.endswith('/subnet/backend_jobs.py'):
  g=frame.f_globals;sys.settrace(None)
  def probe(envelope,authority,workspace,cache):
   import importlib,requests
   from unittest.mock import patch
   b=importlib.import_module('subnet.backend_jobs')
   import subnet.backend_jobs as package_import
   assert b is sys.modules['__main__'] and package_import is b
   root=Path(workspace);out=root/'jobs/original';transfer=out/'.fp32-state-transfer-probe';transfer.mkdir(parents=True)
   raw=b'full approved parent shard';path=transfer/'state-000000.safetensors';row=dict(name=path.name,sha256=hashlib.sha256(raw).hexdigest(),size=len(raw))
   job=dict(role='train',job_id='original',training_policy=b.PERSISTENT_POLICY,expires_at=time.time()+60,persistent_training=dict(parent_read_urls={path.name:'original-parent-capability'}))
   # Same namespace assignment used by actual execute; subsequent worker-side
   # canonical import must observe it. No scientific job/model is executed.
   g['_PARENT_READ_CONTEXT']=(job,{},authority,str(root))
   class Response:
    status_code=200
    def __init__(self,bad=False):self.bad=bad
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_content(self,n):
     if self.bad:
      yield b'partial-prefix'
      raise requests.ConnectionError('stream ReadTimeoutError CPU control')
     yield raw
   with patch('subnet.persistent_training_protocol.validate_job',return_value=({},dict(shards=[row]))),patch.object(b,'r2_url',side_effect=lambda u,op:u),patch('requests.get',side_effect=[Response(True),Response()])as get:
    from subnet.backend_jobs import get_object
    assert get_object is g['get_object'];get_object('original-parent-capability',row['sha256'],path,row['size'])
    assert get.call_count==2 and path.read_bytes()==raw and not path.with_suffix('.safetensors.partial').exists()
   evidence=json.loads((out/'parent-state-read-retries'/(row['name']+'.json')).read_bytes())
   assert evidence['verified']is True and len(evidence['attempts'])==2
   Path(os.environ['ENTRYPOINT_CPU_RESULT']).write_text(json.dumps(dict(status='passed',canonical_is_main=True,context_shared=b._PARENT_READ_CONTEXT is g['_PARENT_READ_CONTEXT'],stream_attempts=2,full_SHA_verified=True,torch_imported='torch'in sys.modules,GPU=False,execute_probe_only=True)))
   return dict(job_id='CPU-entrypoint-probe',role='train',checkpoint='00'*32)
  g['execute']=probe
 return trace
sys.settrace(trace)
'''

class BackendModuleEntrypointControls(unittest.TestCase):
 def test_actual_dash_m_canonical_worker_read_observes_context_and_retries(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);(root/'sitecustomize.py').write_text(PROBE);job=root/'job.json';job.write_text('{}');result=root/'result.json';env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(Path(__file__).resolve().parents[1]),ENTRYPOINT_CPU_RESULT=str(result))
   q=subprocess.run([sys.executable,'-m','subnet.backend_jobs',str(job),'--authority','00'*32,'--workspace',str(root/'workspace')],env=env,capture_output=True,text=True,timeout=30)
   self.assertEqual(q.returncode,0,q.stderr);v=json.loads(result.read_text());self.assertTrue(v['canonical_is_main']);self.assertTrue(v['context_shared']);self.assertEqual(v['stream_attempts'],2);self.assertTrue(v['full_SHA_verified']);self.assertFalse(v['torch_imported']);self.assertTrue(v['execute_probe_only'])
 def test_preloaded_conflicting_module_refused_in_fresh_entrypoint(self):
  code="import subnet.backend_jobs,runpy;runpy.run_module('subnet.backend_jobs',run_name='__main__')"
  q=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=20)
  self.assertNotEqual(q.returncode,0);self.assertIn('fresh canonical module',q.stderr)
if __name__=='__main__':unittest.main()
