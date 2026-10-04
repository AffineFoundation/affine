"""Actual prepared-runtime controls for prospective native MATH grading."""
import ast,json,os,resource,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from subnet.native_math_grader import ASSET,dependency_binding,runtime_lock,isolated_argv

PYTHON=Path('/home/const/.cache/uv/environments-v2/verifiers-cp3.12.3-242b229c745212ce/bin/python')

class NativeMathGrader(unittest.TestCase):
 def run_grader(self,*args,source=None):
  if not PYTHON.is_file():self.skipTest('approved prepared native interpreter unavailable')
  with tempfile.TemporaryDirectory(prefix='native-math-control-') as directory:
   path=Path(directory)/'verify.py';path.write_text(source or ASSET.read_text())
   def limits():
    resource.setrlimit(resource.RLIMIT_CPU,(8,9));resource.setrlimit(resource.RLIMIT_AS,(768*1024*1024,768*1024*1024));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
   return subprocess.run(isolated_argv(PYTHON,path,args),cwd=directory,env={'PATH':'/usr/bin:/bin','HOME':directory,'TMPDIR':directory},capture_output=True,text=True,timeout=12,preexec_fn=limits)
 def grade(self,reply,gold='1',source=None):return self.run_grader('--json-arguments',json.dumps([gold,reply]),source=source)
 def test_approved_runtime_readiness_and_binding(self):
  result=self.run_grader('--runtime-check');self.assertEqual(result.returncode,0,result.stderr)
  self.assertEqual(json.loads(result.stdout)['runtime'],runtime_lock());self.assertEqual(len(next(iter(dependency_binding().values()))),64)
 def test_legitimate_correct_wrong_arithmetic_and_no_box(self):
  for reply,score in ((r'\boxed{1}','1.0'),(r'\boxed{2}','0.0'),(r'\boxed{1+(2-2)}','1.0'),('no answer','0.0')):
   with self.subTest(reply=reply):
    result=self.grade(reply);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout.strip(),score)
 def test_legitimate_fraction_radical_set_and_symbolic_grading(self):
  cases=((r'\frac{1}{2}',r'\boxed{0.5}','1.0'),
         (r'\sqrt{2}',r'\boxed{\sqrt{2}}','1.0'),
         (r'x^{2}+2x+1',r'\boxed{(x+1)^{2}}','1.0'),
         (r'\{1,2\}',r'\boxed{\{2,1\}}','1.0'),
         (r'\frac{1}{2}',r'\boxed{0.6}','0.0'))
  for gold,reply,expected in cases:
   with self.subTest(gold=gold,reply=reply):
    result=self.grade(reply,gold);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout.strip(),expected)
 def test_last_boxed_rule_preserved(self):
  result=self.grade(r'\boxed{2} then \boxed{1}');self.assertEqual(result.returncode,0);self.assertEqual(result.stdout.strip(),'1.0')
 def test_actual_parse_and_comparison_timeouts_are_indeterminate(self):
  for reply in (r'\boxed{'+'{'*150+'1'+'}'*151,r'\boxed{2^{2^{1000000}}}'):
   with self.subTest(reply=reply[:30]):
    result=self.grade(reply);self.assertEqual(result.returncode,75,result.stderr);self.assertEqual(result.stdout,'');self.assertEqual(json.loads(result.stderr)['reason'],'grader_timeout')
 def test_runtime_mismatch_rejected_before_claim_and_grading(self):
  original=runtime_lock()['distributions']['sympy']['python_files_sha256']
  source=ASSET.read_text().replace(original,'0'*64)
  for args in (('--runtime-check',),('--json-arguments',json.dumps(['1',r'\boxed{1}']))):
   result=self.run_grader(*args,source=source);self.assertEqual(result.returncode,75);self.assertEqual(json.loads(result.stderr)['reason'],'grader_runtime_mismatch')
 def test_import_isolation_ignores_pythonpath_and_site_startup(self):
  if not PYTHON.is_file():self.skipTest('approved prepared native interpreter unavailable')
  with tempfile.TemporaryDirectory(prefix='native-math-import-control-') as directory:
   directory=Path(directory);script=directory/'verify.py';script.write_text(ASSET.read_text())
   sentinel=directory/'startup-executed'
   (directory/'sitecustomize.py').write_text('from pathlib import Path;Path('+repr(str(sentinel))+').write_text("bad")')
   result=subprocess.run(isolated_argv(PYTHON,script,['--runtime-check']),cwd=directory,env={'PATH':'/usr/bin:/bin','HOME':str(directory),'PYTHONPATH':str(directory)},capture_output=True,text=True,timeout=12)
   self.assertEqual(result.returncode,0,result.stderr);self.assertFalse(sentinel.exists())
   self.assertTrue(json.loads(result.stdout)['site_disabled'])
 def test_binary_and_stdlib_profiles_cannot_be_mixed(self):
  lock=runtime_lock();source=ASSET.read_text().replace(lock['profiles'][0]['stdlib']['files_sha256'],lock['profiles'][1]['stdlib']['files_sha256'])
  result=self.run_grader('--runtime-check',source=source);self.assertEqual(result.returncode,75);self.assertEqual(json.loads(result.stderr)['reason'],'grader_runtime_mismatch')
 def test_actual_native_task_boundary_does_not_record_timeout_as_reward(self):
  import asyncio,importlib.util
  from types import SimpleNamespace
  import verifiers.v1 as vf
  from verifiers.v1.errors import TaskError
  module_spec=importlib.util.spec_from_file_location('native_math_task_security_control',ASSET.with_name('taskset.py'))
  module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
  task=module.MathTask(module.MathData(name='owned-control',prompt='Compute 1',problem='Compute 1',answer='1',subject='test',level='test'),vf.TaskConfig())
  outer=self
  class Runtime:
   async def prepare_uv_script(self,script,env=None):
    import hashlib
    return [str(PYTHON),'/tmp/vf-scripts/'+hashlib.sha256(script).hexdigest()+'.py']
   async def run(self,argv,env):
    outer.assertIn('-I',argv)
    outer.assertIn('-S',argv)
    result=outer.run_grader(*argv[7:])
    return SimpleNamespace(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
  async def check():
   for reply,expected in ((r'\boxed{1}',1.0),(r'\boxed{2}',0.0),(r'\boxed{2^{2^{1000000}}}',None)):
    rewards=[]
    trace=SimpleNamespace(last_reply=reply,metrics={},rewards={},record_metric=lambda *args:None,record_reward=lambda *args:rewards.append(args))
    if expected is None:
     with self.assertRaises(TaskError):await task.score(trace,Runtime())
     self.assertEqual(rewards,[])
    else:
     await task.score(trace,Runtime());self.assertEqual(rewards[0][1],expected)
  asyncio.run(check())
 def test_actual_native_task_rejects_malformed_grader_outputs(self):
  import asyncio,hashlib,importlib.util
  from types import SimpleNamespace
  import verifiers.v1 as vf
  from verifiers.v1.errors import TaskError
  spec=importlib.util.spec_from_file_location('native_math_output_control',ASSET.with_name('taskset.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  task=module.MathTask(module.MathData(name='owned-output-control',prompt='Compute 1',problem='Compute 1',answer='1',subject='test',level='test'),vf.TaskConfig())
  class Runtime:
   async def prepare_uv_script(self,script,env=None):return [str(PYTHON),'/tmp/vf-scripts/'+hashlib.sha256(script).hexdigest()+'.py']
   async def run(self,argv,env):return SimpleNamespace(exit_code=0,stdout=self.output,stderr='')
  async def check():
   for output in ('','not-a-score','NaN','inf','-1.0','0.5','1.0\n0.0','0.0 extra'):
    rewards=[];runtime=Runtime();runtime.output=output
    trace=SimpleNamespace(last_reply=r'\boxed{1}',metrics={},rewards={},record_metric=lambda *a:None,record_reward=lambda *a:rewards.append(a))
    with self.assertRaises(TaskError):await task.score(trace,runtime)
    self.assertEqual(rewards,[])
    with self.assertRaises(RuntimeError):await task.validate(runtime)
  asyncio.run(check())
 def test_python_strings_are_not_executed(self):
  result=self.grade(r"\boxed{__import__('builtins').print('SAFE_SENTINEL')}");self.assertNotIn('SAFE_SENTINEL\n',result.stdout);self.assertNotEqual(result.stdout.strip(),'1.0')

if __name__=='__main__':unittest.main()
