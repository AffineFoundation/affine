"""Real fresh-process envelope parser controls, before model construction."""
import base64,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from nacl.signing import SigningKey
from subnet.storage import canonical

class ProspectiveJobEnvelope(unittest.TestCase):
 def test_each_nontraining_role_reaches_original_fresh_loader_without_protocol_imports(self):
  key=SigningKey.generate();authority=key.verify_key.encode().hex()
  def sign(v):return {'payload':v,'signer':authority,'signature':base64.b64encode(key.sign(canonical(v)).signature).decode()}
  m=dict(max_batches=9,K=4,L=4,samples_per_batch=8,training_policy='bf16-cpu-fp32-master-task-normalized-persistent-v4',training_input_policy='committed-unaudited-training-v1',training_task_capacity={'version':'signed-training-task-capacity-v1','max_tasks':512})
  source=Path(os.environ.get('ENVELOPE_TEST_SOURCE',Path(__file__).resolve().parents[1]))
  code='''import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from subnet import backend_jobs as b
before={n for n in sys.modules if n.startswith('subnet.')}
v=b.load_job_envelope(sys.argv[2],sys.argv[3])
assert {n for n in sys.modules if n.startswith('subnet.')}==before
b.install_source_loader(Path(sys.argv[1]))
assert 'torch'not in sys.modules
assert not any(n in sys.modules for n in ('subnet.distributed_roles','subnet.forced_sampling','subnet.cache_lifecycle'))
print('FRESH_CPU_PASS')
'''
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'job.json'
   for role in ('mine','verify','evaluate','upload'):
    for padding in (10,4_100_000):
     p.write_bytes(canonical(sign(dict(role=role,manifest=sign(m),padding='x'*padding))))
     r=subprocess.run([sys.executable,'-I','-B','-c',code,str(source),str(p),authority],capture_output=True,text=True,timeout=30)
     with self.subTest(role=role,padding=padding):self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout,'FRESH_CPU_PASS\n')
if __name__=='__main__':unittest.main()
