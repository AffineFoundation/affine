"""Fresh process old-v1 inventory, no compact file/import access.

The local qualification control starts from the actual immutable sealed v1
archive. It overlays only prospective caller modules to test their compatibility
with its 141-module inventory. Synthetic signatures authorize this test package,
not the original sealed source or any live job. No archive is changed in place.
"""
import hashlib
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.backend_jobs import SOURCE_FILES
from training_receipt_fixtures import transport_fixture,sign

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'state/live-math-launch-preparation-v1/distributed-preparation/future-source-receipt-939af544-v1/source-candidate.tar.gz'
CALLERS=('backend_jobs','remote_backend','role_router','gpu_service','persistent_training_controller',
         'persistent_training_protocol','persistent_training_evidence','persistent_training_worker')


class OldV1SourceCompatibilityTests(unittest.TestCase):
    def test_genuine_sealed_v1_inventory_with_prospective_callers_has_no_compact_dependency(self):
        if not ARCHIVE.is_file():self.skipTest('local sealed v1 qualification archive not present')
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
            '94ff74eb335e24d4702da2ec10cc0aee068076b003e6c0b13b0e81f5090bc79c')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package=root/'source';(package/'subnet').mkdir(parents=True)
            originals={}
            with tarfile.open(ARCHIVE,'r:gz') as archive:
                for member in archive.getmembers():
                    if member.isfile() and member.name.startswith('subnet/') and member.name.endswith('.py') and len(Path(member.name).parts)==2:
                        originals[member.name]=archive.extractfile(member).read()
            self.assertEqual(len(originals),141)
            self.assertNotIn('subnet/compact_training_inputs.py',originals)
            for name,data in originals.items():(package/name).write_bytes(data)
            for name in CALLERS:(package/'subnet'/(name+'.py')).write_bytes((ROOT/'subnet'/(name+'.py')).read_bytes())
            # Inventory derives from the genuine original archive, not a current
            # worktree glob. Updated callers keep its exact module membership.
            files={name:hashlib.sha256((package/name).read_bytes()).hexdigest()for name in originals}
            key=SigningKey.generate();fixture=transport_fixture(key)
            job=dict(schema=1,job_id='old-source-v1',role='train',created_at=22,expires_at=100,
                manifest=sign(key,fixture['manifest']),source_files=files,
                runtime_versions=dict(torch='approved',transformers='approved',toploc='approved'),
                submissions=[fixture['submission']],steps=3,training_policy=fixture['manifest']['training_policy'],
                training_input_policy='authenticated-verifier-receipts-v1')
            (root/'job.json').write_bytes(canonical(sign(key,job)))
            (root/'original.zip').write_bytes(fixture['data'])
            script=r'''
import sys,json,importlib.abc,os
from pathlib import Path
from unittest.mock import patch
source,jobpath,zippath,authority,mode=sys.argv[1:]
sys.path.insert(0,source)
class DenyCompact(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='subnet.compact_training_inputs':raise AssertionError('unadmitted compact import')
sys.meta_path.insert(0,DenyCompact())
from subnet import backend_jobs as backend
job,manifest=backend.validate(json.loads(Path(jobpath).read_bytes()),authority,now=50)
assert len(job['source_files'])==141
assert not Path(source,'subnet/compact_training_inputs.py').exists()

if mode=='execute':
    import torch
    from types import SimpleNamespace
    def download(url,digest,path,limit):path.write_bytes(Path(zippath).read_bytes())
    def factory(checkpoint,files,spec,harness):
        runtime=SimpleNamespace(model=torch.nn.Linear(1,1),harness=harness)
        from subnet import covered_epoch_optimizer,model
        def optimize(runtime,pairs,out,**kwargs):
            assert len(pairs)==1
            with torch.no_grad():runtime.model.weight.add_(1)
            return out/'final',[{'synthetic_update':True}]*3
        covered_epoch_optimizer.train_epoch=optimize
        model.model_files=lambda path:dict(files,**{'model.safetensors':'3'*64})
        return runtime
    with patch('subnet.backend_jobs.version',return_value='approved'), \
         patch('subnet.backend_jobs.time.time',return_value=50), \
         patch('subnet.backend_jobs.checkpoint',return_value=Path(source)), \
         patch('subnet.backend_jobs.get_object',side_effect=download), \
         patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
        report=backend.execute(json.loads(Path(jobpath).read_bytes()),authority,Path(source)/'workspace',runtime_factory=factory)
    assert report['training']['training_input_policy']=='authenticated-verifier-receipts-v1'
    assert report['audits']==[] and len(report['training_admissions'])==1
    assert 'subnet.compact_training_inputs' not in sys.modules
    print('old-source-v1-real-loader-pass')
    raise SystemExit(0)
# The real source loader executes approved v1 receipt/protocol helpers, rather
# than patching the loader or authorizing whatever a worktree glob contains.
backend.install_source_loader(Path(source),('subnet/training_receipts.py',))
from subnet.training_receipts import admitted_submission
summary,pairs=admitted_submission(Path(zippath),job['submissions'][0],manifest,authority)
assert summary['version']=='authenticated-verifier-receipts-v1' and len(pairs)==1
assert 'subnet.compact_training_inputs' not in sys.modules
# Every-role v2 pin refusal happens without even trying the unadmitted import.
bad=dict(manifest,training_input_policy='authenticated-verifier-compact-inputs-v2')
from nacl.signing import SigningKey
# Mock only already-authenticated envelope unwrapping; source/policy admission
# remains real. No forged signature claims are made by this missing-pin control.
with patch('subnet.backend_jobs.signed',side_effect=[dict(job,manifest={}),bad]):
    try:backend._validate({},authority,50,resolve_source=False)
    except ValueError as error:assert 'source pins' in str(error)
    else:raise AssertionError('v2 missing source pin admitted')
print('old-source-v1-real-loader-pass')
'''
            for mode in ('admission','execute'):
                result=subprocess.run([str(ROOT/'.venv/bin/python'),'-I','-B','-c',script,str(package),str(root/'job.json'),
                    str(root/'original.zip'),key.verify_key.encode().hex(),mode],cwd=root,text=True,capture_output=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                self.assertIn('old-source-v1-real-loader-pass',result.stdout)
