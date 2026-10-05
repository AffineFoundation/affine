"""Exercise worker admission in genuinely fresh processes, without GPU/network."""
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT = r'''
import base64, os, sys, tempfile, time
from pathlib import Path
from nacl.signing import SigningKey
from unittest.mock import patch
from subnet import backend_jobs as b

mode=sys.argv[1]
key=SigningKey.generate(); authority=key.verify_key.encode().hex()
def sign(payload):
    return dict(payload=payload,signer=authority,signature=base64.b64encode(key.sign(b.canonical(payload)).signature).decode())
files={'config.json':'1'*64,'model.safetensors':'2'*64}
now=time.time()
manifest=dict(epoch='fresh-worker-control',start=now-1,deadline=now+60,
    checkpoint=dict(id=b.file_map(files),files=files),model_runtime_revision=b.REVISION,
    numerical_policy=b.NUMERICAL_POLICY,backend_profile=b.BACKEND_PROFILE,
    K=1,L=1,audit_policy={'mode':'full'},environment={'id':'control'},harness={},indices=[0,1])
root=Path(b.__file__).resolve().parent.parent
job=dict(schema=1,job_id='fresh-owned',role='mine',created_at=now-1,expires_at=now+60,
    manifest=sign(manifest),source_files={n:b.digest(root/n) for n in b.SOURCE_FILES},
    runtime_versions=dict(torch='approved',transformers='approved',toploc='approved'),
    miner_id='b'*64,search_budget=1,seed_start=1,mining_subset={'control':[1]},
    capability=dict(put_url='https://account.r2.cloudflarestorage.com/bucket/artifact?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test',headers={'Content-Type':'application/octet-stream'}))
if mode=='invalid':job['mining_subset']={'control':[99]}
if mode=='tampered':job['source_files']['subnet/protocol.py']='0'*64
if mode=='preimport':import subnet.protocol
if mode in ('publication','publication-preimport'):
    manifest['persistent_publication_policy']=dict(version='parallel-persistent-publication-v1',state_readback='local-full',checkpoint_readback_workers=4)
    job['manifest']=sign(manifest)
    job['source_files']['subnet/persistent_publication.py']=b.digest(root/'subnet/persistent_publication.py')
    if mode=='publication-preimport':import subnet.persistent_publication
if mode=='commitment':
    manifest.update(submission_transport_policy='small-commitment-pairs-v1',max_batches=1)
    job['manifest']=sign(manifest)
    job['miner_identity_file']='/root/scoped-miner.seed'
    job['capability']['batch_put_urls']=[job['capability']['put_url']]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
with tempfile.TemporaryDirectory() as directory:
    def checkpoint(*args):
        assert 'subnet.protocol' in sys.modules
        assert isinstance(sys.modules['subnet.protocol'].__loader__,b.importlib.abc.Loader)
        assert any(isinstance(f,b.FreshSourceFinder) for f in sys.meta_path)
        assert b.mining_definitions(manifest,job)[0]['indices']==[1]
        raise RuntimeError('qualified checkpoint boundary')
    with patch.object(b,'version',return_value='approved'),patch.object(b,'checkpoint',side_effect=checkpoint) as cp:
        try:b.execute(sign(job),authority,directory)
        except (ValueError,RuntimeError) as error:
            expected={'valid':'qualified checkpoint boundary','invalid':'outside authorized training indices',
                      'tampered':'worker source mismatch','preimport':'requires fresh process',
                      'commitment':'qualified checkpoint boundary','publication':'qualified checkpoint boundary',
                      'publication-preimport':'qualified checkpoint boundary'}[mode]
            assert expected in str(error),repr(error)
        else:raise AssertionError('worker must stop at test boundary')
        assert cp.call_count==(1 if mode in ('valid','commitment','publication','publication-preimport') else 0)
        if mode not in ('valid','commitment','publication','publication-preimport'):assert not list(Path(directory).iterdir())
        if mode=='tampered':assert 'subnet.protocol' not in sys.modules
print('fresh admission '+mode+' passed')
'''


class FreshOwnedWorkerAdmission(unittest.TestCase):
    def test_actual_fresh_loader_precedes_subset_semantics_and_artifacts(self):
        root=Path(__file__).resolve().parent.parent
        for mode in ('valid','invalid','tampered','preimport','commitment','publication','publication-preimport'):
            with self.subTest(mode=mode):
                result=subprocess.run([sys.executable,'-B','-c',SCRIPT,mode],cwd=root,
                                      capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('passed',result.stdout)
