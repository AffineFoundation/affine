import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from subnet.backend_jobs import FIXED_POLICY,REVISION,signed
from subnet.controller import Controller,ENV
from subnet.environments import legacy_spec,legacy_harness
from subnet.gpu_service import contract
from subnet.remote_backend import RemoteController
from subnet.storage import Gateway,Identity,canonical

class MemoryBucket:
    def __init__(self):self.objects={}
    def json(self,key,value):self.objects[key]=canonical(value)
    def presign(self,key,operation='get_object',expires=3600):
        return 'https://fixture.r2.cloudflarestorage.com/bucket/'+key+'?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=fixture'

class RealGPUEpochOpening(unittest.TestCase):
    def test_actual_gpu_contract_opens_and_first_signature_binds_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            spec=legacy_spec(ENV);harness=legacy_harness(spec.config)
            config=dict(environments=[dict(spec=spec.to_dict(),indices=[0,1],harness=harness)],
                heldout=[dict(env_id=spec.id,indices=[2,3])],source_bundle={'sha256':'a'*64,'size':1,'key':'public/source.tar.gz'},duration=60)
            bucket=MemoryBucket();gateway=Gateway(bucket,state_path=Path(folder)/'gateway.json',direct_r2=True)
            try:
                with patch('subnet.remote_backend.RemoteJobs'):
                    controller=RemoteController(bucket,gateway,Path(folder)/'controller',{})
                miner=Identity();manifest=controller.open('nonpayable-real-contract',dict(id='fixture',files={'config.json':'a'*64}),[miner.id],max_batches=3,**contract(config,0))
                published=signed(json.loads(bucket.objects['public/nonpayable-real-contract/manifest.json']),controller.authority.id)
                self.assertEqual(published,manifest);self.assertEqual(manifest['training_policy'],FIXED_POLICY)
                self.assertEqual(manifest['model_runtime_revision'],REVISION);self.assertEqual(manifest['max_batches'],3)
                self.assertEqual(manifest['heldout_indices'],{spec.id:[2,3]});self.assertIs(manifest['payable'],False)
                self.assertEqual(json.loads((controller.state/'nonpayable-real-contract-manifest.json').read_text()),manifest)
                self.assertEqual(miner.decrypt(manifest['capabilities'][miner.id])['transport'],'direct-r2-v1')
                self.assertIn('public/nonpayable-real-contract/current.json',bucket.objects)
                self.assertNotIn('public/current.json',bucket.objects)
            finally:gateway.server.shutdown();gateway.server.server_close();gateway.thread.join()

    def test_unsupported_policy_refuses_before_gateway_or_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            from unittest.mock import Mock
            gateway=Mock();bucket=MemoryBucket();controller=Controller(bucket,gateway,folder)
            for policy,revision in [('unknown',REVISION),(FIXED_POLICY,'cpu-float32-eager-v2-bounded-toploc'),(True,REVISION)]:
                with self.subTest(policy=policy,revision=revision),self.assertRaises(ValueError):
                    controller.open('nonpayable-invalid',{},[],training_policy=policy,model_runtime_revision=revision)
            gateway.open.assert_not_called();self.assertEqual(bucket.objects,{})
            self.assertFalse(list(Path(folder).glob('*-manifest.json')))

if __name__=='__main__':unittest.main()
