import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np

from subnet import cli
from subnet.artifact_budget import LEGACY,LONG,LONG_REVISION
from subnet.backend_profiles import HOPPER_REVISION,profile
from subnet.batches import unpack
from subnet.harness import source_hash
from subnet.miner import Miner
from subnet.storage import Identity,encrypt


class PublicMinerContract(unittest.TestCase):
    def manifest(self,identity='registered'):
        revision,backend,numerical=profile(HOPPER_REVISION)
        return dict(epoch='nonpayable-public',checkpoint={'id':'pinned'},K=1,L=1,
            max_batches=1,deadline=1700001000,transport_policy='direct-r2-v1',
            harness_source_hash=source_hash(),
            capabilities={identity:{}},model_runtime_revision=revision,
            backend_profile=backend,numerical_policy=numerical,artifact_policy=LONG_REVISION,
            source_bundle={'sha256':'c'*64},
            environments=[dict(env_id='math',spec=dict(id='math',version='native-v1',num_samples=1),indices=[0],
                harness=dict(version='text-tools-long-v2',policy='autoregressive',max_output_tokens=1024,temperature=.8,top_p=1.))])

    def capability(self,identity='registered'):
        return dict(identity=identity,epoch='nonpayable-public',deadline=1700001000,
            transport='direct-r2-v1',headers={'Content-Type':'application/octet-stream'},
            put_url='https://account.r2.cloudflarestorage.com/bucket/private/nonpayable-public/staging/'+identity+'.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test')

    def test_decrypted_delegation_preserves_narrow_direct_transport(self):
        owner=Identity();manifest=self.manifest(owner.id);cap=self.capability(owner.id)
        encrypted=encrypt(owner.id,{k:v for k,v in cap.items() if k not in ('identity','epoch')})
        decrypted=owner.decrypt(encrypted)
        delegated=dict(decrypted,identity=owner.id,epoch=manifest['epoch'])
        self.assertEqual(cli.delegated_capability(manifest,delegated),decrypted)

    def test_cross_epoch_identity_deadline_and_object_delegations_refuse(self):
        manifest=self.manifest();good=self.capability()
        for change in [dict(epoch='old'),dict(identity='unregistered'),dict(deadline=1001),
                       dict(deadline=True),dict(transport='gateway-v1'),dict(headers={}),
                       dict(put_url=good['put_url'].replace('/registered.zip','/other.zip'))]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                cli.delegated_capability(manifest,dict(good,**change))

    def test_public_cli_upload_and_restart_keep_signed_long_tensor_budget(self):
        manifest=self.manifest();cap=self.capability()
        runtime=SimpleNamespace(spec=SimpleNamespace(version='native-v1'))
        def rollout(index,seed):
            positive=seed%2==0
            return dict(classification='positive' if positive else 'negative',reward=float(positive),
                        turns=[{'output':[seed]}]),[np.arange(1024*2,dtype=np.float32).reshape(1024,2)]
        runtime.rollout=rollout
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);capfile=root/'cap.json';capfile.write_text(json.dumps(cap))
            args=SimpleNamespace(search_budget=2,env_id='math',indices=[0],cap_file=str(capfile),key=None,
                state=str(root/'state'),manifest_url='https://manifest.invalid',current_url=None,
                gateway='https://unused.invalid',authority='trusted',max_batches=1,once=True,source_bundle_sha256='c'*64)
            response=Mock(status_code=200)
            # Scale the legacy compressed ceiling down to exercise both real
            # packing paths without allocating a 100 MB unit-test artifact.
            with patch.dict(LEGACY,compressed_bytes=1),patch.object(cli,'fetch_signed',return_value=manifest),patch.object(cli,'checkpoint_download',return_value=root),\
                 patch.object(cli,'identity') as key_access,patch('subnet.miner.make_runtime',return_value=runtime),\
                 patch('subnet.miner.requests.put',return_value=response) as upload,patch.object(cli.time,'time',return_value=1700000010),\
                 patch.object(cli.time,'time_ns',return_value=100):
                cli.run(args)
                key_access.assert_not_called()
                self.assertEqual(upload.call_args.kwargs['headers'],cap['headers'])
                data=upload.call_args.kwargs['data']
            # The real NPY framing exceeds legacy's 512 rows and must stay intact.
            with self.assertRaisesRegex(ValueError,'tensor header'):unpack(data)
            records=unpack(data,budget=LONG);self.assertEqual(len(records),1)
            self.assertEqual(records[0][1][1][0].shape,(1024,2))
            state=root/'state'/(manifest['epoch']+'-registered.zip')
            restored=Miner(SimpleNamespace(id='registered'),manifest,root,
                capability=cli.delegated_capability(manifest,cap),state_path=state)
            self.assertEqual(len(restored.batches),1)
            with patch.dict(LEGACY,compressed_bytes=1),patch('subnet.miner.requests.put',return_value=response) as upload,patch('subnet.miner.time.time',return_value=1700000010):
                restored.upload()
            self.assertEqual(unpack(upload.call_args.kwargs['data'],budget=LONG)[0][1][1][0].shape,(1024,2))
            legacy=copy.deepcopy(manifest);legacy.pop('artifact_policy')
            with self.assertRaisesRegex(ValueError,'tensor header'):
                Miner(SimpleNamespace(id='registered'),legacy,root,
                    capability=cli.delegated_capability(legacy,cap),state_path=state)

    def test_bad_delegation_refuses_before_checkpoint_download(self):
        with tempfile.TemporaryDirectory() as temp:
            capfile=Path(temp)/'cap.json';capfile.write_text(json.dumps(dict(self.capability(),epoch='old')))
            args=SimpleNamespace(cap_file=str(capfile),key=None,state=temp,manifest_url='https://manifest.invalid',
                current_url=None,authority='trusted',search_budget=2,env_id='math',indices=[0])
            with patch.object(cli,'fetch_signed',return_value=self.manifest()),patch.object(cli,'checkpoint_download') as download:
                with self.assertRaisesRegex(ValueError,'epoch mismatch'):cli.run(args)
                download.assert_not_called()

    def test_compression_assertion_cannot_override_signed_manifest_before_download(self):
        with tempfile.TemporaryDirectory()as temp:
            capfile=Path(temp)/'cap.json';capfile.write_text(json.dumps(self.capability()))
            args=SimpleNamespace(cap_file=str(capfile),key=None,state=temp,manifest_url='https://manifest.invalid',current_url=None,authority='trusted',compression_level=1)
            with patch.object(cli,'fetch_signed',return_value=self.manifest()),patch.object(cli,'checkpoint_download')as download,self.assertRaisesRegex(ValueError,'must match signed manifest'):cli.run(args)
            download.assert_not_called()

    def test_unapproved_artifact_escalation_refuses_before_runtime(self):
        manifest=self.manifest();manifest['model_runtime_revision']='cuda-bf16-eager-sm86-v1'
        with patch('subnet.miner.make_runtime') as runtime,self.assertRaisesRegex(ValueError,'artifact policy'):
            Miner(SimpleNamespace(id='registered'),manifest,'unused',capability=self.capability())
        runtime.assert_not_called()


if __name__=='__main__':unittest.main()
