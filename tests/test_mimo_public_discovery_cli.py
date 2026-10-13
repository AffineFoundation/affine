"""Actual signed discovery/bootstrap/CLI and compact upload; no GPU or network."""
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
from urllib.parse import unquote,urlparse
import unittest
from unittest.mock import patch

import test_mimo_public_worker as public_fixtures
from test_mimo_lifecycle_adapters import ConditionalBucket
from test_complete_group_mimo import MimoCollectionRuntime
from subnet.complete_group_transport import CompleteGroupTransport,paths
from subnet.mimo_lifecycle_adapters import R2Services
from subnet.mimo_release_policy import ReleasePolicy,sign
from subnet.storage import canonical


class PublicDiscoveryTests(unittest.TestCase):
    def test_actual_release_archive_admits_only_the_pinned_vendor_subtree(self):
        from subnet.mimo_release import build,MODEL_ID,MODEL_REVISION,VERL_REVISION
        from subnet.source_bootstrap import admitted_files,public_path,TASK_ASSET
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)/'source';(root/'subnet').mkdir(parents=True)
            (root/'subnet/__init__.py').write_text('')
            (root/'subnet/cli.py').write_text('# authenticated miner fixture\n')
            vendor=root/'vendor/mimo_verl_pinned';vendor.mkdir(parents=True)
            (vendor/'LICENSE').write_text('Apache-2.0 fixture\n')
            (vendor/'algorithm.py').write_text('# pinned upstream fixture\n')
            task=root/TASK_ASSET;task.parent.mkdir();task.write_bytes(b'{"tasks":[]}')
            settings=dict(model=dict(id=MODEL_ID,revision=MODEL_REVISION),reference_revision=VERL_REVISION,
                task_assets={TASK_ASSET:dict(sha256=hashlib.sha256(task.read_bytes()).hexdigest(),bytes=task.stat().st_size)})
            destination=Path(directory)/'release';release=build(root,destination,settings=settings)
            files=admitted_files((destination/'source.tar.gz').read_bytes(),
                dict(sha256=release['archive_sha256'],size=release['archive_bytes']))
            self.assertEqual(set(files),set(release['source']))
            self.assertIn('vendor/mimo_verl_pinned/algorithm.py',files)
        for name in ('vendor/other/code.py','vendor/mimo_verl_pinned2/code.py',
                     'vendor/mimo_verl_pinned/../secrets.py','vendor/mimo_verl_pinned/key.seed'):
            with self.subTest(name=name),self.assertRaises(ValueError):public_path(name)

    def test_actual_adapter_descriptor_feeds_signed_bootstrap_and_real_miner_cli(self):
        fixture=public_fixtures.PublicWorkerTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        f=fixture.f;m=fixture.m
        policy=ReleasePolicy(fixture.policy,fixture.operator.id,allow_public=True)
        bucket=ConditionalBucket(f.bucket)
        adapter=R2Services(bucket,fixture.operator,policy,None,None,
            discovery_key=policy.namespace()+'/current.json',
            source_bundle_key='sources/private-test-source.tar.gz',
            initial_checkpoint_keys={n:'initial/'+n for n in m['checkpoint']['files']},
            source_files={},runtime_versions={},trainer_worker=f.identity.id,publisher_id=f.identity.id,
            trainer_role_config_file='/private/fixture.json',trainer_config={},readback_state=f.root/'readbacks')
        opening=sign(fixture.operator,m)
        key=policy.namespace()+'/epochs/'+m['epoch']+'/mimo/opening.json'
        f.bucket.store(key,canonical(opening))
        adapter.discovery(opening,None)
        current_url=bucket.presign(adapter.discovery_key,'get_object',86400)
        def fetch(url,limit):
            object_key=unquote(urlparse(url).path).removeprefix('/candidate/')
            data=f.bucket.objects[object_key][0]
            self.assertLessEqual(len(data),limit)
            return data
        from subnet.source_bootstrap import manifest,signed
        self.assertEqual(manifest(current_url,f.authority,fetch=fetch),m)
        from subnet.cli import run
        from subnet.miner import Miner
        actors=[];runtimes=[]
        def actor(*args,**kwargs):
            value=Miner(*args,**kwargs,complete_group_transport=CompleteGroupTransport(http=f.http))
            actors.append(value);return value
        def runtime(*args,**kwargs):
            value=MimoCollectionRuntime(m,f.identity.id);runtimes.append(value);return value
        args=SimpleNamespace(gateway='https://unused.invalid',authority=f.authority,manifest_url=None,
            current_url=current_url,key='private-fixture',cap_file=None,state=str(f.root/'public-cli'),
            once=True,max_batches=1,search_budget=32)
        with patch('subnet.cli.identity',return_value=f.identity),\
             patch('subnet.cli.fetch_signed',side_effect=lambda url,authority:signed(fetch(url,32*1024**2),authority)),\
             patch('subnet.cli.checkpoint_download',return_value='CPU-fixture'),\
             patch('subnet.cli.check_runtime_profile'),patch('subnet.model.check_runtime_profile'),\
             patch('subnet.cli.Miner',side_effect=actor),\
             patch('subnet.miner.make_runtime',side_effect=runtime):
            run(args)
        self.assertEqual(len(runtimes),1)
        self.assertEqual([nonce for _,nonce in runtimes[0].calls],list(range(8)))
        commit=json.loads(f.bucket.objects[paths(m,f.identity.id)][0])['payload']
        self.assertEqual(len(commit['groups']),1)
        self.assertEqual([r['seed']for r in actors[0].batches[0][0]['rollouts']],list(range(8)))
        self.assertTrue(actors[0]._closed);self.assertIsNone(actors[0]._group_journal)


if __name__=='__main__':unittest.main()
