"""Public mining status requires signed release and original live process.

Qualification signatures here are CPU fixtures only, never launch evidence.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import unittest

from ops import publish_mimo_public_discovery as observer
from subnet.mimo_lifecycle_adapters import R2Services
from subnet.mimo_release_policy import ReleasePolicy, sign, digest, build_policy
from subnet.storage import canonical
from test_mimo_public_worker import PublicWorkerTests
from test_mimo_lifecycle_adapters import ConditionalBucket


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.fixture = PublicWorkerTests(); self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture; self.root = f.f.root; self.authority = f.operator.id
        self.policy = ReleasePolicy(f.policy, self.authority, allow_public=True)
        self.opening = sign(f.operator, f.m)
        bucket = ConditionalBucket(f.f.bucket)
        self.adapter = R2Services(bucket, f.operator, self.policy, None, None,
            discovery_key=self.policy.namespace()+'/current.json', source_bundle_key='source.tar.gz',
            initial_checkpoint_keys={n:'initial/'+n for n in f.m['checkpoint']['files']},
            source_files={}, runtime_versions={}, trainer_worker=f.f.identity.id,
            publisher_id=f.f.identity.id, trainer_role_config_file='/private/test.json',
            trainer_config={}, readback_state=self.root/'readbacks')
        key=self.policy.namespace()+'/epochs/'+f.m['epoch']+'/mimo/opening.json'
        f.f.bucket.store(key,canonical(self.opening))
        self.receipt=self.adapter.discovery(self.opening,None)
        self.current=dict(policy_sha256=self.policy.policy_sha256,checkpoint=f.m['checkpoint'])
        self.now=f.m['start']+1

    def project(self, **kwargs):
        return observer.project(kwargs.get('policy',self.policy),kwargs.get('current',self.current),
            kwargs.get('opening',self.opening),kwargs.get('receipt',self.receipt),
            authority=self.authority,live=kwargs.get('live',True),now=kwargs.get('now',self.now))

    def test_current_public_group_contract_projects_without_upload_capabilities(self):
        result=self.project()
        self.assertEqual(result['status'],'mining')
        self.assertEqual((result['samples_per_batch'],result['nonce_min'],result['nonce_max']),(8,0,7))
        self.assertEqual(result['min_success'],1)
        self.assertEqual(result['min_completed_failure'],1)
        self.assertEqual(result['max_batches'],9)
        self.assertNotIn('capabilities',result)
        self.assertNotIn('read_urls',result)
        self.assertFalse(result['chain_weight_submission'])
        self.assertEqual(self.project(now=self.opening['payload']['deadline'])['status'],'epoch_closed')
        self.assertEqual(self.project(live=False)['status'],'controller_unavailable')
        self.assertEqual(self.project(opening=None)['status'],'between_epochs')

    def test_private_stale_and_tampered_records_never_open_mining(self):
        m=deepcopy(self.fixture.m);m['payable']=False
        private=build_policy(self.fixture.operator,m,task_assets=self.fixture.assets,enabled=True)
        with self.assertRaisesRegex(ValueError,'private qualification'):
            self.project(policy=ReleasePolicy(private,self.authority))
        bad=deepcopy(self.current);bad['policy_sha256']='0'*64
        with self.assertRaises(ValueError):self.project(current=bad)
        bad=deepcopy(self.receipt);bad['discovery_sha256']='0'*64
        with self.assertRaises(ValueError):self.project(receipt=bad)
        bad=deepcopy(self.receipt);bad['document']['payload']['current_url']='https://evil.invalid/'
        with self.assertRaises(Exception):self.project(receipt=bad)
        with self.assertRaises(ValueError):self.project(now=self.receipt['document']['payload']['current_url_expires_at'])

    def test_selector_requires_exact_source_config_physical_entry_and_opt_in(self):
        source=self.root/'source';(source/'ops').mkdir(parents=True);(source/'subnet').mkdir()
        (source/'subnet/__init__.py').write_text('# CPU fixture\n')
        entry=source/'ops/mimo_lifecycle.py';entry.write_text('# exact lifecycle entry\n')
        from subnet.mimo_worker import source_inventory
        config=dict(enabled=True,release_policy=self.fixture.policy,
            adapters=dict(source_files=source_inventory(source)))
        config_path=self.root/'lifecycle.private.json';config_path.write_bytes(canonical(config));config_path.chmod(0o600)
        def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        selection=dict(enabled=True,authority=self.authority,lifecycle_unit='fixture.service',
            lifecycle_entry=ref(entry),lifecycle_config=ref(config_path),
            source_sha256=self.fixture.m['source_bundle']['sha256'])
        proc=self.root/'proc';p=proc/'42';p.mkdir(parents=True)
        argv=['python','-B',str(entry),'--config',str(config_path),'--allow-public-mimo']
        (p/'cmdline').write_bytes(b'\0'.join(v.encode()for v in argv)+b'\0')
        (p/'stat').write_text('42 (python) '+' '.join(['S']+['0']*18+['12345']+['0']*10))
        inspect=lambda:'MainPID=42\nActiveState=active\n'
        self.assertEqual(observer.selector(selection,proc=proc,inspect_unit=inspect)[2]['child_ticks'],'12345')
        (p/'cmdline').write_bytes(b'\0'.join(v.encode()for v in argv[:-1])+b'\0')
        with self.assertRaisesRegex(ValueError,'physical MiMo'):observer.selector(selection,proc=proc,inspect_unit=inspect)
        (p/'cmdline').write_bytes(b'\0'.join(v.encode()for v in argv)+b'\0')
        (source/'subnet/__init__.py').write_text('# source drift\n')
        with self.assertRaisesRegex(ValueError,'source inventory'):observer.selector(selection,proc=proc,inspect_unit=inspect)
        config_path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'pinned observer'):observer.selector(selection,proc=proc,inspect_unit=inspect)

    def test_failed_binding_replaces_stale_public_status_without_private_data(self):
        target=self.root/'mining.json';target.write_text('{"accepting_submissions":true,"current_url":"stale"}')
        with patch.object(observer,'selector',side_effect=ValueError('private diagnostic')):
            result=observer.publish(dict(authority=self.authority,target=str(target)),now=self.now)
        self.assertEqual(result['status'],'discovery_binding_error')
        saved=json.loads(target.read_bytes());self.assertNotIn('current_url',saved)
        self.assertFalse(saved['accepting_submissions'])
        self.assertEqual(target.stat().st_mode&0o777,0o644)


if __name__=='__main__':unittest.main()
