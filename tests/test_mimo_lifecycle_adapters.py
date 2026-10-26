"""Conditional discovery/readback and disabled host entrypoint, no live services."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from botocore.exceptions import ClientError

import test_mimo_lifecycle as lifecycle_fixture
from subnet.mimo_lifecycle_adapters import R2Services
from subnet.mimo_release_policy import descriptor,digest,sign,build_policy,ReleasePolicy
from subnet.storage import canonical
from subnet.backend_jobs import signed


class ConditionalBucket:
    def __init__(self,original):self.original=original;self.name=original.name;self.client=self;self.reads=0;self.lose_ack=False
    def get_object(self,**kwargs):
        self.reads+=1
        try:response=self.original.get_object(**kwargs)
        except KeyError:raise ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject')
        response['ETag']='"'+hashlib.sha256(self.original.objects[kwargs['Key']][0]).hexdigest()+'"'
        return response
    def put_object(self,*,IfMatch=None,**kwargs):
        if IfMatch is not None:
            existing=self.original.objects.get(kwargs['Key'])
            if existing is None or '"'+hashlib.sha256(existing[0]).hexdigest()+'"'!=IfMatch:
                raise ClientError({'Error':{'Code':'PreconditionFailed'},'ResponseMetadata':{'HTTPStatusCode':412}},'PutObject')
        result=self.original.put_object(**kwargs)
        if self.lose_ack:self.lose_ack=False;raise OSError('conditional write ack lost')
        return result
    def presign(self,*a):return self.original.presign(*a)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.f=lifecycle_fixture.LifecycleTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.bucket=ConditionalBucket(self.f.bucket)
        self.a=R2Services(self.bucket,self.f.authority,self.f.lifecycle.policy,None,None,
            discovery_key=self.f.lifecycle.policy.namespace()+'/current.json',
            source_bundle_key='sources/test-release.tar.gz',
            initial_checkpoint_keys={n:'initial/'+n for n in self.f.base['files']},source_files={},runtime_versions={},
            trainer_worker=self.f.worker.id,publisher_id=self.f.publisher.id,
            trainer_role_config_file='/private/config.json',trainer_config={},readback_state=self.f.root/'readbacks')
        self.f.lifecycle.tick(now=self.f.now)
        self.opening=self.f.discovery_value

    def test_lost_conditional_put_ack_reads_back_same_exact_signed_opening(self):
        self.bucket.lose_ack=True
        with self.assertRaises(OSError):self.a.discovery(self.opening,None)
        receipt=self.a.discovery(self.opening,None)
        self.assertEqual(receipt['sha256'],digest(self.opening))
        data=self.f.bucket.objects[self.a.discovery_key][0]
        current=signed(json.loads(data),self.f.authority.id)
        self.assertEqual(receipt['discovery_sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(current['manifest_sha256'],digest(self.opening))
        self.assertIn('/mimo/opening.json?',current['manifest_url'])

    def test_source_capability_renews_after_one_bounded_archive_readback(self):
        raw=b'exact-authorized-source-archive';source={'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)}
        manifest=dict(self.opening['payload'],source_bundle=dict(source,read_url='https://old.invalid'))
        envelope=build_policy(self.f.authority,manifest,
            task_assets=self.a.policy.scope['task_assets'],enabled=True)
        self.a.policy=ReleasePolicy(envelope,self.f.authority.id)
        self.f.bucket.store(self.a.source_bundle_key,raw)
        calls=self.bucket.reads
        with patch.object(self.bucket,'presign',side_effect=['fresh-first','fresh-second']):
            self.assertEqual(self.a.source_bundle(source),dict(source,url='fresh-first'))
            self.assertEqual(self.a.source_bundle(source),dict(source,url='fresh-second'))
        self.assertEqual(self.bucket.reads-calls,1)
        self.a._source_checked=None;self.f.bucket.store(self.a.source_bundle_key,b'x'*len(raw))
        with self.assertRaisesRegex(ValueError,'source archive full readback hash'):
            self.a.source_bundle(source)

    def test_stale_predecessor_cannot_overwrite_other_discovery(self):
        original=self.a.discovery(self.opening,None)
        raw=self.f.bucket.objects[self.a.discovery_key][0]
        changed=json.loads(canonical(self.opening));changed['payload']['epoch']='nonpayable-other-epoch'
        changed=sign(self.f.authority,changed['payload'])
        with self.assertRaisesRegex(ValueError,'changed from'):
            self.a.discovery(changed,'0'*64)
        self.assertEqual(self.f.bucket.objects[self.a.discovery_key][0],raw)
        key=self.a.policy.namespace()+'/epochs/'+changed['payload']['epoch']+'/mimo/opening.json'
        self.f.bucket.store(key,canonical(changed))
        self.assertEqual(self.a.discovery(changed,original['discovery_sha256'])['sha256'],digest(changed))

    def published(self,corrupt=False):
        bodies={'config.json':b'{}','model.safetensors':b'real-test-model-bytes'}
        cp={'files':{n:hashlib.sha256(b).hexdigest()for n,b in bodies.items()}};cp['id']=digest(cp['files'])
        objects={}
        for name,body in bodies.items():
            key=self.a.policy.namespace()+'/checkpoints/'+cp['id']+'/model/'+name
            objects[name]=dict(key=key,sha256=cp['files'][name],size=len(body))
            self.f.bucket.store(key,b'X'*len(body)if corrupt and name.endswith('safetensors')else body)
        return dict(descriptor=cp,output_checkpoint=cp['id'],objects=objects)

    def test_model_readback_hashes_every_byte_once_and_keeps_no_model_cache(self):
        p=self.published();self.assertEqual(self.a.readback_publication(p),p['descriptor'])
        calls=self.bucket.reads;self.assertEqual(calls,2)
        self.assertEqual(self.a.readback_publication(p),p['descriptor']);self.assertEqual(self.bucket.reads,calls)
        self.assertEqual([x.suffix for x in self.a.readback_state.iterdir()],['.json'])

    def test_corrupt_or_cross_namespace_publication_never_advances(self):
        p=self.published(corrupt=True)
        with self.assertRaisesRegex(ValueError,'readback hash'):self.a.readback_publication(p)
        self.assertEqual(list(self.a.readback_state.iterdir()),[])
        p=self.published();p['objects']['config.json']['key']='another-run/model/config.json'
        with self.assertRaisesRegex(ValueError,'scoped bounded'):self.a.readback_publication(p)

    def test_four_independent_readbacks_overlap_before_receipt_cache(self):
        import threading
        bodies={f'model-{i}.safetensors':bytes([i])*100 for i in range(4)};bodies['config.json']=b'{}'
        cp={'files':{n:hashlib.sha256(b).hexdigest()for n,b in bodies.items()}};cp['id']=digest(cp['files'])
        objects={}
        for name,body in bodies.items():
            key=self.a.policy.namespace()+'/checkpoints/'+cp['id']+'/model/'+name
            objects[name]=dict(key=key,sha256=cp['files'][name],size=len(body));self.f.bucket.store(key,body)
        barrier=threading.Barrier(4);original=self.bucket.get_object
        def simultaneous(**kwargs):
            self.assertEqual(list(self.a.readback_state.iterdir()),[])
            if kwargs['Key'].endswith('.safetensors'):barrier.wait(timeout=5)
            return original(**kwargs)
        with patch.object(self.bucket,'get_object',side_effect=simultaneous):
            self.assertEqual(self.a.readback_publication(dict(descriptor=cp,output_checkpoint=cp['id'],objects=objects)),cp)
        self.assertEqual(self.bucket.reads,5)
        self.assertEqual(len(list(self.a.readback_state.iterdir())),1)

    def test_disabled_cli_never_opens_credentials_or_remote_services(self):
        from ops.mimo_lifecycle import run
        path=self.f.root/'disabled.json';path.write_text('{"enabled":false}');path.chmod(0o600)
        with patch('subnet.storage.Bucket')as bucket:
            with self.assertRaisesRegex(ValueError,'disabled'):
                run(SimpleNamespace(config=str(path),allow_public_mimo=False,once=True,interval=10))
        bucket.assert_not_called()


if __name__=='__main__':unittest.main()
