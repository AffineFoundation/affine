"""Prospective quotas reach the original signed opening, never a later rewrite."""
import json
import base64
from nacl.signing import VerifyKey
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from subnet.storage import Gateway,Identity,canonical
from subnet.remote_backend import RemoteController
from subnet.gpu_service import contract,initial_manifest,run
from subnet.forced_sampling import VERSION


class Bucket:
    def __init__(self):self.objects={}
    def json(self,key,value):self.objects[key]=canonical(value)
    def snapshot(self,key,**kwargs):return None
    def presign(self,key,*args,**kwargs):return 'https://test.r2.cloudflarestorage.com/'+key


class QuotaOpeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.bucket=Bucket();self.gateway=Gateway.__new__(Gateway)
        self.gateway.lock=threading.Lock();self.gateway.epochs={};self.gateway.direct_r2=True
        self.gateway.state_path=None;self.gateway.bucket=self.bucket;self.gateway.secret=b'CPU-TEST'
        with patch('subnet.remote_backend.RemoteJobs'):
            self.controller=RemoteController(self.bucket,self.gateway,Path(self.tmp.name)/'compute',{})
        self.spec=dict(id='affine_math',version='prime-v1-1',adapter='prime_v1',config={},max_turns=1,max_output_tokens=256,num_samples=2,success_reward=1.,source_hash='f'*64)
        self.row=dict(spec=self.spec,indices=[0],harness=dict(version='text-tools-v1',policy='autoregressive',max_output_tokens=256,temperature=.8,top_p=1.))
        self.checkpoint={'id':'b'*64,'files':{'config.json':'e'*64}}
        self.miner=Identity().id
        self.config=dict(source_bundle={'sha256':'a'*64,'size':1},heldout=[],epoch_prefix='nonpayable-quota')

    def opening(self,**kwargs):
        return self.controller.open('nonpayable-quota',self.checkpoint,[self.miner],duration=60,environments=[self.row],**kwargs)

    def test_K2L2_is_in_first_signed_public_and_local_manifest(self):
        with patch('subnet.gpu_service.definitions',return_value=[self.row]):
            selected=contract(dict(self.config,K=2,L=2,sampling_policy=dict(version=VERSION,max_attempts=16)),0)
        selected.pop('duration');selected.pop('heldout_indices');selected.pop('environments')
        manifest=self.opening(**selected)
        first=json.loads(self.bucket.objects['public/nonpayable-quota/manifest.json'])
        VerifyKey(bytes.fromhex(first['signer'])).verify(canonical(first['payload']),base64.b64decode(first['signature']))
        self.assertEqual((manifest['K'],manifest['L']),(2,2))
        self.assertEqual(first['payload'],manifest)
        self.assertEqual(self.bucket.objects['public/nonpayable-quota/manifest.json'],canonical(first))
        self.assertEqual(json.loads((self.controller.state/'nonpayable-quota-manifest.json').read_bytes()),manifest)
        self.assertEqual(manifest['sampling_contract']['max_attempts'],16)

    def test_default_manifest_bytes_equal_explicit_one_one(self):
        with patch('subnet.controller.time.time',return_value=1234),patch('subnet.storage.time.time',return_value=1234):
            first=self.opening()
        self.gateway.epochs.clear()
        with patch('subnet.controller.time.time',return_value=1234),patch('subnet.storage.time.time',return_value=1234):
            # Gateway capability secrets are random, so hold its real output constant.
            with patch.object(self.gateway,'open',return_value=first['capabilities']):
                self.gateway.epochs['nonpayable-quota']={'start':first['start']}
                second=self.opening(K=1,L=1)
        self.assertEqual(canonical(first),canonical(second))

    def test_unequal_and_single_configured_quota_preserved(self):
        with patch('subnet.gpu_service.definitions',return_value=[self.row]):
            choice=contract(dict(self.config,K=2),0)
            self.assertEqual((choice['K'],choice['L']),(2,1))
        manifest=self.opening(K=2,L=1)
        self.assertEqual((manifest['K'],manifest['L']),(2,1))
        # The existing aligned zipper consumes one pair, not two positives.
        self.assertEqual(len(list(zip(range(manifest['K']),range(manifest['L'])))),1)

    def test_bad_quotas_refused_before_gateway_or_publication(self):
        bad=[dict(K=x,L=1) for x in (True,None,1.,'2',0,-1,128)]
        bad+=[dict(K=1,L=x) for x in (False,None,2.,'2',0,-1,128)]
        bad+=[dict(K=2,L=2,sampling_policy=dict(version=VERSION,max_attempts=3))]
        for fields in bad:
            with self.subTest(fields=fields),patch.object(self.gateway,'open')as issue:
                with self.assertRaisesRegex(ValueError,'class quotas'):self.opening(**fields)
                issue.assert_not_called();self.assertEqual(self.bucket.objects,{})

    def test_contract_and_initial_manifest_match_and_defaults_unchanged(self):
        with patch('subnet.gpu_service.definitions',return_value=[self.row]):
            default=contract(self.config,0)
            explicit=contract(dict(self.config,K=1,L=1),0)
            self.assertNotIn('K',default);self.assertNotIn('L',default)
            self.assertEqual(default,{k:v for k,v in explicit.items()if k not in ('K','L')})
            self.assertEqual(canonical(initial_manifest(self.config,self.checkpoint)),canonical(initial_manifest(dict(self.config,K=1,L=1),self.checkpoint)))
            initial=initial_manifest(dict(self.config,K=2,L=2,sampling_policy=dict(version=VERSION,max_attempts=4)),self.checkpoint)
            self.assertEqual((initial['K'],initial['L']),(2,2))
            for fields in (dict(K=True),dict(L=None),dict(K=2,L=2,sampling_policy=dict(version=VERSION,max_attempts=3))):
                with self.subTest(fields=fields):
                    with self.assertRaisesRegex(ValueError,'class quotas'):contract(dict(self.config,**fields),0)
                    with self.assertRaisesRegex(ValueError,'class quotas'):initial_manifest(dict(self.config,**fields),self.checkpoint)

    def test_loop_rejects_malformed_config_before_storage_creation(self):
        state=Path(self.tmp.name)/'never-created'
        with self.assertRaisesRegex(ValueError,'class quotas'):run(dict(K=None,state=str(state)))
        self.assertFalse(state.exists())
