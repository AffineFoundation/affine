"""Submitted rejection must never hide native/transport failure on admission."""
import io
from dataclasses import replace
import json
from unittest.mock import patch
import unittest

from botocore.exceptions import ClientError
import test_mimo_lifecycle as fixtures
from subnet.complete_group_capture import SubmissionRejected
from subnet.complete_group_transport import paths


class CaptureTaxonomyTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.LifecycleTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.lifecycle.tick(now=self.f.now);self.f.upload()

    def assert_unsealed(self):
        self.assertFalse((self.f.root/'run/epochs/0/coordinator/POPULATION.json').exists())
        self.assertEqual(json.loads((self.f.root/'run/CURRENT.json').read_bytes())['checkpoint'],self.f.base)

    def test_trusted_native_valueerror_does_not_discard_original_miner(self):
        def broken(*args):raise ValueError('native pinned source temporarily unavailable')
        self.f.lifecycle.resolve_task=broken
        with self.assertRaisesRegex(ValueError,'native pinned'):
            self.f.lifecycle.tick(now=self.f.now+120)
        self.assert_unsealed()

    def test_truncated_body_is_infrastructure_not_miner_rejection(self):
        original=self.f.bucket.get_object
        def short(**kwargs):
            result=original(**kwargs)
            if '/commitments/' in kwargs['Key']:result['Body']=io.BytesIO(b'')
            return result
        with patch.object(self.f.bucket,'get_object',side_effect=short):
            with self.assertRaisesRegex(OSError,'transport read'):
                self.f.lifecycle.tick(now=self.f.now+120)
        self.assert_unsealed()

    def test_bad_miner_signature_is_isolated_without_training(self):
        m=self.f.discovery_value['payload'];key=paths(m,self.f.miner.id)
        raw,stamp=self.f.bucket.objects[key];bad=json.loads(raw)
        bad['signature']='A'*88
        self.f.bucket.objects[key]=(json.dumps(bad).encode(),stamp)
        result=self.f.lifecycle.tick(now=self.f.now+120)
        self.assertEqual(result['training'],'no-update');self.assertEqual(self.f.jobs,{})
        rejected=json.loads((self.f.root/'run/epochs/0/CAPTURE-REJECTIONS.json').read_bytes())['payload']
        self.assertIn(self.f.miner.id,rejected['rejected'])

    def test_absent_committed_proof_isolated_but_provider_error_defers(self):
        original=self.f.bucket.head_object
        for code in ('InternalError','NoSuchKey'):
            def fail(**kwargs):
                if '/staging/' in kwargs['Key']:
                    raise ClientError({'Error':{'Code':code}},'HeadObject')
                return original(**kwargs)
            with patch.object(self.f.bucket,'head_object',side_effect=fail):
                if code=='InternalError':
                    with self.assertRaises(ClientError):self.f.lifecycle.tick(now=self.f.now+120)
                    self.assert_unsealed()
                else:
                    result=self.f.lifecycle.tick(now=self.f.now+120)
                    self.assertEqual(result['training'],'no-update')


if __name__=='__main__':unittest.main()
