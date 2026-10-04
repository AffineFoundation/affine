"""Real SQLite/signature concurrency controls; no R2 request or credential use."""
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlparse

import boto3
from botocore.config import Config
from nacl.signing import SigningKey

from subnet.upload_broker import (
    VERSION, TRANSPORT, CONTENT_TYPE, SizeBoundUploadBroker, UploadBrokerServer,
    authenticate, immutable_snapshot_key, sha, signed,
)


class FakePresigner:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.omit_length = False
        self.omit_condition = False

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        self.calls.append((operation, dict(Params), ExpiresIn))
        headers = ['content-type', 'host']
        if 'ContentLength' in Params and not self.omit_length:
            headers.append('content-length')
        if 'IfNoneMatch' in Params and not self.omit_condition:
            headers.append('if-none-match')
        query = dict(X_Amz_Algorithm='AWS4-HMAC-SHA256', X_Amz_Signature='00' * 32,
            X_Amz_SignedHeaders=';'.join(sorted(headers)), X_Amz_Expires=str(ExpiresIn),
            X_Amz_Date=datetime.fromtimestamp(self.clock(), timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        return 'https://unit-test.r2.cloudflarestorage.com/unit-test/' + Params['Key'] + '?' + urlencode(
            {k.replace('_', '-'): v for k, v in query.items()})


class UploadBrokerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.clock = 100.
        self.operator = SigningKey.generate()
        self.miners = [SigningKey.generate(), SigningKey.generate()]
        self.identities = [key.verify_key.encode().hex() for key in self.miners]
        self.authority = self.operator.verify_key.encode().hex()
        self.client = FakePresigner(lambda: self.clock)
        self.bucket = SimpleNamespace(name='unit-test', client=self.client)
        self.path = Path(self.folder.name)/'uploads.sqlite3'
        self.broker = SizeBoundUploadBroker(self.path, self.operator, self.bucket, clock=lambda: self.clock)
        self.epoch = 'nonpayable-broker-TEST'
        self.registrations = signed(dict(epoch=self.epoch, snapshot_block=123,
            registrations={str(i): dict(uid=i+1, public_key=key) for i, key in enumerate(self.identities)}), self.operator)
        self.policy = dict(version=VERSION, transport=TRANSPORT, epoch=self.epoch,
            start=90, deadline=300, identities=sorted(self.identities),
            registration_document_sha256=sha(self.registrations), max_compressed_bytes=1000,
            grant_ttl_seconds=30, max_grants_per_identity=3,
            max_declared_bytes_per_identity=2000, minimum_issue_interval_seconds=5,
            snapshot_mode='immutable-snapshot')
        self.open()

    def open(self):
        return self.broker.register_epoch(signed(self.policy, self.operator), self.registrations)

    def payload(self, miner=0, nonce=1, size=500):
        return dict(version=VERSION, action='grant', epoch=self.epoch,
            identity=self.identities[miner], nonce=format(nonce, '064x'), bytes=size, at=self.clock)

    def request(self, miner=0, nonce=1, size=500):
        return self.broker.request(dict(authentication='ed25519',
            document=signed(self.payload(miner, nonce, size), self.miners[miner])))

    def test_exact_size_immutable_condition_scope_and_signed_grant(self):
        grant = authenticate(self.request(), self.authority)
        operation, params, expiry = self.client.calls[-1]
        self.assertEqual(operation, 'put_object')
        self.assertEqual(params['ContentLength'], 500)
        self.assertEqual(params['IfNoneMatch'], '*')
        self.assertEqual(params['ContentType'], CONTENT_TYPE)
        self.assertEqual(params['Key'], immutable_snapshot_key(self.epoch, self.identities[0], format(1, '064x')))
        self.assertEqual(grant['headers'], {'Content-Type': CONTENT_TYPE, 'Content-Length': '500', 'If-None-Match': '*'})
        self.assertEqual(grant['deadline'], 300)
        self.assertEqual(grant['expires_at'], 130)
        self.assertEqual(grant['quota']['grants_remaining'], 2)

    def test_nonce_is_idempotent_and_conflicts_do_not_issue(self):
        request = dict(authentication='ed25519', document=signed(self.payload(), self.miners[0]))
        first = self.broker.request(request)
        self.assertEqual(self.broker.request(request), first)
        self.assertEqual(len(self.client.calls), 1)
        with self.assertRaisesRegex(ValueError, 'nonce conflict'):
            self.request(size=600)
        self.clock = 131
        with self.assertRaisesRegex(ValueError, 'expired'):
            self.broker.request(request)
        self.assertEqual(len(self.client.calls), 1)

    def test_quota_rate_byte_budget_and_restart_persist(self):
        self.request(size=1000)
        with self.assertRaisesRegex(ValueError, 'rate'):
            self.request(nonce=2, size=1000)
        self.clock += 5
        self.request(nonce=2, size=1000)
        resumed = SizeBoundUploadBroker(self.path, self.operator, self.bucket, clock=lambda: self.clock)
        with self.assertRaisesRegex(ValueError, 'quota'):
            resumed.request(dict(authentication='ed25519', document=signed(self.payload(nonce=3, size=1), self.miners[0])))
        self.assertEqual(len(self.client.calls), 2)
        self.request(miner=1)
        self.assertEqual(len(self.client.calls), 3)

    def test_operator_delegation_needs_no_miner_private_key_and_is_scoped(self):
        authorization = self.broker.delegate(self.epoch, self.identities[0])
        request = dict(authentication='delegated', payload=self.payload(), authorization=authorization)
        grant = authenticate(self.broker.request(request), self.authority)
        self.assertEqual(grant['identity'], self.identities[0])
        bad = copy.deepcopy(request)
        bad['payload'] = self.payload(miner=1, nonce=2)
        with self.assertRaisesRegex(ValueError, 'delegated'):
            self.broker.request(bad)
        bad = copy.deepcopy(request)
        bad['authorization']['payload']['deadline'] = 400
        with self.assertRaises(Exception):
            self.broker.request(bad)

    def test_wrong_signature_unknown_identity_bad_size_and_deadline_refuse(self):
        wrong = dict(authentication='ed25519', document=signed(self.payload(), self.miners[1]))
        with self.assertRaises(ValueError): self.broker.request(wrong)
        for size in (True, 0, -1, 1001, 2_000_000_001):
            with self.subTest(size=size), self.assertRaises(ValueError): self.request(size=size)
        outsider = SigningKey.generate()
        request = self.payload()
        request['identity'] = outsider.verify_key.encode().hex()
        with self.assertRaisesRegex(ValueError, 'scope'):
            self.broker.request(dict(authentication='ed25519', document=signed(request, outsider)))
        self.clock = 300
        with self.assertRaisesRegex(ValueError, 'scope'): self.request()
        self.assertFalse(self.client.calls)

    def test_missing_size_or_immutable_signed_header_rolls_back_issuance(self):
        for field in ('omit_length', 'omit_condition'):
            setattr(self.client, field, True)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'size-signed'):
                self.request()
            setattr(self.client, field, False)
        grant = authenticate(self.request(), self.authority)
        self.assertEqual(grant['quota']['grants_remaining'], 2)
        self.assertEqual(grant['quota']['declared_bytes_remaining'], 1500)

    def test_atomic_concurrent_nonce_deduplication_across_connections(self):
        barrier = threading.Barrier(2)
        # Construct independent connections before simultaneous requests.
        peers = [SizeBoundUploadBroker(self.path, self.operator, self.bucket, clock=lambda: self.clock) for _ in range(2)]
        request = dict(authentication='ed25519', document=signed(self.payload(), self.miners[0]))
        def attempt(index):
            barrier.wait()
            return peers[index].request(request)
        with ThreadPoolExecutor(2) as pool:
            grants = list(pool.map(attempt, range(2)))
        self.assertEqual(grants[0], grants[1])
        self.assertEqual(len(self.client.calls), 1)

    def test_atomic_distinct_nonce_issuance_preserves_rate_limit(self):
        barrier = threading.Barrier(2)
        def attempt(index):
            barrier.wait()
            try: return self.request(nonce=index+1)
            except ValueError: return None
        with ThreadPoolExecutor(2) as pool:
            grants = list(pool.map(attempt, range(2)))
        self.assertEqual(sum(grant is not None for grant in grants), 1)
        self.assertEqual(len(self.client.calls), 1)

    def test_immutable_epoch_registration_and_duplicate_identity_refuse(self):
        self.assertEqual(self.open(), self.open())
        changed = copy.deepcopy(self.policy)
        changed['deadline'] += 1
        with self.assertRaisesRegex(ValueError, 'collision'):
            self.broker.register_epoch(signed(changed, self.operator), self.registrations)
        changed = copy.deepcopy(self.registrations)
        changed['payload']['registrations']['0']['public_key'] = 'ff'*32
        changed = signed(changed['payload'], self.operator)
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            self.broker.register_epoch(signed(self.policy, self.operator), changed)

    def test_finalization_lists_only_original_known_keys_without_capabilities(self):
        self.request()
        self.clock = 105
        self.request(nonce=2)
        with self.assertRaisesRegex(ValueError, 'closed'):
            self.broker.finalization_candidates(self.epoch, self.identities[0])
        self.clock = 300
        candidates = self.broker.finalization_candidates(self.epoch, self.identities[0])
        self.assertEqual(len(candidates), 2)
        self.assertNotEqual(candidates[0]['staging_key'], candidates[1]['staging_key'])
        self.assertTrue(all('put_url' not in row and 'bearer_secret' not in row for row in candidates))

    def test_real_offline_sigv4_signs_exact_length_and_condition(self):
        client = boto3.client('s3', endpoint_url='https://unit-test.r2.cloudflarestorage.com',
            region_name='auto', aws_access_key_id='UNIT_TEST_ONLY',
            aws_secret_access_key='UNIT_TEST_NOT_A_REAL_SECRET', config=Config(signature_version='s3v4'))
        url = client.generate_presigned_url('put_object', Params=dict(Bucket='unit-test',
            Key='private/test.zip', ContentType=CONTENT_TYPE, ContentLength=123, IfNoneMatch='*'), ExpiresIn=60)
        headers = set(parse_qs(urlparse(url).query)['X-Amz-SignedHeaders'][0].split(';'))
        self.assertTrue({'host', 'content-type', 'content-length', 'if-none-match'} <= headers)

    def test_http_loopback_route_and_safe_error_body(self):
        import requests
        server = UploadBrokerServer(('127.0.0.1', 0), self.broker)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = 'http://127.0.0.1:' + str(server.server_port) + '/v1/upload-grants'
        request = dict(authentication='ed25519', document=signed(self.payload(), self.miners[0]))
        response = requests.post(url, json=request, timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(authenticate(response.json(), self.authority)['bytes'], 500)
        response = requests.post(url, json={'authorization': 'PRIVATE_SENTINEL'}, timeout=5)
        self.assertEqual(response.status_code, 403)
        self.assertNotIn('PRIVATE_SENTINEL', response.text)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        with self.assertRaisesRegex(ValueError, 'loopback'):
            UploadBrokerServer(('0.0.0.0', 0), self.broker)


if __name__ == '__main__':
    unittest.main()
