"""v8.2.3: signed per-miner group cap parameter and sealed upload-grant delivery.

All HTTP/S3 are in-memory. The grant pointer in the signed manifest has constant
size; the grant object carries the unchanged v1 capability (4 URLs per slot)."""
from test_complete_group_fixtures import *
import hashlib
import requests
from urllib.parse import urlparse, unquote
import test_complete_group_transport as base
from subnet.storage import Identity
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
from subnet.complete_group_transport import (capability, resolve_capability, validate_pointer, grant_key,
    GRANT_POLICY, delivery, validate_capability)
from subnet.complete_group_codec import limits
from subnet import backend_jobs, committed_training_inputs


class GrantHTTP:
    """requests-like GET for grant objects (stream + iter_content)."""
    def __init__(self, bucket):
        self.bucket = bucket; self.gets = []; self.fail = 0; self.tamper = None
    def get(self, url, *, timeout, stream):
        assert stream is True and 0 < timeout <= 120
        if self.fail:
            self.fail -= 1; raise requests.ConnectionError('mock transient')
        key = unquote(urlparse(url).path).removeprefix('/candidate/'); self.gets.append(key)
        data, _ = self.bucket.objects[key]
        if self.tamper: data = self.tamper(data)
        return SimpleNamespace(raise_for_status=lambda: None, iter_content=lambda n: iter([data]))


def grant_manifest(n=8, cap=3):
    m = fixture(n); m['max_batches'] = cap; m['capability_delivery'] = {'version': GRANT_POLICY}; return m


class _Harness(unittest.TestCase):
    setUp = base.TransportTests.setUp
    manifest = base.TransportTests.manifest
    actor = base.TransportTests.actor
    search = base.TransportTests.search
    resolve = base.TransportTests.resolve
    capture = base.TransportTests.capture


class UploadGrantTests(_Harness):
    def gmanifest(self, cap=3):
        m = grant_manifest(cap=cap)
        m['capabilities'] = {self.identity.id: capability(self.bucket, m, self.identity.id, registered_miners={self.identity.id})}
        return m

    def test_pointer_is_small_constant_and_grant_holds_full_v1_capability(self):
        sizes = {}
        for cap in (3, 9, 125):
            m = self.gmanifest(cap); sizes[cap] = len(m['capabilities'][self.identity.id])
            got = resolve_capability(self.identity, m, http=GrantHTTP(self.bucket))
            validate_capability(got, m, self.identity.id)
            self.assertEqual(len(got['batch_put_urls']), cap); self.assertEqual(len(got['training_get_urls']), cap)
        self.assertLess(max(sizes.values()) - min(sizes.values()), 64)
        self.assertLess(max(sizes.values()), 2500)
        inline = fixture(); inline['max_batches'] = 125
        self.assertGreater(len(capability(self.bucket, inline, self.identity.id, registered_miners={self.identity.id})), 100 * sizes[125])

    def test_grant_object_is_content_addressed_private_and_sealed(self):
        m = self.gmanifest(); pointer = self.identity.decrypt(m['capabilities'][self.identity.id])
        key = grant_key(m, self.identity.id, pointer['grant_sha256'])
        self.assertTrue(key.startswith('private/' + m['epoch'] + '/grants/' + self.identity.id + '/'))
        blob, _ = self.bucket.objects[key]
        self.assertEqual(hashlib.sha256(blob).hexdigest(), pointer['grant_sha256'])
        self.assertNotIn(b'X-Amz', blob)  # sealed box, not plaintext URLs
        with self.assertRaises(Exception): self.other.decrypt(blob.decode())

    def test_tampered_truncated_or_oversized_grant_rejected(self):
        m = self.gmanifest()
        for tamper in (lambda d: d[:-1] + bytes([d[-1] ^ 1]), lambda d: d[:-4], lambda d: d + b'x'):
            http = GrantHTTP(self.bucket); http.tamper = tamper
            with self.assertRaisesRegex(ValueError, 'grant'): resolve_capability(self.identity, m, http=http)

    def test_transient_get_failures_retry_then_fail_closed(self):
        m = self.gmanifest(); http = GrantHTTP(self.bucket); http.fail = 2
        self.assertIn('put_url', resolve_capability(self.identity, m, http=http, sleep=lambda s: None))
        http.fail = 9
        with self.assertRaisesRegex(RuntimeError, 'grant unavailable'):
            resolve_capability(self.identity, m, http=http, sleep=lambda s: None)

    def test_pointer_binding_identity_epoch_deadline_and_path(self):
        m = self.gmanifest(); p = self.identity.decrypt(m['capabilities'][self.identity.id])
        validate_pointer(p, m, self.identity.id)
        with self.assertRaises(ValueError): validate_pointer(p, m, self.other.id)
        for field, value in (('deadline', p['deadline'] + 1), ('grant_bytes', 0), ('grant_bytes', 9 * 1024**2),
                             ('transport', 'complete-group-commitment-v1'), ('grant_sha256', '0' * 64),
                             ('grant_url', p['grant_url'].replace('/grants/', '/staging/'))):
            bad = dict(p, **{field: value})
            with self.assertRaises(ValueError): validate_pointer(bad, m, self.identity.id)
        bad = dict(p); bad['extra'] = 1
        with self.assertRaises(ValueError): validate_pointer(bad, m, self.identity.id)
        other_epoch = deepcopy(m); other_epoch['epoch'] = m['epoch'] + 'x'
        with self.assertRaises(ValueError): validate_pointer(p, other_epoch, self.identity.id)

    def test_inline_delivery_unchanged_and_unknown_delivery_rejected(self):
        m = self.manifest(); self.assertIsNone(delivery(m))
        cap = resolve_capability(self.identity, m, http=None)
        validate_capability(cap, m, self.identity.id)
        for value in ({'version': 'other'}, {'version': GRANT_POLICY, 'x': 1}, 'r2-upload-grant-v1'):
            bad = deepcopy(m); bad['capability_delivery'] = value
            with self.assertRaises(ValueError): delivery(bad)

    def test_retry_of_same_grant_bytes_is_idempotent_and_collision_fails(self):
        from subnet.complete_group_transport import _put_immutable
        _put_immutable(self.bucket, 'private/e/grants/x.json', b'abc')
        _put_immutable(self.bucket, 'private/e/grants/x.json', b'abc')
        with self.assertRaisesRegex(ValueError, 'collision'): _put_immutable(self.bucket, 'private/e/grants/x.json', b'abd')

    def test_real_miner_end_to_end_with_grant_delivery_and_cap_125(self):
        m = self.gmanifest(cap=125); http = GrantHTTP(self.bucket)
        with patch('subnet.complete_group_transport.requests', http):
            a = self.actor(m, self.state)
        self.assertEqual(len(a.cap['batch_put_urls']), 125); self.assertEqual(len(http.gets), 1)
        for index in range(4):
            self.search(a, index); self.assertEqual(a.upload(), 200)
        self.assertEqual(self.capture(m)['new_groups'], 4)
        self.assertEqual(limits(m).commitment_bytes, 4096 + 125 * 1024)

    def test_cap_bounds_and_job_envelope_scopes(self):
        m = fixture(); m['max_batches'] = 125; limits(m)
        m['max_batches'] = 257
        with self.assertRaises(ValueError): limits(m)
        self.assertEqual(backend_jobs.SIGNED_ENVELOPE_GROUP_CAPS, frozenset((9, 125)))


class ReleasePolicyCapTests(unittest.TestCase):
    def setUp(self):
        from test_mimo_public_worker import PublicWorkerTests
        self.f = PublicWorkerTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)

    def policy_for(self, **changes):
        from test_mimo_public_worker import QUALIFICATION_VERSION
        from subnet.mimo_release_policy import build_policy, policy_scope, digest, sign, QUALIFICATIONS
        m = deepcopy(self.f.m); m.pop('mimo_release_policy', None); m.update(changes)
        scope = digest(policy_scope(m, self.f.assets))
        q = [sign(self.f.operator, dict(version=QUALIFICATION_VERSION, scope_sha256=scope, kind=k, passed=True,
                  evidence_sha256='a' * 64)) for k in sorted(QUALIFICATIONS)]
        return m, build_policy(self.f.operator, m, task_assets=self.f.assets, mode='public', enabled=True, qualifications=q)

    def test_signed_scope_pins_cap_125_and_grant_delivery(self):
        from subnet.mimo_release_policy import ReleasePolicy, POLICY_KEY
        m, envelope = self.policy_for(max_batches=125, capability_delivery={'version': GRANT_POLICY})
        policy = ReleasePolicy(envelope, self.f.operator.id, allow_public=True)
        m[POLICY_KEY] = envelope; policy.validate_manifest(m)
        for field, value in (('max_batches', 9), ('capability_delivery', None)):
            bad = deepcopy(m)
            if value is None: bad.pop(field)
            else: bad[field] = value
            with self.assertRaisesRegex(ValueError, 'differs from qualified'): policy.validate_manifest(bad)

    def test_out_of_bound_cap_or_unknown_delivery_cannot_be_qualified(self):
        from subnet.mimo_release_policy import ReleasePolicy
        for changes in (dict(max_batches=0), dict(max_batches=257), dict(max_batches=True),
                        dict(capability_delivery={'version': 'x'})):
            with self.assertRaises(ValueError): self.policy_for(**changes)


if __name__ == '__main__':
    unittest.main()


class ParallelMintTests(unittest.TestCase):
    """Real subprocess workers + real botocore presigning (fake credentials, no network)."""
    def test_parallel_mint_matches_serial_contract_for_many_identities(self):
        import subnet.complete_group_transport as t
        ids = [Identity(bytes([i]) * 32) for i in range(70)]
        bucket = base.FakeBucket()
        bucket._client_options = dict(endpoint_url='https://test.r2.cloudflarestorage.com', region_name='auto',
                                      aws_access_key_id='FAKE', aws_secret_access_key='FAKE')
        bucket.name = 'candidate'
        m = grant_manifest(cap=125); reg = {i.id for i in ids}
        with patch.object(t, 'MINT_WORKERS', 4):
            caps = {i.id: capability(bucket, m, i.id, registered_miners=reg) for i in ids}
        self.assertEqual(len([k for k in bucket.objects if '/grants/' in k]), 70)
        m['capabilities'] = caps
        for i in (ids[0], ids[33], ids[69]):
            got = resolve_capability(i, m, http=GrantHTTP(bucket))
            self.assertEqual(len(got['batch_put_urls']), 125)
            self.assertIn('/staging/' + i.id + '/124.zip', unquote(got['batch_put_urls'][124]))
        self.assertEqual(len(t._GRANT_CACHE), 1)
