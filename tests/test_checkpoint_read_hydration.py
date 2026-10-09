import base64
import copy
import hashlib
import tempfile
import unittest
from unittest.mock import Mock
from pathlib import Path

from nacl.signing import SigningKey

from ops import checkpoint_read_hydration as hydration


class CheckpointReadHydrationTests(unittest.TestCase):
    def setUp(self):
        self.operator = SigningKey.generate()
        self.authority = self.operator.verify_key.encode().hex()
        self.files = {'config.json': 'a' * 64, 'model.safetensors': 'b' * 64}
        checkpoint = hashlib.sha256(hydration.canonical(self.files)).hexdigest()
        descriptor = self.sign({'id': checkpoint, 'files': self.files})
        self.plan = dict(kind='immutable-checkpoint-read-hydration-v1', role='mine',
                         retained_UUID='test-node', helper_sha256=hydration.sha(hydration.__file__),
                         GPU_runs=0, optimizer_runs=0, chain_transactions=0, publication_writes=0,
                         created_at=10, expires_at=110, checkpoint=checkpoint,
                         checkpoint_authority=self.authority, checkpoint_descriptor=descriptor,
                         checkpoint_descriptor_sha256=hashlib.sha256(hydration.canonical(descriptor)).hexdigest(),
                         objects={n: {'sha256': h, 'bytes': 2} for n, h in self.files.items()},
                         read_urls={n: self.url(checkpoint, n) for n in self.files},
                         destination='/root/test/checkpoints/' + checkpoint,
                         allows_concurrent_scientific_reads=True)

    def sign(self, payload):
        return dict(payload=payload, signer=self.authority,
                    signature=base64.b64encode(self.operator.sign(hydration.canonical(payload)).signature).decode())

    @staticmethod
    def url(checkpoint, name):
        return ('https://example.r2.cloudflarestorage.com/bucket/public/checkpoints/'
                + checkpoint + '/' + name
                + '?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=' + 'c' * 64)

    def validate(self, payload):
        return hydration.validate(self.sign(payload), self.authority, 'mine', 'test-node', now=20)

    def test_arbitrary_pinned_successor_and_shard_count_are_supported(self):
        plan, descriptor = self.validate(self.plan)
        self.assertEqual(descriptor['files'], self.files)
        self.assertEqual(plan['checkpoint'], descriptor['id'])
        self.assertNotIn('all_scientific_roles_idle', plan)
        self.assertNotIn('next_opening_held', plan)

    def test_invalid_actor_deadline_or_write_scope_is_refused(self):
        for change in ({'role': 'train'}, {'retained_UUID': 'other-node'},
                       {'helper_sha256': '0' * 64}, {'expires_at': 20},
                       {'created_at': 21}, {'expires_at': 10000}, {'GPU_runs': True},
                       {'optimizer_runs': 1}, {'chain_transactions': 1},
                       {'publication_writes': 1}, {'allows_concurrent_scientific_reads': False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate({**self.plan, **change})

    def test_modified_envelope_and_descriptor_inventory_are_refused(self):
        envelope = self.sign(copy.deepcopy(self.plan))
        envelope['payload']['checkpoint'] = '0' * 64
        with self.assertRaises(Exception):
            hydration.validate(envelope, self.authority, 'mine', 'test-node', now=20)
        for change in ({'checkpoint': '0' * 64}, {'objects': {}}, {'read_urls': {}},
                       {'checkpoint_descriptor_sha256': '0' * 64},
                       {'destination': '/root/../outside'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate({**self.plan, **change})
        descriptor = self.sign({'id': self.plan['checkpoint'], 'files': {**self.files, 'extra': 'd' * 64}})
        plan = {**self.plan, 'checkpoint_descriptor': descriptor,
                'checkpoint_descriptor_sha256': hashlib.sha256(hydration.canonical(descriptor)).hexdigest()}
        with self.assertRaises(ValueError):
            self.validate(plan)

    def test_capabilities_are_bound_to_exact_file_and_r2(self):
        checkpoint = self.plan['checkpoint']
        valid = self.url(checkpoint, 'config.json')
        for url in (valid.replace('https:', 'http:'),
                    valid.replace('example.r2.cloudflarestorage.com', 'example.com'),
                    valid.replace('/config.json', '/other.json'),
                    valid.replace(checkpoint, '0' * 64), valid + '#fragment',
                    valid.replace('https://', 'https://user:password@')):
            with self.subTest(url=url), self.assertRaises(ValueError):
                hydration.read_url(url, checkpoint, 'config.json')

    def test_existing_full_cache_requires_bytes_sizes_and_exact_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'config.json').write_bytes(b'{}')
            (root / 'model.safetensors').write_bytes(b'weights')
            files = {p.name: hydration.sha(p) for p in root.iterdir()}
            objects = {p.name: {'bytes': p.stat().st_size} for p in root.iterdir()}
            self.assertTrue(hydration.complete(root, files, objects))
            (root / 'extra').write_bytes(b'extra')
            self.assertFalse(hydration.complete(root, files, objects))
            (root / 'extra').unlink()
            (root / 'model.safetensors').write_bytes(b'changed')
            self.assertFalse(hydration.complete(root, files, objects))
            self.assertEqual((root / 'model.safetensors').read_bytes(), b'changed')

    def test_mismatching_complete_partial_is_preserved_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            partial = Path(temp) / 'weights.partial'
            partial.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'preserve mismatching'):
                hydration.download(None, 'unused', partial, 3, '0' * 64, 100, clock=lambda: 20)
            self.assertEqual(partial.read_bytes(), b'bad')

    def test_kernel_lock_prevents_duplicate_hydration_without_stale_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = hydration.acquire_lock(temp, self.plan['checkpoint'])
            try:
                with self.assertRaisesRegex(ValueError, 'already active'):
                    hydration.acquire_lock(temp, self.plan['checkpoint'])
            finally:
                lock.close()
            hydration.acquire_lock(temp, self.plan['checkpoint']).close()

    def transfer(self,body,*,partial=b'',failure_at=None,mutate=None,chunk=4,retries=3):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        p=Path(temp.name)/'checkpoint.partial'
        if partial:p.write_bytes(partial)
        calls=[]
        def get(url,**kw):
            self.assertEqual(kw['headers']['Connection'],'close')
            start,end=map(int,kw['headers']['Range'][6:].split('-'));calls.append((start,end))
            response=Mock(status_code=206,headers={'Content-Range':f'bytes {start}-{end}/{len(body)}','Content-Length':str(end-start+1)})
            response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
            def content(_):
                for i in range(start,end+1):
                    if len(calls)==1 and i==failure_at:raise hydration.requests.exceptions.ChunkedEncodingError('cut')
                    yield body[i:i+1]
            response.iter_content=content
            if mutate:mutate(response)
            return response
        session=Mock(get=get)
        return p,calls,lambda:hydration.download(session,'url',p,len(body),hashlib.sha256(body).hexdigest(),100,clock=lambda:20,chunk_bytes=chunk,max_retries=retries)

    def test_bounded_ranges_assemble_original_bytes_and_hash(self):
        p,calls,run=self.transfer(b'abcdefghij')
        result=run();self.assertEqual(calls,[(0,3),(4,7),(8,9)])
        self.assertEqual(p.read_bytes(),b'abcdefghij');self.assertEqual(result['downloaded_bytes'],10)

    def test_interrupted_range_resumes_from_written_bytes(self):
        p,calls,run=self.transfer(b'abcdefghij',failure_at=2)
        result=run();self.assertEqual(calls,[(0,3),(2,5),(6,9)])
        self.assertEqual(result['transfer_retries'],1);self.assertEqual(p.read_bytes(),b'abcdefghij')

    def test_existing_partial_is_retained_and_completed(self):
        p,calls,run=self.transfer(b'abcdefghij',partial=b'abc')
        result=run();self.assertEqual(calls,[(3,6),(7,9)])
        self.assertEqual(result['resumed_bytes'],3);self.assertEqual(p.read_bytes(),b'abcdefghij')

    def test_empty_interrupted_range_can_retry(self):
        p,calls,run=self.transfer(b'abc',failure_at=0)
        self.assertEqual(run()['transfer_retries'],1);self.assertEqual(p.read_bytes(),b'abc')

    def test_wrong_range_or_ignored_range_preserves_partial(self):
        for mutate in [lambda r:setattr(r,'status_code',200),lambda r:r.headers.update({'Content-Range':'bytes 0-3/10'})]:
            p,calls,run=self.transfer(b'abcdefghij',partial=b'ab',mutate=mutate)
            with self.assertRaises(ValueError):run()
            self.assertEqual(p.read_bytes(),b'ab');self.assertEqual(len(calls),1)

    def test_exhausted_retry_keeps_partial_and_no_fake_completion(self):
        p,calls,run=self.transfer(b'abcdefghij',failure_at=2,retries=0)
        with self.assertRaises(hydration.requests.RequestException):run()
        self.assertEqual(p.read_bytes(),b'ab')

    def test_corrupt_retained_prefix_never_passes_full_sha(self):
        p,calls,run=self.transfer(b'abcdefghij',partial=b'bad')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):run()
        self.assertEqual(p.read_bytes(),b'baddefghij')


if __name__ == '__main__':
    unittest.main()
