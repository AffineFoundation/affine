"""Synthetic active-v2 transport controls: no real identity, network or model."""
import fcntl
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nacl.signing import SigningKey
from subnet import forced_sampling as forced
from subnet.miner import Miner
from subnet.commitment_transport import validate as validate_commitment
from subnet.training_documents import validate as validate_training
import test_miner_search_progress as fixtures


class StateTransaction(unittest.TestCase):
    roll = fixtures.ProgressTests.roll
    generate = fixtures.ProgressTests.generate
    unfinished = fixtures.ProgressTests.unfinished

    def setUp(self):
        fixtures.ProgressTests.setUp(self)
        key = SigningKey.generate()
        self.identity = SimpleNamespace(key=key, id=key.verify_key.encode().hex())
        self.manifest.update(max_batches=9, indices=list(range(10)),
                             transport_policy='direct-r2-v1',
                             submission_transport_policy='small-commitment-pairs-v2')
        self.context = forced.binding(self.manifest, self.identity.id)
        self.kind = lambda n: 'positive' if n < 4 else 'negative'
        prefix = 'https://' + 'a'*32 + '.r2.cloudflarestorage.com/bucket/'
        query = '?X-Amz-Signature=synthetic&X-Amz-Algorithm=AWS4-HMAC-SHA256'
        self.cap = dict(transport='small-commitment-pairs-v2', put_url=prefix+'commit'+query,
                        batch_put_urls=[prefix+'proof/'+str(i)+query for i in range(9)],
                        training_put_urls=[prefix+'training/'+str(i)+query for i in range(9)])
        self.uploads = []
        self.put = self.enterContext(patch('subnet.miner.requests.put', side_effect=self.record_put))

    def record_put(self, url, **kwargs):
        self.uploads.append((url, kwargs['data']))
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None)

    def miner(self):
        miner = Miner(self.identity, self.manifest, 'unused', capability=self.cap, state_path=self.state)
        self.addCleanup(miner.close)
        return miner

    def fd_target(self, descriptor):
        opened = os.fstat(descriptor)
        for path in (self.state, self.state.with_name(self.state.name+'.pairs'), self.state.parent):
            try:
                expected = path.stat()
            except FileNotFoundError:
                continue
            if (opened.st_dev, opened.st_ino) == (expected.st_dev, expected.st_ino):
                return str(path)
        return None

    def test_stale_a_b_refuses_overwrite_then_recovers_b_without_generation(self):
        a, b = self.miner(), self.miner()
        a.search(0, max_attempts=8)
        b.search(1, max_attempts=8)
        with patch('subnet.miner.requests.put', side_effect=RuntimeError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                a.upload()
        durable_a = self.state.read_bytes()
        with self.assertRaisesRegex(ValueError, 'state changed; reopen'):
            b.upload()
        self.put.assert_not_called()
        self.assertEqual(self.state.read_bytes(), durable_a)
        self.assertTrue(b.search_state.load('math', 0)[1])
        self.assertFalse(b.search_state.load('math', 1)[1])
        self.assertEqual(len(b.search_state.load('math', 1)[2]), 8)
        b.close(); a.close()
        restored = self.miner()
        with patch.object(self.runtime, 'rollout', side_effect=AssertionError('no new generation')):
            restored.search(1, max_attempts=1)
        restored.upload()
        self.assertEqual([batch['index'] for batch, _ in self.miner().batches], [0, 1])
        self.assertEqual(len(self.calls), 16)

    def test_stale_client_search_refused_before_nonce_consumption(self):
        a, b = self.miner(), self.miner()
        a.search(0, max_attempts=8); a.upload()
        with self.assertRaisesRegex(ValueError, 'state changed; reopen'):
            b.search(1, max_attempts=8)
        self.assertEqual(b.search_state.load('math', 1), (0, False, [], []))

    def test_lock_covers_all_puts_through_final_commitment(self):
        miner = self.miner(); miner.search(0, max_attempts=8)
        descriptor = os.open(miner.search_state.path.with_name(miner.search_state.path.name+'.lock'), os.O_RDWR)
        self.addCleanup(os.close, descriptor)
        def locked_put(url, **kwargs):
            with self.assertRaises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self.record_put(url, **kwargs)
        with patch('subnet.miner.requests.put', side_effect=locked_put):
            miner.upload()
        self.assertEqual(len(self.uploads), 3)
        self.assertEqual(self.uploads[-1][0], self.cap['put_url'])
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)

    def test_durability_order_before_partial_retirement(self):
        miner = self.miner(); miner.search(0, max_attempts=8)
        events = []; real_sync = os.fsync; real_complete = miner.search_state.completed
        def sync(fd):
            events.append(('dir' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file',
                           self.fd_target(fd)))
            real_sync(fd)
        def complete(batches):
            events.append(('retire', None)); real_complete(batches)
        with patch('os.fsync', side_effect=sync), patch.object(miner.search_state, 'completed', side_effect=complete):
            miner.upload()
        offset = events.index(('retire', None))
        self.assertEqual(events[offset-3:offset], [('file', str(self.state)),
            ('dir', str(self.state.with_name(self.state.name+'.pairs'))), ('dir', str(self.state.parent))])

    def test_each_durability_barrier_failure_keeps_partial_and_recovers(self):
        for failure in ('file', 'pairs_directory', 'parent_directory'):
            with self.subTest(failure=failure):
                self.state = self.root/(failure+'.state')
                miner = self.miner(); miner.search(0, max_attempts=8)
                real_sync = os.fsync
                def sync(fd):
                    target = self.fd_target(fd)
                    expected = {'file': str(self.state), 'pairs_directory': str(self.state)+'.pairs',
                                'parent_directory': str(self.state.parent)}[failure]
                    if target == expected:
                        raise OSError('interrupted durability barrier')
                    real_sync(fd)
                with patch('os.fsync', side_effect=sync):
                    with self.assertRaisesRegex(OSError, 'interrupted durability'):
                        miner.upload()
                self.assertFalse(miner.search_state.load('math', 0)[1])
                self.assertEqual(len(miner.search_state.load('math', 0)[2]), 8)
                self.put.assert_not_called()
                miner.close()
                restored = self.miner()
                self.assertEqual(len(restored.batches), 1)
                self.assertEqual(restored.search_state.load('math', 0)[1:], (True, [], []))

    def test_failed_barrier_can_retry_same_client(self):
        miner = self.miner(); miner.search(0, max_attempts=8)
        with patch.object(miner, '_sync_complete_state', side_effect=OSError('barrier')):
            with self.assertRaisesRegex(OSError, 'barrier'): miner.upload()
        self.assertFalse(miner.search_state.load('math', 0)[1])
        miner.upload()
        self.assertTrue(miner.search_state.load('math', 0)[1])

    def test_interrupted_index_replace_preserves_prior_state_and_new_partial(self):
        miner = self.miner(); miner.search(0, max_attempts=8); miner.upload()
        prior = self.state.read_bytes()
        miner.search(1, max_attempts=8)
        real_replace = Path.replace
        def replace(source, target):
            if Path(target) == self.state:
                raise OSError('interrupted before index replacement')
            return real_replace(source, target)
        self.put.reset_mock()
        with patch.object(Path, 'replace', replace):
            with self.assertRaisesRegex(OSError, 'interrupted before index'):
                miner.upload()
        self.assertEqual(self.state.read_bytes(), prior)
        self.assertFalse(miner.search_state.load('math', 1)[1])
        self.assertEqual(len(miner.search_state.load('math', 1)[2]), 8)
        self.put.assert_not_called(); miner.close()
        restored = self.miner()
        with patch.object(self.runtime, 'rollout', side_effect=AssertionError('no regeneration')):
            restored.search(1, max_attempts=1)
        # A restarted process uses another temporary filename. Model this one
        # detail without killing the test runner or deleting the crash artifact.
        with patch('os.getpid', return_value=os.getpid()+1000000):
            restored.upload()
        self.assertEqual([b['index'] for b, _ in self.miner().batches], [0, 1])
        self.assertEqual(len(self.calls), 16)

    def test_restore_failed_barrier_does_not_retire_partials(self):
        miner = self.miner(); miner.search(0, max_attempts=8)
        with patch.object(miner, '_sync_complete_state', side_effect=OSError('barrier')):
            with self.assertRaises(OSError): miner.upload()
        with patch.object(Miner, '_sync_complete_state', side_effect=OSError('restore barrier')):
            with self.assertRaisesRegex(OSError, 'restore barrier'): self.miner()
        self.assertFalse(miner.search_state.load('math', 0)[1])
        self.assertEqual(len(miner.search_state.load('math', 0)[2]), 8)

    def test_v2_partial_resume_cap9_training_framing_and_upload_reuse(self):
        miner = self.miner(); self.unfinished(miner, budget=2); miner.close()
        miner = self.miner(); miner.search(0, max_attempts=6)
        self.assertEqual(self.calls, list(range(8)))
        for task in range(1, 9): miner.search(task, max_attempts=8)
        with self.assertRaisesRegex(ValueError, 'owned commitment slot cap'):
            miner.search(9, max_attempts=8)
        self.assertEqual(len(miner.batches), 9)
        miner.upload()
        self.assertEqual(len(self.uploads), 19)
        envelope = validate_commitment(self.uploads[-1][1], self.manifest['epoch'], self.identity.id, 9)
        for entry in envelope['payload']['batches']:
            body = dict(self.uploads)[self.cap['training_put_urls'][entry['slot']]]
            validate_training(body, self.manifest['epoch'], self.manifest['checkpoint']['id'], self.identity.id, entry)
        miner.close(); restored = self.miner(); self.uploads.clear(); restored.upload()
        self.assertEqual([url for url, _ in self.uploads], [self.cap['put_url']])

    def test_legacy_zip_file_and_parent_fenced_before_retirement(self):
        self.manifest.pop('transport_policy'); self.manifest.pop('submission_transport_policy')
        self.context = forced.binding(self.manifest, self.identity.id)
        miner = self.miner(); miner.search(0, max_attempts=8)
        real_sync = os.fsync
        def sync(fd):
            if self.fd_target(fd) == str(self.state.parent):
                raise OSError('legacy parent barrier')
            real_sync(fd)
        with patch('os.fsync', side_effect=sync):
            with self.assertRaisesRegex(OSError, 'legacy parent barrier'): miner.upload()
        self.assertFalse(miner.search_state.load('math', 0)[1])
        self.assertEqual(len(miner.search_state.load('math', 0)[2]), 8)
        self.put.assert_not_called()
        miner.close(); restored = self.miner()
        self.assertTrue(restored.search_state.load('math', 0)[1])


class TokenTransportCompatibility(unittest.TestCase):
    def setUp(self):
        import test_token_only_production_contract as token_fixture
        from subnet.token_only_protocol import TRANSPORT
        case = token_fixture.TokenBackendControls(); case.setUp()
        self.manifest = dict(case.m, deadline=time.time()+3600, max_batches=1,
                             source_bundle={'sha256': 'b'*64}, transport_policy='direct-r2-v1')
        self.batch = case.batch
        key = SigningKey.generate()
        self.identity = SimpleNamespace(key=key, id=key.verify_key.encode().hex())
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)/'complete.state'
        prefix = 'https://'+'a'*32+'.r2.cloudflarestorage.com/bucket/'
        query = '?X-Amz-Signature=synthetic&X-Amz-Algorithm=AWS4-HMAC-SHA256'
        self.cap = dict(transport=TRANSPORT, put_url=prefix+'commit'+query,
                        batch_put_urls=[prefix+'pair'+query], training_put_urls=[prefix+'training'+query])
        self.enterContext(patch('subnet.miner.check_runtime_profile'))
        self.uploads = []
        def put(url, **kwargs):
            self.uploads.append((url, kwargs['data']))
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None)
        self.put = self.enterContext(patch('subnet.miner.requests.put', side_effect=put))

    def miner(self):
        miner = Miner(self.identity, self.manifest, 'unused', capability=self.cap, state_path=self.state)
        self.addCleanup(miner.close)
        return miner

    def test_v3_upload_and_retry_preserve_token_document_framing(self):
        from subnet.token_only_protocol import TRANSPORT
        from subnet.training_documents import TOKEN_VERSION
        miner = self.miner(); miner.batches = [(self.batch, [[], []])]; miner.upload()
        self.assertEqual([url for url, _ in self.uploads],
                         [self.cap['batch_put_urls'][0], self.cap['training_put_urls'][0], self.cap['put_url']])
        claim = validate_commitment(self.uploads[-1][1], self.manifest['epoch'], self.identity.id, 1)['payload']
        self.assertEqual(claim['version'], TRANSPORT)
        document = validate_training(self.uploads[1][1], self.manifest['epoch'],
                                     self.manifest['checkpoint']['id'], self.identity.id,
                                     claim['batches'][0], transport=TRANSPORT)
        self.assertEqual(document['version'], TOKEN_VERSION)
        self.uploads.clear(); miner.upload()
        self.assertEqual([url for url, _ in self.uploads], [self.cap['put_url']])

    def test_v3_still_requires_bound_training_upload_slots(self):
        self.cap['training_put_urls'] = []
        with self.assertRaisesRegex(ValueError, 'bound token upload slots'):
            self.miner()
        self.put.assert_not_called()


if __name__ == '__main__':
    unittest.main()
