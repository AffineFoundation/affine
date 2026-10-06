import copy
import datetime
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import test_training_documents as document_tests
from subnet.commitment_transport import canonical, make, sha
from subnet.storage import Identity
from subnet.training_documents import capture, document
from subnet.token_capture_journal import CaptureJournal


class CaptureJournalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = document_tests.DocumentTests()
        self.fixture.setUp()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.gateway, self.state, self.puts, self.calls = self.fixture.gateway()
        self.gateway.state_path = Path(self.folder.name) / 'gateway.json'
        self.gateway.secret = b'private-test-journal-key'
        self.initial = copy.deepcopy(self.state)
        self.persisted = []
        self.gateway.persist = lambda: self.persisted.append(copy.deepcopy(self.state))

    def restart(self):
        self.state = copy.deepcopy(self.initial)
        self.gateway.epochs['token-v2'] = self.state
        self.gateway.persist = lambda: self.persisted.append(copy.deepcopy(self.state))

    def test_durable_receipt_replays_without_mutable_staging_read(self):
        capture(self.gateway, 'token-v2')
        expected = copy.deepcopy(self.state['training_document_snapshots'])
        self.assertEqual(len(self.persisted), 1)
        self.restart()
        self.gateway.bucket.client.get_object = lambda **kw: self.fail('mutable source reread')
        capture(self.gateway, 'token-v2')
        self.assertEqual(self.state['training_document_snapshots'], expected)

    def test_crash_after_receipt_fsync_before_gateway_projection(self):
        original = CaptureJournal.append
        def crash(owner, miner, receipt):
            original(owner, miner, receipt)
            raise RuntimeError('simulated process termination')
        with patch.object(CaptureJournal, 'append', crash):
            with self.assertRaises(RuntimeError):
                capture(self.gateway, 'token-v2')
        self.assertEqual(len(self.puts), 1)
        self.assertFalse(self.persisted)
        self.restart()
        self.gateway.bucket.client.get_object = lambda **kw: self.fail('durable receipt lost')
        capture(self.gateway, 'token-v2')
        self.assertEqual(len(self.state['training_document_snapshots']), 1)

    def test_interrupted_partial_tail_keeps_complete_receipts(self):
        capture(self.gateway, 'token-v2')
        path = CaptureJournal(self.gateway, 'token-v2', self.state).path
        complete = path.read_bytes()
        with path.open('ab') as stream:
            stream.write(b'{"unfinished":')
        self.restart()
        capture(self.gateway, 'token-v2')
        self.assertEqual(path.read_bytes(), complete)
        self.assertEqual(len(self.calls), 1)

    def test_tampered_receipt_and_changed_original_commitment_fail_closed(self):
        capture(self.gateway, 'token-v2')
        path = CaptureJournal(self.gateway, 'token-v2', self.state).path
        original = path.read_bytes()
        value = json.loads(original)
        value['body']['receipt']['sha256'] = 'f' * 64
        path.write_bytes(canonical(value) + b'\n')
        self.restart()
        with self.assertRaises(ValueError):
            capture(self.gateway, 'token-v2')
        path.write_bytes(original)
        child = next(iter(self.state['commitment_pending'].values()))['document']['payload']['batches'][0]
        child['training_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            capture(self.gateway, 'token-v2')

    def test_expired_capture_replays_durable_only_and_preserves_deferred(self):
        capture(self.gateway, 'token-v2')
        self.restart()
        # Original cutoff remains pinned; advancing the clock does not reopen it.
        cutoff = self.state['commitment_binding']['freeze_until']
        with patch('subnet.training_documents.time.time', return_value=cutoff + 1):
            capture(self.gateway, 'token-v2')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.state['commitment_binding']['freeze_until'], cutoff)
        self.assertFalse(self.state['training_document_deferred'])

    def test_expired_uncaptured_slot_is_deferred_without_network_or_reopening(self):
        cutoff = self.state['commitment_binding']['freeze_until']
        self.gateway.bucket.client.get_object = lambda **kw: self.fail('expired GET')
        with patch('subnet.training_documents.time.time', return_value=cutoff + 1):
            capture(self.gateway, 'token-v2')
        self.assertFalse(self.state['training_document_snapshots'])
        deferred = self.state['training_document_deferred'][self.fixture.identity.id]['0']
        self.assertEqual(deferred['reason'], 'infrastructure_deferred')
        self.assertEqual(self.state['commitment_binding']['freeze_until'], cutoff)
        self.assertFalse(self.state['rejections'])

    def test_journal_symlink_is_rejected(self):
        journal = CaptureJournal(self.gateway, 'token-v2', self.state)
        target = Path(self.folder.name) / 'external'
        target.write_bytes(b'')
        journal.path.symlink_to(target)
        with self.assertRaises(OSError):
            capture(self.gateway, 'token-v2')
        self.assertFalse(self.calls)

    def test_slot_fairness_and_only_terminal_whole_gateway_persist(self):
        fixtures = self.fixture
        documents = {}
        self.state['commitment_pending'] = {}
        now = time.time()
        for number in range(6):
            identity = Identity(bytes([number + 1]) * 32)
            packed = []
            for slot in range(3):
                batch = dict(fixtures.batch, index=number * 3 + slot)
                packed.append((batch, b'proof' + bytes([slot])))
                documents[identity.id, slot] = document(batch, fixtures.manifest, identity.id, slot)
            envelope = make(identity, fixtures.manifest, packed)
            self.state['commitment_pending'][identity.id] = dict(
                document=envelope, root='public/' + identity.id,
                sha256=sha(canonical(envelope)), size=len(canonical(envelope)), received_at=now)
        order = []
        def get(**kw):
            parts = kw['Key'].split('/')
            miner, slot = parts[-2], int(parts[-1].split('.')[0])
            order.append((miner, slot))
            return dict(Body=io.BytesIO(documents[miner, slot]),
                        LastModified=datetime.datetime.fromtimestamp(now, datetime.timezone.utc))
        self.gateway.bucket.client.get_object = get
        capture(self.gateway, 'token-v2')
        self.assertEqual([slot for _, slot in order], [0] * 6 + [1] * 6 + [2] * 6)
        self.assertEqual(len(self.persisted), 1)
        self.assertEqual(sum(len(v) for v in self.state['training_document_snapshots'].values()), 18)


if __name__ == '__main__':
    unittest.main()
