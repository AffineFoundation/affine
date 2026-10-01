import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from subnet.replay_commit import commit


class ReplayCommitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'ledger.json'
        self.target = 'a'*64
        self.metrics = {'checkpoint':'b'*64, 'replay_inputs_sha256':'c'*64,
            'replay_training':{'pool_sha256':'d'*64,
                'checks':[{'target_sha256':self.target}],
                'proposed_reuse_increments':{self.target:1}}}

    def test_retry_commits_once_and_rejects_changed_epoch_binding(self):
        first = commit(self.path, 'epoch', self.metrics)
        self.assertEqual(commit(self.path, 'epoch', self.metrics), first)
        self.assertEqual(first['counts'][self.target], 1)
        changed = copy.deepcopy(self.metrics); changed['checkpoint'] = 'e'*64
        with self.assertRaises(ValueError): commit(self.path, 'epoch', changed)
        self.assertEqual(json.loads(self.path.read_bytes()), first)

    def test_crash_before_replace_preserves_counts_and_retry_recovers(self):
        first = commit(self.path, 'first', self.metrics)
        with patch('subnet.replay_commit.os.replace', side_effect=OSError('simulated crash')):
            with self.assertRaises(OSError): commit(self.path, 'second', self.metrics)
        self.assertEqual(json.loads(self.path.read_bytes()), first)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        recovered = commit(self.path, 'second', self.metrics)
        self.assertEqual(recovered['counts'][self.target], 2)

    def test_unconsumed_target_and_negative_increment_rejected(self):
        bad = copy.deepcopy(self.metrics)
        bad['replay_training']['proposed_reuse_increments']['e'*64] = 1
        with self.assertRaises(ValueError): commit(self.path, 'epoch', bad)
        bad = copy.deepcopy(self.metrics)
        bad['replay_training']['proposed_reuse_increments'][self.target] = -1
        with self.assertRaises(ValueError): commit(self.path, 'epoch', bad)
        self.assertFalse(self.path.exists())
