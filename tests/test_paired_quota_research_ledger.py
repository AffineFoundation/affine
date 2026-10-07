import copy
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from ops.paired_quota_qualification import ApprovedTask, digest, select_pairs
from ops.paired_quota_research_ledger import ResearchTrainingLedger, RecoveryRequired


def receipt(binding):
    return dict(job_id=binding['job_id'], binding_digest=digest(binding),
                parent_state_sha256=binding['parent_state_sha256'],
                step_before=binding['step_before'], step_after=binding['step_before'] + 1,
                output_state_sha256='e' * 64, output_checkpoint_sha256='f' * 64)


def crash_after_claim(path, counter, after_update):
    ledger = ResearchTrainingLedger(path, enabled=True)
    def execute(binding):
        if after_update:
            Path(counter).write_text('1')
        os._exit(23)
    ledger.execute_once('job-A', execute, lambda r, b: True)


class DurableResearchLedgerControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'owned.sqlite'
        self.ledger = ResearchTrainingLedger.create(self.path, enabled=True)
        task = ApprovedTask('epoch40', 'cp', 'a' * 64, 'math', 10,
                            'b' * 64, 'c' * 64, 'd' * 64, (0, 1, 2, 3))
        rolls = [dict(task.task_binding(), epoch=task.epoch, harness_sha256=task.harness_sha256,
                      sampling_context_sha256=task.sampling_context_sha256, attempt=i,
                      classification='positive' if i < 2 else 'negative',
                      turns=[dict(prompt=[10], output=[100 + i], actions=[], observations=[])])
                 for i in range(4)]
        self.revision = select_pairs(task, 'miner-A', rolls, quota=2)
        self.other_miner_revision = select_pairs(task, 'miner-B', rolls, quota=2)
        self.task = task
        self.rolls = rolls
        self.arguments = dict(parent_state_sha256='a' * 64, step_before=29,
                              settings_sha256='b' * 64, revisions=[self.revision])

    def reserve(self):
        return self.ledger.reserve('job-A', **self.arguments)

    def test_explicit_opt_in_and_owned_schema_only(self):
        with self.assertRaises(ValueError):
            ResearchTrainingLedger(self.path)
        with self.assertRaises(FileExistsError):
            ResearchTrainingLedger.create(self.path, enabled=True)
        unowned = Path(self.directory.name) / 'other.sqlite'
        sqlite3.connect(unowned).close()
        with self.assertRaisesRegex(ValueError, 'owned'):
            ResearchTrainingLedger(unowned, enabled=True)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_reopen_reserved_binding_and_complete_redelivery_no_second_step(self):
        initial = self.reserve()
        ledger = ResearchTrainingLedger(self.path, enabled=True)
        self.assertEqual(initial, ledger.reserve('job-A', **self.arguments))
        calls = []
        actual = ledger.execute_once('job-A', lambda b: calls.append(b) or receipt(b), lambda r, b: True)
        reopened = ResearchTrainingLedger(self.path, enabled=True)
        duplicate = reopened.execute_once('job-A', lambda b: self.fail('second optimizer call'), lambda r, b: True)
        self.assertEqual(actual, duplicate)
        self.assertEqual(len(calls), 1)
        self.assertEqual(reopened.reserve('job-A', **self.arguments)['status'], 'complete')

    def test_same_slot_cannot_be_moved_to_new_job(self):
        self.reserve()
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.reserve('job-B', **self.arguments)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            self.ledger.inspect('job-B')

    def test_cross_uid_copied_members_not_second_training_contribution(self):
        self.reserve()
        args = dict(self.arguments, revisions=[self.other_miner_revision])
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.reserve('job-B', **args)
        with sqlite3.connect(self.path) as database:
            self.assertEqual(database.execute('SELECT COUNT(*) FROM selections').fetchone()[0], 1)
            self.assertEqual(database.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], 1)

    def test_new_epoch_does_not_make_same_checkpoint_execution_new(self):
        self.reserve()
        task = ApprovedTask('epoch41', self.task.checkpoint,
                            self.task.taskset_sha256, self.task.env_id,
                            self.task.index, self.task.task_sha256,
                            self.task.harness_sha256,
                            self.task.sampling_context_sha256,
                            self.task.approved_attempts)
        rolls = copy.deepcopy(self.rolls)
        for row in rolls:
            row['epoch'] = task.epoch
        revision = select_pairs(task, 'miner-A', rolls, quota=2)
        self.assertNotEqual(revision['slot_id'], self.revision['slot_id'])
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.reserve('job-B', **dict(self.arguments, revisions=[revision]))
        with self.assertRaisesRegex(ValueError, 'unknown'):
            self.ledger.inspect('job-B')

    def test_new_checkpoint_identity_does_not_globally_ban_same_answer(self):
        # Identity admission alone does not prove generation by the new model.
        # That remains the inference audit's responsibility.
        self.reserve()
        task = ApprovedTask('epoch41', 'new-checkpoint',
                            self.task.taskset_sha256, self.task.env_id,
                            self.task.index, self.task.task_sha256,
                            self.task.harness_sha256,
                            self.task.sampling_context_sha256,
                            self.task.approved_attempts)
        rolls = copy.deepcopy(self.rolls)
        for row in rolls:
            row.update(epoch=task.epoch, checkpoint=task.checkpoint)
        revision = select_pairs(task, 'miner-A', rolls, quota=2)
        result = self.ledger.reserve('job-B', **dict(
            self.arguments, parent_state_sha256='c' * 64,
            step_before=30, revisions=[revision]))
        self.assertEqual(result['status'], 'reserved')
        with sqlite3.connect(self.path) as database:
            self.assertEqual(database.execute('SELECT COUNT(*) FROM members').fetchone()[0], 8)

    def test_multi_slot_reservation_rolls_back_on_later_conflict(self):
        self.reserve()
        fresh = copy.deepcopy(self.revision)
        fresh['slot_id'] = '9' * 64
        for index, pair in enumerate(fresh['pairs']):
            for side in ('positive', 'negative'):
                pair[side]['execution_id'] = digest(['fresh-execution', index, side])
                pair[side]['content_id'] = digest(['fresh-content', index, side])
        fresh['revision_id'] = digest({k: v for k, v in fresh.items() if k not in ('revision_id', 'duplicate_content_count')})
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.reserve('job-B', **dict(self.arguments, revisions=[fresh, self.revision]))
        with sqlite3.connect(self.path) as database:
            self.assertEqual(database.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], 1)
            self.assertEqual(database.execute('SELECT COUNT(*) FROM selections').fetchone()[0], 1)
            self.assertEqual(database.execute('SELECT COUNT(*) FROM members').fetchone()[0], 4)
        # Rollback left the fresh selection available for an independent job.
        self.assertEqual(self.ledger.reserve('job-C', **dict(self.arguments, revisions=[fresh]))['status'], 'reserved')

    def test_revision_reordering_repacking_does_not_change_job_binding(self):
        original = self.reserve()
        changed = copy.deepcopy(self.revision); changed['duplicate_content_count'] = 10
        self.assertEqual(original, self.ledger.reserve('job-A', **dict(self.arguments, revisions=[changed])))
        # A selected pair ordering change is an amended selected revision, not redelivery.
        changed['pairs'].reverse()
        changed['revision_id'] = digest({k: v for k, v in changed.items() if k not in ('revision_id', 'duplicate_content_count')})
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            self.ledger.reserve('job-A', **dict(self.arguments, revisions=[changed]))

    def test_changed_parent_step_settings_refused(self):
        self.reserve()
        for key, value in [('parent_state_sha256', 'c' * 64), ('step_before', 30), ('settings_sha256', 'c' * 64)]:
            with self.assertRaisesRegex(ValueError, 'binding changed'):
                self.ledger.reserve('job-A', **dict(self.arguments, **{key: value}))

    def test_same_member_cannot_fill_second_pair(self):
        changed = copy.deepcopy(self.revision)
        changed['pairs'][1]['positive'] = changed['pairs'][0]['positive']
        changed['revision_id'] = digest({k: v for k, v in changed.items() if k not in ('revision_id', 'duplicate_content_count')})
        with self.assertRaisesRegex(ValueError, 'member reuse'):
            self.ledger.reserve('job-A', **dict(self.arguments, revisions=[changed]))

    def test_exception_after_claim_remains_uncertain_without_auto_retry(self):
        self.reserve()
        def fail(binding):
            raise RuntimeError('transport failed after possible update')
        with self.assertRaises(RuntimeError):
            self.ledger.execute_once('job-A', fail, lambda r, b: True)
        reopened = ResearchTrainingLedger(self.path, enabled=True)
        self.assertEqual(reopened.inspect('job-A')['status'], 'uncertain')
        with self.assertRaises(RecoveryRequired):
            reopened.execute_once('job-A', lambda b: self.fail('must not reapply'), lambda r, b: True)

    def test_actual_process_crash_after_update_blocks_reapplication(self):
        self.reserve()
        counter = Path(self.directory.name) / 'fake-optimizer-step'
        context = multiprocessing.get_context('fork')
        process = context.Process(target=crash_after_claim, args=(str(self.path), str(counter), True))
        process.start(); process.join(timeout=10)
        self.assertFalse(process.is_alive()); self.assertEqual(process.exitcode, 23)
        reopened = ResearchTrainingLedger(self.path, enabled=True)
        self.assertEqual(counter.read_text(), '1')
        self.assertEqual(reopened.inspect('job-A')['status'], 'executing')
        with self.assertRaises(RecoveryRequired):
            reopened.execute_once('job-A', lambda b: counter.write_text('2'), lambda r, b: True)
        binding = reopened.inspect('job-A')['binding']
        original = receipt(binding)
        self.assertEqual(reopened.recover_complete('job-A', original, lambda r, b: True), original)
        self.assertEqual(reopened.execute_once('job-A', lambda b: self.fail('second step'), None), original)
        self.assertEqual(counter.read_text(), '1')

    def test_actual_process_crash_before_update_also_requires_recovery(self):
        self.reserve()
        counter = Path(self.directory.name) / 'never-updated'
        process = multiprocessing.get_context('fork').Process(target=crash_after_claim,
                            args=(str(self.path), str(counter), False))
        process.start(); process.join(timeout=10)
        self.assertFalse(process.is_alive()); self.assertEqual(process.exitcode, 23)
        self.assertFalse(counter.exists())
        reopened = ResearchTrainingLedger(self.path, enabled=True)
        with self.assertRaises(RecoveryRequired):
            reopened.execute_once('job-A', lambda b: self.fail('no automatic retry'), lambda r, b: True)
        # The ledger cannot distinguish before-update from after-update crash.
        self.assertEqual(reopened.inspect('job-A')['status'], 'executing')

    def test_concurrent_claims_only_one_executor_call(self):
        self.reserve(); barrier = threading.Barrier(2); results = []; calls = []
        def work():
            ledger = ResearchTrainingLedger(self.path, enabled=True)
            barrier.wait()
            try:
                results.append(ledger.execute_once('job-A', lambda b: calls.append(1) or receipt(b), lambda r, b: True))
            except RecoveryRequired:
                results.append('recovery-required')
        threads = [threading.Thread(target=work) for _ in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=10)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(len(calls), 1); self.assertEqual(len(results), 2)
        self.assertEqual(self.ledger.inspect('job-A')['status'], 'complete')

    def test_invalid_receipt_cannot_complete_or_make_execution_retriable(self):
        for mutation in ('parent_state_sha256', 'step_after', 'job_id', 'binding_digest'):
            # New independent ledger per adversarial case.
            path = Path(self.directory.name) / (mutation + '.sqlite')
            ledger = ResearchTrainingLedger.create(path, enabled=True)
            ledger.reserve('job-A', **self.arguments)
            def execute(binding):
                value = receipt(binding)
                value[mutation] = 999 if mutation == 'step_after' else 'c' * 64
                return value
            with self.assertRaises(ValueError):
                ledger.execute_once('job-A', execute, lambda r, b: True)
            self.assertEqual(ledger.inspect('job-A')['status'], 'uncertain')
        self.reserve()
        with self.assertRaisesRegex(ValueError, 'authenticated'):
            self.ledger.execute_once('job-A', receipt, lambda r, b: False)
        self.assertEqual(self.ledger.inspect('job-A')['status'], 'uncertain')

    def test_reserved_cannot_be_marked_complete_and_completed_receipt_immutable(self):
        self.reserve(); binding = self.ledger.inspect('job-A')['binding']
        original = receipt(binding)
        with self.assertRaisesRegex(ValueError, 'not started'):
            self.ledger.recover_complete('job-A', original, lambda r, b: True)
        self.ledger.execute_once('job-A', receipt, lambda r, b: True)
        changed = dict(original, output_state_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            self.ledger.recover_complete('job-A', changed, lambda r, b: True)


if __name__ == '__main__':
    unittest.main()
