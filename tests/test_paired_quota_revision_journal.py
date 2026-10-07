import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from nacl.signing import SigningKey
import test_paired_quota_batch_adapter as fixtures
from ops.paired_quota_revision_journal import ResearchRevisionJournal, boundary_receipt, scope
from ops.paired_quota_qualification import digest
from subnet.storage import canonical
from subnet.training_documents import VERSION


class DurableRevisionControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'journal.sqlite'
        self.j = ResearchRevisionJournal.create(self.path, enabled=True)
        self.f = fixtures.CurrentWireAdapterControls(); self.f.setUp()
        self.key = SigningKey.generate(); self.miner = self.key.verify_key.encode().hex()
        self.manifest = digest(self.f.manifest)

    def original(self, rolls, **wrapper):
        batch = self.f.batch(rolls); batch.update(wrapper)
        return canonical(dict(version=VERSION, epoch=self.f.adapter.task.epoch,
            checkpoint=self.f.adapter.task.checkpoint, miner=self.miner, slot=0, batch=batch))

    def append(self, data, journal=None, manifest=None):
        manifest = manifest or self.manifest
        b = scope(self.f.adapter, self.miner, manifest)
        signature = self.key.sign(data + canonical(b)).signature
        def authenticate(original, evidence, proposed):
            # Test boundary performs real signature verification AND independent
            # expected scope equality. A live caller also authenticates commitment
            # and ROOT cheap-admission/captured SHA before returning this receipt.
            if proposed != b: raise ValueError('scope not independently approved')
            self.key.verify_key.verify(original + canonical(b), bytes.fromhex(evidence['signature']))
            return boundary_receipt(hashlib.sha256(original).hexdigest(), digest(b))
        return (journal or self.j).append_authenticated(self.f.adapter, self.miner, manifest,
            data, dict(signature=signature.hex()), authenticate)

    def test_default_off_new_path_and_public_key_boundary(self):
        with self.assertRaises(ValueError): ResearchRevisionJournal(self.path)
        with self.assertRaises(FileExistsError): ResearchRevisionJournal.create(self.path, enabled=True)
        with self.assertRaises(ValueError): scope(self.f.adapter, 'uid-85', self.manifest)

    def test_restart_append_and_old_exact_replay_does_not_move_head(self):
        first = self.original(self.f.rolls[:1]); full = self.original(self.f.rolls)
        one = self.append(first)
        restarted = ResearchRevisionJournal(self.path, enabled=True)
        two = self.append(full, restarted)
        replay = self.append(first, restarted)
        self.assertTrue(replay['redelivery']); self.assertFalse(replay['original_recorded'])
        self.assertEqual(replay['revision_id'], one['revision_id'])
        self.assertEqual(replay['head_revision_id'], two['revision_id'])
        state = restarted.inspect(one['slot_id'])
        self.assertEqual((state['revision_count'], state['original_count']), (2, 2))

    def test_restart_lost_changed_token_and_class_attempt_reject_rollback(self):
        original = self.original(self.f.rolls); one = self.append(original)
        before = self.j.inspect(one['slot_id'])
        for change in ('remove', 'token', 'label'):
            rolls = copy.deepcopy(self.f.rolls)
            if change == 'remove': rolls.pop()
            elif change == 'token': rolls[0]['turns'][0].update(output=[120], text='x')
            else: rolls[0]['classification'] = 'negative'
            with self.assertRaises(ValueError):
                self.append(self.original(rolls), ResearchRevisionJournal(self.path, enabled=True))
            self.assertEqual(self.j.inspect(one['slot_id']), before)

    def test_concurrent_exact_replay_one_original_one_revision(self):
        data = self.original(self.f.rolls)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.append(data, ResearchRevisionJournal(self.path, enabled=True)), range(16)))
        self.assertEqual(sum(r['original_recorded'] for r in results), 1)
        state = self.j.inspect(results[0]['slot_id'])
        self.assertEqual((state['revision_count'], state['original_count']), (1, 1))

    def test_concurrent_incomparable_extensions_one_wins_one_rejected(self):
        initial = self.append(self.original(self.f.rolls[:1]))
        documents = [self.original(self.f.rolls[:2]), self.original([self.f.rolls[0], self.f.rolls[2]])]
        def run(data):
            try: return self.append(data, ResearchRevisionJournal(self.path, enabled=True))
            except ValueError: return 'rejected'
        with ThreadPoolExecutor(max_workers=2) as pool: results = list(pool.map(run, documents))
        self.assertEqual(results.count('rejected'), 1)
        self.assertEqual(self.j.inspect(initial['slot_id'])['revision_count'], 2)

    def test_repacked_same_revision_preserves_both_originals(self):
        one = self.append(self.original(self.f.rolls))
        two = self.append(self.original(self.f.rolls[::-1], research_wrapper='repack'))
        self.assertTrue(two['redelivery']); self.assertTrue(two['original_recorded'])
        self.assertEqual(one['revision_id'], two['revision_id'])
        state = self.j.inspect(one['slot_id'])
        self.assertEqual((state['revision_count'], state['original_count']), (1, 2))
        with sqlite3.connect(self.path) as db:
            with self.assertRaises(sqlite3.IntegrityError): db.execute("UPDATE originals SET bytes=X'00'")
            with self.assertRaises(sqlite3.IntegrityError): db.execute('DELETE FROM revisions')

    def test_self_asserted_boolean_and_bad_signature_cannot_create_slot(self):
        data = self.original(self.f.rolls)
        for callback in (lambda *args: True, lambda *args: dict(version='fake')):
            with self.assertRaises(ValueError): self.j.append_authenticated(self.f.adapter, self.miner, self.manifest, data, {}, callback)
        with sqlite3.connect(self.path) as db: self.assertEqual(db.execute('SELECT COUNT(*) FROM slots').fetchone()[0], 0)
        signature = self.key.sign(data).signature
        def bad(original, evidence, binding):
            self.key.verify_key.verify(original+canonical(binding), signature)
        with self.assertRaises(Exception): self.j.append_authenticated(self.f.adapter, self.miner, self.manifest, data, {}, bad)

    def test_claimed_execution_content_ids_never_override_adapter_computation(self):
        one = self.append(self.original(self.f.rolls))
        rolls = copy.deepcopy(self.f.rolls)
        for row in rolls:
            row.update(execution_id='e'*64, content_id='d'*64, uid=123)
        two = self.append(self.original(rolls))
        self.assertEqual(one['revision_id'], two['revision_id'])
        self.assertTrue(two['redelivery'])
        with sqlite3.connect(self.path) as db:
            for sha, data in db.execute('SELECT sha256,bytes FROM originals'):
                self.assertEqual(hashlib.sha256(data).hexdigest(), sha)
            with self.assertRaises(sqlite3.IntegrityError): db.execute("UPDATE slots SET binding='rebound'")

    def test_changed_authentication_evidence_on_original_replay_is_refused(self):
        data = self.original(self.f.rolls); one = self.append(data)
        binding = scope(self.f.adapter, self.miner, self.manifest)
        evidence = dict(signature=self.key.sign(data+canonical(binding)).signature.hex(), changed='new')
        def auth(original, ev, proposed):
            self.key.verify_key.verify(original+canonical(proposed), bytes.fromhex(ev['signature']))
            return boundary_receipt(hashlib.sha256(original).hexdigest(), digest(proposed))
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            self.j.append_authenticated(self.f.adapter,self.miner,self.manifest,data,evidence,auth)
        self.assertEqual(self.j.inspect(one['slot_id'])['original_count'], 1)

    def test_mutated_adapter_spec_cannot_reuse_old_approved_scope(self):
        self.append(self.original(self.f.rolls))
        self.f.adapter.definition['spec']['config']['seed'] = 99
        with self.assertRaisesRegex(ValueError, 'adapter approved'):
            self.append(self.original(self.f.rolls))

    def test_changed_authenticated_manifest_refused_even_exact_document(self):
        data = self.original(self.f.rolls); one = self.append(data)
        with self.assertRaisesRegex(ValueError, 'slot binding changed'): self.append(data, manifest='f'*64)
        self.assertEqual(self.j.inspect(one['slot_id'])['original_count'], 1)

    def test_abrupt_process_death_mid_transaction_leaves_prior_head_no_orphans(self):
        one = self.append(self.original(self.f.rolls[:1]))
        data = self.original(self.f.rolls); target = Path(self.tmp.name)/'input.json'; target.write_bytes(data)
        # Real abrupt process death AFTER SQL inserts and BEFORE COMMIT. Recovery
        # must expose only the durable earlier snapshot and permit the uncommitted
        # cumulative append once; this is revision persistence, not optimizer retry.
        code = '''import contextlib,os,sqlite3,json,sys
import test_paired_quota_batch_adapter as f
from ops.paired_quota_revision_journal import *
from ops.paired_quota_qualification import digest
fixture=f.CurrentWireAdapterControls();fixture.setUp();j=ResearchRevisionJournal(sys.argv[1],enabled=True)
class Proxy:
 def __init__(self,c):self.c=c
 def execute(self,sql,args=()):
  r=self.c.execute(sql,args)
  if sql.startswith('UPDATE slots SET head'):os._exit(77)
  return r
 def commit(self):self.c.commit()
 def rollback(self):self.c.rollback()
@contextlib.contextmanager
def connect():
 c=sqlite3.connect(j.path,isolation_level=None);c.execute('PRAGMA foreign_keys=ON');c.execute('PRAGMA synchronous=FULL')
 try:yield Proxy(c)
 finally:c.close()
j._connect=connect
data=open(sys.argv[2],'rb').read();j.append_authenticated(fixture.adapter,sys.argv[3],digest(fixture.manifest),data,{},lambda original,evidence,binding:boundary_receipt(__import__('hashlib').sha256(original).hexdigest(),digest(binding)))
'''
        run = subprocess.run([sys.executable, '-c', code, str(self.path), str(target), self.miner],
            env=dict(__import__('os').environ, PYTHONPATH=str(Path.cwd())+':'+str(Path.cwd()/'tests')), capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 77, run.stderr.decode())
        restarted = ResearchRevisionJournal(self.path, enabled=True)
        self.assertEqual((restarted.inspect(one['slot_id'])['revision_count'],restarted.inspect(one['slot_id'])['original_count']), (1, 1))
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM originals').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 1)
        self.assertEqual(self.append(data, restarted)['redelivery'], False)

if __name__ == '__main__': unittest.main()
