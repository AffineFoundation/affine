import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import os
import tempfile
import unittest
from nacl.signing import SigningKey
import test_paired_quota_batch_adapter as fixtures
from ops.paired_quota_revision_journal import ResearchRevisionJournal,scope,boundary_receipt
from ops.paired_quota_journal_selector_bridge import ResearchQuotaBridge
from ops.paired_quota_nested_selector import GRADE_VERSION,selection_scope
from ops.paired_quota_qualification import digest
from subnet.storage import canonical
from subnet.training_documents import VERSION


class CapturedNativeJoinControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.jp=Path(self.tmp.name)/'original.sqlite';self.bp=Path(self.tmp.name)/'bridge.sqlite'
        self.j=ResearchRevisionJournal.create(self.jp,enabled=True);self.b=ResearchQuotaBridge.create(self.bp,enabled=True)
        self.f=fixtures.CurrentWireAdapterControls();self.f.setUp()
        for row in self.f.rolls:
            row['turns'][0]['output'].append(999);row['turns'][0]['text']=self.f.decode(row['turns'][0]['output'])
        self.key=SigningKey.generate();self.native_key=SigningKey.generate();self.miner=self.key.verify_key.encode().hex()
        self.manifest=digest(self.f.manifest);self.eos=(999,)
        self.approved_scope=scope(self.f.adapter,self.miner,self.manifest)
        self.rows=self.f.adapter.normalize(self.f.batch(self.f.rolls))

    def auth_original(self,data,evidence,binding):
        self.assertEqual(binding,self.approved_scope)
        self.key.verify_key.verify(data+canonical(binding),bytes.fromhex(evidence['signature']))
        return boundary_receipt(hashlib.sha256(data).hexdigest(),digest(binding))

    def capture(self,rolls=None,doc_slot=0):
        data=canonical(dict(version=VERSION,epoch=self.f.adapter.task.epoch,checkpoint=self.f.adapter.task.checkpoint,
                            miner=self.miner,slot=doc_slot,batch=self.f.batch(self.f.rolls if rolls is None else rolls)))
        ev=dict(signature=self.key.sign(data+canonical(self.approved_scope)).signature.hex())
        return self.j.append_authenticated(self.f.adapter,self.miner,self.manifest,data,ev,self.auth_original)

    def outcome(self,row=None,*,attempt=None,status='native-graded',done=True,classification=None,reward=None):
        attempt=row['attempt']if row is not None else attempt
        label=classification or row['classification']if row is not None else None
        if status=='native-graded':reward=int(label=='positive')if reward is None else reward
        else:label=reward=done=None
        p=dict(attempt=attempt,row_sha256=digest(row)if row is not None else None,
               selection_scope_sha256=digest(selection_scope(self.f.adapter.task,self.miner,self.eos)),
               status=status,classification=label,reward=reward,native_done=done)
        return dict(attempt=attempt,evidence=dict(payload=p,signature=self.native_key.sign(canonical(p)).signature.hex()))

    def auth_native(self,record,binding):
        ev=record['evidence'];p=ev['payload'];self.native_key.verify_key.verify(canonical(p),bytes.fromhex(ev['signature']))
        for k in ('attempt','row_sha256','selection_scope_sha256'):self.assertEqual(p[k],binding[k])
        return dict(binding,version=GRADE_VERSION,**{k:p[k]for k in ('status','classification','reward','native_done')})

    def join(self,revision,outcomes,bridge=None,**options):
        kwargs=dict(eos_token_ids=self.eos,authenticate_original=self.auth_original,authenticate_admitted_native=self.auth_native);kwargs.update(options)
        return (bridge or self.b).select_revision(self.j,self.f.adapter,self.miner,self.manifest,revision,outcomes,**kwargs)

    def test_default_off_and_end_to_end_restart_exact_redelivery(self):
        with self.assertRaises(ValueError):ResearchQuotaBridge(self.bp)
        captured=self.capture();outcomes=[self.outcome(r)for r in self.rows]
        first=self.join(captured['revision_id'],outcomes)
        second=self.join(captured['revision_id'],list(reversed(outcomes)),ResearchQuotaBridge(self.bp,enabled=True))
        self.assertFalse(first['redelivery']);self.assertTrue(second['redelivery']);self.assertEqual(first['result'],second['result'])
        selected=first['result']['selection'];self.assertTrue(selected['complete_K2']);self.assertEqual(len(selected['K2L2']['pairs']),2)
        self.assertEqual(first['result']['input_assurance'],'cheap-eligible-unaudited');self.assertFalse(first['result']['inference_verified'])
        self.assertFalse(first['result']['optimizer_application_performed'])

    def test_cumulative_documents_outcomes_and_historical_redelivery(self):
        first=self.capture(self.f.rolls[:1]);one=[self.outcome(self.rows[0])]
        early=self.join(first['revision_id'],one)
        full=self.capture();all_outcomes=[self.outcome(r)for r in self.rows]
        late=self.join(full['revision_id'],all_outcomes)
        self.assertFalse(early['result']['selection']['complete_K1']);self.assertTrue(late['result']['selection']['complete_K2'])
        replay=self.join(first['revision_id'],one,ResearchQuotaBridge(self.bp,enabled=True))
        self.assertTrue(replay['redelivery']);self.assertEqual(replay['result'],early['result'])
        with self.assertRaisesRegex(ValueError,'removed original outcome'):
            self.join(full['revision_id'],all_outcomes[1:],ResearchQuotaBridge(self.bp,enabled=True))

    def test_checkpoint_epoch_task_and_public_key_mismatch_refused(self):
        c=self.capture();outcomes=[self.outcome(r)for r in self.rows]
        variants=[(replace(self.f.adapter,task=replace(self.f.adapter.task,checkpoint='f'*64)),self.miner),
                  (replace(self.f.adapter,task=replace(self.f.adapter.task,epoch='different')),self.miner),
                  (replace(self.f.adapter,task=replace(self.f.adapter.task,task_sha256='e'*64)),self.miner),
                  (self.f.adapter,'d'*64)]
        for adapter,miner in variants:
            with self.assertRaises(ValueError):self.b.select_revision(self.j,adapter,miner,self.manifest,c['revision_id'],outcomes,eos_token_ids=self.eos,authenticate_original=self.auth_original,authenticate_admitted_native=self.auth_native)
        with self.assertRaises(ValueError):self.b.select_revision(self.j,self.f.adapter,self.miner,'f'*64,c['revision_id'],outcomes,eos_token_ids=self.eos,authenticate_original=self.auth_original,authenticate_admitted_native=self.auth_native)

    def test_changed_authenticated_native_outcome_refused_after_restart(self):
        c=self.capture();outcomes=[self.outcome(r)for r in self.rows];self.join(c['revision_id'],outcomes)
        changed=copy.deepcopy(outcomes);changed[0]=self.outcome(self.rows[0],done=False)
        with self.assertRaisesRegex(ValueError,'outcome/evidence changed'):
            self.join(c['revision_id'],changed,ResearchQuotaBridge(self.bp,enabled=True))
        with sqlite3.connect(self.bp)as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM packets').fetchone()[0],1)

    def test_native_outcome_for_uncaptured_row_cannot_fill_quota(self):
        c=self.capture(self.f.rolls[:1]);outcomes=[self.outcome(r)for r in self.rows]
        with self.assertRaises(AssertionError):self.join(c['revision_id'],outcomes)
        replacement=dict(attempt=0,evidence=outcomes[0]['evidence'],row=self.rows[0])
        with self.assertRaisesRegex(ValueError,'replacement rows'):self.join(c['revision_id'],[replacement])

    def test_missing_prefix_blocks_and_authenticated_failure_restores_observation(self):
        c=self.capture();outcomes=[self.outcome(r)for r in self.rows[1:]]
        result=self.join(c['revision_id'],outcomes)['result']['selection']
        self.assertFalse(result['complete_K1']);self.assertFalse(result['complete_K2'])
        # Same captured row has real signed indeterminate status; it is observed
        # but ineligible, so the original grade must not be forged as success.
        prefix=self.outcome(self.rows[0],status='indeterminate')
        observed=self.join(c['revision_id'],[prefix]+outcomes)['result']['selection']
        self.assertTrue(observed['complete_K1']);self.assertFalse(observed['complete_K2'])
        self.assertEqual(observed['supply'][0]['reason'],'indeterminate')

    def test_gap_between_K1_and_K2_blocks_only_K2_until_authenticated_failure(self):
        rolls=copy.deepcopy([self.f.rolls[0],self.f.rolls[2],self.f.rolls[1],self.f.rolls[3]])
        from subnet.forced_sampling import receipt
        for row,attempt in zip(rolls,[0,1,3,4]):
            row['seed']=attempt;row['sampling']=receipt(self.f.adapter.sampling_context,attempt)
        captured=self.capture(rolls);rows=self.f.adapter.normalize(self.f.batch(rolls))
        outcomes=[self.outcome(r)for r in rows]
        partial=self.join(captured['revision_id'],outcomes)['result']['selection']
        self.assertTrue(partial['complete_K1']);self.assertFalse(partial['complete_K2'])
        self.assertEqual(partial['completion_prefix_gaps']['K2L2'],[2])
        failure=self.outcome(None,attempt=2,status='infrastructure')
        complete=self.join(captured['revision_id'],outcomes+[failure],ResearchQuotaBridge(self.bp,enabled=True))['result']['selection']
        self.assertTrue(complete['complete_K2']);self.assertEqual(complete['first_K2_prefix_length'],5)
        self.assertTrue(all(r['reason']=='not-observed'for r in complete['supply'][5:]))

    def test_duplicate_content_does_not_fill_from_distinct_captured_attempts(self):
        rolls=copy.deepcopy(self.f.rolls);rolls[1]['turns']=copy.deepcopy(rolls[0]['turns'])
        c=self.capture(rolls);rows=self.f.adapter.normalize(self.f.batch(rolls));outcomes=[self.outcome(r)for r in rows]
        result=self.join(c['revision_id'],outcomes)['result']['selection']
        self.assertTrue(result['complete_K1']);self.assertFalse(result['complete_K2']);self.assertEqual(result['duplicate_contents'],1)

    def test_repacked_original_does_not_change_frozen_selection(self):
        c=self.capture();outcomes=[self.outcome(r)for r in self.rows];first=self.join(c['revision_id'],outcomes)
        # Equivalent original document slot wrappers are immutable redeliveries,
        # not new trajectory IDs. Stable first recorded original must be retained.
        for slot in range(1,10):self.capture(doc_slot=slot)
        replay=self.join(c['revision_id'],outcomes,ResearchQuotaBridge(self.bp,enabled=True))
        self.assertTrue(replay['redelivery']);self.assertEqual(replay['result'],first['result'])

    def test_original_and_native_signature_tamper_refused_and_rollback(self):
        c=self.capture();outcomes=[self.outcome(r)for r in self.rows]
        bad=copy.deepcopy(outcomes);bad[0]['evidence']['payload']['reward']=0
        with self.assertRaises(Exception):self.join(c['revision_id'],bad)
        with self.assertRaisesRegex(ValueError,'reauthenticated'):self.join(c['revision_id'],outcomes,authenticate_original=lambda *args:True)
        with self.assertRaisesRegex(ValueError,'authenticated'):self.join(c['revision_id'],outcomes,authenticate_admitted_native=lambda *args:True)
        with sqlite3.connect(self.bp)as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM contexts').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0],0)

    def test_concurrent_replay_one_packet_and_immutable_outcomes(self):
        c=self.capture();outcomes=[self.outcome(r)for r in self.rows]
        with ThreadPoolExecutor(max_workers=8)as pool:
            results=list(pool.map(lambda _:self.join(c['revision_id'],outcomes,ResearchQuotaBridge(self.bp,enabled=True)),range(16)))
        self.assertEqual(sum(not r['redelivery']for r in results),1)
        with sqlite3.connect(self.bp)as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0],4)
            with self.assertRaises(sqlite3.IntegrityError):db.execute("UPDATE outcomes SET grade='changed'")

    def test_abrupt_bridge_process_death_rolls_back_outcomes_and_packet(self):
        captured=self.capture();outcomes=[self.outcome(r)for r in self.rows]
        target=Path(self.tmp.name)/'outcomes.json';target.write_bytes(canonical(outcomes))
        program='''import contextlib,os,sqlite3,json,sys,hashlib
from nacl.signing import VerifyKey
import test_paired_quota_batch_adapter as f
from ops.paired_quota_revision_journal import *
from ops.paired_quota_journal_selector_bridge import *
from ops.paired_quota_nested_selector import GRADE_VERSION
from subnet.storage import canonical
fixture=f.CurrentWireAdapterControls();fixture.setUp()
for r in fixture.rolls:r['turns'][0]['output'].append(999);r['turns'][0]['text']=fixture.decode(r['turns'][0]['output'])
j=ResearchRevisionJournal(sys.argv[1],enabled=True);b=ResearchQuotaBridge(sys.argv[2],enabled=True)
class Proxy:
 def __init__(self,c):self.c=c
 def execute(self,sql,args=()):
  r=self.c.execute(sql,args)
  if sql.startswith('INSERT INTO packets'):os._exit(78)
  return r
 def commit(self):self.c.commit()
 def rollback(self):self.c.rollback()
@contextlib.contextmanager
def connect():
 c=sqlite3.connect(b.path,isolation_level=None);c.execute('PRAGMA foreign_keys=ON');c.execute('PRAGMA synchronous=FULL')
 try:yield Proxy(c)
 finally:c.close()
b._connect=connect
miner=sys.argv[4];native_key=VerifyKey(bytes.fromhex(sys.argv[5]))
def original(data,evidence,binding):
 VerifyKey(bytes.fromhex(miner)).verify(data+canonical(binding),bytes.fromhex(evidence['signature']))
 return boundary_receipt(hashlib.sha256(data).hexdigest(),digest(binding))
def native(record,binding):
 ev=record['evidence'];p=ev['payload'];native_key.verify(canonical(p),bytes.fromhex(ev['signature']))
 for k in ('attempt','row_sha256','selection_scope_sha256'):assert p[k]==binding[k]
 return dict(binding,version=GRADE_VERSION,**{k:p[k]for k in ('status','classification','reward','native_done')})
b.select_revision(j,fixture.adapter,miner,digest(fixture.manifest),sys.argv[3],json.load(open(sys.argv[6])),eos_token_ids=(999,),authenticate_original=original,authenticate_admitted_native=native)
'''
        run=subprocess.run([sys.executable,'-c',program,str(self.jp),str(self.bp),captured['revision_id'],self.miner,self.native_key.verify_key.encode().hex(),str(target)],
            env=dict(os.environ,PYTHONPATH=str(Path.cwd())+':'+str(Path.cwd()/'tests')),capture_output=True,timeout=30)
        self.assertEqual(run.returncode,78,run.stderr.decode())
        with sqlite3.connect(self.bp)as db:
            for table in ('contexts','outcomes','packets'):self.assertEqual(db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0],0)
        self.assertEqual(self.j.inspect(captured['slot_id'])['revision_count'],1)
        recovered=self.join(captured['revision_id'],outcomes,ResearchQuotaBridge(self.bp,enabled=True))
        self.assertFalse(recovered['redelivery']);self.assertTrue(recovered['result']['selection']['complete_K2'])

    def test_genuine_authenticated_failed_missing_row_in_prefix(self):
        rolls=copy.deepcopy(self.f.rolls)
        for row in rolls:row['seed']+=1;row['sampling']=__import__('subnet.forced_sampling',fromlist=['receipt']).receipt(self.f.adapter.sampling_context,row['seed'])
        c=self.capture(rolls);rows=self.f.adapter.normalize(self.f.batch(rolls))
        outcomes=[self.outcome(None,attempt=0,status='infrastructure')]+[self.outcome(r)for r in rows]
        result=self.join(c['revision_id'],outcomes)['result']['selection'];self.assertTrue(result['complete_K2'])
        self.assertEqual(result['first_K2_prefix_length'],5)

if __name__=='__main__':unittest.main()
