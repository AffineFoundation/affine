import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from nacl.signing import SigningKey

from ops.consumed_application_ledger import ConsumedApplicationLedger, VERSION, raw
from ops.paired_quota_qualification import ApprovedTask, digest, select_pairs
from ops.paired_quota_research_ledger import RecoveryRequired


class ActualApplicationControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'claim.sqlite'
        self.ledger = ConsumedApplicationLedger.create(self.path, enabled=True)
        self.key = SigningKey.generate()
        task = ApprovedTask('e1','cp','a'*64,'math',10,'b'*64,'c'*64,'d'*64,(0,1,2,3))
        rows = [dict(task.task_binding(), epoch=task.epoch,harness_sha256=task.harness_sha256,
          sampling_context_sha256=task.sampling_context_sha256,attempt=i,
          classification='positive' if i<2 else 'negative',
          turns=[dict(prompt=[10],output=[100+i],actions=[],observations=[])]) for i in range(4)]
        revision = select_pairs(task,'miner',rows,quota=2)
        self.binding = dict(version=VERSION, branch_sha256='1'*64,plan_sha256='2'*64,
          parent_state_sha256='3'*64,input_checkpoint_sha256='4'*64,step_before=33,
          settings_sha256='5'*64,selected_revisions=[revision],groups=[[revision['slot_id']]]*3)

    def original(self, binding=None, original='a'*64):
        b = copy.deepcopy(self.binding if binding is None else binding)
        payload = dict(original_sha256=original,binding=b)
        return dict(payload=payload,signature=self.key.sign(raw(payload).encode()).signature.hex())

    def authenticate(self, original, evidence):
        self.key.verify_key.verify(raw(evidence['payload']).encode(),bytes.fromhex(evidence['signature']))
        if evidence['payload']['original_sha256'] != original: raise ValueError('original mismatch')
        return evidence['payload']['binding']

    def reserve(self, binding=None, original='a'*64):
        return self.ledger.reserve_original(original,self.original(binding,original),self.authenticate)

    def publication(self, aid):
        b = self.ledger.inspect(aid)['binding']
        payload = dict(version='consumed-application-publication-research-v1',application_id=aid,
          parent_state_sha256=b['parent_state_sha256'],step_before=b['step_before'],
          step_after=b['step_before']+len(b['groups']),output_state_sha256='e'*64,
          output_checkpoint_sha256='f'*64)
        return dict(payload=payload,signature=self.key.sign(raw(payload).encode()).signature.hex())

    def auth_publication(self,e,b,aid):
        self.key.verify_key.verify(raw(e['payload']).encode(),bytes.fromhex(e['signature']))
        return e['payload']

    def test_default_off_and_private_schema(self):
        with self.assertRaises(ValueError): ConsumedApplicationLedger(self.path)
        with self.assertRaises(FileExistsError): ConsumedApplicationLedger.create(self.path,enabled=True)
        self.assertEqual(self.path.stat().st_mode & 0o777,0o600)

    def test_repacked_original_aliases_one_application_after_restart(self):
        aid=self.reserve(); other=self.reserve(original='b'*64)
        self.assertEqual(aid,other)
        calls=[]
        receipt=self.ledger.execute_once(aid,lambda b:calls.append(b) or self.publication(aid),self.auth_publication)
        reopened=ConsumedApplicationLedger(self.path,enabled=True)
        self.assertEqual(reopened.execute_once(other,lambda b:self.fail('double application'),None),receipt)
        self.assertEqual(len(calls),1)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM applications').fetchone()[0],1)
            self.assertEqual(db.execute('SELECT count(*) FROM originals').fetchone()[0],2)

    def test_changed_parent_application_rejected_despite_new_wrapper_plan(self):
        self.reserve()
        for field,value in [('settings_sha256','6'*64),('plan_sha256','7'*64),('step_before',34),
                            ('input_checkpoint_sha256','8'*64),('groups',[self.binding['groups'][0]])]:
            b=copy.deepcopy(self.binding); b[field]=value
            with self.assertRaisesRegex(ValueError,'parent already'):
                self.reserve(b,original='b'*64)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM originals').fetchone()[0],1)

    def test_intentional_declared_schedule_reuse_not_replayed_receipt(self):
        aid=self.reserve()
        seen=[]
        result=self.ledger.execute_once(aid,lambda b:seen.extend(b['groups']) or self.publication(aid),self.auth_publication)
        self.assertEqual(len(seen),3); self.assertEqual(result['step_after'],36)
        b=copy.deepcopy(self.binding);b['parent_state_sha256']='e'*64;b['step_before']=36
        # A genuinely authenticated NEW parent/application may prescribe reuse;
        # this fixture does not establish actual optimizer lineage or authority.
        self.assertNotEqual(self.reserve(b,original='c'*64),aid)

    def test_signed_original_tamper_boolean_and_extra_wrapper_fields_refused(self):
        e=self.original();e['payload']['binding']['step_before']=34
        with self.assertRaises(Exception): self.ledger.reserve_original('a'*64,e,self.authenticate)
        with self.assertRaises(ValueError): self.ledger.reserve_original('a'*64,{},lambda s,e:True)
        b=copy.deepcopy(self.binding);b['job_id']='another-label'
        with self.assertRaises(ValueError):self.reserve(b)
        with sqlite3.connect(self.path) as db:self.assertEqual(db.execute('SELECT count(*) FROM applications').fetchone()[0],0)

    def test_concurrent_distinct_repack_claims_one_executor(self):
        ids=[]
        with ThreadPoolExecutor(max_workers=12) as pool:
            ids=list(pool.map(lambda i:self.reserve(original=digest({'wrapper':i})),range(24)))
        self.assertEqual(len(set(ids)),1)
        counter=[];lock=threading.Lock()
        def execute(b):
            with lock:counter.append(1)
            time.sleep(.1)
            return self.publication(ids[0])
        def call(_):
            try:return self.ledger.execute_once(ids[0],execute,self.auth_publication)
            except RecoveryRequired:return 'unresolved'
        with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(call,range(24)))
        self.assertEqual(counter,[1]);self.assertEqual(self.ledger.inspect(ids[0])['state'],'complete')

    def run_crash(self, after):
        aid=self.reserve(); marker=Path(self.tmp.name)/'simulated-update'
        evidence=self.publication(aid); output=Path(self.tmp.name)/'synthetic-durable-output.json'
        code='''import json,os,sys
from pathlib import Path
from ops.consumed_application_ledger import ConsumedApplicationLedger
ledger=ConsumedApplicationLedger(sys.argv[1],enabled=True)
def execute(binding):
 if sys.argv[4]=='after':
  for path,data in [(sys.argv[3],b'one simulated update'),(sys.argv[5],sys.argv[6].encode())]:
   fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,data);os.fsync(fd);os.close(fd)
 os._exit(78)
ledger.execute_once(sys.argv[2],execute,None)
'''
        run=subprocess.run([sys.executable,'-c',code,str(self.path),aid,str(marker),after,str(output),raw(evidence)],capture_output=True,text=True)
        self.assertEqual(run.returncode,78,run.stderr)
        reopened=ConsumedApplicationLedger(self.path,enabled=True)
        self.assertEqual(reopened.inspect(aid)['state'],'executing')
        with self.assertRaises(RecoveryRequired):reopened.execute_once(aid,lambda b:self.fail('reapplied'),self.auth_publication)
        self.assertEqual(marker.exists(),after=='after')
        if after=='after':
            actual=json.loads(output.read_text())
            reopened.record_publication(aid,actual,self.auth_publication)
            reopened.execute_once(aid,lambda b:self.fail('reapplied after recovery'),None)
        return aid

    def test_actual_process_death_before_first_update_blocks_auto_retry(self):self.run_crash('before')
    def test_actual_process_death_after_simulated_update_and_output_metadata_recovery(self):self.run_crash('after')

    def test_publication_requires_started_application_exact_lineage_and_signature(self):
        aid=self.reserve();e=self.publication(aid)
        with self.assertRaisesRegex(ValueError,'not started'):self.ledger.record_publication(aid,e,self.auth_publication)
        with self.assertRaises(RuntimeError):
            self.ledger.execute_once(aid,lambda b:(_ for _ in ()).throw(RuntimeError('uncertain')),self.auth_publication)
        changed=copy.deepcopy(e);changed['payload']['step_after']=37
        with self.assertRaises(Exception):self.ledger.record_publication(aid,changed,self.auth_publication)
        changed['signature']=self.key.sign(raw(changed['payload']).encode()).signature.hex()
        with self.assertRaises(ValueError):self.ledger.record_publication(aid,changed,self.auth_publication)
        with self.assertRaises(ValueError):self.ledger.record_publication(aid,e,lambda e,b,a:True)
        self.ledger.record_publication(aid,e,self.auth_publication)
        changed=copy.deepcopy(e);changed['payload']['output_state_sha256']='9'*64
        changed['signature']=self.key.sign(raw(changed['payload']).encode()).signature.hex()
        with self.assertRaisesRegex(ValueError,'changed'):self.ledger.record_publication(aid,changed,self.auth_publication)

    def test_reset_delete_and_original_rewrite_forbidden(self):
        aid=self.reserve()
        with self.assertRaises(RuntimeError):self.ledger.execute_once(aid,lambda b:(_ for _ in ()).throw(RuntimeError()),None)
        with sqlite3.connect(self.path) as db:
            for sql in ["UPDATE applications SET state='reserved'",'DELETE FROM applications',"UPDATE originals SET evidence='{}'",'DELETE FROM originals']:
                with self.assertRaises(sqlite3.IntegrityError):db.execute(sql)

    def test_cross_task_copied_members_cannot_fill_population(self):
        b=copy.deepcopy(self.binding)
        r=copy.deepcopy(b['selected_revisions'][0]);r['slot_id']='9'*64
        r['revision_id']=digest({k:v for k,v in r.items() if k not in ('revision_id','duplicate_content_count')})
        b['selected_revisions'].append(r);b['groups']=[[b['selected_revisions'][0]['slot_id'],r['slot_id']]]
        with self.assertRaisesRegex(ValueError,'duplicate execution/content'):self.reserve(b)

    def test_competing_changed_applications_transactionally_reserve_one_parent(self):
        def call(i):
            b=copy.deepcopy(self.binding);b['plan_sha256']=digest({'distinct_plan':i})
            try:return self.reserve(b,original=digest({'attempt':i}))
            except ValueError:return None
        with ThreadPoolExecutor(max_workers=12) as pool:out=list(pool.map(call,range(24)))
        self.assertEqual(len([v for v in out if v is not None]),1)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM applications').fetchone()[0],1)
            self.assertEqual(db.execute('SELECT count(*) FROM originals').fetchone()[0],1)

    def test_original_task_schedule_reuses_one_task_for_three_updates(self):
        from subnet.task_normalized_training import task_groups
        p=dict(env_id='math',index=10,task_hash='b'*64,classification='positive',
               turns=[dict(prompt=[1],output=[2])])
        n=dict(p,classification='negative',turns=[dict(prompt=[1],output=[3])])
        pairs,tasks,groups,ids=task_groups([(dict(env_id='math'),p,n)],3,'a'*64)
        self.assertEqual(groups,[[0],[0],[0]])
        self.assertEqual(len(pairs),1);self.assertEqual(len(tasks),1)

    def test_original_pair_identity_is_full_row_not_token_canonical(self):
        from subnet.covered_epoch_optimizer import pair_identity, distinct_verified_pairs
        from subnet.trajectory_identity import token_trace_sha256
        p=dict(env_id='math',index=10,task_hash='b'*64,classification='positive',
               turns=[dict(prompt=[1],output=[2])])
        n=dict(p,classification='negative',turns=[dict(prompt=[1],output=[3])])
        original=(dict(env_id='math'),p,n)
        repack=(dict(env_id='math'),dict(p,uid=99),copy.deepcopy(n))
        self.assertEqual(token_trace_sha256(p['turns']),token_trace_sha256(repack[1]['turns']))
        self.assertNotEqual(pair_identity(original),pair_identity(repack))
        self.assertEqual(len(distinct_verified_pairs([original,repack])),2)
        # This demonstrates helper scope, not actual accepted live duplication:
        # current cheap admission separately enforces one committed task.

    def test_duplicate_task_within_group_and_changed_evidence_refused(self):
        b=copy.deepcopy(self.binding);b['groups'][0]=b['groups'][0]*2
        with self.assertRaises(ValueError):self.reserve(b)
        self.reserve();e=self.original();e['extra_wrapper']='changed'
        with self.assertRaisesRegex(ValueError,'evidence changed'):self.ledger.reserve_original('a'*64,e,self.authenticate)
