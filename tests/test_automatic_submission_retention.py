import hashlib
import io
import copy
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
import unittest
from ops.automatic_submission_retention import archived, run, retention_writer, retention_configs
from ops.live_reward_exporter import sign
from subnet.live_reward_bridge import sha
from ops.live_reward_source_approval import source_verifiers
from ops.verifier_workforce import authorize_worker

class AutomaticRetention(unittest.TestCase):
    def fixture(self, data, expected=b'archive'):
        body=io.BytesIO(data)
        client=SimpleNamespace(get_object=lambda **kw:{'Body':body})
        bucket=SimpleNamespace(client=client,name='durable')
        plan={'archive_key':'public/frozen.zip','size':len(expected),'sha256':hashlib.sha256(expected).hexdigest()}
        return bucket,plan,body

    def test_retirement_requires_complete_matching_archive(self):
        bucket,plan,body=self.fixture(b'archive')
        verified=archived(bucket,plan)
        self.assertTrue(verified['archive_verified'])
        self.assertNotIn('archive_verified',plan)
        self.assertTrue(body.closed)

    def test_corrupt_truncated_or_oversized_archive_refuses_retirement(self):
        for payload in [b'corrupt',b'arch',b'archive-too-long']:
            bucket,plan,body=self.fixture(payload)
            with self.assertRaises(ValueError):archived(bucket,plan)
            self.assertTrue(body.closed)
            self.assertNotIn('archive_verified',plan)

    def test_unbounded_retention_rate_refuses_before_read_or_network(self):
        for limit in [0,33,True,1.5]:
            with self.assertRaises(ValueError):run('absent','absent','absent','absent',per_worker=limit)

class RetentionAuthority(unittest.TestCase):
    def setUp(self):
        from test_live_reward_source_approval import ApprovalTests
        self.fixture=ApprovalTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;f.c['anchor_sha256']=sha(f.anchor);f.cutover=sign(f.c,f.key)
        f.payload['original_cutover_sha256']=sha(f.cutover)
        self.root=Path(f.t.name);self.cutover=self.root/'cutover.json';self.cutover.write_text(json.dumps(f.cutover))
        self.anchor=self.root/'anchor.json';self.anchor.write_text(json.dumps(f.anchor))
        self.pointer={'cutover_path':str(self.cutover),'cutover_sha256':hashlib.sha256(self.cutover.read_bytes()).hexdigest(),'anchor_path':str(self.anchor)}
        self.approval=sign(f.payload,f.key)

    def test_signed_addition_preserves_original_and_authorizes_only_new_source_workers(self):
        f=self.fixture;before=copy.deepcopy(f.cutover)
        c=retention_writer(self.pointer,f.auth,[self.approval],[])
        self.assertEqual(json.loads(self.cutover.read_bytes()),before)
        m={'source_bundle':{'sha256':f.digest},'start':201,'epoch':'nonpayable-live-reward-math-v1-test'}
        j={'source_files':f.payload['runtime_source_files'],'runtime_versions':f.c['runtime_versions']}
        self.assertEqual(source_verifiers(c,m,j),f.ids)
        m['source_bundle']['sha256']=f.old
        self.assertEqual(source_verifiers(c,m,{}),f.ids[:2])
        with self.assertRaisesRegex(ValueError,'approved verifier'):
            authorize_worker(f.ids[2],m,{}, {},None,source_verifiers(c,m,{}),c['_verifier_workforce'])

    def test_forged_source_or_workforce_document_refused_before_retirement(self):
        f=self.fixture;bad=copy.deepcopy(self.approval);bad['payload']['verifier_identities'][-1]='f'*64
        with self.assertRaises(Exception):retention_writer(self.pointer,f.auth,[bad],[])
        forged=sign({'version':'operational-verifier-workforce-v1'},f.key)
        forged['signature']='not-a-signature'
        with self.assertRaises(Exception):retention_writer(self.pointer,f.auth,[],[forged])

    def test_historical_supplement_retains_original_claim_time_requirement(self):
        from test_verifier_workforce import WorkforceTests
        fixture=WorkforceTests();fixture.setUp();self.addCleanup(fixture.tearDown)
        f=self.fixture;m=copy.deepcopy(fixture.manifest);m['source_bundle']['sha256']=f.old
        payload=dict(fixture.payload,cutover_document_sha256=sha(f.cutover),
                     existing_verifier_identities=f.ids[:2],additional_verifier_identities=f.ids[2:],
                     source_sha256=f.old,opening_manifest_sha256=sha(m))
        c=retention_writer(self.pointer,f.auth,[],[sign(payload,f.key)])
        db=fixture.db;db.execute('delete from events')
        fixture.claim(101,worker=f.ids[2])
        authorized=authorize_worker(f.ids[2],m,fixture.job,fixture.row,db,
                                    source_verifiers(c,m,fixture.job),c['_verifier_workforce'])
        self.assertEqual(authorized['worker_claimed_at'],101)
        db.execute('delete from events');fixture.claim(99,worker=f.ids[2])
        with self.assertRaisesRegex(ValueError,'predates'):
            authorize_worker(f.ids[2],m,fixture.job,fixture.row,db,
                             source_verifiers(c,m,fixture.job),c['_verifier_workforce'])

    def test_wrong_pointer_and_anchor_binding_refused(self):
        f=self.fixture;p=dict(self.pointer,cutover_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'pointer'):retention_writer(p,f.auth,[],[])
        self.anchor.write_text(json.dumps(sign(dict(f.anchor['payload'],effective_at=999),f.key)))
        with self.assertRaisesRegex(ValueError,'anchor'):retention_writer(self.pointer,f.auth,[self.approval],[])

    def config(self,source):
        return {'source_bundle':{'sha256':source},'state':str(self.root),'bucket':{'name':'durable'},
                'remote':{'roles':{'verify':[{'worker_identity':'1'*64,'workspace':'/root/exact-worker'}]}}}

    def test_multiple_source_configs_preserved_without_overwrite(self):
        old=self.config('a'*64);new=self.config('b'*64);third=self.config('c'*64)
        writer={'queue_database':str(self.root/'roles/verifier-queue.sqlite3')}
        before=copy.deepcopy([old,new,third])
        configs=retention_configs(old,[new,third],writer,{k:{}for k in ['a'*64,'b'*64,'c'*64]})
        self.assertEqual(len(configs),3);self.assertEqual([old,new,third],before)
        conflicting=copy.deepcopy(new);conflicting['remote']['roles']['verify'][0]['workspace']='/root/different'
        with self.assertRaisesRegex(ValueError,'ambiguous'):retention_configs(old,[new,conflicting],writer,configs)

    def test_unsupported_source_wrong_queue_or_workspace_refused(self):
        old=self.config('a'*64);writer={'queue_database':str(self.root/'roles/verifier-queue.sqlite3')}
        for field in ['unknown-source','wrong-state','duplicate-worker','relative-workspace']:
            c=copy.deepcopy(old)
            if field=='unknown-source':c['source_bundle']['sha256']='b'*64
            if field=='wrong-state':c['state']=str(self.root/'other')
            if field=='duplicate-worker':c['remote']['roles']['verify']*=2
            if field=='relative-workspace':c['remote']['roles']['verify'][0]['workspace']='relative'
            with self.subTest(field=field),self.assertRaises(ValueError):retention_configs(c,[],writer,{'a'*64:{}})
        with self.assertRaisesRegex(ValueError,'bounded'):retention_configs(old,[old]*17,writer,{'a'*64:{}})
