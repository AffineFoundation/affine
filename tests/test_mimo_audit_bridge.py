"""Actual Miner/compact capture/queue signatures; no GPU or chain writes.

The native grading callback and model execution are explicit existing fixtures;
these tests establish evidence plumbing, not numerical qualification.
"""
from copy import deepcopy
import json
import time
import unittest
from unittest.mock import patch

from nacl.signing import SigningKey
import test_mimo_worker as worker_fixtures
from test_mimo_worker import sign
from test_complete_group_mimo import mimo_manifest
from subnet.storage import canonical
from subnet.complete_group_admission import TaskAuthority
from subnet.complete_group_capture import AdmissionLedger, capture
from subnet.complete_group_native_receipts import compact_native_grades
from subnet.distributed_roles import Coordinator
from subnet.mimo_audit_bridge import build_population, authenticate_population, build_audit_job, admit_queue_reports, digest
from subnet.mimo_assessment_bridge import evidence, merge, compose_loader
from subnet.continuous_audit_policy import snapshot


class AuditBridgeTests(unittest.TestCase):
    def setUp(self):
        def manifest():
            from subnet.mimo_answer_projection import POLICY_KEY, POLICY
            m=mimo_manifest();m[POLICY_KEY]=deepcopy(POLICY);return m
        self.f=worker_fixtures.WorkerIntegrationTests('test_signed_group_dispatch_bypasses_old_backend_quotas_and_artifact_budget')
        with patch('test_mimo_worker.mimo_manifest',side_effect=manifest):self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f=self.f
        from subnet.storage import Identity
        from subnet.mimo_release_policy import build_policy,ReleasePolicy,POLICY_KEY
        policy=build_policy(Identity(f.operator.encode()),f.m,task_assets={'assets/test-tasks.json':{'sha256':'a'*64,'bytes':10}},
                            enabled=True)
        f.m[POLICY_KEY]=policy;self.release=ReleasePolicy(policy,f.authority)
        # The release policy is part of the first opening, before any miner
        # commitment. Never retrofit it onto an already committed group.
        from subnet.complete_group_transport import capability,CompleteGroupTransport
        from subnet.complete_group_admission import signed_commitment
        from subnet.miner import Miner
        from test_complete_group_mimo import MimoCollectionRuntime
        from subnet.mimo_worker import submission_reference
        f.m['capabilities']={f.identity.id:capability(f.bucket,f.m,f.identity.id,registered_miners={f.identity.id})}
        with patch('subnet.model.check_runtime_profile'):
            f.actor=Miner(f.identity,f.m,'fixture',complete_group_transport=CompleteGroupTransport(http=f.http))
        self.addCleanup(f.actor._group_journal.close)
        with patch('subnet.miner.make_runtime',return_value=MimoCollectionRuntime(f.m,f.identity.id)):
            f.batch=f.actor.search(0)
        f.data=f.actor._group_prepared[0][1]
        f.commit=json.loads(signed_commitment(f.identity,f.m,f.actor._group_prepared));f.claim=f.commit['payload']['groups'][0]
        f.obj=submission_reference(f.m,f.identity.id,f.commit,0,f.url)
        f.job['submissions']=[f.obj];f.job['group_commitments']={f.identity.id:f.commit}
        self.opening=sign(f.operator,f.m);f.job['manifest']=self.opening;self.at=f.m['deadline']
        self.guard=self.release.validate_manifest
        self.assertEqual(f.actor.upload(),200)
        b=f.batch;t=b['rollouts'][0]['turns'][0]
        resolver=lambda m,e,i:TaskAuthority(m['epoch'],m['checkpoint']['id'],e,b['environment_version'],i,
            b['task_hash'],tuple(t['prompt']),248046,248320,(248046,248044))
        captured=capture(f.bucket,AdmissionLedger(f.root/'ledger.db'),f.m,f.identity.id,
            registered_miners={f.identity.id},resolve_task=resolver,now=f.bucket.clock+1)
        c=dict(version='mimo-complete-group-coordinator-v1',epoch=f.m['epoch'],miner=f.identity.id,
            manifest_sha256=digest(f.m),**captured)
        self.captures={f.identity.id:sign(f.operator,c)};self.commitments={f.identity.id:f.commit}
        grade=dict(version='complete-group-native-grading-v1',batch_sha256=digest(b),manifest_sha256=digest(f.m),
            miner=f.identity.id,task_hash=b['task_hash'],prompt=t['prompt'],sampling_verified=False,results=[])
        for nonce,roll in enumerate(b['rollouts']):
            turn=roll['turns'][0]
            grade['results'].append(dict(nonce=nonce,classification=roll['classification'],reward=roll['reward'],
                stop_reason=turn['stop_reason'],output_sha256=digest(turn['output']),
                native_classification=roll['classification'],done=True,observations=[],
                prompt=t['prompt'],output=turn['output'],task_hash=b['task_hash']))
        row=dict(miner=f.identity.id,env_id=b['env_id'],index=b['index'],document=captured['documents'][0],
            batch_sha256=digest(b),native_grades=compact_native_grades(grade),assurance='natively-graded-sampling-unaudited')
        population=dict(version='mimo-complete-group-coordinator-v1',epoch=f.m['epoch'],
            manifest_sha256=digest(f.m),captures=list(self.captures.values()),sealed_at=self.at)
        self.dispatch=sign(f.operator,dict(version='mimo-complete-group-coordinator-v1',epoch=f.m['epoch'],
            run_id=f.m['run_id'],checkpoint=f.m['checkpoint']['id'],manifest_sha256=digest(f.m),
            population_sha256=digest(population),selected=[row],excluded=[],training_requires_sampling_audit=False))
        self.regs=sign(f.operator,dict(epoch=f.m['epoch'],snapshot_block=123,registrations={
            'fixture-registered-hotkey':dict(uid=85,public_key=f.identity.id,snapshot_block=123)}))
        self.policy=dict(version='continuous-probabilistic-audit-v2',recent_epochs=16,decay=.9,
            prior_alpha=1.,prior_beta=1.,invalid_multiplier=.1,zero_epoch_after=3,blacklist_after=5,blacklist_epochs=3)
        self.execution={f.m['source_bundle']['sha256']:{k:f.job[k] for k in ('source_files','runtime_versions')}}

    def population(self):
        f=self.f
        return build_population(self.opening,self.captures,self.commitments,self.dispatch,self.regs,
            round=181,committed_at=self.at,authority=f.authority,manifest_guard=self.guard)

    def signed_population(self):return sign(self.f.operator,self.population())

    def queue_evidence(self,invalid=False):
        f=self.f;p=self.population();self.worker=SigningKey.generate();key=self.worker.verify_key.encode().hex()
        self.now=self.at+1
        job=build_audit_job(p,p['records'],authority=f.authority,manifest_guard=self.guard,
            job_id=f.job['job_id'],created_at=f.job['created_at'],expires_at=f.job['expires_at'],
            source_files=f.job['source_files'],runtime_versions=f.job['runtime_versions'],read_url=lambda _:f.url)
        self.assertEqual(job,f.job)
        with patch('time.time',return_value=self.now):report=f.execute()
        if invalid:
            report['audits'][0].update(accepted=[],outcomes=[dict(batch=0,valid=False,fully_audited=False,failure_kind='invalid_sample')])
        queue=Coordinator(f.root/'queue-evidence.db',f.authority,{key:['verify']},clock=lambda:self.now)
        queue.enqueue(sign(f.operator,job))
        claim=queue.request(sign(self.worker,dict(action='claim',role='verify',at=self.now,nonce='a'*32)))['claim']
        request=sign(self.worker,dict(action='report',job_id=job['job_id'],token=claim['token'],report=report,at=self.now,nonce='b'*32))
        queue.request(request)
        with queue.transaction() as db:row=dict(db.execute('SELECT * FROM jobs WHERE id=?',(job['job_id'],)).fetchone())
        self.verifiers={key:['verify']};return row

    def admit(self,rows):
        return admit_queue_reports(rows,[self.signed_population()],authority=self.f.authority,
            verifiers=self.verifiers,approved_execution=self.execution,manifest_guard=self.guard,
            cutoff=int((self.at+3600)//3600)*3600)

    def test_real_complete_group_capture_to_original_queue_signature_and_score(self):
        p=self.population();self.assertEqual(len(p['records']),1);self.assertEqual(len(p['eligible_evidence_ids']),1)
        self.assertEqual(authenticate_population(sign(self.f.operator,p),self.f.authority,manifest_guard=self.guard),p)
        row=self.queue_evidence();admitted,refused=self.admit([row,row])
        self.assertEqual(len(admitted),1);self.assertEqual(refused,[])
        self.assertEqual(next(iter(admitted.values()))['observations'][0]['outcome'],'verified_valid')
        r=evidence([self.signed_population()],[row,row],authority=self.f.authority,
            cutoff=int((self.at+3600)//3600)*3600,verifiers=self.verifiers,approved_execution=self.execution,
            manifest_guard=self.guard,snapshot=snapshot,audit_policy=self.policy,release_policy_sha256='a'*64)
        detail=r['snapshots'][0]['miners'][self.f.identity.id]
        self.assertEqual(detail['unique_eligible_batches'],1)
        self.assertAlmostEqual(detail['validity_probability'],2/3)
        self.assertFalse(r['snapshots'][0]['training_waits_for_audits'])

    def test_training_compute_limit_does_not_erase_valid_reward_contribution(self):
        d=deepcopy(self.dispatch['payload']);r=d['selected'].pop();d['excluded']=[dict(r,reason='training-task-budget')]
        self.dispatch=sign(self.f.operator,d)
        self.assertEqual(len(self.population()['eligible_evidence_ids']),1)

    def test_missing_nonce_document_capture_registration_or_scope_rejected(self):
        for name,mutate in (
            ('dispatch',lambda d:d['selected'][0]['native_grades']['results'].pop()),
            ('dispatch',lambda d:d['selected'][0]['document'].update(sha256='e'*64)),
            ('regs',lambda d:d['registrations']['fixture-registered-hotkey'].update(public_key='e'*64)),
            ('dispatch',lambda d:d.update(population_sha256='e'*64))):
            original=getattr(self,name);value=deepcopy(original['payload']);mutate(value);setattr(self,name,sign(self.f.operator,value))
            with self.subTest(name=name),self.assertRaises((ValueError,AssertionError)):self.population()
            setattr(self,name,original)

    def test_changed_opening_source_or_unsigned_worker_cannot_mint_observation(self):
        row=self.queue_evidence()
        for column,change in [('envelope',lambda x:x['payload']['source_files'].update({'subnet/mimo_worker.py':'e'*64})),
                              ('report_request',lambda x:x['payload']['report'].update(success=False))]:
            bad=deepcopy(row);value=json.loads(bad[column]);change(value);bad[column]=json.dumps(value)
            admitted,refused=self.admit([bad,row]);self.assertEqual(len(admitted),1);self.assertEqual(len(refused),1)
            self.assertFalse(refused[0]['fraud_claim'])

    def test_incomplete_token_coverage_is_not_a_valid_audit(self):
        row=self.queue_evidence();request=json.loads(row['report_request']);r=json.loads(row['report'])
        r['audits'][0]['outcomes'][0]['rollouts'][0]['all_intervals_verified']=False
        row['report']=json.dumps(r);row['report_digest']=digest(r);request['payload']['report']=r
        row['report_request']=json.dumps(sign(self.worker,request['payload']))
        admitted,refused=self.admit([row]);self.assertEqual(admitted,{});self.assertEqual(len(refused),1)

    def test_operator_signature_alone_cannot_approve_an_unlisted_worker_source(self):
        row=self.queue_evidence();job=json.loads(row['envelope'])['payload']
        job['source_files']['subnet/mimo_worker.py']='e'*64
        row['envelope']=json.dumps(sign(self.f.operator,job));row['digest']=digest(job)
        admitted,refused=self.admit([row]);self.assertEqual(admitted,{});self.assertEqual(len(refused),1)


    def test_confirmed_invalid_is_penalized_but_failed_queue_is_not_fraud(self):
        row=self.queue_evidence(invalid=True);admitted,refused=self.admit([row])
        self.assertEqual(next(iter(admitted.values()))['observations'][0]['outcome'],'confirmed_invalid')
        failed=deepcopy(row);failed['status']='failed';admitted,refused=self.admit([failed])
        self.assertEqual(admitted,{});self.assertFalse(refused[0]['fraud_claim'])

    def test_scheduler_persists_selection_and_original_job_across_lost_enqueue_ack(self):
        from subnet.mimo_audit_scheduler import AuditScheduler
        from subnet.storage import Identity
        f=self.f;identity=Identity(f.operator.encode());worker=SigningKey.generate()
        q=Coordinator(f.root/'scheduled.db',f.authority,{worker.verify_key.encode().hex():['verify']},clock=lambda:self.at+1)
        root=f.root/'scheduler'
        def create():return AuditScheduler(root,f.bucket,q,identity,manifest_guard=self.guard,
            metadata=lambda m:self.execution[m['source_bundle']['sha256']],max_inflight=1)
        scheduler=create();scheduler.admit(self.signed_population())
        original=q.enqueue
        def lost(envelope):original(envelope);raise OSError('lost enqueue ACK')
        with patch.object(q,'enqueue',side_effect=lost),self.assertRaises(OSError):scheduler.tick(now=self.at+1)
        state=json.loads((root/'state.json').read_bytes());self.assertEqual(len(state['draws']),1)
        first=next(iter(state['draws'].values()));self.assertEqual(first['row'],self.population()['records'][0])
        result=create().tick(now=self.at+2);self.assertTrue(result['backpressure'])
        with q.transaction() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],1)
        self.assertEqual(create().populations(),[self.signed_population()])

    def test_scheduler_expired_execution_retries_original_group_with_refreshed_job(self):
        from subnet.mimo_audit_scheduler import AuditScheduler
        from subnet.storage import Identity
        f=self.f;clock=[self.at+1];worker=SigningKey.generate()
        q=Coordinator(f.root/'scheduled-retry.db',f.authority,{worker.verify_key.encode().hex():['verify']},clock=lambda:clock[0])
        scheduler=AuditScheduler(f.root/'retry-scheduler',f.bucket,q,Identity(f.operator.encode()),
            manifest_guard=self.guard,metadata=lambda m:self.execution[m['source_bundle']['sha256']],job_seconds=60)
        scheduler.admit(self.signed_population());self.assertEqual(scheduler.tick(now=clock[0])['enqueued'],1)
        clock[0]+=61;self.assertEqual(scheduler.tick(now=clock[0])['enqueued'],1)
        state=scheduler._load();draw=next(iter(state['draws'].values()))
        self.assertEqual(len(state['draws']),1);self.assertEqual(draw['generation'],1);self.assertEqual(len(draw['prior_jobs']),1)
        with q.transaction() as db:
            jobs=[json.loads(r[0])['payload'] for r in db.execute('SELECT envelope FROM jobs ORDER BY rowid')]
        self.assertEqual(jobs[0]['group_commitments'],jobs[1]['group_commitments'])
        self.assertEqual(jobs[0]['manifest'],jobs[1]['manifest']);self.assertNotEqual(jobs[0]['job_id'],jobs[1]['job_id'])

    def test_scheduler_expired_job_not_enqueued_renews_same_original_group(self):
        from subnet.mimo_audit_scheduler import AuditScheduler
        from subnet.storage import Identity
        f=self.f;clock=[self.at+1];worker=SigningKey.generate()
        q=Coordinator(f.root/'missing-job.db',f.authority,{worker.verify_key.encode().hex():['verify']},clock=lambda:clock[0])
        scheduler=AuditScheduler(f.root/'missing-job-scheduler',f.bucket,q,Identity(f.operator.encode()),
            manifest_guard=self.guard,metadata=lambda m:self.execution[m['source_bundle']['sha256']],job_seconds=60)
        scheduler.admit(self.signed_population())
        with patch.object(q,'enqueue',side_effect=OSError('stopped before enqueue')),self.assertRaises(OSError):
            scheduler.tick(now=clock[0])
        before=next(iter(scheduler._load()['draws'].values()))
        self.assertIsNone(scheduler._queue_status(before['job_id']))
        clock[0]+=61
        self.assertEqual(scheduler.tick(now=clock[0])['enqueued'],1)
        state=scheduler._load();after=next(iter(state['draws'].values()))
        self.assertEqual(len(state['draws']),1);self.assertEqual(after['row'],before['row'])
        self.assertEqual(after['generation'],1);self.assertEqual(after['prior_jobs'],[before['job_id']])
        self.assertNotEqual(after['job_id'],before['job_id'])
        self.assertEqual(q.status(after['job_id'])['status'],'queued')

    def test_signed_readonly_source_reconstructs_durable_scheduler_and_rejects_tamper(self):
        from subnet.mimo_audit_scheduler import AuditScheduler
        from subnet.mimo_assessment_bridge import EvidenceSource,VERSION,TRANSITION
        from subnet.storage import Identity
        f=self.f;row=self.queue_evidence();queue=Coordinator(f.root/'queue-evidence.db',f.authority,self.verifiers,clock=lambda:self.at+1)
        directory=f.root/'reader-scheduler'
        scheduler=AuditScheduler(directory,f.bucket,queue,Identity(f.operator.encode()),manifest_guard=self.guard,
            metadata=lambda m:self.execution[m['source_bundle']['sha256']])
        scheduler.admit(self.signed_population())
        state=scheduler._load();record=self.population()['records'][0]
        state['draws'][digest(record)]={'row':record,'job_id':row['id'],'expires_at':self.at+100,'generation':0}
        scheduler._save(state)
        config=sign(f.operator,dict(version=VERSION,directory=str(directory),queue_path=queue.path,
            release_policy=self.release.envelope,approved_execution=self.execution,audit_policy=self.policy,
            verifiers=list(self.verifiers),transition_policy=TRANSITION))
        from subnet.continuous_audit_policy import observations
        def global_reducer(audits,miners,round,policy):
            # Explicit callback fixture: the production composition supplies
            # its unchanged, separately source-pinned global reducer.
            self.assertEqual(len(audits),1);self.assertEqual(round,181)
            self.assertEqual(audits[0]['outcome'],'verified_valid')
            return {m:dict(validity_probability=.4,reward_multiplier=.1,resolution_coverage_factor=1.) for m in miners}
        source=EvidenceSource(config,f.authority,snapshot=snapshot,current_estimates=global_reducer,observation_reader=observations)
        cutoff=int((self.at+3600)//3600)*3600
        got=source(authority=f.authority,cutoff=cutoff,verifiers=list(self.verifiers))
        self.assertEqual(got['snapshots'][0]['miners'][f.identity.id]['unique_eligible_batches'],1)
        self.assertAlmostEqual(got['snapshots'][0]['miners'][f.identity.id]['validity_probability'],.4)
        self.assertAlmostEqual(got['snapshots'][0]['miners'][f.identity.id]['reward_multiplier'],.1)
        self.assertEqual(got['evidence_hashes']['mimo_reader_policy_sha256'],digest(config))
        # Admit a source-only successor without rewriting original populations
        # or old queue jobs; the sole writer keeps the same historical credit.
        from subnet.mimo_release_policy import build_policy
        from subnet.mimo_source_migration import SourceReleaseRegistry,REGISTRY
        newer=deepcopy(f.m);newer['source_bundle']['sha256']='9'*64
        new=build_policy(Identity(f.operator.encode()),newer,task_assets=self.release.scope['task_assets'],enabled=True)
        profiles=deepcopy(self.execution);changed=deepcopy(next(iter(profiles.values())))
        changed['source_files']['subnet/mimo_grpo/transaction.py']='e'*64;profiles['9'*64]=changed
        old_files=next(iter(self.execution.values()))['source_files']
        changes=[k for k in set(old_files)|set(changed['source_files']) if old_files.get(k)!=changed['source_files'].get(k)]
        registry=sign(f.operator,dict(version=REGISTRY,policies=[self.release.envelope,new],active_policy_sha256=digest(new),
            execution=profiles,source_changes=changes,trainer_config={'fixture':'same'},trainer_identity=f.identity.id))
        cfg=dict(config['payload'],release_policy=new,source_release_registry=registry,approved_execution=profiles)
        migrated=EvidenceSource(sign(f.operator,cfg),f.authority,snapshot=snapshot,current_estimates=global_reducer,observation_reader=observations)
        recovered=migrated(authority=f.authority,cutoff=cutoff,verifiers=list(self.verifiers))
        self.assertEqual(recovered['snapshots'],got['snapshots'])
        self.assertEqual(recovered['committed_at_by_epoch'],got['committed_at_by_epoch'])
        self.assertEqual(recovered['evidence_hashes']['mimo_admitted_jobs'],got['evidence_hashes']['mimo_admitted_jobs'])
        # Missing original source execution is rejected before the reader can
        # hand an empty or reset assessment to the existing hourly writer.
        cfg['approved_execution']={'9'*64:changed}
        with self.assertRaisesRegex(ValueError,'exact admitted source registry'):
            EvidenceSource(sign(f.operator,cfg),f.authority,snapshot=snapshot,current_estimates=global_reducer,observation_reader=observations)
        path=next(directory.glob('*.population.json'));path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'population file digest'):
            source(authority=f.authority,cutoff=cutoff,verifiers=list(self.verifiers))


class AssessmentTransitionTests(unittest.TestCase):
    def data(self,epoch,round,probability,multiplier):
        d=dict(unique_eligible_batches=3,validity_probability=probability,reward_multiplier=multiplier,blacklisted=False)
        return dict(snapshots=[dict(epoch=epoch,round=round,cutoff=7200,miners={'miner':d})],
            committed_at_by_epoch={epoch:3600},evidence_hashes={epoch:'a'*64})
    def test_old_history_and_penalty_survive_model_switch_without_chain_calls(self):
        old=self.data('old',180,.2,.1);new=self.data('new',181,.9,1.)
        merged=merge(old,new)
        self.assertEqual(len(merged['snapshots']),2)
        for s in merged['snapshots']:
            self.assertEqual(s['miners']['miner']['reward_multiplier'],.1)
            self.assertEqual(s['miners']['miner']['validity_probability'],.2)
        self.assertEqual(new['snapshots'][0]['miners']['miner']['validity_probability'],.9)
    def test_new_unavailable_reader_raises_into_existing_last_best_fallback(self):
        old=self.data('old',180,.5,1.)
        def unavailable(**kw):raise OSError('registry unavailable')
        load=compose_loader(lambda *a,**k:old,unavailable)
        with self.assertRaises(OSError):load('unchanged',authority='a'*64,cutoff=7200,verifiers=[])
        with self.assertRaisesRegex(ValueError,'historical epoch'):merge(old,self.data('old',181,1.,1.))
        with self.assertRaisesRegex(ValueError,'global epoch order'):merge(old,self.data('new',1,1.,1.))


class PopulationMembershipIndexTests(unittest.TestCase):
    def test_index_preserves_whole_job_membership_and_avoids_repeated_population_scan(self):
        from subnet.mimo_assessment_bridge import _admissions_for_population
        reads=[]
        class Row(dict):
            def __getitem__(self,key):
                reads.append(key)
                return super().__getitem__(key)
        records=[Row(epoch='epoch-'+str(i//100),miner='miner-'+str(i%100),batch_sha256=str(i))
                 for i in range(2000)]
        def observation(i):return dict(records[i])
        admissions={str(i):dict(observations=[observation(i),observation(i+1)])for i in range(1000)}
        admissions['future']=dict(observations=[observation(0),dict(epoch='future',miner='miner-0',batch_sha256='0')])
        admissions['other-miner']=dict(observations=[dict(observation(0),miner='unknown-miner')])
        admissions['other-batch']=dict(observations=[dict(observation(0),batch_sha256='not-committed')])
        # Compare exact prior semantics, including rejecting a whole job when
        # just one observation lies outside this original population.
        plain=[dict(row)for row in records]
        expected={k:v for k,v in admissions.items() if all(any(
            all(row[f]==o[f]for f in ('epoch','miner','batch_sha256'))for row in plain)
            for o in v['observations'])}
        reads.clear();actual=_admissions_for_population(admissions,records)
        self.assertEqual(actual,expected)
        self.assertEqual(set(actual),{str(i)for i in range(1000)})
        self.assertEqual(len(reads),3*len(records))
        self.assertTrue(all(actual[k]is admissions[k]for k in actual))


if __name__=='__main__':unittest.main()
