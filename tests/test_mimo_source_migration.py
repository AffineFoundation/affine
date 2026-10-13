"""CPU-only source migration fixtures; these do not qualify a public deployment."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_mimo_public_worker import PublicWorkerTests
from subnet.mimo_release_policy import (ReleasePolicy, build_policy, policy_scope, sign, digest,
    QUALIFICATIONS, QUALIFICATION_VERSION, POLICY_KEY)
from subnet.mimo_source_migration import (REGISTRY, SourceReleaseRegistry, prepare_transition,
    validate_transition, planned_current, apply_role_current, validate_ack)
from subnet.storage import canonical


class SourceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.f=PublicWorkerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.operator=self.f.operator;self.trainer=self.f.f.identity
        self.old=ReleasePolicy(self.f.policy,self.operator.id,allow_public=True)
        self.m=deepcopy(self.f.m);self.m['source_bundle']['sha256']='c'*64
        scope=policy_scope(self.m,self.f.assets)
        q=[sign(self.operator,dict(version=QUALIFICATION_VERSION,scope_sha256=digest(scope),
             kind=k,passed=True,evidence_sha256=digest(['synthetic source migration fixture',k])))for k in sorted(QUALIFICATIONS)]
        self.new=build_policy(self.operator,self.m,task_assets=self.f.assets,mode='public',enabled=True,qualifications=q)
        self.m[POLICY_KEY]=self.new
        self.config={'learning_rate':1e-6,'objective':'fixture-GRPO'}
        self.profile=lambda h:dict(source_files={'subnet/mimo_worker.py':'b'*64,'subnet/mimo_grpo/transaction.py':h},runtime_versions={k:'fixture' for k in ('torch','transformers','toploc')})
        self.payload=dict(version=REGISTRY,policies=[self.f.policy,self.new],active_policy_sha256=digest(self.new),
            execution={self.f.m['source_bundle']['sha256']:self.profile('a'*64),'c'*64:self.profile('c'*64)},
            source_changes=['subnet/mimo_grpo/transaction.py'],trainer_config=self.config,trainer_identity=self.trainer.id)
        self.registry=self.registry_for(self.payload)
        cp={'files':dict(self.f.m['checkpoint']['files'],**{'config.json':'d'*64})};cp['id']=digest(cp['files'])
        self.cp=cp
        cert=self.old.authorize_checkpoint(cp,input_checkpoint=self.f.m['checkpoint']['id'],optimizer_step=2,
                                          publication_receipt_sha256='f'*64,identity=self.operator)
        result=dict(status='updated',checkpoint=cp,optimizer_step=2,checkpoint_authorization=cert,
                    publication_receipt_sha256='f'*64)
        self.opening=sign(self.operator,self.f.m)
        self.final=sign(self.operator,dict(opening_sha256=digest(self.opening),policy_sha256=self.old.policy_sha256,epoch=self.f.m['epoch'],
            round=190,input_checkpoint=self.f.m['checkpoint']['id'],result=result))
        self.host=dict(policy_sha256=self.old.policy_sha256,checkpoint=cp,checkpoint_authorization=cert,
            optimizer_step=2,round=191,epoch_index=2,collection_seconds=1800,max_tasks=32)
        report={'config':self.config,'original':'preserved'}
        self.transaction=dict(state='published',job=dict(job_id='original-job',epoch=self.f.m['epoch'],optimizer_step_before=1,input_checkpoint=self.f.m['checkpoint']['id'],source_closure_sha256=digest(self.profile('a'*64)['source_files'])),
                              output_checkpoint=cp['id'],report=report,report_sha256=digest(report))
        self.current=dict(release_policy_sha256=self.old.policy_sha256,run=self.old.run_id,
            source_closure_sha256=digest(self.profile('a'*64)['source_files']),checkpoint=cp['id'],descriptor=cp,
            optimizer_step=2,epoch=self.f.m['epoch'],input_checkpoint=self.f.m['checkpoint']['id'],
            publication_receipt_sha256='f'*64,job_id='original-job',report_sha256=digest(report))
        self.host_bytes=json.dumps(self.host,indent=2).encode()+b'\n'
        self.trainer_bytes=json.dumps(self.current,indent=2).encode()+b'\n'
        self.transition=self.prepare()
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def registry_for(self,payload):
        return SourceReleaseRegistry(sign(self.operator,payload),self.operator.id,allow_public=True)

    def prepare(self,**kw):
        args=dict(host_bytes=self.host_bytes,trainer_bytes=self.trainer_bytes,
                  original_transaction=self.transaction,identity=self.operator);args.update(kw)
        return prepare_transition(self.registry,self.opening,self.final,**args)

    def role(self,role,**kw):
        path=self.root/(role+'-CURRENT.json')
        if not path.exists():path.write_bytes(self.host_bytes if role=='host' else self.trainer_bytes)
        current=planned_current(self.registry,self.transition,role,identity=self.operator)
        planned=sign(self.operator,dict(transition_sha256=digest(self.transition),role=role,current=current))
        args=dict(path=path,journal_directory=self.root/(role+'-migration'),
                  identity=self.operator if role=='host' else self.trainer,
                  assert_quiescent=lambda:None,locked=nullcontext);args.update(kw)
        return apply_role_current(self.registry,self.transition,role,planned,**args)

    def test_exact_old_and_new_execution_profiles_retained(self):
        for manifest in (self.f.m,self.m):
            profile=self.registry.execution_for(manifest)
            job=dict(manifest=sign(self.operator,manifest),**profile)
            self.registry.validate_job_execution(job)
        job.update(self.profile('b'*64))
        with self.assertRaisesRegex(ValueError,'original executable'):self.registry.validate_job_execution(job)
        unsigned=deepcopy(self.m);unsigned[POLICY_KEY]=deepcopy(self.f.policy)
        with self.assertRaises(ValueError):self.registry.resolve(unsigned)

    def test_runtime_source_allowlist_and_objective_are_explicit(self):
        for field in ('runtime','file','policy','empty-config'):
            p=deepcopy(self.payload)
            if field=='runtime':p['execution']['c'*64]['runtime_versions']['torch']='different'
            elif field=='file':p['source_changes']=[]
            elif field=='policy':p['active_policy_sha256']='e'*64
            else:p['trainer_config']={}
            with self.subTest(field=field),self.assertRaises(ValueError):self.registry_for(p)

    def test_different_model_sampler_or_grader_policy_not_source_only(self):
        for key in ('harness_source_hash','sampling_source_hash','environments'):
            p=deepcopy(self.payload);changed=deepcopy(p['policies'][1]['payload'])
            changed['scope']['immutable_manifest'][key]='e'*64
            changed['scope_sha256']=digest(changed['scope'])
            for row in changed['qualifications']:
                row['payload']['scope_sha256']=changed['scope_sha256']
            changed['qualifications']=[sign(self.operator,row['payload'])for row in changed['qualifications']]
            p['policies'][1]=sign(self.operator,changed);p['active_policy_sha256']=digest(p['policies'][1])
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'source-only'):self.registry_for(p)

    def test_changed_step_or_optimizer_cannot_migrate(self):
        trainer=deepcopy(self.current);trainer['optimizer_step']=3
        with self.assertRaises(ValueError):self.prepare(trainer_bytes=canonical(trainer))
        transaction=deepcopy(self.transaction);transaction['report']['config']={'different':'objective'}
        with self.assertRaises(ValueError):self.prepare(original_transaction=transaction)
        host=deepcopy(self.host);host['round']=192
        with self.assertRaises(ValueError):self.prepare(host_bytes=canonical(host))

    def test_original_bytes_and_generation_preserved_host_requires_trainer_ack(self):
        with self.assertRaisesRegex(ValueError,'trainer acknowledgement'):self.role('host')
        ack=self.role('trainer');self.role('host',trainer_ack=ack)
        trainer=json.loads((self.root/'trainer-CURRENT.json').read_bytes())
        self.assertEqual(trainer['optimizer_step'],2);self.assertEqual(trainer['job_id'],'original-job')
        record=json.loads((self.root/'trainer-migration/PREPARED.json').read_bytes())
        self.assertEqual(record['transition'],self.transition)
        self.assertEqual(self.role('trainer'),ack)
        self.assertEqual(self.role('host',trainer_ack=ack),self.role('host',trainer_ack=ack))
        validate_transition(self.registry,self.transition)

    def test_current_semantically_same_but_bytes_changed_rejected_before_prepare(self):
        (self.root/'trainer-CURRENT.json').write_bytes(canonical(self.current))
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.role('trainer')

    def test_quiescence_required_and_no_writes_on_active_work(self):
        def active():raise ValueError('active training or issued epoch')
        with self.assertRaisesRegex(ValueError,'active training'):self.role('trainer',assert_quiescent=active)
        self.assertFalse((self.root/'trainer-migration/PREPARED.json').exists())
        self.assertEqual((self.root/'trainer-CURRENT.json').read_bytes(),self.trainer_bytes)

    def test_crash_after_every_durable_write_resumes_once(self):
        from subnet.mimo_grpo.transaction import _write
        for fail_after in (1,2,3):
            with self.subTest(fail_after=fail_after),tempfile.TemporaryDirectory() as directory:
                self.root=Path(directory);calls=[]
                def crash(path,value):
                    _write(path,value);calls.append(path)
                    if len(calls)==fail_after:raise RuntimeError('crash')
                with patch('subnet.mimo_source_migration._write',side_effect=crash),self.assertRaises(RuntimeError):self.role('trainer')
                ack=self.role('trainer');self.assertEqual(ack,self.role('trainer'))
                self.assertEqual(json.loads((self.root/'trainer-CURRENT.json').read_bytes())['optimizer_step'],2)

    def test_host_crashes_resume_after_same_trainer_ack(self):
        from subnet.mimo_grpo.transaction import _write
        for fail_after in (1,2,3):
            with self.subTest(fail_after=fail_after),tempfile.TemporaryDirectory() as directory:
                self.root=Path(directory);ack=self.role('trainer');calls=[]
                def crash(path,value):
                    _write(path,value);calls.append(path)
                    if len(calls)==fail_after:raise RuntimeError('crash')
                with patch('subnet.mimo_source_migration._write',side_effect=crash),self.assertRaises(RuntimeError):
                    self.role('host',trainer_ack=ack)
                result=self.role('host',trainer_ack=ack)
                self.assertEqual(result,self.role('host',trainer_ack=ack))
                self.assertEqual(json.loads((self.root/'host-CURRENT.json').read_bytes())['round'],191)

    def test_signed_target_cannot_change_optimizer_or_round(self):
        for role,field in (('trainer','optimizer_step'),('host','round')):
            with self.subTest(role=role):
                target=planned_current(self.registry,self.transition,role,identity=self.operator)
                target[field]+=1
                planned=sign(self.operator,dict(transition_sha256=digest(self.transition),role=role,current=target))
                path=self.root/(role+'-CURRENT.json');path.write_bytes(self.trainer_bytes if role=='trainer' else self.host_bytes)
                with self.assertRaisesRegex(ValueError,'preserve every'):
                    apply_role_current(self.registry,self.transition,role,planned,path=path,
                        journal_directory=self.root/(role+'-migration'),identity=self.trainer if role=='trainer' else self.operator,
                        assert_quiescent=lambda:None,locked=nullcontext,trainer_ack=self.role('trainer') if role=='host' else None)

    def test_queue_old_job_waits_for_old_executable_without_consuming_attempt(self):
        from subnet.distributed_roles import Coordinator
        from subnet.storage import Identity
        other=Identity(bytes.fromhex('42'*32))
        workers={self.trainer.id:['verify'],other.id:['verify']}
        queue=Coordinator(self.root/'queue.sqlite3',self.operator.id,workers,
            allow_public_mimo=True,source_registry=self.registry,
            worker_execution={self.trainer.id:'c'*64,other.id:self.f.m['source_bundle']['sha256']})
        job=deepcopy(self.f.job);job.update(self.registry.execution_for(self.f.m))
        envelope=sign(self.operator,job)
        queue.enqueue(envelope)
        import time
        def claim(identity,nonce):
            return queue.request(sign(identity,dict(action='claim',role='verify',at=time.time(),nonce=nonce)))
        self.assertIsNone(claim(self.trainer,'a'*32)['claim'])
        with queue.transaction() as db:
            row=db.execute('SELECT status,attempt FROM jobs WHERE id=?',(job['job_id'],)).fetchone()
            self.assertEqual(tuple(row),('queued',0))
        self.assertEqual(claim(other,'b'*32)['claim']['job'],envelope)

    def test_stale_ack_or_different_state_rejected(self):
        ack=self.role('trainer');bad=deepcopy(ack['payload']);bad['transition_sha256']='e'*64
        with self.assertRaises(ValueError):self.role('host',trainer_ack=sign(self.trainer,bad))
        path=self.root/'trainer-CURRENT.json';state=json.loads(path.read_bytes());state['optimizer_step']=3;path.write_bytes(canonical(state))
        with self.assertRaisesRegex(ValueError,'CURRENT changed'):self.role('trainer')


if __name__=='__main__':unittest.main()
