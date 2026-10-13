"""Real capture/native selection/history with explicit CPU model/queue fixtures.

These tests do not certify a GPU release. Existing trainer-worker tests exercise
the actual optimizer/publication; here the signed worker result is a fixture.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_complete_group_fixtures import fixture,FixtureRuntime
from test_complete_group_transport import FakeBucket,FakeHTTP
from test_mimo_coordinator import NativeSession,NativeTokenizer
from subnet.storage import Identity
from subnet.mimo_release_policy import build_policy,sign,digest
from subnet.mimo_lifecycle import Lifecycle,Services
from subnet.mimo_lifecycle import TrainingRecoveryRequired
from subnet.complete_group_native import resolve_task,grade_batch
from subnet.complete_group_transport import CompleteGroupTransport
from subnet.miner import Miner


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.bucket=FakeBucket();self.http=FakeHTTP(self.bucket)
        self.authority=Identity(bytes([42])*32);self.miner=Identity(bytes([1])*32)
        self.worker=Identity(bytes([2])*32);self.publisher=Identity(bytes([3])*32)
        self.now=int(self.bucket.clock)-1;self.committed=[];self.discovery_value=None;self.discovery_calls=[]
        self.actors=[];self.addCleanup(lambda:[a._group_journal.close()for a in self.actors])
        self.tokenizer=NativeTokenizer()
        p=patch('subnet.complete_group_native.create_session',side_effect=lambda spec:NativeSession());p.start();self.addCleanup(p.stop)
        p=patch('subnet.complete_group_native.EnvironmentSpec.from_dict',side_effect=lambda v:SimpleNamespace(**v));p.start();self.addCleanup(p.stop)
        m=fixture();m.update(run_id='lifecycle-fixture',payable=False,mimo_runtime={'fixture_only':True},mimo_answer_projection=None)
        m['checkpoint']={'files':{'config.json':'a'*64,'model.safetensors':'b'*64}}
        m['checkpoint']['id']=digest(m['checkpoint']['files'])
        m['sampling_contract']['calibration']['checkpoint']=m['checkpoint']['id']
        self.base=m['checkpoint'];self.policy=build_policy(self.authority,m,
            task_assets={'assets/test.tasks.json':{'sha256':'c'*64,'bytes':1}},enabled=True)
        self.jobs={};self.result=None;self.readbacks=0
        self.services=Services(self.registrations,lambda m:list(self.committed),self.discovery,
            lambda cp:{n:self.bucket.presign('models/'+cp['id']+'/'+n,'get_object',1000)for n in cp['files']},
            self.make_job,self.enqueue,lambda job:self.result,self.readback)
        self.lifecycle=self.make()

    def registrations(self):return dict(snapshot_block=123,registrations={'registered-miner':dict(uid=85,public_key=self.miner.id,snapshot_block=123)})
    def discovery(self,opening,previous):
        actual=None if self.discovery_value is None else digest(self.discovery_value)
        if actual!=digest(opening):self.assertEqual(actual,previous)
        self.discovery_value=opening;self.discovery_calls.append(digest(opening))
        return dict(sha256=digest(opening),discovery_sha256=digest(opening))
    def make(self,services=None):return Lifecycle(self.root/'run',self.bucket,self.authority,self.policy,
        services or self.services,resolve_task=lambda m,e,i:resolve_task(m,e,i,self.tokenizer),
        grade_batch=lambda b,m,miner:grade_batch(b,m,self.tokenizer,miner=miner),collection_seconds=120,first_round=200)
    def make_job(self,opening,dispatch,step,namespace):
        return sign(self.authority,dict(role='train',job_id='fixture-job-'+opening['payload']['epoch'],
            manifest=opening,trainer_worker=self.worker.id,training=dict(optimizer_step=step,
                publication_namespace=namespace,publisher_id=self.publisher.id,dispatch={'sha256':digest(dispatch)})))
    def enqueue(self,envelope):
        key=envelope['payload']['job_id']
        if key in self.jobs:self.assertEqual(self.jobs[key],envelope)
        self.jobs[key]=envelope;return key
    def readback(self,published):self.readbacks+=1;return published['descriptor']
    def upload(self):
        m=self.discovery_value['payload']
        with patch('subnet.model.check_runtime_profile'):
            actor=Miner(self.miner,m,'fixture-only',complete_group_transport=CompleteGroupTransport(http=self.http))
        self.actors.append(actor)
        runtime=FixtureRuntime(m,self.miner.id)
        runtime.mutate=lambda r,row:row['turns'][0].update(text=self.tokenizer.decode(row['turns'][0]['output']))
        with patch('subnet.miner.make_runtime',return_value=runtime):actor.search(0,env_id='math')
        self.assertEqual(actor.upload(),200);self.committed=[self.miner.id]

    def test_empty_epoch_publishes_audit_population_then_next_epoch_without_fake_step(self):
        first=self.lifecycle.tick(now=self.now);self.assertEqual(first['status'],'collecting')
        first_opening=deepcopy(self.discovery_value)
        result=self.lifecycle.tick(now=self.now+120)
        self.assertEqual(result['training'],'no-update');self.assertEqual(self.jobs,{})
        current=json.loads((self.root/'run/CURRENT.json').read_bytes())
        self.assertEqual((current['round'],current['optimizer_step']),(201,0))
        self.assertEqual(current['checkpoint'],self.base)
        self.lifecycle.tick(now=self.now+121)
        self.assertNotEqual(first_opening['payload']['sampling_contract']['randomness'],
                            self.discovery_value['payload']['sampling_contract']['randomness'])
        self.assertEqual(first_opening['payload']['checkpoint'],self.discovery_value['payload']['checkpoint'])

    def test_lost_discovery_ack_retries_exact_opening_and_registered_capabilities(self):
        calls=0
        def lose(opening,previous):
            nonlocal calls
            result=self.discovery(opening,previous);calls+=1
            if calls==1:raise OSError('reply lost')
            return result
        lifecycle=self.make(replace(self.services,discovery=lose))
        with self.assertRaises(OSError):lifecycle.tick(now=self.now)
        original=deepcopy(self.discovery_value)
        self.assertEqual(self.make().tick(now=self.now+1)['status'],'collecting')
        self.assertEqual(self.discovery_value,original)
        self.assertEqual(set(original['payload']['capabilities']),{self.miner.id})

    def test_listing_infrastructure_error_cannot_freeze_partial_population(self):
        self.lifecycle.tick(now=self.now);self.upload()
        bad=self.make(replace(self.services,committed_miners=lambda m:(_ for _ in ()).throw(OSError('R2 outage'))))
        with self.assertRaises(OSError):bad.tick(now=self.now+120)
        self.assertFalse((self.root/'run/epochs/0/coordinator/POPULATION.json').exists())
        self.assertEqual(json.loads((self.root/'run/CURRENT.json').read_bytes())['checkpoint'],self.base)

    def test_crash_after_plan_before_opening_expires_without_permanent_stall(self):
        current=self.lifecycle._current();directory=self.root/'run/epochs/0';directory.mkdir(parents=True)
        self.lifecycle._plan(current,directory,self.now)
        result=self.lifecycle.tick(now=self.now+121)
        self.assertEqual(result['status'],'abandoned-before-opening')
        following=json.loads((self.root/'run/CURRENT.json').read_bytes())
        self.assertEqual(following,dict(current,epoch_index=1))
        self.assertIsNone(self.discovery_value)
        self.assertEqual(self.lifecycle.tick(now=self.now+122)['status'],'collecting')
        self.assertEqual(self.discovery_value['payload']['checkpoint']['id'],self.base['id'])

    def test_real_capture_dispatch_is_independent_of_audits_and_waits_for_signed_training(self):
        self.lifecycle.tick(now=self.now);self.upload()
        with (patch('subnet.mimo_training_worker.validate_job') as validate,
              patch('subnet.complete_group_verification.verify_rollout',side_effect=AssertionError('audit barrier'))):
            result=self.lifecycle.tick(now=self.now+120)
        self.assertEqual(result['status'],'training');validate.assert_called_once()
        self.assertEqual(len(self.jobs),1)
        population=json.loads((self.root/'run/epochs/0/AUDIT-POPULATION.json').read_bytes())['payload']
        self.assertEqual(len(population['eligible_evidence_ids']),1)
        self.assertFalse(population['training_requires_sampling_audit'])
        with patch('subnet.mimo_training_worker.validate_job'):
            self.assertEqual(self.make().tick(now=self.now+121)['status'],'training')
        self.assertEqual(len(self.jobs),1)

    def test_training_result_needs_authorized_signer_and_model_readback_before_advance(self):
        self.lifecycle.tick(now=self.now);self.upload()
        with patch('subnet.mimo_training_worker.validate_job'):
            self.lifecycle.tick(now=self.now+120)
        job=next(iter(self.jobs.values()))['payload']
        output={'files':dict(self.base['files'],**{'model.safetensors':'d'*64})};output['id']=digest(output['files'])
        receipt=sign(self.publisher,dict(descriptor=output))
        request=dict(action='report',job_id=job['job_id'],report=dict(training=dict(status='updated',optimizer_step=1),publication_receipt=receipt))
        self.result=sign(self.miner,request)
        with patch('subnet.mimo_training_worker.validate_job'),patch('subnet.mimo_worker.validate_report'):
            with self.assertRaises(ValueError):self.lifecycle.tick(now=self.now+121)
        self.result=sign(self.worker,request)
        with patch('subnet.mimo_training_worker.validate_job'),patch('subnet.mimo_worker.validate_report') as validated:
            result=self.lifecycle.tick(now=self.now+122)
        validated.assert_called_once();self.assertEqual(self.readbacks,1)
        self.assertEqual(result['checkpoint'],output['id'])
        current=json.loads((self.root/'run/CURRENT.json').read_bytes())
        self.assertEqual(current['optimizer_step'],1)
        self.assertEqual(current['checkpoint_authorization']['payload']['publication_receipt_sha256'],digest(receipt))
        self.committed=[];self.lifecycle.tick(now=self.now+123)
        self.assertEqual(self.discovery_value['payload']['checkpoint']['id'],output['id'])

    def _report_clock_fixture(self):
        # Real worker report binding/deadline validation, with only GPU result
        # contents stubbed. Queue receipt occurs after a deliberately old tick.
        from subnet.mimo_worker import VERSION as WORKER_VERSION
        def full_job(opening,dispatch,step,namespace):
            job=self.make_job(opening,dispatch,step,namespace)['payload']
            job.update(source_files={},runtime_versions={},created_at=self.now,
                       expires_at=self.now+2000)
            return sign(self.authority,job)
        lifecycle=self.make(replace(self.services,make_training_job=full_job))
        lifecycle.tick(now=self.now);self.upload()
        with patch('subnet.mimo_training_worker.validate_job'):
            lifecycle.tick(now=self.now+120)
        job=next(iter(self.jobs.values()))['payload'];manifest=job['manifest']['payload']
        output={'files':dict(self.base['files'],**{'model.safetensors':'d'*64})}
        output['id']=digest(output['files'])
        report=dict(job_id=job['job_id'],job_sha256=digest(job),operator=self.authority.id,
            worker_policy=WORKER_VERSION,role='train',epoch=manifest['epoch'],
            checkpoint=manifest['checkpoint']['id'],source_files={},runtime_versions={},
            backend_profile=manifest['backend_profile'],numerical_policy=manifest['sampling_contract']['calibration'],
            chain_transactions=False,success=True,completed_at=self.now+131,
            training=dict(status='updated',optimizer_step=1),
            publication_receipt=sign(self.publisher,dict(descriptor=output)))
        return lifecycle,job,report,output

    def test_report_completed_during_slow_tick_uses_receipt_time(self):
        lifecycle,job,report,output=self._report_clock_fixture()
        self.result=sign(self.worker,dict(action='report',job_id=job['job_id'],report=report))
        with (patch('subnet.mimo_training_worker.validate_job'),
              patch('subnet.mimo_training_worker.validate_result'),
              patch('subnet.mimo_lifecycle.time.time',return_value=self.now+132)):
            result=lifecycle.tick(now=self.now+121)
        self.assertEqual(result['status'],'finalized')
        self.assertEqual(result['checkpoint'],output['id'])
        self.assertEqual(len(self.jobs),1)

    def test_fresh_receipt_clock_still_rejects_future_completion(self):
        lifecycle,job,report,_=self._report_clock_fixture()
        report['completed_at']=self.now+138
        self.result=sign(self.worker,dict(action='report',job_id=job['job_id'],report=report))
        with (patch('subnet.mimo_training_worker.validate_job'),
              patch('subnet.mimo_training_worker.validate_result') as training_validation,
              patch('subnet.mimo_lifecycle.time.time',return_value=self.now+132)):
            with self.assertRaisesRegex(ValueError,'report signed deadline'):
                lifecycle.tick(now=self.now+121)
        training_validation.assert_not_called()
        self.assertEqual(self.readbacks,0)

    def test_fresh_receipt_clock_preserves_original_job_expiry(self):
        lifecycle,job,report,_=self._report_clock_fixture()
        report['completed_at']=job['expires_at']
        self.result=sign(self.worker,dict(action='report',job_id=job['job_id'],report=report))
        with (patch('subnet.mimo_training_worker.validate_job'),
              patch('subnet.mimo_training_worker.validate_result') as training_validation,
              patch('subnet.mimo_lifecycle.time.time',return_value=job['expires_at']+1)):
            with self.assertRaisesRegex(ValueError,'report signed deadline'):
                lifecycle.tick(now=self.now+121)
        training_validation.assert_not_called()
        self.assertEqual(self.readbacks,0)

    def test_terminal_training_failure_is_recovery_required_and_preserves_checkpoint(self):
        self.lifecycle.tick(now=self.now);self.upload()
        def failed(envelope):raise TrainingRecoveryRequired(envelope['payload']['job_id'],'failed')
        lifecycle=self.make(replace(self.services,enqueue_training=failed))
        with patch('subnet.mimo_training_worker.validate_job'):
            status=lifecycle.tick(now=self.now+120)
        self.assertEqual(status['status'],'recovery-required');self.assertEqual(status['queue_status'],'failed')
        current=json.loads((self.root/'run/CURRENT.json').read_bytes())
        self.assertEqual(current['checkpoint'],self.base);self.assertEqual(current['optimizer_step'],0)
        self.assertTrue((self.root/'run/epochs/0/TRAINING-JOB.json').exists())
        self.assertFalse((self.root/'run/epochs/0/FINALIZED.json').exists())


if __name__=='__main__':unittest.main()
