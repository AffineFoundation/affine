"""Real signed queue/evaluator loop using explicit CPU scripted model fixtures."""
from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import version
import json
import os
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

from test_mimo_evaluation import EvaluationTests, Runtime
from test_complete_group_transport import FakeBucket
from subnet.storage import Identity
from subnet.mimo_release_policy import build_policy, ReleasePolicy, sign, POLICY_KEY
from subnet.mimo_evaluation_scheduler import EvaluationScheduler
from subnet.mimo_worker import source_inventory
from subnet.distributed_roles import Coordinator
from subnet.distributed_worker import Worker


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.f=EvaluationTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.identity=Identity(bytes([43])*32);self.worker=Identity(bytes([44])*32)
        self.m=deepcopy(self.f.m);self.m['evaluation_sets'][0]['indices']=[4]
        policy=build_policy(self.identity,self.m,task_assets={'assets/test.json':{'sha256':'a'*64,'bytes':1}},enabled=True)
        self.m[POLICY_KEY]=policy;self.policy=ReleasePolicy(policy,self.identity.id)
        self.opening=sign(self.identity,self.m);self.bucket=FakeBucket()
        self.now=time.time();self.counter=iter(range(1000))
        self.queue=Coordinator(self.f.root/'queue.sqlite',self.identity.id,{self.worker.id:['evaluate']},clock=lambda:self.now)
        self.scheduler=EvaluationScheduler(self.f.root/'eval',self.bucket,self.queue,self.identity,
            policy=self.policy,metadata=lambda m:dict(source_files=source_inventory(),
                runtime_versions={p:version(p)for p in ('torch','transformers','toploc')}),
            workers=[self.worker.id],set_name='fixed4',job_seconds=120,max_retries=1)

    def run_worker(self, runtime, name='worker'):
        class Factory:
            @contextmanager
            def runtime_for_job(self,*a,**kw):yield SimpleNamespace(backend=runtime)
            def close(self):pass
        actor=Worker('http://localhost:99',self.worker.key.encode(),self.identity.id,self.f.root/name,
                     role='evaluate',mimo_factory=Factory())
        self.addCleanup(actor.close)
        actor.request=lambda action,**kw:self.queue.request(sign(self.worker,
            dict(action=action,at=self.now,nonce=str(next(self.counter)).zfill(32),**kw)))
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):self.assertTrue(actor.once())

    def test_actual_queue_report_becomes_immutable_receipt_without_epoch_wait(self):
        result=self.scheduler.tick([self.opening],now=self.now)
        self.assertEqual(result['status'],'enqueued')
        self.assertEqual(self.scheduler.tick([self.opening],now=self.now)['status'],'evaluating')
        self.run_worker(Runtime(self.m))
        self.assertEqual(self.scheduler.tick([self.opening],now=time.time())['status'],'caught-up')
        receipts=list(self.scheduler.root.glob('*.receipt.json'));self.assertEqual(len(receipts),1)
        report=json.loads(receipts[0].read_bytes())['payload']['worker_report']['payload']['report']
        self.assertEqual(report['evaluation']['mean_reward'],1.)
        self.assertFalse(report['chain_transactions'])

    def test_source_registry_preserves_old_pending_report_and_routes_new_job(self):
        from subnet.mimo_source_migration import SourceReleaseRegistry, REGISTRY
        from subnet.mimo_release_policy import digest, CHECKPOINT_KEY
        self.m['source_bundle']={'sha256':'a'*64}
        assets=self.policy.scope['task_assets']
        old=build_policy(self.identity,self.m,task_assets=assets,enabled=True)
        self.m[POLICY_KEY]=old;self.policy=ReleasePolicy(old,self.identity.id)
        self.opening=sign(self.identity,self.m)
        metadata=self.scheduler.metadata
        self.scheduler=EvaluationScheduler(self.f.root/'eval',self.bucket,self.queue,self.identity,
            policy=self.policy,metadata=metadata,workers=[self.worker.id],set_name='fixed4',job_seconds=120)
        first=self.scheduler.tick([self.opening],now=self.now)
        self.run_worker(Runtime(self.m))
        m=deepcopy(self.m);m['source_bundle']['sha256']='b'*64
        new=build_policy(self.identity,m,task_assets=assets,enabled=True);m[POLICY_KEY]=new
        new_policy=ReleasePolicy(new,self.identity.id)
        before=metadata(self.m);after=deepcopy(before)
        after['source_files']['subnet/mimo_grpo/transaction.py']='e'*64
        registry=SourceReleaseRegistry(sign(self.identity,dict(version=REGISTRY,policies=[old,new],
            active_policy_sha256=digest(new),execution={'a'*64:before,'b'*64:after},
            source_changes=['subnet/mimo_grpo/transaction.py'],trainer_config={'fixture':'unchanged'},
            trainer_identity=self.worker.id)),self.identity.id)
        self.scheduler=EvaluationScheduler(self.f.root/'eval',self.bucket,self.queue,self.identity,
            policy=new_policy,metadata=metadata,workers=[self.worker.id],set_name='fixed4',source_registry=registry)
        self.assertEqual(self.scheduler.tick([self.opening],now=time.time())['status'],'caught-up')
        receipt=json.loads((self.scheduler.root/(first['job_id']+'.receipt.json')).read_bytes())['payload']
        self.assertEqual(receipt['policy_sha256'],self.policy.policy_sha256)
        self.assertEqual(receipt['job']['payload']['source_files'],before['source_files'])
        m['checkpoint']={'files':dict(m['checkpoint']['files'],**{'config.json':'f'*64})};m['checkpoint']['id']=digest(m['checkpoint']['files'])
        m['sampling_contract']['calibration']['checkpoint']=m['checkpoint']['id']
        m[CHECKPOINT_KEY]=new_policy.authorize_checkpoint(m['checkpoint'],input_checkpoint=self.m['checkpoint']['id'],
            optimizer_step=1,publication_receipt_sha256='f'*64,identity=self.identity)
        m['epoch']='nonpayable-new-source';m['start']+=1;m['deadline']+=1
        self.now=time.time()
        result=self.scheduler.tick([self.opening,sign(self.identity,m)],now=self.now)
        self.assertEqual(result['status'],'enqueued')
        job=json.loads((self.scheduler.root/(result['job_id']+'.job.json')).read_bytes())['payload']
        self.assertEqual(job['source_files'],after['source_files'])
        self.assertEqual(job['manifest']['payload'][POLICY_KEY],new)

    def test_enqueue_lost_ack_keeps_exact_original_job(self):
        enqueue=self.queue.enqueue
        def lose(envelope):enqueue(envelope);raise OSError('lost acknowledgement')
        with patch.object(self.queue,'enqueue',side_effect=lose),self.assertRaises(OSError):
            self.scheduler.tick([self.opening],now=self.now)
        before=(self.scheduler.root/'state.json').read_bytes()
        self.assertEqual(self.scheduler.tick([self.opening],now=self.now)['status'],'evaluating')
        self.assertEqual((self.scheduler.root/'state.json').read_bytes(),before)
        self.assertEqual(len(list(self.scheduler.root.glob('*.job.json'))),1)

    def test_native_error_report_is_preserved_and_retried_with_a_bound(self):
        class Broken(Runtime):
            def next_token_logits(self,*args,**kwargs):raise OSError('fixture device failure')
        first=self.scheduler.tick([self.opening],now=self.now)
        self.run_worker(Broken(self.m),'broken1')
        self.now=time.time();second=self.scheduler.tick([self.opening],now=self.now)
        self.assertEqual(second['status'],'enqueued');self.assertNotEqual(first['job_id'],second['job_id'])
        self.assertEqual(len(list(self.scheduler.root.glob('*.receipt.json'))),1)
        self.run_worker(Broken(self.m),'broken2')
        self.now=time.time();self.scheduler.tick([self.opening],now=self.now)
        state=json.loads((self.scheduler.root/'state.json').read_bytes())
        self.assertEqual(next(iter(state['checkpoints'].values()))['status'],'exhausted')
        receipts=list(self.scheduler.root.glob('*.receipt.json'));self.assertEqual(len(receipts),2)
        for path in receipts:
            evaluation=json.loads(path.read_bytes())['payload']['worker_report']['payload']['report']['evaluation']
            self.assertEqual(evaluation['status'],'error');self.assertEqual(evaluation['count'],0)

    def test_expired_execution_retries_original_checkpoint_then_exhausts_no_fake_metric(self):
        first=self.scheduler.tick([self.opening],now=self.now)
        self.now+=121
        second=self.scheduler.tick([self.opening],now=self.now)
        self.assertNotEqual(first['job_id'],second['job_id'])
        jobs=[json.loads(p.read_bytes())['payload']for p in self.scheduler.root.glob('*.job.json')]
        self.assertTrue(all(j['manifest']==self.opening for j in jobs))
        self.now+=121;self.scheduler.tick([self.opening],now=self.now)
        self.assertEqual(list(self.scheduler.root.glob('*.receipt.json')),[])
        state=json.loads((self.scheduler.root/'state.json').read_bytes())
        self.assertEqual(next(iter(state['checkpoints'].values()))['status'],'exhausted')
