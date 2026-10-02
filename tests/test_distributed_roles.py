import base64
import copy
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from nacl.signing import SigningKey
from subnet.distributed_roles import Coordinator, CoordinatorServer, authenticate, digest
from subnet.storage import canonical
from subnet.backend_profiles import for_config


def sign(key,payload):
    return dict(signer=key.verify_key.encode().hex(),payload=payload,
                signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        self.operator=SigningKey.generate(); self.workers=[SigningKey.generate(),SigningKey.generate()]
        self.authority=self.operator.verify_key.encode().hex(); self.now=100.
        self.queue=Coordinator(Path(self.folder.name)/'queue.sqlite3',self.authority,
            {k.verify_key.encode().hex():['verify'] for k in self.workers},lease_seconds=10,max_attempts=2,clock=lambda:self.now)
        self.manifest=dict(epoch='nonpayable-math',payable=False,checkpoint={'id':'CP1'},
            backend_profile={'sm':[9,0]},numerical_policy={'logprob_atol':0.00001},
            audit_frozen_receipts={'miner':{'sha256':'frozen-sha'}})
        revision,profile,policy=for_config({'model_runtime_revision':'cuda-bf16-eager-sm90-v1'})
        self.manifest.update(model_runtime_revision=revision,backend_profile=profile,numerical_policy=policy)
        self.job=dict(schema=1,job_id='job1',role='verify',created_at=90.,expires_at=200.,
            manifest=sign(self.operator,self.manifest),source_files={'subnet/file.py':'source-sha'},
            runtime_versions={'torch':'pinned'},submissions=[{'url':'private-capability','sha256':'frozen-sha'}])
        self.queue.enqueue(sign(self.operator,self.job));self.nonce=0

    def request(self,worker=0,action='claim',**fields):
        self.nonce+=1
        return self.queue.request(sign(self.workers[worker],dict(action=action,at=self.now,nonce=str(self.nonce).zfill(32),**fields)))

    def claim(self,worker=0):return self.request(worker,role='verify')['claim']

    def report(self):
        return dict(job_id='job1',job_sha256=digest(self.job),operator=self.authority,role='verify',
            epoch=self.manifest['epoch'],checkpoint='CP1',source_files=self.job['source_files'],
            runtime_versions=self.job['runtime_versions'],backend_profile=self.manifest['backend_profile'],
            numerical_policy=self.manifest['numerical_policy'],chain_transactions=False,success=True,completed_at=self.now,
            audits=[{'epoch':self.manifest['epoch'],'submission_sha256':'frozen-sha','accepted':[
                {'epoch':self.manifest['epoch'],'checkpoint':'CP1'}]}])

    def submit(self,claim,report=None,worker=0):
        return self.request(worker,'report',job_id='job1',token=claim['token'],report=report or self.report())

    def test_concurrent_claim_is_atomic_across_independent_connections(self):
        barrier=threading.Barrier(2)
        def claim(i):
            barrier.wait()
            envelope=sign(self.workers[i],dict(action='claim',role='verify',at=self.now,nonce=str(i).zfill(32)))
            # Independent coordinator instances reproduce multiple processes.
            coordinator=Coordinator(self.queue.path,self.authority,self.queue.workers,lease_seconds=10,clock=lambda:self.now)
            return coordinator.request(envelope)['claim']
        with ThreadPoolExecutor(2) as pool: claims=list(pool.map(claim,[0,1]))
        self.assertEqual(sum(c is not None for c in claims),1)

    def test_two_workers_claim_distinct_jobs(self):
        other=dict(self.job,job_id='job2');self.queue.enqueue(sign(self.operator,other))
        first=self.claim();second=self.claim(1)
        self.assertNotEqual(first['job']['payload']['job_id'],second['job']['payload']['job_id'])

    def test_idempotent_enqueue_and_collision_rejection(self):
        self.queue.enqueue(sign(self.operator,self.job))
        with self.assertRaisesRegex(ValueError,'collision'):
            self.queue.enqueue(sign(self.operator,dict(self.job,source_files={'evil':'bad'})))

    def test_expired_lease_retried_and_old_token_rejected(self):
        old=self.claim();self.now=111.;fresh=self.claim(1)
        self.assertEqual(fresh['attempt'],2);self.assertNotEqual(old['token'],fresh['token'])
        with self.assertRaisesRegex(ValueError,'stale'):self.submit(old)
        self.assertTrue(self.submit(fresh,worker=1)['accepted'])

    def test_renew_does_not_extend_signed_job(self):
        lease=self.claim();self.now=109.
        result=self.request(action='renew',job_id='job1',token=lease['token'])
        self.assertEqual(result['lease_until'],119.)
        self.now=119.
        with self.assertRaisesRegex(ValueError,'expired'):self.request(action='renew',job_id='job1',token=lease['token'])

    def test_original_job_expiry_prevents_claim_and_report(self):
        lease=self.claim();self.now=200.
        with self.assertRaisesRegex(ValueError,'expired'):self.submit(lease)
        self.assertIsNone(self.claim(1));self.assertEqual(self.queue.status('job1')['status'],'expired')

    def test_bounded_failed_worker_retry(self):
        lease=self.claim();self.assertEqual(self.request(action='fail',job_id='job1',token=lease['token'])['status'],'queued')
        lease=self.claim(1);self.assertEqual(self.request(1,'fail',job_id='job1',token=lease['token'])['status'],'failed')
        self.assertIsNone(self.claim())

    def test_duplicate_report_idempotence_and_conflict(self):
        lease=self.claim();report=self.report()
        self.assertFalse(self.submit(lease,report)['duplicate']);self.now=250.
        self.assertTrue(self.submit(lease,report)['duplicate'])
        self.assertEqual(self.queue.enqueue(sign(self.operator,self.job)),'job1')
        changed=dict(report,execution_resources_enforced=True)
        with self.assertRaisesRegex(ValueError,'conflicting'):self.submit(lease,changed)

    def test_forged_cross_epoch_checkpoint_source_runtime_or_metrics_rejected(self):
        lease=self.claim()
        changes=[('epoch','other'),('checkpoint','CP2'),('source_files',{'evil':'hash'}),
                 ('runtime_versions',{'torch':'unapproved'}),('backend_profile',dict(self.manifest['backend_profile'],tf32=0)),('job_sha256','other'),('operator','fake'),
                 ('chain_transactions',True),('chain_transactions',0),('success',1),
                 ('completed_at',201.),('completed_at',True),('numerical_policy',{'logprob_atol':1})]
        for field,value in changes:
            report=self.report();report[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.submit(lease,report)
        report=self.report();report['audits'][0]['submission_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'frozen'):self.submit(lease,report)
        report=self.report();report['audits'][0]['accepted'][0]['checkpoint']='CP2'
        with self.assertRaisesRegex(ValueError,'audit'):self.submit(lease,report)
        with self.assertRaisesRegex(ValueError,'stale'):self.submit(lease,worker=1)
        self.assertTrue(self.submit(lease)['accepted'])

    def test_request_replay_signature_and_unregistered_identity(self):
        payload=dict(action='claim',role='verify',at=self.now,nonce='nonce-000000000000000000')
        envelope=sign(self.workers[0],payload);self.queue.request(envelope)
        with self.assertRaisesRegex(ValueError,'replay'):self.queue.request(envelope)
        with self.assertRaises(ValueError):self.queue.request(sign(SigningKey.generate(),payload))
        forged=sign(self.workers[1],dict(payload,nonce='different000000000000'));forged['payload']['role']='train'
        with self.assertRaises(Exception):self.queue.request(forged)

    def test_nonpayable_and_frozen_hash_required(self):
        for changes in ({'payable':True},{'epoch':'payable-math'},{'audit_frozen_receipts':{}}):
            job=dict(self.job,job_id='new-job',manifest=sign(self.operator,dict(self.manifest,**changes)))
            with self.assertRaises(ValueError):self.queue.enqueue(sign(self.operator,job))

    def test_http_transport_authenticates_both_directions(self):
        import requests
        server=CoordinatorServer(('127.0.0.1',0),self.queue,lambda p:sign(self.operator,p))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            request=sign(self.workers[0],dict(action='claim',role='verify',at=self.now,nonce='a'*32))
            response=requests.post('http://127.0.0.1:'+str(server.server_port)+'/request',json=request,timeout=5)
            self.assertEqual(response.status_code,200)
            value=authenticate(response.json(),self.authority);self.assertEqual(value['claim']['job_sha256'],digest(self.job))
            response=requests.post('http://127.0.0.1:'+str(server.server_port)+'/request',json=request,timeout=5)
            self.assertEqual(response.status_code,403);self.assertNotIn('private-capability',response.text)
        finally:server.shutdown();server.server_close();thread.join()

    def test_r2_history_conditional_and_authenticated(self):
        class Client:
            def __init__(self):self.objects={};self.calls=[]
            def put_object(self,**kwargs):
                self.calls.append(kwargs);self.objects[kwargs['Key']]=kwargs['Body']
        from types import SimpleNamespace
        client=Client();bucket=SimpleNamespace(client=client,name='private-bucket')
        self.submit(self.claim());self.queue.archive('job1',bucket,'private/roles')
        self.assertTrue(all(call['IfNoneMatch']=='*' for call in client.calls))
        saved=json.loads(client.objects['private/roles/job1/worker-report.json'])
        self.assertEqual(authenticate(saved,self.workers[0].verify_key.encode().hex())['report'],self.report())

if __name__=='__main__':unittest.main()
