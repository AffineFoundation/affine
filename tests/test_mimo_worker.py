"""Real signed job/queue/Miner/codec boundaries with CPU-only runtime fixtures."""
import base64
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from test_complete_group_mimo import mimo_manifest, MimoCollectionRuntime
from test_complete_group_transport import FakeBucket, FakeHTTP
from subnet.storage import Identity, canonical
from subnet.mimo_runtime import MimoRuntimeSpec, MODEL_ID, VERSION as RUNTIME_VERSION, EOS_TOKEN_IDS
from subnet.complete_group_transport import CompleteGroupTransport, capability, paths
from subnet.complete_group_admission import signed_commitment, digest
from subnet.miner import Miner
from subnet import backend_jobs, mimo_worker
from subnet.distributed_roles import Coordinator
from subnet.distributed_worker import Worker


def sign(key, payload):
    return dict(payload=deepcopy(payload),signer=key.verify_key.encode().hex(),
        signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())


class WorkerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.operator=SigningKey.generate();self.authority=self.operator.verify_key.encode().hex()
        self.identity=Identity(bytes(range(32)));self.bucket=FakeBucket();self.http=FakeHTTP(self.bucket)
        self.m=mimo_manifest();self.m.update(epoch='nonpayable-mimo-worker-fixture',payable=False,model_runtime_revision=RUNTIME_VERSION)
        self.m['mimo_runtime']=MimoRuntimeSpec(RUNTIME_VERSION,MODEL_ID,{'versions':{},'sources':{}},'a'*64,True,64,EOS_TOKEN_IDS).to_dict()
        self.m['checkpoint']={'files':{'config.json':'1'*64,'model.safetensors':'2'*64}}
        self.m['checkpoint']['id']=backend_jobs.file_map(self.m['checkpoint']['files'])
        self.m['sampling_contract']['calibration']['checkpoint']=self.m['checkpoint']['id']
        self.m['sampling_contract']['calibration']['model_runtime_revision']=RUNTIME_VERSION
        self.m['capabilities']={self.identity.id:capability(self.bucket,self.m,self.identity.id,registered_miners={self.identity.id})}
        self.cap=self.identity.decrypt(self.m['capabilities'][self.identity.id])
        with patch('subnet.model.check_runtime_profile'):
            self.actor=Miner(self.identity,self.m,'not-a-model',complete_group_transport=CompleteGroupTransport(http=self.http))
        self.addCleanup(self.actor._group_journal.close)
        runtime=MimoCollectionRuntime(self.m,self.identity.id)
        with patch('subnet.miner.make_runtime',return_value=runtime):self.batch=self.actor.search(0)
        self.data=self.actor._group_prepared[0][1]
        self.commit=json.loads(signed_commitment(self.identity,self.m,self.actor._group_prepared))
        self.claim=self.commit['payload']['groups'][0]
        self.key=paths(self.m,self.identity.id,0)[0]
        self.url='https://account.r2.cloudflarestorage.com/bucket/'+self.key+'?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=fixture'
        self.obj=dict(url=self.url,sha256=self.claim['artifact_sha256'],commitment_miner=self.identity.id,
            commitment_ref=dict(miner=self.identity.id,commitment_sha256=digest(self.commit),
                slot=0,env_id='math',index=0,batch_sha256=self.claim['batch_sha256'],size=self.claim['artifact_size'],
                group_id=self.claim['group_id'],artifact_key=self.key))
        now=time.time()
        self.job=dict(schema=1,worker_policy=mimo_worker.VERSION,job_id='mimo-verify-1',role='verify',
            created_at=now-5,expires_at=now+180,manifest=sign(self.operator,self.m),source_files=mimo_worker.source_inventory(),
            runtime_versions={p:version(p) for p in ('torch','transformers','toploc')},submissions=[self.obj],
            group_commitments={self.identity.id:self.commit})
        env=patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'});env.start();self.addCleanup(env.stop)

    def envelope(self): return sign(self.operator,self.job)
    def fetch(self,url,expected,path,limit):
        self.assertEqual(expected,hashlib.sha256(self.data).hexdigest());self.assertEqual(limit,len(self.data))
        path.write_bytes(self.data)
    def verify(self,data,manifest,checkpoint,*,miner,runtime_factory):
        self.assertEqual(data,self.data);self.assertEqual(miner,self.identity.id)
        self.assertIs(runtime_factory,self.factory)
        return dict(epoch=manifest['epoch'],miner=miner,submission_sha256=self.obj['sha256'],accepted=[self.batch],
            outcomes=[dict(batch=0,valid=True,fully_audited=True,rollouts=[
                dict(positions=len(r['turns'][0]['output']),all_intervals_verified=True) for r in self.batch['rollouts']])])
    def execute(self,name='backend'):
        self.factory=object()
        with patch('subnet.backend_jobs.get_object',side_effect=self.fetch),patch('subnet.complete_group_verification.verify',side_effect=self.verify),patch('subnet.backend_jobs.checkpoint') as legacy:
            result=backend_jobs.execute(self.envelope(),self.authority,self.root/name,runtime_factory=self.factory)
            legacy.assert_not_called()
        return result
    def test_signed_group_dispatch_bypasses_old_backend_quotas_and_artifact_budget(self):
        with patch('subnet.backend_profiles.resolve',side_effect=AssertionError('legacy')),patch('subnet.artifact_budget.for_manifest',side_effect=AssertionError('legacy')):
            job,manifest=backend_jobs.validate(self.envelope(),self.authority)
            self.assertEqual(manifest['group_contract']['group_size'],8)
            result=self.execute()
        self.assertEqual(result['audits'][0]['accepted'][0]['rollout_count'],8)
        self.assertNotIn('rollouts',result['audits'][0]['accepted'][0])
        mimo_worker.validate_report(result,self.job,self.m,self.authority,digest(self.job),time.time())
    def test_missing_policy_unknown_runtime_or_missing_source_fail_before_read(self):
        for key,value in [('worker_policy','unknown'),('source_files',{}),('role','train')]:
            job=deepcopy(self.job);job[key]=value
            with self.subTest(key=key),patch('subnet.backend_jobs.get_object') as fetch:
                with self.assertRaises(ValueError):backend_jobs.execute(sign(self.operator,job),self.authority,self.root/'unused')
                fetch.assert_not_called()
    def test_signed_slot_references_cannot_swap_miner_task_body_or_key(self):
        for field,value in [('miner','a'*64),('slot',True),('index',1),('batch_sha256','a'*64),('size',1),('artifact_key','private/other')]:
            rows=deepcopy([self.obj]);rows[0]['commitment_ref'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):mimo_worker.validate_submissions(self.m,rows,{self.identity.id:self.commit})
        with self.assertRaises(ValueError):mimo_worker.validate_submissions(self.m,[self.obj,self.obj],{self.identity.id:self.commit})
        changed=deepcopy(self.commit);changed['payload']['groups'][0]['index']=1
        with self.assertRaises(Exception):mimo_worker.validate_submissions(self.m,[self.obj],{self.identity.id:changed})
    def test_reference_builder_reuses_original_signed_slot_and_accepts_new_inventory_bound(self):
        self.assertEqual(mimo_worker.submission_reference(self.m,self.identity.id,self.commit,0,self.url),self.obj)
        with self.assertRaises(ValueError):mimo_worker.submission_reference(self.m,self.identity.id,self.commit,1,self.url)
        # New runtime permits a sharded inventory beyond legacy file_map's 32;
        # this only checks admission and never downloads or loads these fixtures.
        manifest=deepcopy(self.m)
        files={'config.json':'1'*64,**{f'model-{i:03d}.safetensors':'2'*64 for i in range(40)}}
        manifest['checkpoint']={'files':files,'id':digest(files)}
        manifest['sampling_contract']['calibration']['checkpoint']=manifest['checkpoint']['id']
        with patch('subnet.mimo_worker.validate_submissions'):
            job=deepcopy(self.job);job['manifest']=sign(self.operator,manifest)
            _,admitted=mimo_worker.validate(sign(self.operator,job),self.authority)
            self.assertEqual(len(admitted['checkpoint']['files']),41)

    def test_source_closure_change_rejects_before_download(self):
        self.job['source_files']['subnet/miner.py']='a'*64
        with patch('subnet.backend_jobs.get_object') as fetch:
            with self.assertRaisesRegex(ValueError,'source closure'):backend_jobs.execute(self.envelope(),self.authority,self.root/'unused')
            fetch.assert_not_called()
    def test_local_checkpoint_candidate_avoids_network_but_still_requires_cache_digest_admission(self):
        from contextlib import contextmanager
        root=self.root/'original-model';root.mkdir()
        contents={'config.json':b'{"fixture":true}','model.safetensors':b'fixture model bytes'}
        for name,data in contents.items():(root/name).write_bytes(data)
        m=deepcopy(self.m);files={n:hashlib.sha256(data).hexdigest() for n,data in contents.items()}
        m['checkpoint']={'id':digest(files),'files':files}
        class CacheAdmission:
            def __init__(inner,*args,fetch_file,**kwargs):inner.fetch=fetch_file
            @contextmanager
            def runtime_for_job(inner,checkpoint,manifest,*args,**kwargs):
                for name,expected in manifest['checkpoint']['files'].items():
                    staged=self.root/('copied-'+name)
                    inner.fetch(manifest,name,staged)
                    if hashlib.sha256(staged.read_bytes()).hexdigest()!=expected:
                        raise ValueError('cache digest mismatch')
                yield 'digest-admitted-fixture'
            def close(inner):pass
        with patch('subnet.mimo_runtime.PersistentMimoRuntimeFactory',CacheAdmission),patch('subnet.backend_jobs.get_object') as network:
            factory=mimo_worker.CheckpointTransportFactory(self.root/'cache',max_bytes=128*1024**3)
            factory.bind_transport(m,{})
            with factory.runtime_for_job(root,m,{}, {},miner='a'*64) as result:
                self.assertEqual(result,'digest-admitted-fixture')
                with self.assertRaisesRegex(ValueError,'in flight'):factory.bind_transport(m,{})
            network.assert_not_called();self.assertIsNone(factory.local_directory)
            (root/'model.safetensors').write_bytes(b'changed model bytes')
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                with factory.runtime_for_job(root,m,{}, {},miner='a'*64):pass
            self.assertIsNone(factory.local_directory)
            (root/'extra.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'inventory'):
                with factory.runtime_for_job(root,m,{}, {},miner='a'*64):pass
            factory.close()
    def test_invalid_artifact_does_not_load_model_and_infra_does_not_mint_invalid(self):
        with patch('subnet.backend_jobs.get_object',side_effect=backend_jobs.ArtifactRejected('wrong digest')),patch('subnet.complete_group_verification.verify') as verify:
            result=backend_jobs.execute(self.envelope(),self.authority,self.root/'invalid',runtime_factory=object())
            self.assertEqual(result['audits'][0]['outcomes'][0]['failure_kind'],'invalid_sample');verify.assert_not_called()
        with patch('subnet.backend_jobs.get_object',side_effect=OSError('unavailable')):
            with self.assertRaises(OSError):backend_jobs.execute(self.envelope(),self.authority,self.root/'infra',runtime_factory=object())
        self.assertFalse((self.root/'infra/jobs'/self.job['job_id']/'report.json').exists())
    def test_queue_original_group_report_and_idempotent_lost_ack(self):
        worker=SigningKey.generate();queue=Coordinator(self.root/'queue.db',self.authority,{worker.verify_key.encode().hex():['verify']})
        queue.enqueue(self.envelope());counter=iter(range(100))
        def request(action,**kw):return queue.request(sign(worker,dict(action=action,at=time.time(),nonce=str(next(counter)).zfill(32),**kw)))
        claim=request('claim',role='verify')['claim'];result=self.execute()
        changed=deepcopy(result);changed['audits'][0]['accepted'][0]['batch_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'original committed'):request('report',job_id=self.job['job_id'],token=claim['token'],report=changed)
        self.assertFalse(request('report',job_id=self.job['job_id'],token=claim['token'],report=result)['duplicate'])
        self.assertTrue(request('report',job_id=self.job['job_id'],token=claim['token'],report=result)['duplicate'])
    def test_worker_reuses_one_factory_across_jobs_and_deletes_inputs_after_ack(self):
        key=SigningKey.generate();queue=Coordinator(self.root/'queue.db',self.authority,{key.verify_key.encode().hex():['verify']})
        queue.enqueue(self.envelope());second=deepcopy(self.job);second['job_id']='mimo-verify-2';queue.enqueue(sign(self.operator,second))
        from types import SimpleNamespace
        self.factory=SimpleNamespace(close=lambda:None)
        worker=Worker('http://localhost:99',key.encode(),self.authority,self.root/'worker',mimo_factory=self.factory)
        count=iter(range(100)); reports=[]
        def request(action,**kw):
            result=queue.request(sign(key,dict(action=action,at=time.time(),nonce=str(next(count)).zfill(32),**kw)))
            if action=='report':reports.append(kw['report'])
            return result
        worker.request=request
        with patch('subnet.backend_jobs.get_object',side_effect=self.fetch),patch('subnet.complete_group_verification.verify',side_effect=self.verify),patch('subnet.distributed_worker.subprocess.run') as subprocess:
            self.assertTrue(worker.once());self.assertTrue(worker.once());subprocess.assert_not_called()
        self.assertEqual(len(reports),2);self.assertIs(worker.mimo_factory,self.factory)
        self.assertFalse(list((self.root/'worker/backend').rglob('submission-*.zip')))
        self.assertEqual(queue.status('mimo-verify-2')['status'],'complete')
    def test_worker_lost_ack_retries_same_report_without_reverification(self):
        import requests
        from types import SimpleNamespace
        key=SigningKey.generate();queue=Coordinator(self.root/'queue.db',self.authority,{key.verify_key.encode().hex():['verify']})
        queue.enqueue(self.envelope())
        self.factory=SimpleNamespace(close=lambda:None)
        worker=Worker('http://localhost:99',key.encode(),self.authority,self.root/'worker',mimo_factory=self.factory)
        count=iter(range(100));reports=[]
        def request(action,**kw):
            result=queue.request(sign(key,dict(action=action,at=time.time(),nonce=str(next(count)).zfill(32),**kw)))
            if action=='report':
                self.assertTrue(list((self.root/'worker/backend').rglob('submission-*.zip')))
                reports.append(kw['report'])
                if len(reports)==1:raise requests.ConnectionError('ack lost after commit')
            return result
        worker.request=request
        with patch('subnet.backend_jobs.get_object',side_effect=self.fetch),patch('subnet.complete_group_verification.verify',side_effect=self.verify) as verify,patch('subnet.distributed_worker.time.sleep'):
            self.assertTrue(worker.once());self.assertEqual(verify.call_count,1)
        self.assertEqual(reports[0],reports[1]);self.assertFalse(list((self.root/'worker/backend').rglob('submission-*.zip')))

    def test_actual_backend_training_dispatch_allows_recovery_directory(self):
        from subnet.mimo_grpo import TrainerConfig
        self.job={k:v for k,v in self.job.items() if k not in ('submissions','group_commitments')}
        self.job.update(role='train',job_id='mimo-training-dispatch',trainer_worker=self.identity.id,training=dict(
            run_id=self.m['run_id'],optimizer_step=0,config=TrainerConfig().document(),
            dispatch={'url':self.url,'sha256':'a'*64,'size':100},documents={},
            publication_namespace='private/worker-test',publisher_id=self.identity.id,role_config_file='/private/test-only.json'))
        def train(job,manifest,authority,workspace,out,report,**kwargs):
            self.assertEqual(authority,self.authority);self.assertEqual(job['training']['optimizer_step'],0)
            return dict(report,training={'status':'no-update'})
        with patch('subnet.mimo_training_worker.execute',side_effect=train) as train_call,patch('subnet.backend_jobs.checkpoint') as oldload:
            first=backend_jobs.execute(self.envelope(),self.authority,self.root/'train',assert_lease=lambda:None,trainer_state_root=self.root/'stable-trainer')
            second=backend_jobs.execute(self.envelope(),self.authority,self.root/'train',assert_lease=lambda:None,trainer_state_root=self.root/'stable-trainer')
            self.assertEqual(first['training'],second['training']);self.assertEqual(train_call.call_count,2);oldload.assert_not_called()
        out=self.root/'train/jobs/mimo-training-dispatch';out.chmod(0o755)
        with patch('subnet.mimo_training_worker.execute') as train_call:
            with self.assertRaisesRegex(ValueError,'owner-only'):backend_jobs.execute(self.envelope(),self.authority,self.root/'train',assert_lease=lambda:None,trainer_state_root=self.root/'stable-trainer')
            train_call.assert_not_called()

    def test_ack_cleanup_removes_inputs_and_preserves_durable_trainer_state(self):
        out=self.root/'jobs/finished';inputs=out/'training-inputs';inputs.mkdir(parents=True)
        (inputs/'DISPATCH.json').write_text('{}');(out/'submission-0.zip').write_bytes(b'original downloaded bytes')
        optimizer=self.root/'mimo-training/run/history/step/optimizer';optimizer.mkdir(parents=True)
        state=optimizer/'master.bin';state.write_bytes(b'keep durable local state')
        mimo_worker.cleanup_acknowledged_inputs(out)
        self.assertFalse(inputs.exists());self.assertFalse((out/'submission-0.zip').exists())
        self.assertEqual(state.read_bytes(),b'keep durable local state')
        inputs.symlink_to(optimizer,target_is_directory=True)
        with self.assertRaises(ValueError):mimo_worker.cleanup_acknowledged_inputs(out)
        self.assertTrue(state.exists())

    def training_job(self, worker_identity):
        from subnet.mimo_grpo import TrainerConfig
        job={k:v for k,v in self.job.items() if k not in ('submissions','group_commitments')}
        job.update(role='train',job_id='mimo-train-queue',trainer_worker=worker_identity,training=dict(
            run_id=self.m['run_id'],optimizer_step=0,config=TrainerConfig().document(),
            dispatch={'url':self.url,'sha256':'a'*64,'size':100},documents={},
            publication_namespace='private/worker-test',publisher_id=self.identity.id,role_config_file='/private/test-only.json'))
        return job

    def empty_training_report(self,report):
        return dict(report,training=dict(status='no-update',reason='no-eligible-complete-groups',group_count=0,
            rollout_count=0,input_checkpoint=self.m['checkpoint']['id'],output_checkpoint=self.m['checkpoint']['id'],
            optimizer_step=0,training_requires_sampling_audit=False))

    def test_training_released_job_uses_same_stable_state_root_and_named_worker(self):
        key=SigningKey.generate();other=SigningKey.generate();kid=key.verify_key.encode().hex()
        self.job=self.training_job(kid)
        queue=Coordinator(self.root/'trainqueue.db',self.authority,{kid:['train'],other.verify_key.encode().hex():['train']})
        queue.enqueue(self.envelope());counter=iter(range(100))
        def request(action,**kw):return queue.request(sign(key,dict(action=action,at=time.time(),nonce=str(next(counter)).zfill(32),**kw)))
        self.assertIsNone(queue.request(sign(other,dict(action='claim',role='train',at=time.time(),nonce='other'.zfill(32))))['claim'])
        stable=self.root/'stable-trainer';worker=Worker('http://localhost:99',key.encode(),self.authority,self.root/'worker',role='train',trainer_state_root=stable)
        worker.request=request;calls=[]
        def train(job,manifest,authority,workspace,out,report,**kw):
            kw['assert_lease']();calls.append((Path(workspace),Path(out),kw['trainer_state_root']))
            stable.mkdir(exist_ok=True);marker=stable/'preserved-original-step';
            if not marker.exists():marker.write_text('original private fixture state')
            if len(calls)==1:raise OSError('original prepared work interrupted')
            return self.empty_training_report(report)
        with patch('subnet.mimo_training_worker.execute',side_effect=train),patch('subnet.mimo_worker.persistent_factory') as verifier_factory:
            self.assertTrue(worker.once());self.assertTrue(worker.once());verifier_factory.assert_not_called()
        self.assertNotEqual(calls[0][0],calls[1][0]);self.assertEqual(calls[0][2],calls[1][2]);self.assertEqual(calls[0][2],stable)
        self.assertEqual((stable/'preserved-original-step').read_text(),'original private fixture state')
        self.assertEqual(queue.status(self.job['job_id'])['status'],'complete');worker.close()

    def test_training_worker_lease_expiry_blocks_side_effect_callback(self):
        key=SigningKey.generate();kid=key.verify_key.encode().hex();self.job=self.training_job(kid)
        queue=Coordinator(self.root/'trainqueue.db',self.authority,{kid:['train']},lease_seconds=10)
        queue.enqueue(self.envelope());counter=iter(range(100));claimed=[]
        def request(action,**kw):
            result=queue.request(sign(key,dict(action=action,at=queue.clock(),nonce=str(next(counter)).zfill(32),**kw)))
            if action=='claim':claimed.append(result['claim'])
            return result
        worker=Worker('http://localhost:99',key.encode(),self.authority,self.root/'worker',role='train',trainer_state_root=self.root/'stable')
        worker.request=request;side_effects=[]
        def train(job,manifest,authority,workspace,out,report,**kw):
            with patch('subnet.distributed_worker.time.time',return_value=claimed[0]['lease_until']+1):
                kw['assert_lease']()
            side_effects.append('optimizer/publication/current')
            return self.empty_training_report(report)
        with patch('subnet.mimo_training_worker.execute',side_effect=train):self.assertTrue(worker.once())
        self.assertEqual(side_effects,[]);self.assertEqual(queue.status(self.job['job_id'])['status'],'queued');worker.close()

    def test_refreshed_checkpoint_transport_keeps_original_manifest_and_draws(self):
        from types import SimpleNamespace
        original=deepcopy(self.m)
        urls={name:self.url.replace(self.key,'checkpoints/'+name) for name in self.m['checkpoint']['files']}
        self.job['checkpoint_read_urls']=urls
        observed=[];self.factory=SimpleNamespace(bind_transport=lambda manifest,transport:observed.append((deepcopy(manifest),dict(transport))))
        with patch('subnet.backend_jobs.get_object',side_effect=self.fetch),patch('subnet.complete_group_verification.verify',side_effect=self.verify):
            backend_jobs.execute(self.envelope(),self.authority,self.root/'refresh',runtime_factory=self.factory)
        self.assertEqual(observed,[(original,urls)]);self.assertEqual(self.m,original)
        self.job['checkpoint_read_urls']={}
        with self.assertRaisesRegex(ValueError,'inventory'):mimo_worker.validate(self.envelope(),self.authority)

    def test_owned_mine_uses_real_group_miner_and_exact_nonce_count(self):
        seed=self.root/'owned.seed';seed.write_text(self.identity.key.encode().hex());seed.chmod(0o600)
        self.job={k:v for k,v in self.job.items() if k not in ('submissions','group_commitments')}
        self.job.update(job_id='mimo-owned-mine',role='mine',miner_id=self.identity.id,miner_identity_file=str(seed),
            capability=self.cap,search_budget=32,seed_start=0,mining_subset={'math':[1]})
        runtime=MimoCollectionRuntime(self.m,self.identity.id)
        adapter=CompleteGroupTransport(http=self.http)
        with patch('subnet.backend_jobs.checkpoint',return_value=self.root/'no-model'),patch('subnet.complete_group_transport.CompleteGroupTransport',return_value=adapter),patch('subnet.model.check_runtime_profile'),patch('subnet.miner.make_runtime',return_value=runtime):
            result=backend_jobs.execute(self.envelope(),self.authority,self.root/'mine')
        self.assertEqual(runtime.calls,[(1,n)for n in range(8)])
        self.assertEqual(result['submitted_groups'],1);self.assertEqual(result['exact_group_size'],8)
    def test_mining_budget_and_offset_reject_before_checkpoint(self):
        self.job={k:v for k,v in self.job.items() if k not in ('submissions','group_commitments')}
        self.job.update(role='mine',miner_id=self.identity.id,miner_identity_file='/private/owned.seed',capability=self.cap)
        for budget,offset in [(7,0),(False,0),(32,1),(32,False)]:
            self.job.update(search_budget=budget,seed_start=offset)
            with patch('subnet.backend_jobs.checkpoint') as checkpoint:
                with self.assertRaises(ValueError):backend_jobs.execute(self.envelope(),self.authority,self.root/'unused')
                checkpoint.assert_not_called()

if __name__=='__main__':unittest.main()
