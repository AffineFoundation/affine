import copy
import base64
import unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.backend_jobs import validate,execute,canonical,file_map,REVISION,NUMERICAL_POLICY,BACKEND_PROFILE,SOURCE_FILES,r2_url

class TrainingPairAttribution(unittest.TestCase):
    def test_exact_consumed_pair_is_fingerprinted(self):
        import hashlib
        from subnet.backend_jobs import pair_attribution
        positive={'env_id':'science','index':7,'turns':[{'output':[1,2]}]}
        negative={'env_id':'science','index':7,'turns':[{'output':[3]}]}
        row=pair_attribution({'env_id':'science'},positive,negative,2)
        self.assertEqual(row['optimizer_step'],3)
        self.assertEqual(row['positive_rollout_sha256'],hashlib.sha256(canonical(positive)).hexdigest())
        changed=copy.deepcopy(positive);changed['turns'][0]['output'][0]=9
        self.assertNotEqual(row['positive_rollout_sha256'],pair_attribution({'env_id':'science'},changed,negative,2)['positive_rollout_sha256'])

    def test_different_task_or_environment_cannot_be_attributed(self):
        from subnet.backend_jobs import pair_attribution
        with self.assertRaisesRegex(ValueError,'binding'):
            pair_attribution({'env_id':'science'},{'env_id':'science','index':7},{'env_id':'science','index':8},0)
        with self.assertRaisesRegex(ValueError,'binding'):
            pair_attribution({'env_id':'science'},{'env_id':'logic','index':7},{'env_id':'logic','index':7},0)

class BackendJobAuthorization(unittest.TestCase):
    def setUp(self):
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        self.files={'config.json':'1'*64,'model.safetensors':'2'*64}
        self.manifest={'epoch':'nonpayable-gpu-test','checkpoint':{'id':file_map(self.files),'files':self.files},
            'model_runtime_revision':REVISION,'numerical_policy':NUMERICAL_POLICY,'backend_profile':BACKEND_PROFILE,'K':1,'L':1,'audit_policy':{'mode':'full'}}
        self.job={'schema':1,'job_id':'verify-1','role':'verify','created_at':10,'expires_at':100,'manifest':self.sign(self.manifest),
            'source_files':{n:'a'*64 for n in SOURCE_FILES},'runtime_versions':{'torch':'approved','transformers':'approved','toploc':'approved'},
            'submissions':[{'url':'https://account.r2.cloudflarestorage.com/bucket/artifact?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test','sha256':'3'*64}]}
    def sign(self,payload,key=None):
        key=key or self.key
        return {'payload':copy.deepcopy(payload),'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(canonical(payload)).signature).decode()}
    def test_honest_signed_verify_policy(self):
        job,manifest=validate(self.sign(self.job),self.authority,now=50)
        self.assertEqual(job['role'],'verify');self.assertEqual(manifest['checkpoint']['id'],file_map(self.files))
    def test_wrong_operator_rejected_before_artifact_reads(self):
        with patch('subnet.backend_jobs.checkpoint') as checkpoint,patch('subnet.backend_jobs.get_object') as fetch:
            with self.assertRaises(ValueError):execute(self.sign(self.job,SigningKey.generate()),self.authority,'unused')
            checkpoint.assert_not_called();fetch.assert_not_called()
    def test_changed_unsigned_role_is_rejected(self):
        envelope=self.sign(self.job);envelope['payload']['role']='upload'
        with self.assertRaises(Exception):validate(envelope,self.authority,now=50)
    def test_unapproved_manifest_signer(self):
        self.job['manifest']=self.sign(self.manifest,SigningKey.generate())
        with self.assertRaises(ValueError):validate(self.sign(self.job),self.authority,now=50)
    def test_invalid_job_roles_or_expiry(self):
        for field,value in [('role','set_weights'),('role','shell'),('expires_at',50),('job_id','../../wallet')]:
            job=copy.deepcopy(self.job);job[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):validate(self.sign(job),self.authority,now=50)
    def test_no_cross_backend_tolerance_relaxation(self):
        for field,value in [('model_runtime_revision','cpu-float32'),('numerical_policy',{**NUMERICAL_POLICY,'logprob_atol':1}),('backend_profile',{**BACKEND_PROFILE,'tf32':True})]:
            manifest=copy.deepcopy(self.manifest);manifest[field]=value;self.job['manifest']=self.sign(manifest)
            with self.subTest(field=field),self.assertRaises(ValueError):validate(self.sign(self.job),self.authority,now=50)
    def test_checkpoint_identity_and_unsafe_files(self):
        for cp in [{'files':self.files,'id':'a'*64},{'files':{'config.json':'1'*64,'../model.safetensors':'2'*64},'id':'a'*64}]:
            self.manifest['checkpoint']=cp;self.job['manifest']=self.sign(self.manifest)
            with self.assertRaises(ValueError):validate(self.sign(self.job),self.authority,now=50)
    def test_sampling_and_missing_proofs_are_not_training_authorization(self):
        self.manifest['audit_policy']={'mode':'sample','count':1};self.job['role']='train';self.job['steps']=1;self.job['manifest']=self.sign(self.manifest)
        with self.assertRaises(ValueError):validate(self.sign(self.job),self.authority,now=50)
    def test_no_redirect_proxy_or_arbitrary_remote_upload(self):
        for url in ['http://a.r2.cloudflarestorage.com/a','https://evil.test/a?X-Amz-Signature=s','https://a.r2.cloudflarestorage.com.evil.test/a','https://user:secret@a.r2.cloudflarestorage.com/a']:
            with self.subTest(url=url),self.assertRaises(ValueError):r2_url(url,'GET')
    def test_all_worker_sources_must_be_pinned(self):
        del self.job['source_files'][SOURCE_FILES[0]]
        with self.assertRaises(ValueError):validate(self.sign(self.job),self.authority,now=50)
    def test_upload_requires_one_put_per_exact_file(self):
        self.job['role']='upload';self.job['put_urls']={'config.json':self.job['submissions'][0]['url']}
        with self.assertRaises(ValueError):validate(self.sign(self.job),self.authority,now=50)
    def test_heldout_rejects_curated_candidate_policy(self):
        self.job.update(role='evaluate',heldout=[dict(env_id='original',indices=[2],seeds=[202],harness={'policy':'candidates','turn_overrides':{}})])
        with self.assertRaises(ValueError):validate(self.sign(self.job),self.authority,now=50)

if __name__=='__main__':unittest.main()

class BackendFullAudit(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from types import SimpleNamespace
        self.manifest=dict(epoch='nonpayable-test',checkpoint={'id':'approved'},K=1,L=1,max_batches=4,environment={'id':'env'},harness={},indices=[0])
        self.batch=dict(schema=2,epoch='nonpayable-test',checkpoint='approved',env_id='env',environment_version='v1',index=0,sample_index=0,
            rollouts=[dict(index=0,env_id='env',classification=c,reward=r,turns=[{'output':[i]}]) for c,r,i in [('positive',1,1),('negative',0,2)]])
        self.arrays=[[np.zeros((1,2),dtype=np.float32)],[np.zeros((1,2),dtype=np.float32)]]
        self.runtime=SimpleNamespace(spec=SimpleNamespace(version='v1'),verify=lambda doc,arrays:True)
        self.runtime.for_environment=lambda spec,harness:self.runtime
    def run_audit(self):
        from subnet.backend_jobs import audit
        from subnet.batches import pack
        return audit(pack([(self.batch,self.arrays)]),self.manifest,self.runtime)
    def test_full_pair_becomes_training_eligible(self):
        report,pairs=self.run_audit();self.assertTrue(report['outcomes'][0]['fully_audited']);self.assertEqual(len(pairs),1)
    def test_wrong_checkpoint_does_not_train(self):
        self.batch['checkpoint']='other-model';report,pairs=self.run_audit();self.assertFalse(report['outcomes'][0]['valid']);self.assertEqual(pairs,[])
    def test_failed_inference_proof_does_not_train(self):
        def reject(doc,arrays):raise ValueError('TOPLOC mismatch')
        self.runtime.verify=reject;report,pairs=self.run_audit();self.assertFalse(report['outcomes'][0]['valid']);self.assertEqual(pairs,[])
    def test_duplicate_tokens_do_not_train(self):
        self.batch['rollouts'][1]['turns']=self.batch['rollouts'][0]['turns'];report,pairs=self.run_audit();self.assertFalse(report['outcomes'][0]['valid']);self.assertEqual(pairs,[])
    def test_forged_class_counts_do_not_train(self):
        self.batch['rollouts'][1]['classification']='positive';report,pairs=self.run_audit();self.assertFalse(report['outcomes'][0]['valid']);self.assertEqual(pairs,[])

class FreshSourceLoading(unittest.TestCase):
    def test_ignores_forged_cached_bytecode(self):
        import tempfile,py_compile,importlib.util
        from pathlib import Path
        from subnet.backend_jobs import FreshSourceFinder
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'subnet').mkdir();path=root/'subnet/cache_probe.py'
            path.write_text('value = "forged"\n');py_compile.compile(str(path),doraise=True)
            path.write_text('value = "approved source"\n')
            spec=FreshSourceFinder(root).find_spec('subnet.cache_probe');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.assertEqual(module.value,'approved source')

class MiningEpochWindow(unittest.TestCase):
    sign = BackendJobAuthorization.sign
    def setUp(self):
        BackendJobAuthorization.setUp(self)
        self.manifest.update(start=10,deadline=90)
        self.job.update(role='mine',miner_id='b'*64,search_budget=1,seed_start=100,
            capability={'put_url':self.job['submissions'][0]['url'],'headers':{'Content-Type':'application/octet-stream'}})
        self.job['manifest']=self.sign(self.manifest)
    def test_signed_epoch_boundary_is_half_open(self):
        from subnet.backend_jobs import mining_window
        mining_window(self.manifest,now=10);mining_window(self.manifest,now=89.999)
        for now in (9.999,90,91):
            with self.subTest(now=now),self.assertRaisesRegex(ValueError,'epoch window closed'):mining_window(self.manifest,now=now)
    def test_expired_mining_manifest_rejected_before_artifact_reads(self):
        with patch('subnet.backend_jobs.time.time',return_value=90),patch('subnet.backend_jobs.checkpoint') as read:
            with self.assertRaisesRegex(ValueError,'epoch window closed'):execute(self.sign(self.job),self.authority,'unused')
            read.assert_not_called()
    def test_deadline_crossing_before_search_does_not_generate(self):
        import tempfile
        from types import SimpleNamespace
        runtime=SimpleNamespace(for_environment=lambda *a:runtime,rollout=unittest.mock.Mock())
        from subnet.gpu_runtime import GPURuntime
        calls=iter([50])
        def clock():return next(calls,90)
        with tempfile.TemporaryDirectory() as directory,patch('subnet.backend_jobs.time.time',side_effect=clock),patch('subnet.backend_jobs.digest',return_value='a'*64),patch('subnet.backend_jobs.version',return_value='approved'),patch('subnet.backend_jobs.install_source_loader'),patch('subnet.backend_jobs.checkpoint',return_value=directory),patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}),patch('subnet.protocol.entries',return_value=[dict(spec={},harness={},indices=[0])]):
            with self.assertRaisesRegex(ValueError,'epoch window closed'):execute(self.sign(self.job),self.authority,directory,runtime_factory=lambda *a:runtime)
        runtime.rollout.assert_not_called()

class CumulativeMining(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        self.now=20
        self.manifest=dict(start=10,deadline=90,epoch='nonpayable-cumulative',checkpoint={'id':'approved'},K=1,L=1,max_batches=3)
        self.job=dict(search_budget=4,seed_start=0)
        self.definitions=[dict(env_id='one',spec={},harness={},indices=[0,1]),dict(env_id='two',spec={},harness={},indices=[2])]
        self.runtime=SimpleNamespace(spec=SimpleNamespace(version='v1'))
        self.runtime.for_environment=lambda *args:self.runtime
        self.runtime.rollout=lambda index,seed:({'classification':'positive' if seed%2==0 else 'negative','turns':[{'output':[index,seed]}]},[])
        self.uploads=[]
    def run_miner(self,upload=None):
        from subnet.backend_jobs import mine_cumulative
        import json
        def pack(rows):return json.dumps([b['index'] for b,a in rows]).encode()
        with patch('subnet.protocol.entries',return_value=self.definitions),patch('subnet.batches.pack',side_effect=pack):
            return mine_cumulative(self.runtime,self.manifest,self.job,upload or (lambda data,timeout:self.uploads.append(data)),clock=lambda:self.now)
    def test_each_complete_batch_overwrites_with_cumulative_snapshot(self):
        data,report=self.run_miner()
        self.assertEqual(self.uploads,[b'[0]',b'[0, 1]',b'[0, 1, 2]'])
        self.assertEqual(data,self.uploads[-1]);self.assertEqual(report['cumulative_uploads'],3)
    def test_later_slow_search_keeps_acknowledged_snapshot(self):
        honest=self.runtime.rollout
        def rollout(index,seed):
            if index==1:self.now=91
            return honest(index,seed)
        self.runtime.rollout=rollout
        data,report=self.run_miner()
        self.assertEqual(self.uploads,[b'[0]']);self.assertEqual(data,b'[0]')
        self.assertEqual(report['batches'],1);self.assertTrue(report['search_stopped_at_deadline'])
    def test_max_batches_applies_across_environment_groups(self):
        self.manifest['max_batches']=1
        data,report=self.run_miner()
        self.assertEqual(self.uploads,[b'[0]']);self.assertEqual(len(report['search']),1)
    def test_owned_subset_keeps_public_manifest_and_searches_only_selected_tasks(self):
        self.job['mining_subset']={'one':[1]}
        data,report=self.run_miner()
        self.assertEqual(self.uploads,[b'[1]']);self.assertEqual(data,self.uploads[0])
        self.assertEqual(self.definitions[0]['indices'],[0,1])
        self.assertEqual([(r['env_id'],r['index'])for r in report['search']],[('one',1)])
    def test_owned_subset_cannot_authorize_other_environment_or_heldout(self):
        from subnet.backend_jobs import mining_definitions
        for subset in [{'unknown':[0]},{'one':[99]},{'one':[True]},{'one':[0,0]},{'one':[]},{}]:
            with self.subTest(subset=subset),patch('subnet.protocol.entries',return_value=self.definitions):
                with self.assertRaises(ValueError):mining_definitions(self.manifest,{'mining_subset':subset})
    def test_owned_subset_projects_only_the_pinned_indexed_policy(self):
        from subnet.backend_jobs import mining_definitions
        from subnet.sample_harness import VERSION,resolve
        a=dict(version='text-tools-v1',policy='autoregressive',max_output_tokens=128)
        b=dict(a,max_output_tokens=256)
        self.definitions[0]['harness']=dict(version=VERSION,by_index={'0':a,'1':b})
        with patch('subnet.protocol.entries',return_value=self.definitions):
            selected=mining_definitions(self.manifest,{'mining_subset':{'one':[1]}})[0]
        self.assertEqual(set(selected['harness']['by_index']),{'1'})
        self.assertEqual(resolve(selected['harness'],1,selected['indices'])['max_output_tokens'],256)
        self.assertEqual(set(self.definitions[0]['harness']['by_index']),{'0','1'})
    def test_failed_put_is_not_reported_as_success(self):
        def fail(data,timeout):raise ValueError('R2 PUT status 403')
        with self.assertRaisesRegex(ValueError,'R2 PUT status'):self.run_miner(fail)
    def test_capacity_limit_preserves_last_acknowledged_upload(self):
        from subnet.backend_jobs import mine_cumulative
        from subnet.batches import UploadBudgetExceeded
        with patch('subnet.protocol.entries',return_value=self.definitions),patch('subnet.batches.pack',side_effect=[b'first-complete',UploadBudgetExceeded('full')]):
            data,report=mine_cumulative(self.runtime,self.manifest,self.job,lambda data,timeout:self.uploads.append(data),clock=lambda:self.now)
        self.assertEqual(data,b'first-complete');self.assertEqual(self.uploads,[data])
        self.assertEqual(report['batches'],1);self.assertTrue(report['search_stopped_at_capacity'])
        self.assertEqual(report['search'][-1]['submission_status'],'exceeds_cumulative_upload_budget')
    def test_oversize_first_batch_does_not_prevent_smaller_later_batch(self):
        from subnet.backend_jobs import mine_cumulative
        from subnet.batches import UploadBudgetExceeded
        self.manifest['max_batches']=1
        with patch('subnet.protocol.entries',return_value=self.definitions),patch('subnet.batches.pack',side_effect=[UploadBudgetExceeded('full'),b'smaller-complete']):
            data,report=mine_cumulative(self.runtime,self.manifest,self.job,lambda data,timeout:self.uploads.append(data),clock=lambda:self.now)
        self.assertEqual(self.uploads,[b'smaller-complete']);self.assertEqual(data,self.uploads[0])
        self.assertEqual(report['search'][1]['index'],1);self.assertFalse(report['search_stopped_at_capacity'])
    def test_rollout_finishing_in_reserve_does_not_start_late_overwrite(self):
        honest=self.runtime.rollout
        def rollout(index,seed):
            if index==1 and seed==1:self.now=81
            return honest(index,seed)
        self.runtime.rollout=rollout
        data,report=self.run_miner()
        self.assertEqual(self.uploads,[b'[0]']);self.assertEqual(report['batches'],1)


class EmptyBoundedMining(CumulativeMining):
    def run_empty(self,label):
        from subnet.backend_jobs import mine_cumulative
        self.runtime.rollout=lambda index,seed:({'classification':label,'turns':[{'output':[index,seed]}]},[])
        with patch('subnet.protocol.entries',return_value=self.definitions),patch('subnet.batches.pack') as pack:
            data,report=mine_cumulative(self.runtime,self.manifest,self.job,lambda data,timeout:self.uploads.append(data),clock=lambda:self.now,allow_empty=True)
            pack.assert_not_called()
        return data,report
    def test_all_positive_is_truthful_terminal_empty_not_failed_search(self):
        data,report=self.run_empty('positive')
        self.assertIsNone(data);self.assertEqual(self.uploads,[])
        self.assertEqual(report['batches'],0);self.assertEqual(report['cumulative_uploads'],0)
        self.assertEqual(report['mining_status'],'no_complete_KL_batch')
        self.assertEqual([(s['attempts'],s['positive'],s['negative']) for s in report['search']],[(4,1,0)]*3)
        self.assertEqual([(s['observed_positive'],s['observed_negative']) for s in report['search']],[(4,0)]*3)
    def test_all_negative_retains_missing_class_and_counts(self):
        data,report=self.run_empty('negative')
        self.assertIsNone(data);self.assertEqual(self.uploads,[])
        self.assertEqual([(s['attempts'],s['positive'],s['negative']) for s in report['search']],[(4,0,1)]*3)
        self.assertEqual([(s['observed_positive'],s['observed_negative']) for s in report['search']],[(0,4)]*3)
    def test_infrastructure_exception_is_not_an_empty_success(self):
        from subnet.backend_jobs import mine_cumulative
        self.runtime.rollout=unittest.mock.Mock(side_effect=RuntimeError('native worker unavailable'))
        with patch('subnet.protocol.entries',return_value=self.definitions),self.assertRaisesRegex(RuntimeError,'native worker unavailable'):
            mine_cumulative(self.runtime,self.manifest,self.job,lambda *args:self.fail('unexpected upload'),clock=lambda:self.now,allow_empty=True)


class EmptyMineExecution(MiningEpochWindow):
    def test_zero_batch_executor_succeeds_without_put_or_submission_file(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        runtime=SimpleNamespace(spec=SimpleNamespace(version='v1'))
        runtime.for_environment=lambda *args:runtime
        runtime.rollout=lambda index,seed:({'classification':'positive','turns':[{'output':[seed]}]},[])
        definition=dict(env_id='one',spec={},harness={},indices=[0])
        with tempfile.TemporaryDirectory() as directory,patch('subnet.backend_jobs.time.time',return_value=50),patch('subnet.backend_jobs.digest',return_value='a'*64),patch('subnet.backend_jobs.version',return_value='approved'),patch('subnet.backend_jobs.install_source_loader'),patch('subnet.backend_jobs.checkpoint',return_value=directory),patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}),patch('subnet.protocol.entries',return_value=[definition]),patch('requests.put') as put:
            report=execute(self.sign(self.job),self.authority,directory,runtime_factory=lambda *args:runtime)
            put.assert_not_called()
            self.assertEqual(report['batches'],0);self.assertEqual(report['submission_size'],0);self.assertIsNone(report['submission_sha256'])
            self.assertEqual(report['mining_status'],'no_complete_KL_batch')
            self.assertEqual(report['search'][0]['observed_positive'],1)
            self.assertFalse((Path(directory)/'jobs'/self.job['job_id']/'submission.zip').exists())
            self.assertNotIn('training',report);self.assertNotIn('scores',report)
