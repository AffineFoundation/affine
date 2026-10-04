"""CPU integration of signed lineage, durable authority commit and recovery."""
import base64
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import torch
from nacl.signing import SigningKey
from botocore.exceptions import ClientError

from subnet import forced_sampling as sampling
from subnet.backend_jobs import (BACKEND_PROFILE,NUMERICAL_POLICY,REVISION,SOURCE_FILES,
    file_map,validate,canonical)
from subnet.persistent_cpu_adamw import POLICY,HYPERPARAMETERS,PersistentCPUAdamW,parameter_inventory,genesis,sha
from subnet.persistent_training_state import resource_plan,admit_resources,export_state,cgroup_headroom
from subnet.persistent_training_protocol import (EXECUTION_FILES,opening_binding,prepare_job,
    validate_job,validate_output,validate_report,independently_commit,state_pointer,validate_parent)
from subnet.persistent_training_worker import admitted_submission,capacity_requirement,report_updates
from training_receipt_fixtures import transport_fixture
from subnet.training_receipts import VERSION as INPUT_POLICY, receipt_inventory
from subnet.distributed_roles import Coordinator
import sqlite3
from subnet.persistent_training_controller import commit_latest,train
from subnet.training_policy import coverage_manifest,epoch_policy
from subnet.remote_backend import RemoteJobs,save


class MemoryBucket:
    def __init__(self):self.objects={};self.events=[];self.name='test';self.client=self
    def get(self,key):
        if key not in self.objects:raise ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject')
        return self.objects[key]
    def json(self,key,value):self.events.append(('authority-write',key));self.objects[key]=canonical(value)
    def get_object(self,*,Bucket,Key):return {'Body':io.BytesIO(self.get(Key))}
    def presign(self,key,operation='get_object',expires=3600):
        return 'https://test.r2.cloudflarestorage.com/'+key+'?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test'


class PersistentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        self.bucket=MemoryBucket();self.controller=SimpleNamespace(state=self.root,bucket=self.bucket,
            authority=SimpleNamespace(id=self.authority),signed=self.sign)
        (self.root/'roles').mkdir()
        self.parameters=[('weight',torch.nn.Parameter(torch.tensor([.02,.02],dtype=torch.bfloat16)))]
        _,self.inventory=parameter_inventory(self.parameters)
        files={'config.json':'1'*64,'model.safetensors':'2'*64};self.cp=dict(id=file_map(files),files=files)
        admission=dict(parameters=self.inventory,parameters_sha256=sha(self.inventory),source_sha256='7'*64,
            gpu_qualification_sha256='8'*64,genesis_round=10,genesis_checkpoint=self.cp['id'],
            genesis_sha256=sha(genesis(self.inventory,self.cp['id'])))
        self.config=dict(training_policy=POLICY,source_bundle={'sha256':'7'*64},persistent_training_admission=admission)
        self.status=dict(round=10,checkpoint=self.cp)
        fixture=transport_fixture(self.key,policy=POLICY)
        self.manifest=fixture['manifest'];self.rollouts=fixture['batch']['rollouts'];self.batch=fixture['batch']
        self.miner=fixture['miner'];self.receipts=fixture['receipts'];self.challenge=fixture['challenge']
        self.manifest['trainer_state_binding']=opening_binding(self.config,self.status,self.manifest['epoch'])
        self.submission=fixture['submission'];self.data=fixture['data'];self.verifier_audit=dict(fixture['audit'],remote_job_id='synthetic-verify')
        request=fixture['worker_request'];verify_job=fixture['verify_job'];report=request['payload']['report']
        self.queue=Coordinator(self.root/'queue.sqlite',self.authority,{request['signer']:['verify']})
        with sqlite3.connect(self.queue.path)as db:
            db.execute('INSERT INTO jobs(id,digest,envelope,role,expires,status,worker,report,report_digest,report_request) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (verify_job['payload']['job_id'],sha(verify_job['payload']),canonical(verify_job).decode(),'verify',100,'complete',request['signer'],
                 canonical(report).decode(),sha(report),canonical(request).decode()))

    def sign(self,payload):
        return dict(payload=copy.deepcopy(payload),signer=self.authority,
            signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())

    def job(self,manifest=None,identifier='v4-train',steps=3):
        manifest=manifest or self.manifest
        return dict(schema=1,job_id=identifier,role='train',created_at=22,expires_at=100,
            manifest=self.sign(manifest),source_files={n:'b'*64 for n in set(SOURCE_FILES)|set(EXECUTION_FILES)},
            runtime_versions=dict(torch='approved',transformers='approved',toploc='approved'),
            steps=steps,training_policy=POLICY,training_input_policy=INPUT_POLICY,submissions=[self.submission],
            persistent_training=prepare_job(self.controller,manifest,identifier,steps,3600))

    def report(self,job=None):
        job=job or self.job();manifest=job['manifest']['payload'];binding=manifest['trainer_state_binding']
        plan=resource_plan(self.inventory,bf16_export_bytes=100,ram_reserve_bytes=0,disk_reserve_bytes=0)
        admission=admit_resources(self.root,plan)
        optimizer=PersistentCPUAdamW(self.parameters,self.cp['id'],approved_genesis=binding['genesis'],
            approved_genesis_sha256=binding['genesis_sha256'],resource_admission=admission)
        from subnet.task_normalized_training import task_groups
        pairs,tasks,groups,identities=task_groups([(manifest['environments'][0],*self.rollouts)],job['steps'],manifest['training_coverage']['seed'])
        updates=[]
        for step in range(job['steps']):
            self.parameters[0][1].grad=torch.ones_like(self.parameters[0][1]);optimizer.step()
            updates.append(dict(training_policy=POLICY,steps=1,epoch_optimizer_step=step+1,
                global_optimizer_step=step+1,input_checkpoint=self.cp['id'],reference_scope='immutable-BF16-epoch-input',
                unique_tasks=1,unique_verified_pairs=1,gradient_tasks=1,gradient_pairs=1,cumulative_unique_gradient_tasks=1,
                loss=.6931471805599453,gradient_norm_before_clip=1.,full_model_finetune=True,gradient_tensors=1,
                hyperparameters=copy.deepcopy(HYPERPARAMETERS),precision=copy.deepcopy(optimizer.last_update),
                task_weight_rule='mean-pair-within-task-then-mean-task-within-group',optimizer_lifecycle='persistent-across-epochs',
                pairs=[dict(pair_index=0,task_index=0,task_sha256=tasks[0]['task_sha256'],pair_sha256=identities[0],
                    gradient_weight=1.,reference_margin=0.,margin_before=0.,loss=.6931471805599453)]))
        diagnostics=dict(training_policy=POLICY,optimizer_steps=job['steps'],global_optimizer_step_before=0,
            global_optimizer_step_after=job['steps'],epoch=manifest['epoch'],input_checkpoint=self.cp['id'],
            task_count=1,pair_count=1,heldout_gain_claimed=False,state_publication_required=True,
            training_pair_margin_before=[0.],training_pair_margin_after=[0.],training_pair_margin_delta=[0.],
            master_state_updated=True,inference_tensors_changed_during_updates=False)
        namespace=job['persistent_training']['output_namespace']
        def publish(name,path):self.bucket.events.append(('shard-put',name));self.bucket.objects[namespace+'/'+name]=path.read_bytes()
        def readback(name):
            self.bucket.events.append(('worker-readback',name));yield self.bucket.get(namespace+'/'+name)
        def stage(document):
            self.bucket.objects[namespace+'/staged-state.json']=canonical(document)
            return dict(descriptor_sha256=sha(document),durable_readback_verified=True)
        descriptor,_=export_state(optimizer,epoch=manifest['epoch'],inference_checkpoint=self.cp['id'],workspace=self.root,
            publish_shard=publish,readback_shard=readback,commit_descriptor=stage,resource_admission=admission)
        audit=dict(epoch=manifest['epoch'],submission_sha256='3'*64,accepted=[self.batch],
            outcomes=[dict(batch=0,fully_audited=True,valid=True)],training_eligibility='fully-audited-only',
            sampling_assurance=sampling.assurance(manifest))
        report=dict(job_id=job['job_id'],role='train',operator=self.authority,success=True,chain_transactions=False,
            job_sha256=sha(job),epoch=manifest['epoch'],checkpoint=self.cp['id'],backend_profile=BACKEND_PROFILE,
            numerical_policy=NUMERICAL_POLICY,source_files=job['source_files'],runtime_versions=job['runtime_versions'],completed_at=40,
            training=dict(training_policy=POLICY,steps=job['steps'],weights_changed=False,global_step_before=0,
                global_step_after=job['steps'],state_updated=True,training_input_policy=INPUT_POLICY,
                trainer_verification_performed=False,all_pairs_authenticated_verifier_receipts=True,
                parameter_values_sha256_before='d'*64,parameter_values_sha256_after='d'*64,updates=updates,
                persistent_diagnostics=diagnostics),
            new_checkpoint=dict(self.cp,path='/original-final'),audits=[],training_admissions=[dict(
                version=INPUT_POLICY,epoch=manifest['epoch'],submission_sha256=self.submission['sha256'],
                verifier_receipt_sha256=sha(self.submission['verifier_receipt']),accepted_batch_sha256=self.submission['accepted_batch_sha256'],
                accepted=[self.batch],original_verify_job_id='synthetic-verify',
                original_report_sha256=self.submission['verifier_receipt']['payload']['original_report_sha256'],
                trainer_verification_performed=False,verification_performed_by='registered-verifier')],
            persistent_training_state=dict(namespace=namespace,descriptor_sha256=sha(descriptor),descriptor=descriptor))
        return report,job

    def test_new_signed_policy_requires_exact_genesis_source_and_state_execution_pins(self):
        job=self.job();validate(self.sign(job),self.authority,now=30)
        self.assertEqual(epoch_policy({'training_policy':POLICY}),POLICY)
        for mutation in ('source','genesis','steps','name','missing-pin','missing-sampling','namespace','capability','batch'):
            changed=copy.deepcopy(job);manifest=changed['manifest']['payload']
            if mutation=='source':manifest['trainer_state_binding']['source_sha256']='c'*64
            elif mutation=='genesis':manifest['trainer_state_binding']['genesis_sha256']='c'*64
            elif mutation=='steps':changed['persistent_training']['global_step_after']=4
            elif mutation=='name':manifest['trainer_state_binding']['parameters'][0]['name']='different'
            elif mutation=='missing-pin':del changed['source_files'][EXECUTION_FILES[0]]
            elif mutation=='missing-sampling':del manifest['sampling_contract']
            elif mutation=='namespace':changed['persistent_training']['output_namespace']='private/other'
            elif mutation=='capability':changed['persistent_training']['output_shards']['state-000000.safetensors']['put_url']=self.bucket.presign('private/different','put_object')
            else:changed['submissions'][0]['accepted_batch_sha256']=[]
            changed['manifest']=self.sign(manifest)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate(self.sign(changed),self.authority,now=30)

    def test_no_automatic_genesis_for_later_round_or_missing_committed_parent(self):
        for status in (dict(self.status,round=11),dict(self.status,persistent_state_committed=True),
                       dict(self.status,checkpoint={'id':'9'*64})):
            with self.subTest(status=status),self.assertRaises(ValueError):opening_binding(self.config,status,'next')

    def test_same_bf16_output_commits_advanced_state_authority_last(self):
        report,job=self.report();descriptor=validate_report(report,job,self.manifest)
        pointer=independently_commit(self.controller,report,job,self.manifest)
        self.assertEqual(pointer['inference_checkpoint'],self.cp['id']);self.assertEqual(pointer['optimizer_steps'],3)
        self.assertFalse(report['training']['weights_changed'])
        self.assertTrue(self.bucket.events[-1][0]=='authority-write')
        namespace=job['persistent_training']['output_namespace']
        self.assertIn(namespace+'/authority-state.json',self.bucket.objects)
        self.assertEqual(pointer['descriptor_sha256'],sha(descriptor))
        before=len(self.bucket.events)
        self.assertEqual(independently_commit(self.controller,report,job,self.manifest),pointer)
        self.assertEqual(len(self.bucket.events),before)
        commit_latest(self.controller,self.manifest['trainer_state_binding'],pointer)
        commit_latest(self.controller,self.manifest['trainer_state_binding'],pointer)
        next_binding=opening_binding(self.config,dict(self.status,round=11,trainer_state=pointer,persistent_state_committed=True),'next')
        self.assertEqual(next_binding['global_step_before'],3);self.assertIsNone(next_binding['genesis'])
        parent=validate_parent(json.loads(self.bucket.get(pointer['descriptor_key'])),next_binding,self.authority)
        self.assertEqual(parent['optimizer_steps'],3)

    def test_self_consistent_descriptor_wrong_parent_counters_and_model_cannot_pass(self):
        report,job=self.report()
        for field,value in [('parent_state_sha256','9'*64),('genesis_sha256','9'*64),
                            ('input_checkpoint','9'*64),('epoch','other'),('optimizer_steps',6)]:
            changed=copy.deepcopy(report['persistent_training_state']['descriptor']);changed[field]=value
            if field=='optimizer_steps':changed['parameter_steps']={'weight':6}
            with self.subTest(field=field),self.assertRaises(ValueError):validate_output(changed,job,self.manifest)
        forged=copy.deepcopy(report);forged['training']['weights_changed']=True
        with self.assertRaises(ValueError):validate_report(forged,job,self.manifest)

    def test_corrupt_durable_shard_never_creates_authority_commit(self):
        report,job=self.report();namespace=job['persistent_training']['output_namespace']
        name=report['persistent_training_state']['descriptor']['shards'][0]['name']
        self.bucket.objects[namespace+'/'+name]=b'corrupt'
        with self.assertRaises(ValueError):independently_commit(self.controller,report,job,self.manifest)
        self.assertNotIn(namespace+'/authority-state.json',self.bucket.objects)
        self.assertFalse(any(e[0]=='authority-write'for e in self.bucket.events))

    def test_wrong_parent_namespace_or_old_state_same_checkpoint_refused(self):
        report,job=self.report();pointer=independently_commit(self.controller,report,job,self.manifest)
        next_status=dict(self.status,round=11,trainer_state=pointer,persistent_state_committed=True)
        next_manifest=dict(self.manifest,epoch='nonpayable-v4-11')
        next_manifest['trainer_state_binding']=opening_binding(self.config,next_status,next_manifest['epoch'])
        next_job=self.job(next_manifest,'v4-next');validate_job(next_job,next_manifest,self.authority)
        for field,value in [('optimizer_steps',2),('descriptor_sha256','e'*64),('namespace','private/trainer-state/older/job')]:
            changed=copy.deepcopy(next_job);changed['manifest']['payload']['trainer_state_binding']['parent'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_job(changed,changed['manifest']['payload'],self.authority)
        stale=dict(pointer,descriptor_sha256='e'*64,optimizer_steps=2)
        commit_latest(self.controller,self.manifest['trainer_state_binding'],pointer)
        with self.assertRaises(ValueError):commit_latest(self.controller,dict(next_manifest['trainer_state_binding'],parent=stale),dict(pointer,optimizer_steps=6))

    def test_training_input_retired_only_after_exact_verifier_receipt_admission(self):
        path=self.root/'submission.zip';path.write_bytes(self.data)
        obj=copy.deepcopy(self.submission);obj['accepted_batch_sha256']=['e'*64]
        with self.assertRaises(ValueError):admitted_submission(path,obj,self.manifest,self.authority)
        self.assertTrue(path.exists())
        path.write_bytes(b'tampered')
        with self.assertRaises(ValueError):admitted_submission(path,self.submission,self.manifest,self.authority)
        self.assertTrue(path.exists());path.write_bytes(self.data)
        with patch('subnet.backend_jobs.audit',side_effect=AssertionError('trainer re-verification forbidden')):
            actual,pairs=admitted_submission(path,self.submission,self.manifest,self.authority)
        self.assertEqual(len(pairs),1);self.assertEqual(actual['accepted'],[self.batch]);self.assertFalse(path.exists())

    def test_capacity_uses_one_input_and_one_shard_not_total_state_disk(self):
        probe=dict(free_bytes=30*1024**3,available_ram_bytes=50*1024**3)
        admitted=capacity_requirement(self.manifest,probe,checkpoint_bytes=100,missing_input=True)
        self.assertTrue(admitted['state_streaming']);self.assertFalse(admitted['full_state_disk_hydration'])
        self.assertFalse(admitted['gpu_forward_backward_capacity_qualified'])
        for change in ({'free_bytes':10},{'available_ram_bytes':10}):
            with self.subTest(change=change),self.assertRaises(ValueError):capacity_requirement(self.manifest,dict(probe,**change),checkpoint_bytes=100,missing_input=True)

    def test_cache_aware_ram_admission_excludes_dirty_mapped_and_pinned_pages(self):
        stats=dict(file=218,inactive_file=109,active_file=109,file_dirty=4,file_writeback=5,
                   file_mapped=6,unevictable=7,shmem=8)
        observed=cgroup_headroom(248,219,stats)
        self.assertEqual(observed['hard_headroom_bytes'],29)
        self.assertEqual(observed['conservative_clean_inactive_file_bytes'],79)
        self.assertEqual(observed['usable_bytes'],108)
        self.assertFalse(observed['active_file_cache_counted']);self.assertFalse(observed['drop_caches_requested'])
        self.assertEqual(cgroup_headroom(248,219,dict(stats,active_file=10_000))['usable_bytes'],108)
        self.assertEqual(cgroup_headroom(248,219,dict(stats,file_dirty=109))['usable_bytes'],29)
        self.assertEqual(cgroup_headroom(248,219,{})['usable_bytes'],29)
        for change in ({'file_dirty':-1},{'inactive_file':True}):
            with self.assertRaises(ValueError):cgroup_headroom(248,219,dict(stats,**change))

    def test_completed_original_training_resume_never_relaunches_or_steps(self):
        report,job=self.report();record=dict(job_id=job['job_id'],role='train',job_sha256=sha(job),
            manifest_sha256=sha(self.manifest),source_files=job['source_files'],runtime_versions=job['runtime_versions'])
        save(self.root/'roles'/'v4-train.json',record);save(self.root/'roles'/(job['job_id']+'-job.json'),self.sign(job))
        save(self.root/'roles'/(job['job_id']+'-report.json'),report)
        jobs=RemoteJobs.__new__(RemoteJobs);jobs.controller=self.controller;jobs.state=self.root/'roles'
        jobs.remote_status=Mock(side_effect=AssertionError('completed local report needs no remote launch'))
        jobs.command=Mock(side_effect=AssertionError('no new train process'))
        result=jobs.training_resume('v4-train',self.manifest,[self.submission],3,None)
        self.assertFalse(result['new_training_started']);self.assertEqual(result['original_job_id'],job['job_id'])
        recovered=jobs.run('v4-train','train',self.manifest,submissions=[self.submission],steps=3,training_policy=POLICY)
        self.assertEqual(recovered['persistent_training_state']['descriptor_sha256'],report['persistent_training_state']['descriptor_sha256'])
        jobs.command.assert_not_called()
        changed=dict(self.submission,accepted_batch_sha256=['e'*64])
        with self.assertRaises(ValueError):jobs.training_resume('v4-train',self.manifest,[changed],3,None)

    def test_controller_same_bf16_checkpoint_recovers_original_state_without_retraining(self):
        report,job=self.report()
        save(self.root/(self.manifest['epoch']+'-scores.json'),{'receipts':self.receipts})
        save(self.root/(self.manifest['epoch']+'-audit-challenge.json'),self.challenge)
        reports={self.miner:self.verifier_audit}
        launches=[]
        def run(label,role,manifest,cache,**fields):
            self.assertEqual(manifest,self.manifest);self.assertEqual(fields['training_policy'],POLICY)
            launches.append(job['job_id'])
            record=dict(job_id=job['job_id'],role='train',job_sha256=sha(job),manifest_sha256=sha(manifest))
            save(self.root/'roles'/(self.manifest['epoch']+'-train.json'),record)
            save(self.root/'roles'/(job['job_id']+'-job.json'),self.sign(job))
            save(self.root/'roles'/(job['job_id']+'-report.json'),report)
            return report
        self.controller.jobs=SimpleNamespace(queue=self.queue,training_resume=Mock(return_value=None),
            training_capacity=Mock(return_value={'actual_resource_test':True}),run=Mock(side_effect=run))
        def publish(manifest,path):
            save(self.root/(self.manifest['epoch']+'-checkpoint-publication.json'),
                dict(checkpoint=self.cp['id'],operator_independent_hashes=True,
                    objects={n:{'sha256':h}for n,h in self.cp['files'].items()}))
            return self.cp
        self.controller.publish_remote_checkpoint=Mock(side_effect=publish)
        self.controller.checkpoint_with_reads=lambda value:value
        output,first=train(self.controller,self.manifest,reports,'/input',steps=3)
        self.assertEqual(output['id'],self.cp['id']);self.assertFalse(first['weights_changed'])
        self.assertTrue(first['state_updated']);self.assertEqual(first['trainer_state']['optimizer_steps'],3)
        self.assertEqual(first['updates'],report['training']['updates']);self.assertIsInstance(first['updates'],list)
        self.assertEqual(len(first['updates']),3);self.assertNotIn('updates',first['persistent_diagnostics'])
        self.assertEqual(first['persistent_diagnostics'],report['training']['persistent_diagnostics'])
        _,second=train(self.controller,self.manifest,reports,'/input',steps=3)
        self.assertEqual(second['trainer_state'],first['trainer_state'])
        self.assertEqual(launches,[job['job_id']]);self.controller.jobs.run.assert_called_once()
        self.controller.publish_remote_checkpoint.assert_called_once()
        corrupted=copy.deepcopy(second);corrupted['trainer_state']['optimizer_steps']=2
        save(self.root/(self.manifest['epoch']+'-training-metrics.json'),corrupted)
        with self.assertRaises(ValueError):train(self.controller,self.manifest,reports,'/input',steps=3)
        self.controller.jobs.run.assert_called_once()

    def test_worker_backend_reports_exact_update_list_with_separate_diagnostics(self):
        from subnet.backend_jobs import execute
        original,job=self.report();definition=dict(env_id='math',spec={})
        runtime=SimpleNamespace(model=torch.nn.Module())
        runtime.model.register_parameter('weight',self.parameters[0][1])
        diagnostics=dict(original['training']['persistent_diagnostics'],updates=original['training']['updates'])
        model_path=self.root/'checkpoint-final'
        def download(url,digest,path,limit):path.write_bytes(b'synthetic frozen transport')
        with patch('subnet.backend_jobs._validate',return_value=(job,self.manifest)), \
                patch('subnet.backend_jobs.digest',return_value='b'*64), \
                patch('subnet.backend_jobs.version',return_value='approved'), \
                patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}), \
                patch('subnet.backend_jobs.install_source_loader'), \
                patch('subnet.task_assets.hydrate_manifest'), \
                patch('subnet.backend_jobs.checkpoint',return_value=self.root), \
                patch('subnet.protocol.entries',return_value=[definition]), \
                patch('subnet.backend_jobs.initial_configuration',return_value=(definition,{})), \
                patch('subnet.forced_sampling.bind_runtime'), \
                patch('subnet.backend_jobs.get_object',side_effect=download), \
                patch('subnet.training_receipts.admitted_submission',
                    return_value=(original['training_admissions'][0],[(definition,*self.rollouts)])), \
                patch('subnet.persistent_training_worker.train',
                    return_value=(model_path,diagnostics,original['persistent_training_state'])), \
                patch('subnet.model.model_files',return_value=self.cp['files']):
            actual=execute(self.sign(job),self.authority,self.root,runtime_factory=lambda *args:runtime)
        self.assertIsInstance(actual['training']['updates'],list)
        self.assertEqual(actual['training']['updates'],original['training']['updates'])
        self.assertEqual(len(actual['training']['updates']),job['steps'])
        self.assertNotIn('updates',actual['training']['persistent_diagnostics'])
        self.assertEqual(actual['training']['persistent_diagnostics'],original['training']['persistent_diagnostics'])
        self.assertFalse(actual['training']['weights_changed'])
        validate_report(actual,job,self.manifest)

    def test_v4_evidence_rejects_wrong_update_shape_counter_and_task_weights(self):
        report,job=self.report()
        from subnet.persistent_training_evidence import validate_updates
        self.assertEqual(validate_updates(report,job,self.manifest)['update_count'],3)
        diagnostics=dict(report['training']['persistent_diagnostics'],updates=report['training']['updates'])
        updates,summary=report_updates(diagnostics,job,self.manifest)
        self.assertEqual(updates,report['training']['updates']);self.assertNotIn('updates',summary)
        for kind in ('dict','count','counter','task-weight','precision-count','duplicate-summary'):
            changed=copy.deepcopy(report)
            if kind=='dict':changed['training']['updates']=diagnostics
            elif kind=='count':changed['training']['updates']=changed['training']['updates'][:-1]
            elif kind=='counter':changed['training']['updates'][0]['global_optimizer_step']=2
            elif kind=='task-weight':changed['training']['updates'][0]['pairs'][0]['gradient_weight']=.5
            elif kind=='precision-count':changed['training']['updates'][0]['precision']['master_changed_elements']=0
            else:changed['training']['persistent_diagnostics']['updates']=updates
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_report(changed,job,self.manifest)


if __name__=='__main__':unittest.main()
