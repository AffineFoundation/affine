"""CPU future-v2 integration. Network, live signing and GPU execution excluded."""
import copy
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from subnet import backend_jobs as backend, compact_training_inputs as compact
from subnet import training_receipts as v1
from subnet.distributed_roles import Coordinator
from subnet.remote_backend import RemoteController, RemoteJobs, save
from subnet.role_router import RoutedJobs
from subnet.storage import canonical
from training_receipt_fixtures import sign
from test_compact_training_inputs import make_setup


class MemoryBucket:
    def __init__(self):self.name='synthetic';self.client=self;self.objects={};self.events=[];self.corrupt=False
    def put(self,key,data,content_type):self.events.append('put');self.objects[key]=data
    def get_object(self,*,Bucket,Key):
        self.events.append('readback');data=self.objects[Key]
        return dict(ContentLength=len(data),Body=io.BytesIO(data+b'bad' if self.corrupt else data))
    def presign(self,key):return 'https://synthetic.r2.cloudflarestorage.com/'+key+'?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test'
    def json(self,key,value):self.objects[key]=canonical(value)


class CompactIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.s=make_setup(self.root)
        self.manifest=self.s['manifest'];self.authority=self.s['authority']
        self.bucket=MemoryBucket()
        self.queue=Coordinator(self.root/'queue.sqlite',self.authority,{self.s['row']['worker']:['verify']})
        row=dict(self.s['row'],expires=100)
        with sqlite3.connect(self.queue.path) as database:
            fields=list(row)
            database.execute('INSERT INTO jobs ('+','.join(fields)+') VALUES ('+','.join('?' for _ in fields)+')',tuple(row.values()))
        self.controller=RemoteController.__new__(RemoteController)
        self.controller.state=self.root;self.controller.authority=SimpleNamespace(id=self.authority)
        self.controller.signed=lambda payload:sign(self.s['key'],payload)
        self.controller.bucket=self.bucket;self.controller.jobs=SimpleNamespace(queue=self.queue)
        self.audit=dict(self.s['fx']['audit'],remote_job_id=row['id'])

    def prepare(self):
        return compact.prepare_submissions(self.controller,self.manifest,
            {self.s['fx']['miner']:self.audit},self.s['fx']['receipts'])

    def job(self,submissions=None):
        names=set(backend.SOURCE_FILES)|{'subnet/training_policy.py','subnet/covered_epoch_optimizer.py',
            'subnet/epoch_optimizer.py','subnet/training_receipts.py','subnet/compact_training_inputs.py'}
        return dict(schema=1,job_id='prospective-compact-train',role='train',created_at=22,expires_at=100,
            manifest=self.controller.signed(self.manifest),source_files={n:'a'*64 for n in names},
            runtime_versions=dict(torch='approved',transformers='approved',toploc='approved'),
            submissions=submissions or self.prepare(),steps=3,training_policy=backend.COVERED_POLICY,
            training_input_policy=compact.VERSION)

    def test_actual_coordinator_reads_complete_row_and_readbacks_before_v2_signature(self):
        signed=[]
        def signing(payload):
            if payload.get('version')==compact.VERSION:
                self.assertEqual(self.bucket.events[-1],'readback');signed.append(payload)
            return sign(self.s['key'],payload)
        self.controller.signed=signing
        first=self.prepare();second=self.prepare()
        self.assertEqual(compact.receipt_inventory(first),compact.receipt_inventory(second))
        self.assertEqual(first[0]['sha256'],self.s['obj']['sha256'])
        self.assertEqual(compact.original_submissions(first,self.manifest,self.authority)[0]['sha256'],self.s['fx']['frozen']['sha256'])
        self.assertEqual(len(signed),2)

    def test_bad_durable_readback_refuses_v2_signing(self):
        self.bucket.corrupt=True
        signed=Mock(side_effect=self.controller.signed);self.controller.signed=signed
        with self.assertRaisesRegex(ValueError,'readback'):self.prepare()
        self.assertEqual(sum(call.args[0].get('version')==compact.VERSION for call in signed.call_args_list),0)

    def test_historical_v1_report_cannot_be_relabelled_compact(self):
        row=dict(self.s['row']);job=json.loads(row['envelope']);job['payload']['manifest']['payload']['training_input_policy']=v1.VERSION
        # Authentic original job changed for this adversarial fixture; this first
        # policy check must reject even before accepting report lineage.
        job['payload']['manifest']=self.controller.signed(job['payload']['manifest']['payload'])
        job=self.controller.signed(job['payload']);row.update(envelope=json.dumps(job),digest=v1.sha(job['payload']))
        with self.assertRaisesRegex(ValueError,'prospective original'):
            compact.prepare_from_completed_row(row,self.authority,self.queue.workers,self.manifest,
                self.s['fx']['miner'],self.s['fx']['frozen'],self.audit,self.s['fx']['receipt'])

    def test_backend_coverage_uses_original_frozen_identity_and_v2_pins(self):
        job=self.job();backend.validate(self.controller.signed(job),self.authority,now=50)
        self.assertNotEqual(job['submissions'][0]['sha256'],self.s['fx']['frozen']['sha256'])
        for changed in ('missing-pin','old-input-policy','old-amendment','wrong-inventory'):
            bad=copy.deepcopy(job)
            if changed=='missing-pin':del bad['source_files']['subnet/compact_training_inputs.py']
            elif changed=='old-input-policy':bad['training_input_policy']=v1.VERSION
            elif changed=='old-amendment':
                bad['manifest']['payload']['training_execution_amendment']={'v1':'amendment'}
                bad['manifest']=self.controller.signed(bad['manifest']['payload'])
            else:bad['submissions'][0]['accepted_batch_sha256']=[]
            with self.subTest(changed=changed),self.assertRaises((ValueError,KeyError)):
                backend.validate(self.controller.signed(bad),self.authority,now=50)

    def test_every_role_requires_compact_policy_source_pins(self):
        job=self.job();job['role']='verify';del job['source_files']['subnet/compact_training_inputs.py']
        with self.assertRaisesRegex(ValueError,'every role'):
            backend.validate(self.controller.signed(job),self.authority,now=50)

    def test_backend_downloads_only_compact_json_and_calls_unchanged_covered_objective(self):
        job=self.job();runtime=SimpleNamespace(model=torch.nn.Linear(1,1),harness=self.manifest['environments'][0]['harness'])
        downloads=[];seen_pairs=[];files=dict(self.manifest['checkpoint']['files'],**{'model.safetensors':'3'*64})
        def download(url,digest,path,limit):
            self.assertEqual(path.suffix,'.json');self.assertEqual(limit,job['submissions'][0]['size'])
            self.assertEqual(digest,job['submissions'][0]['sha256']);downloads.append(path)
            path.write_bytes(next(iter(self.bucket.objects.values())))
        def optimize(runtime,pairs,out,**kwargs):
            seen_pairs.extend(pairs);self.assertEqual(kwargs,dict(seed=self.manifest['training_coverage']['seed'],steps=3))
            with torch.no_grad():runtime.model.weight.add_(1)
            return out/'final',[dict(synthetic_update=True)]*3
        forbidden=AssertionError('trainer re-verification or ZIP access')
        with patch('subnet.backend_jobs.time.time',return_value=50), \
             patch('subnet.backend_jobs.digest',return_value='a'*64), \
             patch('subnet.backend_jobs.version',return_value='approved'), \
             patch('subnet.backend_jobs.install_source_loader') as loader, \
             patch('subnet.backend_jobs.checkpoint',return_value=self.root), \
             patch('subnet.backend_jobs.get_object',side_effect=download), \
             patch('subnet.backend_jobs.audit',side_effect=forbidden), \
             patch('subnet.batches.submission_records',side_effect=forbidden), \
             patch('subnet.training_receipts.admitted_submission',side_effect=forbidden), \
             patch('subnet.covered_epoch_optimizer.train_epoch',side_effect=optimize), \
             patch('subnet.model.model_files',return_value=files), \
             patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
            report=backend.execute(self.controller.signed(job),self.authority,self.root,runtime_factory=lambda *args,**kw:runtime)
        self.assertEqual(len(downloads),1);self.assertEqual(seen_pairs[0][1:],tuple(self.s['fx']['batch']['rollouts']))
        self.assertIn('subnet/compact_training_inputs.py',loader.call_args.args[1])
        self.assertEqual(report['audits'],[]);self.assertFalse(report['training']['trainer_verification_performed'])
        self.assertEqual(report['training']['training_input_policy'],compact.VERSION)
        compact.validate_report(report,job,self.manifest,self.authority)

    def test_controller_selects_v2_and_measures_compact_download_bytes(self):
        epoch=self.manifest['epoch'];(self.root/'roles').mkdir()
        save(self.root/(epoch+'-scores.json'),{'receipts':self.s['fx']['receipts']})
        save(self.root/(epoch+'-audit-challenge.json'),self.s['fx']['challenge'])
        output=dict(files={'config.json':'1'*64,'model.safetensors':'3'*64},path='/future/final')
        output['id']=backend.file_map(output['files'])
        self.controller.jobs=SimpleNamespace(queue=self.queue,training_resume=Mock(return_value=None),
            training_capacity=Mock(return_value={'compact_capacity':True}),run=Mock(return_value=dict(
                new_checkpoint=output,job_id='future-job',training=dict(updates=[],training_policy=backend.COVERED_POLICY,
                training_coverage=self.manifest['training_coverage'],training_input_policy=compact.VERSION),
                covered_training_inputs=dict(unique_verified_pairs=1))))
        self.controller.publish_remote_checkpoint=Mock(return_value={k:v for k,v in output.items()if k!='path'})
        _,metrics=self.controller.train(self.manifest,{self.s['fx']['miner']:self.audit},'/future/input',steps=3)
        submissions=self.controller.jobs.run.call_args.kwargs['submissions']
        self.assertEqual(self.controller.jobs.training_capacity.call_args.kwargs['submission_bytes'],sum(o['size']for o in submissions))
        self.assertEqual(metrics['training_input_policy'],compact.VERSION)
        self.assertEqual(metrics['verifier_receipt_inventory'][0]['submission_sha256'],self.s['fx']['frozen']['sha256'])

    def test_remote_dispatch_mints_explicit_v2_job_through_original_authorized_path(self):
        submissions=self.prepare();jobs=RemoteJobs.__new__(RemoteJobs)
        jobs.state=self.root/'roles';jobs.state.mkdir();jobs.controller=self.controller
        jobs.config={};jobs.workspace='/synthetic/work';jobs.code='/synthetic/code';jobs.python='python'
        template=self.job(submissions);jobs.metadata={k:template[k]for k in ('source_files','runtime_versions')}
        jobs.command=Mock();jobs.copy_to=Mock(side_effect=RuntimeError('test ends before external dispatch'))
        with patch('subnet.remote_backend.time.time',return_value=50),self.assertRaisesRegex(RuntimeError,'test ends'):
            jobs.run('future-new','train',self.manifest,submissions=submissions,steps=3,training_policy=backend.COVERED_POLICY)
        record=json.loads((jobs.state/'future-new.json').read_text())
        job=v1.authenticate(json.loads((jobs.state/(record['job_id']+'-job.json')).read_text()),self.authority)
        self.assertEqual(job['training_input_policy'],compact.VERSION)
        self.assertEqual(job['manifest']['payload'],self.manifest)
        compact.validate_job(job,self.manifest,self.authority)

    def test_future_opening_carries_policy_in_first_signed_public_manifest(self):
        from subnet.controller import Controller
        def base_open(controller,*args,**kwargs):
            manifest=dict(self.manifest);manifest.pop('training_input_policy')
            controller.bucket.json('public/'+manifest['epoch']+'/manifest.json',controller.signed(manifest))
            return manifest
        with patch.object(Controller,'open',base_open):
            actual=self.controller.open(self.manifest['epoch'],training_policy=backend.COVERED_POLICY,
                training_input_policy=compact.VERSION)
        published=v1.authenticate(json.loads(self.bucket.objects['public/'+actual['epoch']+'/manifest.json']),self.authority)
        self.assertEqual(published['training_input_policy'],compact.VERSION)
        with self.assertRaises(ValueError):self.controller.open('future',training_policy=backend.FIXED_POLICY,
            training_input_policy=compact.VERSION)


    def test_router_capacity_uses_compact_bounds_keeps_checkpoint_and_safety_reserves(self):
        router=RoutedJobs.__new__(RoutedJobs);router.initial_role='mine';router.caches={'train':{self.manifest['checkpoint']['id']:'/cache'}}
        router.roles={'mine':SimpleNamespace(workspace='/miner'),'train':SimpleNamespace(capacity=Mock(return_value=dict(checkpoint_bytes=100,free_bytes=10**10)))}
        result=router.training_capacity(self.manifest,3,submission_bytes=1234)
        self.assertEqual(result['download_reserve_bytes'],compact.MAX_BYTES)
        self.assertEqual(result['raw_working_reserve_bytes'],compact.MAX_BYTES)
        self.assertEqual(result['required_bytes'],200+2*compact.MAX_BYTES+2*1024**3)
        self.assertEqual(result['retained_step_checkpoints'],0);self.assertEqual(result['temporary_export_copies'],1)

    def test_persistent_worker_retires_authenticated_compact_only(self):
        from subnet.persistent_training_worker import admitted_submission
        summary,pairs=admitted_submission(self.s['path'],self.s['obj'],self.manifest,self.authority)
        self.assertFalse(self.s['path'].exists());self.assertEqual(summary['submission_sha256'],self.s['fx']['frozen']['sha256'])
        self.assertEqual(pairs[0][1:],tuple(self.s['fx']['batch']['rollouts']))

    def test_remote_resume_binds_v2_receipts_and_original_job(self):
        jobs=RemoteJobs.__new__(RemoteJobs);jobs.state=self.root/'roles';jobs.state.mkdir();jobs.controller=self.controller
        job=self.job();label='future-train';jobid=job['job_id']
        save(jobs.state/(jobid+'-job.json'),self.controller.signed(job))
        save(jobs.state/(label+'.json'),dict(role='train',job_id=jobid,job_sha256=v1.sha(job),manifest_sha256=v1.sha(self.manifest)))
        jobs.remote_status=Mock(return_value={'phase':'running'})
        result=jobs.training_resume(label,self.manifest,job['submissions'],3,None)
        self.assertTrue(result['resuming_original_training'])
        bad=copy.deepcopy(job['submissions']);bad[0]['verifier_receipt']['payload']['compact_size']+=1
        with self.assertRaisesRegex(ValueError,'receipts changed'):
            jobs.training_resume(label,self.manifest,bad,3,None)


class CompactPersistentIntegrationTests(unittest.TestCase):
    """Exercise unchanged persistent CPU state lineage with v2 transport."""
    from test_persistent_training_integration import PersistentIntegrationTests as Original
    sign=Original.sign

    def setUp(self):
        self.Original.setUp(self)
        from training_receipt_fixtures import signed_receipt
        self.manifest['training_input_policy']=compact.VERSION
        receipt,audit,job,request=signed_receipt(self.key,self.manifest,self.miner,self.receipts[self.miner],self.batch)
        self.legacy_submission=dict(self.submission,verifier_receipt=receipt)
        self.verifier_audit=dict(audit,remote_job_id='synthetic-verify')
        self.queue.workers={request['signer']:['verify']}
        with sqlite3.connect(self.queue.path) as database:
            database.execute('UPDATE jobs SET digest=?,envelope=?,worker=?,report=?,report_digest=?,report_request=? WHERE id=?',
                (v1.sha(job['payload']),canonical(job).decode(),request['signer'],canonical(request['payload']['report']).decode(),
                 v1.sha(request['payload']['report']),canonical(request).decode(),'synthetic-verify'))
        def put(key,data,content_type):self.bucket.objects[key]=data
        def get_object(*,Bucket,Key):
            data=self.bucket.get(Key);return dict(ContentLength=len(data),Body=io.BytesIO(data))
        self.bucket.put=put;self.bucket.get_object=get_object
        self.controller.jobs=SimpleNamespace(queue=self.queue)
        self.submission=compact.prepare_submissions(self.controller,self.manifest,
            {self.miner:self.verifier_audit},self.receipts)[0]
        self.data=self.bucket.get('private/compact-training-inputs/'+self.submission['sha256']+'.json')

    def job(self,*args,**kwargs):
        job=self.Original.job(self,*args,**kwargs)
        job['training_input_policy']=compact.VERSION
        job['source_files']['subnet/compact_training_inputs.py']='b'*64
        return job

    def report(self,job=None):
        job=job or self.job();compact_submission=self.submission
        self.submission=self.legacy_submission
        try:report,job=self.Original.report(self,job)
        finally:self.submission=compact_submission
        path=self.root/'prospective-input.json';path.write_bytes(self.data)
        report['training_admissions']=[compact.admitted_submission(path,self.submission,self.manifest,self.authority)[0]]
        report['training']['training_input_policy']=compact.VERSION
        return report,job

    def test_persistent_controller_recovers_exact_original_state_without_retraining(self):
        self.Original.test_controller_same_bf16_checkpoint_recovers_original_state_without_retraining(self)

    def test_persistent_report_and_state_lineage_accept_compact_exact_pairs(self):
        from subnet.persistent_training_protocol import validate_report
        report,job=self.report()
        backend.validate(self.sign(job),self.authority,now=30)
        validate_report(report,job,self.manifest)
        bad=copy.deepcopy(report);bad['training_admissions'][0]['submission_sha256']=self.submission['sha256']
        with self.assertRaises(ValueError):validate_report(bad,job,self.manifest)

    def test_persistent_compact_capacity_keeps_state_model_and_memory_reserves(self):
        from subnet.persistent_training_worker import capacity_requirement
        result=capacity_requirement(self.manifest,dict(free_bytes=10**12,available_ram_bytes=10**12),checkpoint_bytes=100,missing_input=True)
        self.assertEqual(result['download_reserve_bytes'],256*compact.MAX_BYTES)
        self.assertEqual(result['raw_working_reserve_bytes'],compact.MAX_BYTES)
        self.assertGreater(result['required_bytes'],result['download_reserve_bytes']+compact.MAX_BYTES)
        self.assertTrue(result['state_streaming']);self.assertFalse(result['full_state_disk_hydration'])
