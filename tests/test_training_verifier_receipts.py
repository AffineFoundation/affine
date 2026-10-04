"""Local receipt integrity controls: synthetic keys, arrays and SQLite only."""
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from nacl.signing import SigningKey
from subnet import training_receipts as r
from subnet.storage import canonical
from subnet.distributed_roles import Coordinator
from subnet.remote_backend import RemoteController
from training_receipt_fixtures import sign,transport_fixture


class VerifierReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        self.f=transport_fixture(self.key);self.manifest=self.f['manifest'];self.obj=self.f['submission']
        self.path=self.root/'frozen.zip';self.path.write_bytes(self.f['data'])

    def receipt(self,payload):return sign(self.key,payload)

    def job(self):
        return dict(role='train',job_id='synthetic-train',created_at=50,steps=3,
            training_policy=self.manifest['training_policy'],training_input_policy=r.VERSION,
            manifest=sign(self.key,self.manifest),source_files={'subnet/training_receipts.py':'b'*64},
            submissions=[self.obj])

    def test_honest_receipt_admission_never_runs_inference_sampler_toploc_or_environment(self):
        prohibited=AssertionError('trainer must never reverify')
        with patch('subnet.model.Runtime.verify',side_effect=prohibited), \
                patch('subnet.model.Runtime.sample_output',side_effect=prohibited), \
                patch('subnet.model.Runtime.compute',side_effect=prohibited), \
                patch('subnet.model.create_session',side_effect=prohibited), \
                patch('subnet.proofs.verify_mapped_proofs',side_effect=prohibited), \
                patch('subnet.backend_jobs.audit',side_effect=prohibited):
            summary,pairs=r.admitted_submission(self.path,self.obj,self.manifest,self.authority)
        self.assertEqual(len(pairs),1);self.assertEqual(summary['accepted'],[self.f['batch']])
        self.assertIs(summary['trainer_verification_performed'],False)
        self.assertEqual(summary['verification_performed_by'],'registered-verifier')
        self.assertTrue(self.path.exists())

    def test_signature_forgery_or_wrong_authority_refused_before_zip_read(self):
        obj=copy.deepcopy(self.obj);obj['verifier_receipt']['payload']['trainer_verification_required']=True
        with self.assertRaises(ValueError):r.admitted_submission(self.path,obj,self.manifest,self.authority)
        with self.assertRaises(ValueError):r.admitted_submission(self.path,self.obj,self.manifest,SigningKey.generate().verify_key.encode().hex())
        self.assertTrue(self.path.exists())

    def test_cross_epoch_checkpoint_source_sampler_harness_and_zip_substitution_refused(self):
        for field in ('epoch','checkpoint','source_bundle','sampling_contract','harness_source_hash','environments'):
            manifest=copy.deepcopy(self.manifest)
            if field=='epoch':manifest[field]='nonpayable-other'
            elif field=='checkpoint':manifest[field]['id']='c'*64
            elif field=='source_bundle':manifest[field]['sha256']='c'*64
            elif field=='sampling_contract':manifest[field]['randomness']='c'*64
            elif field=='harness_source_hash':manifest[field]='c'*64
            else:manifest[field][0]['harness']['temperature']=1.
            with self.subTest(field=field),self.assertRaises(ValueError):r.admitted_submission(self.path,self.obj,manifest,self.authority)
        self.path.write_bytes(self.f['data'][:-1]+b'x')
        with self.assertRaises(ValueError):r.admitted_submission(self.path,self.obj,self.manifest,self.authority)

    def test_unchecked_or_changed_pair_never_gets_operator_receipt(self):
        for kind in ('unchecked','invalid','missing-outcome','changed-batch','changed-sampling','unregistered-worker','report-source','report-job'):
            job=copy.deepcopy(self.f['verify_job']);request=copy.deepcopy(self.f['worker_request'])
            # Resign mutations as synthetic registered verifier; operator checks semantic bindings too.
            worker=SigningKey.generate();worker_id=worker.verify_key.encode().hex()
            payload=request['payload'];audit=payload['report']['audits'][0]
            if kind=='unchecked':audit['outcomes'][0]['fully_audited']=False
            elif kind=='invalid':audit['outcomes'][0]['valid']=False
            elif kind=='missing-outcome':audit['outcomes']=[]
            elif kind=='changed-batch':audit['accepted'][0]['checkpoint']='c'*64
            elif kind=='changed-sampling':audit['accepted'][0]['rollouts'][0]['sampling']['binding_sha256']='c'*64
            elif kind=='report-source':payload['report']['source_files']={}
            elif kind=='report-job':payload['report']['job_sha256']='c'*64
            request=sign(worker,payload)
            workers={}if kind=='unregistered-worker'else{worker_id:['verify']}
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                r.receipt_payload(job,request,self.authority,workers,self.manifest,self.f['miner'],self.f['frozen'],audit)

    def test_completed_authoritative_queue_lineage_is_required_and_stable_on_recovery(self):
        queue=Coordinator(self.root/'queue.sqlite',self.authority,{self.f['worker_request']['signer']:['verify']})
        job=self.f['verify_job'];request=self.f['worker_request'];report=request['payload']['report']
        with sqlite3.connect(queue.path)as db:
            db.execute('INSERT INTO jobs(id,digest,envelope,role,expires,status,worker,report,report_digest,report_request) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (job['payload']['job_id'],r.sha(job['payload']),canonical(job).decode(),'verify',100,'complete',request['signer'],
                 canonical(report).decode(),r.sha(report),canonical(request).decode()))
        controller=SimpleNamespace(jobs=SimpleNamespace(queue=queue),authority=SimpleNamespace(id=self.authority),signed=lambda p:sign(self.key,p))
        audit=dict(self.f['audit'],remote_job_id=job['payload']['job_id'])
        first=r.issue(controller,self.manifest,self.f['miner'],self.f['frozen'],audit)
        self.assertEqual(r.issue(controller,self.manifest,self.f['miner'],self.f['frozen'],audit),first)
        for column,value in [('status','leased'),('report_digest','c'*64),('worker','c'*64)]:
            with sqlite3.connect(queue.path)as db:old=db.execute('SELECT '+column+' FROM jobs').fetchone()[0];db.execute('UPDATE jobs SET '+column+'=?',(value,))
            with self.subTest(column=column),self.assertRaises(ValueError):r.issue(controller,self.manifest,self.f['miner'],self.f['frozen'],audit)
            with sqlite3.connect(queue.path)as db:db.execute('UPDATE jobs SET '+column+'=?',(old,))

    def test_pair_commitment_slots_size_and_acceptance_inventory_are_exact(self):
        for kind in ('pair','slot','duplicate','accepted-hash','size','extra-field','class-quota'):
            obj=copy.deepcopy(self.obj);value=obj['verifier_receipt']['payload']
            if kind=='pair':value['fully_audited_batches'][0]['positive_rollout_sha256']=['c'*64]
            elif kind=='slot':value['fully_audited_batches'][0]['batch_number']=2
            elif kind=='duplicate':value['fully_audited_batches']*=2
            elif kind=='accepted-hash':obj['accepted_batch_sha256']=['c'*64]
            elif kind=='size':obj['size']+=1
            elif kind=='extra-field':value['unchecked_admission']=True
            else:value['fully_audited_batches'][0]['negative_rollout_sha256']=[]
            obj['verifier_receipt']=self.receipt(value)
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.admitted_submission(self.path,obj,self.manifest,self.authority,retire=True)
            self.assertTrue(self.path.exists())
        _,pairs=r.admitted_submission(self.path,self.obj,self.manifest,self.authority,retire=True)
        self.assertEqual(len(pairs),1);self.assertFalse(self.path.exists())

    def test_pending_e9_amendment_gates_before_capacity_signing_or_any_request(self):
        c=RemoteController.__new__(RemoteController);c.training_execution_amendment_required_epochs=[self.manifest['epoch']]
        c.training_execution_amendment_files={};c.jobs=Mock();c.signed=Mock();c.state=self.root
        with self.assertRaisesRegex(ValueError,'amendment is pending'):c.train(self.manifest,{},'/unused',steps=3)
        self.assertEqual(c.jobs.mock_calls,[]);c.signed.assert_not_called()

    def amendment(self):
        original=copy.deepcopy(self.manifest);original.pop('training_input_policy')
        new_source={'sha256':'8'*64}
        payload=dict(version=r.AMENDMENT_VERSION,epoch=self.manifest['epoch'],original_signed_manifest=sign(self.key,original),
            original_signed_manifest_sha256=r.sha(sign(self.key,original)),training_source_bundle=new_source,
            training_policy=original['training_policy'],training_input_policy=r.VERSION,steps=3,
            verifier_receipt_inventory=r.receipt_inventory([self.obj]),created_at=45,expires_at=90)
        derived=dict(self.manifest,source_bundle=new_source,training_execution_amendment=sign(self.key,payload))
        return derived,payload

    def test_private_amendment_changes_only_training_execution_and_preserves_original_challenge(self):
        original=canonical(self.manifest);derived,payload=self.amendment()
        r.validate_amendment(derived,[self.obj],3,self.authority,50)
        r.validate_receipt(self.obj['verifier_receipt'],self.obj,derived,self.authority)
        self.assertEqual(canonical(self.manifest),original)
        for kind in ('model','sampler','task','policy','steps','receipts','created'):
            manifest=copy.deepcopy(derived);value=copy.deepcopy(payload)
            if kind=='model':manifest['checkpoint']['id']='c'*64
            elif kind=='sampler':manifest['sampling_contract']['randomness']='c'*64
            elif kind=='task':manifest['environments'][0]['indices']=[1]
            elif kind=='policy':value['training_policy']='different'
            elif kind=='steps':value['steps']=4
            elif kind=='receipts':value['verifier_receipt_inventory']=[]
            else:value['created_at']=51
            manifest['training_execution_amendment']=sign(self.key,value)
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.validate_amendment(manifest,[self.obj],3,self.authority,50)

    def test_reports_are_truthful_admissions_without_fabricated_trainer_audits(self):
        summary,_=r.admitted_submission(self.path,self.obj,self.manifest,self.authority)
        report=dict(training_admissions=[summary],audits=[],training=dict(training_input_policy=r.VERSION,
            trainer_verification_performed=False,all_pairs_authenticated_verifier_receipts=True))
        r.validate_report(report,self.job(),self.manifest,self.authority)
        for kind in ('audits','claim','receipt','accepted','input-policy'):
            bad=copy.deepcopy(report)
            if kind=='audits':bad['audits']=[self.f['audit']]
            elif kind=='claim':bad['training']['all_pairs_independently_reaudited']=True
            elif kind=='receipt':bad['training_admissions'][0]['verifier_receipt_sha256']='c'*64
            elif kind=='accepted':bad['training_admissions'][0]['accepted']=[]
            else:bad['training']['training_input_policy']='unchecked'
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.validate_report(bad,self.job(),self.manifest,self.authority)
        unchecked=self.job();unchecked.pop('training_input_policy')
        with self.assertRaises(ValueError):r.validate_job(unchecked,self.manifest,self.authority)


if __name__=='__main__':unittest.main()
