import base64,copy,json,tempfile,unittest
from pathlib import Path
from nacl.signing import SigningKey
from dashboard.mimo_live_projection import Projection,canonical,digest,evidence_availability

class Policy:
    run_id='new-mimo';policy_sha256='a'*64
    scope={'initial_checkpoint':{'id':'b'*64},'immutable_manifest':{'source_bundle':{'sha256':'c'*64}}}
    def validate_manifest(self,m):
        if m['checkpoint']['id']!='b'*64 or m['source_bundle']['sha256']!='c'*64:raise ValueError('pin')

class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.addCleanup(self.temp.cleanup)
        self.authority=SigningKey.generate();self.worker=SigningKey.generate();self.authority_id=self.authority.verify_key.encode().hex();self.worker_id=self.worker.verify_key.encode().hex()
        self.epoch=self.root/'epochs/0';(self.epoch/'coordinator').mkdir(parents=True);(self.root/'evaluator').mkdir()
        self.manifest=dict(epoch='new-mimo-e000000',run_id='new-mimo',checkpoint={'id':'b'*64},source_bundle={'sha256':'c'*64},payable=True,start=10,deadline=20,environments=[dict(indices=[1,2,3])],group_contract={'min_success':1,'min_completed_failure':1},max_batches=9,capabilities={'d'*64:dict(secret='NEVER-PUBLISH')})
        self.opening=self.sign(self.manifest);self.write('coordinator/OPENING.json',self.opening)
        self.calls=[]
        def validate(report,job,m,authority,jobhash,now):
            self.calls.append(job['job_id'])
            if report.get('job_sha256')!=jobhash:raise ValueError('job binding')
        self.p=Projection(self.root,Policy(),self.authority_id,{self.worker_id:['evaluate','train']},validate,first_round=189,started_at=10,model='MiMo9B')
        self.prior=dict(epochs=[dict(id='old-qwen')],evaluations=[dict(mean_reward=.9)],incentive={},incentive_history=[],incentive_prices={})
    def sign(self,value,key=None):
        key=key or self.authority
        return dict(payload=value,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(value)).signature).decode())
    def write(self,name,value):
        p=self.epoch/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canonical(value))
    def seal(self):
        capture=self.sign(dict(epoch=self.manifest['epoch'],manifest_sha256=digest(self.manifest),miner='d'*64,total_groups=2,private_url='SECRET-URL',rollouts='SECRET-TOKENS'))
        pop=dict(epoch=self.manifest['epoch'],manifest_sha256=digest(self.manifest),sealed_at=20,captures=[capture]);self.write('coordinator/POPULATION.json',self.sign(pop))
        dispatch=dict(epoch=self.manifest['epoch'],checkpoint='b'*64,manifest_sha256=digest(self.manifest),population_sha256=digest(pop),selected_task_count=1);self.write('coordinator/DISPATCH.json',self.sign(dispatch));return pop,dispatch
    def evaluation(self):
        job=dict(job_id='eval-1',role='evaluate',manifest=self.opening)
        result=dict(job_sha256=digest(job),completed_at=22,evaluation=dict(suite={'env_id':'math','seed':7},harness={'max_output_tokens':8192,'temperature':1,'top_p':.95,'top_k':20},dataset_id='e'*64,count=32,successes=18,mean_reward=18/32,status='complete',requested_count=32,rows=[{'output':'SECRET'}]))
        receipt=self.sign(dict(policy_sha256='a'*64,job=self.sign(job),worker_report=self.sign(dict(action='report',job_id='eval-1',report=result),self.worker)))
        path=self.root/'evaluator/eval-1.receipt.json';path.write_bytes(canonical(receipt));return path,receipt
    def test_new_run_pending_never_stitches_old_curve(self):
        d=self.p.project(self.prior,15);self.assertEqual(d['evaluations'],[]);self.assertEqual(d['training_run']['first_round'],189);self.assertEqual(d['epochs'][0]['display_epoch'],1);self.assertIsNone(d['epochs'][0]['training'])
        self.assertNotIn('SECRET',json.dumps(d));self.assertNotIn('NEVER-PUBLISH',json.dumps(d))
    def test_open_epoch_does_not_read_population_or_capture_artifacts(self):
        self.write('coordinator/POPULATION.json',{'invalid':True});self.write('coordinator/captured/garbage.json',{'tokens':'SECRET'});self.write('coordinator/DISPATCH.json',{'invalid':True})
        d=self.p.project(self.prior,15);self.assertFalse(d['epochs'][0]['batches_available']);self.assertEqual(d['epochs'][0]['batches'],0)
    def test_frozen_counts_only_without_artifacts_or_fake_audits(self):
        self.seal();d=self.p.project(self.prior,21);e=d['epochs'][0];self.assertEqual(e['batches'],2);self.assertEqual(e['unchecked'],2);self.assertEqual(e['accepted'],0);self.assertFalse(e['audit_breakdown_available']);self.assertNotIn('SECRET',json.dumps(d))
    def test_changed_opening_signature_rejected(self):
        altered=copy.deepcopy(self.opening);altered['payload']['checkpoint']['id']='f'*64;self.write('coordinator/OPENING.json',altered)
        with self.assertRaises(Exception):self.p.project(self.prior,15)
    def test_signed_unapproved_checkpoint_rejected(self):
        m=dict(self.manifest,checkpoint={'id':'f'*64});self.write('coordinator/OPENING.json',self.sign(m))
        with self.assertRaises(ValueError):self.p.project(self.prior,15)
    def test_completed_eval_requires_original_job_and_worker_signature(self):
        self.evaluation();d=self.p.project(self.prior,25);self.assertEqual(d['evaluations'][0]['mean_reward'],18/32);self.assertEqual(self.calls,['eval-1']);self.assertNotIn('SECRET',json.dumps(d))
    def test_wrong_worker_role_rejected(self):
        self.evaluation();self.p.workers={self.worker_id:['verify']}
        with self.assertRaises(ValueError):self.p.project(self.prior,25)
    def test_substituted_job_rejected_even_when_receipt_signed(self):
        path,receipt=self.evaluation();job=receipt['payload']['job']['payload'];job['job_id']='other';receipt['payload']['job']=self.sign(job);path.write_bytes(canonical(self.sign(receipt['payload'])))
        with self.assertRaises(ValueError):self.p.project(self.prior,25)
    def test_population_count_change_rejected(self):
        self.seal();p=self.epoch/'coordinator/POPULATION.json';d=json.loads(p.read_bytes());d['payload']['captures'][0]['payload']['total_groups']=9;p.write_bytes(canonical(d))
        with self.assertRaises(Exception):self.p.project(self.prior,25)
    def trained(self):
        _,dispatch=self.seal();publisher=SigningKey.generate();cp={'id':'f'*64}
        job=dict(job_id='train-1',role='train',manifest=self.opening,trainer_worker=self.worker_id,training={'publisher_id':publisher.verify_key.encode().hex()})
        receipt=self.sign({'descriptor':cp},publisher)
        report=dict(job_sha256=digest(job),publication_receipt=receipt,training=dict(output_checkpoint='f'*64,optimizer_step=1,optimizer_step_before=0,group_count=1,objective_loss=.25))
        self.write('TRAINING-JOB.json',self.sign(job));self.write('TRAINING-RESULT.json',self.sign(dict(action='report',job_id='train-1',report=report),self.worker))
        final=dict(policy_sha256='a'*64,epoch=self.manifest['epoch'],opening_sha256=digest(self.opening),dispatch_sha256=digest(self.sign(dispatch)),input_checkpoint='b'*64,round=189,result=dict(status='updated',checkpoint=cp,optimizer_step=1,publication_receipt_sha256=digest(receipt)))
        self.write('FINALIZED.json',self.sign(final));return final
    def test_real_update_requires_publication_and_finalization_binding(self):
        self.trained();row=self.p.project(self.prior,25)['epochs'][0]
        self.assertTrue(row['finalized']);self.assertEqual(row['training']['losses'],[.25]);self.assertEqual(row['training']['input_pairs'],1)
    def test_root_final_wrong_publication_rejected(self):
        final=self.trained();final['result']['checkpoint']={'id':'0'*64};self.write('FINALIZED.json',self.sign(final))
        with self.assertRaises(ValueError):self.p.project(self.prior,25)
    def test_abandoned_unsigned_epoch_does_not_increment_display_or_global_round(self):
        self.trained();self.epoch.rename(self.root/'epochs/1');(self.root/'epochs/0').mkdir()
        row=self.p.project(self.prior,25)['epochs'][0];self.assertEqual(row['display_epoch'],1)
    def test_missing_training_metrics_stay_null(self):
        self.seal();d=self.p.project(self.prior,25);self.assertIsNone(d['epochs'][0]['training'])
    def evidence_history(self,**changes):
        entry=dict(version='mimo-finalized-evidence-v1',run_id='new-mimo',first_round=189,
            epoch_count=1,last_finalized_round=189,created_at=24,catalog={'method':'GET','expires_at':40})
        entry.update(changes)
        return self.sign({'mimo_finalized_epoch_evidence':entry})
    def test_published_status_requires_signed_matching_closed_epoch_catalog(self):
        self.trained();snap=self.p.project(self.prior,25)
        result=evidence_availability(snap,self.evidence_history(),self.authority_id,25)
        self.assertEqual(result['summary']['evidence_status'],'mimo-finalized-artifacts-available')
        self.assertEqual(result['summary']['published_evidence_epochs'],1)
        self.assertEqual(snap['summary']['evidence_status'],'mimo-finalized-artifacts-not-yet-published')
    def test_published_open_or_wrong_run_catalog_rejected(self):
        snap=self.p.project(self.prior,15)
        result=evidence_availability(snap,self.evidence_history(),self.authority_id,25)
        self.assertEqual(result['summary']['evidence_status'],'mimo-finalized-evidence-unavailable')
        self.trained();snap=self.p.project(self.prior,25)
        result=evidence_availability(snap,self.evidence_history(run_id='old-run'),self.authority_id,25)
        self.assertEqual(result['summary']['evidence_status'],'mimo-finalized-evidence-unavailable')
    def test_expired_and_bad_signature_links_not_advertised_available(self):
        self.trained();snap=self.p.project(self.prior,25)
        result=evidence_availability(snap,self.evidence_history(),self.authority_id,41)
        self.assertEqual(result['summary']['evidence_status'],'mimo-finalized-artifact-links-expired')
        forged=self.evidence_history();forged['payload']['mimo_finalized_epoch_evidence']['epoch_count']=0
        result=evidence_availability(snap,forged,self.authority_id,25)
        self.assertEqual(result['summary']['evidence_status'],'mimo-finalized-evidence-unavailable')

if __name__=='__main__':unittest.main()

class WrapperTests(unittest.TestCase):
    def test_legacy_refresh_bypassed_and_dynamic_snapshot_retained(self):
        from ops.publish_mimo_dashboard import database_view
        class Original:
            def refresh(self):raise AssertionError('closed legacy rescan must not run')
            def snapshot(self,current_only=False):return {'incentive':self.latest}
        class View:
            def project(self,prior,now):return dict(prior,projected=True)
        db=database_view(Original,View())();db.latest=1;db.refresh()
        self.assertEqual(db.snapshot()['incentive'],1)
        db.latest=2;self.assertEqual(db.snapshot(True),{'incentive':2,'projected':True})
    def test_legacy_projection_public_exports_preserved(self):
        from dashboard.mimo_projection import project,from_selector
        self.assertTrue(callable(project));self.assertTrue(callable(from_selector))
    def test_existing_frontend_selects_new_rows(self):
        f=ProjectionTests();f.setUp()
        try:
            row=f.p.project(f.prior,15)['epochs'][0]
            self.assertEqual(row['source'],'live-reward-math')
            self.assertEqual(row['mechanism'],'mimo-fixed-group')
        finally:f.doCleanups()
