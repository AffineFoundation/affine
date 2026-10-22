import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from dashboard.test_mimo_projection import ProjectionTests
from ops.test_publish_finalized_epoch_evidence import PublisherTests
from ops import mimo_finalized_evidence as m
from ops import publish_finalized_epoch_evidence as p

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.f=ProjectionTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.legacy=PublisherTests();self.legacy.setUp();self.addCleanup(self.legacy.tearDown)
        self.bucket=self.legacy.bucket
        self.f.manifest['group_contract']['group_size']=8
        self.f.opening=self.f.sign(self.f.manifest);self.f.write('coordinator/OPENING.json',self.f.opening)
    def trained(self):
        f=self.f;final=f.trained();path=f.epoch/'coordinator/DISPATCH.json';d=json.loads(path.read_bytes())['payload']
        doc=dict(version='complete-group-training-document-v1',epoch=f.manifest['epoch'],checkpoint='b'*64,miner='d'*64,slot=0,batch=dict(env_id='math',index=1,task_hash='a'*64,rollouts=[dict(seed=0,classification='positive',turns=[dict(prompt=[1,2],output=[3,4],stop_reason='eos',logprobs=[-1],full_vocab_logprobs=['NO'])])]))
        doc['batch']['rollouts']=[dict(doc['batch']['rollouts'][0],seed=i)for i in range(8)]
        raw=p.canonical(doc);self.bucket.objects['private/doc']=raw
        d.update(selected=[dict(miner='d'*64,env_id='math',index=1,document={'key':'private/doc','sha256':p.digest(raw),'slot':0},batch_sha256=m.digest(doc['batch']))],excluded=[],selected_rollout_count=8)
        f.write('coordinator/DISPATCH.json',f.sign(d));final['dispatch_sha256']=m.digest(f.sign(d))
        job=json.loads((f.epoch/'TRAINING-JOB.json').read_bytes())['payload'];job['training']['dispatch']={'sha256':final['dispatch_sha256']};job['training']['documents']={p.digest(raw):{'size':len(raw),'url':'https://private.invalid/do-not-publish'}};f.write('TRAINING-JOB.json',f.sign(job))
        report=json.loads((f.epoch/'TRAINING-RESULT.json').read_bytes())['payload'];report['report']['job_sha256']=m.digest(job);f.write('TRAINING-RESULT.json',f.sign(report,f.worker));f.write('FINALIZED.json',f.sign(final));return final
    def test_open_never_reads_inputs_or_corrupt_opening(self):
        self.f.write('coordinator/OPENING.json',{'bad':True});self.assertEqual(m.collect(self.f.p,self.bucket,25),[]);self.assertEqual(self.bucket.reads,[])
    def test_finalization_arriving_after_snapshot_waits_until_next_cycle(self):
        inventory=m.snapshot(self.f.p)
        self.trained()
        self.assertEqual(m.collect(self.f.p,self.bucket,25,inventory),[])
        self.assertEqual(self.bucket.reads,[])
        self.assertEqual(len(m.collect(self.f.p,self.bucket,25)),1)

    def test_evaluation_arriving_after_snapshot_waits_until_next_cycle(self):
        self.trained();inventory=m.snapshot(self.f.p);self.f.evaluation()
        rows=m.collect(self.f.p,self.bucket,25,inventory)
        self.assertEqual(rows[0]['categories']['evaluations']['status'],'pending')
        self.assertEqual(len(m.collect(self.f.p,self.bucket,25)[0]['categories']['evaluations']['reports']),1)

    def test_frozen_finalization_and_eval_hashes_cannot_change(self):
        self.trained();path,evaluation=self.f.evaluation();inventory=m.snapshot(self.f.p)
        evaluation['payload']['unexpected']='changed'
        path.write_bytes(p.canonical(self.f.sign(evaluation['payload'])))
        with self.assertRaisesRegex(ValueError,'frozen evaluation changed'):
            m.collect(self.f.p,self.bucket,25,inventory)
        inventory=m.snapshot(self.f.p)
        final=json.loads((self.f.epoch/'FINALIZED.json').read_bytes())['payload']
        final['round']=999;self.f.write('FINALIZED.json',self.f.sign(final))
        with self.assertRaisesRegex(ValueError,'frozen finalization changed'):
            m.collect(self.f.p,self.bucket,25,inventory)

    def test_arrival_during_preput_revalidation_does_not_change_plan(self):
        self.trained();plan=self.legacy.prepare()
        with patch.object(m,'collector',return_value=self.f.p):
            m.extend_plan(plan,{},self.bucket)
            self.f.evaluation()
            m.validate_extension(plan,self.bucket)
        self.assertEqual(plan['mimo_extension']['inventory']['evaluations'],{})

    def test_prepare_freezes_inventory_before_legacy_network_reads(self):
        original=p.read_object
        started=[]
        def arrive(bucket,key,**kwargs):
            if key==p.HISTORY_KEY and not started:
                started.append(True);self.trained()
            return original(bucket,key,**kwargs)
        with patch.object(m,'collector',return_value=self.f.p),patch.object(p,'read_object',side_effect=arrive):
            plan=self.legacy.prepare(mimo={})
        self.assertTrue(started)
        self.assertEqual(plan['history_payload'][m.FIELD]['epoch_count'],0)
        self.assertEqual(plan['mimo_extension']['inventory']['epochs'],{})

    def test_default_cutoff_is_captured_after_inventory_scan(self):
        events=[];original=m.snapshot
        def inventory(view):
            events.append('snapshot');self.trained();return original(view)
        def clock():
            events.append('clock');return self.legacy.now
        with patch.object(m,'collector',return_value=self.f.p),patch.object(m,'snapshot',side_effect=inventory),patch.object(p.time,'time',side_effect=clock):
            plan=self.legacy.prepare(mimo={},now=None)
        self.assertEqual(events[:2],['snapshot','clock'])
        self.assertEqual(plan['history_payload'][m.FIELD]['epoch_count'],1)

    def test_original_report_validator_still_receives_original_cutoff(self):
        self.trained();original=self.f.p.validate_report
        calls=[]
        def reject_future(report,job,manifest,authority,jobhash,now):
            calls.append(now)
            original(report,job,manifest,authority,jobhash,now)
            raise ValueError('report signed deadline')
        self.f.p.validate_report=reject_future
        with self.assertRaisesRegex(ValueError,'report signed deadline'):
            m.collect(self.f.p,self.bucket,25)
        self.assertEqual(calls,[25])

    def test_full_tokens_without_vocab_or_credentials(self):
        self.trained();rows=m.collect(self.f.p,self.bucket,25);text=json.dumps(rows)
        self.assertNotIn('logprobs',text);self.assertNotIn('private/doc',text);self.assertNotIn('NEVER-PUBLISH',text)
        self.assertEqual(rows[0]['categories']['training_inputs']['groups'][0]['document']['batch']['rollouts'][0]['turns'][0]['output'],[3,4])
    def test_finalized_before_deadline_rejected(self):
        self.trained()
        with self.assertRaises(ValueError):m.collect(self.f.p,self.bucket,15)
    def test_forged_finalization_rejected(self):
        final=self.trained();final['round']=999;self.f.write('FINALIZED.json',self.f.sign(final))
        with self.assertRaises(ValueError):m.collect(self.f.p,self.bucket,25)
    def test_document_mutation_rejected(self):
        self.trained();self.bucket.objects['private/doc']+=b' '
        with self.assertRaises(ValueError):m.collect(self.f.p,self.bucket,25)
    def test_wrong_worker_rejected(self):
        self.trained();self.f.p.workers={}
        with self.assertRaises(ValueError):m.collect(self.f.p,self.bucket,25)
    def test_eval_only_after_original_epoch_finalizes(self):
        self.f.evaluation();self.assertEqual(m.collect(self.f.p,self.bucket,25),[])
        self.trained();self.assertEqual(len(m.collect(self.f.p,self.bucket,25)[0]['categories']['evaluations']['reports']),1)
    def test_additive_get_only_and_tamper_guard(self):
        self.trained();plan=self.legacy.prepare();old=copy.deepcopy(plan['history_payload'])
        with patch.object(m,'collector',return_value=self.f.p):
            m.extend_plan(plan,{},self.bucket);stripped=m.validate_extension(plan,self.bucket)
            self.assertEqual(stripped['history_payload'],old)
            self.assertTrue(all(op=='get_object' for _,op,_ in self.bucket.routes))
            plan['history_payload'][m.FIELD]['epoch_count']=999
            with self.assertRaises(ValueError):m.validate_extension(plan,self.bucket)
    def test_staged_category_modification_rejected(self):
        self.trained();plan=self.legacy.prepare()
        with patch.object(m,'collector',return_value=self.f.p):
            m.extend_plan(plan,{},self.bucket)
            item=next(x for x in plan['artifacts']if '/mimo/'in x['key']and x.get('encoding')=='gzip');Path(item['local_path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):m.validate_extension(plan,self.bucket)

    def test_existing_single_publisher_can_publish_both_catalogs(self):
        self.trained();plan=self.legacy.prepare();old=copy.deepcopy(self.legacy.history)
        with patch.object(m,'collector',return_value=self.f.p):
            m.extend_plan(plan,{},self.bucket)
            result=p.publish_prepared(plan,bucket=self.bucket,identity=self.legacy.identity)
        self.assertTrue(result['published'])
        payload=p.authenticated(json.loads(self.bucket.objects[p.EVIDENCE_HISTORY_KEY]),self.legacy.authority)
        self.assertEqual(payload['epochs'],old['epochs'])
        self.assertIn(m.FIELD,payload);self.assertIn('finalized_epoch_evidence',payload)

    def rewrite(self,dispatch=None,documents=None):
        f=self.f
        d=dispatch or json.loads((f.epoch/'coordinator/DISPATCH.json').read_bytes())['payload']
        f.write('coordinator/DISPATCH.json',f.sign(d))
        final=json.loads((f.epoch/'FINALIZED.json').read_bytes())['payload'];final['dispatch_sha256']=m.digest(f.sign(d));f.write('FINALIZED.json',f.sign(final))
        job=json.loads((f.epoch/'TRAINING-JOB.json').read_bytes())['payload'];job['training']['dispatch']['sha256']=final['dispatch_sha256']
        if documents is not None:job['training']['documents']=documents
        f.write('TRAINING-JOB.json',f.sign(job))
        report=json.loads((f.epoch/'TRAINING-RESULT.json').read_bytes())['payload'];report['report']['job_sha256']=m.digest(job);f.write('TRAINING-RESULT.json',f.sign(report,f.worker))
    def test_extra_or_missing_signed_job_document_rejected(self):
        self.trained();job=json.loads((self.f.epoch/'TRAINING-JOB.json').read_bytes())['payload']
        for documents in ({},dict(job['training']['documents'],**{'f'*64:{'size':1}})):
            self.rewrite(documents=documents)
            with self.assertRaisesRegex(ValueError,'document set'):m.collect(self.f.p,self.bucket,25)
    def test_duplicate_selected_task_or_document_rejected(self):
        self.trained();original=json.loads((self.f.epoch/'coordinator/DISPATCH.json').read_bytes())['payload']
        for field in ('task','document','slot'):
            d=copy.deepcopy(original);second=copy.deepcopy(d['selected'][0]);second.update(index=2)
            second['document'].update(key='private/other',sha256='f'*64,slot=1)
            if field=='task':second['index']=1
            elif field=='document':second['document']['key']='private/doc'
            else:second['document']['slot']=0
            d['selected'].append(second);d.update(selected_task_count=2,selected_rollout_count=16);self.rewrite(d)
            with self.assertRaisesRegex(ValueError,'unique selected'):m.collect(self.f.p,self.bucket,25)
    def test_signed_document_row_identity_mismatch_rejected(self):
        self.trained();original=json.loads(self.bucket.objects['private/doc'])
        for field,value in [('miner','e'*64),('slot',1),('env_id','other'),('index',99)]:
            doc=copy.deepcopy(original)
            if field in ('env_id','index'):doc['batch'][field]=value
            else:doc[field]=value
            raw=p.canonical(doc);self.bucket.objects['private/doc']=raw
            d=json.loads((self.f.epoch/'coordinator/DISPATCH.json').read_bytes())['payload'];d['selected'][0]['document']['sha256']=p.digest(raw);d['selected'][0]['batch_sha256']=m.digest(doc['batch'])
            self.rewrite(d,{p.digest(raw):{'size':len(raw)}})
            with self.assertRaisesRegex(ValueError,'document binding'):m.collect(self.f.p,self.bucket,25)
    def test_multiple_tasks_from_one_identity_remain_allowed(self):
        self.trained();d=json.loads((self.f.epoch/'coordinator/DISPATCH.json').read_bytes())['payload'];second=copy.deepcopy(d['selected'][0]);second['index']=2;second['document'].update(key='private/other',sha256='f'*64,slot=1)
        d['selected'].append(second);d.update(selected_task_count=2,selected_rollout_count=16)
        self.assertEqual(len(m.selection_closure(d,self.f.manifest)),2)
