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
    def trained(self):
        f=self.f;final=f.trained();path=f.epoch/'coordinator/DISPATCH.json';d=json.loads(path.read_bytes())['payload']
        doc=dict(version='complete-group-training-document-v1',epoch=f.manifest['epoch'],checkpoint='b'*64,batch=dict(env_id='math',index=1,task_hash='a'*64,rollouts=[dict(seed=0,classification='positive',turns=[dict(prompt=[1,2],output=[3,4],stop_reason='eos',logprobs=[-1],full_vocab_logprobs=['NO'])])]))
        raw=p.canonical(doc);self.bucket.objects['private/doc']=raw
        d.update(selected=[dict(miner='d'*64,document={'key':'private/doc','sha256':p.digest(raw)},batch_sha256=m.digest(doc['batch']))],excluded=[])
        f.write('coordinator/DISPATCH.json',f.sign(d));final['dispatch_sha256']=m.digest(f.sign(d))
        job=json.loads((f.epoch/'TRAINING-JOB.json').read_bytes())['payload'];job['training']['dispatch']={'sha256':final['dispatch_sha256']};f.write('TRAINING-JOB.json',f.sign(job))
        report=json.loads((f.epoch/'TRAINING-RESULT.json').read_bytes())['payload'];report['report']['job_sha256']=m.digest(job);f.write('TRAINING-RESULT.json',f.sign(report,f.worker));f.write('FINALIZED.json',f.sign(final));return final
    def test_open_never_reads_inputs_or_corrupt_opening(self):
        self.f.write('coordinator/OPENING.json',{'bad':True});self.assertEqual(m.collect(self.f.p,self.bucket,25),[]);self.assertEqual(self.bucket.reads,[])
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
