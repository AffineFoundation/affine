"""Portable publisher controls: tiny signed fixtures, no network or live state."""
import copy
import gzip
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import boto3
from botocore.config import Config
from nacl.signing import SigningKey

from ops import publish_finalized_epoch_evidence as p


class ConditionalConflict(Exception):
    response={'Error':{'Code':'PreconditionFailed'}}


class Bucket:
    def __init__(self):
        self.objects={};self.reads=[];self.writes=[];self.routes=[]
        self.client=self;self.name='test';self.race=False;self.corrupt=False
        self.signing_client=boto3.client('s3',endpoint_url='https://example.invalid',region_name='auto',
            aws_access_key_id='synthetic-access',aws_secret_access_key='synthetic-secret',
            config=Config(signature_version='s3v4'))
        self._request_signer=self.signing_client._request_signer

    def snapshot(self,key,limit):
        self.reads.append(key)
        if key not in self.objects:return None
        raw=self.objects[key]
        if len(raw)>limit:raise ValueError('bounded GET')
        return {'data':raw,'size':len(raw),'etag':p.digest(raw)}

    def presign(self,key,operation,expires):
        self.routes.append((key,operation,expires))
        return self.signing_client.generate_presigned_url(operation,Params={'Bucket':self.name,'Key':key},ExpiresIn=expires)

    def put(self,key,data,content_type):
        self.writes.append((key,content_type))
        self.objects[key]=data+b'!' if self.corrupt else data

    def put_object(self,**kwargs):
        if self.race or kwargs['IfMatch']!=p.digest(self.objects[kwargs['Key']]):raise ConditionalConflict()
        self.put(kwargs['Key'],kwargs['Body'],kwargs['ContentType'])


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.state=self.root/'state';self.state.mkdir()
        self.work=self.root/'work';self.bucket=Bucket()
        key=SigningKey(bytes(range(32)))
        self.identity=SimpleNamespace(key=key,id=key.verify_key.encode().hex())
        self.authority=self.identity.id;self.now=1700001000;self.calls=[]
        self.epoch=self.proofs(14)
        self.history={'version':1,'authority':self.authority,'refreshed_at':17,
            'routes_expire_at':18,'epochs':[{'epoch_id':'legacy-4','objects':{'training':{'url':'https://old.invalid/a'}}}],
            'infrastructure_skips':[{'round':12}],'source_reconstruction_supplements':[]}
        self.bucket.objects[p.HISTORY_KEY]=p.canonical(p.signed(self.history,self.identity))
        self.training_key='public/'+self.epoch+'/training.json'
        self.bucket.objects[self.training_key]=p.canonical(p.signed({'source_epoch':self.epoch,
            'input_checkpoint':'a'*64,'checkpoint':'b'*64,'state_authority_committed':True},self.identity))

    def tearDown(self):self.temp.cleanup()

    def proofs(self,round_number,*,complete=True,at=1700000300):
        epoch='nonpayable-live-reward-math-v1--1700000000-'+str(round_number)
        manifest={'epoch':epoch,'checkpoint':{'id':'a'*64},'deadline':1700000100}
        completion={'epoch':epoch,'round':round_number,'checkpoint':'a'*64,'next_checkpoint':'b'*64,'completed_at':at}
        (self.state/(epoch+'-first-signed-manifest.json')).write_bytes(p.canonical(p.signed(manifest,self.identity)))
        if complete:(self.state/(epoch+'-signed-learner-completion.json')).write_bytes(p.canonical(p.signed(completion,self.identity)))
        return epoch

    def project(self,state,epoch,*,finalized_record,authority,input_loader):
        self.calls.append(epoch)
        if 'training_envelope' not in finalized_record:
            with self.assertRaises(FileNotFoundError):input_loader('public/'+epoch+'/training.json')
        with self.assertRaises(ValueError):input_loader('private/anything')
        return {name:{'status':'available','payload':{'prompt':[1,2],'output':[3,4],
                    'text':'See https://example.org/reference'}} for name in p.CATEGORIES[:-1]}

    def prepare(self,**kwargs):
        return p.prepare_cycle(state=self.state,source_root=self.root/'empty-source',bucket=self.bucket,
            workdir=self.work,authority=self.authority,now=kwargs.pop('now',self.now),
            training_projector=kwargs.pop('training_projector',self.project),**kwargs)

    def catalog(self,plan):
        item=next(i for i in plan['artifacts'] if '/catalogs/' in i['key'])
        return json.loads(Path(item['local_path']).read_bytes())

    def category(self,plan,category):
        item=next(i for i in plan['artifacts'] if '/'+category+'/' in i['key'])
        return json.loads(gzip.decompress(Path(item['local_path']).read_bytes()))

    def test_finalization_requires_original_signed_manifest_and_durable_completion(self):
        for n in (13,77,78,90,91):self.proofs(n)
        self.proofs(122,complete=False);self.proofs(123,at=self.now+1)
        tampered=self.proofs(124)
        path=self.state/(tampered+'-signed-learner-completion.json')
        value=json.loads(path.read_bytes());value['payload']['next_checkpoint']='c'*64
        path.write_bytes(p.canonical(value))
        found,_,rejected=p.discover_finalized(self.state,self.authority,now=self.now)
        self.assertEqual([r['round'] for r in found.values()],[14,77,78,90,91])
        self.assertEqual(len(rejected),2)
        self.assertEqual(found[self.epoch]['run_label'],'stable-old-14-77')
        self.assertFalse(list(self.state.glob('*scores*')))

    def test_prepare_readonly_preserves_legacy_and_real_evaluation_missing_status(self):
        plan=self.prepare();payload=copy.deepcopy(plan['history_payload'])
        payload.pop('finalized_epoch_evidence')
        self.assertEqual(payload,self.history);self.assertEqual(self.bucket.writes,[])
        self.assertEqual(self.calls,[self.epoch]);self.assertTrue(all(op=='get_object' for _,op,_ in self.bucket.routes))
        category=self.category(plan,'evaluations')
        self.assertEqual(category['status'],'unavailable')
        self.assertEqual(category['availability'],'not_retained_or_not_evaluated')
        self.assertNotEqual(category.get('reason'),'projection_status_missing')

    def test_real_signed_training_projection_and_tokens_cross_boundary(self):
        from dashboard import test_training_evidence_projection as fixtures
        from dashboard.training_evidence_projection import project_epoch
        fixture=fixtures.TrainingEvidenceTests();fixture.setUp()
        try:
            self.identity=SimpleNamespace(key=fixture.key,id=fixture.authority);self.authority=fixture.authority
            self.state=fixture.state;self.epoch=fixture.epoch
            for suffix,key in (('-first-signed-manifest.json','manifest_envelope'),('-signed-learner-completion.json','completion_envelope')):
                (self.state/(self.epoch+suffix)).write_bytes(p.canonical(fixture.record[key]))
            self.history['authority']=self.authority
            self.bucket.objects[p.HISTORY_KEY]=p.canonical(p.signed(self.history,self.identity))
            self.bucket.objects['public/'+self.epoch+'/training.json']=p.canonical(fixture.record['training_envelope'])
            plan=self.prepare(training_projector=project_epoch)
            p.validate_prepared(plan,self.identity)
            inputs=self.category(plan,'training_inputs')
            self.assertEqual(inputs['status'],'available')
            self.assertEqual(inputs['payload']['batches'][0]['rollouts'][0]['turns'][0]['output'],[3,4])
            self.assertNotIn('PRIVATE',json.dumps(inputs))
        finally:fixture.tearDown()

    def test_real_authenticated_evaluation_output_survives_category_wrapper(self):
        from dashboard import test_evaluation_evidence_projection as fixtures
        from dashboard import evaluation_evidence_projection as evaluation
        fixture=fixtures.EvaluationEvidenceTests();fixture.setUp()
        with patch.object(evaluation,'AUTHORITY',fixture.authority):
            row=evaluation.project_ack(fixture.fixture(),'b'*64)
        def collect(root,finalized):
            return {epoch:{'version':'evaluation-version','epoch_id':epoch,'status':'available','evaluations':[row]} for epoch in finalized}
        plan=self.prepare(evaluation_collector=collect)
        self.assertEqual(len(self.category(plan,'evaluations')['evaluations'][0]['tasks']),32)
        p.validate_prepared(plan,self.identity)

    def test_unsigned_history_rejected_before_projection_or_put(self):
        value=json.loads(self.bucket.objects[p.HISTORY_KEY]);value['payload']['epochs']=[]
        self.bucket.objects[p.HISTORY_KEY]=p.canonical(value)
        with self.assertRaises(Exception):self.prepare()
        self.assertEqual(self.calls,[]);self.assertEqual(self.bucket.writes,[])

    def test_projection_sanitizer_withholds_secrets_but_keeps_benign_rollout_text(self):
        for value in ({'url':'https://secret'}, {'output':[1],'full_vocab_logprobs':[[.1]]},
                      {'raw_logs':'example'}, {'metadata':'/home/private/file'}, {'text':'https://host/?token=secret'}):
            with self.assertRaises(ValueError):p.safe_projection(value)
        p.safe_projection({'text':'Use /tmp/example or https://example.org','output':[1,2]})
        def unsafe(*args,**kwargs):return {'training':{'status':'available','url':'https://secret'}}
        plan=self.prepare(training_projector=unsafe)
        self.assertEqual(self.category(plan,'training')['reason'],'projection_withheld_by_publication_sanitizer')
        p.validate_prepared(plan,self.identity)

    def test_missing_training_fetched_once_and_cached_for_short_retry(self):
        del self.bucket.objects[self.training_key]
        self.prepare();self.prepare(now=self.now+60)
        self.assertEqual(self.bucket.reads.count(self.training_key),1)
        self.prepare(now=self.now+301)
        self.assertEqual(self.bucket.reads.count(self.training_key),2)

    def test_cached_receipt_and_projection_skip_unchanged_work_then_refresh(self):
        a=self.prepare();b=self.prepare(now=self.now+60)
        self.assertEqual(self.bucket.reads.count(self.training_key),1);self.assertEqual(len(self.calls),1)
        self.assertEqual(self.catalog(a)['epochs'][0]['artifacts']['training_inputs']['sha256'],
                         self.catalog(b)['epochs'][0]['artifacts']['training_inputs']['sha256'])
        self.prepare(now=self.now+p.PROJECTION_CACHE_SECONDS+1)
        self.assertEqual(self.bucket.reads.count(self.training_key),2);self.assertEqual(len(self.calls),2)

    def test_local_metrics_change_invalidates_cache_without_authorizing_new_bytes(self):
        self.prepare()
        (self.state/(self.epoch+'-training-metrics.json')).write_bytes(p.canonical({'loss':999}))
        plan=self.prepare(now=self.now+60)
        self.assertEqual(self.bucket.reads.count(self.training_key),2);self.assertEqual(len(self.calls),2)
        self.assertNotIn('999',json.dumps(self.category(plan,'training')))

    def test_recent_root_seed_is_authenticated_and_reused_without_duplicate_get(self):
        self.work.mkdir(mode=0o700)
        directory=self.work/'training-cache';directory.mkdir(mode=0o700)
        path=directory/(self.epoch+'.signed.json')
        path.write_bytes(self.bucket.objects[self.training_key]);os.utime(path,(self.now-60,self.now-60))
        self.prepare()
        self.assertNotIn(self.training_key,self.bucket.reads)
        self.assertTrue(path.with_name(self.epoch+'.meta.json').exists())

    def test_bad_signed_training_never_enters_cache_or_publication(self):
        value=json.loads(self.bucket.objects[self.training_key]);value['payload']['checkpoint']='c'*64
        self.bucket.objects[self.training_key]=p.canonical(value)
        plan=self.prepare()
        self.assertEqual(plan['private_projection_errors'][0]['category'],'training_source')
        self.assertEqual(self.bucket.writes,[])
        self.assertFalse(list((self.work/'training-cache').glob('*.signed.json')))

    def test_publish_hashes_compressed_bytes_and_exports_exact_signed_history(self):
        plan=self.prepare();destination=self.root/'history.json'
        receipt=p.publish_prepared(plan,bucket=self.bucket,identity=self.identity,history_export_path=destination)
        raw=self.bucket.objects[p.EVIDENCE_HISTORY_KEY]
        self.assertEqual(destination.read_bytes(),raw);self.assertEqual(self.bucket.objects[p.HISTORY_KEY],raw)
        self.assertTrue(receipt['legacy_history_extension_written'])
        self.assertEqual(p.authenticated(json.loads(raw),self.authority)['epochs'],self.history['epochs'])
        for item in plan['artifacts']:
            stored=self.bucket.objects[item['key']]
            self.assertEqual(p.digest(stored),item['sha256']);self.assertEqual(len(stored),item['size'])

    def test_legacy_fields_cannot_be_resigned_after_plan_tampering(self):
        plan=self.prepare();plan['history_payload']['epochs']=[]
        with self.assertRaisesRegex(ValueError,'legacy'):p.publish_prepared(plan,bucket=self.bucket,identity=self.identity)
        self.assertEqual(self.bucket.writes,[])

    def test_changed_history_aborts_before_any_write(self):
        plan=self.prepare();self.history['refreshed_at']=999
        self.bucket.objects[p.HISTORY_KEY]=p.canonical(p.signed(self.history,self.identity))
        with self.assertRaisesRegex(ValueError,'changed'):p.publish_prepared(plan,bucket=self.bucket,identity=self.identity)
        self.assertEqual(self.bucket.writes,[])

    def test_conditional_race_preserves_independent_and_static_index(self):
        plan=self.prepare();old=self.bucket.objects[p.HISTORY_KEY];self.bucket.race=True
        destination=self.root/'history.json'
        receipt=p.publish_prepared(plan,bucket=self.bucket,identity=self.identity,history_export_path=destination)
        self.assertFalse(receipt['legacy_history_extension_written'])
        self.assertEqual(self.bucket.objects[p.HISTORY_KEY],old)
        self.assertEqual(destination.read_bytes(),self.bucket.objects[p.EVIDENCE_HISTORY_KEY])

    def test_changed_staging_and_corrupt_remote_readback_never_sign_history(self):
        plan=self.prepare();path=Path(plan['artifacts'][0]['local_path']);original=path.read_bytes()
        path.write_bytes(original+b'!')
        with self.assertRaises(ValueError):p.publish_prepared(plan,bucket=self.bucket,identity=self.identity)
        self.assertEqual(self.bucket.writes,[])
        path.write_bytes(original);self.bucket.corrupt=True
        with self.assertRaises(ValueError):p.publish_prepared(plan,bucket=self.bucket,identity=self.identity)
        self.assertNotIn(p.EVIDENCE_HISTORY_KEY,self.bucket.objects)

    def test_put_route_and_unreferenced_artifact_rejected_before_write(self):
        plan=self.prepare();plan['history_payload']['finalized_epoch_evidence']['catalog']['method']='PUT'
        with self.assertRaises(ValueError):p.publish_prepared(plan,bucket=self.bucket,identity=self.identity)
        self.assertEqual(self.bucket.writes,[])
        plan=self.prepare();plan['artifacts'].append(dict(plan['artifacts'][0],key='public/epoch-evidence/extra.json'))
        with self.assertRaises(ValueError):p.publish_prepared(plan,bucket=self.bucket,identity=self.identity)
        self.assertEqual(self.bucket.writes,[])

    def test_mutated_artifact_with_new_hash_still_cannot_cross_sanitizer(self):
        plan=self.prepare();catalog=self.catalog(plan)
        ref=catalog['epochs'][0]['artifacts']['training']
        item=next(i for i in plan['artifacts'] if i['key']==ref['key'])
        document=json.loads(gzip.decompress(Path(item['local_path']).read_bytes()));document['url']='https://secret'
        plain=p.canonical(document);raw=gzip.compress(plain,mtime=0)
        old_hash=item['sha256'];new_hash=p.digest(raw)
        item.update(sha256=new_hash,size=len(raw),uncompressed_size=len(plain),key=item['key'].replace(old_hash,new_hash))
        Path(item['local_path']).write_bytes(raw)
        ref.update({k:item[k] for k in ('sha256','size','key','uncompressed_size')})
        catitem=next(i for i in plan['artifacts'] if '/catalogs/' in i['key']);catraw=p.canonical(catalog)
        old_hash=catitem['sha256'];new_hash=p.digest(catraw)
        catitem.update(sha256=new_hash,size=len(catraw),key=catitem['key'].replace(old_hash,new_hash))
        Path(catitem['local_path']).write_bytes(catraw)
        plan['history_payload']['finalized_epoch_evidence']['catalog'].update({k:catitem[k] for k in ('sha256','size','key')})
        with self.assertRaises(ValueError):p.publish_prepared(plan,bucket=self.bucket,identity=self.identity)
        self.assertEqual(self.bucket.writes,[])

    def test_repeated_publication_skips_existing_content_uploads(self):
        plan=self.prepare();p.publish_prepared(plan,bucket=self.bucket,identity=self.identity)
        old_keys={key for key,_ in self.bucket.writes if '/epoch-evidence/' in key}
        self.bucket.writes=[];self.bucket.reads=[]
        plan=self.prepare(now=self.now+60);p.publish_prepared(plan,bucket=self.bucket,identity=self.identity)
        self.assertFalse(old_keys&{key for key,_ in self.bucket.writes})
        self.assertFalse(old_keys&set(self.bucket.reads))

    def test_extra_extension_secret_and_unrelated_url_rejected_before_put(self):
        plan=self.prepare();plan['history_payload']['finalized_epoch_evidence']['secret']='synthetic'
        with self.assertRaises(ValueError):p.publish_prepared(plan,bucket=self.bucket,identity=self.identity)
        self.assertEqual(self.bucket.writes,[])
        plan=self.prepare();plan['history_payload']['finalized_epoch_evidence']['catalog']['url']='https://unrelated.invalid/?token=synthetic'
        with self.assertRaises(ValueError):p.publish_prepared(plan,bucket=self.bucket,identity=self.identity)
        self.assertEqual(self.bucket.writes,[])

    def test_exact_historical_empty_receipt_requires_unchanged_checkpoint(self):
        closure={'input_checkpoint':'a'*64,'output_checkpoint':'a'*64}
        p._training_binding({'checkpoint':'a'*64,'status':'closed_no_eligible_batches'},self.epoch,closure)
        for document in ({'checkpoint':'a'*64,'status':'unexpected'},
                         {'checkpoint':'a'*64,'status':'closed_no_eligible_batches','extra':1}):
            with self.assertRaises(ValueError):p._training_binding(document,self.epoch,closure)
        with self.assertRaises(ValueError):p._training_binding({'checkpoint':'a'*64,'status':'closed_no_eligible_batches'},
            self.epoch,{'input_checkpoint':'a'*64,'output_checkpoint':'b'*64})

    def test_optional_population_is_separately_authenticated_and_cached(self):
        key='public/'+self.epoch+'/learner-population.json'
        self.bucket.objects[key]=p.canonical(p.signed({'epoch':self.epoch,'checkpoint':'a'*64,'exclusions':[]},self.identity))
        def project(*args,**kwargs):
            output=self.project(*args,**kwargs)
            if 'population_envelope' not in kwargs['finalized_record']:
                output['exclusions']=p.unavailable('signed_population_receipt_unavailable')
            return output
        first=self.prepare(training_projector=project)
        self.assertEqual(self.category(first,'exclusions')['status'],'available')
        self.prepare(training_projector=project,now=self.now+60)
        self.assertEqual(self.bucket.reads.count(key),1)

    def test_invalid_population_does_not_discard_authenticated_training_categories(self):
        key='public/'+self.epoch+'/learner-population.json'
        self.bucket.objects[key]=p.canonical(p.signed({'epoch':'wrong','checkpoint':'a'*64},self.identity))
        def project(*args,**kwargs):
            output=self.project(*args,**kwargs);output['exclusions']=p.unavailable('signed_population_receipt_unavailable')
            return output
        plan=self.prepare(training_projector=project)
        self.assertEqual(self.category(plan,'training')['status'],'available')
        self.assertEqual(self.category(plan,'exclusions')['reason'],'signed_population_evidence_unavailable')

    def test_log_and_receipt_hashes_invalidate_projection_cache(self):
        directory=self.root/'private-logs';directory.mkdir()
        raw=directory/(self.epoch+'.worker.log');receipt=directory/(self.epoch+'.receipt.json')
        raw.write_bytes(b'first');receipt.write_bytes(p.canonical({'fixture':1}))
        def project(*args,**kwargs):
            self.assertIn('trainer_log_record',kwargs['finalized_record'])
            return self.project(*args,**kwargs)
        self.prepare(training_projector=project,trainer_log_directory=directory)
        raw.write_bytes(b'second')
        self.prepare(training_projector=project,trainer_log_directory=directory,now=self.now+60)
        self.assertEqual(len(self.calls),2)

    def test_owned_pruning_only_after_success_keeps_four_plans_and_referenced_objects(self):
        plan=self.prepare();objects=self.work/'objects'
        for index in range(6):
            row=copy.deepcopy(plan);raw=p.canonical({'catalog_fixture':index});h=p.digest(raw);path=objects/(h+'.json')
            path.write_bytes(raw);row['artifacts'].append({'sha256':h,'local_path':str(path)})
            (self.work/f'prepared-{index}.private.json').write_bytes(p.canonical(row))
            (self.work/f'receipt-{index}.json').write_bytes(b'{}')
        orphan=objects/('0'*64+'.json');orphan.write_bytes(b'old owned projection')
        untouched=objects/'operator-note.txt';untouched.write_text('keep')
        external=self.root/'external.json';external.write_text('unchanged')
        link=objects/('1'*64+'.json');link.symlink_to(external)
        before={v.name for v in objects.iterdir()}
        with self.assertRaises(ValueError):p.prune_owned_staging(self.work,publication_receipt={'published':False})
        self.assertEqual(before,{v.name for v in objects.iterdir()})
        result=p.prune_owned_staging(self.work,publication_receipt={'published':True,'history_sha256':'a'*64})
        self.assertEqual(result,{'objects_removed':3,'prepared_plans_removed':2,'receipts_removed':2})
        self.assertEqual(len(list(self.work.glob('prepared-*.private.json'))),4)
        self.assertTrue(untouched.exists());self.assertTrue(link.is_symlink());self.assertEqual(external.read_text(),'unchanged')
        self.assertTrue(all(Path(item['local_path']).exists() for item in plan['artifacts']))

    def test_owned_pruning_refuses_unowned_workdir_and_escaping_plan(self):
        plan=self.prepare();path=self.work/'prepared-1.private.json'
        plan['artifacts'][0]['local_path']=str(self.root/'outside.json');path.write_bytes(p.canonical(plan))
        with self.assertRaises(ValueError):p.prune_owned_staging(self.work,publication_receipt={'published':True,'history_sha256':'a'*64})
        self.assertTrue(path.exists())
        (self.work/'.publisher-owned.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):p.prune_owned_staging(self.work,publication_receipt={'published':True,'history_sha256':'a'*64})


if __name__=='__main__':unittest.main()
