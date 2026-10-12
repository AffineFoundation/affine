"""Synthetic CPU integration with genuine ephemeral signatures; zero network,
model, grader, production-state or production-authority operations."""
import copy, hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import test_training_task_representatives as f
from test_training_task_representatives import r, AUTH, signed, fixture, make_wave, sha, canonical
from subnet import committed_training_inputs as inputs
from subnet import unaudited_training_execution as execution
from subnet import learner_selection_operator_bridge as peer
from ops import native_task_priority_history as history
from ops import native_task_representative_selection as selector_module
from ops.native_training_eligibility import _create, NativeNoUpdate


def prospective(tasks,miner_groups=None):
    p,a=fixture(tasks,miner_groups);m=p['payload']['original_signed_manifest']['payload']
    m['environments']=[{'env_id':'math','indices':sorted(set(tasks))}]
    m[r.FIELD].update(version=r.PRIORITY_VERSION,history_epochs=12,task_priority='untrained-in-prior12-first-stable-rank')
    p['payload']['original_signed_manifest']=signed(m)
    return signed(p['payload']),a

def snapshot(p,prior=(1,),fallback=None):
    m=p['payload']['original_signed_manifest']['payload'];binding=r.history_binding(m);rows=[]
    for j,number in enumerate(range(binding['round']-12,binding['round'])):
        inp=sha(['checkpoint',j]);out=sha(['checkpoint',j+1])if j<11 else m['checkpoint']['id']
        epoch='fixture--1000-'+str(number)
        closure=signed(dict(round=number,epoch=epoch,checkpoint=inp,next_checkpoint=out,completed_at=900+j,input_assurance='unaudited'))
        rows.append(dict(round=number,epoch=epoch,input_checkpoint=inp,output_checkpoint=out,
            updates=[dict(global_optimizer_step=9+j,indices=sorted(set(prior)))],
            provenance={n:dict(path='/synthetic/'+str(number)+'/'+n,sha256=sha([n,number]))for n in('closure','job','report')},signed_closure=closure))
    return signed(dict(binding,created_at=2001,outcome='original_order_fallback'if fallback else'priority',reason_code=fallback,records=[]if fallback else rows))

def freeze(p,path,h=None):
    with patch.object(r.secrets,'token_hex',return_value='f'*64):
        return r.freeze_draw(p,AUTH,path,now=2001,history_snapshot=h)

def finalized(p,a,d,invalid=()):
    waves=[]
    while True:
        progress=r.replay(p,AUTH,d,a,waves)
        if progress['complete']:break
        waves.append(make_wave(p,d,a,progress['next_submissions'],invalid=invalid))
    return dict(pool=p,draw=signed(d),authorization=a,waves=waves,result=signed(r.finalize(progress,d,r._policy(p['payload']['original_signed_manifest']['payload']),2002)))

def derived(documents):
    m=documents['pool']['payload']['original_signed_manifest']['payload'];result=documents['result']['payload'];draw=documents['draw']['payload']
    out=inputs.coverage_manifest(m,result['accepted_submissions'],seed=draw['seed'],captured_at=draw['captured_at'])
    out['native_training_eligibility_receipt']=r.derivation_receipt(documents,AUTH)
    if draw['blacklist_selection'] is not None: out['learner_blacklist_selection_snapshot']=draw['blacklist_selection']
    return out

class PriorityIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.path=self.root/'draw.json'
    def test_stable_partition_preserves_miner_order_reward_and_final_input_order(self):
        p,a=prospective([1,2,1,3,4],[0,1,2,3,4]);h=snapshot(p,[1,3]);d=freeze(p,self.path,h)
        fallback=copy.deepcopy(d);fallback['history_snapshot']=snapshot(p,fallback='missing_history')
        original=r.validate_draw(p,AUTH,fallback);actual=r.validate_draw(p,AUTH,d)
        self.assertEqual(actual[4],[t for t in original[4]if t[1]not in(1,3)]+[t for t in original[4]if t[1]in(1,3)])
        self.assertEqual(actual[5],original[5]);self.assertEqual(actual[3],original[3])
        docs=finalized(p,a,d);self.assertEqual(docs['result']['payload']['accepted_submissions'],[o for o in p['payload']['submissions']if o in docs['result']['payload']['accepted_submissions']])
    def test_real_history_signatures_and_earliest_cutoff_required(self):
        p,a=prospective([1,2]);h=snapshot(p)
        for field,value in [('signature','bad'),('signer','0'*64)]:
            x=copy.deepcopy(h);x[field]=value
            with self.assertRaises(ValueError):r.validate_history_snapshot(x,p['payload']['original_signed_manifest']['payload'],AUTH,at=2001)
        x=copy.deepcopy(h['payload']);x['records'][0]['signed_closure']['payload']['completed_at']=1001;x['records'][0]['signed_closure']=signed(x['records'][0]['signed_closure']['payload'])
        with self.assertRaisesRegex(ValueError,'closed before'):r.validate_history_snapshot(signed(x),p['payload']['original_signed_manifest']['payload'],AUTH,at=2001)
    def test_current_round_wrong_source_parent_counters_and_task_types_refused(self):
        p,a=prospective([1]);m=p['payload']['original_signed_manifest']['payload'];h=snapshot(p)['payload']
        mutations=[lambda x:x.update(source_sha256='0'*64),lambda x:x['records'][0].update(round=900),lambda x:x['records'][-1].update(output_checkpoint='0'*64),lambda x:x['records'][4]['updates'][0].update(global_optimizer_step=999),lambda x:x['records'][0]['updates'][0].update(indices=[True]),lambda x:x['records'][0]['updates'][0].update(indices=[1,1]),lambda x:x['records'].pop()]
        for mutate in mutations:
            x=copy.deepcopy(h);mutate(x)
            with self.assertRaises(ValueError):r.validate_history_snapshot(signed(x),m,AUTH,at=2001)
    def test_validly_signed_closures_cannot_bridge_checkpoint_or_step_discontinuity(self):
        p,a=prospective([1]);m=p['payload']['original_signed_manifest']['payload']
        for mode in ('checkpoint','step'):
            x=copy.deepcopy(snapshot(p)['payload'])
            if mode=='checkpoint':
                x['records'][4]['input_checkpoint']='9'*64
                closure=copy.deepcopy(x['records'][4]['signed_closure']['payload']);closure['checkpoint']='9'*64
                x['records'][4]['signed_closure']=signed(closure)
            else:
                x['records'][4]['updates'][0]['global_optimizer_step']+=1
            with self.assertRaises(ValueError):r.validate_history_snapshot(signed(x),m,AUTH,at=2001)
    def test_corrupt_closure_not_hidden_by_outer_signature(self):
        p,a=prospective([1]);h=snapshot(p)['payload'];h['records'][0]['signed_closure']['signature']='bad'
        with self.assertRaises(ValueError):freeze(p,self.path,signed(h))
    def test_old_history_tasks_need_not_be_current_cooled_members(self):
        p,a=prospective([1,2]);h=snapshot(p,[7000]);d=freeze(p,self.path,h)
        self.assertEqual(len(r.validate_draw(p,AUTH,d)[4]),2)
    def test_fallback_is_explicit_immutable_and_original_hash_order(self):
        p,a=prospective([1,2,3]);h=snapshot(p,fallback='missing_training_report');d=freeze(p,self.path,h);before=self.path.read_bytes()
        with patch.object(r.secrets,'token_hex',side_effect=AssertionError('reroll')):
            self.assertEqual(r.freeze_draw(p,AUTH,self.path,now=4000),d)
        with self.assertRaisesRegex(ValueError,'cannot be replaced'):r.freeze_draw(p,AUTH,self.path,now=4000,history_snapshot=snapshot(p,[2]))
        self.assertEqual(before,self.path.read_bytes());common=dict(seed=d['seed'],epoch=p['payload']['original_signed_manifest']['payload']['epoch'],checkpoint='a'*64,source='b'*64)
        self.assertEqual(r.validate_draw(p,AUTH,d)[4],sorted([('math',i)for i in(1,2,3)],key=lambda t:(sha(dict(common,domain='distinct-training-task-v1',task=t)),t)))
    def test_fallback_cannot_smuggle_priority_records(self):
        p,a=prospective([1]);h=snapshot(p)['payload'];h.update(outcome='original_order_fallback',reason_code='missing')
        with self.assertRaisesRegex(ValueError,'fallback'):freeze(p,self.path,signed(h))
    def test_native_invalid_and_unresolved_follow_existing_same_task_fallback(self):
        p,a=prospective([1,1,2],[0,1,2]);d=freeze(p,self.path,snapshot(p,[2]));first=r.replay(p,AUTH,d,a,[])['next_submissions'];bad=next(o['sha256']for o in first if o['learner_admission']['payload']['original_commitment']['payload']['batches'][0]['index']==1)
        for k in('invalid','unresolved'):
            wave=make_wave(p,d,a,first,**{k:[bad]});middle=r.replay(p,AUTH,d,a,[wave]);self.assertEqual(len(middle['next_submissions']),1);self.assertNotEqual(middle['next_submissions'][0]['sha256'],bad)
            last=make_wave(p,d,a,middle['next_submissions']);self.assertEqual(r.replay(p,AUTH,d,a,[wave,last])['accepted_count'],2)
    def test_600s_partial_and_zero_update_never_forge_native_failure(self):
        p,a=prospective([1,2,3]);m=p['payload']['original_signed_manifest']['payload'];m[r.FIELD]['max_native_documents_per_wave']=1;p['payload']['original_signed_manifest']=signed(m);p=signed(p['payload']);d=freeze(p,self.path,snapshot(p));progress=r.replay(p,AUTH,d,a,[])
        with self.assertRaises(ValueError):r.finalize(progress,d,r._policy(m),2600)
        zero=r.finalize(progress,d,r._policy(m),2601);self.assertEqual(zero['disposition'],'no_update');self.assertEqual(zero['checked_count'],0)
        wave=make_wave(p,d,a,progress['next_submissions']);progress=r.replay(p,AUTH,d,a,[wave]);partial=r.finalize(progress,d,r._policy(m),2601);self.assertEqual(partial['accepted_count'],1);self.assertEqual(partial['disposition'],'train');self.assertEqual(partial['completion_reason'],'native_budget_exhausted')
    def test_full_preparation_scope_reconstructs_signed_v3_history_and_selected_inputs(self):
        p,a=prospective([1,2,3]);d=freeze(p,self.path,snapshot(p,[1]));docs=finalized(p,a,d);m=derived(docs);envelope=signed(m)
        scope=execution.preparation_scope(p['payload']['original_signed_manifest'],envelope,docs['result']['payload']['accepted_submissions'],docs,AUTH)
        self.assertEqual(scope['native_eligibility_receipt']['history_snapshot_sha256'],sha(d['history_snapshot']))
        x=copy.deepcopy(docs);x['draw']['payload']['history_snapshot']=snapshot(p,[2]);x['draw']=signed(x['draw']['payload'])
        with self.assertRaises(ValueError):execution.preparation_scope(p['payload']['original_signed_manifest'],envelope,docs['result']['payload']['accepted_submissions'],x,AUTH)
    def test_committed_job_rejects_old_receipt_for_new_policy_and_tampered_inputs(self):
        p,a=prospective([1,2]);d=freeze(p,self.path,snapshot(p));docs=finalized(p,a,d);m=derived(docs)
        job=dict(role='train',training_input_policy=m['training_input_policy'],training_policy=m['training_policy'],source_files={'subnet/committed_training_inputs.py':'e'*64},submissions=docs['result']['payload']['accepted_submissions'])
        inputs.validate_job(job,m,AUTH)
        n=copy.deepcopy(m);n['native_training_eligibility_receipt'].pop('history_snapshot_sha256');n['native_training_eligibility_receipt']['version']=r.RECEIPT_VERSION
        with self.assertRaisesRegex(ValueError,'receipt version'):inputs.validate_job(job,n,AUTH)
        j=copy.deepcopy(job);j['submissions'][0]['size']+=1
        with self.assertRaises(ValueError):inputs.validate_job(j,m,AUTH)

class ActualSelectorCaller(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.state=Path(self.tmp.name);(self.state/'roles').mkdir()
    def setup_selector(self,tasks=(1,1,2),prior=(2,),missing=False):
        p,a=prospective(list(tasks),list(range(len(tasks))));m=p['payload']['original_signed_manifest']['payload'];pop=dict(manifest=m,submissions=p['payload']['submissions'])
        raw=canonical(pop);p['payload']['original_population_file_sha256']=hashlib.sha256(raw).hexdigest();p=signed(p['payload'])
        poolpath,_=r.collection_paths(self.state,m['epoch']);_create(poolpath,p);(self.state/(m['epoch']+'-learner-population.json')).write_bytes(raw);_create(self.state/(m['epoch']+'-learner-training-selection.json'),{})
        controller=SimpleNamespace(state=self.state,authority=SimpleNamespace(id=AUTH),signed=signed)
        selector=SimpleNamespace(controller=controller,authority=AUTH,authorization=a,policy=a['payload'])
        h=snapshot(p,prior);rows=copy.deepcopy(h['payload']['records'])
        for row in rows:
            path=self.state/(row['epoch']+'-signed-learner-completion.json');raw=canonical(row.pop('signed_closure'));path.write_bytes(raw);row['provenance']['closure']=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())
        data=dict(version='finalized-trained-task-exposure-input-v1',assurance='signed-closure-and-job-linked-unsigned-trainer-report',records=rows)
        provider=patch.object(history.exposure,'evidence_provider',side_effect=history.exposure.ExposureUnavailable('missing_training_report')if missing else None,return_value=data)
        return p,a,m,selector,provider
    def run_selector(self,p,a,m,selector,provider,all_invalid=False):
        def grade(_selector,root,paths,context,expires,fd):
            draw=json.loads((root.parent.parent/'draw.json').read_bytes())
            objects=context['payload']['submissions']
            return make_wave(p,draw,a,objects,invalid=[o['sha256']for o in objects]if all_invalid else ())['grades']['payload']
        with provider,patch.object(selector_module.time,'time',return_value=2001),patch.object(selector_module,'prepare_documents',return_value=[]),patch.object(selector_module,'grade_wave',side_effect=grade):
            return selector_module.select(selector,m,p['payload']['submissions'])
    def test_actual_selector_freezes_history_and_signed_waves_then_reentry_no_provider(self):
        p,a,m,sel,provider=self.setup_selector();result=self.run_selector(p,a,m,sel,provider)
        root=self.state/'native-outcome-eligibility'/m['epoch'];docs=selector_module.read_documents(root);self.assertEqual(r.derivation_receipt(docs,AUTH),result[0]['native_training_eligibility_receipt'])
        with patch.object(history.exposure,'evidence_provider',side_effect=AssertionError('new history')),patch.object(selector_module,'grade_wave',side_effect=AssertionError('repeat grade')),patch.object(selector_module.time,'time',return_value=4000):
            self.assertEqual(selector_module.select(sel,m,p['payload']['submissions']),result)
    def test_actual_reentry_after_checkpoint_read_url_refresh_keeps_original_history(self):
        p,a,m,sel,provider=self.setup_selector();result=self.run_selector(p,a,m,sel,provider)
        root=self.state/'native-outcome-eligibility'/m['epoch'];history_before=(root/'priority-history.ROOT-SIGNED.json').read_bytes();draw_before=(root/'draw.json').read_bytes()
        refreshed=copy.deepcopy(m);refreshed['checkpoint']['read_urls']={'shard':'https://synthetic.invalid/refreshed-read-only-capability'}
        with patch.object(history.exposure,'evidence_provider',side_effect=AssertionError('history refreshed')),patch.object(selector_module,'grade_wave',side_effect=AssertionError('native rerun')),patch.object(selector_module.time,'time',return_value=4000):
            actual=selector_module.select(sel,refreshed,p['payload']['submissions'])
        self.assertEqual(actual[1],result[1]);self.assertEqual(actual[0]['native_training_eligibility_receipt'],result[0]['native_training_eligibility_receipt']);self.assertEqual((root/'priority-history.ROOT-SIGNED.json').read_bytes(),history_before);self.assertEqual((root/'draw.json').read_bytes(),draw_before)
    def test_actual_selector_missing_history_runs_original_order_fallback(self):
        p,a,m,sel,provider=self.setup_selector(missing=True);result=self.run_selector(p,a,m,sel,provider)
        h=json.loads((self.state/'native-outcome-eligibility'/m['epoch']/'priority-history.ROOT-SIGNED.json').read_bytes())['payload'];self.assertEqual(h['outcome'],'original_order_fallback');self.assertEqual(h['reason_code'],'missing_training_report');self.assertEqual(len(result[1]),2)
    def test_actual_selector_all_invalid_truthful_no_update(self):
        p,a,m,sel,provider=self.setup_selector()
        with self.assertRaises(NativeNoUpdate):self.run_selector(p,a,m,sel,provider,all_invalid=True)
        result=json.loads((self.state/'native-outcome-eligibility'/m['epoch']/'result.ROOT-SIGNED.json').read_bytes())['payload'];self.assertEqual(result['disposition'],'no_update');self.assertEqual(result['checked_count'],3);self.assertFalse(list((self.state/'roles').iterdir()))
    def test_existing_corrupt_snapshot_refuses_not_replaced_or_fallback(self):
        p,a,m,sel,provider=self.setup_selector();root=self.state/'native-outcome-eligibility'/m['epoch'];root.mkdir(parents=True);_create(root/'priority-history.ROOT-SIGNED.json',signed({'corrupt':True}))
        with provider,patch.object(selector_module.time,'time',return_value=2001),self.assertRaises(ValueError):selector_module.select(sel,m,p['payload']['submissions'])



if __name__=='__main__':unittest.main(verbosity=2)
