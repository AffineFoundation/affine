"""CPU orchestration tests: real raw admissions/signatures; synthetic native scores.

The native grading boundary is deliberately patched for deterministic failure
and replay controls. These are not GPU or real MATH-label qualification claims.
"""
import base64,copy,hashlib,importlib.util,json,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
from nacl.signing import SigningKey
from subnet import committed_training_inputs as c, training_task_representatives as r
from subnet.training_receipts import sha, computation_binding
from subnet.storage import canonical
from ops.native_training_eligibility import NativeEligibilitySelector,NativeNoUpdate,_create
from ops import native_task_representative_selection as ns
from ops.native_training_outcome_filter import AUTHORIZATION_VERSION,MULTI_VERSION

KEY=SigningKey(hashlib.sha256(b'representative-integration-test').digest());AUTH=KEY.verify_key.encode().hex()
def signed(v,key=KEY):return dict(payload=copy.deepcopy(v),signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(v)).signature).decode())

class Bucket:
    def __init__(self):self.objects={}
    def put(self,k,v):self.objects[k]=v
    def json(self,k,v):self.put(k,canonical(v))
    def get(self,k):return self.objects[k]
    def get_bounded(self,k,limit):return self.objects[k][:limit+1]
    def presign(self,k,*args):return 'memory:'+k

def fixture(root,tasks=(1,1,2),bad=(),representatives=True):
    from subnet import harness
    now=time.time();bucket=Bucket();receipts={};keys=[SigningKey(hashlib.sha256(('miner'+str(i)).encode()).digest()) for i in range(len(tasks))]
    m=dict(epoch='test--100-900',checkpoint=dict(id='a'*64),source_bundle=dict(sha256='b'*64),
        start=now-1220,deadline=now-20,K=4,L=4,max_batches=9,
        capabilities={k.verify_key.encode().hex():'synthetic' for k in keys},
        trainer_state_binding=dict(global_step_before=21,parent=dict(optimizer_steps=21,inference_checkpoint='a'*64)),
        training_policy='bf16-cpu-fp32-master-task-normalized-persistent-v4',training_input_policy=c.VERSION,
        training_task_capacity=dict(version=c.CAPACITY_VERSION,max_tasks=512),
        harness_source_hash=harness.source_hash(),
        environments=[dict(env_id='math',indices=list(range(10)),spec=dict(id='math',version='synthetic-v1'),harness=dict(harness.DEFAULT))])
    if representatives:m[r.FIELD]=dict(version=r.VERSION,max_candidate_documents=2304,
        max_total_input_bytes=4_608_000_000,max_native_documents_per_wave=2,max_native_wall_seconds=600,
        exhausted_task_rule='advance-fixed-task-order')
    for i,(task,key) in enumerate(zip(tasks,keys)):
        miner=key.verify_key.encode().hex();rolls=[]
        for j in range(8):
            rolls.append(dict(env_id='math',index=task,sample_index=task,environment_version='synthetic-v1',
                task_hash=sha(['task',task]),classification='positive' if j<4 else 'negative',reward=1 if j<4 else 0,
                turns=[dict(prompt=[1,2],output=[100+i*10+j,99],text='synthetic')]))
        if i in bad:rolls[0]['turns'][0]['output']=[True]
        batch=dict(schema=2,epoch=m['epoch'],checkpoint='a'*64,env_id='math',index=task,sample_index=task,
            environment_version='synthetic-v1',rollouts=rolls)
        raw=canonical(dict(version=c.ARTIFACT_VERSION,epoch=m['epoch'],checkpoint='a'*64,miner=miner,slot=0,batch=batch))
        digest=hashlib.sha256(raw).hexdigest();proof=sha(['proof',i]);frozen='frozen/'+str(i)
        commitment=signed(dict(version='small-commitment-pairs-v2',epoch=m['epoch'],miner=miner,checkpoint='a'*64,source='b'*64,
            batches=[dict(slot=0,env_id='math',index=task,sha256=proof,batch_sha256=sha(batch),training_sha256=digest,training_size=len(raw))]),key)
        receipts[miner]=dict(commitment_document=commitment,training_documents=[dict(slot=0,sha256=digest,size=len(raw),frozen_key=frozen,captured_at=now-10)])
        bucket.put(frozen,raw)
        bucket.put('public/'+m['epoch']+'/submissions/'+miner+'/'+sha(commitment)+'/training/0.json',raw)
    controller=SimpleNamespace(state=root,bucket=bucket,gateway=SimpleNamespace(capture_learner=lambda epoch:receipts),authority=SimpleNamespace(id=AUTH),signed=signed)
    return controller,m,receipts

def grades(context,invalid=()):
    decisions=[];rows=[]
    for o in context['payload']['submissions']:
        ids=[]
        for j in range(4):
            ident=sha([o['sha256'],j]);ids.append(ident);ok=o['sha256'] not in invalid
            gs=[dict(claim='positive',native_score=1 if ok else 0,label_matches=ok,terminal_framing_valid=True),
                dict(claim='negative',native_score=0,label_matches=True,terminal_framing_valid=True)]
            rows.append(dict(pair_sha256=ident,status='accepted_native_labels' if ok else 'excluded_label_mismatch',grades=gs))
        decisions.append(dict(document_sha256=o['sha256'],learner_admission_sha256=sha(o['learner_admission']),pair_sha256=ids,accepted=o['sha256'] not in invalid))
    return dict(version=MULTI_VERSION,context_sha256=sha(context),rows=rows,document_decisions=decisions,
        terminal_rule='max-or-eos-v1',sampling_assurance='unaudited',proof_verification_performed=False,claims_rewritten=False,cheating_penalties=False)

class Integration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.controller,self.manifest,self.receipts=fixture(self.root)
        self.auth=signed(dict(version=AUTHORIZATION_VERSION,source_sha256='b'*64,source_root=str(HERE),
            source_files={'subnet/native_math_prompt.py':'c'*64},sampling_assurance='unaudited',no_credit=True,no_relabel=True,
            limits=dict(version=MULTI_VERSION,terminal_rule='max-or-eos-v1')))
        self.selector=NativeEligibilitySelector(self.controller,self.auth,'/unused',sys.executable)

    def collect(self):return c.collect(self.controller,self.manifest)
    def select(self,invalid_first=False):
        manifest,objects,pop=self.collect();calls=[]
        def get(url,timeout):
            import io
            return io.BytesIO(self.controller.bucket.get(url.removeprefix('memory:')))
        def grade(selector,root,paths,context,expires,fd):
            # Actual unchanged raw validator runs on every staged input again.
            for path,obj in zip(paths,context['payload']['submissions']):
                c.admitted_submission(path,obj,self.manifest,AUTH)
            bad=[o['sha256'] for o in context['payload']['submissions'] if o['learner_admission']['payload']['original_commitment']['payload']['batches'][0]['index']==1] if invalid_first and not calls else []
            calls.append(context);return grades(context,bad)
        with patch.object(ns.urllib.request,'urlopen',side_effect=get),patch.object(ns,'grade_wave',side_effect=grade):
            out=self.selector.select(manifest,objects)
        return out,calls

    def test_real_structural_collector_preserves_reward_singletons_and_rejects_bad_raw(self):
        self.controller,self.manifest,self.receipts=fixture(self.root,tasks=(1,1,2,3),bad=(3,))
        m,objs,pop=self.collect();pool=json.loads(r.collection_paths(self.root,m['epoch'])[0].read_bytes())
        self.assertEqual(pop['eligible_count'],1);self.assertEqual(len(pool['payload']['submissions']),3)
        self.assertEqual(len(pop['exclusions']),3);self.assertEqual(len(objs),1)
        self.assertEqual(r.admit_pool(pool,AUTH)[-1],objs)

    def test_end_to_end_fallback_final_preparation_and_job_admission(self):
        m,objects,pop=self.collect();originals={p:p.read_bytes() for p in self.root.glob('*learner-*.json')}
        (native,chosen),calls=self.select(invalid_first=True)
        self.assertEqual(len(calls),2);self.assertEqual(len(chosen),2)
        docs=ns.read_documents(self.root/'native-outcome-eligibility'/m['epoch'])
        from subnet.unaudited_training_execution import preparation_scope
        scope=preparation_scope(signed(self.manifest),signed(native),chosen,docs,AUTH)
        self.assertEqual(scope['input_inventory_sha256'],sha(c.receipt_inventory(chosen)))
        job=dict(role='train',training_input_policy=c.VERSION,training_policy=m['training_policy'],submissions=chosen,
            source_files={'subnet/committed_training_inputs.py':'e'*64})
        c.validate_job(job,native,AUTH)
        for p,b in originals.items():self.assertEqual(p.read_bytes(),b)
        self.assertEqual(computation_binding(self.manifest),computation_binding(native))

    def test_restart_no_capture_no_grading_no_seed_renewal(self):
        expected,_=self.select();self.controller.gateway.capture_learner=lambda epoch:self.fail('recapture')
        with patch.object(ns,'grade_wave',side_effect=AssertionError('regrade')),patch.object(r.secrets,'token_hex',side_effect=AssertionError('reroll')):
            manifest,objects,_=self.collect();self.assertEqual(self.selector.select(manifest,objects),expected)

    def test_all_collision_pool_is_trainable_despite_empty_reward_inputs(self):
        self.controller,self.manifest,self.receipts=fixture(self.root,tasks=(1,1))
        self.selector.controller=self.controller
        self.assertEqual(self.collect()[1],[])
        (native,chosen),_=self.select();self.assertEqual(len(chosen),1)

    def test_missing_policy_retains_original_collection_and_selector(self):
        self.controller,self.manifest,self.receipts=fixture(self.root,representatives=False)
        m,objs,pop=self.collect();self.assertEqual(len(objs),1)
        self.assertFalse(r.collection_paths(self.root,m['epoch'])[0].exists())

    def test_signed_aggregate_tamper_and_omitted_wave_rejected(self):
        (native,chosen),_=self.select(invalid_first=True)
        docs=ns.read_documents(self.root/'native-outcome-eligibility'/native['epoch'])
        altered=copy.deepcopy(docs);altered['waves'].pop()
        with self.assertRaises(ValueError):r.derivation_receipt(altered,AUTH)
        altered=copy.deepcopy(docs);altered['result']['payload']['accepted_count']+=1
        altered['result']=signed(altered['result']['payload'])
        with self.assertRaises(ValueError):r.derivation_receipt(altered,AUTH)

    def test_transport_failure_preserves_fixed_unfinished_wave(self):
        m,objects,_=self.collect()
        with patch.object(ns.urllib.request,'urlopen',side_effect=ConnectionError('synthetic transport')):
            with self.assertRaises(ConnectionError):self.selector.select(m,objects)
        root=self.root/'native-outcome-eligibility'/m['epoch'];before=(root/'draw.json').read_bytes()
        self.assertFalse((root/'result.ROOT-SIGNED.json').exists())
        self.select();self.assertEqual((root/'draw.json').read_bytes(),before)

    def test_issued_without_final_receipt_refused(self):
        m,objects,_=self.collect();(self.root/'roles').mkdir();(self.root/'roles'/(m['epoch']+'-train.json')).write_text('{}')
        with self.assertRaisesRegex(ValueError,'issued job'):self.selector.select(m,objects)

    def test_transaction_recovery_after_pool_before_population(self):
        m,objects,pop=self.collect();(self.root/(m['epoch']+'-learner-population.json')).unlink()
        self.controller.gateway.capture_learner=lambda epoch:self.fail('recapture')
        self.assertEqual(self.collect(),(m,objects,pop))

    def test_real_owned_inputs_retire_only_after_completed_full_R2_readbacks(self):
        (m,chosen),_=self.select(invalid_first=True)
        from ops.native_task_representative_lifecycle import retire_completed
        self.assertEqual(retire_completed(self.controller,m['epoch'])['status'],'deferred_no_signed_completion')
        _create(self.root/(m['epoch']+'-signed-learner-completion.json'),signed(dict(epoch=m['epoch'],round=900,input_assurance='unaudited')))
        _create(self.root/'controller.json',dict(round=901,active=None))
        self.assertEqual(retire_completed(self.controller,m['epoch'],max_documents=1)['status'],'full_readback_in_progress')
        result=retire_completed(self.controller,m['epoch']);self.assertGreater(result['retired_bytes'],0)
        self.assertEqual(retire_completed(self.controller,m['epoch'])['retired_bytes'],0)
        self.assertTrue((self.root/(m['epoch']+'-learner-population.json')).exists())
        self.assertTrue(self.controller.bucket.objects['frozen/0'])


    def test_capture_budget_refuses_before_any_raw_GET(self):
        self.manifest[r.FIELD]['max_total_input_bytes']=1
        with patch.object(self.controller.bucket,'get_bounded',side_effect=AssertionError('raw read')):
            with self.assertRaisesRegex(ValueError,'before downloads'):self.collect()

    def test_budget_expiry_trains_only_verified_prefix_without_invalidating_ungraded(self):
        self.manifest[r.FIELD]['max_native_documents_per_wave']=1
        m,objects,_=self.collect();clock=[time.time()]
        def prepare(selector,root,objects,expires):return []
        def grade(selector,root,paths,context,expires,fd):
            result=grades(context);clock[0]+=601;return result
        with patch.object(ns.time,'time',side_effect=lambda:clock[0]),patch.object(ns,'prepare_documents',side_effect=prepare),patch.object(ns,'grade_wave',side_effect=grade):
            native,chosen=self.selector.select(m,objects)
        self.assertEqual(len(chosen),1)
        docs=ns.read_documents(self.root/'native-outcome-eligibility'/m['epoch'])
        result=docs['result']['payload'];self.assertEqual(result['completion_reason'],'native_budget_exhausted')
        self.assertEqual(result['checked_count'],1);self.assertFalse(result['complete'])
        r.derivation_receipt(docs,AUTH)

    def test_deadline_before_first_grade_parent_preserving_no_update(self):
        m,objects,_=self.collect();clock=[time.time()]
        def prepare(*args):clock[0]+=601;raise TimeoutError('fixed deadline')
        with patch.object(ns.time,'time',side_effect=lambda:clock[0]),patch.object(ns,'prepare_documents',side_effect=prepare):
            with self.assertRaises(NativeNoUpdate) as raised:self.selector.select(m,objects)
        pointer=m['trainer_state_binding']['parent'];_create(self.root/'latest-trainer-state.json',pointer)
        status=dict(trainer_state=pointer,checkpoint=m['checkpoint'],persistent_state_committed=True,training_steps=21,checkpoint_path='/unchanged')
        from ops.native_training_lifecycle import close_no_update
        before=(self.root/'latest-trainer-state.json').read_bytes()
        checkpoint,metrics=close_no_update(self.controller,raised.exception,m,status)
        self.assertEqual(metrics['steps'],0);self.assertEqual(checkpoint,m['checkpoint'])
        self.assertEqual((self.root/'latest-trainer-state.json').read_bytes(),before)


    def terminal(self,m):
        _create(self.root/(m['epoch']+'-signed-learner-completion.json'),signed(dict(epoch=m['epoch'],round=900,input_assurance='unaudited')))
        _create(self.root/'controller.json',dict(round=901,active=None))

    def test_missing_R2_original_never_deletes_owned_input(self):
        (m,_),_=self.select();self.terminal(m)
        from ops.native_task_representative_lifecycle import retire_completed
        root=self.root/'native-outcome-eligibility'/m['epoch']
        files={p:p.read_bytes() for p in root.glob('waves/*/document-*.owned/document.json')}
        self.assertTrue(files)
        for key in list(self.controller.bucket.objects):
            if key.startswith('public/') and '/submissions/' in key:del self.controller.bucket.objects[key]
        with self.assertRaises(KeyError):retire_completed(self.controller,m['epoch'])
        self.assertEqual(files,{p:p.read_bytes() for p in files})

    def test_replaced_owned_input_never_deleted_after_archive(self):
        (m,_),_=self.select();self.terminal(m)
        from ops.native_task_representative_lifecycle import retire_completed
        root=self.root/'native-outcome-eligibility'/m['epoch']
        target=next(root.glob('waves/*/document-*.owned/document.json'))
        raw=target.read_bytes();target.unlink();target.write_bytes(raw)
        with self.assertRaises(ValueError):retire_completed(self.controller,m['epoch'])
        self.assertEqual(target.read_bytes(),raw)

    def test_empty_interrupted_next_wave_does_not_block_retirement(self):
        (m,_),_=self.select();self.terminal(m)
        from ops.native_task_representative_lifecycle import retire_completed
        root=self.root/'native-outcome-eligibility'/m['epoch']/'waves'
        (root/str(len(list(root.iterdir()))).zfill(4)).mkdir()
        self.assertGreater(retire_completed(self.controller,m['epoch'])['retired_bytes'],0)

    def test_native_manifest_records_full_pool_blacklist_without_rewriting_singleton_snapshot(self):
        blocked=next(iter(self.manifest['capabilities']));cutoff=int(self.manifest['start'])//3600*3600
        audit=dict(version='continuous-probabilistic-audit-v3',recent_epochs=6,decay=.9,prior_alpha=1.,prior_beta=1.,invalid_multiplier=.2,zero_epoch_after=2,blacklist_after=2,blacklist_epochs=4)
        assessment=signed(dict(version='hourly-current-miner-assessment-v1',assessment_stale=False,cutoff=cutoff,evidence_cutoff=cutoff,writer_policy_sha256='f'*64,miner_estimates={blocked:dict(blacklisted=True,confirmed_invalid_recent=2,latest_bad_round=899,current_estimate_round=900,unresolved_is_fraud=False,infrastructure_counted_in_coverage=False)}))
        self.manifest['learner_blacklist_selection_policy']=signed(dict(version='confirmed-blacklist-training-selection-v1',checkpoint='a'*64,source_sha256='b'*64,target_round=900,maximum_age_seconds=7200,assessment_document=assessment,writer_policy_sha256='f'*64,audit_policy=audit))
        self.manifest['learner_blacklist_selection_round']=900
        register=lambda manifest,receipts,round_number,at,authority,eligible_pairs:dict(manifest_document=manifest,receipts=receipts,round=round_number,at=at,eligible_evidence_ids=eligible_pairs)
        with patch.dict(sys.modules,{'subnet.continuous_audit_service':SimpleNamespace(register_population=register)}):
            c.collect(self.controller,self.manifest,round_number=900)
        old=(self.root/(self.manifest['epoch']+'-learner-population.json')).read_bytes()
        (m,chosen),_=self.select()
        self.assertNotIn(blocked,[o['learner_admission']['payload']['miner_identity'] for o in chosen])
        self.assertEqual(m['learner_blacklist_selection_snapshot']['structural_eligible_count'],3)
        self.assertEqual(json.loads(old)['manifest']['learner_blacklist_selection_snapshot']['structural_eligible_count'],1)
        self.assertEqual((self.root/(m['epoch']+'-learner-population.json')).read_bytes(),old)
        documents=ns.read_documents(self.root/'native-outcome-eligibility'/m['epoch'])
        from subnet.unaudited_training_execution import preparation_scope
        preparation_scope(signed(self.manifest),signed(m),chosen,documents,AUTH)
        bad=copy.deepcopy(m);bad['learner_blacklist_selection_snapshot']=json.loads(old)['manifest']['learner_blacklist_selection_snapshot']
        with self.assertRaisesRegex(ValueError,'full-pool blacklist'):preparation_scope(signed(self.manifest),signed(bad),chosen,documents,AUTH)


if __name__=='__main__':unittest.main(verbosity=2)
