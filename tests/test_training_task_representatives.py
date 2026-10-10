"""Real-signature planner controls; native grades are explicit synthetic fixtures.

Uses the exact installed predecessor admission and bind_subset implementations.
No model, native grader subprocess, production key or network is used.
"""
import copy
import base64
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subnet import training_task_representatives as r
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.training_receipts import sha
from subnet.committed_training_inputs import receipt_inventory
from ops.native_training_outcome_filter import (AUTHORIZATION_VERSION, CONTEXT_VERSION,
                                               MULTI_VERSION)

def key(label): return SigningKey(hashlib.sha256(label.encode()).digest())
AUTH_KEY = key('representative-test-authority')
AUTH = AUTH_KEY.verify_key.encode().hex()

def signed(value, signer=AUTH_KEY):
    return dict(payload=value, signer=signer.verify_key.encode().hex(),
                signature=base64.b64encode(signer.sign(canonical(value)).signature).decode())

def fixture(tasks, miner_groups=None):
    miners = miner_groups or [i//9 for i in range(len(tasks))]
    keys = {i: key('miner-'+str(i)) for i in set(miners)}
    identities = {i:k.verify_key.encode().hex() for i,k in keys.items()}
    manifest = dict(epoch='fixture--1000-900', deadline=2000, start=1000,
        checkpoint=dict(id='a'*64), source_bundle=dict(sha256='b'*64), max_batches=9,
        K=4, L=4, capabilities={v:'fixture-private-capability' for v in identities.values()},
        training_input_policy='committed-unaudited-training-v1',
        training_policy='bf16-cpu-fp32-master-task-normalized-persistent-v4',
        training_task_capacity=dict(version='signed-training-task-capacity-v1', max_tasks=512),
        trainer_state_binding=dict(global_step_before=20,fixture=True),
        **{r.FIELD:dict(version=r.VERSION,max_candidate_documents=2304,
            max_total_input_bytes=4_608_000_000,max_native_documents_per_wave=256,max_native_wall_seconds=600,
            exhausted_task_rule='advance-fixed-task-order')})
    children={i:[] for i in keys}; positions=[]
    for number,(task,miner) in enumerate(zip(tasks,miners)):
        slot=len(children[miner]); assert slot<9
        child=dict(slot=slot,env_id='math',index=task,sha256=sha(['proof',number]),
            batch_sha256=sha(['batch',number]),training_sha256=sha(['document',number]),
            training_size=1000)
        children[miner].append(child); positions.append((miner,child))
    commitments={i:signed(dict(version='small-commitment-pairs-v2',epoch=manifest['epoch'],
        checkpoint=manifest['checkpoint']['id'],source=manifest['source_bundle']['sha256'],
        miner=identities[i],batches=rows),keys[i]) for i,rows in children.items()}
    objects=[]
    for miner,child in positions:
        admission=dict(version='committed-unaudited-training-v1',epoch=manifest['epoch'],
            checkpoint=manifest['checkpoint']['id'],source_sha256=manifest['source_bundle']['sha256'],
            miner_identity=identities[miner],slot=child['slot'],original_commitment=commitments[miner],
            commitment_sha256=sha(commitments[miner]),proof_sha256=child['sha256'],
            batch_sha256=child['batch_sha256'],document_sha256=child['training_sha256'],
            document_size=child['training_size'],captured_at=2001,assurance='unaudited')
        objects.append(dict(sha256=child['training_sha256'],size=1000,
                            url='https://unused.invalid/'+child['training_sha256'],
                            learner_admission=signed(admission)))
    pool=signed(dict(version=r.POOL_VERSION,original_signed_manifest=signed(manifest),
        submissions=objects,capture_receipts_sha256='c'*64,original_population_file_sha256='d'*64,
        structural_inventory_sha256=sha(receipt_inventory(objects)),sampling_assurance='unaudited',
        proof_verification_performed=False))
    auth=signed(dict(version=AUTHORIZATION_VERSION,source_sha256='b'*64,
        source_files={'ops/native_training_outcome_filter.py':'e'*64},
        sampling_assurance='unaudited',no_credit=True,no_relabel=True,
        limits=dict(version=MULTI_VERSION,terminal_rule='max-or-eos-v1')))
    return pool,auth

def make_wave(pool,draw,auth,objects,invalid=(),unresolved=()):
    m=pool['payload']['original_signed_manifest']['payload']
    context=signed(dict(version=CONTEXT_VERSION,
        original_signed_manifest=pool['payload']['original_signed_manifest'],submissions=objects,
        source_files=auth['payload']['source_files'],authorization_sha256=sha(auth),
        original_population_file_sha256=sha(pool),original_selection_file_sha256=sha(draw),
        parent_binding_sha256=sha(m['trainer_state_binding'])))
    rows=[]; decisions=[]
    for obj in objects:
        pair_ids=[]
        for i in range(4):
            pair=sha([obj['sha256'],i]);pair_ids.append(pair)
            bad=obj['sha256'] in invalid and i==3
            unknown=obj['sha256'] in unresolved and i==3
            positive=dict(claim='positive',native_score=None if unknown else 0 if bad else 1,
                label_matches=None if unknown else not bad,terminal_framing_valid=True)
            negative=dict(claim='negative',native_score=0,label_matches=True,terminal_framing_valid=True)
            status='excluded_indeterminate' if unknown else 'excluded_label_mismatch' if bad else 'accepted_native_labels'
            rows.append(dict(pair_sha256=pair,status=status,grades=[positive,negative]))
        decisions.append(dict(document_sha256=obj['sha256'],
            learner_admission_sha256=sha(obj['learner_admission']),
            batch_sha256=obj['learner_admission']['payload']['batch_sha256'],pair_sha256=pair_ids,
            accepted=obj['sha256'] not in invalid and obj['sha256'] not in unresolved))
    grades=signed(dict(version=MULTI_VERSION,context_sha256=sha(context),sampling_assurance='unaudited',
        proof_verification_performed=False,claims_rewritten=False,cheating_penalties=False,
        terminal_rule='max-or-eos-v1',rows=rows,document_decisions=decisions))
    return dict(context=context,grades=grades)

class RepresentativeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'draw.json'

    def draw(self,pool):
        with patch.object(r.secrets,'token_hex',return_value='f'*64):
            return r.freeze_draw(pool,AUTH,self.path,now=2001)

    def test_absent_policy_stays_explicitly_historical(self):
        self.assertIsNone(r._policy({}))
        p,a=fixture([1]);del p['payload']['original_signed_manifest']['payload'][r.FIELD]
        p['payload']['original_signed_manifest']=signed(p['payload']['original_signed_manifest']['payload'])
        with self.assertRaisesRegex(ValueError,'historical'):r.admit_pool(signed(p['payload']),AUTH)

    def test_reward_unique_only_byte_exact_and_audit_pool_untouched(self):
        p,a=fixture([4,2,4,3],[0,1,2,3]);before=canonical(p);d=self.draw(p)
        out=r.replay(p,AUTH,d,a,[])
        expected=receipt_inventory([p['payload']['submissions'][i] for i in (1,3)])
        self.assertEqual(canonical(out['reward_eligible_inventory']),canonical(expected))
        self.assertEqual(before,canonical(p));self.assertEqual(len(out['next_submissions']),3)
        w=make_wave(p,d,a,out['next_submissions']);final=r.replay(p,AUTH,d,a,[w])
        self.assertEqual(final['accepted_count'],3)
        self.assertEqual(canonical(final['reward_eligible_inventory']),canonical(expected))
        self.assertEqual(before,canonical(p))

    def test_invalid_first_has_valid_peer_fallback(self):
        p,a=fixture([7,7],[0,1]);d=self.draw(p);s=r.replay(p,AUTH,d,a,[])
        first=s['next_submissions'];w1=make_wave(p,d,a,first,invalid=[first[0]['sha256']])
        middle=r.replay(p,AUTH,d,a,[w1]);self.assertEqual(middle['accepted_count'],0)
        self.assertNotEqual(middle['next_submissions'][0]['sha256'],first[0]['sha256'])
        w2=make_wave(p,d,a,middle['next_submissions']);final=r.replay(p,AUTH,d,a,[w1,w2])
        self.assertTrue(final['complete']);self.assertEqual(final['accepted_count'],1)
        self.assertEqual(final['checked_count'],2);self.assertEqual(final['reward_eligible_inventory'],[])

    def test_all_invalid_refills_fixed_distinct_task_order(self):
        p,a=fixture([1,1,2,3],[0,1,2,3]);m=p['payload']['original_signed_manifest']['payload']
        m[r.FIELD]['max_native_documents_per_wave']=1
        p['payload']['original_signed_manifest']=signed(m);p=signed(p['payload']);d=self.draw(p)
        waves=[];seen=[]
        while True:
            s=r.replay(p,AUTH,d,a,waves)
            if s['complete']:break
            objects=s['next_submissions'];seen.extend(o['sha256'] for o in objects)
            waves.append(make_wave(p,d,a,objects,invalid=[o['sha256'] for o in objects]))
        self.assertEqual(len(set(seen)),4);self.assertEqual(s['disposition'],'no_update')
        self.assertEqual(s['accepted_count'],0)

    def test_unresolved_not_counted_valid_and_peer_still_checked(self):
        p,a=fixture([3,3],[0,1]);d=self.draw(p);first=r.replay(p,AUTH,d,a,[])['next_submissions']
        w=make_wave(p,d,a,first,unresolved=[first[0]['sha256']])
        out=r.replay(p,AUTH,d,a,[w]);self.assertEqual(out['accepted_count'],0)
        self.assertEqual(len(out['next_submissions']),1)

    def test_missing_native_evidence_does_not_advance_or_convert_error(self):
        p,a=fixture([3,3],[0,1]);d=self.draw(p)
        self.assertEqual(r.replay(p,AUTH,d,a,[]),r.replay(p,AUTH,d,a,[]))
        with self.assertRaisesRegex(ValueError,'exact original native wave'):
            r.replay(p,AUTH,d,a,[dict(error='network timeout')])

    def test_crash_replay_reuses_seed_context_and_completed_waves(self):
        p,a=fixture([1,1],[0,1]);d=self.draw(p);before=self.path.read_bytes()
        with patch.object(r.secrets,'token_hex',side_effect=AssertionError('reroll')):
            self.assertEqual(d,r.freeze_draw(p,AUTH,self.path,now=4000))
        self.assertEqual(before,self.path.read_bytes())
        w=make_wave(p,d,a,r.replay(p,AUTH,d,a,[])['next_submissions'])
        x=r.replay(p,AUTH,d,a,[w]);self.assertEqual(x,r.replay(p,AUTH,d,a,[w]))

    def test_changed_inventory_checkpoint_policy_reject_original_draw(self):
        p,a=fixture([1,2]);d=self.draw(p)
        for field,value in [('capture_receipts_sha256','1'*64),('original_population_file_sha256','2'*64)]:
            with self.subTest(field=field):
                q=copy.deepcopy(p['payload']);q[field]=value
                with self.assertRaisesRegex(ValueError,'immutable'):r.validate_draw(signed(q),AUTH,d)
        other=copy.deepcopy(d);other['policy_sha256']='1'*64
        with self.assertRaisesRegex(ValueError,'immutable'):r.replay(p,AUTH,other,a,[])

    def test_real_signature_tampering_refused(self):
        p,a=fixture([1]);d=self.draw(p)
        q=copy.deepcopy(p);q['signature']='00'*64
        with self.assertRaises(ValueError):r.replay(q,AUTH,d,a,[])
        w=make_wave(p,d,a,r.replay(p,AUTH,d,a,[])['next_submissions'])
        w['grades']['signature']='00'*64
        with self.assertRaises(ValueError):r.replay(p,AUTH,d,a,[w])

    def test_native_wrong_checkpoint_or_other_candidate_cannot_be_inserted(self):
        p,a=fixture([1,1],[0,1]);d=self.draw(p);s=r.replay(p,AUTH,d,a,[])['next_submissions']
        other=[o for o in p['payload']['submissions'] if o['sha256']!=s[0]['sha256']]
        with self.assertRaisesRegex(ValueError,'exact next'):r.replay(p,AUTH,d,a,[make_wave(p,d,a,other)])
        w=make_wave(p,d,a,s);w['context']['payload']['parent_binding_sha256']='0'*64
        w['context']=signed(w['context']['payload'])
        with self.assertRaisesRegex(ValueError,'exact next'):r.replay(p,AUTH,d,a,[w])

    def test_partial_document_and_forged_acceptance_rejected(self):
        p,a=fixture([1]);d=self.draw(p);s=r.replay(p,AUTH,d,a,[])['next_submissions']
        w=make_wave(p,d,a,s);w['grades']['payload']['document_decisions'][0]['pair_sha256'].pop()
        w['grades']=signed(w['grades']['payload'])
        with self.assertRaisesRegex(ValueError,'complete document'):r.replay(p,AUTH,d,a,[w])
        w=make_wave(p,d,a,s,invalid=[s[0]['sha256']]);w['grades']['payload']['document_decisions'][0]['accepted']=True
        w['grades']=signed(w['grades']['payload'])
        with self.assertRaisesRegex(ValueError,'completeness verdict'):r.replay(p,AUTH,d,a,[w])

    def test_task_rank_independent_of_multiplicity(self):
        p,a=fixture([4,2,3],[0,1,2]);d=self.draw(p);order=r.validate_draw(p,AUTH,d)[4]
        self.path.unlink();q,b=fixture([4,2,3,4,4],[0,1,2,3,4]);e=self.draw(q)
        self.assertEqual(order,r.validate_draw(q,AUTH,e)[4])

    def test_same_miner_multiple_docs_not_multiple_miner_rank_tickets(self):
        p,a=fixture([2,2,2],[0,1,0]);d=self.draw(p);groups=r.validate_draw(p,AUTH,d)[5]
        rows=groups[('math',2)];runs=[rows[0]['miner']]
        for row in rows[1:]:
            if row['miner']!=runs[-1]:runs.append(row['miner'])
        self.assertEqual(len(runs),2)

    def test_original_blacklist_filters_training_but_not_reward_or_audit(self):
        p,a=fixture([1,1,2],[0,1,2]);m=p['payload']['original_signed_manifest']['payload']
        blocked=p['payload']['submissions'][0]['learner_admission']['payload']['miner_identity']
        audit=dict(version='continuous-probabilistic-audit-v3',recent_epochs=6,decay=.9,
            prior_alpha=1.,prior_beta=1.,invalid_multiplier=.2,zero_epoch_after=2,
            blacklist_after=2,blacklist_epochs=4)
        assessment=signed(dict(version='hourly-current-miner-assessment-v1',assessment_stale=False,
            cutoff=0,evidence_cutoff=0,writer_policy_sha256='f'*64,
            miner_estimates={blocked:dict(blacklisted=True,confirmed_invalid_recent=2,
                latest_bad_round=899,current_estimate_round=900,unresolved_is_fraud=False,
                infrastructure_counted_in_coverage=False)}))
        m['learner_blacklist_selection_policy']=signed(dict(version='confirmed-blacklist-training-selection-v1',
            checkpoint='a'*64,source_sha256='b'*64,target_round=900,maximum_age_seconds=7200,
            assessment_document=assessment,writer_policy_sha256='f'*64,audit_policy=audit))
        m['learner_blacklist_selection_round']=900;p['payload']['original_signed_manifest']=signed(m)
        p=signed(p['payload']);before=canonical(p);d=self.draw(p);out=r.replay(p,AUTH,d,a,[])
        self.assertEqual(len(out['next_submissions']),2)
        self.assertNotIn(blocked,[o['learner_admission']['payload']['miner_identity'] for o in out['next_submissions']])
        self.assertEqual(out['reward_eligible_inventory'],receipt_inventory([p['payload']['submissions'][2]]))
        self.assertEqual(before,canonical(p))

    def test_register_population_eligible_pairs_argument_is_byte_exact(self):
        p,a=fixture([1,1,2,3],[0,1,2,3]);_,_,objects,rows,reward=r.admit_pool(p,AUTH)
        from collections import Counter
        counts=Counter(tuple(row['task']) for row in rows)
        original=[obj for obj,row in zip(objects,rows) if counts[tuple(row['task'])]==1]
        def arguments(chosen):
            return [dict(miner=o['learner_admission']['payload']['miner_identity'],
                commitment_sha256=o['learner_admission']['payload']['commitment_sha256'],
                batch_sha256=o['learner_admission']['payload']['batch_sha256'],
                proof_sha256=o['learner_admission']['payload']['proof_sha256']) for o in chosen]
        self.assertEqual(canonical(arguments(original)),canonical(arguments(reward)))

    def test_cap512_and_no_extra_wave_after_completion(self):
        p,a=fixture(list(range(513)));d=self.draw(p);s=r.replay(p,AUTH,d,a,[])
        self.assertEqual(len(s['next_submissions']),256)
        w=make_wave(p,d,a,s['next_submissions']);middle=r.replay(p,AUTH,d,a,[w])
        w2=make_wave(p,d,a,middle['next_submissions']);out=r.replay(p,AUTH,d,a,[w,w2])
        self.assertTrue(out['complete']);self.assertEqual(out['accepted_count'],512)
        self.assertEqual(out['checked_count'],512)
        with self.assertRaisesRegex(ValueError,'extra native waves'):r.replay(p,AUTH,d,a,[w,w2,w])

    def test_final_gradient_order_remains_original_population_order(self):
        p,a=fixture([1,2,3,4]);d=self.draw(p);s=r.replay(p,AUTH,d,a,[])
        out=r.replay(p,AUTH,d,a,[make_wave(p,d,a,s['next_submissions'])])
        self.assertEqual(out['accepted_submissions'],p['payload']['submissions'])

    def test_budget_checked_before_seed_or_grading(self):
        p,a=fixture([1,2]);m=p['payload']['original_signed_manifest']['payload']
        m[r.FIELD]['max_total_input_bytes']=1999;p['payload']['original_signed_manifest']=signed(m)
        with self.assertRaisesRegex(ValueError,'bounded original structural'):self.draw(signed(p['payload']))
        self.assertFalse(self.path.exists())

    def test_duplicate_slots_and_unregistered_identity_refused(self):
        p,a=fixture([1,2]);q=copy.deepcopy(p['payload']);q['submissions'][1]=q['submissions'][0]
        q['structural_inventory_sha256']=sha(receipt_inventory(q['submissions']))
        with self.assertRaisesRegex(ValueError,'unique original registered'):r.admit_pool(signed(q),AUTH)
        q=copy.deepcopy(p['payload']);m=q['original_signed_manifest']['payload'];m['capabilities']={}
        q['original_signed_manifest']=signed(m)
        with self.assertRaisesRegex(ValueError,'unique original registered'):r.admit_pool(signed(q),AUTH)

    def test_prefreeze_and_symlink_journal_refused(self):
        p,a=fixture([1])
        with self.assertRaisesRegex(ValueError,'postfreeze'):r.freeze_draw(p,AUTH,self.path,now=1999)
        self.draw(p);link=self.path.with_name('link');link.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError,'owned native'):r.freeze_draw(p,AUTH,link,now=2002)

if __name__=='__main__':unittest.main(verbosity=2)
