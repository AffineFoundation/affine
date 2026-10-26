import copy
from dataclasses import asdict, replace
import json
import unittest

from .contract import (VERSION,BATCH_VERSION,VerifierAssessment,sha,validate_contract,
    validate_batch,sampling_receipt,uniform,credit_id,audited_verdict,audited_credit_once,
    register_eligible_once,record_audit_once,loads_unique)


def fixture(n=8):
    miner='a'*64
    contract=dict(version=VERSION,epoch='epoch-new',checkpoint='b'*64,runtime_sha256='c'*64,
        randomness='d'*64,group_size=n,min_success=1,min_completed_failure=1,
        outcome_policy='completed-eos-math-unresolved-masked-v1',sampler=dict(
            version='cached-eager-inverse-cdf-v1',temperature=.7,top_p=1.,support_size=4,
            eos_token_id=3,max_output_tokens=3,max_context_tokens=8))
    task=dict(env_id='affine_math',environment_version='future-math-v2',index=12,
              task_hash='e'*64,prompt_sha256=sha([0,1]))
    rows=[]
    for nonce in range(n):
        category='positive' if nonce<n//2 else 'negative'
        rows.append(dict(index=12,sample_index=12,env_id=task['env_id'],environment_version=task['environment_version'],
            task_hash=task['task_hash'],seed=nonce,sampling=sampling_receipt(contract,miner,task,nonce),
            classification=category,reward=1. if category=='positive' else 0.,
            turns=[dict(prompt=[0,1],output=[0 if category=='positive' else 1,3],stop_reason='eos')]))
    batch=dict(schema=BATCH_VERSION,epoch=contract['epoch'],checkpoint=contract['checkpoint'],miner=miner,
        env_id=task['env_id'],environment_version=task['environment_version'],index=12,sample_index=12,
        task_hash=task['task_hash'],group_contract_sha256=sha(contract),rollouts=rows)
    return contract,miner,task,batch


def admit(c,m,t,b):return validate_batch(b,c,authenticated_miner=m,trusted_task=t)


def assessments(admission,batch):
    return [VerifierAssessment(i,admission.rollout_sha256[i],admission.context_sha256,
                'valid','valid',r['classification']) for i,r in enumerate(batch['rollouts'])]


class ContractTests(unittest.TestCase):
    def test_generic_N_and_separate_minima(self):
        for n in (2,4,8,16,100,128):
            c,m,t,b=fixture(n);a=admit(c,m,t,b)
            self.assertTrue(a.claimed_qualifies)
            self.assertFalse(a.sampling_verified)
            self.assertEqual(len(a.rollout_sha256),n)
        c,m,t,b=fixture();c['min_success']=2;c['min_completed_failure']=3
        b['group_contract_sha256']=sha(c)
        for i,r in enumerate(b['rollouts']):r['sampling']=sampling_receipt(c,m,t,i)
        self.assertTrue(admit(c,m,t,b).claimed_qualifies)

    def test_bad_size_and_outcome_minimum_types(self):
        for key,value in [('group_size',True),('group_size',1),('group_size',129),
                          ('min_success',0),('min_success',1.),('min_completed_failure',True),('min_completed_failure',8)]:
            c,_,_,_=fixture();c[key]=value
            with self.assertRaises(ValueError):validate_contract(c)

    def test_historical_wire_not_reinterpreted(self):
        c,m,t,b=fixture();c['version']='forced-inverse-cdf-prefill-miner-bound-v5'
        with self.assertRaisesRegex(ValueError,'historical'):admit(c,m,t,b)
        c,m,t,b=fixture();b['schema']=2
        with self.assertRaisesRegex(ValueError,'schema'):admit(c,m,t,b)

    def test_missing_duplicate_cherrypicked_and_reordered_nonces(self):
        for mutation in ('missing','duplicate','extra','offset','reorder'):
            c,m,t,b=fixture()
            if mutation=='missing':b['rollouts'].pop()
            elif mutation=='duplicate':b['rollouts'][3]=copy.deepcopy(b['rollouts'][2])
            elif mutation=='extra':b['rollouts'][-1]['seed']=8
            elif mutation=='offset':
                for r in b['rollouts']:r['seed']+=1
            else:b['rollouts'].reverse()
            with self.assertRaises(ValueError):admit(c,m,t,b)

    def test_extra_salt_rejected_at_every_wire_level(self):
        for level in ('contract','sampler','batch','rollout','turn','sampling','task'):
            c,m,t,b=fixture()
            obj={'contract':c,'sampler':c['sampler'],'batch':b,'rollout':b['rollouts'][0],
                 'turn':b['rollouts'][0]['turns'][0],'sampling':b['rollouts'][0]['sampling'],'task':t}[level]
            obj['salt']='miner-chosen'
            with self.assertRaises(ValueError):admit(c,m,t,b)

    def test_bool_nonce_cannot_alias_integer(self):
        c,m,t,b=fixture();b['rollouts'][1]['seed']=True
        with self.assertRaises(ValueError):admit(c,m,t,b)
        c,m,t,b=fixture();b['rollouts'][1]['sampling']['attempt']=True
        with self.assertRaises(ValueError):admit(c,m,t,b)

    def test_stale_epoch_checkpoint_and_identity_rejected(self):
        for key,value in [('epoch','old'),('checkpoint','f'*64),('miner','f'*64),
                          ('group_contract_sha256','f'*64),('task_hash','f'*64),('index',13),
                          ('environment_version','old-env')]:
            c,m,t,b=fixture();b[key]=value
            with self.assertRaises(ValueError):admit(c,m,t,b)

    def test_runtime_sampler_task_and_prompt_binding(self):
        for mutation in ('runtime','temperature','randomness','task','prompt'):
            c,m,t,b=fixture()
            if mutation=='runtime':c['runtime_sha256']='f'*64
            elif mutation=='temperature':c['sampler']['temperature']=1.
            elif mutation=='randomness':c['randomness']='f'*64
            elif mutation=='task':t['task_hash']='f'*64
            else:b['rollouts'][0]['turns'][0]['prompt']=[1,0]
            with self.assertRaises(ValueError):admit(c,m,t,b)

    def test_receipt_and_draw_binding(self):
        c,m,t,b=fixture()
        self.assertNotEqual(uniform(c,m,t,0,0),uniform(c,m,t,1,0))
        self.assertNotEqual(uniform(c,m,t,0,0),uniform(c,'f'*64,t,0,0))
        self.assertNotEqual(credit_id(c,m,t),credit_id(c,'f'*64,t))
        for nonce in (-1,8,True,1.):
            with self.assertRaises(ValueError):sampling_receipt(c,m,t,nonce)

    def test_duplicate_output_different_valid_nonce_is_accepted(self):
        c,m,t,b=fixture();self.assertEqual(b['rollouts'][0]['turns'],b['rollouts'][1]['turns'])
        a=admit(c,m,t,b);v=audited_verdict(a,assessments(a,b),c)
        self.assertEqual(v['status'],'valid_qualifying')
        self.assertFalse(v['historical_execution_proven'])

    def test_caps_are_retained_neutral_not_completed_failures(self):
        c,m,t,b=fixture();r=b['rollouts'][-1]
        r.update(classification='neutral',reward=None)
        r['turns'][0].update(output=[0,1,2],stop_reason='budget')
        a=admit(c,m,t,b);self.assertEqual(a.claimed_counts,(4,3,1))
        r.update(classification='negative',reward=0.)
        with self.assertRaisesRegex(ValueError,'budget caps'):admit(c,m,t,b)

    def test_only_unresolved_and_correct_cannot_qualify(self):
        c,m,t,b=fixture()
        for r in b['rollouts'][4:]:r.update(classification='neutral',reward=None)
        a=admit(c,m,t,b)
        self.assertFalse(a.claimed_qualifies)
        self.assertEqual(audited_verdict(a,assessments(a,b),c)['status'],'valid_nonqualifying')

    def test_uniform_outcomes_valid_but_nonpayable(self):
        for outcome in ('positive','negative'):
            c,m,t,b=fixture()
            for r in b['rollouts']:r.update(classification=outcome,reward=1. if outcome=='positive' else 0.)
            a=admit(c,m,t,b);v=audited_verdict(a,assessments(a,b),c)
            self.assertFalse(a.claimed_qualifies)
            self.assertEqual(v['status'],'valid_nonqualifying')
            with self.assertRaises(ValueError):audited_credit_once({},a,v)

    def test_malformed_stops_and_unsupported_tokens(self):
        for output,stop in [([0,1],'eos'),([3,0],'eos'),([0,3,3],'eos'),([0,1],'budget'),([0,4,3],'eos'),([True,3],'eos')]:
            c,m,t,b=fixture();b['rollouts'][0]['turns'][0].update(output=output,stop_reason=stop)
            with self.assertRaises(ValueError):admit(c,m,t,b)

    def test_infra_error_and_multiturn_do_not_become_negative(self):
        c,m,t,b=fixture();b['rollouts'][0]['turns'][0]['stop_reason']='infra_error'
        with self.assertRaises(ValueError):admit(c,m,t,b)
        c,m,t,b=fixture();b['rollouts'][0]['turns']*=2
        with self.assertRaises(ValueError):admit(c,m,t,b)

    def test_invalid_proof_rejects_group(self):
        c,m,t,b=fixture();a=admit(c,m,t,b);rows=assessments(a,b)
        rows[3]=replace(rows[3],sampling='invalid')
        v=audited_verdict(a,rows,c)
        self.assertEqual(v['status'],'invalid');self.assertEqual(v['invalid_nonces'],[3])

    def test_indeterminate_and_unchecked_are_not_verified(self):
        for status in ('indeterminate','unchecked'):
            c,m,t,b=fixture();a=admit(c,m,t,b);rows=assessments(a,b)
            rows[2]=replace(rows[2],sampling=status)
            v=audited_verdict(a,rows,c)
            self.assertEqual(v['status'],'indeterminate');self.assertFalse(v['sampling_checked'])

    def test_invalid_terminal_rejected(self):
        c,m,t,b=fixture();a=admit(c,m,t,b);rows=assessments(a,b)
        rows[0]=replace(rows[0],terminal='invalid')
        self.assertEqual(audited_verdict(a,rows,c)['status'],'invalid')

    def test_swapped_outcomes_same_counts_detected_per_nonce(self):
        c,m,t,b=fixture();a=admit(c,m,t,b);rows=assessments(a,b)
        rows[0]=replace(rows[0],outcome='negative');rows[4]=replace(rows[4],outcome='positive')
        v=audited_verdict(a,rows,c)
        self.assertEqual(v['status'],'invalid')
        self.assertEqual(v['mismatched_outcome_nonces'],[0,4])

    def test_forged_assessment_dict_or_stale_digest_not_accepted(self):
        for kind in ('json','digest','nonce','context'):
            c,m,t,b=fixture();a=admit(c,m,t,b);rows=assessments(a,b)
            if kind=='json':rows[0]=asdict(rows[0])
            elif kind=='digest':rows[0]=replace(rows[0],rollout_sha256='f'*64)
            elif kind=='nonce':rows[0]=replace(rows[0],nonce=1)
            else:rows[0]=replace(rows[0],context_sha256='f'*64)
            with self.assertRaises(ValueError):audited_verdict(a,rows,c)

    def test_audit_cannot_change_original_contract(self):
        c,m,t,b=fixture();a=admit(c,m,t,b);rows=assessments(a,b);c['min_success']=2
        with self.assertRaises(ValueError):audited_verdict(a,rows,c)

    def test_credit_once_idempotent_and_conflicting_payload_rejected(self):
        c,m,t,b=fixture();a=admit(c,m,t,b);v=audited_verdict(a,assessments(a,b),c)
        original={};ledger,credit=audited_credit_once(original,a,v)
        self.assertEqual(original,{})
        self.assertEqual(credit,1)
        ledger2,credit2=audited_credit_once(ledger,a,v)
        self.assertEqual(ledger2,ledger);self.assertEqual(credit2,0)
        b['rollouts'][0]['turns'][0]['output']=[2,3]
        altered=admit(c,m,t,b)
        self.assertEqual(altered.credit_id,a.credit_id)
        altered_v=audited_verdict(altered,assessments(altered,b),c)
        with self.assertRaisesRegex(ValueError,'already credited'):audited_credit_once(ledger,altered,altered_v)

    def test_provisional_registration_does_not_require_any_audit(self):
        c,m,t,b=fixture();a=admit(c,m,t,b)
        ledger,delta=register_eligible_once({},a)
        self.assertEqual(delta,1)
        self.assertEqual(ledger[a.credit_id]['assurance'],'unaudited')
        self.assertFalse(ledger[a.credit_id]['sampling_verified'])
        self.assertFalse(ledger[a.credit_id]['reward_assigned'])
        ledger2,delta2=register_eligible_once(ledger,a)
        self.assertEqual(ledger2,ledger);self.assertEqual(delta2,0)

    def test_audit_evidence_separate_from_eligible_counts_and_all_statuses_retained(self):
        c,m,t,b=fixture();a=admit(c,m,t,b)
        ledger,delta=register_eligible_once({},a);original=copy.deepcopy(ledger)
        evidence={}
        for i,status in enumerate(('valid','invalid','indeterminate','unchecked')):
            rows=assessments(a,b);rows[0]=replace(rows[0],sampling=status)
            verdict=audited_verdict(a,rows,c);eid=sha(['authenticated-report',i])
            evidence,added=record_audit_once(evidence,a,verdict,authenticated_evidence_sha256=eid)
            self.assertEqual(added,1)
            unchanged,repeated=record_audit_once(evidence,a,verdict,authenticated_evidence_sha256=eid)
            self.assertEqual(unchanged,evidence);self.assertEqual(repeated,0)
        self.assertEqual(ledger,original)
        self.assertEqual(len(ledger),1);self.assertEqual(len(evidence),4)
        self.assertEqual({r['verdict']['status'] for r in evidence.values()},
                         {'valid_qualifying','invalid','indeterminate'})

    def test_nonqualifying_claim_cannot_inflate_provisional_counts(self):
        c,m,t,b=fixture()
        for r in b['rollouts']:r.update(classification='positive',reward=1.)
        a=admit(c,m,t,b)
        with self.assertRaisesRegex(ValueError,'eligible claimed'):register_eligible_once({},a)

    def test_changed_bytes_cannot_double_register_and_audit_digest_cannot_change(self):
        c,m,t,b=fixture();a=admit(c,m,t,b);ledger,_=register_eligible_once({},a)
        verdict=audited_verdict(a,assessments(a,b),c);eid='f'*64
        evidence,_=record_audit_once({},a,verdict,authenticated_evidence_sha256=eid)
        changed=dict(verdict,status='invalid')
        with self.assertRaisesRegex(ValueError,'different result'):
            record_audit_once(evidence,a,changed,authenticated_evidence_sha256=eid)
        b['rollouts'][0]['turns'][0]['output']=[2,3];altered=admit(c,m,t,b)
        with self.assertRaisesRegex(ValueError,'different bytes'):register_eligible_once(ledger,altered)

    def test_json_duplicate_keys_nonfinite_and_size_rejected(self):
        for raw in (b'{"seed":0,"seed":1}',b'{"x":NaN}',b'{"x":Infinity}'):
            with self.assertRaises(ValueError):loads_unique(raw)
        with self.assertRaises(ValueError):loads_unique(b'{}',max_bytes=1)
        self.assertEqual(loads_unique(b'{"x":1}'),{'x':1})


if __name__=='__main__':unittest.main()
