"""Prospective CPU sampler controls; no GPU qualification or live contract change."""
import copy
import itertools
import math
import unittest
from unittest.mock import patch

import numpy as np

from subnet import verified_distribution_sampling as s


class VerifiedDistributionSamplingTests(unittest.TestCase):
    def setUp(self):
        self.manifest=dict(epoch='nonpayable-prospective-cdf',checkpoint={'id':'c'*64},
            sampling_contract=s.make_contract('a'*64),sampling_source_hash=s.source_hash())
        self.context=s.context_for_manifest(self.manifest)
        self.env='synthetic-math';self.task='d'*64;self.index=73;self.attempt=5
        self.row=np.log(np.array([.1,.2,.3,.4],np.float64));self.temperature=.8

    def draw(self,pos,**changes):
        args=dict(context=self.context,env_id=self.env,task_hash=self.task,index=self.index,
                  attempt=self.attempt,turn=0,position=pos);args.update(changes)
        return s.uniform(**args)

    def output(self,count=4):
        return[s.select_token(self.row,self.draw(i),self.temperature)for i in range(count)]

    def full(self,output):
        prompt=[0,1]
        values=np.tile(np.append(self.row,-100.),(len(prompt)+len(output),1))
        return dict(prompt_tokens=prompt,output_tokens=output,forward_input_ids=prompt+output,
            full_forward_logprobs=values,temperature=self.temperature,max_output_tokens=len(output),eos_token_id=4)

    def verify(self,args):
        return s.verify_turn(self.context,self.env,self.task,self.index,self.attempt,0,**args)

    def test_prospective_contract_cannot_select_e9_or_a_weakened_numeric_policy(self):
        for mutation in ('version','top_p','logprob_atol','arithmetic_margin','extra','source'):
            changed=copy.deepcopy(self.manifest)
            if mutation=='source':changed['sampling_source_hash']='e'*64
            elif mutation=='extra':changed['sampling_contract']['unchecked_tolerance']=True
            else:changed['sampling_contract'][mutation]={'version':'forced-inverse-cdf-replay-v1',
                'top_p':.9,'logprob_atol':1e-4,'arithmetic_margin':0}[mutation]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):s.context_for_manifest(changed)

    def test_public_draw_changes_with_every_context_attempt_and_position_binding(self):
        base=self.draw(0)
        changed=copy.deepcopy(self.context);changed['epoch']='other'
        variants=[self.draw(0,context=changed),self.draw(0,env_id='other'),self.draw(0,task_hash='e'*64),
            self.draw(0,index=74),self.draw(0,attempt=6),self.draw(0,turn=1),self.draw(1)]
        changed=copy.deepcopy(self.context);changed['checkpoint']='e'*64;variants.append(self.draw(0,context=changed))
        changed=copy.deepcopy(self.context);changed['contract']['randomness']='e'*64;variants.append(self.draw(0,context=changed))
        self.assertTrue(all(value!=base for value in variants))
        self.assertEqual(base*2**53,int(base*2**53))
        for changes in ({'attempt':128},{'attempt':-1},{'attempt':True},{'position':2048},{'turn':32},{'index':-1}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.draw(0,**changes)

    def test_all_token_choices_are_checked_without_inference_toploc_or_environment_calls(self):
        output=self.output()
        forbidden=AssertionError('candidate CDF gate must perform no model/environment calls')
        with patch('subnet.model.Runtime.compute',side_effect=forbidden), \
                patch('subnet.model.Runtime.sample_output',side_effect=forbidden), \
                patch('subnet.model.Runtime.verify',side_effect=forbidden), \
                patch('subnet.model.create_session',side_effect=forbidden), \
                patch('subnet.proofs.verify_mapped_proofs',side_effect=forbidden):
            result=self.verify(self.full(output))
        self.assertTrue(result['valid']);self.assertEqual(result['checked_tokens'],4)
        self.assertTrue(result['complete_output_checked']);self.assertFalse(result['historical_execution_proven'])

    def test_copied_or_synthetic_output_with_genuine_model_probabilities_is_refused(self):
        output=self.output();output[2]=(output[2]+1)%4
        result=self.verify(self.full(output))
        self.assertFalse(result['valid']);self.assertEqual(result['status'],'sampler_mismatch')
        self.assertEqual(result['first_rejected_position'],2)
        self.assertEqual(result['checked_tokens'],3)

    def test_probability_tolerance_alone_allows_cumulative_boundary_choice_flip(self):
        row=np.log(np.array([.5,.5],np.float64));altered=row.copy();altered[0]+=2e-6
        u=.5000002
        self.assertTrue(np.allclose(row,altered,atol=s.LOGPROB_ATOL,rtol=0))
        self.assertNotEqual(s.select_token(row,u,.8),s.select_token(altered,u,.8))
        for token in(0,1):self.assertEqual(s.check_token(row,u,token,.8)['status'],'numerical_ambiguity')

    def test_every_per_logprob_extreme_perturbation_obeys_exact_cdf_bounds_and_unique_robust_pick(self):
        base_weights,total=s.distribution(self.row,self.temperature)
        base=[value/total for value in base_weights];count=0
        for u in(.000001,.03,.12,.25,.46,.67,.98,.999999):
            selected=s.select_token(self.row,u,self.temperature)
            self.assertEqual(s.check_token(self.row,u,selected,self.temperature)['status'],'accepted')
            for signs in itertools.product((-1,1),repeat=4):
                altered=self.row+np.array(signs)*s.LOGPROB_ATOL
                weights,total=s.distribution(altered,self.temperature);distribution=[x/total for x in weights]
                self.assertEqual(s.select_token(altered,u,self.temperature),selected)
                for end in range(1,4):
                    lower,upper=s.cdf_uncertainty(math.fsum(base[:end]),self.temperature)
                    mass=math.fsum(distribution[:end]);self.assertGreaterEqual(mass+1e-15,lower);self.assertLessEqual(mass-1e-15,upper)
                count+=1
        self.assertEqual(count,128)

    def test_tail_temperature_and_large_vocabulary_cannot_enable_arbitrary_teacher_tokens(self):
        for temperature in(.05,.8,4):
            row=np.full(200000,-50.,np.float64);row[27]=0.
            self.assertEqual(s.select_token(row,.5,temperature),27)
            self.assertEqual(s.check_token(row,.5,27,temperature)['status'],'accepted')
            for chosen in(0,100000,199999):
                self.assertEqual(s.check_token(row,.5,chosen,temperature)['status'],'sampler_mismatch')
        with self.assertRaises(ValueError):s.distribution(np.zeros(200001,np.float32),.8)

    def test_shift_constant_normalization_and_fp32_input_leave_distribution_semantics_intact(self):
        u=.123
        self.assertEqual(s.select_token(self.row,u,.8),s.select_token(self.row-100,u,.8))
        self.assertEqual(s.select_token(self.row,u,.8),s.select_token(self.row.astype(np.float32),u,.8))
        for row in(np.array([math.nan,0.]),np.array([math.inf,0.]),np.array([1,2]),np.array([[0.,0.]])):
            with self.subTest(row_type=str(row.dtype)),self.assertRaises(ValueError):s.check_token(row,u,0,.8)
        for temperature in(0,.001,math.nan,True):
            with self.assertRaises(ValueError):s.check_token(self.row,u,0,temperature)
        for u in(-1,1,math.nan,True):
            with self.assertRaises(ValueError):s.check_token(self.row,u,0,.8)

    def test_exact_causal_row_offset_uses_prompt_last_token_to_predict_first_output(self):
        output=self.output(2);args=self.full(output)
        full=args['full_forward_logprobs'].copy();full[0]=np.array([-2.,-3.,-4.,-5.,-100.]);full[-1]=np.array([-6.,-7.,-8.,-9.,-100.])
        args['full_forward_logprobs']=full
        rows=s.causal_rows(full,args['forward_input_ids'],args['prompt_tokens'],output)
        self.assertTrue(np.array_equal(rows,full[1:3]))
        self.assertTrue(self.verify(args)['valid'])

    def test_context_sequence_row_count_dtype_or_token_type_substitution_is_refused(self):
        good=self.full(self.output())
        for mutation in('inputs','row-count','dtype','bool-token','prompt','output-index'):
            args=copy.deepcopy(good)
            if mutation=='inputs':args['forward_input_ids'][0]=3
            elif mutation=='row-count':args['full_forward_logprobs']=args['full_forward_logprobs'][:-1]
            elif mutation=='dtype':args['full_forward_logprobs']=args['full_forward_logprobs'].astype(np.int64)
            elif mutation=='bool-token':args['forward_input_ids'][0]=True;args['prompt_tokens'][0]=True
            elif mutation=='prompt':args['prompt_tokens']=[]
            else:args['output_tokens'][0]=5;args['forward_input_ids'][2]=5
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.verify(args)

    def test_eos_or_budget_is_the_only_stopping_rule(self):
        good=self.full(self.output());self.assertTrue(self.verify(good)['valid'])
        shortened=copy.deepcopy(good);shortened['max_output_tokens']+=1
        with self.assertRaisesRegex(ValueError,'stopping'):self.verify(shortened)
        # A correct one-token EOS is permitted even below the token budget.
        one=self.output(1);args=self.full(one);args.update(eos_token_id=one[0],max_output_tokens=4)
        self.assertTrue(self.verify(args)['valid'])
        bad=self.full([one[0],one[0]]);bad.update(eos_token_id=one[0],max_output_tokens=4)
        with self.assertRaisesRegex(ValueError,'stopping'):self.verify(bad)

    def test_quantization_has_its_own_boundary_and_is_not_a_reproducibility_proof(self):
        left=.0005-1e-6;right=.0005+1e-6
        self.assertLess(abs(left-right),s.LOGPROB_ATOL)
        self.assertNotEqual(round(left/.001),round(right/.001))

    def test_math_uncertainty_is_independent_of_physical_execution_or_success_selection(self):
        bound=math.tanh(s.LOGPROB_ATOL/(2*.8))
        lower,upper=s.cdf_uncertainty(.5,.8)
        self.assertGreater(bound,0);self.assertLess(bound,6.250001e-6)
        self.assertLess(upper-.5,bound+1e-15);self.assertLess(.5-lower,bound+1e-15)
        result=self.verify(self.full(self.output()))
        self.assertFalse(result['historical_execution_proven'])
        self.assertFalse(result['numerical_ambiguity_is_fraud'])


if __name__=='__main__':unittest.main()
