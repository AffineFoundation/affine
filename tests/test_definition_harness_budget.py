"""Budget checks consume fully validated choices without per-task revalidation."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from subnet.service import definitions
from subnet.harness import normalize
import subnet.sample_harness as sample

class DefinitionHarnessBudget(unittest.TestCase):
    def row(self,indices=None,harness=None,budget=128,**extra):
        return dict(spec=dict(id='fixture',adapter='prime_v1',config={},num_samples=10000,max_output_tokens=budget),indices=[0,1] if indices is None else indices,harness=harness,**extra)
    def compile(self,row):
        with patch('subnet.service.EnvironmentSpec.from_dict',side_effect=lambda raw:SimpleNamespace(**raw,to_dict=lambda:raw)):
            return definitions(dict(environments=[row]))[0]
    def plain(self,budget=64):return dict(version='text-tools-long-v2',policy='autoregressive',max_output_tokens=budget)
    def test_large_plain_population_validated_once_without_resolve(self):
        row=self.row(list(range(10000)),self.plain())
        with patch.object(sample,'_indices',wraps=sample._indices) as indices,patch.object(sample,'normalize',wraps=sample.normalize) as normalized,patch.object(sample,'resolve',side_effect=AssertionError('per-task revalidation')):
            result=self.compile(row)
        self.assertEqual(indices.call_count,1);self.assertEqual(normalized.call_count,1)
        self.assertEqual(result['indices'],row['indices']);self.assertEqual(result['harness'],normalize(self.plain()))
    def test_plain_budget_checked_for_every_population_size(self):
        for population in ([0],list(range(10000))):
            with self.assertRaisesRegex(ValueError,'harness exceeds environment budget'):
                self.compile(self.row(population,self.plain(129)))
    def test_indexed_choices_validated_once_and_all_budgets_checked(self):
        population=list(range(10000));harness=dict(version=sample.VERSION,by_index={str(i):self.plain() for i in population})
        with patch.object(sample,'_indices',wraps=sample._indices) as indices,patch.object(sample,'normalize',wraps=sample.normalize) as normalized,patch.object(sample,'resolve',side_effect=AssertionError('per-task revalidation')):
            self.compile(self.row(population,harness))
        self.assertEqual(indices.call_count,1);self.assertEqual(normalized.call_count,len(population))
        harness['by_index']['9999']=self.plain(129)
        with self.assertRaisesRegex(ValueError,'harness exceeds environment budget'):self.compile(self.row(population,harness))
    def test_indexed_exact_coverage_and_authorization_remain_required(self):
        valid=dict(version=sample.VERSION,by_index={'0':self.plain(),'1':self.plain()})
        for mutation in ('missing','extra','nested'):
            harness=copy.deepcopy(valid)
            if mutation=='missing':harness['by_index'].pop('1')
            elif mutation=='extra':harness['by_index']['2']=self.plain()
            else:harness['by_index']['1']=valid
            with self.assertRaises(ValueError):self.compile(self.row(harness=harness))
        for indices in ([0,0],[0,-1],[True],list(range(10001)),[10000]):
            with self.assertRaises(ValueError):self.compile(self.row(indices,self.plain()))
    def test_none_historical_default_budget_retains_none_policy(self):
        self.assertIsNone(self.compile(self.row())['harness'])
        with self.assertRaisesRegex(ValueError,'harness exceeds environment budget'):self.compile(self.row(budget=1))
    def test_legacy_historical_harness_fallback_is_preserved(self):
        row=self.row();row['spec']['adapter']='legacy_mastermind'
        with patch('subnet.service.legacy_harness',return_value=self.plain()):
            self.assertEqual(self.compile(row)['harness'],normalize(self.plain()))
    def test_evaluation_only_plain_budget_and_empty_indexed_coverage(self):
        with self.assertRaisesRegex(ValueError,'harness exceeds environment budget'):
            self.compile(self.row([],self.plain(129),evaluation_only=True))
        value=self.compile(self.row([],dict(version=sample.VERSION,by_index={}),evaluation_only=True))
        self.assertEqual(value['indices'],[]);self.assertTrue(value['evaluation_only'])
        with self.assertRaisesRegex(ValueError,'exact signed mining index coverage'):
            self.compile(self.row([],dict(version=sample.VERSION,by_index={'0':self.plain()}),evaluation_only=True))
        with self.assertRaisesRegex(ValueError,'challenge indices'):self.compile(self.row([],self.plain()))

if __name__=='__main__':unittest.main()
