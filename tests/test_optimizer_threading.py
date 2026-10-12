"""Optimizer-only thread controls; fake runtime, no model or tensor calls."""
import copy
import unittest
from subnet import optimizer_threading as t

class Runtime:
    def __init__(self, intraop=2, interop=2):
        self.intraop=intraop; self.interop=interop; self.changes=[]
    def get_num_threads(self): return self.intraop
    def get_num_interop_threads(self): return self.interop
    def set_num_threads(self, value):
        self.changes.append(value); self.intraop=value

class OptimizerThreadingTests(unittest.TestCase):
    def test_only_step_uses_sixteen_then_restores(self):
        runtime=Runtime(); gradients=object(); calls=[]
        class Optimizer:
            def step(self, *, gradients):
                calls.append((runtime.intraop,runtime.interop,gradients)); return 'original-result'
        result,evidence=t.step(runtime,Optimizer(),gradients,t.POLICY)
        self.assertEqual(result,'original-result'); self.assertEqual(calls,[(16,2,gradients)])
        self.assertEqual(runtime.changes,[16,2]); self.assertEqual(evidence,t.expected_evidence(t.POLICY))
    def test_exception_restores_threads_and_is_propagated(self):
        runtime=Runtime()
        class Optimizer:
            def step(self, **kwargs): raise RuntimeError('failed original update')
        with self.assertRaisesRegex(RuntimeError,'failed original update'):t.step(runtime,Optimizer(),{},t.POLICY)
        self.assertEqual(runtime.changes,[16,2]); self.assertEqual(runtime.intraop,2)
    def test_wrong_entry_profile_never_steps(self):
        for runtime in (Runtime(4,2),Runtime(2,4)):
            with self.assertRaises(ValueError):t.step(runtime,object(),{},t.POLICY)
            self.assertEqual(runtime.changes,[])
    def test_policy_is_explicit_exact_and_copied(self):
        self.assertIsNone(t.declared_policy({}))
        p=t.configuration(t.POLICY);p['optimizer_intraop']=1;self.assertEqual(t.POLICY['optimizer_intraop'],16)
        for field,value in [('optimizer_intraop',8),('optimizer_intraop',16.0),('interop',True)]:
            p=dict(t.POLICY);p[field]=value
            with self.assertRaises(ValueError):t.configuration(p)
    def test_every_update_needs_exact_restoration_evidence(self):
        good={'training':{'updates':[{'optimizer_threading':t.expected_evidence(t.POLICY)}]}}
        t.validate_report(good,t.POLICY,1)
        for report in ({'training':{'updates':[]}}, {'training':{'updates':[{}]}}, {'training':{'updates':good['training']['updates']*2}}):
            with self.assertRaises(ValueError):t.validate_report(report,t.POLICY,1)
        for field,value in [('intraop',16),('interop',2.0)]:
            report=copy.deepcopy(good);report['training']['updates'][0]['optimizer_threading']['after'][field]=value
            with self.assertRaises(ValueError):t.validate_report(report,t.POLICY,1)
        with self.assertRaises(ValueError):t.validate_report(good,None,1)

if __name__=='__main__':unittest.main()
