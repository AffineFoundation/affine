"""Fresh-worker startup controls; portable CPU tests without a model."""
import copy
import unittest
from ops.learner_selection_cpu_peer import initialize_optimizer_threads
from subnet import optimizer_threading as threading

GRANT={'version':'cpu-selection-peer-basic-eligibility-optimizer-cpu16-authorization-v11'}
def job(policy=None):
    return {'payload':{'unaudited_training_execution':{'payload':{
        'optimizer_threading':copy.deepcopy(threading.POLICY if policy is None else policy)}}}}

class Runtime:
    def __init__(self,inter=96):self.intra=96;self.inter=inter;self.calls=[]
    def get_num_threads(self):return self.intra
    def get_num_interop_threads(self):return self.inter
    def set_num_threads(self,n):self.calls.append(('intra',n));self.intra=n
    def set_num_interop_threads(self,n):self.calls.append(('inter',n));self.inter=n

class Tests(unittest.TestCase):
    def test_original_partial_initialization_fails_then_startup_repairs(self):
        runtime=Runtime();runtime.set_num_threads(2)
        with self.assertRaises(ValueError):threading.require_steady(runtime,threading.POLICY)
        receipt=initialize_optimizer_threads(job(),GRANT,torch_module=runtime)
        self.assertEqual(receipt['before'],{'intraop':2,'interop':96})
        self.assertEqual(receipt['after'],{'intraop':2,'interop':2})
        threading.require_steady(runtime,threading.POLICY)
        class Optimizer:
            def step(self,**kwargs):
                self.observed=(runtime.intra,runtime.inter);return 'unchanged-step'
        opt=Optimizer();result,evidence=threading.step(runtime,opt,{},threading.POLICY)
        self.assertEqual(result,'unchanged-step');self.assertEqual(opt.observed,(16,2))
        self.assertEqual(evidence['after'],{'intraop':2,'interop':2})
    def test_legacy_does_not_initialize_or_import_torch(self):
        runtime=Runtime();self.assertIsNone(initialize_optimizer_threads({'payload':{}},{},torch_module=runtime))
        self.assertEqual(runtime.calls,[])
    def test_wrong_policy_or_peer_refused_before_any_runtime_mutation(self):
        cases=[(job(),{}),(job(dict(threading.POLICY,interop=True)),GRANT),
               (job(dict(threading.POLICY,optimizer_intraop=8)),GRANT)]
        for document,grant in cases:
            runtime=Runtime()
            with self.assertRaises(ValueError):initialize_optimizer_threads(document,grant,torch_module=runtime)
            self.assertEqual(runtime.calls,[])
    def test_already_initialized_interop_is_not_set_again(self):
        runtime=Runtime(2);initialize_optimizer_threads(job(),GRANT,torch_module=runtime)
        self.assertEqual(runtime.calls,[('intra',2)])
    def test_reused_interpreter_refusal_is_not_suppressed(self):
        class Used(Runtime):
            def set_num_interop_threads(self,n):raise RuntimeError('interop work already started')
        with self.assertRaisesRegex(RuntimeError,'already started'):
            initialize_optimizer_threads(job(),GRANT,torch_module=Used())
    def test_runtime_that_ignores_setter_is_rejected(self):
        class Broken(Runtime):
            def set_num_interop_threads(self,n):pass
        with self.assertRaisesRegex(ValueError,'initialization failed'):
            initialize_optimizer_threads(job(),GRANT,torch_module=Broken())

if __name__=='__main__':unittest.main()
