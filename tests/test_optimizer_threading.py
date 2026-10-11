"""Mechanical controls and synthetic signed admission; no GPU execution claim."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from subnet import optimizer_threading as t
from subnet import unaudited_training_execution as a
from subnet.training_receipts import sha
import test_positive_sft_execution_route as route
import test_positive_only_training_evidence as evidence


class FakeTorch:
    def __init__(self, intra=2, inter=2):
        self.intra=intra; self.inter=inter; self.changes=[]
    def get_num_threads(self): return self.intra
    def get_num_interop_threads(self): return self.inter
    def set_num_threads(self, n): self.changes.append(n); self.intra=n


class OptimizerThreadingTests(unittest.TestCase):
    def test_only_optimizer_has_sixteen_threads(self):
        torch=FakeTorch(); gradients=object()
        def update(*, gradients):
            self.assertEqual(t.snapshot(torch), dict(intraop=16,interop=2))
            return dict(marker=42)
        optimizer=SimpleNamespace(step=Mock(side_effect=update))
        result, observed=t.step(torch,optimizer,gradients,t.POLICY)
        optimizer.step.assert_called_once_with(gradients=gradients)
        self.assertEqual(result,dict(marker=42))
        self.assertEqual(observed,t.expected_evidence(t.POLICY))
        self.assertEqual(torch.changes,[16,2])

    def test_failed_optimizer_restores_threads_and_propagates_failure(self):
        torch=FakeTorch(); optimizer=SimpleNamespace(step=Mock(side_effect=RuntimeError('failed')))
        with self.assertRaisesRegex(RuntimeError,'failed'):
            t.step(torch,optimizer,{},t.POLICY)
        self.assertEqual(t.snapshot(torch),dict(intraop=2,interop=2))
        self.assertEqual(torch.changes,[16,2])

    def test_wrong_entry_runtime_never_runs_optimizer(self):
        for torch in (FakeTorch(16,2),FakeTorch(2,1)):
            optimizer=SimpleNamespace(step=Mock())
            with self.assertRaises(ValueError):t.step(torch,optimizer,{},t.POLICY)
            optimizer.step.assert_not_called();self.assertEqual(torch.changes,[])

    def test_unqualified_overrides_rejected(self):
        for mutation in ({'optimizer_intraop':32},{'steady_intraop':16},{'interop':True},
                         {'other':1},{'version':'anything'}):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                t.configuration(dict(t.POLICY,**mutation))
        with self.assertRaises(ValueError):t.configuration(None)

    def test_report_expected_policy_is_independent(self):
        d,u,p,args=evidence.fixture();args['optimizer_threading']=copy.deepcopy(t.POLICY)
        d['optimizer_threading']=copy.deepcopy(t.POLICY)
        for update in u:update['optimizer_threading']=t.expected_evidence(t.POLICY)
        evidence.validate_diagnostics(d,u,p,**args)
        for mutate in (
            lambda d,u:u[0]['optimizer_threading']['during'].update(intraop=2),
            lambda d,u:u[1]['optimizer_threading']['after'].update(intraop=16),
            lambda d,u:d['optimizer_threading'].update(optimizer_intraop=8),
            lambda d,u:u[0].pop('optimizer_threading')):
            dd,uu=copy.deepcopy(d),copy.deepcopy(u);mutate(dd,uu)
            with self.assertRaises(ValueError):evidence.validate_diagnostics(dd,uu,p,**args)
        oldargs=dict(args);oldargs.pop('optimizer_threading')
        with self.assertRaises(ValueError):evidence.validate_diagnostics(d,u,p,**oldargs)

    def fixture(self):
        f=route.PositiveSFTRouteTests();f.setUp();self.addCleanup(f.doCleanups)
        r=copy.deepcopy(f.release['payload']);r['optimizer_threading']=copy.deepcopy(t.POLICY)
        r['execution_source_files']['subnet/optimizer_threading.py']='a'*64
        r['changed_source_files']={n:h for n,h in r['execution_source_files'].items()
            if r['original_source_files'].get(n)!=h}
        q=r['execution_qualification']['payload']
        q.update(optimizer_threading=copy.deepcopy(t.POLICY),
            optimizer_threading_equivalence_evidence_sha256='b'*64,
            execution_source_files_sha256=sha(r['execution_source_files']))
        r['execution_qualification']=f.sign(q)
        f.fx.f.job['source_files']=copy.deepcopy(r['execution_source_files'])
        return f,r

    def test_signed_policy_propagates_from_release_to_original_job_and_report(self):
        f,r=self.fixture();signed=f.sign(r)
        a.release(signed,f.authority,now=100)
        for n,step in ((90,0),(91,8)):
            amended=f.amend(f.job(n,step),signed)
            actual=a.validate(f.sign(amended),f.authority)
            self.assertEqual(actual['optimizer_threading'],t.POLICY)
            self.assertEqual(a.provenance(f.sign(amended),f.authority)['optimizer_threading'],t.POLICY)

    def test_missing_source_or_qualification_cannot_enable_threading(self):
        f,r=self.fixture()
        def missing_source(x):
            x['execution_source_files'].pop('subnet/optimizer_threading.py')
            x['changed_source_files'].pop('subnet/optimizer_threading.py')
            x['execution_qualification']['payload']['execution_source_files_sha256']=sha(x['execution_source_files'])
        def missing_equivalence(x):
            x['execution_qualification']['payload'].pop('optimizer_threading_equivalence_evidence_sha256')
        def missing_policy(x): x['execution_qualification']['payload'].pop('optimizer_threading')
        for mutate in (missing_source,missing_equivalence,missing_policy):
            bad=copy.deepcopy(r);mutate(bad)
            bad['execution_qualification']=f.sign(bad['execution_qualification']['payload'])
            with self.assertRaises(ValueError):a.release(f.sign(bad),f.authority)

    def test_historical_objectives_reject_override(self):
        for version in (a.VERSION,a.OBJECTIVE_VERSION,a.GENESIS_VERSION):
            with self.subTest(version=version),self.assertRaises(ValueError):
                a.configured_objective(dict(version=version,optimizer_threading=t.POLICY))

    def test_historical_v6_shape_still_valid(self):
        f=route.PositiveSFTRouteTests();f.setUp();self.addCleanup(f.doCleanups)
        self.assertNotIn('optimizer_threading',a.release(f.release,f.authority))


if __name__=='__main__': unittest.main()
