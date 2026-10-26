"""Isolated original-controller gate controls; no live import or services."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=ROOT/'state/root-audits/e9-receipt-pending-controller-gate-20261004-v1/remote_backend.candidate.py'
E9='nonpayable-live-reward-math-v1--1791128338-9'


@unittest.skipUnless(CANDIDATE.is_file(),'operator-local pending gate candidate absent')
class PendingReceiptControllerGateTests(unittest.TestCase):
    def method(self):
        parsed=ast.parse(CANDIDATE.read_text())
        controller=next(row for row in parsed.body if isinstance(row,ast.ClassDef)and row.name=='RemoteController')
        method=next(row for row in controller.body if isinstance(row,ast.FunctionDef)and row.name=='train')
        # Compile only the first statement. In the signed E9 path it must raise
        # before any existing imports, capacity probes or original job writes.
        method.body=[method.body[0],ast.Return(ast.Constant('gate-not-selected'))]
        module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]));namespace={}
        exec(compile(module,'synthetic gate only','exec'),namespace)
        return namespace['train']

    def test_e9_refused_before_jobs_capacity_and_signature(self):
        controller=SimpleNamespace(training_execution_amendment_required_epochs=(E9,),jobs=Mock(),signed=Mock())
        with self.assertRaisesRegex(ValueError,'no training request permitted'):
            self.method()(controller,{'epoch':E9},{},'/unchanged/cache')
        controller.jobs.assert_not_called();controller.signed.assert_not_called()

    def test_gate_does_not_modify_other_epoch_or_manifest(self):
        controller=SimpleNamespace(training_execution_amendment_required_epochs=(E9,))
        manifest={'epoch':'other-epoch','checkpoint':{'id':'same'}}
        self.assertEqual(self.method()(controller,manifest,{},'/cache'),'gate-not-selected')
        self.assertEqual(manifest,{'epoch':'other-epoch','checkpoint':{'id':'same'}})

    def test_candidate_keeps_original_worker_source_and_queue_dispatch_bytes(self):
        original=ROOT/'state/live-math-launch-preparation-v1/distributed-preparation/future-source-forced-310492f6-v1/source-candidate/subnet/remote_backend.py'
        candidate=CANDIDATE.read_text()
        candidate=candidate.replace("        self.training_execution_amendment_required_epochs=tuple(remote.get('training_execution_amendment_required_epochs',()))\n",'')
        candidate=candidate.replace("        if manifest['epoch'] in self.training_execution_amendment_required_epochs:\n            raise ValueError('E9 training held pending authenticated verifier receipt amendment; no training request permitted')\n",'')
        self.assertEqual(candidate,original.read_text())


if __name__=='__main__':unittest.main()
