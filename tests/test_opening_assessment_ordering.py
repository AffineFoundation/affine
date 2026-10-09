import copy
import importlib.util
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ops.trainer_lifecycle.opening_assessment_ordering import install


class OpeningOrderTests(unittest.TestCase):
    def fixture(self):
        calls=[];clock=[3599.0];record={}
        def prepare(c,cfg,status,contract):
            calls.append(('assessment',clock[0]))
            return dict(contract,assessment_cutoff=int(clock[0]//3600)*3600)
        def calibrate(c,cfg,status,contract):
            calls.append(('calibration',clock[0]))
            if 'completed' not in record:
                clock[0]+=3;record['completed']='exact-original-report';calls.append(('gpu-dispatch',clock[0]))
            return dict(contract,calibration=record['completed'])
        selection=types.SimpleNamespace(prepare_opening=prepare)
        calibration=types.SimpleNamespace(before_open=calibrate)
        install(selection,calibration,earliest_round=93)
        def run(round_number=93,contract=None):
            contract={'input':'unchanged'} if contract is None else contract
            first=selection.prepare_opening(None,{},dict(round=round_number),contract)
            return calibration.before_open(None,{},dict(round=round_number),first)
        return calls,clock,record,selection,calibration,run

    def test_cutoff_crossing_snapshots_after_completed_calibration(self):
        calls,clock,record,s,c,run=self.fixture();result=run()
        self.assertEqual(result['assessment_cutoff'],3600)
        self.assertEqual([x[0]for x in calls],['calibration','gpu-dispatch','assessment'])
        self.assertEqual(result['calibration'],'exact-original-report')

    def test_round92_preserves_original_order(self):
        calls,clock,record,s,c,run=self.fixture();result=run(92)
        self.assertEqual(result['assessment_cutoff'],0)
        self.assertEqual([x[0]for x in calls],['assessment','calibration','gpu-dispatch'])

    def test_failed_calibration_does_not_acquire_assessment(self):
        calls=[]
        def fail(*a):calls.append('calibration');raise TimeoutError('original pending')
        s=types.SimpleNamespace(prepare_opening=lambda *a:calls.append('assessment'))
        c=types.SimpleNamespace(before_open=fail);install(s,c,earliest_round=93)
        contract=s.prepare_opening(None,{},dict(round=93),{})
        with self.assertRaisesRegex(TimeoutError,'original pending'):c.before_open(None,{},dict(round=93),contract)
        self.assertEqual(calls,['calibration'])

    def test_completed_original_report_reused_on_retry(self):
        calls,clock,record,s,c,run=self.fixture();first=run();clock[0]=7201;second=run()
        self.assertEqual(sum(v[0]=='gpu-dispatch'for v in calls),1)
        self.assertEqual(first['calibration'],second['calibration'])
        self.assertEqual(second['assessment_cutoff'],7200)

    def test_assessment_failure_remains_closed_and_does_not_repeat_completed_gpu(self):
        calls,clock,record,s,c,run=self.fixture()
        # Original acquisition errors must propagate unchanged.
        marker=ValueError('fresh original authenticated blacklist assessment')
        calls=[];calibration=types.SimpleNamespace(before_open=lambda *a:dict(a[-1],calibration='original'))
        def failed(*a):raise marker
        selection=types.SimpleNamespace(prepare_opening=failed)
        install(selection,calibration,earliest_round=93)
        with self.assertRaises(ValueError)as caught:calibration.before_open(None,{},dict(round=93),{})
        self.assertIs(caught.exception,marker)

    def test_issued_opening_is_never_refreshed(self):
        calls,clock,record,s,c,run=self.fixture()
        for field in ('start','deadline'):
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'immutable'):run(contract={field:123})
        self.assertEqual(calls,[])

    def test_contract_not_mutated_and_no_op_configuration_stays_supported(self):
        contract={'nested':{'a':[1]}};saved=copy.deepcopy(contract)
        s=types.SimpleNamespace(prepare_opening=lambda *a:a[-1]);c=types.SimpleNamespace(before_open=lambda *a:a[-1])
        install(s,c,earliest_round=93)
        out=c.before_open(None,{},dict(round=93),s.prepare_opening(None,{},dict(round=93),contract))
        self.assertIs(out,contract);self.assertEqual(contract,saved)

    def test_invalid_boundaries_and_rounds_rejected(self):
        for v in (True,0,-1,'93'):
            with self.subTest(v=v),self.assertRaises(ValueError):install(None,None,earliest_round=v)
        calls,clock,record,s,c,run=self.fixture()
        for v in (True,-1,'93',None):
            with self.subTest(v=v),self.assertRaises(ValueError):run(v)


if __name__=='__main__':unittest.main()
