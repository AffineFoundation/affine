"""Instrumentation preserves original operations and failure semantics."""
import unittest
from unittest.mock import Mock,patch
from subnet.backend_jobs import measured_phase

class RolePhaseTiming(unittest.TestCase):
    def test_original_return_arguments_and_accumulated_calls(self):
        result=object();operation=Mock(return_value=result);times={}
        with patch('subnet.backend_jobs.time.monotonic',side_effect=[1,3,5,8]):
            self.assertIs(measured_phase(times,'downloads',operation,'original',limit=31),result)
            self.assertIs(measured_phase(times,'downloads',operation,'second',limit=33),result)
        self.assertEqual(operation.call_count,2)
        self.assertEqual(operation.call_args_list[0].args,('original',))
        self.assertEqual(operation.call_args_list[0].kwargs,{'limit':31})
        self.assertEqual(times,{'downloads':{'seconds':5.0,'calls':2}})

    def test_failure_propagates_without_retry_or_success_timing(self):
        operation=Mock(side_effect=ValueError('original integrity failure'));times={}
        with self.assertRaisesRegex(ValueError,'original integrity failure'):
            measured_phase(times,'admission',operation)
        operation.assert_called_once_with();self.assertEqual(times,{})
