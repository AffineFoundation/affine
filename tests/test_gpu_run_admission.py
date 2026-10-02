"""Unsigned preparation plans must fail before local or external side effects."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from subnet.gpu_service import run

class ReachedBucket(Exception):pass

class GPURunAdmission(unittest.TestCase):
    def blocked(self,flags,message):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            state=Path(directory)/'untouched'
            mocks={name:stack.enter_context(patch('subnet.gpu_service.'+name)) for name in ('Bucket','Gateway','RemoteController','ChainAdapter','save','initial_manifest')}
            path=stack.enter_context(patch('subnet.gpu_service.Path',side_effect=AssertionError('state path touched')))
            with self.assertRaisesRegex(ValueError,message):run(dict(state=str(state),**flags))
            path.assert_not_called()
            for mock in mocks.values():mock.assert_not_called()
            self.assertFalse(state.exists());self.assertEqual(list(Path(directory).iterdir()),[])
    def test_preparation_only_blocks_without_needing_any_runtime_fields(self):
        self.blocked({'preparation_only':True},'preparation-only config cannot run')
        self.blocked({'preparation_only':True,'activation_allowed':True},'preparation-only config cannot run')
    def test_activation_false_blocks_unsigned_plans_before_state(self):
        self.blocked({'activation_allowed':False},'config activation is not allowed')
        self.blocked({'preparation_only':False,'activation_allowed':False},'config activation is not allowed')
    def test_present_flags_require_literal_booleans(self):
        for field in ('preparation_only','activation_allowed'):
            for value in (None,0,1,'false','true',[],{}):
                with self.subTest(field=field,value=value):self.blocked({field:value},field+' must be boolean')
    def test_old_omitted_and_explicit_active_defaults_reach_constructor(self):
        for flags in ({},{'preparation_only':False},{'activation_allowed':True},{'preparation_only':False,'activation_allowed':True}):
            with self.subTest(flags=flags),tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
                state=Path(directory)/'active'
                bucket=stack.enter_context(patch('subnet.gpu_service.Bucket',side_effect=ReachedBucket))
                others=[stack.enter_context(patch('subnet.gpu_service.'+name)) for name in ('Gateway','RemoteController','ChainAdapter','save')]
                with self.assertRaises(ReachedBucket):run(dict(state=str(state),bucket={'fixture':True},**flags))
                bucket.assert_called_once_with({'fixture':True})
                self.assertTrue(state.is_dir())
                for mock in others:mock.assert_not_called()

if __name__=='__main__':unittest.main()
