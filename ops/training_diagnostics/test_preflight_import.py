"""Exercise the actual run entry without importing a numerical library/model."""
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import evaluate


class ReachedRuntimeCheck(Exception): pass


class PreflightTests(unittest.TestCase):
    def exercise(self,module):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('job','source','model'):(root/name).mkdir()
            now=time.time()
            plan=dict(version='fresh-approved-training128-plan-v1',program_sha256=module.file_hash(module.__file__),
                      production_mutation=False,optimizer_allocation=False,task_count=128,created_at=now-1,expires_at=now+60,
                      directory=str(root/'job'),source=str(root/'source'),checkpoint=dict(path=str(root/'model')),
                      source_files={},runtime_versions={'synthetic-runtime':'1'})
            path=root/'plan.json';path.write_text('{}')
            with mock.patch.object(module,'signed',return_value=plan),mock.patch.object(module.importlib.metadata,'version',side_effect=ReachedRuntimeCheck) as reached:
                try:module.run(path,check_only=True)
                finally:self.runtime_check_calls=reached.call_count

    def test_actual_run_entry_reaches_runtime_guard_without_model_import(self):
        with self.assertRaises(ReachedRuntimeCheck):self.exercise(evaluate)
        self.assertEqual(self.runtime_check_calls,1)



if __name__=='__main__':unittest.main()
