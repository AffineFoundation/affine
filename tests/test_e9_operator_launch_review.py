"""Synthetic launch preflight/body/repeat controls; no credentials or network."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


HELPER=Path(__file__).resolve().parents[1]/'state/root-audits/launch_e9_probability_diagnostic_20261004_v1.py'
MODULE=runpy.run_path(str(HELPER),run_name='operator_launch_test_definition')if HELPER.is_file()else None


@unittest.skipIf(MODULE is None,'operator-local launch helper is absent')
class E9OperatorLaunchReviewTests(unittest.TestCase):
    def preflight(self,model='NVIDIA H200',applications='',processes=''):
        with tempfile.TemporaryDirectory(prefix='diagnostic paths ')as directory:
            base=Path(directory);source=base/'source';source.mkdir();cache=base/'cache';cache.mkdir()
            (source/'synthetic.py').write_bytes(b'approved source')
            (cache/'config.json').write_bytes(b'approved checkpoint')
            plan=dict(endpoint={'code':str(source)},checkpoint_cache=str(cache),
                      workspace=str(base/'new workspace'),remote_control=str(base/'new control'))
            program=MODULE['remote_preflight_program'](plan)
            def output(command,**kwargs):
                if command[0]=='ps':return processes+'\npython -c '+program
                if '--query-gpu=name,uuid'in command:return model+', GPU-SYNTHETIC\n'
                if '--query-compute-apps=pid'in command:return applications
                raise AssertionError('unexpected synthetic command')
            stdout=io.StringIO()
            with patch('subprocess.check_output',side_effect=output),\
                    patch('importlib.metadata.version',side_effect=lambda name:'synthetic-'+name),\
                    patch('os.statvfs',return_value=SimpleNamespace(f_bavail=20*1024**3,f_frsize=1)),\
                    contextlib.redirect_stdout(stdout):
                exec(compile(program,'<synthetic operator preflight>','exec'),{})
            return json.loads(stdout.getvalue())

    def test_idle_evaluator_h200_full_source_and_checkpoint_readback(self):
        result=self.preflight()
        self.assertTrue(result['GPU_idle'])
        self.assertEqual(result['source_files'],{'synthetic.py':hashlib.sha256(b'approved source').hexdigest()})
        self.assertEqual(result['checkpoint_files'],{'config.json':hashlib.sha256(b'approved checkpoint').hexdigest()})

    def test_baseline_h100_busy_gpu_and_live_runner_refuse_preflight(self):
        for kwargs in ({'model':'NVIDIA H100'},{'applications':'123\n'},
                       {'processes':'python -m subnet.remote_runner existing-job.json'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(AssertionError):self.preflight(**kwargs)

    def test_descriptor_body_closes_on_truncation(self):
        body=io.BytesIO(b'{}');bucket=Mock();bucket.name='synthetic'
        bucket.client.get_object.return_value={'Body':body,'ContentLength':3}
        with self.assertRaisesRegex(ValueError,'complete authority descriptor GET'):
            MODULE['bounded_get'](bucket,'synthetic/descriptor.json')
        self.assertTrue(body.closed)

    def test_partial_original_run_cannot_launch_again(self):
        with tempfile.TemporaryDirectory()as directory:
            (Path(directory)/'original-run-v1').mkdir()
            globals_=MODULE['run_reviewed'].__globals__
            with patch.dict(globals_,PREPARED=Path(directory)),patch('subprocess.run')as command:
                with self.assertRaises(FileExistsError):
                    MODULE['run_reviewed']({'helper':{}},{'endpoint':{}})
                command.assert_not_called()


if __name__=='__main__':unittest.main()
