import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from subnet.remote_backend import RemoteJobs,RemoteObservationTimeout

class DetachedLaunchTests(unittest.TestCase):
    def job(self,folder):
        job=RemoteJobs.__new__(RemoteJobs);job.workspace=str(folder);job.code=str(folder)
        job.python=sys.executable;job.controller=SimpleNamespace(authority=SimpleNamespace(id='public-key'))
        return job
    def test_actual_detached_child_does_not_hold_transport_pipes_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package=root/'subnet';package.mkdir();(package/'__init__.py').write_text('')
            (package/'remote_runner.py').write_text("import time;from pathlib import Path;Path('ready').write_text('alive');time.sleep(60)\n")
            job=self.job(root);outputs=[]
            def command(text,timeout):
                result=subprocess.check_output(['bash','-c',text],text=True,timeout=2)
                outputs.append(json.loads(result));return result
            job.command=command;started=time.monotonic();job.launch_runner('original',str(root/'job.json'))
            child=outputs[0]['launcher_pid']
            try:
                for _ in range(50):
                    if (root/'ready').exists():break
                    time.sleep(.01)
                self.assertTrue((root/'ready').exists())
                self.assertLess(time.monotonic()-started,2);os.kill(child,0)
                self.assertEqual(os.getsid(child),child)
                marker=json.loads((root/'original-dispatch-attempt.json').read_text())
                self.assertEqual(marker['job_id'],'original')
                self.assertEqual((root/'original-dispatch-attempt.json').stat().st_mode & 0o777,0o600)
                with self.assertRaises(subprocess.CalledProcessError):job.launch_runner('original',str(root/'job.json'))
                self.assertEqual(len(outputs),1)
            finally:os.kill(child,signal.SIGTERM)
    def test_lost_ssh_reply_observes_same_job_without_second_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            job=self.job(directory);job.command=Mock(side_effect=subprocess.TimeoutExpired('ssh',30))
            job.remote_status=Mock(return_value={'phase':'running','job_id':'original'})
            job.launch_runner('original','/job')
            self.assertEqual(job.command.call_count,1);job.remote_status.assert_called_once_with('original',timeout=30)
    def test_unknown_launch_observation_preserves_original_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            job=self.job(directory);job.command=Mock(side_effect=subprocess.TimeoutExpired('ssh',30))
            job.remote_status=Mock(return_value={'phase':'not_launched'})
            with self.assertRaises(RemoteObservationTimeout)as caught:job.launch_runner('original','/job')
            self.assertEqual(caught.exception.job_id,'original');self.assertEqual(job.command.call_count,1)
    def test_transport_error_is_not_an_automatic_relaunch(self):
        with tempfile.TemporaryDirectory() as directory:
            job=self.job(directory);job.command=Mock(side_effect=ConnectionError('transport'));job.remote_status=Mock()
            with self.assertRaises(ConnectionError):job.launch_runner('original','/job')
            self.assertEqual(job.command.call_count,1);job.remote_status.assert_not_called()

    def test_observed_terminal_failure_is_not_reported_as_running(self):
        with tempfile.TemporaryDirectory() as directory:
            from subnet.remote_backend import RemoteJobTerminalError
            job=self.job(directory);job.command=Mock(side_effect=subprocess.TimeoutExpired('ssh',30))
            job.remote_status=Mock(return_value={'phase':'failed','exit_code':1})
            with self.assertRaises(RemoteJobTerminalError):job.launch_runner('original','/job')
            self.assertEqual(job.command.call_count,1)
