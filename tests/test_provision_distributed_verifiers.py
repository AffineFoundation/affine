import socket
import threading
import unittest
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

from ops.provision_distributed_verifiers import wait_for_coordinator,process_identity,matches_process,worker_arguments,remote_seed_path,seed_admission_code
from nacl.signing import SigningKey


class RemoteSeedBindings(unittest.TestCase):
    def test_nondefault_namespace_requires_the_actual_approved_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'current-deployment/private/verifier.seed';path.parent.mkdir(parents=True)
            key=SigningKey.generate();path.write_text(key.encode().hex());path.chmod(0o600)
            original=path.read_bytes();exec(seed_admission_code(str(path),key.verify_key.encode().hex()),{})
            with self.assertRaisesRegex(AssertionError,'approved roster'):
                exec(seed_admission_code(str(path),SigningKey.generate().verify_key.encode().hex()),{})
            self.assertEqual(path.read_bytes(),original)

    def test_public_or_symlinked_seed_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'verifier.seed';key=SigningKey.generate();path.write_text(key.encode().hex());path.chmod(0o644)
            with self.assertRaisesRegex(AssertionError,'private canonical'):
                exec(seed_admission_code(str(path),key.verify_key.encode().hex()),{})
            path.chmod(0o600);alias=Path(directory)/'alias';alias.symlink_to(path)
            with self.assertRaisesRegex(AssertionError,'private canonical'):
                exec(seed_admission_code(str(alias),key.verify_key.encode().hex()),{})

    def test_relative_traversal_or_control_character_paths_are_refused(self):
        for value in ['private/key','/private/../key','/private/key\n','/private/key\0']:
            with self.assertRaises(ValueError):remote_seed_path(value)


class ActualProcessBindings(unittest.TestCase):
    def test_live_process_requires_exact_arguments_start_time_and_directory(self):
        argv=[sys.executable,'-I','-c','import time;time.sleep(30)','--coordinator','19081']
        process=subprocess.Popen(argv,cwd='/tmp')
        try:
            actual=process_identity(process.pid);marker={'ticks':actual['ticks']}
            self.assertTrue(matches_process(actual,marker,argv,'/tmp'))
            self.assertFalse(matches_process(actual,marker,argv[:-1]+['19082'],'/tmp'))
            self.assertFalse(matches_process(actual,marker,argv,os.getcwd()))
            self.assertFalse(matches_process(actual,{'ticks':'old-process'},argv,'/tmp'))
        finally:process.terminate();process.wait()
        self.assertIsNone(process_identity(process.pid))

    def test_zombie_is_not_a_running_tunnel(self):
        process=subprocess.Popen(['/bin/true'])
        try:
            deadline=time.monotonic()+2
            while time.monotonic()<deadline:
                fields=Path('/proc',str(process.pid),'stat').read_text().rsplit(')',1)[1].split()
                if fields[0]=='Z':break
                time.sleep(.01)
            self.assertEqual(fields[0],'Z')
            self.assertIsNone(process_identity(process.pid))
        finally:process.wait()

    def test_worker_reuse_checks_source_coordinator_key_and_checkpoint_binding(self):
        argv=worker_arguments('/python','http://127.0.0.1:19081','approved','/key','/work',{'checkpoint':'/cache'})
        actual={'ticks':'123','argv':argv,'cwd':'/source'};marker={'ticks':'123'}
        self.assertTrue(matches_process(actual,marker,argv,'/source'))
        for position in [5,7,9,11,13]:
            changed=list(argv);changed[position]='different'
            self.assertFalse(matches_process(actual,marker,changed,'/source'))
        self.assertFalse(matches_process(actual,marker,argv,'/another-source'))


class CoordinatorStartupTests(unittest.TestCase):
    def test_waits_for_delayed_listener(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
            ready = threading.Timer(.05, listener.listen)
            ready.start()
            try:
                wait_for_coordinator('127.0.0.1', port, timeout=2)
                listener.settimeout(1)
                connection, _ = listener.accept()
                connection.close()
            finally:
                ready.join()

    def test_closed_coordinator_times_out_before_worker_launch(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            with self.assertRaisesRegex(TimeoutError, 'no verifier workers started'):
                wait_for_coordinator('127.0.0.1', listener.getsockname()[1], timeout=.05)


if __name__ == '__main__':
    unittest.main()
