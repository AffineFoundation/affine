import copy,hashlib,inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ops import mimo_host_capture_launcher as launcher
from subnet.mimo_lifecycle import Lifecycle
from subnet.mimo_coordinator import Coordinator
from subnet.mimo_release_policy import sign
from subnet.storage import Identity,canonical,sha

class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.actor=Identity(bytes([99])*32)
        self.proposal=dict(version=launcher.VERSION,
            host_files={name:launcher.digest_file(launcher.P/name)for name in('mimo_host_capture_launcher.py','mimo_host_capture_cache.py','mimo_host_path_launcher.py')},
            base={'lifecycle_config':{'path':'/fixture/original.json'}},workers=8,
            original_methods={name:hashlib.sha256(inspect.getsource(method).encode()).hexdigest()for name,method in [('Lifecycle._capture',Lifecycle._capture),('Coordinator.collect',Coordinator.collect)]})
        self.source=patch.object(launcher.paths,'checked_entry',return_value=SimpleNamespace(main=lambda:None));self.source_check=self.source.start();self.addCleanup(self.source.stop)
        self.read=patch.object(launcher.paths,'private_json',return_value={'release_policy':{'signer':self.actor.id}});self.read.start();self.addCleanup(self.read.stop)

    def config(self,proposal=None):
        p=copy.deepcopy(self.proposal if proposal is None else proposal)
        return dict(p,admission=sign(self.actor,dict(version=launcher.VERSION,configuration_sha256=sha(canonical(p)),source_unchanged=True,scope='host-capture-scheduling-only')))

    def test_valid_admission_checks_original_source_and_does_not_install(self):
        original=Lifecycle._capture
        module,cls=launcher.checked(self.config())
        self.assertIs(cls,Lifecycle);self.assertIs(Lifecycle._capture,original)
        self.source_check.assert_called_once_with(self.proposal['base'])

    def test_unsigned_worker_count_edit_rejected(self):
        c=self.config();c['workers']=16
        with self.assertRaisesRegex(ValueError,'admission'):launcher.checked(c)

    def test_signed_unbounded_worker_count_rejected(self):
        p=copy.deepcopy(self.proposal);p['workers']=100
        with self.assertRaisesRegex(ValueError,'concurrency'):launcher.checked(self.config(p))

    def test_wrong_original_method_rejected(self):
        p=copy.deepcopy(self.proposal);p['original_methods']['Coordinator.collect']='0'*64
        with self.assertRaisesRegex(ValueError,'method differs'):launcher.checked(self.config(p))

    def test_host_module_hash_mismatch_rejected(self):
        p=copy.deepcopy(self.proposal);p['host_files']['mimo_host_capture_cache.py']='0'*64
        with self.assertRaisesRegex(ValueError,'capture implementation'):launcher.checked(self.config(p))

    def test_wrong_authority_signature_rejected(self):
        p=self.config();p['admission']=sign(Identity(bytes([98])*32),p['admission']['payload'])
        with self.assertRaises(Exception):launcher.checked(p)

    def test_original_source_failure_propagates_before_installation(self):
        original=Lifecycle._capture;self.source_check.side_effect=ValueError('source mismatch')
        with self.assertRaisesRegex(ValueError,'source mismatch'):launcher.checked(self.config())
        self.assertIs(Lifecycle._capture,original)

    def test_install_changes_only_explicit_supplied_host_class(self):
        class Host:
            def _capture(self,*args):raise AssertionError('old')
        original=Lifecycle._capture;launcher.install_capture(Host,8)
        self.assertEqual(Host._capture.__module__,launcher.__name__)
        self.assertIs(Lifecycle._capture,original)

if __name__=='__main__':unittest.main()
