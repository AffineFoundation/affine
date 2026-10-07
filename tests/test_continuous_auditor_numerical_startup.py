"""Portable real-crypto startup controls; no private deployment fixtures."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from test_numerical_resolution import ReviewedUnknownControls
from subnet.continuous_audit_service import (
    ContinuousAuditor, admit_completed_reports, load_numerical_resolution,
    numerical_snapshot_arguments,
)

class NumericalStartup(unittest.TestCase):
    def setUp(self):
        self.fixture=ReviewedUnknownControls();self.fixture.setUp()
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.archive=self.path/'original.tar.gz';self.archive.write_bytes(self.fixture.archives[0]['archive'])
        self.ack=self.path/'ack.json';self.ack.write_text(json.dumps(self.fixture.archives[0]['ack']))
        self.settings=dict(policy_document=self.fixture.document,reference_archives=[dict(archive_path=str(self.archive),ack_path=str(self.ack))])
    def construct(self, settings=None):
        return ContinuousAuditor(SimpleNamespace(authority=SimpleNamespace(id=self.fixture.authority)),SimpleNamespace(),directory=self.path/'auditor',approved_sources={},job_metadata={},audit_policy=self.fixture.audit_policy,numerical_resolution=settings)
    def test_real_constructor_main_keyword_loads_authenticated_archive(self):
        service=self.construct(self.settings)
        self.assertEqual(service.numerical_resolution['numerical_resolution_policy'],self.fixture.document)
        self.assertEqual(service.numerical_resolution['numerical_reference_archives'],self.fixture.archives)
    def test_real_default_off_keeps_prior_admission_signature(self):
        self.assertIsNone(self.construct().numerical_resolution)
        self.assertNotIn('numerical_resolution',inspect.signature(admit_completed_reports).parameters)
    def test_signed_cutoff_does_not_change_prior_snapshot_inputs(self):
        settings=self.construct(self.settings).numerical_resolution
        self.assertEqual(numerical_snapshot_arguments(settings,29),{})
        self.assertEqual(numerical_snapshot_arguments(settings,30)['numerical_resolution_policy'],self.fixture.document)
    def test_forged_policy_fails_constructor_before_state_creation(self):
        settings=copy.deepcopy(self.settings);settings['policy_document']['payload']['entries'][0]['outcome']='verified_valid'
        with self.assertRaises(Exception):self.construct(settings)
        self.assertFalse((self.path/'auditor').exists())
    def test_truncated_archive_fails_real_constructor(self):
        self.archive.write_bytes(self.archive.read_bytes()[:40])
        with self.assertRaises(ValueError):self.construct(self.settings)
    def test_symlink_archive_fails_real_constructor(self):
        target=self.path/'symlink';target.symlink_to(self.archive);settings=copy.deepcopy(self.settings);settings['reference_archives'][0]['archive_path']=str(target)
        with self.assertRaises(ValueError):self.construct(settings)
    def test_unknown_config_schema_refused(self):
        with self.assertRaises(ValueError):self.construct(dict(self.settings,ignore_invalid=True))

if __name__=='__main__':unittest.main()
