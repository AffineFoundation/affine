import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from subnet.role_router import RoutedJobs


class Availability(unittest.TestCase):
    def fixture(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        controller=SimpleNamespace(state=Path(self.temp.name),authority=SimpleNamespace(id='a'*64))
        endpoints={role:dict(host=role,port=22,worker_identity=str(i)*64)
            for i,role in enumerate(['mine','train','evaluate'],1)}
        endpoints['verify']=[dict(host=host,port=22,worker_identity=str(i)*64)
            for i,host in enumerate(['offline','online-one','online-two'],4)]
        return dict(roles=endpoints,verifier_queue=dict(external_api=True)),controller

    def test_offline_node_does_not_block_other_qualified_nodes_or_change_allowlist(self):
        config,controller=self.fixture()
        def remote(endpoint,unused):
            if endpoint['host']=='offline':raise subprocess.CalledProcessError(255,['ssh'])
            return SimpleNamespace(metadata=dict(source_files={'same':'source'},runtime_versions={'same':'runtime'}))
        with patch('subnet.remote_backend.RemoteJobs',side_effect=remote):jobs=RoutedJobs(config,controller)
        self.assertEqual(len(jobs.verifiers),2)
        self.assertEqual(jobs.unavailable_verifiers,['4'*64])
        self.assertEqual(set(jobs.queue.workers),{'4'*64,'5'*64,'6'*64})
        self.assertEqual(jobs.metadata['source_files'],{'same':'source'})

    def test_source_execution_error_is_not_masked_as_network_availability(self):
        config,controller=self.fixture()
        def remote(endpoint,unused):
            if endpoint['host']=='offline':raise subprocess.CalledProcessError(1,['python'])
            return SimpleNamespace(metadata={})
        with patch('subnet.remote_backend.RemoteJobs',side_effect=remote),self.assertRaises(subprocess.CalledProcessError):
            RoutedJobs(config,controller)

    def test_all_offline_never_claims_qualified_verification(self):
        config,controller=self.fixture()
        def remote(endpoint,unused):
            if endpoint['host']in ['offline','online-one','online-two']:raise subprocess.CalledProcessError(255,['ssh'])
            return SimpleNamespace(metadata={})
        with patch('subnet.remote_backend.RemoteJobs',side_effect=remote),self.assertRaisesRegex(RuntimeError,'no reachable qualified'):
            RoutedJobs(config,controller)


if __name__=='__main__':unittest.main()
