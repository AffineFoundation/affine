import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from subnet.commitment_transport import VERSION
from subnet.gpu_service import owned_dispatch_identities, run

class OwnedDispatchScope(unittest.TestCase):
    owned='1'*64
    external='2'*64
    def test_scope_does_not_mutate_public_participants_or_read_any_key(self):
        identities={self.external:'external-hotkey',self.owned:'owned-hotkey'}
        config={'owned_miner_identity_files':{self.owned:'/remote/miner-only.seed','3'*64:'/remote/inactive.seed'}}
        with patch.object(Path,'read_bytes',side_effect=AssertionError('no key reads')):
            self.assertEqual(owned_dispatch_identities(config,{'submission_transport_policy':VERSION},identities),[self.owned])
        self.assertEqual(identities,{self.external:'external-hotkey',self.owned:'owned-hotkey'})
    def test_no_owned_participants_and_legacy_behavior(self):
        self.assertEqual(owned_dispatch_identities({}, {'submission_transport_policy':VERSION},[self.external]),[])
        self.assertEqual(owned_dispatch_identities({}, {},[self.external,self.owned]),[self.external,self.owned])
    def test_bad_scopes_fail_closed(self):
        for mapping in ([], {'bad':'/key'}, {self.owned:'relative'}, {self.owned:None}, {self.owned:'/key\0'}):
            with self.assertRaises(ValueError):
                owned_dispatch_identities({'owned_miner_identity_files':mapping},{'submission_transport_policy':VERSION},[self.owned])
        with self.assertRaises(ValueError):owned_dispatch_identities({}, {'submission_transport_policy':'unknown'},[])
    def test_actual_coordinator_only_dispatches_owned_identity_with_external_first(self):
        class StopAfterDispatch(Exception):pass
        with tempfile.TemporaryDirectory() as d:
            state=Path(d);epoch='nonpayable-owned-dispatch'
            identities={self.external:'external-hotkey',self.owned:'owned-hotkey'}
            status=dict(active=dict(epoch=epoch,phase='mine',identities=identities),round=11,training_steps=21,checkpoint={'id':'learned'},checkpoint_path='cached',initial_published=True)
            manifest=dict(epoch=epoch,deadline=200,max_batches=3,submission_transport_policy=VERSION)
            (state/'controller.json').write_text(json.dumps(status));(state/(epoch+'-manifest.json')).write_text(json.dumps(manifest))
            jobs=SimpleNamespace(run=Mock(side_effect=StopAfterDispatch))
            config=dict(state=d,bucket={},remote={},source_bundle={},epoch_prefix='nonpayable-owned-dispatch',registration_policy='all_activated_subnet',owned_miner_identity_files={self.owned:'/remote/miner-only.seed'})
            bucket=Mock();bucket.presign.return_value='scoped-put'
            with patch('subnet.gpu_service.Bucket',return_value=bucket),patch('subnet.gpu_service.Gateway'),patch('subnet.gpu_service.RemoteController',return_value=SimpleNamespace(jobs=jobs)),patch('subnet.gpu_service.ChainAdapter'),patch('subnet.gpu_service.time.time',return_value=100),patch('subnet.gpu_service.log.exception'):
                with self.assertRaises(StopAfterDispatch):run(config,once=True)
            jobs.run.assert_called_once()
            fields=jobs.run.call_args.kwargs
            self.assertEqual(fields['miner_id'],self.owned)
            self.assertEqual(fields['miner_identity_file'],'/remote/miner-only.seed')
            self.assertEqual(len(fields['capability']['batch_put_urls']),3)
            self.assertEqual(json.loads((state/'controller.json').read_text())['active']['identities'],identities)
            self.assertEqual(json.loads((state/(epoch+'-manifest.json')).read_text()),manifest)
