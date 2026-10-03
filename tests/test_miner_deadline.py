"""Late local work must not become a late submission or a transport retry loop."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from subnet.miner import Miner, EpochClosed

class MinerDeadline(unittest.TestCase):
    def actor(self):
        a=Miner.__new__(Miner);a.manifest={'deadline':20};a.batches=[]
        a.state_path=None;a.cap={'put_url':'https://unused.invalid','headers':{}}
        return a

    def test_already_closed_upload_neither_packs_nor_sends(self):
        a=self.actor()
        with patch('subnet.miner.time.time',return_value=20),patch('subnet.miner.pack') as pack,patch('subnet.miner.requests.put') as put:
            with self.assertRaises(EpochClosed):a.upload()
            pack.assert_not_called();put.assert_not_called()

    def test_packing_crossing_deadline_keeps_local_state_without_sending(self):
        a=self.actor()
        with tempfile.TemporaryDirectory() as root:
            a.state_path=Path(root)/'epoch.zip'
            with patch('subnet.miner.time.time',side_effect=[19,20]),patch('subnet.miner.pack',return_value=b'local candidate'),patch('subnet.miner.requests.put') as put:
                with self.assertRaises(EpochClosed):a.upload()
            self.assertEqual(a.state_path.read_bytes(),b'local candidate');put.assert_not_called()

    def test_expired_inflight_403_is_closed_window_but_early_403_is_transport_error(self):
        import requests
        a=self.actor();response=Mock(status_code=403);response.raise_for_status.side_effect=requests.HTTPError('rejected')
        with patch('subnet.miner.pack',return_value=b'data'),patch('subnet.miner.requests.put',return_value=response):
            with patch('subnet.miner.time.time',side_effect=[19,19,20]),self.assertRaises(EpochClosed):a.upload()
            with patch('subnet.miner.time.time',return_value=19),self.assertRaises(requests.HTTPError):a.upload()

    def test_generation_finishing_late_cannot_replace_previous_batches(self):
        a=self.actor();a.manifest.update(K=1,L=1,epoch='test',checkpoint={'id':'test'})
        prior=[('previous uploaded batch',[])];a.batches=prior;a.runtimes={};a.checkpoint='unused'
        a.runtime=SimpleNamespace(rollout=lambda *args:({'classification':'positive'},[]))
        a.runtime.for_environment=lambda *args:a.runtime
        with patch('subnet.miner.entry',return_value={'env_id':'math','spec':{}}),patch('subnet.miner.harness_for',return_value={}),patch('subnet.miner.time.time',side_effect=[19,19,20]),patch('subnet.miner.pack') as pack:
            with self.assertRaises(EpochClosed):a.search(1,max_attempts=1)
        self.assertIs(a.batches,prior);pack.assert_not_called()

    def test_closed_search_does_not_load_model(self):
        a=self.actor()
        with patch('subnet.miner.time.time',return_value=20),patch('subnet.miner.make_runtime') as runtime:
            with self.assertRaises(EpochClosed):a.search(1)
            runtime.assert_not_called()

    def test_cli_stops_this_epoch_cleanly_without_trying_more_tasks(self):
        from subnet import cli
        manifest=dict(epoch='test',deadline=20,capabilities={'owned':{}},checkpoint={'id':'model'})
        miner=Mock(batches=[]);miner.search.side_effect=EpochClosed('rollout finished late')
        with tempfile.TemporaryDirectory() as root:
            args=SimpleNamespace(search_budget=1,manifest_url='https://unused.invalid',current_url=None,
                cap_file=None,key='owned.seed',state=root,gateway='https://unused.invalid',authority='trusted',
                max_batches=3,once=True,env_id=None,indices=None)
            with patch.object(cli,'fetch_signed',return_value=manifest),patch.object(cli,'identity',return_value=SimpleNamespace(id='owned')),patch.object(cli,'selected_tasks',return_value=[('math',1),('math',2)]),patch.object(cli,'checkpoint_download',return_value=Path(root)),patch.object(cli,'check_runtime_profile'),patch.object(cli,'Miner',return_value=miner),patch.object(cli.time,'time',return_value=19):
                cli.run(args)
        miner.search.assert_called_once();miner.upload.assert_not_called()

if __name__=='__main__':unittest.main()
