import unittest
from unittest.mock import patch
import requests
import tempfile
from pathlib import Path
from types import SimpleNamespace
from subnet import cli


class MinerTransportRecovery(unittest.TestCase):
    argv=['miner','--gateway','https://example.invalid','--authority','a'*64,'--key','owned.seed']

    def test_continuous_transport_failure_reenters_same_configuration(self):
        with patch('sys.argv',self.argv),patch.object(cli,'run',side_effect=[requests.HTTPError('502'),None]) as run,patch.object(cli.time,'sleep') as sleep:
            cli.main()
        self.assertEqual(run.call_count,2)
        self.assertIs(run.call_args_list[0].args[0],run.call_args_list[1].args[0])
        sleep.assert_called_once_with(10)

    def test_integrity_failure_never_retries(self):
        with patch('sys.argv',self.argv),patch.object(cli,'run',side_effect=ValueError('wrong authority')) as run,patch.object(cli.time,'sleep') as sleep:
            with self.assertRaisesRegex(ValueError,'wrong authority'):cli.main()
        self.assertEqual(run.call_count,1);sleep.assert_not_called()

    def test_once_transport_failure_is_reported(self):
        with patch('sys.argv',self.argv+['--once']),patch.object(cli,'run',side_effect=requests.ConnectionError('offline')) as run,patch.object(cli.time,'sleep') as sleep:
            with self.assertRaises(requests.ConnectionError):cli.main()
        self.assertEqual(run.call_count,1);sleep.assert_not_called()

    def test_authority_verified_pointer_renews_direct_discovery(self):
        class Done(Exception):pass
        suffix='?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test'
        initial='https://account.r2.cloudflarestorage.com/bucket/old'+suffix
        renewed='https://account.r2.cloudflarestorage.com/bucket/renewed'+suffix
        manifest_url='https://account.r2.cloudflarestorage.com/bucket/manifest'+suffix
        pointer=dict(epoch='nonpayable-test',transport_policy='direct-r2-v1',manifest_url=manifest_url,current_url=renewed,current_url_expires_at=100)
        manifest=dict(epoch='nonpayable-test',transport_policy='direct-r2-v1',capabilities={'owned':{}},checkpoint={'id':'approved'},deadline=90)
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(manifest_url=None,current_url=initial,cap_file=None,key='owned.seed',state=d,gateway='https://unused.invalid',authority='trusted',max_batches=1,once=False)
            with patch.object(cli,'fetch_signed',side_effect=[pointer,manifest]) as fetch,patch.object(cli,'identity',return_value=SimpleNamespace(id='owned')),patch.object(cli,'checkpoint_download',return_value=Path(d)),patch.object(cli,'check_runtime_profile'),patch.object(cli,'Miner',return_value=SimpleNamespace(batches=[])),patch.object(cli,'entries',return_value=[]),patch.object(cli.time,'time',return_value=0),patch.object(cli.time,'sleep',side_effect=Done):
                with self.assertRaises(Done):cli.run(args)
            self.assertEqual(args.current_url,renewed)
            self.assertEqual([call.args[0] for call in fetch.call_args_list],[initial,manifest_url])


if __name__=='__main__':unittest.main()
