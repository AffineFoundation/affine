import tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from subnet.chain import ChainAdapter


class ZeroSinkControls(unittest.TestCase):
    def adapter(self, path):
        a=ChainAdapter.__new__(ChainAdapter);a.state_dir=Path(path);a.netuid=120;a.owner='owner'
        a.bt=SimpleNamespace(Wallet=lambda **k:SimpleNamespace(hotkey=SimpleNamespace(ss58_address='owner')),
            SetWeights=lambda **k:k)
        a.chain=SimpleNamespace(block=100,plan=lambda intent,wallet:SimpleNamespace(ok=True))
        a.registrations=lambda:{}
        def query(name,params,block):
            return {'SubnetOwnerHotkey':'owner','Uids':0,'Keys':'owner','LastUpdate':[0],
                    'WeightsSetRateLimit':0,'WeightsVersionKey':0}[name]
        a.query=query;return a
    def test_default_empty_behavior_unchanged(self):
        with tempfile.TemporaryDirectory()as d:
            result=self.adapter(d).submit_hour({}, {}, int(time.time())//3600*3600)
            self.assertEqual(result['status'],'zero_points_no_submission')
    def test_explicit_zero_routes_to_actual_owner(self):
        with tempfile.TemporaryDirectory()as d:
            result=self.adapter(d).submit_hour({}, {}, int(time.time())//3600*3600,
                                               zero_total_policy='owner-sink-v1')
            self.assertEqual(result['status'],'planned');self.assertEqual(result['uids'],[0])
            self.assertEqual(result['weights'],[1.]);self.assertEqual(result['zero_total_policy'],'owner-sink-v1')
    def test_bad_reverse_mapping_refuses(self):
        with tempfile.TemporaryDirectory()as d:
            a=self.adapter(d);old=a.query;a.query=lambda n,p,b:'wrong'if n=='Keys'else old(n,p,b)
            with self.assertRaisesRegex(RuntimeError,'sink owner'):a.submit_hour({}, {}, int(time.time())//3600*3600,
                                                                                zero_total_policy='owner-sink-v1')
    def test_unknown_zero_policy_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            with self.assertRaises(ValueError):self.adapter(d).submit_hour({}, {}, 0,zero_total_policy='invented')
    def test_stale_nonzero_recipient_cannot_trigger_sink(self):
        with tempfile.TemporaryDirectory()as d:
            result=self.adapter(d).submit_hour({'departed':1}, {'departed':{'uid':85}},
                int(time.time())//3600*3600,zero_total_policy='owner-sink-v1')
            self.assertEqual(result['status'],'stale_registration_denied')
            self.assertNotIn('zero_total_policy',result)
    def test_rate_limited_sink_is_deferred(self):
        with tempfile.TemporaryDirectory()as d:
            a=self.adapter(d);old=a.query;a.query=lambda n,p,b:200 if n=='WeightsSetRateLimit'else old(n,p,b)
            result=a.submit_hour({}, {}, int(time.time())//3600*3600,zero_total_policy='owner-sink-v1')
            self.assertEqual(result['status'],'deferred_rate_limit');self.assertEqual(result['remaining_blocks'],100)

if __name__=='__main__':unittest.main()
