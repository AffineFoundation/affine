"""Offline payout safety tests. Run python -m ops.new_subnet_checks."""
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bittensor.wallet import Keypair
from subnet.chain import ChainAdapter, OWNER, activation_message, decode_commitment, hourly_points
from subnet.register import activation_payload, scale_bytes


class FakeAdapter(ChainAdapter):
    chain_owner = OWNER
    recycled = False
    rate = 0

    def registrations(self):
        return {'miner': {'uid': 7, 'public_key': 'aa'}}

    def query(self, name, params, block):
        return {
            'SubnetOwnerHotkey': self.chain_owner,
            'Uids': 0 if len(params)>1 and params[1] == OWNER else 7,
            'Keys': 'another' if self.recycled else 'miner',
            'LastUpdate': [0]*10,
            'WeightsSetRateLimit': self.rate,
            'WeightsVersionKey': 0,
        }[name]


class ChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.chain = SimpleNamespace(block=1000, plan=lambda i,w: SimpleNamespace(ok=True))
        self.adapter = FakeAdapter(self.temp.name, subtensor=self.chain)
        self.end = int(time.time()) // 3600 * 3600
        self.wallet = patch.object(self.adapter.bt, 'Wallet', return_value=SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address=OWNER)))
        self.wallet.start()

    def tearDown(self):
        self.wallet.stop()
        self.temp.cleanup()

    def payout(self, points=None):
        return self.adapter.submit_hour({'miner':3} if points is None else points,
                                        self.adapter.registrations(), self.end)

    def test_zero(self):
        self.assertEqual(self.payout({})['status'], 'zero_points_no_submission')

    def test_normalized(self):
        self.assertEqual(self.payout()['weights'], [1.0])

    def test_recycled_uid(self):
        self.adapter.recycled = True
        self.assertEqual(self.payout()['status'], 'stale_registration_denied')

    def test_owner_mismatch(self):
        self.adapter.chain_owner = 'unexpected'
        with self.assertRaises(RuntimeError): self.payout()

    def test_cardinality_policy(self):
        self.chain.plan = lambda i,w: SimpleNamespace(ok=False, reason='minimum recipients')
        self.assertEqual(self.payout()['status'], 'chain_policy_denied')

    def test_rate_limit(self):
        self.adapter.rate = 1100
        self.assertEqual(self.payout()['status'], 'deferred_rate_limit')

    def test_hour_boundary(self):
        reports = [{'epoch_id':'a','finalized_at':3600,'points':{'m':2}},
                   {'epoch_id':'b','finalized_at':7199,'points':{'m':3}},
                   {'epoch_id':'c','finalized_at':7200,'points':{'m':9}}]
        self.assertEqual(hourly_points(reports,7200), {'m':5})
        with self.assertRaises(ValueError): hourly_points(reports,7201)
        with self.assertRaises(ValueError): hourly_points([reports[0], reports[0]],7200)

    def test_nonpayable_epochs_never_enter_weights(self):
        reports=[{'epoch_id':'nonpayable-registered-test','finalized_at':self.end-1,'points':{'miner':999}},
                 {'epoch_id':'some-epoch','payable':False,'finalized_at':self.end-1,'points':{'miner':999}}]
        self.assertEqual(hourly_points(reports,self.end),{})

    def test_provisional_audits_cannot_enter_weights(self):
        reports=[{'epoch_id':'live-sampled','payable':True,'provisional':True,'finalized_at':self.end-1,'points':{'miner':999}},
                 {'epoch_id':'live-unresolved','payable':True,'duplicate_coverage':'incomplete','finalized_at':self.end-1,'points':{'miner':999}}]
        self.assertEqual(hourly_points(reports,self.end),{})

    def test_activation_ownership(self):
        key = Keypair.create_from_seed(bytes(range(32)),crypto_type=0)
        payload = activation_payload(key)
        self.assertEqual(decode_commitment(scale_bytes(payload.encode())), payload)
        self.assertTrue(key.verify(activation_message(key.ss58_address), key.sign(activation_message(key.ss58_address))))
        with self.assertRaises(ValueError): activation_payload(Keypair.create_from_seed(bytes(range(32)),crypto_type=1))

    def test_registration_scan_rejects_wrong_author(self):
        key = Keypair.create_from_seed(bytes(range(32)),crypto_type=0)
        another = Keypair.create_from_seed(bytes(range(1,33)),crypto_type=0)
        wire = scale_bytes(activation_payload(key).encode())
        fake = SimpleNamespace(block=1000,query_map=lambda *a,**kw:[
            (key.ss58_address,[(wire,900)]), (another.ss58_address,[(wire,900)])])
        adapter = ChainAdapter(self.temp.name,subtensor=fake)
        with patch.object(adapter,'query',side_effect=lambda n,p,b: 7 if n=='Uids' else key.ss58_address):
            regs = adapter.registrations()
        self.assertEqual(list(regs),[key.ss58_address])


if __name__ == '__main__':
    unittest.main()
