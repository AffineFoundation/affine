"""Signed registration mapping at the expanded subnet boundary."""
import base64
import unittest

from nacl.signing import SigningKey
from dashboard.mimo_live_projection import canonical, contribution_grid


class UIDGridTests(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()
        self.manifest = {'epoch': 'test-epoch'}

    def grid(self, uids):
        payload = dict(epoch='test-epoch', snapshot_block=50, registrations={
            str(i): dict(uid=uid, public_key=str(i), snapshot_block=50)
            for i, uid in enumerate(uids)})
        document = dict(payload=payload, signer=self.authority,
            signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())
        return contribution_grid(document, [{'miner': '0', 'total_groups': 9}],
                                 self.manifest, self.authority)

    def test_uid999_retains_position(self):
        grid, outside = self.grid([999])
        self.assertEqual((len(grid), grid[999], sum(grid), outside), (1000, 9, 9, 0))

    def test_legacy_grid_retains_positions_and_size(self):
        grid, outside = self.grid([85, 255])
        self.assertEqual((len(grid), grid[85], grid[255], outside), (256, 9, 0, 0))

    def test_invalid_or_duplicate_uid_never_misattributed(self):
        for uids in ([True], [-1], [1000], ['999'], [999, 999]):
            with self.subTest(uids=uids):
                self.assertEqual(self.grid(uids), (None, 9))

    def test_missing_historical_registration_is_unavailable(self):
        self.assertEqual(contribution_grid(None, [{'miner': '0', 'total_groups': 9}],
                                          self.manifest, self.authority), (None, 9))


if __name__ == '__main__':
    unittest.main()
