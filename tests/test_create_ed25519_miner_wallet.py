import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from ops.create_ed25519_miner_wallet import create,main
import inspect
try:
    import bittensor as bt
    SUPPORTED = "on_mnemonic" in inspect.signature(bt.wallets.create).parameters
except (ImportError, AttributeError):
    SUPPORTED = False

@unittest.skipUnless(SUPPORTED, "requires isolated pinned PoW SDK; keep mining runtime unchanged")
class WalletTests(unittest.TestCase):
    def test_real_keys_are_private_matching_and_not_printed(self):
        from bittensor.sp_core import Keypair
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'new-wallet';out=io.StringIO()
            with contextlib.redirect_stdout(out):main(['--directory',str(target)])
            result=json.loads(out.getvalue())
            seed=bytes.fromhex((target/'miner.seed').read_text().strip())
            key=Keypair.create_from_seed(seed,0)
            self.assertEqual(key.ss58_address,result['hotkey'])
            self.assertEqual(bytes(key.public_key).hex(),result['public_key'])
            for p in target.rglob('*'):self.assertEqual(p.stat().st_mode & 0o077,0)
            for p in target.glob('*.mnemonic.private.txt'):self.assertNotIn(p.read_text().strip(),out.getvalue())
            self.assertNotIn(seed.hex(),out.getvalue())
            before=(target/'identity.json').read_bytes()
            with self.assertRaises(FileExistsError):create(target)
            self.assertEqual(before,(target/'identity.json').read_bytes())
    def test_path_traversal_names_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):create(Path(tmp)/'new',wallet_name='../bad')
            self.assertFalse((Path(tmp)/'new').exists())
    def test_symlink_parent_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);(parent/'link').symlink_to(parent,target_is_directory=True)
            with self.assertRaises(ValueError):create(parent/'link'/'new')
            self.assertFalse((parent/'new').exists())

if __name__=='__main__':unittest.main()
