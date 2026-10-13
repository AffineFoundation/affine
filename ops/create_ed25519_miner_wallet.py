"""Create a private Ed25519 miner wallet and matching Affine seed; no chain writes.

Use an isolated Bittensor SDK environment. Secrets stay in the new owner-only
output directory. Back that directory up privately before terminating a GPU.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import stat


def _write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def create(directory, wallet_name='affine-pow', hotkey_name='miner'):
    import bittensor as bt
    from bittensor.sp_core import Keypair

    if 'on_mnemonic' not in inspect.signature(bt.wallets.create).parameters:
        raise RuntimeError('use the pinned PoW SDK environment with private mnemonic callbacks')
    if any(not name or name in ('.', '..') or '/' in name or '\\' in name
           for name in (wallet_name, hotkey_name)):
        raise ValueError('wallet and hotkey names must be single path components')
    directory = Path(directory).absolute()
    if any(parent.is_symlink() for parent in (directory, *directory.parents)):
        raise ValueError('output path must not contain symlinks')
    # Refuse any existing directory, including interrupted or completed wallets.
    # Never overwrite/recreate identities behind an existing miner process.
    old_mask = os.umask(0o077)
    try:
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        def mnemonic(role, value):
            _write(directory / (role + '.mnemonic.private.txt'), value + '\n')
        wallet = bt.wallets.create(
            name=wallet_name, hotkey=hotkey_name, path=str(directory / 'wallets'),
            use_password=False, overwrite=False, hotkey_crypto_type=0,
            on_mnemonic=mnemonic,
        )
        if wallet.hotkey.crypto_type != 0:
            raise ValueError('miner hotkey must be Ed25519')
        raw = json.loads(Path(wallet.hotkey_file.path).read_text())
        seed = bytes.fromhex(raw['secretSeed'].removeprefix('0x'))
        if len(seed) != 32 or bytes(Keypair.create_from_seed(seed, 0).public_key) != bytes(wallet.hotkey.public_key):
            raise ValueError('exported seed does not match wallet hotkey')
        _write(directory / 'miner.seed', seed.hex() + '\n')
        result = dict(wallet=wallet_name, hotkey_name=hotkey_name,
                      wallet_path=str(directory / 'wallets'), seed_file=str(directory / 'miner.seed'),
                      coldkey=wallet.coldkeypub.ss58_address, hotkey=wallet.hotkey.ss58_address,
                      public_key=bytes(wallet.hotkey.public_key).hex(), hotkey_crypto_type=0)
        _write(directory / 'identity.json', json.dumps(result, indent=2) + '\n')
        for path in directory.rglob('*'):
            metadata = path.lstat()
            if path.is_symlink() or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
                raise ValueError('wallet output must remain private and owned by this user')
        return result
    finally:
        os.umask(old_mask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, help='new directory under an existing private parent')
    parser.add_argument('--wallet-name', default='affine-pow')
    parser.add_argument('--hotkey-name', default='miner')
    args = parser.parse_args(argv)
    print(json.dumps(create(args.directory, args.wallet_name, args.hotkey_name)))


if __name__ == '__main__':
    main()
