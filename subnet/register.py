"""Publish existing affine2 Ed25519 activation; never buy/register a subnet slot."""
import argparse
import base64
import json
from pathlib import Path

from .chain import OWNER, activation_message


def activation_payload(key, netuid=120):
    if key.crypto_type != 0:
        raise ValueError('registered miner hotkey must be Ed25519')
    hotkey = key.ss58_address
    if hotkey == OWNER:
        raise ValueError('validator owner hotkey cannot be used as a test miner')
    signature = key.sign(activation_message(hotkey, netuid))
    return f'affine2|activate|{hotkey}|{base64.urlsafe_b64encode(signature).decode().rstrip("=")}'


def scale_bytes(raw):
    n = len(raw)
    if n < 64:
        return bytes([n << 2]) + raw
    if n < 16384:
        return ((n << 2) | 1).to_bytes(2, 'little') + raw
    if n < 1 << 30:
        return ((n << 2) | 2).to_bytes(4, 'little') + raw
    raise ValueError('oversized commitment')


def main():
    import bittensor as bt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wallet', required=True)
    p.add_argument('--hotkey', required=True)
    p.add_argument('--wallet-path', default=str(Path.home()/'.bittensor/wallets'))
    p.add_argument('--network', default='finney')
    p.add_argument('--netuid', type=int, default=120)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey, path=args.wallet_path)
    payload = activation_payload(wallet.hotkey, args.netuid)
    if not args.execute:
        print(json.dumps({'status': 'dry_run', 'payload': payload, 'netuid': args.netuid}))
        return
    chain = bt.subtensor(network=args.network)
    block = int(chain.block)
    uid = chain.query(bt.storage.SubtensorModule.Uids,
                      [args.netuid, wallet.hotkey.ss58_address], block=block)
    if uid is None:
        raise RuntimeError('hotkey must already be registered on subnet; no paid slot registration is performed')
    sealed = bt.timelock.encrypt(scale_bytes(payload.encode()), reveal_in='180s')
    call = bt.calls.Commitments.set_commitment(args.netuid, {'fields': [{'TimelockEncrypted': {
        'encrypted': sealed.ciphertext, 'reveal_round': sealed.reveal_round}}]})
    result = chain.submit_call(call, wallet, signer='hotkey')
    result.raise_for_failure()
    print(json.dumps({'status': 'submitted', 'uid': int(uid), 'block_hash': str(result.block_hash)}))


if __name__ == '__main__':
    main()
