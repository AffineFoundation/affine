"""Equal transition weights for verified new model registrations; dry-run by default."""
import argparse
import fcntl
import json
import logging
import signal
from datetime import datetime, timezone
from pathlib import Path

import bittensor as bt

ROOT = Path.home() / '.local/state/affine-transition'
OWNER = '5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8'
REGISTRATIONS = Path('/home/const/subnet120/affine/state/registrations.json')


def eligible(records, start_block):
    # Activation alone is insufficient: require a verified, uploaded model.
    return sorted({r['hotkey'] for r in records.values()
                   if int(r.get('activate_block') or 0) >= start_block
                   and int(r.get('ready_block') or 0) >= start_block
                   and r.get('state') in ('queued', 'crowned')
                   and r.get('model_digest') and r.get('manifest_sha256')
                   and r.get('netuid') == 120 and r.get('hotkey') != OWNER})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text())
    with (ROOT / 'lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        status = {'start_block': config['start_block'], 'checked_utc': datetime.now(timezone.utc).isoformat()}
        if datetime.now(timezone.utc) >= datetime.fromisoformat(config['end_utc']):
            status['status'] = 'transition_expired_no_submission'
        else:
            records = json.loads(REGISTRATIONS.read_text())['records']
            hotkeys = eligible(records, config['start_block'])
            chain = bt.subtensor(network='finney')
            block = int(chain.block)
            def query(name, params):
                return chain.query(getattr(bt.storage.SubtensorModule, name), params, block=block)
            if query('SubnetOwnerHotkey', [120]) != OWNER:
                raise RuntimeError('owner changed')
            wallet = bt.Wallet(name='default', hotkey='default', path=str(Path.home()/'.bittensor/wallets'))
            if wallet.hotkey.ss58_address != OWNER:
                raise RuntimeError('wallet mismatch')
            uids = sorted({int(uid) for hk in hotkeys if (uid := query('Uids', [120, hk])) is not None})
            status.update(block=block, uids=uids, eligible_hotkeys=hotkeys, count=len(uids))
            if not uids:
                status['status'] = 'waiting_for_new_models'
            else:
                owner_uid = query('Uids', [120, OWNER])
                if owner_uid is None:
                    raise RuntimeError('validator not registered')
                last = int(query('LastUpdate', [120])[int(owner_uid)])
                remaining = max(0, int(query('WeightsSetRateLimit', [120])) - (block-last))
                status.update(weight_per_miner=1/len(uids), remaining_blocks=remaining)
                if remaining:
                    status['status'] = 'deferred_rate_limit'
                else:
                    intent = bt.SetWeights(netuid=120, uids=uids, weights=[1/len(uids)]*len(uids),
                                           version_key=int(query('WeightsVersionKey', [120]) or 0))
                    plan = chain.plan(intent, wallet)
                    if not plan.ok:
                        status['status'] = 'chain_policy_denied'
                    elif args.execute:
                        result = chain.execute(intent, wallet, wait_for_inclusion=True,
                                               wait_for_finalization=True, retries=0)
                        result.raise_for_failure()
                        status.update(status='submitted', block_hash=str(result.block_hash))
                    else:
                        status['status'] = 'planned'
        temp = ROOT / 'status.tmp'
        temp.write_text(json.dumps(status, indent=2)+'\n')
        temp.replace(ROOT / 'status.json')
        print(json.dumps(status))


if __name__ == '__main__':
    def timeout(*_):
        raise TimeoutError('submission timed out')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(660)
    try:
        main()
    except Exception as error:
        logging.error('transition invocation failed: %s', type(error).__name__)
        raise SystemExit(1)
