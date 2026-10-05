"""Live SN120 identities and single-writer payouts; never fabricate recipients."""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

OWNER = '5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8'


def activation_message(hotkey: str, netuid: int = 120) -> bytes:
    body = json.dumps({'hotkey': hotkey, 'netuid': netuid}, sort_keys=True,
                      separators=(',', ':')).encode()
    rid = hashlib.sha256(b'affine-registration-v1\0' + body).hexdigest()
    return f'affine-activate|v1|{netuid}|{hotkey}|{rid}'.encode()


def decode_commitment(value) -> str:
    raw = bytes.fromhex(value[2:]) if isinstance(value, str) and value.startswith('0x') else (
        value.encode('latin1') if isinstance(value, str) else bytes(value))
    if not raw:
        raise ValueError('empty commitment')
    mode = raw[0] & 3
    offset = (1, 2, 4)[mode] if mode < 3 else 1 + (raw[0] >> 2) + 4
    if len(raw) < offset:
        raise ValueError('truncated SCALE length')
    length = int.from_bytes(raw[:offset], 'little') >> 2 if mode < 3 else int.from_bytes(raw[1:offset], 'little')
    if length != len(raw) - offset:
        raise ValueError('SCALE length mismatch')
    return raw[offset:].decode('utf8')


def hourly_points(reports: list[dict], window_end: int) -> dict[str, int]:
    """Authenticated finalized reports only, UTC interval [end-3600,end)."""
    if window_end % 3600:
        raise ValueError('window_end must be an integral UTC hour')
    points, seen = {}, set()
    for report in reports:
        if report.get('payable') is False or report.get('provisional',False) or report.get('duplicate_coverage')=='incomplete' or str(report.get('epoch_id','')).startswith(('nonpayable-', 'test-', 'mock-')):
            continue
        if not window_end - 3600 <= float(report['finalized_at']) < window_end:
            continue
        eid = report['epoch_id']
        if eid in seen:
            raise ValueError('duplicate finalized epoch')
        seen.add(eid)
        for hotkey, value in report['points'].items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('points must be nonnegative integers')
            points[hotkey] = points.get(hotkey, 0) + value
    return points


class ChainAdapter:
    def __init__(self, state_dir: str | Path, network: str = 'finney', netuid: int = 120,
                 expected_owner: str = OWNER, subtensor=None):
        import bittensor as bt
        from bittensor.wallet import Keypair
        self.bt = bt
        self.keypair_type = Keypair
        self.chain = subtensor if subtensor is not None else bt.subtensor(network=network)
        self.netuid, self.owner = netuid, expected_owner
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def query(self, name, params, block):
        return self.chain.query(getattr(self.bt.storage.SubtensorModule, name), params, block=block)

    def registrations(self) -> dict[str, dict]:
        """Chain commitment author AND Ed25519 signature must match a current UID."""
        block, registered = int(self.chain.block), {}
        rows = self.chain.query_map(self.bt.storage.Commitments.RevealedCommitments,
                                    [self.netuid], block=block)
        # Materialize BOTH directions at the identical immutable block. Map
        # ambiguity is an infrastructure error, never a fabricated empty roster.
        uids = {}
        for hotkey, uid in self.chain.query_map(self.bt.storage.SubtensorModule.Uids, [self.netuid], block=block):
            if not isinstance(hotkey, str) or type(uid) is not int or uid < 0 or hotkey in uids or uid in uids.values():
                raise ValueError('ambiguous fixed-block UID map')
            uids[hotkey] = uid
        keys = {}
        for uid, hotkey in self.chain.query_map(self.bt.storage.SubtensorModule.Keys, [self.netuid], block=block):
            if type(uid) is not int or uid < 0 or not isinstance(hotkey, str) or uid in keys or hotkey in keys.values():
                raise ValueError('ambiguous fixed-block hotkey map')
            keys[uid] = hotkey
        for hotkey, entries in rows:
            if hotkey == self.owner:
                continue
            for value, activation_block in entries:
                try:
                    parts = decode_commitment(value).strip().split('|')
                    if len(parts) != 4 or parts[:2] != ['affine2', 'activate'] or parts[2] != hotkey:
                        continue
                    signature = base64.urlsafe_b64decode(parts[3] + '=' * (-len(parts[3]) % 4))
                    key = self.keypair_type(ss58_address=hotkey, crypto_type=0)
                    if len(signature) != 64 or not key.verify(activation_message(hotkey, self.netuid), signature):
                        continue
                    uid = uids.get(hotkey)
                    if uid is None or keys.get(uid) != hotkey:
                        continue
                    registered[hotkey] = {'uid': int(uid), 'public_key': bytes(key.public_key).hex(),
                                         'activate_block': int(activation_block), 'snapshot_block': block}
                except (ValueError, TypeError, KeyError):
                    continue
        return registered

    def submit_hour(self, points: dict[str, int], registrations: dict[str, dict],
                    window_end: int, execute: bool = False) -> dict:
        if window_end % 3600 or window_end > int(time.time()):
            raise ValueError('only completed integral UTC hour windows may pay out')
        if any(isinstance(p, bool) or not isinstance(p, int) or p < 0 for p in points.values()):
            raise ValueError('invalid point count')
        with (self.state_dir / 'weights.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            status_path = self.state_dir / 'weights.json'
            previous = json.loads(status_path.read_text()) if status_path.exists() else {}
            if previous.get('last_submitted_window', -1) >= window_end:
                return {'status': 'already_submitted', 'window_end': window_end}
            block = int(self.chain.block)
            if self.query('SubnetOwnerHotkey', [self.netuid], block) != self.owner:
                raise RuntimeError('subnet owner changed')
            wallet = self.bt.Wallet(name='default', hotkey='default', path=str(Path.home()/'.bittensor/wallets'))
            if wallet.hotkey.ss58_address != self.owner:
                raise RuntimeError('wallet identity mismatch')
            fresh = self.registrations()
            block = int(self.chain.block)
            if self.query('SubnetOwnerHotkey', [self.netuid], block) != self.owner:
                raise RuntimeError('subnet owner changed during preparation')
            status_block = block
            recipients, stale = [], []
            for hotkey, count in points.items():
                if not count:
                    continue
                before, current = registrations.get(hotkey), fresh.get(hotkey)
                if (not before or not current or current['uid'] != before['uid']
                        or current['public_key'] != before['public_key']
                        or self.query('Keys', [self.netuid, current['uid']], block) != hotkey
                        or self.query('Uids', [self.netuid, hotkey], block) != current['uid']):
                    stale.append(hotkey)
                    continue
                recipients.append((current['uid'], hotkey, count))
            status = {'window_end': window_end, 'block': status_block, 'excluded_stale': stale}
            if stale:
                # Never silently renormalize winners after UID recycling.
                status['status'] = 'stale_registration_denied'
            elif not recipients:
                status['status'] = 'zero_points_no_submission'
            else:
                recipients.sort()
                total = sum(p for _, _, p in recipients)
                weights = [p/total for _, _, p in recipients]
                if not math.isclose(sum(weights), 1.0):
                    raise RuntimeError('normalization failed')
                status.update(uids=[u for u, _, _ in recipients], hotkeys=[h for _, h, _ in recipients], weights=weights)
                uid = self.query('Uids', [self.netuid, self.owner], block)
                if uid is None:
                    raise RuntimeError('validator not registered')
                last = int(self.query('LastUpdate', [self.netuid], block)[int(uid)])
                remaining = max(0, int(self.query('WeightsSetRateLimit', [self.netuid], block)) - block + last)
                status['remaining_blocks'] = remaining
                if remaining:
                    status['status'] = 'deferred_rate_limit'
                else:
                    intent = self.bt.SetWeights(netuid=self.netuid, uids=status['uids'], weights=weights,
                        version_key=int(self.query('WeightsVersionKey', [self.netuid], block) or 0))
                    plan = self.chain.plan(intent, wallet)
                    if not plan.ok:
                        status.update(status='chain_policy_denied', detail=str(plan))
                    elif not execute:
                        status['status'] = 'planned'
                    else:
                        marker = Path.home()/'.local/state/affine-transition/active'
                        if not marker.exists() or not (self.state_dir/'writer.enabled').exists():
                            raise RuntimeError('single-writer cutover has not been activated')
                        for name in ('affine-transition-weights', 'affine-hourly-burn'):
                            for suffix in ('timer', 'service'):
                                old = subprocess.run(['systemctl', '--user', 'is-active', f'{name}.{suffix}'],
                                                     capture_output=True, text=True)
                                if old.stdout.strip() in ('active', 'activating', 'reloading'):
                                    raise RuntimeError('another payout writer is still enabled')
                        result = self.chain.execute(intent, wallet, wait_for_inclusion=True,
                                                    wait_for_finalization=True, retries=0)
                        result.raise_for_failure()
                        status.update(status='submitted', block_hash=str(result.block_hash))
                        previous['last_submitted_window'] = window_end
            previous['latest'] = status
            temporary = status_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(previous, indent=2)+'\n')
            temporary.replace(status_path)
            return status
