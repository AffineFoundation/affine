"""READ-ONLY chain verification for one submitted hour (no wallet, no writes except the report file).
usage: verify_onchain.py <cutoff>   -> prints/writes onchain-<cutoff>.json
Checks: owner UID + hotkey, LastUpdate freshness, the hour's commit extrinsic included & successful, currently revealed
weights vs this hour's intended u16 vector, pending commits, and the v7c shadow log (returned == original digest)."""
from pathlib import Path
import json, time, signal, sys, hashlib
O = Path("/home/const/subnet120-rewrite/state/root-audits/20261019-measured-impact-v1"); sys.path.insert(0, '/home/const/subnet120-mimo-discovery-fix')
from subnet.weight_submission_reconciliation import BittensorReadonlyReader, validate_intent
from subnet.chain import OWNER
import bittensor as bt
Q = Path('/home/const/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1')
def expire(*_): raise TimeoutError('readonly180s')
signal.signal(signal.SIGALRM, expire); signal.alarm(180)
c = int(sys.argv[1])
s = bt.subtensor(network='finney', retry_forever=False); r = BittensorReadonlyReader(s); head = r.finalized_head()
uid = r._query('Uids', [120, OWNER], head['hash']); hot = r._query('Keys', [120, uid], head['hash'])
lu = r._query('LastUpdate', [120], head['hash'])[uid]
state = r.owner_state(dict(netuid=120, owner=OWNER, owner_uid=uid, mecid=0), head)
actual = sorted([int(u), int(w)] for u, w in state.pop('weights') if int(w) > 0)
res = dict(observed_at=time.time(), finalized_head=head.get('number'), owner_uid=uid, owner_hotkey=hot, owner_is_OWNER=hot == OWNER,
           last_update_block=lu, last_update_age_blocks=head.get('number') - lu if isinstance(head.get('number'), int) else None,
           revealed_positive_uids=len(actual), revealed_sha=hashlib.sha256(json.dumps(actual).encode()).hexdigest()[:10],
           pending=state.get('owner_pending'), attempts=[])
for p in sorted((Q / 'current-assessment-v1/attempts').glob(str(c) + '-*')):
    idoc = json.loads((p / 'intent.json').read_text()); intent = validate_intent(idoc)
    obs = json.loads((p / 'signed-observation.json').read_text()); obs = obs.get('payload', obs); commit = None
    for n in range(intent['attempt_start_block'] - 2, intent['attempt_start_block'] + 10):
        b = r.block(n)
        for v in b['commits']:
            if v['extrinsic_hash'] == obs['extrinsic_hash']: commit = dict(v, inclusion_block=n); break
        if commit: break
    expected = sorted([int(u), int(w)] for u, w in intent['expected_weights'] if int(w) > 0)
    res['attempts'].append(dict(attempt=p.name, commit=commit, intended_positive_uids=len(expected),
                                intended_sha=hashlib.sha256(json.dumps(expected).encode()).hexdigest()[:10], currently_revealed=expected == actual))
sh = O / 'shadow' / f'blend-{c}.json'
if sh.exists():
    d = json.loads(sh.read_text())
    res['shadow'] = {k: d.get(k) for k in ('blend_configured', 'applied', 'kill_reasons', 'original_digest', 'returned_digest', 'evidence_seconds',
                                           'writer_seconds_remaining', 'blend_ms', 'epochs_used', 'latest_sealed', 'shadow_blend', 'top_gain')}
    res['shadow']['returned_equals_original'] = d.get('original_digest') == d.get('returned_digest')
print(json.dumps(res, default=str)); s.close()
