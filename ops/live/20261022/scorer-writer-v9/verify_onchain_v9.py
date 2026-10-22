"""READ-ONLY chain verification for one submitted hour of the v9-only weight path (no wallet, no writes except the report).
usage: verify_onchain_v9.py <cutoff>  -> prints/writes live-log/onchain-<cutoff>.json
Checks: owner UID == 0 + hotkey, LastUpdate freshness, the hour's timelocked commit extrinsic included, the journaled intent vector
(== the v9 rule output for that hour: UID 0 burn + gated miners), and whether the currently revealed weights equal it."""
from pathlib import Path
import importlib.util, json, time, signal, sys, hashlib
O = Path(__file__).resolve().parent; sys.path.insert(0, '/home/const/subnet120-mimo-discovery-fix')
spec = importlib.util.spec_from_file_location('_w9', str(O / 'weights_v9.py')); w9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(w9)
_, wsr = w9.isolated_journal_modules()          # v9 copy: accepts the UID 0 burn weight in journaled intents
from subnet.chain import OWNER
import bittensor as bt
Q = Path('/home/const/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1')
def expire(*_): raise TimeoutError('readonly180s')
signal.signal(signal.SIGALRM, expire); signal.alarm(180)
c = int(sys.argv[1])
s = bt.subtensor(network='finney', retry_forever=False); r = wsr.BittensorReadonlyReader(s); head = r.finalized_head()
uid = r._query('Uids', [120, OWNER], head['hash']); hot = r._query('Keys', [120, uid], head['hash'])
lu = r._query('LastUpdate', [120], head['hash'])[uid]
state = r.owner_state(dict(netuid=120, owner=OWNER, owner_uid=uid, mecid=0), head)
actual = sorted([int(u), int(w)] for u, w in state.pop('weights') if int(w) > 0)
res = dict(observed_at=time.time(), finalized_head=head.get('number'), owner_uid=uid, owner_hotkey_is_OWNER=hot == OWNER,
           last_update_block=lu, last_update_age_blocks=head.get('number') - lu if isinstance(head.get('number'), int) else None,
           revealed_positive_uids=len(actual), revealed_uid0_share=(dict(actual).get(0, 0) / sum(w for _, w in actual)) if actual else None,
           revealed_sha=hashlib.sha256(json.dumps(actual).encode()).hexdigest()[:10], pending=state.get('owner_pending'), attempts=[])
for p in sorted((Q / 'current-assessment-v1/attempts').glob(str(c) + '-*')):
    intent = wsr.validate_intent(json.loads((p / 'intent.json').read_text()))
    obs = json.loads((p / 'signed-observation.json').read_text()); obs = obs.get('payload', obs); commit = None
    for n in range(intent['attempt_start_block'] - 2, intent['attempt_start_block'] + 10):
        b = r.block(n)
        for v in b['commits']:
            if v['extrinsic_hash'] == obs['extrinsic_hash']: commit = dict(v, inclusion_block=n); break
        if commit: break
    expected = sorted([int(u), int(w)] for u, w in intent['expected_weights'] if int(w) > 0)
    tot = sum(w for _, w in expected)
    res['attempts'].append(dict(attempt=p.name, commit_included=commit is not None, inclusion_block=(commit or {}).get('inclusion_block'),
                                intended_positive_uids=len(expected), intended_uid0_share=dict(expected).get(0, 0) / tot if tot else None,
                                intended_sha=hashlib.sha256(json.dumps(expected).encode()).hexdigest()[:10], currently_revealed=expected == actual))
lg = O / 'live-log' / f'weights-{c}.json'
if lg.exists():
    d = json.loads(lg.read_text()); res['v9_rule'] = {k: d.get(k) for k in ('recipients', 'miners_paid', 'burn_share', 'digest', 'v9')}
    cfg = w9.load_config(str(O / 'LIVE-CONFIG-v9.ACTIVE.signed.json'), '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd')
    sh, burn, info = w9.v9_vector(cfg, (d.get('v9') or {}).get('latest_sealed'))
    res['v9_recomputed'] = dict(miners_paid=len(sh), burn_share=burn, reasons=info['reasons'],
                                matches_logged=abs(burn - d.get('burn_share', -1)) < 1e-9 and len(sh) == d.get('miners_paid'))
(O / 'live-log').mkdir(exist_ok=True, mode=0o700)
(O / 'live-log' / f'onchain-{c}.json').write_text(json.dumps(res, indent=1, default=str)); print(json.dumps(res, default=str)); s.close()
