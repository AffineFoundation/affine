"""Compare merged replay shards with live e76: capture records and the seeded 256 selection. Aggregates only."""
import json, sys, glob
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
from pathlib import Path
EPOCH = int(sys.argv[1])
SRC = '/home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822'
sys.path.insert(0, SRC)
from subnet.storage import canonical, sha
from subnet.backend_jobs import signed
live = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs/%d/coordinator' % EPOCH)
opening = json.loads((live/'OPENING.json').read_bytes()); AUTH = opening['signer']
manifest = signed(opening, AUTH)
out = {}

livecap = {p.stem: p for p in (live/'captured').glob('*.json')}
newcap = {Path(p).stem: Path(p) for p in glob.glob('/tmp/v827-replay-e%d-s*/captured/*.json' % EPOCH)}
strip = lambda d: {k: v for k, v in d.items() if k != 'new_groups'}
bytes_same = sum(newcap[m].read_bytes() == livecap[m].read_bytes() for m in newcap if m in livecap)
payload_same = sum(strip(json.loads(newcap[m].read_bytes())['payload']) == strip(json.loads(livecap[m].read_bytes())['payload'])
                   for m in newcap if m in livecap)
out['captures'] = dict(live=len(livecap), replay=len(newcap), same_miner_set=set(livecap) == set(newcap),
                       byte_identical_records=bytes_same, identical_except_new_groups=payload_same)
# --- seeded selection from the replayed population (live native grades reused; grading is deterministic per batch)
dispatch = signed(json.loads((live/'DISPATCH.json').read_bytes()), AUTH)
policy_sel = json.loads((live/'SELECTION.json').read_bytes())
docs = {}
for m, p in newcap.items():
    cap = signed(json.loads(p.read_bytes()), AUTH)
    for d in cap['documents']: docs[(m, d['key'])] = d
graded_ok = [r for r in dispatch['selected']] + [e for e in dispatch['excluded'] if e['reason'] in ('training-task-budget', 'cross-miner-task-collision')]
cands = [dict({k: v for k, v in r.items() if k != 'reason'}) for r in graded_ok]
missing = [r for r in cands if docs.get((r['miner'], r['document']['key'])) != r['document']]
cands = [r for r in cands if docs.get((r['miner'], r['document']['key'])) == r['document']]
from subnet.claim_order import enabled, resolve
caps = {m: signed(json.loads(p.read_bytes()), AUTH) for m, p in newcap.items()}
if enabled(manifest):
    resolved = resolve((r['miner'], (r['env_id'], r['index']), r['document'].get('received_at', caps[r['miner']].get('received_at'))) for r in cands)
    winners = {k: v[0] for k, v in resolved.items()}
    unique = [r for r in cands if winners[(r['env_id'], r['index'])] == r['miner']]
else:
    from collections import Counter
    c = Counter((r['env_id'], r['index']) for r in cands); unique = [r for r in cands if c[(r['env_id'], r['index'])] == 1]
unique.sort(key=lambda row: sha(canonical([policy_sel['seed'], row['miner'], row['env_id'], row['index']])))
sel = unique[:dispatch['selection']['max_tasks']]
out['selection'] = dict(live_selected=len(dispatch['selected']), replay_selected=len(sel), candidates_missing_from_replay=len(missing),
                        identical=canonical(sel) == canonical(dispatch['selected']),
                        live_sha256=sha(canonical(dispatch['selected'])), replay_sha256=sha(canonical(sel)))
print(json.dumps(out, indent=1))
json.dump(out, open('/home/const/subnet120-rewrite/state/root-audits/20261021-v827-incremental-capture/replay/COMPARE-e%d.json' % EPOCH, 'w'), indent=1)
