#!/usr/bin/env python3
"""Check a closed SN120 prior-v1 round reveal (https://affine.io/gate/heldout/<round>.json) without trusting the validator:
  1. sha256(version|secret|<secret>) == the secret commitment published when the round opened;
  2. the hash of block B (from a public Finney RPC) == the revealed block hash;
  3. seed == sha256(version|seed|secret|B|hash(B));
  4. draw_commitment == sha256(version|seed|sorted drawn ids), and every drawn id is in the public gate pool.
usage: verify_round.py ROUND_JSON_OR_URL [GATE_POOL_JSONL] [--rpc URL]"""
import hashlib, json, sys, urllib.request
sys.path.insert(0, __import__('os').path.dirname(__file__) + '/..')
from baseline_v1 import VERSION, secret_commitment, seed_from, draw_commitment
args = [a for a in sys.argv[1:] if not a.startswith('--')]; rpc = 'https://entrypoint-finney.opentensor.ai:443'
if '--rpc' in sys.argv: rpc = sys.argv[sys.argv.index('--rpc') + 1]; args = [a for a in args if a != rpc]
src = args[0]
r = json.load(urllib.request.urlopen(src)) if src.startswith('http') else json.load(open(src))
def block_hash(n):
    q = json.dumps(dict(jsonrpc='2.0', id=1, method='chain_getBlockHash', params=[int(n)])).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(rpc, q, {'Content-Type': 'application/json'}), timeout=30))['result']
ok = {}
ok['secret_commitment'] = secret_commitment(r['secret']) == r['secret_commitment']
ok['block_hash_on_chain'] = block_hash(r['block_number']).lower() == r['block_hash'].lower()
ok['seed'] = seed_from(r['secret'], r['block_number'], r['block_hash']) == r['seed']
ok['draw_commitment'] = draw_commitment(r['seed'], r['drawn_ids']) == r['commitment']
if len(args) > 1:
    pool = {json.loads(l)['id'] for l in open(args[1])}
    ok['ids_in_pool'] = all(i in pool for i in r['drawn_ids'])
    ok['pool_sha256'] = hashlib.sha256(open(args[1], 'rb').read()).hexdigest() == r['pool']['sha256']
print(json.dumps(dict(round=r['round_id'], version=VERSION, drawn=len(r['drawn_ids']), checks=ok), indent=1))
sys.exit(0 if all(ok.values()) else 1)
