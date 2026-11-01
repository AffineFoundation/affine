#!/usr/bin/env python3
"""usage: make_gate_config.py REPO_GATE_CONFIG LIVE_FLAG|- OUT  - adds the prior-v1 baseline to the public gate-config copy:
heldout.baseline, heldout.prior_v1 (the full public round settings = baseline_v1.DEFAULTS + the flag's prior_v1, no paths),
heldout.prior_v1_history (from the live flag). Numbers only; no seeds, task ids, prior values or paths."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import baseline_v1 as BV
src, flag, out = sys.argv[1:4]
g = json.load(open(src)); f = json.load(open(flag))['heldout'] if flag != '-' else dict(baseline='prior-v1', prior_v1={}, prior_v1_history=[])
pv = dict(BV.DEFAULTS, **{k: v for k, v in (f.get('prior_v1') or {}).items() if k not in ('table', 'pool_file', 'pool_url', 'rpc')})
pv.pop('stall_s', None); pv.pop('baseline_parallel', None)
g['heldout']['baseline'] = f.get('baseline', 'prior-v1'); g['heldout']['prior_v1'] = pv
g['heldout']['prior_v1_history'] = f.get('prior_v1_history') or []
g['about'] = (g['about'].rstrip('.') + '. heldout.baseline = prior-v1: the round rule is heldout.prior_v1 (priors/baseline_v1.py); '
              'the ivw-v2 keys (n, k_cand, sigma_run*, min_kept) apply only when baseline is not prior-v1.')
s = json.dumps(g, indent=1, sort_keys=True) + '\n'
for bad in ('"pi"', '"secret"', '"seed"', '/home/', '.sqlite'): assert bad not in s, bad
open(out, 'w').write(s); print('wrote', out, json.dumps(pv))
