import json, sys
from pathlib import Path
G = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/epochs')
def ml(p):
    try:
        d = json.loads(p.read_text()); L = [x for t in d['per_task'].values() for x in (t.get('L') or [])]
        return round(sum(L) / len(L), 1) if L else None
    except Exception: return None
for e in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    p = G / str(e) / 'DECISION.json'
    if not p.exists(): print(e, 'pending'); continue
    d = json.loads(p.read_text()); ev = d.get('evaluated') or {}
    print(e, d.get('decision'), 'gain %+.5f' % ev['delta'] if 'delta' in ev else '-', 'z %+.3f' % ev['z'] if 'z' in ev else '-',
          'cap %.3f' % ev['c_cap_rate'] if 'c_cap_rate' in ev else '-', 'lenC', ml(G / str(e) / 'C-PROBE.private.json'), 'lenH', ml(G / str(e) / 'H-PROBE.private.json'), d.get('decided_at_cr'))
