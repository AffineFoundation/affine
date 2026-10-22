#!/usr/bin/env python3
# Derive the v8.2.4 dashboard config from the 20261017 one: same runtime/chain/authority,
# lifecycle -> live v8.2.4 ACTIVE, + ROOT-admitted v8.2.4 host publication validator, repinned source.
import json,hashlib,os,sys
from pathlib import Path
N=Path(sys.argv[1]); RA=Path('/home/const/subnet120-rewrite/state/root-audits')
old=json.loads((RA/'20261017-dashboard-succession-v1/config.ROOT-REVIEW.private.json').read_bytes())
V=RA/'20261017-v824/release-v824-signed'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pin=lambda p:{'path':str(p),'sha256':h(p)}
files=lambda root:sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.is_symlink())
LC=V/'lifecycle.v8.2.4.ACTIVE.private.json'
op=json.loads((V/'uid-operator.v8.2.4.ROOT-ADMITTED.private.json').read_bytes())
cap=json.loads(Path(op['parent_operator']['path']).read_bytes())
assert cap['base']['lifecycle_config']=={'path':str(LC),'sha256':h(LC)}, 'live lifecycle config pin'
cfg=dict(old,version='mimo-dashboard-readonly-projection-v3-v824-publication',
    legacy_server=pin(N/'source/dashboard/server.py'),projection_files=[pin(p) for p in files(N/'source')],
    lifecycle_config=pin(LC),
    host_validator=dict(operator_config=pin(V/'uid-operator.v8.2.4.ROOT-ADMITTED.private.json'),helper=pin(V/'mimo_uid_capacity.py')))
for r in cfg['runtime_files']: assert h(Path(r['path']))==r['sha256'], r['path']
out=N/'config.ROOT-REVIEW.private.json'
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f: json.dump(cfg,f,indent=1,sort_keys=True)
print('wrote',out,'sha256',h(out))
