#!/usr/bin/env python3
import json,hashlib,os,sys
from pathlib import Path
N=Path(sys.argv[1]); RA=Path('/home/const/subnet120-rewrite/state/root-audits')
old=json.loads((RA/'20261013-null-pow-v1/dashboard-uid1000/config.REVIEW.private.json').read_bytes())
RT=RA/'20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822'
LC=RA/'20261016-v823/release-v823-signed/lifecycle.v8.2.3.ACTIVE.private.json'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pin=lambda p:{'path':str(p),'sha256':h(p)}
def files(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.is_symlink())
# Cross-check the runtime tree against the v8.2.2 signed manifest where it lists the file.
signed={}
for line in (RT.parent/'SIGNED-MANIFEST.sha256').read_text().splitlines():
    parts=line.split(None,1)
    if len(parts)==2: signed[parts[1].lstrip('*').strip()]=parts[0]
rt=files(RT); checked=mismatch=0
for p in rt:
    for key in (str(p), str(p.relative_to(RT.parent)), './'+str(p.relative_to(RT.parent))):
        if key in signed:
            checked+=1
            if signed[key]!=h(p): mismatch+=1
            break
print('runtime files',len(rt),'checked vs signed manifest',checked,'mismatch',mismatch)
if mismatch: raise SystemExit('runtime tree differs from signed manifest')
lc=json.loads(LC.read_bytes())
sys.path.insert(0,str(RT))
from subnet.mimo_release_succession import ReleaseChain
chain=ReleaseChain(lc['release_succession'],old['authority'],allow_public=True)
cfg=dict(version='mimo-dashboard-readonly-projection-v2-release-succession',authority=old['authority'],
    legacy_server=pin(N/'source/dashboard/server.py'),runtime_root=str(RT),
    runtime_files=[pin(p) for p in rt],projection_files=[pin(p) for p in files(N/'source')],
    lifecycle_config=pin(LC),release_chain_sha256=chain.chain_sha256,
    prior_snapshot=old['prior_snapshot'],public_root=old['public_root'],started_at=old['started_at'],
    model=old['model'],history_export_path=old['history_export_path'])
out=N/'config.ROOT-REVIEW.private.json'
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f: json.dump(cfg,f,indent=1,sort_keys=True)
print('wrote',out,'sha256',h(out),'chain',chain.chain_sha256[:12],'members',[(r,p.policy_sha256[:8]) for r,p in chain.members])
