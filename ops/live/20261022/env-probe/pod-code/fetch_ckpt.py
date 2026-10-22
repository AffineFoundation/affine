"""Materialize a checkpoint for the probe: copy from the evaluator's read-only cache if present, else download
from presigned read URLs (R2 GET only). Every file is sha256-verified against the signed checkpoint descriptor."""
import hashlib, json, os, shutil, sys, urllib.request, time
desc = json.load(open(sys.argv[1])); dest_root = sys.argv[2]; urls = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else {}
CACHE = '/root/affine-mimo-public-evaluator-20261013/workspace/mimo-cache'
cid = desc['id']; dest = os.path.join(dest_root, cid); tmp = dest + '.partial'
if os.path.exists(os.path.join(dest, '.verified')): print(json.dumps(dict(ckpt=cid, status='present'))); sys.exit(0)
os.makedirs(tmp, exist_ok=True); t0 = time.time(); src_kind = {}
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
for name, want in sorted(desc['files'].items()):
    out = os.path.join(tmp, name)
    if os.path.exists(out) and sha(out) == want: src_kind[name] = 'resumed'; continue
    src = os.path.join(CACHE, cid, name); ok = False
    if os.path.exists(src):
        shutil.copyfile(src, out); ok = sha(out) == want; src_kind[name] = 'evaluator-cache'
    if not ok and name in urls:
        with urllib.request.urlopen(urls[name], timeout=600) as r, open(out, 'wb') as f: shutil.copyfileobj(r, f, 1 << 24)
        ok = sha(out) == want; src_kind[name] = 'r2-presigned-get'
    if not ok: raise SystemExit(f'cannot materialize {name} with sha256 {want}')
open(os.path.join(tmp, '.verified'), 'w').write(json.dumps(dict(files=desc['files'], at=time.time())))
os.replace(tmp, dest)
print(json.dumps(dict(ckpt=cid, status='materialized', seconds=round(time.time() - t0, 1), sources=sorted(set(src_kind.values())))))
