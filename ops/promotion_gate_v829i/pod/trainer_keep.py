"""Runs ON the trainer pod (golden-raven-7a). Promotion gate (Jacob pt682u 2026-10-26).
keep H    : while pod trainer CURRENT == H, hardlink H's committed generation (model files) + optimizer marker + CURRENT.json
            into private/promotion-gate/keep/H/ so a RED (candidate discarded) can restore the trainer to H after the
            trainer prunes H's model on committing C (retain_committed_generations=1). Hardlinks: no copy, no extra disk
            until the prune. Keeps the newest 3 keeps.
restore H : (worker must be stopped) put H's generation files back and install H's CURRENT.json (backup kept).
unlink H  : (worker must be stopped) replace hardlinks in an existing keep with private copies.
2026-10-27 (t694): keeps and restores now COPY instead of hardlinking. The owned worker's CheckpointGuard pins
(dev, ino, size, mtime, ctime) of the live checkpoint files; adding or removing a hardlink changes the inode ctime,
so hardlinking the just-promoted checkpoint failed the next job with "checkpoint local state changed: config.json"
(e154 after the first GREEN). Copies never touch the live inodes. Old keeps whose head is the pod CURRENT are never pruned.
Same method as rollback2-e141/restore_e141_pod.py, from local hardlinks instead of a download. Trainer math untouched."""
import json, os, shutil, sys, time
def cp(src, dst):  # private copy; never adds a link to (and so never changes the ctime of) the live inode
    shutil.copyfile(src, dst); shutil.copystat(src, dst)
from pathlib import Path
R = Path('/workspace/affine-mimo-trainer-prod/trainer-state/affine-mimo9b-grpo-20261013')
K = Path('/workspace/affine-mimo-trainer-prod/private/promotion-gate/keep')
def out(**kw): print(json.dumps(kw)); sys.exit(0 if kw.get('ok') else 1)
def gen_for(H):
    for g in sorted((R / 'history').glob('update-*')):
        try: t = json.loads((g / 'TRANSACTION.json').read_bytes())
        except Exception: continue
        if t.get('state') == 'published' and t.get('output_checkpoint') == H: return g, t
    return None, None
cmd, H = sys.argv[1], sys.argv[2]; assert len(H) == 64
kd = K / H
if cmd == 'keep':
    if (kd / 'KEEP.json').exists(): out(ok=True, already=True, **json.loads((kd / 'KEEP.json').read_text()))
    cur = json.loads((R / 'CURRENT.json').read_bytes())
    if cur.get('checkpoint') != H: out(ok=False, error='pod CURRENT is %s, not H' % str(cur.get('checkpoint'))[:12])
    g, t = gen_for(H)
    if g is None or not (g / 'model').is_dir() or not (g / 'optimizer' / 'NOT-PERSISTED.json').is_file():
        out(ok=False, error='no committed generation with model for H', gen=str(g))
    arts = t['artifacts']
    for k, a in arts.items():
        if k.startswith('model/'): assert (g / k).stat().st_size == a['size'], k
    tmp = K / (H + '.tmp'); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True, mode=0o700)
    (tmp / 'model').mkdir()
    for p in (g / 'model').iterdir(): cp(p, tmp / 'model' / p.name)
    for k, a in arts.items():
        if k.startswith('model/'): assert (tmp / k).stat().st_size == a['size'], k
    shutil.copy2(g / 'optimizer' / 'NOT-PERSISTED.json', tmp / 'NOT-PERSISTED.json')
    shutil.copy2(R / 'CURRENT.json', tmp / 'CURRENT.json')
    meta = dict(head=H, gen=g.name, optimizer_step=cur.get('optimizer_step'), kept_at=time.time())
    (tmp / 'KEEP.json').write_text(json.dumps(meta)); os.replace(tmp, kd)
    live = cur.get('checkpoint')
    for old in sorted((p for p in K.iterdir() if p.is_dir() and len(p.name) == 64 and p.name != live), key=lambda p: p.stat().st_mtime)[:-2]:
        shutil.rmtree(old, ignore_errors=True)
    out(ok=True, **meta)
elif cmd == 'restore':
    meta = json.loads((kd / 'KEEP.json').read_text()); g = R / 'history' / meta['gen']
    t = json.loads((g / 'TRANSACTION.json').read_bytes()); assert t['output_checkpoint'] == H
    cur = json.loads((R / 'CURRENT.json').read_bytes())
    if cur.get('checkpoint') == H: out(ok=True, already=True, gen=meta['gen'])
    if not (g / 'model').is_dir():
        tmp = g / 'model.gate-tmp'; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(mode=0o700)
        for p in (kd / 'model').iterdir(): cp(p, tmp / p.name)
        for k, a in t['artifacts'].items():
            if k.startswith('model/'): assert (tmp / k[6:]).stat().st_size == a['size'], k
        os.rename(tmp, g / 'model')
    if not (g / 'optimizer' / 'NOT-PERSISTED.json').is_file():
        (g / 'optimizer').mkdir(mode=0o755, exist_ok=True); shutil.copy2(kd / 'NOT-PERSISTED.json', g / 'optimizer' / 'NOT-PERSISTED.json')
    ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    shutil.copy2(R / 'CURRENT.json', R / ('CURRENT.json.pre-gate-red-' + ts))
    tmpc = R / 'CURRENT.json.gate-tmp'; shutil.copy2(kd / 'CURRENT.json', tmpc); os.chmod(tmpc, 0o600); os.replace(tmpc, R / 'CURRENT.json')
    out(ok=True, restored=H, gen=meta['gen'], backup='CURRENT.json.pre-gate-red-' + ts)
elif cmd == 'unlink':
    n = 0
    for p in sorted((kd / 'model').iterdir()):
        if p.stat().st_nlink > 1:
            t = p.with_name(p.name + '.copy-tmp'); cp(p, t); assert t.stat().st_size == p.stat().st_size; os.replace(t, p); n += 1
    out(ok=True, unlinked=n, keep=str(kd))
elif cmd == 'status':
    out(ok=(kd / 'KEEP.json').exists(), keep=str(kd))
