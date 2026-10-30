"""Hardlink byte-identical large files inside plain-disk copies (/opt/plain/...) to save space. Pod side.
usage: python3 pd_dedupe.py [--apply] [--min-mb 64] <dir under /opt/plain> ...
Groups regular files by size, then sha256; files in a group with the same sha256, mode, uid and gid become hardlinks to
the first one (link to a temp name in the same dir, then rename over: atomic). Content stays byte-identical; only inode
sharing changes, so use it only on cold copies nobody rewrites in place (e.g. content-addressed checkpoint shards).
Skips files any process holds (cwd/fd/mmap). Dry run without --apply. Prints TSV: action, bytes, sha256, kept, replaced."""
import os, sys, hashlib, collections
from concurrent.futures import ThreadPoolExecutor
args = [a for a in sys.argv[1:]]
apply = '--apply' in args; args = [a for a in args if a != '--apply']
mb = 64
if '--min-mb' in args:
    i = args.index('--min-mb'); mb = int(args[i + 1]); del args[i:i + 2]
roots = [os.path.realpath(a) for a in args]
for r in roots:
    if not r.startswith('/opt/plain/') or not os.path.isdir(r): sys.exit('refuse: %s is not a dir under /opt/plain' % r)
def held_paths():
    s = set()
    for p in os.listdir('/proc'):
        if not p.isdigit(): continue
        try:
            for f in os.listdir('/proc/%s/fd' % p):
                try: s.add(os.readlink('/proc/%s/fd/%s' % (p, f)))
                except OSError: pass
            with open('/proc/%s/maps' % p) as m:
                for l in m:
                    x = l.split(None, 5)
                    if len(x) == 6: s.add(x[5].strip())
        except OSError: pass
    return s
held = held_paths()
by_size = collections.defaultdict(list)
for r in roots:
    for d, _, fs in os.walk(r):
        for f in fs:
            p = os.path.join(d, f); st = os.lstat(p)
            if os.path.islink(p) or not os.path.isfile(p) or st.st_size < mb << 20: continue
            by_size[st.st_size].append((p, st))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(1 << 22): h.update(b)
    return h.hexdigest()
cands = [x for v in by_size.values() if len({s.st_ino for _, s in v}) > 1 for x in v]
with ThreadPoolExecutor(8) as ex: hs = dict(zip([p for p, _ in cands], ex.map(sha, [p for p, _ in cands])))
groups = collections.defaultdict(list)
for p, st in cands: groups[(st.st_size, hs[p], st.st_mode, st.st_uid, st.st_gid)].append((p, st))
saved = 0
for (size, h, *_), v in sorted(groups.items()):
    v.sort(); keep, kst = v[0]
    for p, st in v[1:]:
        if st.st_ino == kst.st_ino: continue
        if p in held: print('skip-held\t%d\t%s\t%s\t%s' % (size, h, keep, p)); continue
        if apply:
            if sha(p) != h or sha(keep) != h: print('skip-changed\t%d\t%s\t%s\t%s' % (size, h, keep, p)); continue
            tmp = p + '.pd-dedupe-tmp'; os.link(keep, tmp); os.rename(tmp, p)
        saved += size; print('%s\t%d\t%s\t%s\t%s' % ('linked' if apply else 'would-link', size, h, keep, p))
print('total\t%d\t%s' % (saved, 'applied' if apply else 'dry-run'))
