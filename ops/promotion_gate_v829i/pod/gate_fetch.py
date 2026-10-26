"""Promotion gate (SN120): parallel materialization of a checkpoint on the probe pod.
usage: gate_fetch.py DESCRIPTOR.json DEST_ROOT URLS.private.json [--threads N] [--pidfile PATH]
Same on-disk result as fetch_ckpt.py (DEST_ROOT/<id>/ + .verified, sha256-checked against the signed descriptor);
files are downloaded with parallel ranged GETs (R2 presigned GET only). The URL file is deleted afterwards."""
import hashlib, json, os, shutil, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
args = sys.argv[1:]
threads = int(args[args.index('--threads') + 1]) if '--threads' in args else 32
pidfile = args[args.index('--pidfile') + 1] if '--pidfile' in args else None
desc = json.load(open(args[0])); dest_root = args[1]; uf = args[2]; urls = json.load(open(uf))
if pidfile: open(pidfile, 'w').write(str(os.getpid()))
cid = desc['id']; dest = os.path.join(dest_root, cid); tmp = dest + '.partial'
try:
    if os.path.exists(os.path.join(dest, '.verified')):
        print(json.dumps(dict(ckpt=cid, status='present'))); sys.exit(0)
    os.makedirs(tmp, exist_ok=True); t0 = time.time()
    CH = 64 << 20
    def size_of(url):
        req = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
        with urllib.request.urlopen(req, timeout=120) as r:
            cr = r.headers.get('Content-Range')
            return int(cr.split('/')[-1]) if cr else int(r.headers.get('Content-Length'))
    jobs = []
    for name in sorted(desc['files']):
        out = os.path.join(tmp, name)
        n = size_of(urls[name])
        with open(out, 'wb') as f: f.truncate(n)
        for off in range(0, n, CH): jobs.append((name, out, off, min(n, off + CH) - 1))
    def get(job):
        name, out, a, b = job
        for attempt in range(6):
            try:
                req = urllib.request.Request(urls[name], headers={'Range': 'bytes=%d-%d' % (a, b)})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if len(data) != b - a + 1: raise IOError('short read')
                fd = os.open(out, os.O_WRONLY); os.pwrite(fd, data, a); os.close(fd); return
            except Exception:
                time.sleep(2 + 3 * attempt)
        raise SystemExit('range failed %s %d-%d' % (name, a, b))
    with ThreadPoolExecutor(threads) as ex: list(ex.map(get, jobs))
    t_dl = time.time() - t0
    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 24), b''): h.update(blk)
        return p, h.hexdigest()
    with ThreadPoolExecutor(16) as ex: got = dict(ex.map(sha, [os.path.join(tmp, n) for n in desc['files']]))
    bad = [n for n, w in desc['files'].items() if got[os.path.join(tmp, n)] != w]
    if bad: raise SystemExit('sha256 mismatch: %s' % bad)
    open(os.path.join(tmp, '.verified'), 'w').write(json.dumps(dict(files=desc['files'], at=time.time(), by='promotion-gate')))
    if os.path.exists(dest): shutil.rmtree(dest)
    os.replace(tmp, dest)
    print(json.dumps(dict(ckpt=cid, status='materialized', seconds=round(time.time() - t0, 1), download_s=round(t_dl, 1),
                          sources=['r2-presigned-get-parallel'], threads=threads)))
finally:
    try: os.remove(uf)
    except Exception: pass
    if pidfile:
        try: os.remove(pidfile)
        except Exception: pass
