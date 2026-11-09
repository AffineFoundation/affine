#!/usr/bin/env python3
"""Runs ON the shadow pod (RA/20261102-per-sub-pod-test run2). Streams one software-bundle part (zstd tar) from a presigned R2 GET
with 32 parallel ranged GETs (64 MB chunks, bounded in-order window), hashing in order and piping straight into zstd -d | tar -x
(no temp file). Verifies the sha256 at the end (exit 1 on mismatch). stdin: {"url": ...} (never written to disk).
usage: bundle_fetch.py NAME BYTES SHA256 DEST_ROOT"""
import hashlib, json, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
name, size, sha, dest = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
url = json.load(sys.stdin)['url']; CH = 64 << 20; n = (size + CH - 1) // CH; WIN = 64
def get(i):
    a = i * CH; b = min(size, a + CH) - 1; err = None
    for k in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=%d-%d' % (a, b)}), timeout=180) as r: d = r.read()
            if len(d) == b - a + 1: return d
            err = 'short %d' % len(d)
        except Exception as x: err = type(x).__name__
        time.sleep(1 + 2 * k)
    raise RuntimeError('chunk %d failed: %s' % (i, err))
t0 = time.time(); h = hashlib.sha256(); first = None
p = subprocess.Popen('zstd -dc -T0 | tar -C %s -xpf -' % dest, shell=True, stdin=subprocess.PIPE)
with ThreadPoolExecutor(32) as ex:
    futs = {}; nxt = 0
    for i in range(n):
        while nxt < n and nxt - i < WIN: futs[nxt] = ex.submit(get, nxt); nxt += 1
        d = futs.pop(i).result(); h.update(d); p.stdin.write(d)
        if first is None: first = time.time() - t0
dl = time.time() - t0
p.stdin.close(); rc = p.wait(); tot = time.time() - t0; ok = h.hexdigest() == sha
print(json.dumps(dict(part=name, bytes=size, seconds=round(tot, 1), stream_s=round(dl, 1), first_chunk_s=round(first or 0, 1), gbps=round(size * 8 / tot / 1e9, 2), sha_ok=ok, tar_rc=rc)), flush=True)
sys.exit(0 if ok and rc == 0 else 1)
