"""Download + authenticate a published checkpoint from the read-only index.

Every file is streamed from its presigned GET URL, sha256-checked against the
signed filemap, and the filemap itself must hash to the checkpoint id. Files land
in <cache>/<id>/ ; a READY marker is written only after every hash matches.
Concurrent instances on one host share the cache safely (per-checkpoint lock).
"""
import fcntl, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _get(url, path, expected, chunk=8 << 20):
    tmp = path.with_suffix(path.suffix + '.part')
    h = hashlib.sha256()
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, 'wb') as f:
                    for b in r.iter_content(chunk):
                        f.write(b); h.update(b)
            break
        except Exception:
            if attempt == 4:
                raise
            h = hashlib.sha256(); time.sleep(2 + 3 * attempt)
    if h.hexdigest() != expected:
        tmp.unlink(missing_ok=True)
        raise ValueError(f'hash mismatch for {path.name}')
    os.replace(tmp, path)


def ensure(cp, cache):
    """Return local dir of an authenticated checkpoint (downloading if needed)."""
    files = cp['files']
    if hashlib.sha256(canonical(files)).hexdigest() != cp['checkpoint_id']:
        raise ValueError('filemap does not hash to checkpoint id')
    d = Path(cache) / cp['checkpoint_id']
    d.mkdir(parents=True, exist_ok=True)
    if (d / 'READY').exists():
        return d, 0.0
    t0 = time.time()
    with open(d / '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if (d / 'READY').exists():
            return d, time.time() - t0
        if cp['read_urls_expire_at'] and cp['read_urls_expire_at'] < time.time():
            raise RuntimeError('presigned URLs expired and checkpoint not cached; re-export index')
        todo = [n for n in files if not (d / n).exists()]
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda n: _get(cp['read_urls'][n], d / n, files[n]), todo))
        # re-hash files that were already present (from an interrupted run)
        for n in set(files) - set(todo):
            h = hashlib.sha256()
            with open(d / n, 'rb') as f:
                for b in iter(lambda: f.read(64 << 20), b''):
                    h.update(b)
            if h.hexdigest() != files[n]:
                (d / n).unlink(); raise ValueError('cached file corrupted: ' + n)
        (d / 'READY').write_text(json.dumps(dict(id=cp['checkpoint_id'], t=time.time())))
    return d, time.time() - t0


if __name__ == '__main__':
    idx = json.loads(Path(sys.argv[1]).read_text())
    want = sys.argv[3:]
    for cp in idx['checkpoints']:
        if any(cp['checkpoint_id'].startswith(w) for w in want):
            d, s = ensure(cp, sys.argv[2]); print(cp['checkpoint_id'][:8], d, f'{s:.1f}s', flush=True)
