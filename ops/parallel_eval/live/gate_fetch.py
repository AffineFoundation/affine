"""Promotion gate (SN120): parallel materialization of a checkpoint on the probe pod.
usage: gate_fetch.py DESCRIPTOR.json DEST_ROOT URLS.private.json [--threads N] [--pidfile PATH]
Same on-disk result as fetch_ckpt.py (DEST_ROOT/<id>/ + .verified, sha256-checked against the signed descriptor);
files are downloaded with parallel ranged GETs (R2 presigned GET only). The URL file is deleted afterwards.

Retries: each request (the size request of a file and each range) is tried again after an error, with a pause that
doubles. GATE_FETCH_ATTEMPTS is the number of tries (default 10). The pause starts at GATE_FETCH_BACKOFF_BASE seconds
(default 2) and stops at GATE_FETCH_BACKOFF_MAX seconds (default 60). An HTTP answer that a new try cannot change
(400, 401, 403, 404, 416) stops at once. The first request that fails for good stops the other threads.

One fetch of an id at a time: a second process for the same id waits for the first, then finds the result.

Output: the last line of stdout is one JSON record. Success: status "present" or "materialized", exit code 0.
Failure: status "failed" with the real error ("error", "errno", "errno_name", "attempts", "retryable"), exit code 1.
The record has no URL in it."""
import errno, fcntl, hashlib, json, os, random, re, shutil, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

ATTEMPTS = max(1, int(os.environ.get('GATE_FETCH_ATTEMPTS') or 10))
BACKOFF_BASE = float(os.environ.get('GATE_FETCH_BACKOFF_BASE') or 2)
BACKOFF_MAX = float(os.environ.get('GATE_FETCH_BACKOFF_MAX') or 60)
PERMANENT_HTTP = (400, 401, 403, 404, 416)
CHUNK = 64 << 20
URL_TEXT = re.compile(r'https?://\S+')


class FetchError(Exception):
    """A request that failed for good. ``cause`` is the last error of the request."""

    def __init__(self, what, cause, attempts):
        Exception.__init__(self, what)
        self.what, self.cause, self.attempts = what, cause, attempts


class Stopped(Exception):
    """A different request failed for good; this one did not finish."""


def root_cause(exc):
    """The network error under a urllib wrapper (URLError.reason), else the error itself."""
    reason = getattr(exc, 'reason', None)
    return reason if isinstance(reason, BaseException) else exc


def describe(exc):
    """The class and the text of an error, with no URL in it. Example: URLError: OSError: [Errno 101] Network is unreachable."""
    inner = root_cause(exc)
    text = '%s: %s' % (type(inner).__name__, inner)
    if inner is not exc:
        text = '%s: %s' % (type(exc).__name__, text)
    return URL_TEXT.sub('<url>', text)


def permanent(exc):
    return isinstance(exc, urllib.error.HTTPError) and exc.code in PERMANENT_HTTP


def failure_record(cid, exc):
    """The JSON record of a fetch that failed."""
    cause, attempts, what = exc, 1, None
    if isinstance(exc, FetchError):
        cause, attempts, what = exc.cause, exc.attempts, exc.what
    number = getattr(root_cause(cause), 'errno', None)
    text = describe(cause) if what is None else '%s: %s' % (what, describe(cause))
    return dict(ckpt=cid, status='failed', error=text[:600], errno=number if isinstance(number, int) else None,
                errno_name=errno.errorcode.get(number) if isinstance(number, int) else None, attempts=attempts,
                retryable=isinstance(exc, FetchError) and not permanent(cause))


class Fetcher:
    def __init__(self, urls, threads):
        self.urls, self.threads = urls, threads
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.failure = None          # the first request that failed for good

    def pause(self, attempt):
        """Seconds before try ``attempt + 1``: 2, 4, 8 ... up to the maximum, each made shorter by a random part."""
        return min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempt)) * (0.5 + random.random() / 2)

    def request(self, what, name, first, last, timeout, exact=True):
        """Bytes first..last of the file ``name``. Returns (data, headers)."""
        error = None
        for attempt in range(ATTEMPTS):
            if self.stop.is_set():
                raise Stopped(what)
            try:
                req = urllib.request.Request(self.urls[name], headers={'Range': 'bytes=%d-%d' % (first, last)})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data, headers = r.read(), r.headers
                if exact and len(data) != last - first + 1:
                    raise IOError('short read: %d of %d bytes' % (len(data), last - first + 1))
                return data, headers
            except Exception as exc:
                error = exc
                if permanent(exc) or attempt == ATTEMPTS - 1:
                    break
                sys.stderr.write('retry %s (try %d of %d): %s\n' % (what, attempt + 1, ATTEMPTS, describe(exc)))
                sys.stderr.flush()
                if self.stop.wait(self.pause(attempt)):
                    raise Stopped(what)
        failed = FetchError(what, error, attempt + 1)
        with self.lock:
            if self.failure is None:
                self.failure = failed
        self.stop.set()
        raise failed

    def size_of(self, name):
        data, headers = self.request('size of %s' % name, name, 0, 0, 120, exact=False)
        total = headers.get('Content-Range')
        return int(total.split('/')[-1]) if total else int(headers.get('Content-Length'))

    def get(self, job):
        name, out, first, last = job
        data, _ = self.request('range %s %d-%d' % (name, first, last), name, first, last, 300)
        fd = os.open(out, os.O_WRONLY)
        try:
            os.pwrite(fd, data, first)
        finally:
            os.close(fd)

    def download(self, jobs):
        """All ranges, in threads. The error that comes back is the first request that failed, not a thread that stopped."""
        with ThreadPoolExecutor(self.threads) as ex:
            futures = [ex.submit(self.get, job) for job in jobs]
        errors = [f.exception() for f in futures if f.exception() is not None]
        if self.failure is not None:
            raise self.failure
        if errors:
            raise errors[0]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 24), b''):
            h.update(blk)
    return path, h.hexdigest()


def materialize(desc, dest_root, urls, threads):
    cid = desc['id']
    dest = os.path.join(dest_root, cid)
    tmp = dest + '.partial'
    os.makedirs(dest_root, exist_ok=True)
    with open(dest + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(os.path.join(dest, '.verified')):
            return dict(ckpt=cid, status='present')
        os.makedirs(tmp, exist_ok=True)
        t0 = time.time()
        fetcher = Fetcher(urls, threads)
        jobs = []
        for name in sorted(desc['files']):
            out = os.path.join(tmp, name)
            n = fetcher.size_of(name)
            with open(out, 'wb') as f:
                f.truncate(n)
            for off in range(0, n, CHUNK):
                jobs.append((name, out, off, min(n, off + CHUNK) - 1))
        fetcher.download(jobs)
        t_dl = time.time() - t0
        with ThreadPoolExecutor(16) as ex:
            got = dict(ex.map(sha256_of, [os.path.join(tmp, n) for n in desc['files']]))
        bad = [n for n, w in desc['files'].items() if got[os.path.join(tmp, n)] != w]
        if bad:
            raise ValueError('sha256 mismatch: %s' % bad)
        with open(os.path.join(tmp, '.verified'), 'w') as f:
            f.write(json.dumps(dict(files=desc['files'], at=time.time(), by='promotion-gate')))
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.replace(tmp, dest)
        return dict(ckpt=cid, status='materialized', seconds=round(time.time() - t0, 1), download_s=round(t_dl, 1),
                    sources=['r2-presigned-get-parallel'], threads=threads)


def main(args):
    threads = int(args[args.index('--threads') + 1]) if '--threads' in args else 32
    pidfile = args[args.index('--pidfile') + 1] if '--pidfile' in args else None
    dest_root, uf = args[1], args[2]
    cid = None
    try:
        desc = json.load(open(args[0]))
        cid = desc['id']
        urls = json.load(open(uf))
        if pidfile:
            open(pidfile, 'w').write(str(os.getpid()))
        record, code = materialize(desc, dest_root, urls, threads), 0
    except Exception as exc:
        record, code = failure_record(cid, exc), 1
        sys.stderr.write('gate_fetch failed: %s\n' % record['error'])
        sys.stderr.flush()
    finally:
        for path in (uf, pidfile):
            try:
                os.remove(path)
            except Exception:
                pass
    print(json.dumps(record), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
