"""Pod side (trainer pod): PUT one fp32-master store entry to R2 with presigned URLs read from stdin (never written to disk/logs).
usage: python pod_upload.py ENTRY < {"urls": {file: url}}   -> prints JSON {file: {status, bytes, seconds}}"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
STORE = '/workspace/affine-mimo-trainer-prod/fp32-master'
entry = sys.argv[1]; d = os.path.join(STORE, entry); urls = json.load(sys.stdin)['urls']
assert os.path.isfile(os.path.join(d, 'meta.json')), 'entry not complete'
open(os.path.join(d, 'PIN'), 'w').write(str(time.time()))
def put(name):
    p = os.path.join(d, name); n = os.path.getsize(p); t0 = time.time(); err = None
    for attempt in range(4):
        try:
            with open(p, 'rb') as f:
                req = urllib.request.Request(urls[name], data=f, method='PUT',
                                             headers={'Content-Length': str(n), 'Content-Type': 'application/octet-stream'})
                with urllib.request.urlopen(req, timeout=1800) as r:
                    if r.status in (200, 201):
                        return name, dict(status=r.status, bytes=n, seconds=round(time.time() - t0, 1), etag=r.headers.get('ETag'))
        except Exception as x:
            err = '%s: %s' % (type(x).__name__, str(x)[:120]); time.sleep(5 + 10 * attempt)
    return name, dict(status='failed', error=err, bytes=n)
t0 = time.time()
with ThreadPoolExecutor(int(os.environ.get('UP_THREADS', '6'))) as ex:
    res = dict(ex.map(put, sorted(urls)))
print(json.dumps(dict(entry=entry, files=res, seconds=round(time.time() - t0, 1))))
