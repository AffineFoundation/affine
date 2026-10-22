# v829 A3 offline timing: sequential vs parallel read-only GETs of e81's 256 signed input docs (sha256-verified by get_object)
import json, sys, time, shutil, threading, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/workspace/affine-mimo-trainer-prod/source-final-v822b')
from subnet.backend_jobs import get_object
import requests
J = '/workspace/affine-mimo-trainer-prod/workspace/mimo-train-6d98b9c8d7c03846a7a92fce706a190680cba85700d07e7e/attempt-1/job.json'
p = json.load(open(J)); p = p.get('payload', p); docs = p['training']['documents']
root = Path('/tmp/v829-fetch-test'); shutil.rmtree(root, ignore_errors=True)
res = {}
for mode, workers in (('sequential', 1), ('parallel8', 8), ('parallel16', 16)):
    out = root/mode; out.mkdir(parents=True)
    local = threading.local()
    def one(item):
        k, o = item
        if workers == 1:
            get_object(o['url'], k, out/(k + '.json'), o['size'], record_lifecycle=False)
        else:
            if not hasattr(local, 's'): local.s = requests.Session()
            get_object(o['url'], k, out/(k + '.json'), o['size'], session=local.s, record_lifecycle=False)
    t = time.monotonic()
    if workers == 1:
        for it in docs.items(): one(it)
    else:
        with ThreadPoolExecutor(workers) as ex: list(ex.map(one, docs.items()))
    dt = time.monotonic() - t
    ok = all(hashlib.sha256((out/(k + '.json')).read_bytes()).hexdigest() == k for k in docs)
    res[mode] = dict(seconds=round(dt, 1), files=len(list(out.iterdir())), all_sha256_ok=ok)
    print(mode, res[mode], flush=True)
same = all((root/'sequential'/(k+'.json')).read_bytes() == (root/m/(k+'.json')).read_bytes() for m in ('parallel8','parallel16') for k in docs)
print(json.dumps(dict(res, byte_identical_across_modes=same)))
shutil.rmtree(root)
