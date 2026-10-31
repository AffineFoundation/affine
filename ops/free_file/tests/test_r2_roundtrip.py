"""R2 round trip with the live bucket under a TEST prefix: sealed link (ff_service method) -> open with the stdlib reference
script -> presigned PUT (twice: overwrite) -> GET -> server-side snapshot copy -> format check. Prints no secrets/URLs."""
import json, os, sys, time, urllib.request, base64
from pathlib import Path
D = Path('/home/const/subnet120-rewrite/state/root-audits/20261030-free-file')
sys.path.insert(0, str(D / 'release'))
import ff_service as S, ff_format as F, affine_batch as AB
from nacl.signing import SigningKey
from subnet.storage import encrypt

b = S.bucket()
seed = os.urandom(32); sk = SigningKey(seed); ident = sk.verify_key.encode().hex()
hk = 'TESTff%d' % int(time.time())
key = 'private/ff/v1/test/%s/batch.json' % hk
link = dict(version=S.LINK_VERSION, hotkey=hk, object_key=key, issued_at=int(time.time()), expires_at=int(time.time()) + 3600,
            put_url=b.presign(key, 'put_object', 3600), put_headers={'Content-Type': 'application/octet-stream'},
            get_url=b.presign(key, 'get_object', 3600))
pub = dict(sealed=encrypt(ident, link))
opened = json.loads(AB.seal_open(base64.b64decode(pub['sealed']), seed))
assert opened == link, 'sealed link round trip'
tok = F.tokenizer('/home/const/subnet120-rewrite/state/root-audits/20261030-determinism/tok')
for n in (1, 2):
    raw = json.dumps(dict(version=F.FORMAT, hotkey=hk, groups=[dict(prompt='p%d' % i, rollouts=[dict(output='x' * n, reward=1.0)] * n)
                                                              for i in range(n)])).encode()
    req = urllib.request.Request(opened['put_url'], data=raw, method='PUT', headers=opened['put_headers'])
    with urllib.request.urlopen(req, timeout=60) as r:
        assert r.status == 200
    with urllib.request.urlopen(opened['get_url'], timeout=60) as r:
        assert r.read() == raw, 'get_url returns the latest upload (overwrite %d)' % n
copy = 'private/ff/v1/test/turns/%s.json' % hk
b.client.copy_object(Bucket=b.name, Key=copy, CopySource=dict(Bucket=b.name, Key=key))
snap = b.client.get_object(Bucket=b.name, Key=copy)['Body'].read()
assert snap == raw
ok, info, groups = F.check(snap, hk, tok, 125, 1000)
assert ok and info['groups'] == 2 and info['rollouts'] == 4, info
# wrong key cannot open the link
try:
    AB.seal_open(base64.b64decode(pub['sealed']), os.urandom(32)); raise AssertionError('opened with a wrong key')
except SystemExit:
    pass
# a PUT without the signed content type is refused
try:
    urllib.request.urlopen(urllib.request.Request(opened['put_url'], data=b'{}', method='PUT', headers={'Content-Type': 'text/plain'}), timeout=60)
    ct = 'accepted'
except urllib.error.HTTPError as e:
    ct = 'refused %d' % e.code
print(json.dumps(dict(ok=True, test_hotkey=hk, sealed_link='opened by stdlib reference script', overwrite='ok', snapshot_copy='ok',
                      format_check=info, wrong_content_type=ct, wrong_key='refused')))
