"""Upload the v8.2.2 source archive to the lifecycle's private R2 source_bundle_key; verify by full read-back.
Never overwrites a different object. Run only by activate_v822.sh --arm."""
import sys, json, hashlib, time
from pathlib import Path
sys.dont_write_bytecode = True
D = Path('/home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession'); SG = D/'release-v822-signed'
sys.path.insert(0, str(SG/'deployment-final-source-v822'))
from subnet.storage import Bucket
life = json.loads((SG/'lifecycle.v8.2.2.ACTIVE.private.json').read_bytes()); key = life['adapters']['source_bundle_key']
src = D/'staged-v822-successor/source-release-v822/source.tar.gz'
want = json.loads((D/'staged-v822-successor/SUCCESSOR-BINDING.json').read_bytes())['archive_sha256']
assert hashlib.sha256(src.read_bytes()).hexdigest() == want and want in key
b = Bucket(life['bucket'])
def remote():
    try: return hashlib.sha256(b.get(key)).hexdigest()
    except Exception as e: return 'absent:' + type(e).__name__
before = remote()
if before != want:
    assert before.startswith('absent'), 'different object already at key: ' + before
    b.upload(key, str(src))
after = remote()
r = dict(version='mimo-v822-source-upload-v1', at=time.time(), key=key, bytes=src.stat().st_size, before=before, after=after, readback_verified=after == want)
(D/'R2-SOURCE-UPLOAD-v822.json').write_text(json.dumps(r, indent=1, sort_keys=True) + '\n'); print(json.dumps(r))
assert after == want
