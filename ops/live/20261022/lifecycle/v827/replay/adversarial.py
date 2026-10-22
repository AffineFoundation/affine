"""Adversarial test (READ-ONLY R2): a miner edits/swaps an OLD document (or its commitment) after an honest capture.
Root A = original code path (memo off). Root B = v8.2.7 memo on and warm. Each step's outcome must be identical."""
import importlib.util, json, sys, time
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
import rohost
from pathlib import Path
STAGE = Path(sys.argv[1]); EPOCH = int(sys.argv[2])
config, actor, policy, tok, real = rohost.build()
stats = rohost.new_stats(); bucket = rohost.ROBucket(real, stats)
from subnet.complete_group_native import resolve_task
from subnet.backend_jobs import signed
from subnet.storage import canonical, sha
from subnet.complete_group_transport import paths
from subnet.complete_group_capture import SubmissionRejected
spec = importlib.util.spec_from_file_location('affine_host_capture_cache', STAGE/'mimo_host_capture_cache.py')
M = importlib.util.module_from_spec(spec); sys.modules[spec.name] = M; spec.loader.exec_module(M)
M._install_pure_memo(); M._install_incremental_memo()
overlay = {}
RC = rohost.ROClient
orig_get = RC.__getattr__
class Body:
    def __init__(s, d): s.d = d
    def read(s, n=-1): return s.d if n is None or n < 0 else s.d[:n]
    def close(s): pass
def patched_getattr(self, name):
    f = orig_get(self, name)
    if name != 'get_object': return f
    def g(**kw):
        r = f(**kw)
        if kw.get('Key') in overlay:
            r['Body'].read(); r['Body'].close(); data = overlay[kw['Key']]
            r = dict(r, Body=Body(data), ContentLength=len(data))
        return r
    return g
RC.__getattr__ = patched_getattr
resolve = lambda m, e, i: resolve_task(m, e, i, tok)
A, live = rohost.coordinator_copy(EPOCH, Path('/tmp/v827-adv-A'), bucket, actor, resolve, policy)
B, _ = rohost.coordinator_copy(EPOCH, Path('/tmp/v827-adv-B'), bucket, actor, resolve, policy)
manifest = signed(json.loads((live/'OPENING.json').read_bytes()), actor.id); scope_sha = sha(canonical(manifest))
cp = M.CapturePass(workers=4); counters = dict(document_memo_hits=0, commitment_memo_hits=0, write_skips=0)
def run(coord, memo, miner):
    if memo: M._active.memo = cp.memo_frame(scope_sha, {}, counters)
    try:
        rec = coord.collect(miner, now=int(time.time()))
        return ('accepted', sha((coord.root/'captured'/(miner+'.json')).read_bytes()))
    except SubmissionRejected as e:
        return ('rejected', str(e)[:300])
    except Exception as e:
        return ('error', type(e).__name__ + ': ' + str(e)[:200])
    finally:
        M._active.memo = None
caps = sorted(((signed(json.loads(p.read_bytes()), actor.id)) for p in (live/'captured').glob('*.json')), key=lambda c: c['miner'])
targets = [c['miner'] for c in caps if len(c['documents']) >= 2][:3]
results = []
for miner in targets:
    ck = paths(manifest, miner); d0 = paths(manifest, miner, 0)[1]; d1 = paths(manifest, miner, 1)[1]
    raw0 = real.client.get_object(Bucket=real.name, Key=d0)['Body'].read(); raw1 = real.client.get_object(Bucket=real.name, Key=d1)['Body'].read()
    rawc = real.client.get_object(Bucket=real.name, Key=ck)['Body'].read()
    edit = bytearray(raw0); i = len(edit) // 2; edit[i] = (edit[i] + 1) % 256
    scenarios = [('honest', {}), ('edit-old-doc-slot0', {d0: bytes(edit)}), ('swap-old-docs-0-1', {d0: raw1, d1: raw0}),
                 ('edit-commitment', {ck: rawc.replace(b'"slot":0', b'"slot":7', 1) if b'"slot":0' in rawc else rawc[:-2] + b' }'}),
                 ('truncate-old-doc-slot1', {d1: raw1[:len(raw1)//2]}), ('restored-honest', {})]
    for name, ov in scenarios:
        overlay.clear(); overlay.update(ov)
        a = run(A, False, miner); b = run(B, True, miner)
        results.append(dict(miner_uid_prefix=miner[:8], scenario=name, original=a[0], v827=b[0], identical=a == b, detail=a[1][:120] if a[0] != 'accepted' else 'record sha ' + a[1][:16]))
out = dict(epoch=EPOCH, all_identical=all(r['identical'] for r in results), memo_counters=counters, results=results)
print(json.dumps(out, indent=1))
json.dump(out, open('/home/const/subnet120-rewrite/state/root-audits/20261021-v827-incremental-capture/replay/ADVERSARIAL-e%d.json' % EPOCH, 'w'), indent=1)
