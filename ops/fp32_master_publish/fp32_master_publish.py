#!/usr/bin/env python3
"""SN120 fp32 head master publisher v2 (RA/20261104-fp32-master-publish; Jacob 2026-11-04 05:42 CR "Publish the fp32 master
tensors per GREEN head"). Replaces the transient fp32-publisher unit (v1 = RA/20261030-fp32-verifier/release/fp32_publish.py).

Every GREEN in the gate ledger (applied == decision == green) -> the GREEN candidate's fp32 master (trainer-pod store entry
named by turn.json training.optimizer.sgd.fp32_master.output_store_entry, checked against the committed output_master_digest)
is uploaded to the SAME R2 bucket as the bf16 head:  mimo/affine-mimo9b-grpo-20261013/checkpoints/<C>/fp32-master/<shard>
and announced at https://affine.io/k1/fp32-master/<C>.json (+ index.json, current.json). Public download = presigned R2 GET
URLs in the manifest's `download` block (same mechanism as the bf16 head's read_urls), refreshed before they expire for the
newest PUBLIC_HEADS heads.

Only the master tensors are published: 427 F32 tensors (model.language_model.*, lm_head.*). The upload is refused if any
shard header holds another dtype or name, or if the store entry contains anything besides the shards / meta.json / PIN.
Read-only towards the gate, lifecycle, trainer and weights: it only reads the public ledger / turn.json, PINs a store entry
while uploading and writes website/k1/fp32-master/*. A failure is logged + written to state/ALERT.json; it never blocks,
reverts or changes the gate, the king or weights. Idempotent: state/publisher.json records done epochs; re-running re-uses
the R2 objects (size check) and rewrites the same manifest.
usage: fp32_master_publish.py [--once] [--epoch N --force]"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits/20261104-fp32-master-publish')
LEDGER = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/LEDGER.public.jsonl')
WEB = Path('/home/const/subnet120/affine/website'); OUTD = WEB / 'k1' / 'fp32-master'
STATE = RA / 'state' / 'publisher.json'; LOG = RA / 'state' / 'publisher.log'; ALERT = RA / 'state' / 'ALERT.json'
V1_STATE = Path('/home/const/subnet120-rewrite/state/root-audits/20261030-fp32-verifier/state/publisher.json')
RSSH = '/home/const/subnet120-rewrite/state/root-audits/20261026-rollback-189de412/scripts/rssh.sh'   # = current trainer pod
POD_PY = '/workspace/affine-mimo-trainer-prod/runtime-exact/venv/bin/python'
POD_DIR = '/workspace/affine-mimo-trainer-prod/fp32-publish'; POD_UP = POD_DIR + '/pod_upload.py'
LOCAL_UP = RA / 'code' / 'pod_upload.py'
STORE = '/workspace/affine-mimo-trainer-prod/fp32-master'
PREFIX = 'mimo/affine-mimo9b-grpo-20261013/checkpoints/%s/fp32-master/'
PUBLIC_HEADS = int(os.environ.get('FP32_PUBLIC_HEADS', '3'))     # newest N published heads get live download links
URL_SECONDS = 7 * 24 * 3600; URL_REFRESH_BELOW = 3 * 24 * 3600   # presigned GET lifetime / refresh when less is left
MAX_ATTEMPTS = 12; RETRY_SECONDS = 300
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code')
def cr(t=None): return time.strftime('%Y-%m-%d %H:%M:%S CR', time.gmtime((t or time.time()) - 6 * 3600))
def log(**k):
    k['at_cr'] = cr(); LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, 'a') as f: f.write(json.dumps(k, sort_keys=True) + '\n')
def rj(p, d=None):
    try: return json.loads(Path(p).read_text())
    except Exception: return d
def wj(p, d, mode=0o644):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(d, indent=1, sort_keys=True) + '\n'); os.chmod(t, mode); os.replace(t, p)
def alert(kind, **k):
    k.update(kind=kind, at_cr=cr()); log(event='ALERT', **k); wj(ALERT, k, 0o600)
def pod(cmd, inp=None, timeout=3600):
    r = subprocess.run([RSSH, cmd], input=inp, capture_output=True, text=True, timeout=timeout)
    if r.returncode: raise RuntimeError('pod rc %d: %s' % (r.returncode, r.stderr[-300:]))
    return r.stdout
_B = None
def bucket():
    global _B
    if _B is None:
        from presign import SRC, LIFE_CFG
        if SRC not in sys.path: sys.path.insert(0, SRC)
        from subnet.storage import Bucket
        _B = Bucket(json.loads(Path(LIFE_CFG).read_text())['bucket'])
    return _B
def greens():
    out = []
    for line in LEDGER.read_text().splitlines():
        try: d = json.loads(line)
        except Exception: continue
        if d.get('applied') == 'green' and d.get('decision') == 'green': out.append(d)
    return out
def ensure_uploader():
    """Self-heal after a trainer pod move: (re)install pod_upload.py when missing or changed."""
    want = hashlib.sha256(LOCAL_UP.read_bytes()).hexdigest()
    have = pod('sha256sum %s 2>/dev/null | cut -c1-64; true' % POD_UP, timeout=60).strip()
    if have != want:
        pod('mkdir -p %s && cat > %s.tmp && mv %s.tmp %s' % (POD_DIR, POD_UP, POD_UP, POD_UP), inp=LOCAL_UP.read_text(), timeout=60)
        log(event='pod_uploader_installed', sha256=want[:16], previous=have[:16] or None)
CHECK = r'''python3 - <<'PY'
import json, os, struct, sys
d = '%s/%s'; m = json.load(open(d + '/meta.json')); bad = []
extra = sorted(set(os.listdir(d)) - set(m['files']) - {'meta.json', 'PIN'})
n = 0
for f in sorted(m['files']):
    b = open(os.path.join(d, f), 'rb'); h = json.loads(b.read(struct.unpack('<Q', b.read(8))[0])); h.pop('__metadata__', None)
    for k, v in h.items():
        n += 1
        if v['dtype'] != 'F32' or not (k.startswith('model.language_model.') or k.startswith('lm_head.')): bad.append(k)
print(json.dumps(dict(extra=extra, bad=bad[:5], tensors=n, files=len(m['files']))))
PY'''
def r2_meta(C):
    """meta.json already in R2 with every shard at its size -> the upload is complete (no pod needed)."""
    b = bucket()
    try: meta = json.loads(b.client.get_object(Bucket=b.name, Key=PREFIX % C + 'meta.json')['Body'].read())
    except Exception: return None
    for f, v in meta['files'].items():
        try:
            if b.client.head_object(Bucket=b.name, Key=PREFIX % C + f)['ContentLength'] != v['bytes']: return None
        except Exception: return None
    return meta
def publish(e, C, fm):
    t0 = time.time(); entry = fm['output_store_entry']
    if not entry or len(entry) != 64 or any(c not in '0123456789abcdef' for c in entry): raise RuntimeError('bad store entry')
    meta = r2_meta(C); res = dict(seconds=0.0, files={}); pinned = False
    if meta is None:   # upload from the trainer pod store (pinned while uploading)
        meta = json.loads(pod('cat %s/%s/meta.json && touch %s/%s/PIN' % (STORE, entry, STORE, entry), timeout=120)); pinned = True
    if meta['master_digest'] != fm['output_master_digest'] or meta['bf16_sha256'] != entry:
        raise RuntimeError('store entry does not match the committed digest')
    files = sorted(meta['files']) + ['meta.json']; b = bucket()
    if pinned:
        chk = json.loads(pod(CHECK % (STORE, entry), timeout=300))
        if chk['extra'] or chk['bad'] or chk['tensors'] != len(meta['tensor_sha256']):
            raise RuntimeError('refused: store entry is not weights-only %s' % chk)
    sizes = {}
    for f in files:
        try: sizes[f] = b.client.head_object(Bucket=b.name, Key=PREFIX % C + f)['ContentLength']
        except Exception: sizes[f] = None
    need = [f for f in files if sizes[f] is None or (f in meta['files'] and sizes[f] != meta['files'][f]['bytes'])]
    if need and not pinned: raise RuntimeError('R2 incomplete: %s' % need)
    if need:   # idempotent: only what is missing / wrong in R2
        ensure_uploader()
        urls = {f: b.presign(PREFIX % C + f, 'put_object', 7200) for f in need}
        res = json.loads(pod('%s -I -B %s %s' % (POD_PY, POD_UP, entry), inp=json.dumps(dict(urls=urls)), timeout=7200)); urls = None
        bad = [f for f, r in res['files'].items() if r.get('status') not in (200, 201)]
        if bad: raise RuntimeError('upload failed: %s' % bad)
        for f in files: sizes[f] = b.client.head_object(Bucket=b.name, Key=PREFIX % C + f)['ContentLength']
    for f, v in meta['files'].items():
        if sizes[f] != v['bytes']: raise RuntimeError('R2 size mismatch ' + f)
    old = rj(OUTD / ('%s.json' % C), {}) or {}
    man = dict(version='sn120-fp32-head-master-v1', epoch=e, checkpoint=C, bf16_sha256=entry, master_digest=meta['master_digest'],
               master_digest_rule='sha256 of canonical JSON {tensor name: sha256(little-endian fp32 bytes)} (sorted keys, separators , :)',
               dtype='float32', format='safetensors shards; tensor names = trainable parameter names (model.language_model.*, lm_head.*)',
               bf16_rule='bf16 head = round-to-nearest-even(master); bf16_sha256 = sha256 of canonical {tensor name: sha256(bf16 bytes)}',
               r2_prefix=PREFIX % C, files={f: dict(meta['files'][f], r2_key=PREFIX % C + f) for f in meta['files']},
               meta_json=dict(r2_key=PREFIX % C + 'meta.json'), tensor_sha256=meta['tensor_sha256'], tensor_file=meta['tensor_file'],
               produced_from=dict(head_bf16_sha256=meta['input_bf16_sha256'], head_master_digest=meta['input_master_digest'],
                                  head_master_source=meta['source'], optimizer_step=meta['optimizer_step']),
               turn='/k1/turns/%d/turn.json' % e, published_at=old.get('published_at') or time.time(),
               published_at_cr=old.get('published_at_cr') or cr(), upload_seconds=res['seconds'] if res['files'] else old.get('upload_seconds', 0.0), publish_seconds=round(time.time() - t0, 1),
               bytes=sum(v['bytes'] for v in meta['files'].values()), tensors=len(meta['tensor_sha256']),
               contents='fp32 master weights only (no optimizer state: the step is stateless, fresh zero moments every turn)',
               access='public: presigned R2 GET links in `download` (same bucket and mechanism as the bf16 head read_urls); '
                      'miner/kit/fetch_fp32_master.py downloads and verifies them',
               fetch='python3 miner/kit/fetch_fp32_master.py --head %s --out ./head-fp32-master' % C)
    wj(OUTD / ('%s.json' % C), man)
    idx = rj(OUTD / 'index.json', {}) or {}
    idx.setdefault('version', 'sn120-fp32-head-master-index-v1'); idx.setdefault('heads', [])
    idx['heads'] = [h for h in idx['heads'] if h.get('checkpoint') != C] + [dict(epoch=e, checkpoint=C, master_digest=meta['master_digest'],
                     manifest='/k1/fp32-master/%s.json' % C, r2_prefix=PREFIX % C, published_at_cr=man['published_at_cr'])]
    idx['heads'].sort(key=lambda h: h['epoch'])
    idx['updated_at_cr'] = cr(); wj(OUTD / 'index.json', idx)
    if pinned: pod('rm -f %s/%s/PIN' % (STORE, entry), timeout=60)
    log(event='published', epoch=e, checkpoint=C[:12], digest=meta['master_digest'][:16], upload_s=res['seconds'],
        uploaded=sorted(res['files']), total_s=man['publish_seconds'], gb=round(man['bytes'] / 1e9, 2))
def refresh_links(force=False):
    """Live download links for the newest PUBLIC_HEADS published heads; older heads keep their manifest (sha256s), no links."""
    idx = rj(OUTD / 'index.json', {}) or {}; heads = sorted(idx.get('heads') or [], key=lambda h: h['epoch'])
    live = heads[-PUBLIC_HEADS:]; live_c = {h['checkpoint'] for h in live}; b = None; now = time.time()
    for h in heads:
        p = OUTD / ('%s.json' % h['checkpoint']); man = rj(p)
        if not man: continue
        if h['checkpoint'] not in live_c:
            if man.get('download'):
                man.pop('download'); wj(p, man)
            continue
        dl = man.get('download') or {}
        if not force and dl.get('expires_at', 0) - now > URL_REFRESH_BELOW: continue
        b = b or bucket(); C = h['checkpoint']; exp = now + URL_SECONDS
        try:
            for f in man['files']: b.client.head_object(Bucket=b.name, Key=man['files'][f]['r2_key'])
        except Exception as x:
            alert('link_error', checkpoint=C[:12], error=repr(x)[:200]); continue
        man['download'] = dict(expires_at=exp, expires_at_cr=cr(exp), signed_at_cr=cr(now),
                               files={f: b.presign(man['files'][f]['r2_key'], 'get_object', URL_SECONDS) for f in sorted(man['files'])},
                               meta_json=b.presign(man['meta_json']['r2_key'], 'get_object', URL_SECONDS),
                               note='presigned R2 GET links, valid 7 days, re-signed when fewer than 3 days are left; '
                                    'check every file against files.<name>.sha256')
        wj(p, man); log(event='links_signed', checkpoint=C[:12], expires_at_cr=cr(exp))
    if heads:
        k = heads[-1]
        wj(OUTD / 'current.json', dict(version='sn120-fp32-head-master-current-v1', epoch=k['epoch'], checkpoint=k['checkpoint'],
                                         manifest=k['manifest'], master_digest=k['master_digest'], updated_at_cr=cr(),
                                         note='newest GREEN head with a published fp32 master; compare checkpoint with mining.json checkpoint'))
def tick():
    st = rj(STATE, {}) or {}; done = st.setdefault('done', {}); fail = st.setdefault('fail', {}); nxt = st.setdefault('next_try', {})
    for d in greens():
        e = int(d['epoch']); C = d['candidate']
        if str(e) in done or e < int(st.get('from_epoch', 10 ** 9)) or time.time() < nxt.get(str(e), 0): continue
        turn = rj(WEB / 'k1' / 'turns' / str(e) / 'turn.json')
        if not turn: continue   # k1t has not published the turn yet
        fm = ((turn.get('training') or {}).get('optimizer') or {}).get('sgd', {}).get('fp32_master')
        if not fm:
            done[str(e)] = dict(status='no-fp32-master (pre-v8.3.3f turn)'); wj(STATE, st); continue
        try:
            publish(e, C, fm); done[str(e)] = dict(status='published', checkpoint=C, at_cr=cr()); fail.pop(str(e), None)
            refresh_links(force=True)
        except Exception as x:
            n = fail.get(str(e), 0) + 1; fail[str(e)] = n; nxt[str(e)] = time.time() + RETRY_SECONDS
            gone = 'No such file' in repr(x) and 'meta.json' in repr(x)
            log(event='publish_error', epoch=e, checkpoint=C[:12], error=repr(x)[:300], attempt=n)
            if gone or n >= MAX_ATTEMPTS:
                done[str(e)] = dict(status='store-entry-gone' if gone else 'failed', checkpoint=C, error=repr(x)[:300], at_cr=cr())
                alert('publish_failed', epoch=e, checkpoint=C[:12], error=repr(x)[:300], attempts=n)
        wj(STATE, st)
    try: refresh_links()
    except Exception as x: alert('refresh_error', error=repr(x)[:300])
    st['last_tick_cr'] = cr(); wj(STATE, st)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--once', action='store_true'); ap.add_argument('--epoch', type=int)
    ap.add_argument('--force', action='store_true'); a = ap.parse_args()
    if not STATE.exists():   # inherit v1's done-list so nothing old is retried
        v1 = rj(V1_STATE, {}) or {}; wj(STATE, dict(done=v1.get('done', {}), fail={}, from_epoch=v1.get('from_epoch', 274)))
    if a.epoch is not None and a.force:
        st = rj(STATE); st['done'].pop(str(a.epoch), None); st['fail'].pop(str(a.epoch), None); st.get('next_try', {}).pop(str(a.epoch), None); wj(STATE, st)
    while True:
        try: tick()
        except Exception as x: alert('tick_error', error=repr(x)[:300])
        if a.once: break
        time.sleep(60)
