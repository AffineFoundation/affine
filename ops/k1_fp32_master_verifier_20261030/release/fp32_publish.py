#!/usr/bin/env python3
"""SN120 fp32 head master publisher (host; Jacob t831u 2026-10-30, RA/20261030-fp32-verifier).
On every GREEN gate decision whose turn ran trainer v8.3.3f, publish the candidate's fp32 master (the new head master):
  R2:   mimo/affine-mimo9b-grpo-20261013/checkpoints/<C>/fp32-master/<shard>.safetensors + meta.json (pod PUTs via presigned URLs)
  web:  /k1/fp32-master/<C>.json (manifest: master digest, per-file sha256/bytes/R2 key, per-tensor sha256) + /k1/fp32-master/index.json
The master digest is the one the trainer committed in turn.json training.optimizer.sgd.fp32_master.output_master_digest.
Read-only towards lifecycle/gate/trainer; never touches weights. usage: fp32_publish.py [--once]"""
import json, os, subprocess, sys, time, hashlib
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits/20261030-fp32-verifier')
LEDGER = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/LEDGER.public.jsonl')
WEB = Path('/home/const/subnet120/affine/website'); OUTD = WEB / 'k1' / 'fp32-master'
STATE = RA / 'state' / 'publisher.json'; LOG = RA / 'state' / 'publisher.log'
RSSH = '/home/const/subnet120-rewrite/state/root-audits/20261026-rollback-189de412/scripts/rssh.sh'
POD_PY = '/workspace/affine-mimo-trainer-prod/runtime-exact/venv/bin/python'
POD_UP = '/workspace/affine-mimo-trainer-prod/fp32-publish/pod_upload.py'
STORE = '/workspace/affine-mimo-trainer-prod/fp32-master'
PREFIX = 'mimo/affine-mimo9b-grpo-20261013/checkpoints/%s/fp32-master/'
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code')
def cr(): return time.strftime('%Y-%m-%d %H:%M:%S CR', time.gmtime(time.time() - 6 * 3600))
def log(**k):
    k['at_cr'] = cr(); LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, 'a') as f: f.write(json.dumps(k, sort_keys=True) + '\n')
def rj(p, d=None):
    try: return json.loads(Path(p).read_text())
    except Exception: return d
def wj(p, d, mode=0o644):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(d, indent=1, sort_keys=True) + '\n'); os.chmod(t, mode); os.replace(t, p)
def pod(cmd, inp=None, timeout=3600):
    r = subprocess.run([RSSH, cmd], input=inp, capture_output=True, text=True, timeout=timeout)
    if r.returncode: raise RuntimeError('pod rc %d: %s' % (r.returncode, r.stderr[-300:]))
    return r.stdout
def bucket():
    from presign import SRC, LIFE_CFG
    if SRC not in sys.path: sys.path.insert(0, SRC)
    from subnet.storage import Bucket
    return Bucket(json.loads(Path(LIFE_CFG).read_text())['bucket'])
def greens():
    out = []
    for line in LEDGER.read_text().splitlines():
        try: d = json.loads(line)
        except Exception: continue
        if d.get('applied') == 'green' and d.get('decision') == 'green': out.append(d)
    return out
def publish(e, C, fm, st):
    t0 = time.time(); entry = fm['output_store_entry']
    meta = json.loads(pod('cat %s/%s/meta.json && touch %s/%s/PIN' % (STORE, entry, STORE, entry), timeout=120))
    if meta['master_digest'] != fm['output_master_digest'] or meta['bf16_sha256'] != entry:
        raise RuntimeError('store entry does not match the committed digest')
    files = sorted(meta['files']) + ['meta.json']
    b = bucket(); urls = {f: b.presign(PREFIX % C + f, 'put_object', 7200) for f in files}
    res = json.loads(pod('%s -I -B %s %s' % (POD_PY, POD_UP, entry), inp=json.dumps(dict(urls=urls)), timeout=7200)); urls = None
    bad = [f for f, r in res['files'].items() if r.get('status') not in (200, 201)]
    if bad: raise RuntimeError('upload failed: %s' % bad)
    sizes = {}
    for f in files:   # R2 read-back of sizes
        h = b.client.head_object(Bucket=b.name, Key=PREFIX % C + f); sizes[f] = h['ContentLength']
    for f, v in meta['files'].items():
        if sizes[f] != v['bytes']: raise RuntimeError('R2 size mismatch ' + f)
    man = dict(version='sn120-fp32-head-master-v1', epoch=e, checkpoint=C, bf16_sha256=entry, master_digest=meta['master_digest'],
               master_digest_rule='sha256 of canonical JSON {tensor name: sha256(little-endian fp32 bytes)} (sorted keys, separators , :)',
               dtype='float32', format='safetensors shards; tensor names = trainable parameter names (model.language_model.*, lm_head.*)',
               bf16_rule='bf16 head = round-to-nearest-even(master); bf16_sha256 = sha256 of canonical {tensor name: sha256(bf16 bytes)}',
               r2_prefix=PREFIX % C, files={f: dict(meta['files'][f], r2_key=PREFIX % C + f) for f in meta['files']},
               meta_json=dict(r2_key=PREFIX % C + 'meta.json'), tensor_sha256=meta['tensor_sha256'], tensor_file=meta['tensor_file'],
               produced_from=dict(head_bf16_sha256=meta['input_bf16_sha256'], head_master_digest=meta['input_master_digest'],
                                  head_master_source=meta['source'], optimizer_step=meta['optimizer_step']),
               turn='/k1/turns/%d/turn.json' % e, published_at=time.time(), published_at_cr=cr(),
               upload_seconds=res['seconds'], publish_seconds=round(time.time() - t0, 1),
               access='same R2 bucket/access as the bf16 head checkpoints/<C>/model/*')
    wj(OUTD / ('%s.json' % C), man)
    idx = rj(OUTD / 'index.json', {}) or {}
    idx.setdefault('version', 'sn120-fp32-head-master-index-v1'); idx.setdefault('heads', [])
    idx['heads'] = [h for h in idx['heads'] if h.get('checkpoint') != C] + [dict(epoch=e, checkpoint=C, master_digest=meta['master_digest'],
                     manifest='/k1/fp32-master/%s.json' % C, r2_prefix=PREFIX % C, published_at_cr=man['published_at_cr'])]
    idx['updated_at_cr'] = cr(); wj(OUTD / 'index.json', idx)
    pod('rm -f %s/%s/PIN' % (STORE, entry), timeout=60)
    log(event='published', epoch=e, checkpoint=C[:12], digest=meta['master_digest'][:16], upload_s=res['seconds'], total_s=man['publish_seconds'],
        gb=round(sum(v['bytes'] for v in meta['files'].values()) / 1e9, 2))
def bootstrap_note(e, fm, turn):
    idx = rj(OUTD / 'index.json', {}) or {}
    if idx.get('bootstrap'): return
    idx.setdefault('version', 'sn120-fp32-head-master-index-v1'); idx.setdefault('heads', [])
    idx['bootstrap'] = dict(epoch=e, head=turn['training'].get('input_checkpoint'), head_bf16_sha256=fm['input_bf16_sha256'],
                            head_master_digest=fm['input_master_digest'],
                            rule='first v8.3.3f turn: the head master is the exact fp32 upcast of the bf16 head (not uploaded; reproducible)')
    idx['updated_at_cr'] = cr(); wj(OUTD / 'index.json', idx)
def tick():
    st = rj(STATE, {}) or {}; done = st.setdefault('done', {})
    for d in greens():
        e = int(d['epoch']); C = d['candidate']
        if str(e) in done or e < int(st.get('from_epoch', 10 ** 9)): continue
        turn = rj(WEB / 'k1' / 'turns' / str(e) / 'turn.json')
        if not turn:
            continue   # k1t has not published yet
        fm = ((turn.get('training') or {}).get('optimizer') or {}).get('sgd', {}).get('fp32_master')
        if not fm:
            done[str(e)] = dict(status='no-fp32-master (pre-v8.3.3f turn)'); wj(STATE, st); continue
        if fm.get('source') == 'bootstrap-upcast': bootstrap_note(e, fm, turn)
        try:
            publish(e, C, fm, st); done[str(e)] = dict(status='published', checkpoint=C, at_cr=cr())
        except Exception as x:
            n = st.setdefault('fail', {}).get(str(e), 0) + 1; st['fail'][str(e)] = n
            log(event='publish_error', epoch=e, error=repr(x)[:300], attempt=n)
            if n >= 5: done[str(e)] = dict(status='failed', error=repr(x)[:300])
        wj(STATE, st)
    # turns that ran v8.3.3f but were RED: record the bootstrap if the first v833f turn was RED
    st['last_tick_cr'] = cr(); wj(STATE, st)
if __name__ == '__main__':
    once = '--once' in sys.argv
    while True:
        try: tick()
        except Exception as x: log(event='tick_error', error=repr(x)[:300])
        if once: break
        time.sleep(60)
