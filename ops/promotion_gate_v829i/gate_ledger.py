#!/usr/bin/env python3
"""SN120 promotion-gate verdict ledger (Jacob t690u, 2026-10-26).

Append-only public ledger of every gate verdict -> state/LEDGER.public.jsonl and https://affine.io/gate.json,
plus per-epoch gate timing (pre-probe on H, post-probe on C) for the site's epoch timing chart.
Private: state/REJECTED.private.jsonl keeps every rejected candidate id + batch manifest refs.
Never publishes seeds or the private sample task ids.
"""
import hashlib, json, os, time
from pathlib import Path

VERSION = 'sn120-promotion-gate-ledger-v1'
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
V = RA / '20261026-promotion-gate'
ST = V / 'state'; EP = ST / 'epochs'
LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
PROBE_RESULTS = RA / '20261022-env-probe/state/results'
LEDGER = ST / 'LEDGER.public.jsonl'
REJECTED = ST / 'REJECTED.private.jsonl'
SITE_OUT = Path('/home/const/subnet120/affine/website/gate.json')
FORBIDDEN = ('seed', 'indices', 'per_task', 'sample_tasks')


def rj(p, default=None):
    try:
        return json.loads(Path(p).read_bytes())
    except Exception:
        return default


def sha_file(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:
        return None


def mtime(p):
    try:
        return round(Path(p).stat().st_mtime, 1)
    except Exception:
        return None


def cr(t):
    return time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)', time.gmtime(t - 6 * 3600)) if t else None


def manifest_window(e):
    plan = rj(LIFE / 'epochs' / str(e) / 'PLAN.json') or {}
    m = plan.get('manifest') or {}
    m = m.get('payload', m) if isinstance(m, dict) else {}
    return m.get('start'), m.get('deadline')


def pre_probe(head, gather_start):
    r = rj(PROBE_RESULTS / ('%s.private.json' % head)) or {}
    if not r:
        return None
    return dict(checkpoint=head, started_at=r.get('started_at'), finished_at=r.get('finished_at'),
                tasks=r.get('samples'), partial=r.get('partial'),
                reused=bool(gather_start and r.get('finished_at') and r['finished_at'] < gather_start))


def entry(e):
    d = EP / str(e); dec = rj(d / 'DECISION.json')
    if not dec or dec.get('decision') not in ('green', 'red'):
        return None
    life = LIFE / 'epochs' / str(e)
    applied = rj(life / 'PROMOTION-GATE.json') or {}
    tl = rj(d / 'TIMELINE.json') or {}
    ev = dec.get('evaluated') or {}
    start, deadline = manifest_window(e)
    verdict = 'promoted' if dec['decision'] == 'green' else 'rejected'
    return dict(
        epoch=e, head=dec['head'], candidate=dec['candidate'], decision=dec['decision'], verdict=verdict,
        applied=applied.get('decision'), mode=dec.get('mode'),
        z=ev.get('z'), candidate_cap_rate=ev.get('c_cap_rate'), n=ev.get('n'),
        head_pass=ev.get('h_pass'), candidate_pass=ev.get('c_pass'),
        batch_tasks_dropped=dec.get('batch_overlap'), reasons=dec.get('reasons'),
        reason=('; '.join(dec.get('reasons') or []) or (('z %s %.1f and cap-hit < %.2f' % ('>=' if dec.get('z_inclusive') else '>', dec.get('z_min', -1.0), dec.get('cap_max', 0.20)))
                if verdict == 'promoted' else None)),
        thresholds=dict(z_min=dec.get('z_min', -1.0), z_inclusive=bool(dec.get('z_inclusive')), cap_max=dec.get('cap_max', 0.20),
                        min_tasks=dec.get('min_tasks', 200)), red_credit=dec.get('red_credit'),
        commitment=dec.get('commitment'),
        timestamps=dict(gather_start=start, upload_deadline=deadline,
                        training_result=tl.get('result_mtime') or mtime(life / 'TRAINING-RESULT.json'),
                        pre_probe=pre_probe(dec['head'], start),
                        post_probe=dict(start=tl.get('fetch_start') or tl.get('h_ready'), fetch_start=tl.get('fetch_start'),
                                        fetched=tl.get('fetched'), serve_start=tl.get('serve_start'), served=tl.get('served'),
                                        run_start=tl.get('run_start'), end=tl.get('run_done')),
                        decided_at=dec.get('decided_at'), finalized=mtime(life / 'FINALIZED.json')),
        decided_at_cr=dec.get('decided_at_cr'), timing_s=dec.get('timing'))


def private_reject(e, ent):
    life = LIFE / 'epochs' / str(e)
    tr = rj(life / 'TRAINING-RESULT.json') or {}
    desc = ((((tr.get('payload') or {}).get('report') or {}).get('publication_receipt') or {}).get('payload') or {}).get('descriptor') or {}
    return dict(epoch=e, head=ent['head'], candidate=ent['candidate'], z=ent['z'], cap=ent['candidate_cap_rate'], n=ent['n'],
                training_result_sha256=sha_file(life / 'TRAINING-RESULT.json'),
                dispatch_sha256=sha_file(life / 'coordinator' / 'DISPATCH.json'),
                training_job_sha256=sha_file(life / 'TRAINING-JOB.json'),
                finalized_sha256=sha_file(life / 'FINALIZED.json'),
                checkpoint_descriptor=desc,
                r2_prefix='mimo/affine-mimo9b-grpo-20261013/checkpoints/%s/' % ent['candidate'],
                retention='checkpoint files kept in R2 and on the probe pod (gate ckpt cache, newest 4 + served) while disk allows; hash recorded regardless',
                recorded_at=time.time())


def load_jsonl(p):
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                try: out.append(json.loads(line))
                except Exception: pass
    return out


def append(p, obj, mode=0o644):
    new = not p.exists()
    with open(p, 'a') as f:
        f.write(json.dumps(obj, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
    if new: os.chmod(p, mode)


def current_status():
    s = rj(ST / 'status.json') or {}
    e = s.get('epoch'); out = dict(epoch=e, head=s.get('head'), mode=s.get('mode'), stage=(s.get('state') or {}).get(str(e)))
    if e is not None:
        tl = rj(EP / str(e) / 'TIMELINE.json') or {}
        start, deadline = manifest_window(e)
        out.update(gather_start=start, upload_deadline=deadline, pre_probe=pre_probe(s.get('head'), start) if s.get('head') else None,
                   post_probe=dict(start=tl.get('fetch_start') or tl.get('h_ready'), end=tl.get('run_done')) if tl else None,
                   candidate=tl.get('candidate'))
        prev = s.get('state') or {}
        out['stages'] = {k: v for k, v in prev.items() if k.isdigit()}
    return out


def tick():
    ledger = load_jsonl(LEDGER); seen = {(r['epoch'], r['candidate']) for r in ledger}
    rej = load_jsonl(REJECTED); rseen = {(r['epoch'], r['candidate']) for r in rej}
    for d in sorted((p for p in EP.glob('*') if p.name.isdigit()), key=lambda p: int(p.name)):
        e = int(d.name); ent = entry(e)
        if not ent: continue
        if (e, ent['candidate']) not in seen:
            if ent['timestamps']['finalized'] is None and ent['applied'] is None:
                continue  # wait until the lifecycle applied it so `applied` is recorded
            ent['ledgered_at'] = time.time(); append(LEDGER, ent); ledger.append(ent); seen.add((e, ent['candidate']))
        if ent['decision'] == 'red' and (e, ent['candidate']) not in rseen:
            append(REJECTED, private_reject(e, ent), 0o600); rseen.add((e, ent['candidate']))
    doc = dict(version=VERSION, updated_at=time.time(), updated_at_cr=cr(time.time()),
               rule='Candidate promoted unless clearly worse: paired z (candidate vs head, k=1) >= -1.5 (from E154; > -1 for E152-E153) and candidate cap-hit rate < 0.20 on a private '
                    'sample of 1000 affine_math probe tasks minus the epoch batch (>= 200 tasks). RED = rejected: discarded, next '
                    'epoch mines from the head again, zero flip credit for that epoch. Append-only; sample task ids are never published.',
               counts=dict(total=len(ledger), promoted=sum(r['verdict'] == 'promoted' for r in ledger),
                           rejected=sum(r['verdict'] == 'rejected' for r in ledger)),
               current=current_status(), entries=ledger)
    s = json.dumps(doc, sort_keys=True)
    for bad in ('"seed"', '"indices"', '"per_task"', 'revealed_'):
        if bad in s: raise ValueError('refusing to publish %s' % bad)
    tmp = SITE_OUT.with_name(SITE_OUT.name + '.tmp'); tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    os.chmod(tmp, 0o644); os.replace(tmp, SITE_OUT)
    return doc['counts']


if __name__ == '__main__':
    import sys
    if '--watch' not in sys.argv:
        print(tick()); raise SystemExit
    last = None
    while True:
        try:
            tick(); last = None
        except Exception as ex:
            if str(ex) != last: print(json.dumps(dict(event='error', error=str(ex)[:300], at=time.time())), flush=True)
            last = str(ex)
        time.sleep(20)
