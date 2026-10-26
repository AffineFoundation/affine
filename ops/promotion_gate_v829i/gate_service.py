#!/usr/bin/env python3
"""SN120 promotion gate service (affinevalidator, systemd --user affine-promotion-gate.service).
Jacob pt682u / t684u 2026-10-26. Reads lifecycle state (read-only), drives the probe pod for the candidate probe,
writes gate decisions that the v8.2.9i lifecycle wrapper consumes. Never writes TRAIN-CONTROL, bans, weights or chain.

Per epoch N (head H = CURRENT checkpoint when N plans):
  1. N has PLAN.json -> draw private sample S (os.urandom seed, 0600), record a commitment; keep a hardlinked copy of
     H's trainer generation on the trainer pod (for a RED restore).
  2. N has TRAINING-RESULT.json -> candidate C. T = S minus N's DISPATCH tasks. H side = env-probe daemon's full
     k=1 private result for H (waits for it); C side = gate probe of C on T on the probe pod (fetch/serve/run).
  3. decide (gate_core) -> state/epochs/N/DECISION.json.
  4. Live RED finalized -> restore pod trainer CURRENT to H (trainer_restore.sh) -> TRAINER-RESTORED.json.
"""
import hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_core as G

V = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate')
ST = V / 'state'; EP = ST / 'epochs'
LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
FLAG = Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json')
PROBE = Path('/home/const/subnet120-rewrite/state/root-audits/20261022-env-probe/state')
SITE = Path('/home/const/subnet120/affine/website/gate')
BUSY = ST / 'GATE-BUSY.json'
PY = '/home/const/subnet120-rewrite/.venv/bin/python'
PODD = '/root/affine-env-probe'
PROBE_SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
             '-o', 'UserKnownHostsFile=/home/const/.lium/known_hosts/b4f3da9d-df66-436a-af38-829d1606966d', '-p', '40198', 'root@38.88.142.89']
RSSH = '/home/const/subnet120-rewrite/state/root-audits/20261026-rollback-189de412/scripts/rssh.sh'
DEFAULT_FLAG = dict(enabled=False, shadow=True, sample_size=1000, z_min=-1.0, cap_max=0.20, min_tasks=200,
                    red_credit='zero', pool=7496, env='affine_math', h_wait_seconds=2400, gmu='0.9', fetch_threads=32)


def cr(ts=None):
    ts = time.time() if ts is None else ts
    return time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)', time.gmtime(ts - 6 * 3600))


def log(**kw):
    kw['at'] = round(time.time(), 1); kw['at_cr'] = cr(kw['at']); print(json.dumps(kw, default=str), flush=True)


def wj(path, obj, mode=0o600):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    t = path.with_name('.' + path.name + '.tmp'); fd = os.open(t, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, 'w') as f: f.write(json.dumps(obj, indent=1, sort_keys=True) + '\n')
    os.chmod(t, mode); os.replace(t, path)


def rj(path, default=None):
    try: return json.loads(Path(path).read_text())
    except Exception: return default


def flag():
    d = rj(FLAG)
    return dict(DEFAULT_FLAG, **d) if isinstance(d, dict) else dict(DEFAULT_FLAG)


def mode(f):
    return 'live' if f.get('enabled') is True else ('shadow' if f.get('shadow', True) else 'off')


def pod(cmd, stdin=None, timeout=180):
    p = subprocess.run(PROBE_SSH + ([] if stdin is not None else ['-n']) + [cmd], input=stdin, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0: raise RuntimeError('probe pod rc=%d %s' % (p.returncode, p.stderr[-300:]))
    return p.stdout


def podctl(*args, stdin=None, timeout=180):
    out = pod('python3 %s/podctl.py %s' % (PODD, ' '.join(args)), stdin=stdin, timeout=timeout)
    return json.loads(out.strip().splitlines()[-1])


def busy(on, **info):
    if on: wj(BUSY, dict(at=time.time(), at_cr=cr(), **info), 0o644)
    else:
        try: BUSY.unlink()
        except FileNotFoundError: pass


def dispatch_tasks(e):
    d = rj(LIFE / 'epochs' / str(e) / 'coordinator' / 'DISPATCH.json') or {}
    return sorted({int(s['index']) for s in ((d.get('payload') or d).get('selected') or []) if 'index' in s})


def candidate_of(e):
    r = rj(LIFE / 'epochs' / str(e) / 'TRAINING-RESULT.json')
    if not r: return None
    rep = (r.get('payload') or r)['report']
    desc = rep['publication_receipt']['payload']['descriptor']
    if rep.get('training', {}).get('status') not in (None, 'updated'): return None
    return dict(id=desc['id'], files=desc['files'], input=rep.get('input_checkpoint'))


def h_results(H):
    res = rj(PROBE / 'results' / ('%s.private.json' % H))
    pt = (((res or {}).get('envs') or {}).get('affine_math') or {}).get('per_task') or {}
    return {int(k): v for k, v in pt.items() if isinstance(v, dict)}


def daemon_state(H):
    return ((rj(PROBE / 'ledger.json') or {}).get(H) or {}).get('state')


def trainer_pod(cmd, timeout=600):
    p = subprocess.run([RSSH, cmd], capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout[-2000:], p.stderr[-500:]


# ---------------- step 1: private sample + trainer keep ----------------
def ensure_sample(e, H, f):
    d = EP / str(e); sp = d / 'SAMPLE.private.json'
    if sp.exists(): return rj(sp)
    seed = os.urandom(32).hex(); idx = G.draw_sample(seed, int(f['pool']), int(f['sample_size']))
    s = dict(version=G.VERSION, epoch=e, head=H, seed=seed, indices=idx, drawn_at=time.time(), drawn_at_cr=cr())
    wj(sp, s, 0o600)
    wj(d / 'COMMIT.json', dict(version=G.VERSION, epoch=e, head=H, sample_size=len(idx), commitment=G.sample_commitment(seed, idx),
                               drawn_at=s['drawn_at'], drawn_at_cr=s['drawn_at_cr'],
                               rule='commitment = sha256(version|seed|comma-joined sorted indices); sample task ids are never published'), 0o644)
    log(event='sample_drawn', epoch=e, head=H[:12], n=len(idx))
    return s


def ensure_keep(e, H):
    kp = ST / 'keep' / ('%s.json' % H)
    k = rj(kp)
    if k and k.get('ok'): return k
    if k and time.time() - k.get('tried_at', 0) < 120: return k
    rc, out, err = trainer_pod('python3 /workspace/affine-mimo-trainer-prod/private/promotion-gate/trainer_keep.py keep %s' % H, timeout=900)
    try: res = json.loads(out.strip().splitlines()[-1])
    except Exception: res = dict(ok=False, error=(err or out)[-300:])
    res.update(rc=rc, tried_at=time.time(), tried_at_cr=cr(), epoch=e)
    wj(kp, res); log(event='trainer_keep', epoch=e, head=H[:12], ok=res.get('ok'), info=str(res)[:300])
    return res


# ---------------- step 2: candidate probe ----------------
def fetch_and_serve(C, f, tl):
    st = podctl('status')
    if C not in st['ckpts']:
        if st.get('fetching'):
            return 'fetching-other'
        tl.setdefault('fetch_start', time.time())
        blob = subprocess.run([PY, '-B', str(V / 'code' / 'presign.py'), C['id'] if isinstance(C, dict) else C],
                              input=json.dumps(tl['files']), capture_output=True, text=True, timeout=120)
        if blob.returncode != 0: raise RuntimeError('presign failed: %s' % blob.stderr[-200:])
        ck = tl['candidate']
        pod(("umask 077; mkdir -p {D}/ckpt {D}/gate/work; cd {D}/gate/work && python3 -c \"import json,sys;d=json.load(sys.stdin);"
             "json.dump(d['descriptor'],open('{D}/ckpt/{c}.descriptor.json','w'));json.dump(d['urls'],open('{c}.urls.private.json','w'))\" && "
             "for x in $(ls -t {D}/ckpt | grep -E '^[0-9a-f]{{64}}$' | tail -n +5); do [ \"$x\" != \"$(cat {D}/serve.ckpt)\" ] && rm -rf {D}/ckpt/$x; done; "
             "(setsid nohup python3 {D}/gate/gate_fetch.py {D}/ckpt/{c}.descriptor.json {D}/ckpt {c}.urls.private.json --threads {t} --pidfile {D}/fetch.pid "
             ">> {D}/ckpt/{c}.fetch.log 2>&1 &); echo ok").format(D=PODD, c=ck, t=int(f['fetch_threads'])), stdin=blob.stdout)
        return 'fetching'
    tl.setdefault('fetched', time.time())
    if st.get('served') != tl['candidate']:
        if not st.get('serve_procs') or st.get('serve_ckpt') != tl['candidate']:
            podctl('serve', tl['candidate'], str(f['gmu'])); tl['serve_start'] = time.time()
        return 'serving'
    tl.setdefault('served', time.time())
    return 'ready'


def run_candidate(e, C, tasks, seed, f, tl):
    name = 'gate-e%06d-%s' % (e, C[:12])
    st = podctl('status'); r = st['runs'].get(name, {})
    if r.get('result'):
        res = json.loads(pod('cat %s/runs/%s/result.json' % (PODD, name)))
        return res
    if not r.get('running') and 'run_start' not in tl:
        plan = dict(run_name=name, checkpoint_id=C, seed=hashlib.sha256(('%s:%s:gate-run' % (seed, C)).encode()).hexdigest(),
                    concurrency=4096, grade_workers=160, resume=False, bandit_topk=False, concurrency_per_replica=512,
                    envs=[dict(env_id=f['env'], indices=list(tasks), k=1, max_tokens=8192)])
        podctl('start-run', C, stdin=json.dumps(plan)); tl['run_start'] = time.time()
        log(event='gate_run_started', epoch=e, candidate=C[:12], tasks=len(tasks))
    elif not r.get('running') and time.time() - tl.get('run_start', 0) > 60:
        # worker died without a result: salvage graded rows
        fin = podctl('finalize-run', name, '30', timeout=90)
        log(event='gate_run_salvage', epoch=e, info=fin)
    return None


def process_epoch(e, f):
    d = EP / str(e); dp = d / 'DECISION.json'
    if dp.exists(): return 'decided'
    s = rj(d / 'SAMPLE.private.json')
    if not s: return 'no-sample'
    cand = candidate_of(e)
    if not cand: return 'no-candidate'
    H, C = s['head'], cand['id']
    tlp = d / 'TIMELINE.json'; tl = rj(tlp) or dict(candidate=C, head=H, files=cand['files'], result_seen=time.time(),
                                                      result_mtime=(LIFE / 'epochs' / str(e) / 'TRAINING-RESULT.json').stat().st_mtime)
    if cand.get('input') and cand['input'] != H:
        wj(dp, dict(version=G.VERSION, epoch=e, head=H, candidate=C, decision='skip', reasons=['candidate input %s != sampled head' % cand['input'][:12]], mode=mode(f)), 0o600)
        return 'skip'
    batch = dispatch_tasks(e); T = [t for t in s['indices'] if t not in set(batch)]
    hpt = h_results(H)
    cover = sum(1 for t in T if t in hpt)
    if cover < 0.95 * len(T):
        ds = daemon_state(H)
        tl.setdefault('h_wait_start', time.time()); wj(tlp, tl)
        if time.time() - tl['h_wait_start'] < float(f['h_wait_seconds']) and ds in ('seen', 'seeded', 'running'):
            return 'waiting-H-probe(%s %d/%d)' % (ds, cover, len(T))
        log(event='h_probe_incomplete', epoch=e, head=H[:12], cover=cover, need=len(T), daemon_state=ds)
    tl.setdefault('h_ready', time.time())
    busy(True, epoch=e, candidate=C)
    stage = fetch_and_serve(C, f, tl); wj(tlp, tl)
    if stage != 'ready': return stage
    res = run_candidate(e, C, T, s['seed'], f, tl); wj(tlp, tl)
    if res is None: return 'running-candidate'
    tl['run_done'] = time.time()
    cpt = {int(k): v for k, v in (((res.get('envs') or {}).get(f['env']) or {}).get('per_task') or {}).items()}
    wj(d / 'C-PROBE.private.json', dict(candidate=C, per_task=cpt, wall=res.get('wall_seconds'), partial=res.get('partial')))
    dec = G.decide(hpt, cpt, s['indices'], batch, z_min=float(f['z_min']), cap_max=float(f['cap_max']), min_tasks=int(f['min_tasks']))
    dec.update(epoch=e, head=H, candidate=C, mode=mode(f), decided_at=time.time(), decided_at_cr=cr(),
               batch_tasks=len(batch), h_cover=cover,
               timing=dict(result_to_decision_s=round(time.time() - tl['result_mtime'], 1),
                           h_wait_s=round(tl['h_ready'] - tl['result_seen'], 1),
                           fetch_s=round(tl.get('fetched', tl['h_ready']) - tl.get('fetch_start', tl.get('fetched', tl['h_ready'])), 1),
                           serve_s=round(tl.get('served', 0) - tl.get('serve_start', tl.get('served', 0)), 1),
                           run_s=round(tl['run_done'] - tl.get('run_start', tl['run_done']), 1)),
               red_credit=f.get('red_credit', 'zero'), commitment=(rj(d / 'COMMIT.json') or {}).get('commitment'))
    wj(tlp, tl); wj(dp, dec, 0o600); busy(False)
    log(event='decision', epoch=e, decision=dec['decision'], mode=dec['mode'], z=round(dec['evaluated']['z'], 3),
        cap=dec['evaluated']['c_cap_rate'], n=dec['evaluated']['n'], reasons=dec['reasons'], timing=dec['timing'])
    return 'decided'


# ---------------- step 4: RED restore ----------------
def ensure_restore(e, f):
    d = EP / str(e); dec = rj(d / 'DECISION.json') or {}
    if dec.get('decision') != 'red': return None
    fin = rj(LIFE / 'epochs' / str(e) / 'FINALIZED.json')
    if not fin: return 'waiting-finalize'
    res = (fin.get('payload') or fin).get('result') or {}
    if not str(res.get('reason', '')).startswith('promotion-gate-red'): return 'not-gated'
    if (d / 'TRAINER-RESTORED.json').exists(): return 'restored'
    lp = d / 'RESTORE-ATTEMPTS.json'; att = rj(lp, dict(n=0, last=0))
    if time.time() - att['last'] < 90: return 'restore-backoff'
    att['n'] += 1; att['last'] = time.time(); wj(lp, att)
    log(event='trainer_restore_start', epoch=e, head=dec['head'][:12], attempt=att['n'])
    p = subprocess.run(['bash', str(V / 'code' / 'trainer_restore.sh'), dec['head'], str(e)], capture_output=True, text=True, timeout=2400)
    ok = p.returncode == 0
    log(event='trainer_restore_done', epoch=e, ok=ok, tail=(p.stdout + p.stderr)[-600:])
    if ok: wj(d / 'TRAINER-RESTORED.json', dict(epoch=e, head=dec['head'], at=time.time(), at_cr=cr()), 0o644)
    return 'restored' if ok else 'restore-failed'


# ---------------- public record (commitment now, reveal after lag) ----------------
def publish(cur_epoch, lag=2):
    entries = []
    for d in sorted(EP.glob('*'), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not d.name.isdigit(): continue
        e = int(d.name); c = rj(d / 'COMMIT.json')
        if not c: continue
        ent = dict(epoch=e, head=c['head'], sample_size=c['sample_size'], commitment=c['commitment'], drawn_at_cr=c['drawn_at_cr'])
        dec = rj(d / 'DECISION.json')
        if dec:
            ev = dec.get('evaluated') or {}
            ent.update(candidate=dec.get('candidate'), decision=dec.get('decision'), mode=dec.get('mode'), decided_at_cr=dec.get('decided_at_cr'),
                       paired_tasks=ev.get('n'), head_pass=ev.get('h_pass'), candidate_pass=ev.get('c_pass'), paired_z=ev.get('z'),
                       candidate_cap_rate=ev.get('c_cap_rate'), reasons=dec.get('reasons'), batch_tasks_dropped=dec.get('batch_overlap'))
        # Jacob t690u: the private sample task ids are never published (no lagged reveal); only the commitment.
        entries.append(ent)
    wj(SITE / 'index.json', dict(version=G.VERSION, updated_at_cr=cr(), sample_ids_published=False,
        rule=('Each epoch draws a private uniform sample of 1000 affine_math probe tasks; tasks in the epoch\'s training batch are '
              'dropped. The candidate is promoted only if paired z (candidate vs head, k=1) > -1 and candidate cap-hit rate < 0.20. '
              'RED = candidate discarded, next epoch mines from the head again, zero flip credit for that epoch. Only a hash commitment '
              'of the sample is published; the sample task ids stay private. Full verdict ledger: https://affine.io/gate.json'), entries=entries[-200:]), 0o644)


def tick():
    f = flag(); m = mode(f)
    if m == 'off': busy(False); return 'off'
    cur = rj(LIFE / 'CURRENT.json'); N = cur['epoch_index']; H = cur['checkpoint']['id']
    out = {}
    if (LIFE / 'epochs' / str(N) / 'PLAN.json').exists():
        ensure_sample(N, H, f)
        try: out['keep'] = ensure_keep(N, H).get('ok')
        except Exception as ex: out['keep'] = 'error %s' % ex
    any_busy = False
    for e in (N - 2, N - 1, N):
        try:
            r = process_epoch(e, f); out[str(e)] = r
            any_busy |= r not in ('decided', 'no-sample', 'no-candidate', 'skip')
        except Exception as ex:
            out[str(e)] = "error: %s" % ex; log(event='epoch_error', epoch=e, error=str(ex)[:300], tb=traceback.format_exc()[-600:])
        if m == 'live':
            try: out['restore-%d' % e] = ensure_restore(e, f)
            except Exception as ex: out['restore-%d' % e] = 'error %s' % ex
    if not any_busy: busy(False)
    try: publish(N)
    except Exception as ex: log(event='publish_error', error=str(ex)[:200])
    wj(ST / 'status.json', dict(at_cr=cr(), mode=m, epoch=N, head=H, state=out), 0o644)
    return out


def main():
    last = None
    while True:
        try:
            s = tick()
            if s != last: log(event='tick', state=s); last = s
        except Exception as ex:
            log(event='error', error=str(ex)[:300], tb=traceback.format_exc()[-800:])
        if '--once' in sys.argv: break
        time.sleep(15)


if __name__ == '__main__':
    main()
