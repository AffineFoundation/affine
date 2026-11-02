"""SN120 sequential-gate LIVE SHADOW (Jacob Steeves approved 2026-11-02 11:14 CR). Never affects live decisions.

Read-only on the live gate state (rounds, epochs, lifecycle). Writes only under RA/20261102-seqgate-shadow/.
Two record kinds in shadow.jsonl (no task ids, no per-task values, no prior values, no selection probabilities):
  trunc - every decided live turn, sequential test replayed on the live turn's own grades (~475 tasks, head 12 rollouts).
          Zero GPU.
  full  - the design: 2,000-task baseline (live inclusion rule at N=2,000, nested with the live draw), head 8x seeded once per
          (head, task cap); candidate visited in a committed random order, generated in chunks of 500 on the shadow pod
          affine-seqgate-shadow (own pod; the live probe pod is never used), tested every 100 tasks, stopped at GREEN/RED.
Spend guard: the pod is released after MAX_HOURS (config.json, default 12) or when config.json says release.
Node-gone: ssh failing 10 min -> Lium check every 10 min; gone -> log node-gone once, state pod=gone, no automatic re-rent.
"""
import hashlib, json, math, os, random, subprocess, sys, time, traceback, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import seqgate as SG
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); G = RA / '20261026-promotion-gate'; GST = G / 'state'
LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
ROOT = RA / '20261102-seqgate-shadow'; STD = ROOT / 'state'; PRIV = ROOT / 'private' / 'full'; LOG = ROOT / 'shadow.jsonl'
PY = '/home/const/subnet120-rewrite/.venv/bin/python'; PRESIGN = G / 'code/presign.py'; PODD = '/root/affine-env-probe'
spec = importlib.util.spec_from_file_location('bv_ro', G / 'code/baseline_v1.py'); BV = importlib.util.module_from_spec(spec); spec.loader.exec_module(BV)
CONF = ROOT / 'config.json'
DEF = dict(budget=2000, chunk=500, inflight=2, max_hours=12.0, release=False, paired_seeds=False, full=True, gmu='0.9', fetch_threads=32,
           seq=dict())   # seq: overrides of seqgate.DEFAULTS (alpha, alpha_red, delta, batch, green_margin, lam_mode, ...)


def now(): return time.time()
def cr(t=None): return time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)', time.gmtime((t or now()) - 6 * 3600))
def rj(p, d=None):
    try: return json.load(open(p))
    except Exception: return d
def wj(p, o, mode=0o600):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + '.tmp')
    with open(t, 'w') as f: json.dump(o, f, sort_keys=True)
    os.chmod(t, mode); os.replace(t, p)
def append(o):
    o = dict(o, at=now(), at_cr=cr())
    with open(LOG, 'a') as f: f.write(json.dumps(o, sort_keys=True) + '\n')
def ev(**k):
    with open(STD / 'events.jsonl', 'a') as f: f.write(json.dumps(dict(k, at=now(), at_cr=cr()), sort_keys=True, default=str) + '\n')
def conf(): return dict(DEF, **(rj(CONF, {}) or {}))
def seqcfg(c): return dict(SG.DEFAULTS, **(c.get('seq') or {}), paired_seeds=bool(c['paired_seeds']), budget=int(c['budget']))


# ------------------------------------------------------------------ live state (read-only)
def round_ctx(rid):
    d = GST / 'heldout' / rid; r = rj(d / 'ROUND.private.json'); PI = rj(d / 'PI.private.json')
    if not (r and PI and r.get('ids') is not None): return None
    pool = r['pool']['per_env']; P = sum(pool.values()); w = {t: (pool[e] / P) / pool[e] for t, e in PI['env'].items()}
    return dict(r=r, pv=r['cfg'], PI=PI, w=w, N=sum(PI['pi'].values()), pool=pool)


def cells(pt, pv): return {BV.task_of(int(k), pv): v for k, v in pt.items() if isinstance(v, dict) and v.get('n')}


# ------------------------------------------------------------------ trunc: replay every decided live turn on its own grades
def trunc_pass(S):
    done = set(S.setdefault('trunc_done', []))
    eps = sorted(int(x) for x in os.listdir(GST / 'epochs') if x.isdigit() and int(x) >= S.get('trunc_from', 562))
    for e in eps:
        if e in done: continue
        d = GST / 'epochs' / str(e); D = rj(d / 'DECISION.json')
        if not D or (D.get('heldout') or {}).get('baseline') != 'prior-v1' or not (d / 'C-PROBE.private.json').exists(): continue
        rid = D['heldout']['round_id']; X = round_ctx(rid)
        if not X: continue
        head = cells(rj(d / 'H-PROBE.private.json')['per_task'], X['pv']); cand = cells(rj(d / 'C-PROBE.private.json')['per_task'], X['pv'])
        od = SG.visit_order(list(X['r']['ids']), SG.order_seed(X['r']['seed'], e))
        terms = SG.ht_terms(od, head, cand, X['PI']['pi'], X['w'], X['N']); caps = [((cand[t].get('b') or 0), cand[t]['n']) for t, *_ in terms]
        c = seqcfg(conf()); o = SG.run_sequential(terms, caps, c)
        last = o['trace'][-1] if o['trace'] else {}
        evd = D.get('evaluated') or {}
        append(dict(kind='trunc', epoch=e, round=rid, head=D['head'][:12], candidate=D['candidate'][:12], max_tokens=(D.get('heldout') or {}).get('max_tokens'),
                    live=dict(decision=D['decision'], z=evd.get('z'), gain=evd.get('gain'), n=evd.get('n'), cap=evd.get('c_cap_rate')),
                    shadow=dict(decision=o['decision'], reason=o['reason'], tasks_used=o['tasks_used'], e_green=last.get('e_green'), e_red=last.get('e_red'),
                                lcb=last.get('lcb'), ucb=last.get('ucb'), mean_ht=last.get('mean_ht'), cap=last.get('cap')),
                    budget=len(terms), head_rollouts=12, cfg=dict(alpha=c['alpha'], alpha_red=c['alpha_red'], delta=c['delta'], lam_mode=c['lam_mode'])))
        done.add(e); S['trunc_done'] = sorted(done)[-200:]


# ------------------------------------------------------------------ pod
sys.path.insert(0, '/home/const/subnet120-rewrite/ops/pods')
import replace_pod as RP


def pod():
    p = rj(STD / 'pod.json')
    return (RP.Pod(p['id'], p['ip'], p['port'], p.get('huid'), 'affine-seqgate-shadow'), p) if p else (None, None)


def prun(P, cmd, inp=None, timeout=180): return P.run(cmd, inp=inp, timeout=timeout)
def podctl(P, *a, stdin=None, timeout=180):
    out = prun(P, 'python3 %s/podctl.py %s' % (PODD, ' '.join(a)), inp=stdin, timeout=timeout)
    return json.loads(out.strip().splitlines()[-1])


def pod_alive(S, P, p):
    """node-gone handling (same rule as the bench / prior controllers): unreachable 10 min -> ask Lium every 10 min."""
    try:
        prun(P, 'true', timeout=40); S.pop('unreach_since', None)
        if S.get('pod_state') != 'up': ev(event='node-back' if S.get('pod_state') else 'pod-up', huid=p.get('huid')); S['pod_state'] = 'up'
        return True
    except Exception:
        S.setdefault('unreach_since', now())
        if now() - S['unreach_since'] > 600 and now() - S.get('lium_checked', 0) > 600:
            S['lium_checked'] = now()
            try:
                live = next((x for x in RP.lium_ps() if x['id'] == p['id']), None)
                if not live and S.get('pod_state') != 'gone': ev(event='node-gone', huid=p.get('huid'), id=p['id']); S['pod_state'] = 'gone'
            except Exception as x: ev(event='lium-check-failed', err=str(x)[-200:])
        return False


def ensure_ckpt(P, S, ck, files, tag):
    """-> True when ck is verified on the pod; starts a presigned R2 fetch otherwise (urls file deleted right after the fetch)."""
    st = podctl(P, 'status')
    if ck in st['ckpts']: return True
    if st.get('fetching'): return False
    k = 'fetch-' + ck
    if now() - S.get(k, 0) < 900: return False
    S[k] = now()
    blob = subprocess.run([PY, '-B', str(PRESIGN), ck], input=json.dumps(files), capture_output=True, text=True, timeout=120)
    if blob.returncode != 0: ev(event='presign-failed', ck=ck[:12], tag=tag, err=blob.stderr[-200:]); return False
    prun(P, ("umask 077; mkdir -p {D}/ckpt {D}/gate/work; cd {D}/gate/work && python3 -c \"import json,sys;d=json.load(sys.stdin);"
             "json.dump(d['descriptor'],open('{D}/ckpt/{c}.descriptor.json','w'));json.dump(d['urls'],open('{c}.urls.private.json','w'))\" && "
             "for x in $(ls -t {D}/ckpt | grep -E '^[0-9a-f]{{64}}$' | tail -n +4); do [ \"$x\" != \"$(cat {D}/serve.ckpt 2>/dev/null)\" ] && rm -rf {D}/ckpt/$x; done; "
             "(setsid nohup bash -c 'python3 {D}/gate/gate_fetch.py {D}/ckpt/{c}.descriptor.json {D}/ckpt {c}.urls.private.json --threads 32 --pidfile {D}/fetch.pid; rm -f {c}.urls.private.json' "
             ">> {D}/ckpt/{c}.fetch.log 2>&1 < /dev/null &); echo ok").format(D=PODD, c=ck), inp=blob.stdout)
    ev(event='fetch-start', ck=ck[:12], tag=tag); return False


def ensure_ctx(P, mt):
    need = int(mt) + 4096
    out = prun(P, "cd %s && python3 -c \"import json;print(json.load(open('live_math_manifest_subset.json'))['harness']['max_context_tokens'])\"; "
                  "grep -o -- '--max-model-len [0-9]*' serve.sh | head -1" % PODD, timeout=60).strip().splitlines()
    ctx, mml = int(out[0]), int(out[1].split()[-1])
    if ctx >= need and mml >= need: return True
    prun(P, ("cd {D} && cp -p live_math_manifest_subset.json live_math_manifest_subset.json.bak-seqgate && cp -p serve.sh serve.sh.bak-seqgate && "
             "python3 -c \"import json,os;p='live_math_manifest_subset.json';d=json.load(open(p));d['harness']['max_context_tokens']=max({n},d['harness']['max_context_tokens']);"
             "open(p+'.tmp','w').write(json.dumps(d));os.replace(p+'.tmp',p)\" && sed -i -E 's/--max-model-len [0-9]+/--max-model-len {n}/' serve.sh && "
             "python3 podctl.py stop-serve").format(D=PODD, n=need), timeout=120)
    ev(event='ctx-switch', need=need, was=[ctx, mml]); return False


def ensure_served(P, S, ck, c):
    st = podctl(P, 'status'); k = 'serve-' + ck
    if st.get('served') == ck and st.get('serve_procs') == st.get('replicas'): return True
    if st.get('serve_ckpt') == ck and now() - S.get(k, 0) < 600: return False      # cold serve in progress
    podctl(P, 'serve', ck, str(c['gmu'])); S[k] = now(); ev(event='serve', ck=ck[:12], was=str(st.get('served'))[:12]); return False


def run_result(P, name):
    st = podctl(P, 'status'); r = st['runs'].get(name, {})
    if r.get('result'): return json.loads(prun(P, 'cat %s/runs/%s/result.json' % (PODD, name), timeout=120))
    return 'running' if r.get('running') else None


def start_run(P, name, ck, seed, env, indices, k, mt):
    plan = dict(run_name=name, checkpoint_id=ck, seed=seed, concurrency=4096, grade_workers=160, resume=False, bandit_topk=False,
                concurrency_per_replica=512, envs=[dict(env_id=env, indices=list(indices), k=int(k), max_tokens=int(mt))])
    podctl(P, 'start-run', ck, stdin=json.dumps(plan)); ev(event='run-start', run=name, tasks=len(indices), k=k, mt=mt)


# ------------------------------------------------------------------ full design
def shadow_set(X, budget):
    """Same inclusion rule as the live round, scaled to E[N] = budget: pi'_i = (budget / N) pi_i (no cap binds: max pi' << 1);
    drawn with the live round's uniforms u_i (BV.uniform(seed, i)) -> the live set is nested inside the shadow set."""
    s = float(budget) / X['N']; pi2 = {t: min(1.0, s * p) for t, p in X['PI']['pi'].items()}
    ids = sorted(t for t, p in pi2.items() if BV.uniform(X['r']['seed'], t) < p)
    return ids, pi2, sum(pi2.values())


def head_seed(rseed, mt): return hashlib.sha256(('seqgate-shadow-v1:head:%s:%d' % (rseed, int(mt))).encode()).hexdigest()
def cand_seed(rseed, e): return hashlib.sha256(('seqgate-shadow-v1:cand:%s:%d' % (rseed, int(e))).encode()).hexdigest()


def full_pass(S, c):
    P, p = pod()
    if not P or S.get('pod_state') == 'gone': return 'no-pod'
    hrs = (now() - p['rented_at']) / 3600
    if c.get('release') or hrs > float(c['max_hours']):
        r = subprocess.run([PY, str(HERE / 'pod_shadow.py'), 'release', '--why', 'max_hours %.1f' % float(c['max_hours']) if not c.get('release') else 'config release'],
                           capture_output=True, text=True, timeout=600)
        ev(event='release', out=r.stdout[-300:], rc=r.returncode); S['pod_state'] = 'released'; return 'released'
    if not pod_alive(S, P, p): return 'pod-unreachable'
    cur = rj(GST / 'heldout' / 'CURRENT.private.json')
    if not cur: return 'no-round'
    X = round_ctx(cur['id'])
    if not X: return 'round-not-drawn'
    act = S.get('active')
    if act:   # finish the turn in flight first (its own baseline), even if the live head/round moved on meanwhile
        return turn_pass(S, c, P, X, S['baselines'][act['key']], act['key'])
    H = X['r']['head']; mt = int(X['pv'].get('max_tokens') or 8192); env = X['pv']['probe_env']
    bkey = '%s-%d' % (H[:12], mt); B = S.setdefault('baselines', {}).get(bkey)
    if not B or not B.get('done'):
        if not B:
            ids, pi2, N2 = shadow_set(X, c['budget'])
            B = dict(head=H, mt=mt, round=X['r']['id'], rseed=X['r']['seed'], n=len(ids), N=N2, started=now(), name='sg-H-%s-m%d-k8' % (H[:12], mt))
            wj(PRIV / ('BASE-%s.private.json' % bkey), dict(ids=ids, pi=pi2, N=N2, w={t: X['w'][t] for t in ids}, env={t: X['PI']['env'][t] for t in ids}, pv=X['pv']))
            S['baselines'][bkey] = B; ev(event='baseline-open', key=bkey, n=len(ids), round=X['r']['id'])
        files = (rj(LIFE / 'CURRENT.json') or {}).get('checkpoint', {})
        files = files.get('files') if files.get('id') == H else (rj(GST / 'epochs' / str(X['r'].get('opened_epoch')) / 'TIMELINE.json') or {}).get('files')
        if files is None:
            for e in sorted((int(x) for x in os.listdir(GST / 'epochs') if x.isdigit()), reverse=True)[:40]:
                t = rj(GST / 'epochs' / str(e) / 'TIMELINE.json') or {}
                if t.get('candidate') == H: files = t.get('files'); break
        if not files: return 'head-files-unknown'
        if not ensure_ckpt(P, S, H, files, 'head'): return 'baseline-fetch'
        if not ensure_ctx(P, mt): return 'baseline-ctx'
        if not ensure_served(P, S, H, c): return 'baseline-serve'
        bd = rj(PRIV / ('BASE-%s.private.json' % bkey))
        res = run_result(P, B['name'])
        if res is None:
            start_run(P, B['name'], H, head_seed(B['rseed'], mt), env, [BV.probe_index(t, bd['pv']) for t in bd['ids']], 8, mt); B['run_start'] = now(); return 'baseline-start'
        if res == 'running': return 'baseline-running'
        pt = cells(((res.get('envs') or {}).get(env) or {}).get('per_task') or {}, bd['pv'])
        bd['head'] = pt; wj(PRIV / ('BASE-%s.private.json' % bkey), bd)
        B.update(done=now(), tasks=len(pt), wall=res.get('wall_seconds'), head_pass=sum(v['c'] for v in pt.values()) / max(1, sum(v['n'] for v in pt.values())))
        append(dict(kind='baseline', key=bkey, head=H[:12], max_tokens=mt, tasks_drawn=B['n'], tasks_graded=len(pt), wall_s=B['wall'],
                    s_per_task8=(B['wall'] / max(1, len(pt))) if B['wall'] else None, total_s=round(now() - B['started'], 1), gpu=p.get('gpu')))
        return 'baseline-done'
    return turn_pass(S, c, P, X, B, bkey)


def turn_pass(S, c, P, X, B, bkey):
    H = B['head']; mt = B['mt']; T = S.setdefault('turns', {})
    # pick the turn: current one if in flight, else the newest epoch with TRAINING-RESULT whose candidate input is H
    act = S.get('active')
    if not act:
        cands = []
        for e in sorted((int(x) for x in os.listdir(GST / 'epochs') if x.isdigit()), reverse=True)[:6]:
            if str(e) in T: continue
            tl = rj(GST / 'epochs' / str(e) / 'TIMELINE.json') or {}
            if tl.get('head') != H or not tl.get('candidate') or not tl.get('files'): continue
            if not (LIFE / 'epochs' / str(e) / 'TRAINING-RESULT.json').exists(): continue
            if e <= S.get('min_epoch', 0): continue
            cands.append((e, tl))
        if not cands: return 'idle'
        e, tl = cands[0]
        for e2, _ in cands[1:]:
            if str(e2) not in T: T[str(e2)] = dict(skipped='busy'); append(dict(kind='skip', epoch=e2, why='shadow busy with a newer/other turn'))
        act = dict(epoch=e, cand=tl['candidate'], files=tl['files'], key=bkey, t0=now(), result_mtime=(LIFE / 'epochs' / str(e) / 'TRAINING-RESULT.json').stat().st_mtime,
                   chunks={}, paired=bool(c['paired_seeds']))
        S['active'] = act; ev(event='turn-open', epoch=e, cand=act['cand'][:12])
    e = act['epoch']; C = act['cand']
    if not ensure_ckpt(P, S, C, act['files'], 'cand-e%d' % e):
        if now() - act['t0'] > 2400: return finish(S, act, None, 'fetch timeout')
        return 'cand-fetch'
    act.setdefault('fetched', now())
    if not ensure_ctx(P, mt): return 'cand-ctx'
    if not ensure_served(P, S, C, c): return 'cand-serve'
    act.setdefault('served', now())
    bd = rj(PRIV / ('BASE-%s.private.json' % bkey)); head = bd['head']; env = bd['pv']['probe_env']
    order = [t for t in SG.visit_order(bd['ids'], SG.order_seed(B['rseed'], e)) if t in head]
    cs = int(c['chunk']); nchunk = math.ceil(min(len(order), int(c['budget'])) / cs)
    seed = head_seed(B['rseed'], mt) if act['paired'] else cand_seed(B['rseed'], e)
    cand = {}
    for i in range(nchunk):
        ch = act['chunks'].setdefault(str(i), {}); name = 'sg-C-e%d-%s-c%d%s' % (e, C[:12], i, '-p' if act['paired'] else '')
        if ch.get('done'):
            cand.update(cells((rj(PRIV / ('C-e%d-c%d.private.json' % (e, i))) or {}).get('per_task') or {}, bd['pv'])); continue
        res = run_result(P, name)
        if res is None:
            inflight = sum(1 for v in act['chunks'].values() if v.get('start') and not v.get('done'))
            if inflight < int(c['inflight']) and not ch.get('start'):
                start_run(P, name, C, seed, env, [BV.probe_index(t, bd['pv']) for t in order[i * cs:(i + 1) * cs]], 8, mt); ch['start'] = now()
            break
        if res == 'running':
            # keep the next chunk queued so the GPUs never drain between chunks
            if i + 1 < nchunk and not act['chunks'].get(str(i + 1), {}).get('start') and sum(1 for v in act['chunks'].values() if v.get('start') and not v.get('done')) < int(c['inflight']):
                n2 = 'sg-C-e%d-%s-c%d%s' % (e, C[:12], i + 1, '-p' if act['paired'] else '')
                start_run(P, n2, C, seed, env, [BV.probe_index(t, bd['pv']) for t in order[(i + 1) * cs:(i + 2) * cs]], 8, mt)
                act['chunks'].setdefault(str(i + 1), {})['start'] = now()
            break
        pt = ((res.get('envs') or {}).get(env) or {}).get('per_task') or {}
        wj(PRIV / ('C-e%d-c%d.private.json' % (e, i)), dict(per_task=pt, wall=res.get('wall_seconds')))
        ch.update(done=now(), wall=res.get('wall_seconds'), tasks=len(pt)); cand.update(cells(pt, bd['pv']))
    # test on the committed prefix that is complete
    order = order[:int(c['budget'])]
    doneset = set()
    for i, ch in act['chunks'].items():
        if ch.get('done'): doneset.update(order[int(i) * cs:(int(i) + 1) * cs])
    pre = []; k = 0
    for t in order:                      # committed prefix: graded tasks in order; tasks a finished chunk failed to grade are skipped
        if t in cand: pre.append(t)
        elif t in doneset: pass
        else: break
        k += 1
    final = k >= len(order)
    terms = SG.ht_terms(pre, head, cand, bd['pi'], bd['w'], bd['N']); caps = [((cand[t].get('b') or 0), cand[t]['n']) for t, *_ in terms]
    sc = seqcfg(c); sc['B'] = max(bd['w'][t] * bd['N'] / bd['pi'][t] for t in bd['ids'])
    o = SG.run_sequential(terms, caps, sc, final=final)
    act['prefix'] = len(pre)
    if o['decision'] is None: return 'cand-running(prefix %d)' % k
    return finish(S, act, o, None, sc=sc, k=len(pre))


def finish(S, act, o, err, sc=None, k=None):
    P, p = pod(); e = act['epoch']
    try:   # stop any chunk still generating (exact run names of this turn only)
        if P:
            for i, ch in act['chunks'].items():
                if ch.get('start') and not ch.get('done'):
                    try: podctl(P, 'finalize-run', 'sg-C-e%d-%s-c%s%s' % (e, act['cand'][:12], i, '-p' if act['paired'] else ''), '5', timeout=90)
                    except Exception: pass
    except Exception: pass
    act['decided'] = now(); act['shadow'] = (o or {}).get('decision'); act['err'] = err
    S['turns'][str(e)] = dict(decided=now(), decision=(o or {}).get('decision'), err=err); S['active'] = None
    S.setdefault('pending_log', []).append(dict(act=act, o=dict((o or {}), trace=(o or {}).get('trace', [])[-1:]), sc=sc and {x: sc[x] for x in ('alpha', 'alpha_red', 'delta', 'lam_mode', 'batch', 'budget', 'paired_seeds', 'green_margin')}))
    ev(event='turn-decided', epoch=e, decision=(o or {}).get('decision'), err=err, tasks=k)
    return 'decided'


def flush_logs(S):
    keep = []
    for x in S.get('pending_log', []):
        act, o = x['act'], x['o']; e = act['epoch']; D = rj(GST / 'epochs' / str(e) / 'DECISION.json')
        if not D and now() - act['decided'] < 3600: keep.append(x); continue   # wait for the live decision (never influences it)
        last = (o.get('trace') or [{}])[-1] if o.get('trace') else {}
        evd = (D or {}).get('evaluated') or {}
        walls = [v.get('wall') for v in act['chunks'].values() if v.get('wall')]
        append(dict(kind='full', epoch=e, head=(D or {}).get('head', '')[:12], candidate=act['cand'][:12], key=act['key'], err=act.get('err'),
                    live=dict(decision=(D or {}).get('decision'), z=evd.get('z'), gain=evd.get('gain'), n=evd.get('n'), cap=evd.get('c_cap_rate'),
                              result_to_decision_s=((D or {}).get('timing') or {}).get('result_to_decision_s')),
                    shadow=dict(decision=o.get('decision'), reason=o.get('reason'), tasks_used=o.get('tasks_used'), e_green=last.get('e_green'),
                                e_red=last.get('e_red'), mean_ht=last.get('mean_ht'), cap=last.get('cap'), early=bool(o.get('early'))),
                    cfg=x.get('sc'), paired_seeds=act['paired'],
                    timing=dict(result_to_shadow_decision_s=round(act['decided'] - act['result_mtime'], 1), fetch_s=round(act.get('fetched', act['decided']) - act['t0'], 1),
                                serve_s=round(act.get('served', act['decided']) - act.get('fetched', act['decided']), 1),
                                gen_s=round(act['decided'] - act.get('served', act['decided']), 1), chunk_walls=walls, chunks_started=sum(1 for v in act['chunks'].values() if v.get('start')))))
    S['pending_log'] = keep


def main():
    STD.mkdir(parents=True, exist_ok=True); PRIV.mkdir(parents=True, exist_ok=True); os.chmod(ROOT / 'private', 0o700)
    once = '--once' in sys.argv
    while True:
        S = rj(STD / 'state.json', {}) or {}; c = conf(); stage = None
        try:
            trunc_pass(S)
            if c.get('full'): stage = full_pass(S, c)
            flush_logs(S)
        except Exception as x:
            stage = 'error'; ev(event='error', err=str(x)[-400:], tb=traceback.format_exc()[-1200:])
        S['stage'] = stage; S['tick'] = now(); S['tick_cr'] = cr(); wj(STD / 'state.json', S, 0o600)
        wj(STD / 'STATUS.json', dict(stage=stage, at_cr=cr(), active=(S.get('active') or {}).get('epoch'), prefix=(S.get('active') or {}).get('prefix'),
                                     baselines={k: dict(done=bool(v.get('done')), n=v.get('n')) for k, v in (S.get('baselines') or {}).items()}, pod_state=S.get('pod_state')), 0o644)
        if once: print(stage); return
        time.sleep(20)


if __name__ == '__main__': main()
