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
HELD-OUT ROUNDS (heldout.py, Jacob pt808u 2026-10-30; PROMOTION-GATE.json "heldout" {"enabled": true, "from_epoch": E}):
  one committed secret set S per head (round), head baseline = 4 x k4 = 16 rollouts per task computed ONCE per round, every
  candidate of the round probed on S minus its batch (k=4, fresh plan seed) and decided vs that baseline with z >= bar_z.
  GREEN closes the round and reveals it (public/gate/heldout/<round>.json). Nothing per-task is public while a round is open.
  Test hooks (all default to the live values): HG_GATE_STATE, HG_SITE, HG_FLAG, HG_PROBE_SSH (json argv), HG_NO_TRAINER=1.
"""
import hashlib, json, os, shlex, signal, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_core as G
import heldout as HO

V = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate')
ST = Path(os.environ.get('HG_GATE_STATE') or (V / 'state')); EP = ST / 'epochs'
LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
FLAG = Path(os.environ.get('HG_FLAG') or '/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json')
PROBE = Path('/home/const/subnet120-rewrite/state/root-audits/20261022-env-probe/state')
SITE = Path(os.environ.get('HG_SITE') or '/home/const/subnet120/affine/website/gate')
BUSY = ST / 'GATE-BUSY.json'
PY = '/home/const/subnet120-rewrite/.venv/bin/python'
PODD = '/root/affine-env-probe'
PROBE_SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
             '-o', 'UserKnownHostsFile=/home/const/.lium/known_hosts/b4f3da9d-df66-436a-af38-829d1606966d', '-p', '40198', 'root@38.88.142.89']
if os.environ.get('HG_PROBE_SSH'): PROBE_SSH = json.loads(os.environ['HG_PROBE_SSH'])   # test pod only (e2e); live = the probe pod above
NO_TRAINER = os.environ.get('HG_NO_TRAINER') == '1'   # e2e: never call the trainer pod (keep / early staging / delta / restore)
ROUNDS = None   # heldout.Rounds, created lazily (needs wj/rj/log/cr below)
REPLAY_T0 = {}   # e2e replay of historical epochs only: {epoch: t} replaces the TRAINING-RESULT mtime (empty on the live path)
RSSH = '/home/const/subnet120-rewrite/state/root-audits/20261026-rollback-189de412/scripts/rssh.sh'
# Epoch-time pass (Jacob t696u 2026-10-26 19:35 CR): operator toggles, read every tick; absent/false = original behaviour.
#   delta: materialize C on the probe pod from H + XOR delta streamed from the trainer pod (R2 fetch stays the fallback)
#   early: stage (fetch/serve/run) C as soon as the trainer has STAGED it; the DECISION still waits for the lifecycle's
#          TRAINING-RESULT and requires its candidate id + file hashes to equal the staged ones (else start over).
OPT = ST / 'EPOCH-TIME.json'
TRAINER_DELTA = '/workspace/affine-mimo-trainer-prod/private/promotion-gate/gate_delta.py'
POD_DELTA = PODD + '/gate/gate_delta.py'
_early_poll = {}


def opts():
    d = rj(OPT)
    d = d if isinstance(d, dict) else {}
    return dict(delta=d.get('delta') is True and not NO_TRAINER, early=d.get('early') is True and not NO_TRAINER, delta_timeout=float(d.get('delta_timeout', 900)),
                warm_swap=d.get('warm_swap') is True, restore_bg=d.get('restore_bg') is True,   # t767u
                tick_s=max(2.0, float(d.get('tick_s', 15))), early_poll_s=max(2.0, float(d.get('early_poll_s', 20))),
                # v2a (parent decision after Jacob t771u, 2026-10-29): standing private sample + head-probe reuse
                standing=d.get('standing_sample') is True, standing_max_age_s=float(d.get('standing_max_age_s', 86400)),
                standing_min_cover=float(d.get('standing_min_cover', 0.8)),
                # probe-speed 20261030: an early-staged delta make waits for TRAINING-RESULT (trainer idle); early makes ran 4-8x slower
                delta_after_result=d.get('delta_after_result') is True)


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


def podctl_swap(C, gmu, timeout=960):
    # v2b: podctl swap exits 3 on a refusal/failure but still prints its JSON verdict; keep the reason (pod() would drop it)
    p = subprocess.run(PROBE_SSH + ['-n', 'python3 %s/podctl.py swap %s %s' % (PODD, C, gmu)], capture_output=True, text=True, timeout=timeout)
    try: return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception: return dict(swapped=False, reason='probe pod rc=%d %s' % (p.returncode, (p.stderr or p.stdout)[-200:]))


def busy(on, **info):
    if on: wj(BUSY, dict(at=time.time(), at_cr=cr(), **info), 0o644)
    else:
        try: BUSY.unlink()
        except FileNotFoundError: pass


def clear_own_busy():
    """t737u: the idle tick clears only the gate's own marker; offline drivers (khalf, K1 sweep, red-batch-probe) carry 'who' and
    clear their own (busy(False) itself is unchanged for them)."""
    cur = rj(BUSY)
    if isinstance(cur, dict) and cur.get('who'): return
    busy(False)


def dispatch_tasks(e):
    # t712u K=1: the trained batch is the turn UID filtered dispatch when present (else the sealed pooled DISPATCH)
    d = rj(LIFE / 'epochs' / str(e) / 'TRAIN-DISPATCH-K1.json') or rj(LIFE / 'epochs' / str(e) / 'coordinator' / 'DISPATCH.json') or {}
    return sorted({int(s['index']) for s in ((d.get('payload') or d).get('selected') or []) if 'index' in s})


# ---------------- K=1 seeded k-sample probe (Jacob t751u 2026-10-28 "for probe noise obviously do that yes") ----------------
# PROMOTION-GATE.json "k1_probe": {"from_epoch": N, "k": 4, "seeded": true}  (absent / epoch < from_epoch = original k=1 path).
# For such an epoch the head H is pre-probed by the gate itself on the full private sample S, and the candidate C post-probed on
# T = S minus the turn batch, both with k samples per task and the SAME plan seed P = sha256(sample_seed:k1-crn-plan), so every
# request (task index, sample j) carries the identical per-request vLLM seed sha256(P:env:index:j) on both sides (common random
# numbers). Decision rule unchanged (gate_core.decide on per-task pass rates: GREEN iff paired z > z_min (strict) and cap < cap_max).
def k1_probe(f, e):
    k = f.get('k1_probe')
    if not (isinstance(k, dict) and type(k.get('from_epoch')) is int and e >= k['from_epoch']): return None
    kk = int(k.get('k', 4)); assert 1 <= kk <= 16
    return dict(k=kk, seeded=k.get('seeded', True) is not False)


def plan_seed(s, kp, C):
    if kp and kp['seeded']: return hashlib.sha256(('%s:k1-crn-plan' % s['seed']).encode()).hexdigest()
    return hashlib.sha256(('%s:%s:gate-run' % (s['seed'], C)).encode()).hexdigest()


def ckpt_files(ck):
    cur = rj(LIFE / 'CURRENT.json') or {}
    if (cur.get('checkpoint') or {}).get('id') == ck: return cur['checkpoint']['files']
    for d in sorted((LIFE / 'epochs').glob('*'), key=lambda p: -int(p.name) if p.name.isdigit() else 1):
        if not d.name.isdigit(): continue
        fin = rj(d / 'FINALIZED.json')
        c = (((fin or {}).get('payload') or {}).get('result') or {}).get('checkpoint') or {}
        if c.get('id') == ck: return c['files']
    raise RuntimeError('no descriptor for head %s' % ck[:12])


def ensure_h_probe(e, s, f, kp):
    """-> per_task dict of the gate's own seeded k-sample head probe on S, or a stage string while it runs."""
    d = EP / str(e); hp = d / 'H-PROBE.private.json'
    h = rj(hp)
    if h and h.get('per_task'): return {int(k): v for k, v in h['per_task'].items()}
    if s.get('standing'):   # v2a: reuse the cached head probe on the standing sample (RED, or the GREEN candidate's own probe)
        c = rj(standing_cache_path(s['standing_id'], s['head']))
        if c and c.get('source') not in HEAD_BASELINE_SOURCES:
            log(event='head_baseline_guard', action='refused-reuse', epoch=e, head=s['head'][:12], source=c.get('source'), from_epoch=c.get('epoch')); c = None
        if c and c.get('head') == s['head'] and c.get('per_task') and c.get('k') == kp['k'] and c.get('seeded') == kp['seeded'] and \
                standing_cover(dict(id=s['standing_id'], indices=s['indices']), s['head']) >= opts()['standing_min_cover']:
            wj(hp, dict(head=s['head'], per_task=c['per_task'], k=c['k'], seeded=c['seeded'], tasks=len(s['indices']),
                        reused=True, reused_from_epoch=c.get('epoch'), reused_source=c['source'], standing_id=s['standing_id'],
                        run_s=0.0, serve_s=0.0, fetch_s=0.0, done_at=time.time(), done_at_cr=cr()))
            log(event='k1_head_probe_reused', epoch=e, head=s['head'][:12], id=s['standing_id'], from_epoch=c.get('epoch'), source=c.get('source'))
            return {int(k): v for k, v in c['per_task'].items()}
    H = s['head']; htp = d / 'H-TIMELINE.json'; htl = rj(htp) or dict(candidate=H, files=ckpt_files(H), k=kp['k'], seeded=kp['seeded'], start=time.time())
    cur = rj(BUSY)
    if isinstance(cur, dict) and cur.get('who') and time.time() - float(cur.get('at', 0)) < 180:
        wj(htp, htl); return 'waiting-pod(%s)' % cur.get('who')
    busy(True, epoch=e, candidate=H, stage='k1-head-probe')
    if not htl.get('run_done'):
        stage = fetch_and_serve(H, f, htl, None); wj(htp, htl)
        if stage != 'ready': return 'H-' + stage
    res = run_candidate(e, H, s['indices'], s, f, htl, kp=kp, side='H'); wj(htp, htl)
    if res is None: return 'H-running'
    htl.setdefault('run_done', time.time()); wj(htp, htl)
    pt = {int(k): v for k, v in (((res.get('envs') or {}).get(f['env']) or {}).get('per_task') or {}).items()}
    wj(hp, dict(head=H, per_task=pt, k=kp['k'], seeded=kp['seeded'], wall=res.get('wall_seconds'), partial=res.get('partial'),
                tasks=len(s['indices']), run_s=round(htl['run_done'] - htl.get('run_start', htl['run_done']), 1),
                serve_s=round(htl.get('served', 0) - htl.get('serve_start', htl.get('served', 0)), 1),
                fetch_s=round(htl.get('fetched', 0) - htl.get('fetch_start', htl.get('fetched', 0)), 1), done_at=time.time(), done_at_cr=cr()))
    log(event='k1_head_probe_done', epoch=e, head=H[:12], tasks=len(pt), k=kp['k'], seeded=kp['seeded'], run_s=round(htl['run_done'] - htl.get('run_start', htl['run_done']), 1))
    if s.get('standing'): standing_cache_put(s['standing_id'], H, pt, e, 'head-probe', kp, res.get('partial'))
    return pt


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
STD = ST / 'standing'


def standing_cur():
    return rj(STD / 'CURRENT.private.json')


def standing_cache_path(sid, H):
    return STD / sid / ('H-%s.private.json' % H)


def standing_cover(ss, H):
    pt = (rj(standing_cache_path(ss['id'], H)) or {}).get('per_task') or {}
    return sum(1 for t in ss['indices'] if str(t) in pt) / max(1, len(ss['indices']))


def standing_new(f, reason, old=None):
    seed = os.urandom(32).hex(); idx = G.draw_sample(seed, int(f['pool']), int(f['sample_size']))
    sid = 's' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + hashlib.sha256(seed.encode()).hexdigest()[:8]
    ss = dict(version=G.VERSION, id=sid, seed=seed, indices=idx, drawn_at=time.time(), drawn_at_cr=cr(),
              commitment=G.sample_commitment(seed, idx), reason=reason, rotated_from=(old or {}).get('id'))
    (STD / sid).mkdir(parents=True, exist_ok=True)
    wj(STD / sid / 'SAMPLE.private.json', ss, 0o600)
    if old:
        wj(STD / old['id'] / 'ROTATED.json', dict(id=old['id'], rotated_at=time.time(), rotated_at_cr=cr(), successor=sid, reason=reason), 0o644)
    wj(STD / 'CURRENT.private.json', ss, 0o600)
    log(event='standing_sample_drawn', id=sid, reason=reason, rotated_from=(old or {}).get('id'))
    return ss


HEAD_BASELINE_SOURCES = ('head-probe',)   # permanent guard (Jacob 2026-10-29 18:03 CR, bias e226-e235): a candidate's own probe is never a head baseline


def standing_cache_put(sid, H, pt, e, source, kp, partial):
    if source not in HEAD_BASELINE_SOURCES:
        log(event='head_baseline_guard', action='refused-cache', id=sid, head=H[:12], epoch=e, source=source); return False
    if partial or not pt or not (kp and kp.get('seeded')): return False
    wj(standing_cache_path(sid, H), dict(head=H, per_task={str(k): v for k, v in pt.items()}, epoch=e, source=source,
                                         k=kp['k'], seeded=True, cached_at=time.time(), cached_at_cr=cr()), 0o600)
    log(event='standing_head_probe_cached', id=sid, head=H[:12], epoch=e, source=source, tasks=len(pt))
    return True


def ensure_standing_sample(e, H, f, d, sp):
    o = opts(); ss = standing_cur()
    if not ss:
        ss = standing_new(f, 'first-use')
    elif time.time() - float(ss['drawn_at']) > o['standing_max_age_s']:
        ss = standing_new(f, 'age>%ds' % int(o['standing_max_age_s']), ss)
    elif standing_cover(ss, H) < o['standing_min_cover'] and (STD / ss['id'] / 'USED.json').exists():
        ss = standing_new(f, 'head %s must be re-probed' % H[:12], ss)   # rotate whenever the head is re-probed anyway
    if not (STD / ss['id'] / 'USED.json').exists():
        wj(STD / ss['id'] / 'USED.json', dict(id=ss['id'], first_epoch=e, first_head=H, at_cr=cr()), 0o644)
    s = dict(version=G.VERSION, epoch=e, head=H, seed=ss['seed'], indices=ss['indices'], drawn_at=ss['drawn_at'],
             drawn_at_cr=ss['drawn_at_cr'], standing=True, standing_id=ss['id'])
    wj(sp, s, 0o600)
    wj(d / 'COMMIT.json', dict(version=G.VERSION, epoch=e, head=H, sample_size=len(ss['indices']), commitment=ss['commitment'],
                               drawn_at=ss['drawn_at'], drawn_at_cr=ss['drawn_at_cr'], standing_id=ss['id'],
                               rule='commitment = sha256(version|seed|comma-joined sorted indices) of the STANDING private sample '
                                    '(reused across turns so the head probe can be reused after RED; rotated every 24 h or when the head '
                                    'must be re-probed). Seed and sample task ids are revealed after rotation at '
                                    'https://affine.io/gate/reveal/<epoch>.json'), 0o644)
    log(event='sample_standing', epoch=e, head=H[:12], id=ss['id'], n=len(ss['indices']))
    return s


def ensure_sample(e, H, f):
    d = EP / str(e); sp = d / 'SAMPLE.private.json'
    if sp.exists(): return rj(sp)
    if HO.active(f, e):
        return ensure_heldout_sample(e, H, f, d, sp)
    if opts()['standing']:
        return ensure_standing_sample(e, H, f, d, sp)
    seed = os.urandom(32).hex(); idx = G.draw_sample(seed, int(f['pool']), int(f['sample_size']))
    s = dict(version=G.VERSION, epoch=e, head=H, seed=seed, indices=idx, drawn_at=time.time(), drawn_at_cr=cr())
    wj(sp, s, 0o600)
    wj(d / 'COMMIT.json', dict(version=G.VERSION, epoch=e, head=H, sample_size=len(idx), commitment=G.sample_commitment(seed, idx),
                               drawn_at=s['drawn_at'], drawn_at_cr=s['drawn_at_cr'],
                               rule='commitment = sha256(version|seed|comma-joined sorted indices); seed and sample task ids are revealed after DECISION.json at https://affine.io/gate/reveal/<epoch>.json'), 0o644)
    log(event='sample_drawn', epoch=e, head=H[:12], n=len(idx))
    return s


def ensure_keep(e, H):
    if NO_TRAINER: return dict(ok=None, skipped='HG_NO_TRAINER')
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
def early_candidate(e, H):
    """t696u: candidate C of epoch e as STAGED on the trainer pod (before the lifecycle's readback writes TRAINING-RESULT)."""
    life = LIFE / 'epochs' / str(e); ep = EP / str(e) / 'EARLY.json'
    if (not (life / 'TRAINING-JOB.json').exists() or (life / 'TRAINING-RESULT.json').exists()
            or (life / 'FINALIZED.json').exists()): return None
    k = rj(ep)
    if k and k.get('id') and k.get('input') == H: return k
    if time.time() - _early_poll.get(e, 0) < opts()['early_poll_s']: return None
    _early_poll[e] = time.time()
    plan = rj(life / 'PLAN.json') or {}
    name = (plan.get('manifest') or {}).get('epoch')
    if not name: return None
    rc, out, err = trainer_pod('python3 %s staged %s %s' % (TRAINER_DELTA, shlex.quote(name), H), timeout=90)
    try: k = json.loads(out.strip().splitlines()[-1])
    except Exception: k = {}
    if rc != 0 or not k.get('id') or k.get('input') != H or type(k.get('files')) is not dict: return None
    k['seen_at'] = time.time(); k['seen_at_cr'] = cr(); wj(ep, k)
    log(event='gate_early_candidate', epoch=e, head=H[:12], candidate=k['id'][:12], gen=k.get('gen'), state=k.get('state'))
    return k


def _delta_status(e):
    p = EP / str(e) / 'DELTA.rc'
    try: return int(p.read_text().strip())
    except Exception: return None


def start_delta(e, C, H, tl):
    d = EP / str(e)
    for x in ('DELTA.rc',):
        try: (d / x).unlink()
        except FileNotFoundError: pass
    desc = json.dumps(dict(id=C, files=tl['files']))
    # descriptor + bounded cleanup (never the base H, the candidate or the served checkpoint)
    pod(("umask 077; mkdir -p {D}/ckpt {D}/gate; cat > {D}/ckpt/{c}.descriptor.json && "
         "for x in $(ls -t {D}/ckpt | grep -E '^[0-9a-f]{{64}}$' | tail -n +5); do [ \"$x\" != \"$(cat {D}/serve.ckpt)\" ] && [ \"$x\" != {h} ] && rm -rf {D}/ckpt/$x; done; echo ok"
         ).format(D=PODD, c=C, h=H), stdin=desc)
    make = shlex.join([RSSH, 'nice -n 10 python3 %s make %s %s' % (TRAINER_DELTA, H, C)])
    apply = shlex.join(PROBE_SSH + ['nice -n 5 python3 %s apply %s/ckpt/%s %s/ckpt %s/ckpt/%s.descriptor.json --pidfile %s/fetch.pid'
                                    % (POD_DELTA, PODD, H, PODD, PODD, C, PODD)])
    cmd = 'set -o pipefail; %s | %s; echo $? > %s' % (make, apply, shlex.quote(str(d / 'DELTA.rc')))
    lf = open(d / 'delta.log', 'ab')
    p = subprocess.Popen(['bash', '-c', cmd], stdout=lf, stderr=lf, stdin=subprocess.DEVNULL, start_new_session=True)
    tl['delta_start'] = time.time(); tl['delta_pid'] = p.pid
    log(event='gate_delta_start', epoch=e, candidate=C[:12], base=H[:12], pid=p.pid)


def delta_stage(e, C, tl, st):
    """None = delta not used / finished (caller continues); str = stage to report this tick."""
    o = opts()
    if tl.get('delta_done') or not o['delta']: return None
    if 'delta_start' not in tl:
        H = tl['head']
        if st.get('fetching'): return 'fetching-other'
        if H not in st['ckpts']:
            tl['delta_done'] = time.time(); tl['delta_rc'] = 'no-base'
            return None
        if o['delta_after_result'] and not (LIFE / 'epochs' / str(e) / 'TRAINING-RESULT.json').exists(): return 'delta-await-result'
        start_delta(e, C, H, tl); tl.setdefault('fetch_start', tl['delta_start'])
        return 'fetching-delta'
    rc = _delta_status(e)
    if rc is None:
        if time.time() - tl['delta_start'] < o['delta_timeout']: return 'fetching-delta'
        try: os.killpg(tl['delta_pid'], signal.SIGTERM)
        except Exception: pass
        try: pod('kill $(cat %s/fetch.pid) 2>/dev/null; rm -f %s/fetch.pid; echo ok' % (PODD, PODD))
        except Exception: pass
        rc = 'timeout'
    tl['delta_done'] = time.time(); tl['delta_rc'] = rc
    tail = ''
    try: tail = (EP / str(e) / 'delta.log').read_text()[-400:]
    except Exception: pass
    log(event='gate_delta_done', epoch=e, candidate=C[:12], rc=rc, seconds=round(tl['delta_done'] - tl['delta_start'], 1), tail=tail)
    return None


def fetch_and_serve(C, f, tl, e=None):
    st = podctl('status')
    if C not in st['ckpts'] and e is not None:
        ds = delta_stage(e, C, tl, st)
        if ds is not None: return ds
        if tl.get('delta_rc') == 0: st = podctl('status')
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
        # a replica that died after start (e.g. vLLM inductor-cache race) leaves serve_procs < replicas forever: re-serve
        dead = (st.get('serve_ckpt') == tl['candidate'] and st.get('serve_procs') and st.get('serve_procs') < st.get('replicas', 8)
                and time.time() - tl.get('serve_start', 0) > 150)
        if not st.get('serve_procs') or st.get('serve_ckpt') != tl['candidate'] or dead:
            if dead:
                tl['reserves'] = tl.get('reserves', 0) + 1
                tl.pop('served', None)
                log(event='gate_reserve', candidate=tl['candidate'][:12], serve_procs=st.get('serve_procs'), reserves=tl['reserves'])
            if not dead and opts()['warm_swap'] and st.get('served') and st.get('serve_procs') == st.get('replicas'):
                # t767u S1: in-place weight swap on the running replicas (podctl swap); refused/failed -> cold serve below
                t0 = time.time()
                try: sw = podctl_swap(tl['candidate'], str(f['gmu']))
                except Exception as x: sw = dict(swapped=False, reason=str(x)[-200:])
                log(event='gate_warm_swap', candidate=tl['candidate'][:12], previous=str(st.get('served'))[:12], ok=bool(sw.get('swapped')),
                    seconds=round(time.time() - t0, 1), gen=sw.get('gen'), reason=sw.get('reason'))
                if sw.get('swapped'):
                    tl['serve_start'] = t0; tl['warm_swap_s'] = round(time.time() - t0, 1)
                    return 'serving'
                tl['warm_swap_failed'] = str(sw.get('reason'))[:200]
            podctl('serve', tl['candidate'], str(f['gmu'])); tl['serve_start'] = time.time()
        return 'serving'
    tl.setdefault('served', time.time())
    return 'ready'


def run_candidate(e, C, tasks, seed, f, tl, kp=None, side='C', pseed=None):
    att = int(tl.get('run_attempt', 0))
    name = 'gate-e%06d-%s' % (e, C[:12]) + ('' if kp is None else '-%s-k%d%s' % (side, kp['k'], 's' if kp['seeded'] else 'u')) + ('-ho' + hashlib.sha256(('%s:%s' % (pseed, ','.join(map(str, sorted(tasks))))).encode()).hexdigest()[:8] if pseed else '') + ('' if att == 0 else '-r%d' % att)
    st = podctl('status'); r = st['runs'].get(name, {})
    if r.get('result'):
        res = json.loads(pod('cat %s/runs/%s/result.json' % (PODD, name)))
        got = len(((res.get('envs') or {}).get(f['env']) or {}).get('per_task') or {})
        if got < 0.9 * len(tasks) and att < 3:
            # partial run (replica died mid-run): discard and rerun the full candidate probe on fresh replicas
            log(event='gate_run_partial_retry', epoch=e, candidate=C[:12], got=got, need=len(tasks), attempt=att + 1)
            fresh = tl.get('serve_start', 0) > tl.get('run_start', 0)  # already re-served after the failed run
            tl['run_attempt'] = att + 1; tl.pop('run_start', None)
            if not fresh:
                tl.pop('served', None); podctl('serve', C, str(f['gmu'])); tl['serve_start'] = time.time()
            return None
        return res
    if not r.get('running') and 'run_start' not in tl:
        if pseed:
            pass   # held-out: fresh per-turn candidate plan seed sha256(round_seed:heldout-cand:epoch)
        elif kp is None:
            pseed = hashlib.sha256(('%s:%s:gate-run' % (seed, C)).encode()).hexdigest()
        else:
            pseed = plan_seed(seed, kp, C)   # seed = the sample dict here: H and C share P (common random numbers)
        plan = dict(run_name=name, checkpoint_id=C, seed=pseed,
                    concurrency=4096, grade_workers=160, resume=False, bandit_topk=False, concurrency_per_replica=512,
                    envs=[dict(env_id=f['env'], indices=list(tasks), k=(1 if kp is None else kp['k']), max_tokens=8192)])
        podctl('start-run', C, stdin=json.dumps(plan)); tl['run_start'] = time.time(); tl['plan_seed'] = pseed
        log(event='gate_run_started', epoch=e, candidate=C[:12], tasks=len(tasks), side=side, k=(1 if kp is None else kp['k']), seeded=(kp or {}).get('seeded'))
    elif not r.get('running') and time.time() - tl.get('run_start', 0) > 60:
        # worker died without a result: salvage graded rows
        fin = podctl('finalize-run', name, '30', timeout=90)
        log(event='gate_run_salvage', epoch=e, info=fin)
    return None


# ---------------- held-out rounds (heldout.py; Jacob pt808u 2026-10-30, steps 1-3) ----------------
def rounds():
    global ROUNDS
    if ROUNDS is None: ROUNDS = HO.Rounds(ST / 'heldout', wj, rj, log, cr)
    return ROUNDS


def ensure_heldout_sample(e, H, f, d, sp):
    c = HO.cfg(f); R = rounds(); cur = R.current(); r = R.get(cur['id']) if cur else None
    if r and r['head'] != H:
        R.close(r['id'], 'head changed %s -> %s at epoch %d (no GREEN of this round)' % (r['head'][:12], H[:12], e), e); r = None
    if r and c.get('max_round_age_s') and time.time() - float(r['opened_at']) > float(c['max_round_age_s']):
        R.close(r['id'], 'round age > %ds' % int(c['max_round_age_s']), e); r = None
    if r and c.get('max_round_turns') and sum(1 for t in R.turns(r['id']) if t.get('decision') in ('red', 'green')) >= int(c['max_round_turns']):
        # bounds the life of one baseline session (a session offset would otherwise bias a whole long round)
        R.close(r['id'], 'round reached %d decided turns without GREEN (re-baseline)' % int(c['max_round_turns']), e); r = None
    if not r:
        prev = R.get(R.all_ids()[-1]) if R.all_ids() else None
        r = R.open(G, f, c, H, e, 'first round' if not prev else 'new head %s' % H[:12], prev)
    s = dict(version=G.VERSION, epoch=e, head=H, seed=r['seed'], indices=r['indices'], drawn_at=r['opened_at'],
             drawn_at_cr=r['opened_at_cr'], heldout=True, round_id=r['id'])
    wj(sp, s, 0o600)
    wj(d / 'COMMIT.json', dict(version=G.VERSION, epoch=e, head=H, sample_size=len(r['indices']), commitment=r['commitment'],
                               drawn_at=r['opened_at'], drawn_at_cr=r['opened_at_cr'], heldout_round=r['id'],
                               rule='HELD-OUT ROUND %s: commitment = sha256(version|seed|comma-joined sorted indices) of the round\'s secret '
                                    'set (one set per head; reused by every turn of the round). Seed, task ids and all per-task grades of the '
                                    'round are revealed when the round closes (GREEN or head change) at https://affine.io/gate/heldout/<round>.json '
                                    'and https://affine.io/gate/reveal/<epoch>.json' % r['id']), 0o644)
    log(event='sample_heldout', epoch=e, head=H[:12], round=r['id'], n=len(r['indices']))
    return s


def pod_fingerprint():
    """Probe runtime identity for the cached round baseline: worker/serve/ctl code, env specs, switches, GPU UUIDs, replicas."""
    out = pod('cd %s && sha256sum probe_worker.py serve.sh podctl.py envs.json live_math_manifest_subset.json | cut -c1-16; '
              'cat PREP-CACHE.json GRADE-SETUP-ONCE.json 2>/dev/null; echo; ls WARM-SWAP 2>/dev/null; nvidia-smi -L; cat serve.replicas' % PODD, timeout=60)
    return hashlib.sha256(out.encode()).hexdigest()


def round_settings():
    """Frozen per-round sampling/runtime settings (published with the round; the fingerprint check keeps them fixed for the round)."""
    out = pod("cd %s && python3 -c \"import json;d=json.load(open('live_math_manifest_subset.json'));print(json.dumps(dict(harness=d.get('harness'),"
              "sampling_contract=d.get('sampling_contract'))))\"; sha256sum serve.sh probe_worker.py podctl.py envs.json | cut -c1-16; "
              "grep -o 'vllm serve.*' serve.sh; grep -A2 'vllm serve' serve.sh | tail -2; ls /root/affine-mimo-20261013/venv/lib/python3*/site-packages | grep -i '^vllm-'; "
              "nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c" % PODD, timeout=60)
    lines = out.strip().splitlines(); first = json.loads(lines[0])
    return dict(harness=first.get('harness'), sampling_contract=first.get('sampling_contract'), runtime=lines[1:],
                request_seed_rule='seed(task t, sample j) = int(sha256(P + ":" + env + ":" + t + ":" + j)[:12], 16); P = plan seed of the run')


def _plan(name, ck, pseed, f, indices, k, grade_workers=160):
    return dict(run_name=name, checkpoint_id=ck, seed=pseed, concurrency=4096, grade_workers=int(grade_workers), resume=False, bandit_topk=False,
                concurrency_per_replica=512, envs=[dict(env_id=f['env'], indices=list(indices), k=int(k), max_tokens=8192)])


def ensure_round_baseline(e, r, f):
    """-> merged per-task head baseline {task: cell} of the round (n = blocks*k_block per task), or a stage string while it runs.
    Runs ONCE per round on the probe pod: serve H, then `blocks` k_block runs (up to baseline_parallel at once), each on all of S."""
    c = HO.cfg(f); R = rounds(); rid = r['id']; H = r['head']; bd = R.d(rid)
    b = R.baseline(rid)
    if b: return {int(k): v for k, v in b['per_task'].items()}
    cur = rj(BUSY)
    if isinstance(cur, dict) and cur.get('who') and time.time() - float(cur.get('at', 0)) < 180:
        return 'waiting-pod(%s)' % cur.get('who')
    btp = bd / 'BASELINE-TIMELINE.json'
    btl = rj(btp) or dict(candidate=H, head=H, files=ckpt_files(H), start=time.time(), start_cr=cr(), blocks={})
    busy(True, epoch=e, candidate=H, stage='heldout-baseline', round=rid)
    stage = fetch_and_serve(H, f, btl, None); wj(btp, btl)
    if stage != 'ready': return 'baseline-' + stage
    st = podctl('status')
    running = 0; done = 0
    kept = (rj(bd / 'KEPT.private.json') or {}).get('kept')
    for bi in range(int(r['blocks'])):
        bs = btl['blocks'].setdefault(str(bi), dict(attempt=0))
        if bs.get('done'): done += 1; continue
        if bi > 0 and kept is None: continue          # v2: p_i blocks run on the kept tasks, known once the selection block is done
        tasks = r['indices'] if bi == 0 else kept
        n_need = float(c['min_cover']) * len(tasks)
        name = 'heldout-%s-H-b%d-k%d' % (rid[-8:], bi, int(r['k_block'])) + ('' if not bs['attempt'] else '-r%d' % bs['attempt'])
        ru = st['runs'].get(name, {})
        if ru.get('result'):
            res = json.loads(pod('cat %s/runs/%s/result.json' % (PODD, name)))
            pt = ((res.get('envs') or {}).get(f['env']) or {}).get('per_task') or {}
            pt = {k: v for k, v in pt.items() if isinstance(v, dict) and v.get('n')}
            if len(pt) < n_need and bs['attempt'] < 3:
                log(event='heldout_baseline_partial_retry', round=rid, block=bi, got=len(pt), need=int(n_need), attempt=bs['attempt'] + 1)
                bs['attempt'] += 1; bs.pop('start', None)
                if st.get('serve_procs', 0) < st.get('replicas', 8):
                    btl.pop('served', None); podctl('serve', H, str(f['gmu'])); btl['serve_start'] = time.time()
                wj(btp, btl); return 'baseline-reserve'
            wj(bd / ('BLOCK-%d.private.json' % bi), dict(round=rid, block=bi, head=H, plan_seed=r['baseline_plan_seeds'][bi], k=int(r['k_block']),
                                                         per_task=pt, wall=res.get('wall_seconds'), partial=res.get('partial'), run=name), 0o600)
            bs.update(done=time.time(), run=name, wall=res.get('wall_seconds'), tasks=len(pt)); done += 1
            log(event='heldout_baseline_block_done', round=rid, block=bi, tasks=len(pt), wall=res.get('wall_seconds'))
            if bi == 0:
                kept = HO.kept_tasks(pt)
                wj(bd / 'KEPT.private.json', dict(round=rid, kept=kept, rule='0 < c < n on selection block 0 (k=%d, all of S)' % int(r['k_block']),
                                                  at=time.time()), 0o600)
                log(event='heldout_filter_locked', round=rid, kept=len(kept), of=len(pt))
        elif ru.get('running'):
            running += 1
        elif 'start' not in bs:
            if running < int(c['baseline_parallel']):
                plan = _plan(name, H, r['baseline_plan_seeds'][bi], f, tasks, r['k_block'], c.get('baseline_grade_workers', 160))
                podctl('start-run', H, stdin=json.dumps(plan)); bs['start'] = time.time(); running += 1
                log(event='heldout_baseline_block_started', round=rid, block=bi, tasks=len(tasks), k=int(r['k_block']))
        elif time.time() - bs['start'] > 60:
            fin = podctl('finalize-run', name, '30', timeout=90)
            log(event='heldout_baseline_salvage', round=rid, block=bi, info=fin)
    wj(btp, btl)
    if done < int(r['blocks']): return 'baseline-running(%d/%d done, %d running)' % (done, int(r['blocks']), running)
    blocks = {str(bi): (rj(bd / ('BLOCK-%d.private.json' % bi)) or {}).get('per_task') or {} for bi in range(1, int(r['blocks']))}
    merged = HO.merged_baseline(blocks, kept)
    sel = (rj(bd / 'BLOCK-0.private.json') or {}).get('per_task') or {}
    hp = sum(v['c'] for v in merged.values()) / max(1, sum(v['n'] for v in merged.values()))
    btl['done'] = time.time()
    b = dict(schema=HO.SCHEMA, round=rid, head=H, per_task={str(k): v for k, v in merged.items()}, tasks=len(merged),
             rollouts_per_task=(int(r['blocks']) - 1) * int(r['k_block']), head_pass=hp, plan_seeds=r['baseline_plan_seeds'],
             selection=sel, kept=kept, scoring='ivw-v2', settings=round_settings(), sigma_run_est=HO.run_sigma(sel, blocks, kept),
             fingerprint=pod_fingerprint(), source='heldout-round-baseline',
             timing=dict(total_s=round(btl['done'] - btl['start'], 1),
                         serve_s=round(btl.get('served', btl['start']) - btl.get('serve_start', btl.get('served', btl['start'])), 1),
                         blocks={k: dict(run_s=round(v['done'] - v['start'], 1) if v.get('start') else None, wall=v.get('wall'))
                                 for k, v in btl['blocks'].items()}),
             done_at=btl['done'], done_at_cr=cr(btl['done']))
    b['settings_hash'] = HO.settings_hash(b['settings'])
    wj(btp, btl); wj(bd / 'BASELINE.private.json', b, 0o600)
    wj(bd / 'SETTINGS.json', dict(round=rid, settings=b['settings'], settings_hash=b['settings_hash'], kept_tasks=len(kept),
                                  rollouts_per_task_p=b['rollouts_per_task'], at_cr=cr()), 0o644)
    log(event='heldout_baseline_done', round=rid, head=H[:12], tasks=len(merged), head_pass=round(hp, 4), total_s=b['timing']['total_s'])
    return merged


def heldout_fail_closed(e, s, cand, f, why):
    d = EP / str(e); c = HO.cfg(f)
    dec = dict(version=G.VERSION, epoch=e, head=s['head'], candidate=cand['id'], decision='red', mode=mode(f),
               reasons=['heldout fail-closed: %s' % why], evaluated=dict(z=None, c_cap_rate=None, delta=None),
               z_min=float(c['bar_z']), z_inclusive=bool(c['z_inclusive']), cap_max=float(f['cap_max']), min_tasks=int(f['min_tasks']),
               decided_at=time.time(), decided_at_cr=cr(), red_credit=f.get('red_credit', 'zero'),
               commitment=(rj(d / 'COMMIT.json') or {}).get('commitment'), heldout=dict(round_id=s['round_id'], fail_closed=True))
    wj(d / 'DECISION.json', dec, 0o600); rounds().turn_add(s['round_id'], dict(epoch=e, candidate=cand['id'], decision='red', fail_closed=why, at=time.time()))
    log(event='heldout_fail_closed', epoch=e, candidate=cand['id'][:12], why=why)
    clear_own_busy()   # never drop another driver's ('who') pod marker
    return 'decided'


def skip_red(e, head, C, f, why, rid=None):
    """probe-speed 20261030 (gate owner OK): a candidate the gate will not judge (held-out round closed, candidate input != sampled
    head) is decided RED, never 'skip': the v829i lifecycle hook accepts only green/red and turns anything else into a
    timeout-promote at TRAINING-RESULT + decision_timeout_seconds (fail open). Same schema as a normal red; no promotion, head kept,
    trainer restored to the head by ensure_restore."""
    d = EP / str(e); c = HO.cfg(f)
    dec = dict(version=G.VERSION, epoch=e, head=head, candidate=C, decision='red', mode=mode(f),
               reasons=['%s -> red (not judged, no promotion)' % why], evaluated=dict(z=None, c_cap_rate=None, delta=None),
               z_min=float(c['bar_z']), z_inclusive=bool(c['z_inclusive']), cap_max=float(f['cap_max']), min_tasks=int(f['min_tasks']),
               decided_at=time.time(), decided_at_cr=cr(), red_credit=f.get('red_credit', 'zero'),
               commitment=(rj(d / 'COMMIT.json') or {}).get('commitment'), not_judged=True)
    if rid: dec['heldout'] = dict(round_id=rid, not_judged=True)
    wj(d / 'DECISION.json', dec, 0o600)
    log(event='gate_not_judged_red', epoch=e, candidate=C[:12], why=why)
    clear_own_busy()
    return 'decided'


def process_heldout_epoch(e, s, f):
    d = EP / str(e); dp = d / 'DECISION.json'; c = HO.cfg(f); R = rounds(); rid = s['round_id']; r = R.get(rid)
    cand = candidate_of(e); early = False
    if R.closed(rid):
        if cand:   # same semantics as the live head-mismatch skip
            return skip_red(e, s['head'], cand['id'], f, 'heldout round %s closed before this candidate was decided' % rid, rid)
        return 'heldout-round-closed'
    tr = LIFE / 'epochs' / str(e) / 'TRAINING-RESULT.json'
    tr_t = REPLAY_T0.get(e) or (tr.stat().st_mtime if tr.exists() else None)
    tr_age = (time.time() - tr_t) if (cand and tr_t) else None
    bl = ensure_round_baseline(e, r, f)
    if not isinstance(bl, dict):
        if tr_age is not None and tr_age > float(c['fail_closed_s']):
            return heldout_fail_closed(e, s, cand, f, 'round baseline not ready %ds after TRAINING-RESULT (%s)' % (int(tr_age), bl))
        return 'heldout-' + bl
    if not cand:
        if not opts()['early']: return 'no-candidate'
        cand = early_candidate(e, s['head'])
        if not cand: return 'no-candidate'
        early = True
    H, C = s['head'], cand['id']
    tlp = d / 'TIMELINE.json'; tl = rj(tlp)
    if tl and (tl.get('candidate') != C or tl.get('files') != cand['files']):
        log(event='gate_early_superseded', epoch=e, staged=str(tl.get('candidate'))[:12], candidate=C[:12])
        wj(d / ('TIMELINE.superseded-%d.json' % int(time.time())), tl); tl = None
    if not tl:
        tl = dict(candidate=C, head=H, files=cand['files'], early=early, staged_seen=time.time())
    if not early and 'result_mtime' not in tl:
        tl['result_seen'] = time.time(); tl['result_mtime'] = tr_t
    if not early and cand.get('input') and cand['input'] != H:
        return skip_red(e, H, C, f, 'candidate input %s != sampled head %s' % (cand['input'][:12], H[:12]))
    if not early and tr_age is not None and tr_age > float(c['fail_closed_s']):
        return heldout_fail_closed(e, s, cand, f, 'candidate probe not done %ds after TRAINING-RESULT' % int(tr_age))
    bfull = R.baseline(rid) or {}; kept = bfull.get('kept') or []; sel = bfull.get('selection') or {}
    batch = dispatch_tasks(e); T = [t for t in kept if t not in set(batch)]
    cover = sum(1 for t in T if t in bl)
    tl.setdefault('h_ready', time.time())
    busy(True, epoch=e, candidate=C)
    kp = dict(k=int(c['k_cand']), seeded=True); pseed = HO.cand_plan_seed(s['seed'])
    if not tl.get('run_done'):
        stage = fetch_and_serve(C, f, tl, e); wj(tlp, tl)
        if stage != 'ready': return stage + ('(early)' if early else '')
    res = run_candidate(e, C, T, s, f, tl, kp=kp, pseed=pseed); wj(tlp, tl)
    if res is None: return 'running-candidate' + ('(early)' if early else '')
    tl.setdefault('run_done', time.time())
    if early:
        wj(tlp, tl); busy(False); return 'staged-await-result'
    b = R.baseline(rid); fp = pod_fingerprint()
    if fp != b.get('fingerprint'):
        # the probe runtime changed since the baseline was taken: redo the baseline on the SAME set (C result is kept on the pod)
        os.replace(R.d(rid) / 'BASELINE.private.json', R.d(rid) / ('BASELINE.stale-%d.private.json' % int(time.time())))
        os.replace(R.d(rid) / 'BASELINE-TIMELINE.json', R.d(rid) / ('BASELINE-TIMELINE.stale-%d.json' % int(time.time())))
        log(event='heldout_baseline_stale', round=rid, epoch=e); return 'heldout-baseline-stale'
    tl['decide_at'] = time.time()
    if tl.get('delta_start') and not tl.get('delta_done'):
        try:
            rc = _delta_status(e)
            if rc is not None: tl['delta_rc'] = rc; tl['delta_done'] = (EP / str(e) / 'DELTA.rc').stat().st_mtime
        except Exception: pass
    cpt = {int(k): v for k, v in (((res.get('envs') or {}).get(f['env']) or {}).get('per_task') or {}).items()}
    wj(d / 'C-PROBE.private.json', dict(candidate=C, per_task=cpt, wall=res.get('wall_seconds'), partial=res.get('partial'),
                                        plan_seed=tl.get('plan_seed') or pseed, k=kp['k'], heldout_round=rid))
    wj(d / 'H-PROBE.private.json', dict(head=H, per_task=b['per_task'], k=b['rollouts_per_task'], seeded=True, source='heldout-round-baseline',
                                        heldout_round=rid, plan_seeds=b['plan_seeds'], tasks=b['tasks'], run_s=0.0, serve_s=0.0, fetch_s=0.0,
                                        baseline_done_at_cr=b['done_at_cr']))
    dec = HO.score_ivw(sel, bl, cpt, kept, batch, env=f['env'], bar_z=float(c['bar_z']), z_inclusive=bool(c['z_inclusive']),
                       cap_max=float(f['cap_max']), min_kept=int(c['min_kept']), set_size=len(s['indices']),
                       sigma_run=max(float(c['sigma_run']), float(bfull.get('sigma_run_est') or 0.0)), n_p_blocks=int(r['blocks']) - 1)
    dec['version'] = G.VERSION
    dec.update(epoch=e, head=H, candidate=C, mode=mode(f), decided_at=time.time(), decided_at_cr=cr(),
               batch_tasks=len(batch), h_cover=cover, early_staged=bool(tl.get('early')), delta_rc=tl.get('delta_rc'),
               timing=dict(result_to_decision_s=round(time.time() - tl['result_mtime'], 1),
                           h_wait_s=round(tl['h_ready'] - tl['result_seen'], 1),
                           fetch_s=round(tl.get('fetched', tl['h_ready']) - tl.get('fetch_start', tl.get('fetched', tl['h_ready'])), 1),
                           serve_s=round(tl.get('served', 0) - tl.get('serve_start', tl.get('served', 0)), 1),
                           run_s=round(tl['run_done'] - tl.get('run_start', tl['run_done']), 1)),
               red_credit=f.get('red_credit', 'zero'), commitment=r['commitment'],
               probe=dict(k=kp['k'], seeded=True, h_side='held-out round baseline (%d rollouts/task on the round set, computed once per round)' % b['rollouts_per_task'],
                          common_random_numbers=True, crn='candidate request seeds = baseline block 1 request seeds (same plan seed)',
                          h_run_s=0.0, h_serve_s=0.0, h_fetch_s=0.0),
               heldout=dict(round_id=rid, set_size=len(s['indices']), bar_z=float(c['bar_z']), z_inclusive=bool(c['z_inclusive']),
                            baseline_rollouts_per_task=b['rollouts_per_task'], k_cand=kp['k'], baseline_done_at_cr=b['done_at_cr'],
                            scoring='ivw-v2', settings_hash=b.get('settings_hash')))
    pub, priv = HO.redact_decision(dec)
    wj(tlp, tl); wj(d / 'EVALUATED.private.json', priv, 0o600); wj(dp, pub, 0o600)
    R.turn_add(rid, dict(epoch=e, candidate=C, decision=dec['decision'], z=dec['evaluated']['z'], plan_seed=pseed, batch=batch,
                         evaluated=dec['evaluated'], reasons=dec['reasons'], decided_at=dec['decided_at']))
    if dec['decision'] == 'green': R.close(rid, 'green e%d (candidate %s promoted)' % (e, C[:12]), e)
    busy(False)
    log(event='decision', epoch=e, decision=dec['decision'], mode=dec['mode'], z=round(dec['evaluated']['z'], 3), heldout_round=rid,
        bar=float(c['bar_z']), cap=dec['evaluated']['c_cap_rate'], n=dec['evaluated']['n'], reasons=dec['reasons'], timing=dec['timing'])
    return 'decided'


_HO_FORBID_OPEN = ('"seed"', '"indices"', '"per_task"', '"sample_indices"', '"head_per_task"', '"candidate_per_task"', '"blocks_per_task"')


def _ho_write_public(path, doc, open_round):
    data = json.dumps(doc, indent=1, sort_keys=True)
    if open_round:
        for bad in _HO_FORBID_OPEN:
            if bad in data: raise RuntimeError('heldout publish guard: %s in %s (round open)' % (bad, path))
    for bad in ('r2.cloudflarestorage', 'X-Amz', 'aws_secret', 'secret_access'):
        if bad in data: raise RuntimeError('heldout publish guard: %s in %s' % (bad, path))
    wj(path, doc, 0o644)


def heldout_publish():
    R = rounds(); out = SITE / 'heldout'
    ids = R.all_ids()
    if not ids: return
    # reconcile: an open round with a GREEN decision of its own (crash between DECISION and close) is closed now
    cur = R.current()
    if cur:
        for t in R.turns(cur['id']):
            dd = rj(EP / str(t.get('epoch')) / 'DECISION.json') or {}
            if dd.get('decision') == 'green' and (dd.get('heldout') or {}).get('round_id') == cur['id']:
                R.close(cur['id'], 'green e%s (reconciled)' % t['epoch'], t['epoch']); cur = None; break
    rows = []
    for rid in ids:
        r = R.get(rid); closed = rj(R.d(rid) / 'CLOSED.json'); b = R.baseline(rid); turns = R.turns(rid)
        pubc = R.public_commit(r)
        row = dict(pubc, status=('closed' if closed else 'open'), closed=closed,
                   baseline=dict(pubc['baseline'], done=bool(b), done_at_cr=(b or {}).get('done_at_cr'), head_pass_on_kept=(b or {}).get('head_pass'),
                                 kept_tasks=len((b or {}).get('kept') or []) or None, scoring=(b or {}).get('scoring'),
                                 frozen_settings=(b or {}).get('settings'), settings_hash=(b or {}).get('settings_hash')),
                   turns=[dict(epoch=t.get('epoch'), candidate=t.get('candidate'), decision=t.get('decision'), z=t.get('z')) for t in turns],
                   reveal=('https://affine.io/gate/heldout/%s.json' % rid) if closed else None)
        rows.append(row)
        if not closed:
            _ho_write_public(out / 'current.json', row, True)
            continue
        rp = out / ('%s.json' % rid)
        if rp.exists() or not b: continue
        bd = R.d(rid); blocks = {}
        for bi in range(int(r['blocks'])):
            blk = rj(bd / ('BLOCK-%d.private.json' % bi)) or {}
            blocks[str(bi)] = dict(plan_seed=blk.get('plan_seed'), k=blk.get('k'), per_task=blk.get('per_task'))
        bl = {int(k): v for k, v in b['per_task'].items()}
        tdocs = []
        for t in turns:
            e = t.get('epoch'); cp = rj(EP / str(e) / 'C-PROBE.private.json') or {}; dd = rj(EP / str(e) / 'DECISION.json') or {}
            cpt = {int(k): v for k, v in (cp.get('per_task') or {}).items()}
            re_ = HO.score_ivw(b.get('selection') or {}, bl, cpt, b.get('kept') or [], t.get('batch') or [], env=r.get('env') or 'affine_math',
                               bar_z=float(dd.get('z_min', r['bar_z'])), z_inclusive=bool(dd.get('z_inclusive', r['z_inclusive'])),
                               cap_max=float(dd.get('cap_max', 0.2)), min_kept=int(dd.get('min_tasks', 100)), set_size=len(r['indices']),
                               sigma_run=float(((t.get('evaluated') or {}).get('sigma_run')) or 0.0),
                               n_p_blocks=int(r['blocks']) - 1) if cpt else None
            tdocs.append(dict(epoch=e, candidate=t.get('candidate'), decision=t.get('decision'), fail_closed=t.get('fail_closed'),
                              plan_seed=cp.get('plan_seed'), plan_seed_rule='CRN: = baseline block 1 plan seed sha256(seed + ":heldout-baseline:1")', k=cp.get('k'),
                              batch_tasks=t.get('batch'), evaluated=t.get('evaluated'), reasons=t.get('reasons'),
                              candidate_per_task={str(k): cpt[k] for k in sorted(cpt)},
                              recomputed=(dict(decision=re_['decision'], z=re_['evaluated']['z'], matches=re_['decision'] == t.get('decision'))
                                          if re_ else None)))
        doc = dict(pubc, schema=HO.SCHEMA + '-round-reveal', status='closed', closed=closed, seed=r['seed'], indices=r['indices'],
                   checks=dict(commitment_matches=G.sample_commitment(r['seed'], r['indices']) == r['commitment'],
                               indices_match_seed=G.draw_sample(r['seed'], r['pool'], r['n']) == sorted(r['indices'])),
                   baseline=dict(rollouts_per_task=b['rollouts_per_task'], head_pass=b['head_pass'], plan_seeds=b['plan_seeds'],
                                 plan_seed_rule='block b: sha256(seed + ":heldout-baseline:" + b); block 0 = selection (all of S), blocks 1.. = p_i on kept',
                                 merged_per_task=b['per_task'], blocks=blocks, kept=b.get('kept'), settings=b.get('settings'),
                                 settings_hash=b.get('settings_hash')),
                   turns=tdocs, decision_rule='heldout.score_ivw(selection block 0, merged p_i blocks, candidate cells, kept, batch_tasks, '
                                              'bar_z, z_inclusive, cap_max, min_kept, set_size); https://affine.io/gate/heldout/heldout.py')
        _ho_write_public(rp, doc, False)
        log(event='heldout_round_revealed', round=rid, turns=len(tdocs))
    if not R.current():
        try: (out / 'current.json').unlink()
        except FileNotFoundError: pass
    idx = [{k: v for k, v in row.items() if k != 'turns'} | dict(turns=len(row['turns'])) for row in rows]
    _ho_write_public(out / 'index.json', dict(schema=HO.SCHEMA + '-index', updated_at_cr=cr(), current=(R.current() or {}).get('id'),
                                              rounds=idx[-200:]), True)


def heldout_tick(N, f):
    """Soft-revert support: when the held-out gate is switched off (flag heldout.enabled=false) the open round is closed (and so
    revealed) once no epoch of that round is still waiting for its decision (an undecided epoch keeps its round until decided)."""
    R = rounds(); cur = R.current()
    if not cur or HO.active(f, N): return
    for e in range(N - 3, N + 1):
        s = rj(EP / str(e) / 'SAMPLE.private.json') or {}
        if s.get('round_id') == cur['id'] and not (EP / str(e) / 'DECISION.json').exists() and \
                (LIFE / 'epochs' / str(e) / 'TRAINING-RESULT.json').exists() and not (LIFE / 'epochs' / str(e) / 'FINALIZED.json').exists():
            return
    R.close(cur['id'], 'held-out gate disabled (flag) at epoch %d' % N, N)


def process_epoch(e, f):
    d = EP / str(e); dp = d / 'DECISION.json'
    if dp.exists(): return 'decided'
    s = rj(d / 'SAMPLE.private.json')
    if not s: return 'no-sample'
    if s.get('heldout'):
        return process_heldout_epoch(e, s, f)
    kp = k1_probe(f, e); hk = None
    if kp is not None:
        hk = ensure_h_probe(e, s, f, kp)
        if not isinstance(hk, dict): return 'k1-' + hk
    cand = candidate_of(e); early = False
    if not cand:
        if not opts()['early']: return 'no-candidate'
        cand = early_candidate(e, s['head'])
        if not cand: return 'no-candidate'
        early = True
    H, C = s['head'], cand['id']
    tlp = d / 'TIMELINE.json'; tl = rj(tlp)
    if tl and (tl.get('candidate') != C or tl.get('files') != cand['files']):
        # t696u: the staged candidate is not the lifecycle's result -> discard the staging, start over on the real candidate
        log(event='gate_early_superseded', epoch=e, staged=str(tl.get('candidate'))[:12], candidate=C[:12])
        wj(d / ('TIMELINE.superseded-%d.json' % int(time.time())), tl); tl = None
    if not tl:
        tl = dict(candidate=C, head=H, files=cand['files'], early=early, staged_seen=time.time())
    if not early and 'result_mtime' not in tl:
        tl['result_seen'] = time.time(); tl['result_mtime'] = (LIFE / 'epochs' / str(e) / 'TRAINING-RESULT.json').stat().st_mtime
    if not early and cand.get('input') and cand['input'] != H:
        return skip_red(e, H, C, f, 'candidate input %s != sampled head %s' % (cand['input'][:12], H[:12]))
    batch = dispatch_tasks(e); T = [t for t in s['indices'] if t not in set(batch)]
    hpt = h_results(H) if kp is None else hk
    cover = sum(1 for t in T if t in hpt)
    if kp is None and cover < 0.95 * len(T):
        ds = daemon_state(H)
        if opts()['delta']:
            try:  # t696u: materialize C (CPU/disk only, never serves) while the H probe finishes; no GATE-BUSY
                st = podctl('status')
                if C not in st['ckpts']: delta_stage(e, C, tl, st)
            except Exception as ex: log(event='gate_prefetch_error', epoch=e, error=str(ex)[:200])
        if early:
            wj(tlp, tl); return 'waiting-H-probe(%s %d/%d, early)' % (ds, cover, len(T))
        tl.setdefault('h_wait_start', time.time()); wj(tlp, tl)
        if time.time() - tl['h_wait_start'] < float(f['h_wait_seconds']) and ds in ('seen', 'seeded', 'running'):
            return 'waiting-H-probe(%s %d/%d)' % (ds, cover, len(T))
        log(event='h_probe_incomplete', epoch=e, head=H[:12], cover=cover, need=len(T), daemon_state=ds)
    tl.setdefault('h_ready', time.time())
    busy(True, epoch=e, candidate=C)
    if not tl.get('run_done'):  # t696u: an early-staged run already completed for this exact C: no re-fetch/re-serve
        stage = fetch_and_serve(C, f, tl, e); wj(tlp, tl)
        if stage != 'ready': return stage + ('(early)' if early else '')
    res = run_candidate(e, C, T, (s['seed'] if kp is None else s), f, tl, kp=kp); wj(tlp, tl)
    if res is None: return 'running-candidate' + ('(early)' if early else '')
    tl.setdefault('run_done', time.time())
    if early:
        # the probe of the staged C is complete; the verdict waits for the lifecycle's TRAINING-RESULT (same C, same files)
        wj(tlp, tl); busy(False); return 'staged-await-result'
    tl['decide_at'] = time.time()
    if tl.get('delta_start') and not tl.get('delta_done'):  # t702: record the delta outcome when C was already materialized (cosmetic; was null in DECISION)
        try:
            rc = _delta_status(e)
            if rc is not None: tl['delta_rc'] = rc; tl['delta_done'] = (EP / str(e) / 'DELTA.rc').stat().st_mtime
        except Exception: pass
    cpt = {int(k): v for k, v in (((res.get('envs') or {}).get(f['env']) or {}).get('per_task') or {}).items()}
    wj(d / 'C-PROBE.private.json', dict(candidate=C, per_task=cpt, wall=res.get('wall_seconds'), partial=res.get('partial')))
    _hp = rj(d / 'H-PROBE.private.json') or {}
    if _hp.get('reused') and _hp.get('reused_source') not in HEAD_BASELINE_SOURCES:
        log(event='head_baseline_guard', action='refused-decision', epoch=e, source=_hp.get('reused_source'))
        os.replace(d / 'H-PROBE.private.json', d / ('H-PROBE.refused-%d.private.json' % int(time.time()))); busy(False); return 'head-baseline-refused'
    dec = G.decide(hpt, cpt, s['indices'], batch, z_min=float(f['z_min']), cap_max=float(f['cap_max']), min_tasks=int(f['min_tasks']),
                   z_inclusive=bool(f.get('z_inclusive')))
    dec.update(epoch=e, head=H, candidate=C, mode=mode(f), decided_at=time.time(), decided_at_cr=cr(),
               batch_tasks=len(batch), h_cover=cover,
               early_staged=bool(tl.get('early')), delta_rc=tl.get('delta_rc'),
               timing=dict(result_to_decision_s=round(time.time() - tl['result_mtime'], 1),
                           h_wait_s=round(tl['h_ready'] - tl['result_seen'], 1),
                           fetch_s=round(tl.get('fetched', tl['h_ready']) - tl.get('fetch_start', tl.get('fetched', tl['h_ready'])), 1),
                           serve_s=round(tl.get('served', 0) - tl.get('serve_start', tl.get('served', 0)), 1),
                           run_s=round(tl['run_done'] - tl.get('run_start', tl['run_done']), 1)),
               red_credit=f.get('red_credit', 'zero'), commitment=(rj(d / 'COMMIT.json') or {}).get('commitment'),
               probe=(dict(k=1, seeded=False, h_side='env-probe k=1') if kp is None else
                      dict(k=kp['k'], seeded=kp['seeded'], h_side='gate head pre-probe on S, same plan seed', common_random_numbers=kp['seeded'],
                           h_run_s=(rj(d / 'H-PROBE.private.json') or {}).get('run_s'), h_serve_s=(rj(d / 'H-PROBE.private.json') or {}).get('serve_s'),
                           h_fetch_s=(rj(d / 'H-PROBE.private.json') or {}).get('fetch_s'))))
    if s.get('standing'):
        dec['probe']['h_reused'] = bool((rj(d / 'H-PROBE.private.json') or {}).get('reused'))
        dec['standing_id'] = s['standing_id']
    wj(tlp, tl); wj(dp, dec, 0o600)
    if s.get('standing') and dec.get('decision') == 'green':
        # the promoted candidate's own probe (same standing sample, same CRN plan seed P) becomes the new head's probe
        log(event='head_baseline_guard', action='green-candidate-probe-not-cached', epoch=e, candidate=C[:12])   # the new head gets a fresh head probe
    # t737u: on RED hand the pod straight to the red-batch probe (C is still served) so offline sweeps cannot grab the gap
    # first. Decision/promotion unaffected. Marker goes stale after 180 s if that driver is not running.
    sa = rj(Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/SCORE-ALL-EPOCHS.json')) or {}
    if dec.get('decision') == 'red' and sa.get('enabled') is True and kp is None:   # K=1: no pooled red-batch credit
        busy(True, who='red-batch-probe', epoch=e, candidate=C[:12], stage='handoff-from-gate')
    else:
        busy(False)
    log(event='decision', epoch=e, decision=dec['decision'], mode=dec['mode'], z=round(dec['evaluated']['z'], 3),
        cap=dec['evaluated']['c_cap_rate'], n=dec['evaluated']['n'], reasons=dec['reasons'], timing=dec['timing'])
    return 'decided'


# ---------------- step 4: RED restore ----------------
def ensure_restore(e, f):
    if NO_TRAINER: return 'no-trainer(test)'
    d = EP / str(e); dec = rj(d / 'DECISION.json') or {}
    if dec.get('decision') != 'red': return None
    fin = rj(LIFE / 'epochs' / str(e) / 'FINALIZED.json')
    if not fin: return 'waiting-finalize'
    res = (fin.get('payload') or fin).get('result') or {}
    if not str(res.get('reason', '')).startswith('promotion-gate-red'): return 'not-gated'
    if (d / 'TRAINER-RESTORED.json').exists(): return 'restored'
    lp = d / 'RESTORE-ATTEMPTS.json'; att = rj(lp, dict(n=0, last=0))
    bg = att.get('bg')
    if bg:   # t767u S4: a background restore was launched; poll it (never launches a second one while it runs)
        rcf = d / ('restore-%d.rc' % bg['n'])
        if not rcf.exists():
            act = subprocess.run(['systemctl', '--user', 'is-active', bg['unit']], capture_output=True, text=True).stdout.strip()
            if act in ('active', 'activating', 'deactivating') or time.time() - bg['at'] < 30: return 'restoring'
        try: rc = int(rcf.read_text().strip())
        except Exception: rc = None
        try: tail = (d / ('restore-%d.out' % bg['n'])).read_text()[-600:]
        except Exception: tail = ''
        att.pop('bg'); wj(lp, att); ok = rc == 0
        log(event='trainer_restore_done', epoch=e, ok=ok, rc=rc, background=True, seconds=round(time.time() - bg['at'], 1), tail=tail)
        if ok: wj(d / 'TRAINER-RESTORED.json', dict(epoch=e, head=dec['head'], at=time.time(), at_cr=cr(), background=True), 0o644)
        return 'restored' if ok else 'restore-failed'
    if time.time() - att['last'] < 90: return 'restore-backoff'
    if opts()['restore_bg']:
        att['n'] += 1; att['last'] = time.time(); n = att['n']; unit = 'affine-gate-restore-e%d-%d' % (e, n)
        out, rcp = d / ('restore-%d.out' % n), d / ('restore-%d.rc' % n)
        cmd = ['systemd-run', '--user', '--collect', '--quiet', '--unit', unit, '--setenv=XDG_RUNTIME_DIR=' + os.environ.get('XDG_RUNTIME_DIR', '/run/user/%d' % os.getuid()),
               'bash', '-c', 'bash "$0" "$1" "$2" > "$3" 2>&1; echo $? > "$4.tmp"; mv "$4.tmp" "$4"',
               str(V / 'code' / 'trainer_restore.sh'), dec['head'], str(e), str(out), str(rcp)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if p.returncode == 0:
            att['bg'] = dict(n=n, unit=unit, at=time.time()); wj(lp, att)
            log(event='trainer_restore_start', epoch=e, head=dec['head'][:12], attempt=n, background=True, unit=unit)
            return 'restoring'
        log(event='trainer_restore_bg_launch_failed', epoch=e, attempt=n, err=(p.stdout + p.stderr)[-300:])
        att['n'] -= 1   # fall through to the synchronous restore (counts as this attempt)
    att['n'] += 1; att['last'] = time.time(); wj(lp, att)
    log(event='trainer_restore_start', epoch=e, head=dec['head'][:12], attempt=att['n'])
    p = subprocess.run(['bash', str(V / 'code' / 'trainer_restore.sh'), dec['head'], str(e)], capture_output=True, text=True, timeout=2400)
    ok = p.returncode == 0
    log(event='trainer_restore_done', epoch=e, ok=ok, tail=(p.stdout + p.stderr)[-600:])
    if ok: wj(d / 'TRAINER-RESTORED.json', dict(epoch=e, head=dec['head'], at=time.time(), at_cr=cr()), 0o644)
    return 'restored' if ok else 'restore-failed'


# ---------------- public record (commitment now, reveal after lag) ----------------
def publish(cur_epoch, lag=2):
    try: heldout_publish()
    except Exception as ex: log(event='heldout_publish_error', error=str(ex)[:300])
    entries = []
    for d in sorted(EP.glob('*'), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not d.name.isdigit(): continue
        e = int(d.name); c = rj(d / 'COMMIT.json')
        if not c: continue
        ent = dict(epoch=e, head=c['head'], sample_size=c['sample_size'], commitment=c['commitment'], drawn_at_cr=c['drawn_at_cr'])
        if c.get('heldout_round'): ent['heldout_round'] = c['heldout_round']
        dec = rj(d / 'DECISION.json')
        if dec:
            ev = dec.get('evaluated') or {}
            ent.update(candidate=dec.get('candidate'), decision=dec.get('decision'), mode=dec.get('mode'), decided_at_cr=dec.get('decided_at_cr'),
                       paired_tasks=ev.get('n'), head_pass=ev.get('h_pass'), candidate_pass=ev.get('c_pass'), paired_z=ev.get('z'),
                       candidate_cap_rate=ev.get('c_cap_rate'), reasons=dec.get('reasons'), batch_tasks_dropped=dec.get('batch_overlap'))
        # Before DECISION.json only the commitment is public; afterwards the k1 transparency sidecar reveals seed, ids and grades (/gate/reveal/).
        entries.append(ent)
    wj(SITE / 'index.json', dict(version=G.VERSION, updated_at_cr=cr(), sample_ids_published=('after-rotation (standing sample)' if opts()['standing'] else 'after-decision'),
        standing_sample=('From the first epoch whose COMMIT.json carries standing_id: one PRIVATE standing sample is reused across turns, so after a RED the '
                         'head is not re-probed (its cached probe is reused) and a promoted candidate\'s own probe on that sample (same CRN plan seed) '
                         'becomes the new head probe. The standing sample is rotated every 24 h or whenever the head must be re-probed anyway '
                         '(e.g. after a rollback); its commitment is published at first use and its seed, ids and per-task grades are revealed '
                         'only after rotation, at /gate/reveal/<epoch>.json for every epoch that used it.'), reveal_index='https://affine.io/gate/reveal/index.json',
        rule=('Each epoch draws a private uniform sample of 1000 affine_math probe tasks; tasks in the epoch\'s training batch are '
              'dropped. The candidate is promoted only if its net is positive: paired z (candidate vs head, k=1) > 0, i.e. candidate pass rate > head pass rate on the paired sample (from e168; >= -1.5 for e154-e167, > -1 for e152-e153), and candidate cap-hit rate < 0.20 (>= 200 paired tasks). '
              'RED = candidate discarded, next epoch mines from the head again. From epoch 169 scores update on pass and fail: a RED epoch is scored like a GREEN one (head probe vs the discarded candidate on its own batch tasks; Jacob t737u). Before the decision only a hash commitment '
              'of the sample is published; after DECISION.json the seed, sampled task ids and per-task head/candidate grades are revealed at https://affine.io/gate/reveal/<epoch>.json (index: /gate/reveal/index.json; verify with /gate/reveal/verify_gate_reveal.py). Full verdict ledger: https://affine.io/gate.json'), entries=entries[-200:]), 0o644)


def tick():
    f = flag(); m = mode(f)
    if m == 'off': busy(False); return 'off'
    cur = rj(LIFE / 'CURRENT.json'); N = cur['epoch_index']; H = cur['checkpoint']['id']
    out = {}
    if (LIFE / 'epochs' / str(N) / 'PLAN.json').exists():
        ensure_sample(N, H, f)
        try: heldout_tick(N, f)
        except Exception as ex: log(event='heldout_tick_error', error=str(ex)[:300])
        try: out['keep'] = ensure_keep(N, H).get('ok')
        except Exception as ex: out['keep'] = 'error %s' % ex
    any_busy = False
    for e in (N - 2, N - 1, N):
        try:
            r = process_epoch(e, f); out[str(e)] = r
            any_busy |= r not in ('decided', 'no-sample', 'no-candidate', 'skip', 'staged-await-result', 'heldout-round-closed')
        except Exception as ex:
            out[str(e)] = "error: %s" % ex; log(event='epoch_error', epoch=e, error=str(ex)[:300], tb=traceback.format_exc()[-600:])
        if m == 'live':
            try: out['restore-%d' % e] = ensure_restore(e, f)
            except Exception as ex: out['restore-%d' % e] = 'error %s' % ex
    if not any_busy: clear_own_busy()
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
        try: time.sleep(opts()['tick_s'])   # t767u S4 (default 15)
        except Exception: time.sleep(15)


if __name__ == '__main__':
    main()
