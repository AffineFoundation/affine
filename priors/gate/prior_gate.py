"""SN120 held-out gate: prior-table baseline v1 hooks (RA/20261101-baseline-switch; Jacob Steeves 2026-11-01).
INERT unless PROMOTION-GATE.json heldout.baseline == "prior-v1" (and the held-out gate is active). gate_service.py calls in at four
points (round open/sample, turn, publish); the math is baseline_v1.py. Old path = default; switching the flag back closes an open
prior-v1 round and the next round opens under the old rules.

Round (one per head, as before): open -> secret + public secret_commitment, read-only SNAPSHOT of the prior table, inclusion
probabilities pi (private) -> after the opening epoch's upload deadline a future block number B is fixed and published -> once B is
finalized: seed = sha256(VERSION|seed|secret|B|hash(B)), draw (Poisson, expected N), public draw_commitment and draw size ->
head baseline: head_blocks runs x (head_rollouts / head_blocks) rollouts on every drawn task on the probe pod (once per round
when scoring_mode = "separate"; fresh for every turn when scoring_mode = "same-run") -> every candidate runs k_cand rollouts on every
drawn task (plan seed = cand_seed(seed, epoch): independent of the head blocks) -> baseline_v1.estimate (per-env Hajek Delta, z, cap) -> DECISION.
Closed rounds reveal secret, block, seed and drawn ids (/gate/heldout/<round>.json); per-task grades only for the GREEN turn.
The prior table values (s, n, pi) are never published.
weighting (heldout.prior_v1.weighting, frozen into each round at open): 'prior-v1' (default) = sqrt(E[p(1-p)]) from the prior table;
'irt-info' (live from 2026-11-02) = sqrt(2PL Fisher information) at the head's ability (baseline_v1.irt_score). The IRT a, b and the
head's theta are private like the table; if the IRT weights cannot be computed the round falls back to prior-v1 (event prior_irt_fallback)."""
from __future__ import annotations
import hashlib, json, math, os, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_v1 as BV
import heldout as HO

PASS = object()
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
PATHS = dict(table=str(RA / '20261101-prior-table/private/table/prior_table.private.sqlite'),
             pool_file=str(RA / '20261101-baseline-switch/public/gate-pool-v1.jsonl'),
             pool_url='https://github.com/AffineFoundation/affine/blob/main/miner/data/gate-pool-v1.jsonl',
             irt_params=str(RA / '20261102-irt-baseline/private/live/irt_params.private.json'),
             rpc=['https://entrypoint-finney.opentensor.ai:443'])
_POOL = {}
EPD = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/epochs')
FLAGP = Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json')


def _crs(t, fmt='%Y-%m-%d %H:%M:%S CR'):
    return time.strftime(fmt, time.gmtime(float(t) - 6 * 3600))   # Costa Rica = UTC-6, no DST


def z_bar_for(pv):
    """The z bar: fixed at bar_z (1.00) for every decision from E553 (Jacob 2026-11-02 08:20 CR)."""
    return dict(value=float(pv.get('bar_z', 1.0)), kind='fixed', from_epoch=553)


def z_bar_public(now=None):
    """The z bar for gate.json / mining.json: fixed at z > 1.00."""
    try: pv = pv_of(json.loads(FLAGP.read_text()))
    except Exception: return None
    now = time.time() if now is None else now; v = float(pv.get('bar_z', 1.0))
    return dict(current=v, kind='fixed', from_epoch=553, rule='GREEN needs z > %.2f (fixed).' % v,
                history='Fixed at z > 1.00 from E553; E513-E552 used a time-relaxed bar; the exact bar of every decision: entries[].thresholds.z_min.',
                computed_at=now, computed_at_cr=_crs(now, '%Y-%m-%d %H:%M:%S CR (UTC-6)'))


def pv_of(f) -> dict:
    h = f.get('heldout') if isinstance(f.get('heldout'), dict) else {}
    return _merge(h)


def _merge(h):
    out = dict(BV.DEFAULTS); out.update(PATHS); out.update(h.get('prior_v1') or {}); return out


def wanted(f, e) -> bool:
    h = f.get('heldout') if isinstance(f.get('heldout'), dict) else {}
    return HO.active(f, e) and h.get('baseline') == 'prior-v1'


def is_prior(r) -> bool:
    return isinstance(r, dict) and r.get('schema') == BV.ROUND_SCHEMA


# ------------------------------------------------------------------ chain (finalized block hash, JSON-RPC)
def _rpc(urls, method, params):
    err = None
    for u in urls:
        try:
            req = urllib.request.Request(u, data=json.dumps(dict(id=1, jsonrpc='2.0', method=method, params=params)).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=20) as r: d = json.loads(r.read())
            if d.get('result') is not None: return d['result']
            err = d.get('error')
        except Exception as ex: err = ex
    raise RuntimeError('chain rpc %s failed: %s' % (method, str(err)[:200]))


def finalized_number(urls) -> int:
    return int(_rpc(urls, 'chain_getHeader', [_rpc(urls, 'chain_getFinalizedHead', [])])['number'], 16)


def block_hash(urls, n: int) -> str:
    return _rpc(urls, 'chain_getBlockHash', [int(n)])


# ------------------------------------------------------------------ pool
def load_pool(pv):
    p = pv['pool_file']
    if p not in _POOL:
        raw = open(p, 'rb').read(); sha = hashlib.sha256(raw).hexdigest()
        if pv.get('pool_sha256') and sha != pv['pool_sha256']: raise RuntimeError('gate pool sha256 mismatch')
        rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
        per = {}
        for t in rows: per[t['env']] = per.get(t['env'], 0) + 1
        _POOL[p] = dict(ids=[t['id'] for t in rows], sha256=sha, per_env=per, tasks={t['id']: t for t in rows})
    return _POOL[p]


# ------------------------------------------------------------------ public texts
RULE_V1 = ('prior-v1 baseline: each round (one per head) draws about N tasks from the public gate pool (affine_math + dapo) with inclusion '
        'probability pi_i proportional to sqrt(E[p_i(1-p_i)]) under Beta(s_i+1, n_i-s_i+1) from the private prior table (s/n = the head\'s '
        'solved/sampled counts on task i from the prior sampler; x0.5 on every head change; tasks without samples start at their env '
        'average), mixed with a 10% uniform floor and capped at pi 0.10 (Poisson sampling). Seed = sha256(secret, block B, hash of B): '
        'the secret commitment is published when the round opens and B (a block after the upload deadline) before B exists; the secret '
        'is revealed when the round closes. The head is scored fresh on every drawn task (head_rollouts rollouts), every candidate with '
        'k_cand rollouts per task (independent request seeds); Delta = sum over envs of env share x Hajek '
        '(inverse-probability weighted) mean of (candidate rate - head rate); SE = sqrt(sampling + rollout variance + sigma_run^2), sigma_run = 0.003 (floor; head run-to-run noise measured ~0); '
        'GREEN iff Delta > 0 AND z = Delta/SE > 1.00 AND cap-hit < cap_max AND >= min_scored tasks scored; cap-hit (task-level, from E507; before: Hajek capped share on affine_math < 0.20) = '
        'the share of all the candidate\'s rollouts on the scored baseline tasks that hit the token cap (8,192 tokens): RED if >= 33% '
        'of the candidate\'s baseline answers hit the token cap (cap_max = 0.33). z_bar = 1.00, fixed: GREEN needs z > 1.00 (from E553; '
        'E513-E552 used a time-relaxed bar; the exact bar of every decision is in its entry, thresholds.z_min). '
        'The turn batch is not excluded. Code: https://github.com/AffineFoundation/affine/blob/main/priors/baseline_v1.py')


RULE_IRT = RULE_V1.replace(
    "with inclusion probability pi_i proportional to sqrt(E[p_i(1-p_i)]) under Beta(s_i+1, n_i-s_i+1) from the private prior table (s/n = the head's "
    "solved/sampled counts on task i from the prior sampler; x0.5 on every head change; tasks without samples start at their env average)",
    "with inclusion probability pi_i proportional to sqrt(I_i), I_i = a_i'^2 P_i (1 - P_i) = the Fisher information of task i at the head's "
    "ability theta_h under a 2PL item response model (P_i = 1/(1+exp(-a_i(theta_h - b_i))), fitted on every recorded checkpoint x task "
    "pass/fail, refit daily; a_i' = a_i shrunk halfway (log scale) toward the pool median and clipped to [0.5, 2] x median; theta_h = MAP "
    "of the head's own gate-pool rollouts; per-task a_i, b_i and theta_h are private)", 1).replace(
    'prior-v1 baseline:', 'prior-v1 baseline (irt-info weighting, live from 2026-11-02):', 1)
assert RULE_IRT != RULE_V1


def rule_for(pv):
    return RULE_IRT if (pv or {}).get('weighting') == 'irt-info' else RULE_V1


def __getattr__(name):
    """PG.RULE (gate_ledger's gate.json text) = the rule of the CURRENT flag setting (PEP 562 module attribute)."""
    if name == 'RULE':
        try: return rule_for(pv_of(json.loads(FLAGP.read_text())))
        except Exception: return RULE_V1
    raise AttributeError(name)


def settings_public(pv):
    return {k: pv.get(k) for k in ('N', 'k_cand', 'head_rollouts', 'head_blocks', 'scoring_mode', 'floor', 'cap', 'shares', 'n_init',
                                   'bar_z', 'z_inclusive', 'cap_rule', 'cap_max', 'cap_envs', 'min_scored', 'min_cover', 'block_delay', 'sigma_run_floor') if k in pv} | dict(
        tau_run=(pv['tau_run_separate'] if pv['scoring_mode'] == 'separate' else pv['tau_run_same']),
        weighting=pv.get('weighting') or 'prior-v1') | ({k: pv.get(k, BV.IRT_DEFAULTS[k]) for k in ('a_shrink', 'a_lo', 'a_hi', 'theta_prior_sd')}
                                                           if pv.get('weighting') == 'irt-info' else {})


def public_commit(r) -> dict:
    pv = r['cfg']
    return dict(schema=BV.ROUND_SCHEMA, baseline='prior-v1', round_id=r['id'], head=r['head'], opened_at=r['opened_at'],
                opened_at_cr=r['opened_at_cr'], opened_epoch=r['opened_epoch'], env=pv['probe_env'], set_size=r.get('n'),
                pool=dict(r['pool'], url=pv['pool_url']), secret_commitment=r['secret_commitment'],
                commitment=r.get('draw_commitment') or r['secret_commitment'], upload_deadline=r.get('deadline'),
                target_block=r.get('target_block'), target_block_fixed_at_cr=r.get('target_fixed_at_cr'),
                draw=dict(done=bool(r.get('ids') is not None), n=r.get('n'), expected_n=r.get('expected_n'),
                          draw_commitment=r.get('draw_commitment'), drawn_at_cr=r.get('drawn_at_cr')),
                table_snapshot_sha256=r['table']['sha256'], k_cand=pv['k_cand'], bar_z=pv['bar_z'], z_inclusive=pv['z_inclusive'],
                settings=settings_public(pv), rule=rule_for(pv),
                weighting=r.get('weighting') or 'prior-v1', irt=(r.get('irt_public') if r.get('weighting') == 'irt-info' else None))


def public_params(h: dict, cur) -> dict:
    """mining.json promotion_gate.heldout block when the prior-v1 baseline is on."""
    pv = _merge(h)
    keep = ('round_id', 'head', 'set_size', 'pool', 'env', 'commitment', 'secret_commitment', 'target_block', 'draw', 'opened_at',
            'opened_at_cr', 'opened_epoch', 'baseline', 'k_cand', 'bar_z', 'z_inclusive', 'settings', 'status', 'rule', 'weighting', 'irt')
    return dict(baseline='prior-v1', from_epoch=h.get('from_epoch'), settings_next_round=settings_public(pv),
                pool=dict(url=pv['pool_url'], manifest=pv['pool_url'].replace('gate-pool-v1.jsonl', 'gate-pool-v1.manifest.json')),
                code=dict(baseline='https://github.com/AffineFoundation/affine/blob/main/priors/baseline_v1.py',
                          gate_hooks='https://github.com/AffineFoundation/affine/blob/main/priors/gate/prior_gate.py',
                          prior_sampler='https://github.com/AffineFoundation/affine/tree/main/priors'),
                prior_method=('per task: s/n = head solved/sampled counts from the prior sampler pod (k=4 per pass over the whole pool, continuously); '
                       'x0.5 on every head change; tasks without samples start at their env average; Beta(s+1, n-s+1) posterior. '
                       'The table values are private.'),
                weighting=(pv.get('weighting') or 'prior-v1'),
                irt_method=('2PL item response model P_i(theta) = 1/(1+exp(-a_i(theta-b_i))) fitted (MAP) on every recorded checkpoint x gate-pool '
                            'task pass/fail, refit daily; draw score sqrt(a_i\'^2 P_i(1-P_i)) at the head ability theta_h (MAP of the head\'s '
                            'own gate-pool rollouts, refreshed at every round open); a_i\' = a_i shrunk halfway (log) toward the pool median, '
                            'clipped to [0.5, 2] x median. Per-task a_i, b_i, theta_h and pi_i are private. Code: '
                            'https://github.com/AffineFoundation/affine/blob/main/priors/baseline_v1.py (irt_score, theta_map), '
                            'https://github.com/AffineFoundation/affine/blob/main/priors/irt/') if pv.get('weighting') == 'irt-info' else None,
                current_round=({k: cur.get(k) for k in keep} if isinstance(cur, dict) else None),
                rounds='https://affine.io/gate/heldout/index.json', current='https://affine.io/gate/heldout/current.json', rule=rule_for(pv),
                z_bar=z_bar_public())


# ------------------------------------------------------------------ round open + draw
def _deadline(S, e):
    plan = S.rj(S.LIFE / 'epochs' / str(e) / 'PLAN.json') or {}
    m = plan.get('manifest') or {}
    m = m.get('payload', m) if isinstance(m, dict) else {}
    return m.get('deadline')


def _rjq(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def head_cells(S, pv, H, snap_path):
    """The head's own pass/fail cells on gate-pool tasks: its candidate probe(s) (prior-v1 turns), the head blocks of earlier rounds on
    this head, and prior-table cells of this head. -> ([(task_id, n, solved)], {source: rollouts})."""
    cells, src = [], {}
    def add(per_task, tag, cfg):
        for k, v in (per_task or {}).items():
            if isinstance(v, dict) and v.get('n'):
                cells.append((BV.task_of(int(k), cfg), int(v['n']), int(v.get('c') or 0))); src[tag] = src.get(tag, 0) + int(v['n'])
    eps = sorted(int(p.name) for p in EPD.iterdir() if p.name.isdigit())[-80:] if EPD.exists() else []
    for e in eps:
        d = _rjq(EPD / str(e) / 'C-PROBE.private.json')
        if d and d.get('candidate') == H and d.get('prior_v1') and not d.get('partial'): add(d.get('per_task'), 'candidate_probe', pv)
    R = S.rounds()
    for rd in sorted(R.root.glob('r2*'))[-60:]:
        rr = _rjq(rd / 'ROUND.private.json') or {}
        if rr.get('head') != H or not is_prior(rr): continue
        for b in sorted(rd.glob('BLOCK-*.private.json')): add((_rjq(b) or {}).get('per_task'), 'head_blocks', rr.get('cfg') or pv)
    try:
        import sqlite3
        con = sqlite3.connect('file:%s?mode=ro' % snap_path, uri=True)
        for t, n, c in con.execute('SELECT task_id, sum(n), sum(c) FROM cells WHERE head IN (?, ?) GROUP BY task_id', (H, H[:8])):
            if n: cells.append((t, int(n), int(c or 0))); src['prior_table'] = src.get('prior_table', 0) + int(n)
        con.close()
    except Exception: pass
    return cells, src


def irt_round(S, pv, H, rows, snap_path):
    """-> (private irt record, draw score per row). Raises if the params are missing/inconsistent (open_round falls back to prior-v1)."""
    raw = open(pv['irt_params'], 'rb').read(); P = json.loads(raw); sha = hashlib.sha256(raw).hexdigest()
    ab = {t: (float(v[0]), float(v[1])) for t, v in P['tasks'].items()}
    miss = [r['task_id'] for r in rows if r['task_id'] not in ab]
    if miss: raise RuntimeError('irt params miss %d pool tasks' % len(miss))
    k = {x: float(pv.get(x, BV.IRT_DEFAULTS[x])) for x in BV.IRT_DEFAULTS}
    cells, src = head_cells(S, pv, H, snap_path)
    th, se, nroll = BV.theta_map(cells, ab, k['theta_prior_sd']) if cells else (None, None, 0)
    if nroll >= k['theta_min_rollouts']: how = 'map of the head rollouts'
    elif H[:8] in (P.get('ckpt_theta') or {}): th, se, how = float(P['ckpt_theta'][H[:8]]), None, 'fit'
    else:
        th, se, how = None, None, None
        R = S.rounds()
        for rd in sorted(R.root.glob('r2*'), reverse=True)[:30]:
            rr = _rjq(rd / 'ROUND.private.json') or {}
            if rr.get('weighting') == 'irt-info' and (rr.get('irt') or {}).get('theta') is not None:
                th, how = float(rr['irt']['theta']), 'previous irt round %s (head %s)' % (rr['id'], rr['head'][:12]); break
        if th is None: th, how = float(P['theta_fallback']), 'fit fallback (top checkpoint)'
    th = max(-4.0, min(4.0, th))
    score = BV.irt_score(rows, ab, th, float(P['a_median']), k['a_shrink'], k['a_lo'], k['a_hi'])
    if not all(math.isfinite(x) and x >= 0 for x in score) or sum(score) <= 0: raise RuntimeError('irt score not finite/positive')
    rec = dict(params=pv['irt_params'], params_sha256=sha, fit=P.get('fit'), fitted_at_cr=P.get('fitted_at_cr'), theta=th, theta_se=se,
               theta_source=how, theta_rollouts=nroll, theta_sources=src, a_median=P['a_median'], knobs=k)
    S.log(event='prior_irt_theta', head=H[:12], theta=round(th, 4), se=(round(se, 4) if se else None), rollouts=nroll, source=how)
    return rec, score


def open_round(S, f, H, e, reason, prev=None):
    pv = pv_of(f); R = S.rounds(); R.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    secret = BV.new_secret(); now = time.time(); rid = HO.round_id(secret, now); d = R.d(rid)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    snap = BV.snapshot_table(pv['table'], str(d / 'PRIOR-SNAPSHOT.private.sqlite'))
    pool = load_pool(pv); rows = BV.load_rows(snap['path'], pool['ids'])
    weighting = 'prior-v1'; irt = None; irt_pub = None
    if pv.get('weighting') == 'irt-info':
        try:
            irt, score = irt_round(S, pv, H, rows, snap['path'])
            pi = BV.inclusion_from(rows, score, int(pv['N']), float(pv['floor']), float(pv['cap']), pv['shares'])
            weighting = 'irt-info'
            irt_pub = dict(params_sha256=irt['params_sha256'], fitted_at_cr=irt['fitted_at_cr'], fit=irt['fit'],
                           theta_source=irt['theta_source'], method='sqrt(2PL Fisher information) at the head ability; values private')
        except Exception as ex:
            irt = dict(failed=repr(ex)[:300]); pi = None
            S.log(event='prior_irt_fallback', id=rid, head=H[:12], epoch=e, why=repr(ex)[:300])
    if weighting != 'irt-info':
        pi = BV.inclusion(rows, int(pv['N']), float(pv['floor']), float(pv['cap']), pv['shares'], float(pv['n_init']))
    S.wj(d / 'PI.private.json', dict(round=rid, pi=pi, env={r['task_id']: r['env'] for r in rows}, table_digest=BV.table_digest(rows),
                                     weighting=weighting), 0o600)
    frozen = {k: v for k, v in pv.items() if k not in ('table', 'rpc')}
    r = dict(schema=BV.ROUND_SCHEMA, baseline='prior-v1', id=rid, secret=secret, secret_commitment=BV.secret_commitment(secret),
             head=H, opened_at=now, opened_at_cr=S.cr(now), opened_epoch=e, reason=reason, previous=(prev or {}).get('id'),
             deadline=_deadline(S, e), target_block=None, block_hash=None, seed=None, ids=None, indices=None, n=None,
             expected_n=round(sum(pi.values()), 2), cfg=frozen, k_cand=int(pv['k_cand']), bar_z=float(pv['bar_z']),
             z_inclusive=bool(pv['z_inclusive']), env=pv['probe_env'],
             pool=dict(sha256=pool['sha256'], tasks=len(pool['ids']), per_env=pool['per_env']),
             table=dict(sha256=snap['sha256'], rows=len(rows), digest=BV.table_digest(rows), source='read-only snapshot at round open'),
             gate_version=S.G.VERSION, weighting=weighting, irt=irt, irt_public=irt_pub)
    S.wj(d / 'ROUND.private.json', r, 0o600)
    S.wj(d / 'COMMIT.json', public_commit(r), 0o644)
    S.wj(R.root / 'CURRENT.private.json', dict(id=rid, head=H), 0o600)
    S.log(event='prior_round_open', id=rid, head=H[:12], epoch=e, expected_n=r['expected_n'], reason=reason, weighting=weighting)
    return r


def _save(S, r):
    d = S.rounds().d(r['id']); S.wj(d / 'ROUND.private.json', r, 0o600); S.wj(d / 'COMMIT.json', public_commit(r), 0o644)


def ensure_draw(S, r) -> bool:
    if r.get('ids') is not None: return True
    pv = r['cfg']; urls = PATHS['rpc'] if not pv.get('rpc') else pv['rpc']; now = time.time()
    if r.get('target_block') is None:
        if r.get('deadline') and now < float(r['deadline']): return False
        fin = finalized_number(urls)
        r.update(target_block=fin + int(pv['block_delay']), target_fixed_at=now, target_fixed_at_cr=S.cr(now), finalized_at_fix=fin)
        _save(S, r); S.log(event='prior_target_block', round=r['id'], block=r['target_block'], finalized=fin)
        return False
    if finalized_number(urls) < int(r['target_block']): return False
    bh = block_hash(urls, int(r['target_block']))
    if not bh: return False
    seed = BV.seed_from(r['secret'], int(r['target_block']), bh)
    pv_ = S.rj(S.rounds().d(r['id']) / 'PI.private.json')
    ids = BV.draw(pv_['pi'], seed)
    r.update(block_hash=bh, seed=seed, ids=ids, indices=[BV.probe_index(t, pv) for t in ids], n=len(ids), drawn_at=time.time(),
             drawn_at_cr=S.cr(), draw_commitment=BV.draw_commitment(seed, ids),
             per_env_drawn={e: sum(1 for t in ids if pv_['env'][t] == e) for e in sorted(set(pv_['env'].values()))})
    _save(S, r); S.log(event='prior_drawn', round=r['id'], n=len(ids), block=r['target_block'], per_env=r['per_env_drawn'])
    return True


def hook_sample(S, e, H, f, d, sp):
    """gate_service.ensure_heldout_sample hook. PASS = old path."""
    R = S.rounds(); cur = R.current(); r = R.get(cur['id']) if cur else None
    if not wanted(f, e):
        if is_prior(r): R.close(r['id'], 'prior-v1 baseline switched off (flag) at epoch %d' % e, e)
        return PASS
    c = HO.cfg(f)
    if r and not is_prior(r):
        R.close(r['id'], 'baseline switched to prior-v1 at epoch %d (round never revealed unless GREEN)' % e, e); r = None
    if r and r['head'] != H:
        R.close(r['id'], 'head changed %s -> %s at epoch %d (no GREEN of this round)' % (r['head'][:12], H[:12], e), e); r = None
    if r and c.get('max_round_turns') and sum(1 for t in R.turns(r['id']) if t.get('decision') in ('red', 'green')) >= int(c['max_round_turns']):
        R.close(r['id'], 'round reached %d decided turns without GREEN (re-baseline)' % int(c['max_round_turns']), e); r = None
    if not r:
        prev = R.get(R.all_ids()[-1]) if R.all_ids() else None
        r = open_round(S, f, H, e, 'first prior-v1 round' if not is_prior(prev) else 'new head %s' % H[:12], prev)
    if not ensure_draw(S, r): return None
    s = dict(version=S.G.VERSION, epoch=e, head=H, seed=r['seed'], indices=r['indices'], drawn_at=r['drawn_at'], drawn_at_cr=r['drawn_at_cr'],
             heldout=True, prior_v1=True, round_id=r['id'])
    S.wj(sp, s, 0o600)
    S.wj(d / 'COMMIT.json', dict(version=S.G.VERSION, epoch=e, head=H, sample_size=r['n'], commitment=r['draw_commitment'],
                                 secret_commitment=r['secret_commitment'], target_block=r['target_block'], drawn_at=r['drawn_at'],
                                 drawn_at_cr=r['drawn_at_cr'], heldout_round=r['id'], baseline='prior-v1',
                                 rule='PRIOR-V1 ROUND %s: draw_commitment = sha256(VERSION|seed|comma-joined sorted task ids), seed = '
                                      'sha256(VERSION|seed|secret|B|hash(B)); secret_commitment published at round open, B before it existed. '
                                      'Secret, block, seed and task ids are revealed when the round closes at '
                                      'https://affine.io/gate/heldout/<round>.json' % r['id']), 0o644)
    S.log(event='sample_prior', epoch=e, head=H[:12], round=r['id'], n=r['n'])
    return s


# ------------------------------------------------------------------ head baseline on the probe pod
def _cells_by_task(pt, pv):
    return {BV.task_of(int(k), pv): v for k, v in pt.items() if isinstance(v, dict) and v.get('n')}


def ensure_baseline(S, e, r, f, per_turn: bool):
    """-> {task_id: merged head cell} or a stage string. Runs head_blocks runs of head_rollouts/head_blocks rollouts on every drawn task."""
    pv = r['cfg']; R = S.rounds(); rid = r['id']; H = r['head']; bd = R.d(rid)
    tag = ('-e%d' % e) if per_turn else ''
    bp = bd / ('BASELINE%s.private.json' % tag); btp = bd / ('BASELINE-TIMELINE%s.json' % tag)
    b = S.rj(bp)
    if isinstance(b, dict) and b.get('per_task'): return b['per_task']
    cur = S.rj(S.BUSY)
    if isinstance(cur, dict) and cur.get('who') and time.time() - float(cur.get('at', 0)) < 180:
        return 'waiting-pod(%s)' % cur.get('who')
    now = time.time()
    btl = S.rj(btp) or dict(candidate=H, head=H, files=S.ckpt_files(H), start=now, start_cr=S.cr(now), blocks={}, last_progress=now)
    S.busy(True, epoch=e, candidate=H, stage='prior-baseline', round=rid)
    f2 = dict(f, env=pv['probe_env'])
    stage = S.fetch_and_serve(H, f2, btl, None)
    if stage != 'ready': S.wj(btp, btl); return 'baseline-' + stage
    st = S.podctl('status'); nb = int(pv['head_blocks']); kb = int(pv['head_rollouts']) // nb
    running = done = 0; need = float(pv['min_cover']) * len(r['indices'])
    for bi in range(nb):
        bs = btl['blocks'].setdefault(str(bi), dict(attempt=0))
        if bs.get('done'): done += 1; continue
        name = 'prior-%s%s-H-b%d-k%d' % (rid[-8:], tag, bi, kb) + ('' if not bs['attempt'] else '-r%d' % bs['attempt'])
        ru = st['runs'].get(name, {})
        if ru.get('result'):
            res = json.loads(S.pod('cat %s/runs/%s/result.json' % (S.PODD, name)))
            pt = ((res.get('envs') or {}).get(pv['probe_env']) or {}).get('per_task') or {}
            pt = {k: v for k, v in pt.items() if isinstance(v, dict) and v.get('n')}
            if len(pt) < need and bs['attempt'] < 3:
                S.log(event='prior_baseline_partial_retry', round=rid, block=bi, got=len(pt), need=int(need), attempt=bs['attempt'] + 1)
                bs['attempt'] += 1; bs.pop('start', None)
                if st.get('serve_procs', 0) < st.get('replicas', 8):
                    btl.pop('served', None); S.podctl('serve', H, str(f['gmu'])); btl['serve_start'] = time.time()
                S.wj(btp, btl); return 'baseline-reserve'
            S.wj(bd / ('BLOCK%s-%d.private.json' % (tag, bi)), dict(round=rid, block=bi, head=H, plan_seed=BV.plan_seed(r['seed'], bi), k=kb,
                                                                   per_task=pt, wall=res.get('wall_seconds'), run=name), 0o600)
            bs.update(done=time.time(), run=name, wall=res.get('wall_seconds'), tasks=len(pt)); done += 1; btl['last_progress'] = time.time()
            S.log(event='prior_baseline_block_done', round=rid, block=bi, tasks=len(pt), wall=res.get('wall_seconds'))
        elif ru.get('running'):
            running += 1
            pg_ = ru.get('progress') or {}; ph = '%s/%s' % (pg_.get('generated'), pg_.get('samples'))   # real progress only (progress.at ticks while stuck)
            if ph != bs.get('progress_hash'): bs['progress_hash'] = ph; btl['last_progress'] = time.time()
            if hasattr(S, 'stall_salvage'):   # grade-stall fix 20261101: no-progress finalize-run salvage (same rule as the ivw-v2 path)
                S.stall_salvage(bs, name, ru, len(r['indices']) * kb, 'prior_baseline_stall_salvage', round=rid, block=bi)
        elif 'start' not in bs:
            if running < int(pv['baseline_parallel']):
                plan = dict(run_name=name, checkpoint_id=H, seed=BV.plan_seed(r['seed'], bi), concurrency=4096, grade_workers=160, resume=False,
                            bandit_topk=False, concurrency_per_replica=512,
                            envs=[dict(env_id=pv['probe_env'], indices=list(r['indices']), k=kb, max_tokens=8192)])
                S.podctl('start-run', H, stdin=json.dumps(plan)); bs['start'] = time.time(); running += 1; btl['last_progress'] = time.time()
                S.log(event='prior_baseline_block_started', round=rid, block=bi, tasks=len(r['indices']), k=kb)
        elif time.time() - bs['start'] > 60:
            fin = S.podctl('finalize-run', name, '30', timeout=90)
            S.log(event='prior_baseline_salvage', round=rid, block=bi, info=fin)
    S.wj(btp, btl)
    if done < nb: return 'baseline-running(%d/%d done, %d running)' % (done, nb, running)
    blocks = [(S.rj(bd / ('BLOCK%s-%d.private.json' % (tag, bi))) or {}).get('per_task') or {} for bi in range(nb)]
    merged = {}
    for k in set().union(*[set(x) for x in blocks]):
        cs = [x.get(k) for x in blocks]
        if all(isinstance(c, dict) and c.get('n') for c in cs): merged[k] = HO.merge_cells(cs)
    per_task = _cells_by_task(merged, pv)
    btl['done'] = time.time()
    hp = sum(v['c'] for v in per_task.values()) / max(1, sum(v['n'] for v in per_task.values()))
    b = dict(schema=BV.ROUND_SCHEMA, round=rid, head=H, epoch=(e if per_turn else None), per_task=per_task, tasks=len(per_task),
             rollouts_per_task=kb * nb, head_pass_raw=hp, plan_seeds=[BV.plan_seed(r['seed'], bi) for bi in range(nb)],
             settings=S.round_settings(), fingerprint=S.pod_fingerprint(), source='prior-v1-head-baseline',
             timing=dict(total_s=round(btl['done'] - btl['start'], 1)), done_at=btl['done'], done_at_cr=S.cr(btl['done']))
    b['settings_hash'] = HO.settings_hash(b['settings'])
    S.wj(btp, btl); S.wj(bp, b, 0o600)
    S.log(event='prior_baseline_done', round=rid, epoch=e, tasks=len(per_task), head_pass_raw=round(hp, 4), total_s=b['timing']['total_s'])
    return per_task


# ------------------------------------------------------------------ one turn
def process_epoch(S, e, s, f):
    d = S.EP / str(e); dp = d / 'DECISION.json'; c = HO.cfg(f); R = S.rounds(); rid = s['round_id']; r = R.get(rid); pv = r['cfg']
    cand = S.candidate_of(e); early = False
    if R.closed(rid):
        if cand: return S.skip_red(e, s['head'], cand['id'], f, 'prior-v1 round %s closed before this candidate was decided' % rid, rid)
        return 'heldout-round-closed'
    tr = S.LIFE / 'epochs' / str(e) / 'TRAINING-RESULT.json'
    tr_t = S.REPLAY_T0.get(e) or (tr.stat().st_mtime if tr.exists() else None)
    tr_age = (time.time() - tr_t) if (cand and tr_t) else None
    per_turn = pv['scoring_mode'] == 'same-run'
    if per_turn and not cand and not S.opts()['early']: return 'no-candidate'
    bl = ensure_baseline(S, e, r, f, per_turn)
    if not isinstance(bl, dict):
        if tr_age is not None and tr_age > float(c['fail_closed_s']):
            return S.heldout_fail_closed(e, s, cand, f, 'prior-v1 baseline not ready %ds after TRAINING-RESULT (%s)' % (int(tr_age), bl))
        return 'prior-' + bl
    if not cand:
        if not S.opts()['early']: return 'no-candidate'
        cand = S.early_candidate(e, s['head'])
        if not cand: return 'no-candidate'
        early = True
    H, C = s['head'], cand['id']
    tlp = d / 'TIMELINE.json'; tl = S.rj(tlp)
    if tl and (tl.get('candidate') != C or tl.get('files') != cand['files']):
        S.wj(d / ('TIMELINE.superseded-%d.json' % int(time.time())), tl); tl = None
    if not tl: tl = dict(candidate=C, head=H, files=cand['files'], early=early, staged_seen=time.time())
    if not early and 'result_mtime' not in tl: tl['result_seen'] = time.time(); tl['result_mtime'] = tr_t
    if not early and cand.get('input') and cand['input'] != H:
        return S.skip_red(e, H, C, f, 'candidate input %s != sampled head %s' % (cand['input'][:12], H[:12]))
    if not early and tr_age is not None and tr_age > float(c['fail_closed_s']):
        return S.heldout_fail_closed(e, s, cand, f, 'candidate probe not done %ds after TRAINING-RESULT' % int(tr_age))
    tl.setdefault('h_ready', time.time())
    S.busy(True, epoch=e, candidate=C)
    f2 = dict(f, env=pv['probe_env'])
    kp = dict(k=int(pv['k_cand']), seeded=True); pseed = BV.cand_seed(r['seed'], e)
    if not tl.get('run_done'):
        stage = S.fetch_and_serve(C, f2, tl, e); S.wj(tlp, tl)
        if stage != 'ready': return stage + ('(early)' if early else '')
    res = S.run_candidate(e, C, list(r['indices']), s, f2, tl, kp=kp, pseed=pseed); S.wj(tlp, tl)
    if res is None: return 'running-candidate' + ('(early)' if early else '')
    tl.setdefault('run_done', time.time())
    if early: S.wj(tlp, tl); S.busy(False); return 'staged-await-result'
    bfn = R.d(rid) / ('BASELINE%s.private.json' % (('-e%d' % e) if per_turn else ''))
    b = S.rj(bfn) or {}
    if not per_turn and S.pod_fingerprint() != b.get('fingerprint'):
        os.replace(bfn, R.d(rid) / ('BASELINE.stale-%d.private.json' % int(time.time())))
        tp = R.d(rid) / 'BASELINE-TIMELINE.json'
        if tp.exists(): os.replace(tp, R.d(rid) / ('BASELINE-TIMELINE.stale-%d.json' % int(time.time())))
        S.log(event='prior_baseline_stale', round=rid, epoch=e); return 'prior-baseline-stale'
    tl['decide_at'] = time.time()
    cpt_i = {int(k): v for k, v in (((res.get('envs') or {}).get(pv['probe_env']) or {}).get('per_task') or {}).items()}
    cpt = _cells_by_task(cpt_i, pv)
    PI = S.rj(R.d(rid) / 'PI.private.json')
    drawn = {t: dict(env=PI['env'][t], pi=PI['pi'][t]) for t in r['ids']}
    tau = float(pv['tau_run_same'] if per_turn else pv['tau_run_separate'])
    zb = z_bar_for(pv_of(f)); zbar = float(zb['value'])   # z bar: fixed 1.00
    dec = BV.estimate(drawn, bl, cpt, r['pool']['per_env'], shares=pv['shares'], tau=tau, sigma_run_floor=float(pv['sigma_run_floor']), bar_z=zbar,
                      z_inclusive=bool(pv['z_inclusive']), cap_max=float(pv_of(f).get('cap_max', BV.CAP_HIT_MAX)),   # task-level cap rule: live flag value
                      min_scored=int(pv['min_scored']), min_cover=float(pv['min_cover']))
    S.wj(d / 'C-PROBE.private.json', dict(candidate=C, per_task=cpt_i, wall=res.get('wall_seconds'), partial=res.get('partial'),
                                          plan_seed=tl.get('plan_seed') or pseed, k=kp['k'], heldout_round=rid, prior_v1=True))
    S.wj(d / 'H-PROBE.private.json', dict(head=H, per_task={str(BV.probe_index(t, pv)): v for t, v in bl.items()}, k=b.get('rollouts_per_task'),
                                          seeded=True, source='prior-v1-head-baseline', heldout_round=rid, plan_seeds=b.get('plan_seeds'),
                                          tasks=len(bl), baseline_done_at_cr=b.get('done_at_cr'), prior_v1=True))
    dec.update(version=S.G.VERSION, epoch=e, head=H, candidate=C, mode=S.mode(f), decided_at=time.time(), decided_at_cr=S.cr(),
               batch_tasks=None, early_staged=bool(tl.get('early')), delta_rc=tl.get('delta_rc'),
               timing=dict(result_to_decision_s=round(time.time() - tl['result_mtime'], 1), h_wait_s=round(tl['h_ready'] - tl['result_seen'], 1),
                           fetch_s=round(tl.get('fetched', tl['h_ready']) - tl.get('fetch_start', tl.get('fetched', tl['h_ready'])), 1),
                           serve_s=round(tl.get('served', 0) - tl.get('serve_start', tl.get('served', 0)), 1),
                           run_s=round(tl['run_done'] - tl.get('run_start', tl['run_done']), 1)),
               red_credit=f.get('red_credit', 'zero'), commitment=r['draw_commitment'], z_bar=zb,
               probe=dict(k=kp['k'], seeded=True, common_random_numbers=False, crn='none: candidate plan seed = sha256(seed:prior-baseline:cand:<epoch>), independent of the head blocks',
                          h_side='prior-v1 head baseline (%s rollouts/task on every drawn task, %s)' % (b.get('rollouts_per_task'),
                                 'fresh this turn' if per_turn else 'once per round')),
               heldout=dict(round_id=rid, baseline='prior-v1', set_size=r['n'], bar_z=zbar, z_inclusive=bool(pv['z_inclusive']),
                            k_cand=kp['k'], head_rollouts=b.get('rollouts_per_task'), scoring_mode=pv['scoring_mode'], tau_run=tau, sigma_run_floor=float(pv['sigma_run_floor']),
                            baseline_done_at_cr=b.get('done_at_cr'), scoring=BV.VERSION, settings_hash=b.get('settings_hash')))
    BV.assert_public(dict(evaluated=dec['evaluated'], reasons=dec['reasons'], heldout=dec['heldout']), True)
    S.wj(tlp, tl); S.wj(d / 'EVALUATED.private.json', dict(evaluated=dec['evaluated'], drawn=len(drawn)), 0o600); S.wj(dp, dec, 0o600)
    R.turn_add(rid, dict(epoch=e, candidate=C, decision=dec['decision'], z=dec['evaluated']['z'], plan_seed=pseed, batch=[],
                         evaluated=dec['evaluated'], reasons=dec['reasons'], decided_at=dec['decided_at'], prior_v1=True))
    if dec['decision'] == 'green': R.close(rid, 'green e%d (candidate %s promoted)' % (e, C[:12]), e)
    S.busy(False)
    S.log(event='decision', epoch=e, decision=dec['decision'], mode=dec['mode'], z=round(dec['evaluated']['z'], 3), heldout_round=rid,
          baseline='prior-v1', bar=zbar, z_bar=zb, gain=dec['evaluated']['gain'], cap=dec['evaluated']['c_cap_rate'], cap_rule='task-level', cap_max=dec['cap_max'],
          h_cap=dec['evaluated']['h_cap_rate'], per_env_cap={k: dict(h=v['h_cap'], c=v['c_cap']) for k, v in dec['evaluated']['per_env'].items()},
          n=dec['evaluated']['n'], reasons=dec['reasons'], timing=dec['timing'])
    return 'decided'


# ------------------------------------------------------------------ publish
def publish_round(S, R, rid, r, out):
    closed = S.rj(R.d(rid) / 'CLOSED.json'); turns = R.turns(rid); pv = r['cfg']
    gt = S._green_turn(closed, turns)
    row = dict(public_commit(r), status=('closed' if closed else 'open'), closed=closed,
               turns=[dict(epoch=t.get('epoch'), candidate=t.get('candidate'), decision=t.get('decision'), z=t.get('z'),
                           gain=(t.get('evaluated') or {}).get('gain')) for t in turns],
               reveal=('https://affine.io/gate/heldout/%s.json' % rid) if closed and r.get('ids') is not None else None)
    if not closed:
        BV.assert_public(row, True); S._ho_write_public(out / 'current.json', row, True); return row
    rp = out / ('%s.json' % rid)
    if r.get('ids') is None or rp.exists(): return row
    pool = load_pool(pv)
    doc = dict(public_commit(r), schema=BV.ROUND_SCHEMA + '-round-reveal', status='closed', closed=closed, secret=r['secret'],
               block_number=r['target_block'], block_hash=r['block_hash'], seed=r['seed'], drawn_ids=r['ids'],
               checks=dict(secret_commitment_matches=BV.secret_commitment(r['secret']) == r['secret_commitment'],
                           seed_matches=BV.seed_from(r['secret'], r['target_block'], r['block_hash']) == r['seed'],
                           draw_commitment_matches=BV.draw_commitment(r['seed'], r['ids']) == r['draw_commitment']),
               tasks=[dict(id=t, env=pool['tasks'][t]['env'], problem=pool['tasks'][t]['problem'], answer=pool['tasks'][t]['answer'])
                      for t in r['ids'] if t in pool['tasks']],
               turns=row['turns'], note=('Inclusion probabilities come from the private prior table (snapshot sha256 above) and are not '
                                         'published; u_i = sha256(VERSION|draw|seed|task_id) is public, so anyone can check that every drawn '
                                         'task has u_i below its (private) pi_i bound of cap 0.10.'))
    if gt:
        b = S.rj(R.d(rid) / 'BASELINE.private.json') or {}
        e = gt.get('epoch'); cp = S.rj(S.EP / str(e) / 'C-PROBE.private.json') or {}
        hb = S.rj(R.d(rid) / ('BASELINE-e%d.private.json' % e)) if pv['scoring_mode'] == 'same-run' else b
        cpt = _cells_by_task({int(k): v for k, v in (cp.get('per_task') or {}).items()}, pv)
        doc['green_turn'] = dict(epoch=e, candidate=gt.get('candidate'), evaluated=gt.get('evaluated'), reasons=gt.get('reasons'),
                                 plan_seed=cp.get('plan_seed'), k=cp.get('k'), head_rollouts=(hb or {}).get('rollouts_per_task'),
                                 head_plan_seeds=(hb or {}).get('plan_seeds'), head_per_task=(hb or {}).get('per_task'),
                                 candidate_per_task=cpt)
    BV.assert_public(doc, False); S._ho_write_public(rp, doc, False)
    S.log(event='prior_round_revealed', round=rid, green=bool(gt))
    return row
