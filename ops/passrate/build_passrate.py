#!/usr/bin/env python3
"""affine.io/passrate.json: each king's pass rate over the WHOLE gate pool (gate-pool-v1: affine_math + dapo, 19,305 tasks),
measured from the prior (Jacob pt1130u 2026-11-02: "replace it with the general pass rate across our envs, measured from the prior",
replacing the affine_math-only Performance tile).

Sources, per king (head), best first:
  1. prior-pass: the prior sampler (pod affine-prior-1) runs every task of the pool k=4 times per pass, pass after pass, on the
     current head. Only COMPLETE passes are used: a pass cut short by a head change keeps the tasks that finished first (shorter,
     easier), which overstates the rate by 1-20 points. Per task p_i = correct / graded over the head's complete passes at one
     max_tokens; pass rate = mean_i p_i over all pool tasks (uniform, no weights). SE = rollout noise only (census):
     sqrt(sum_i p_i(1-p_i)/(n_i-1)) / N.
  2. gate-ipw: until a king has a complete prior pass: the gate's head baseline of each prior-v1 round (~500 tasks drawn with
     private inclusion probabilities pi_i, 12 head rollouts each), re-weighted to the pool: per env Hajek
     m_e = sum h_i/pi_i / sum 1/pi_i, combined by env size; Var_e = sum((1-pi) r^2 + pi v)/pi^2 / (sum 1/pi)^2 (the gate's own
     formula, v = h(1-h)/(k-1)); several rounds of one head at one cap are combined by inverse variance.
Public output: one aggregate per king (rate, SE, 95% CI, source, max_tokens, counts). No task ids, no per-task values, no pi,
no prior table values, no IRT parameters, no holdout. Builder reads private files read-only; writes only website/passrate.json."""
import glob, json, math, os, time
from collections import defaultdict
RA = '/home/const/subnet120-rewrite/state/root-audits'
POOL = RA + '/20261101-baseline-switch/public/gate-pool-v1.jsonl'
GRADES = RA + '/20261101-prior-table/private/table/grades.private.jsonl'
ROUNDS = RA + '/20261026-promotion-gate/state/heldout'
PERF_PRIV = RA + '/20261031-performance-eval/state/performance.private.json'
SITE = '/home/const/subnet120/affine/website'; OUT = os.environ.get('PASSRATE_OUT', SITE + '/passrate.json')
CACHE = RA + '/20261102-passrate/state/grades.cache.private.json'
# prior sampler 8,192 -> 12,288 output tokens: worker stopped 12:30:36 CR, restarted at 12K 12:33:19 CR (pod worker.log, 2026-11-02)
PRIOR_12K_T = 1793644300.0
PRIOR_K = 4


def rj(p, d=None):
    try: return json.load(open(p))
    except Exception: return d


def cr(t): return time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)', time.gmtime(t - 6 * 3600))


def pool():
    ids, env = [], {}
    for l in open(POOL):
        d = json.loads(l); ids.append(d['id']); env[d['id']] = d['env']
    return ids, env


def prior_passes(ids):
    """{head: [(cap, {task: [c, n]}) ...]} for COMPLETE passes only. Incremental: per (head, pass) sums cached by file offset."""
    C = rj(CACHE) or {}
    if C.get('file_id') != str(os.stat(GRADES).st_ino): C = {}
    off = C.get('offset', 0); P = C.get('passes', {})
    with open(GRADES, 'rb') as f:
        f.seek(off)
        for raw in f:
            if not raw.endswith(b'\n'): break
            off += len(raw)
            try: d = json.loads(raw)
            except Exception: continue
            k = '%s|%s' % (d['head'], d['pass']); p = P.setdefault(k, dict(t0=d['t'], t1=d['t'], tasks={}))
            p['t0'] = min(p['t0'], d['t']); p['t1'] = max(p['t1'], d['t'])
            x = p['tasks'].setdefault(d['task_id'], [0, 0]); x[0] += int(d.get('c') or 0); x[1] += int(d.get('n') or 0)
    tmp = CACHE + '.tmp'; fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f: json.dump(dict(file_id=str(os.stat(GRADES).st_ino), offset=off, passes=P), f)
    os.replace(tmp, CACHE)
    out = defaultdict(list)
    for k, p in P.items():
        head = k.split('|')[0]; T = p['tasks']
        if not all(t in T and T[t][1] > 0 for t in ids): continue
        cap = 8192 if p['t1'] < PRIOR_12K_T else 12288 if p['t0'] > PRIOR_12K_T else None
        if cap: out[head].append((cap, T, p['t1']))
    return out


def prior_estimate(passes, ids):
    cap = max(c for c, _, _ in passes); use = [T for c, T, _ in passes if c == cap]
    N = len(ids); s = v = 0.0; nr = 0
    for t in ids:
        c = sum(T[t][0] for T in use); n = sum(T[t][1] for T in use); p = c / n; s += p; nr += n
        if n > 1: v += p * (1 - p) / (n - 1)
    return dict(pass_rate=s / N, se=math.sqrt(v) / N, source='prior-pass', max_tokens=cap, passes=len(use), tasks=N,
                rollouts=nr, measured_at_cr=cr(max(t1 for c, _, t1 in passes if c == cap)))


def gate_rounds(env_of):
    size = defaultdict(int)
    for e in env_of.values(): size[e] += 1
    Np = sum(size.values()); out = defaultdict(list)
    for R in sorted(glob.glob(ROUNDS + '/r*')):
        P = rj(R + '/PI.private.json'); r = rj(R + '/ROUND.private.json'); b = rj(R + '/BASELINE.private.json')
        if not (P and r and b and r.get('baseline') == 'prior-v1' and r.get('ids')): continue
        pt = b.get('per_task') or {}; pi = P['pi']; per = defaultdict(list); scored = 0
        for t in r['ids']:
            c = pt.get(t)
            if not (isinstance(c, dict) and c.get('n')) or t not in env_of: continue
            per[env_of[t]].append((float(pi[t]), c['c'] / c['n'], c['n'])); scored += 1
        if set(per) != set(size) or scored < 0.95 * len(r['ids']): continue
        est = var = 0.0
        for e, L in per.items():
            Ne = sum(1 / p for p, _, _ in L); m = sum(y / p for p, y, _ in L) / Ne
            V = sum(((1 - p) * (y - m) ** 2 + p * y * (1 - y) / max(1, n - 1)) / p ** 2 for p, y, n in L) / Ne ** 2
            sh = size[e] / Np; est += sh * m; var += sh * sh * V
        out[r['head']].append(dict(est=est, var=var, cap=int(b.get('max_tokens') or 8192), n=scored, k=int(b.get('rollouts_per_task') or 0),
                                   at=float(b.get('done_at') or 0)))
    return out


def gate_estimate(rs):
    cap = max(x['cap'] for x in rs); rs = [x for x in rs if x['cap'] == cap]
    w = [1 / x['var'] for x in rs]; est = sum(wi * x['est'] for wi, x in zip(w, rs)) / sum(w)
    return dict(pass_rate=est, se=math.sqrt(1 / sum(w)), source='gate-ipw', max_tokens=cap, rounds=len(rs),
                tasks=sum(x['n'] for x in rs), rollouts=sum(x['n'] * x['k'] for x in rs), measured_at_cr=cr(max(x['at'] for x in rs)))


def kings():
    d = rj(PERF_PRIV) or rj(SITE + '/performance.json') or {}
    return sorted((k for k in d.get('kings') or [] if k.get('king') and k.get('epoch') is not None), key=lambda k: k['epoch'])


def main():
    ids, env_of = pool(); pp = prior_passes(ids); gr = gate_rounds(env_of); rows = []
    ks = kings()
    for k in ks:
        h = k['king']
        if h in pp: e = prior_estimate(pp[h], ids)
        elif h in gr: e = gate_estimate(gr[h])
        else: continue
        e['ci95'] = [e['pass_rate'] - 1.96 * e['se'], e['pass_rate'] + 1.96 * e['se']]
        for x in ('pass_rate', 'se'): e[x] = round(e[x], 5)
        e['ci95'] = [round(x, 5) for x in e['ci95']]
        rows.append(dict(king=h, epoch=k['epoch'], crowned_at_cr=k.get('crowned_at_cr'), **e))
    cur = ks[-1]['king'] if ks else None
    out = dict(version='sn120-passrate-v1', updated_at=time.time(), updated_at_cr=cr(time.time()),
               about=('Pass rate of each king (head) over the whole gate pool, all envs (gate-pool-v1: 7,492 affine_math + 11,813 dapo = '
                      '19,305 public tasks), measured from the prior: the prior sampler runs the current head on every pool task, k=4 '
                      'rollouts per task per pass; a king\'s number is the mean over all tasks of its complete passes. Until a king has a '
                      'complete pass (about an hour after it is crowned) its number is the gate\'s head baseline for that round (about 500 '
                      'tasks drawn with unequal inclusion probabilities, 12 rollouts each) re-weighted to the whole pool by 1/pi (Hajek). '
                      'Informational: the gate decides on its own paired estimate (gate.json), not on this.'),
               pool=dict(name='gate-pool-v1', tasks=len(ids), envs=dict((e, sum(1 for v in env_of.values() if v == e)) for e in sorted(set(env_of.values()))),
                         url='https://github.com/AffineFoundation/affine/blob/main/miner/data/gate-pool-v1.jsonl'),
               method=dict(
                   prior_pass='mean over all pool tasks of correct/graded over the king\'s complete prior passes at one max_tokens (passes cut short '
                              'by a head change are dropped: the tasks that finish first are the easier ones). SE = rollout noise only (every task is measured).',
                   gate_ipw='per env Hajek mean of the head baseline pass rates weighted by 1/pi, combined by env size; SE = the gate\'s sampling + rollout '
                            'variance formula; rounds of one king at one max_tokens combined by inverse variance. Inclusion probabilities are private.',
                   ci95='pass_rate +- 1.96 SE', harness='temperature 1.0, top_p 0.95, top_k 20, thinking on, gate prompt and grader; capped replies count as fails',
                   max_tokens='8,192 output tokens before the 12K switch (gate round of 2026-11-02 12:27 CR), 12,288 after; each row says its max_tokens. '
                              'King 100802ca measured on the gate draw at both caps: 79.9% (8K) vs 78.8% (12K), no visible shift.'),
               current=cur, pending=(cur if cur and not any(r['king'] == cur for r in rows) else None), kings=rows)
    s = json.dumps(out, sort_keys=True)
    for bad in ('"pi"', '"ids"', '"per_task"', '"indices"', 'math:', 'dapo:', '"prior_s"', '"a_i"', 'holdout', '.sqlite', 'X-Amz'):
        if bad in s: raise SystemExit('passrate publish guard: %s' % bad)
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f: json.dump(out, f, indent=1)
    os.chmod(tmp, 0o644); os.replace(tmp, OUT)


if __name__ == '__main__':
    main()
