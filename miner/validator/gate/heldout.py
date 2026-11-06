"""SN120 held-out-set promotion gate: committed secret set per round (Jacob pt808u, 2026-10-30; steps 1-3, no confirm set / two tier).

Round = one head. A round opens when the gate first sees a head with no open round (initially, after every GREEN promotion,
and after any other head change such as a rollback):
  * secret seed = os.urandom(32); set S = gate_core.draw_sample(seed, pool, n) (uniform, without replacement)
  * public commitment = gate_core.sample_commitment(seed, S) = sha256("<gate version>|<seed hex>|<comma-joined sorted ids>")
    (the same rule the per-epoch gate commitments already use, so /gate/reveal/verify_gate_reveal.py checks it unchanged)
  * seed + S live only in the gate private state (dir 0700, files 0600). Only round id, n, head, commitment and times are public.
Baseline (once per round): the head is probed on S in `blocks` independent runs of `k_block` samples per task (default 4 x 4 = 16
rollouts per task). Block b uses plan seed sha256("<seed>:heldout-baseline:<b>"). The merged cells (n = 16 per task) are the
round baseline; it is never recomputed for the round (unless the probe pod runtime fingerprint changes: then it is redone).
Turn (every candidate C of the round): C is probed on T = S minus the turn batch tasks with k_cand samples per task and plan
seed sha256("<seed>:heldout-cand:<epoch>") (fresh per turn). d_t = rate_C(t) - rate_H16(t); z = sum d / sqrt(sum d^2)
(gate_core.paired, unchanged). GREEN iff z >= bar_z (default 1.65, one-sided 5 %) and cap-hit < cap_max and n >= min_tasks.
GREEN closes the round: seed, S, the baseline blocks and every candidate's per-task cells of the round become public.
While a round is open nothing per-task (and nothing that reveals |S ∩ batch|, such as n or batch overlap) is published.
Pure functions + private-state helpers; the gate service owns all pod I/O."""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path

SCHEMA = 'sn120-heldout-gate-v2'
DEFAULTS = dict(enabled=False, from_epoch=None, n=1000, blocks=4, k_block=4, k_cand=4, bar_z=1.0, z_inclusive=False,
                baseline_parallel=3, baseline_max_s=3000, fail_closed_s=2400, fetch_fail_closed_s=900, max_round_age_s=None, max_round_turns=8, min_cover=0.95,
                min_kept=100, scoring='ivw-v2', sigma_run=0.022,
                sigma_run_policy='per-round', sigma_pool_rounds=20, sigma_pool_min_rounds=5)
# variance levers (Jacob 2026-10-31 14:07 CR): sigma_run_policy 'pooled' = sigma_run of a round is the pooled run-level sd of the
# last sigma_pool_rounds finished baselines (pooled_sigma_run), frozen in ROUND.private.json when the round opens (max with the
# floor sigma_run); 'per-round' (old) = max(floor, this round's own 2-df estimate run_sigma). k_cand is also read from the round.
# sigma_run (Jacob t820u quick test, 2026-10-30): head runs on the kept tasks differ run-to-run by more than per-task sampling noise
# (7 real k4 runs of one head on the same 201 kept tasks: pass sd 3.1 pp vs 1.6 pp expected; cap-hit 0.28-0.34). The SE therefore
# carries a run-level term: se_total = sqrt(se_task^2 + sigma_run^2 * (1 + 1/n_p_blocks)), sigma_run = max(cfg floor, per-round
# estimate from the p_i blocks).
# v2 (Jacob t818u/t819u/t820u): block 0 = SELECTION run (k_block on all of S) -> locks the filter (keep 0 < c < n) and the weights;
# blocks 1..blocks-1 = p_i runs on the kept tasks only (independent of the selection, so no selection bias); every candidate of the
# round runs k_cand on the kept tasks with the SAME request seeds as block 1 (common random numbers, seed = H(plan seed of block 1,
# env, task, j)). Score = inverse-variance-weighted per-task gain vs p_i; z = gain / empirical SE; GREEN iff gain > 0 and z > bar_z.


def cfg(flag: dict) -> dict:
    h = flag.get('heldout') if isinstance(flag.get('heldout'), dict) else {}
    c = dict(DEFAULTS, **h)
    c['enabled'] = h.get('enabled') is True
    return c


def active(flag: dict, e: int) -> bool:
    c = cfg(flag)
    return c['enabled'] and type(c.get('from_epoch')) is int and e >= c['from_epoch']


def baseline_plan_seed(seed: str, b: int) -> str:
    return hashlib.sha256(('%s:heldout-baseline:%d' % (seed, b)).encode()).hexdigest()


def cand_plan_seed(seed: str, e: int = None) -> str:
    """v2: common random numbers for the whole round: every candidate uses the plan seed of baseline block 1 (the first p_i block),
    so its request seeds sha256(P:env:task:j) equal that block's (epoch-independent)."""
    return baseline_plan_seed(seed, 1)


def kept_tasks(sel: dict) -> list:
    """Filter locked from the selection block only: keep tasks the head neither always nor never solved (0 < c < n)."""
    return sorted(int(t) for t, v in sel.items() if isinstance(v, dict) and v.get('n') and 0 < int(v['c']) < int(v['n']))


def weight(sel_cell: dict) -> float:
    q = (int(sel_cell['c']) + 1) / (int(sel_cell['n']) + 2)
    return 1.0 / (q * (1.0 - q))


def score_ivw(sel: dict, base: dict, cand: dict, kept, batch, *, env: str, bar_z=1.0, z_inclusive=False, cap_max=0.20, min_kept=100,
              set_size=None, sigma_run=0.0, n_p_blocks=3) -> dict:
    """Inverse-variance-weighted per-task gain of the candidate vs the head p_i on the kept tasks minus the batch.
    g_t = rate_C(t) - p_t ; w_t = 1/(q(1-q)), q = (c_sel+1)/(n_sel+2) ; gain = sum w g / sum w ; se = sqrt(sum w^2 g^2) / sum w
    (empirical, the same form as gate_core.paired's sqrt(sum d^2)) ; se = sqrt(se_task^2 + sigma_run^2 (1 + 1/n_p_blocks)) (run-level
    noise, see DEFAULTS); z = gain / se. GREEN iff gain > 0, z > bar_z (>= if inclusive),
    estimated whole-set cap-hit < cap_max and n >= min_kept. Keys are str or int.
    Cap: the kept tasks are the hard ones (head cap-hit ~0.30 there vs ~0.14 on the whole set), so the whole-set cap-hit of the candidate
    is estimated as head whole-set cap-hit (selection block, all of S) + (candidate - head cap-hit on the paired kept tasks) * kept/|S|
    (tasks outside the kept set are assumed unchanged)."""
    def g(dct, t): return dct.get(t) if t in dct else dct.get(str(t))
    bset = set(int(t) for t in batch); T = [int(t) for t in kept if int(t) not in bset]
    num = W = s2 = 0.0; n = 0; hs = cs = 0.0; cap_b = cap_n = hcap_b = hcap_n = 0; gains = losses = 0
    for t in T:
        sc, hc, cc = g(sel, t), g(base, t), g(cand, t)
        if not (isinstance(sc, dict) and isinstance(hc, dict) and isinstance(cc, dict) and sc.get('n') and hc.get('n') and cc.get('n')): continue
        w = weight(sc); xh = hc['c'] / hc['n']; xc = cc['c'] / cc['n']; d = xc - xh
        num += w * d; W += w; s2 += w * w * d * d; n += 1; hs += xh; cs += xc
        gains += xh < 0.5 <= xc; losses += xc < 0.5 <= xh
        cap_b += int(cc.get('b') or 0); cap_n += int(cc.get('n') or 0); hcap_b += int(hc.get('b') or 0); hcap_n += int(hc.get('n') or 0)
    gain = num / W if W else None; se_task = (s2 ** 0.5) / W if W else None
    se = ((se_task ** 2 + float(sigma_run) ** 2 * (1 + 1 / max(1, int(n_p_blocks)))) ** 0.5) if se_task is not None else None
    z = (gain / se) if (se and se > 0) else 0.0
    c_cap_k = (cap_b / cap_n) if cap_n else None; h_cap_k = (hcap_b / hcap_n) if hcap_n else None
    sb = sum(int(v.get('b') or 0) for v in sel.values() if isinstance(v, dict)); sn = sum(int(v.get('n') or 0) for v in sel.values() if isinstance(v, dict))
    h_cap_whole = sb / sn if sn else None; N = set_size or len(sel)
    cap = (h_cap_whole + (c_cap_k - h_cap_k) * len(kept) / N) if None not in (c_cap_k, h_cap_k, h_cap_whole) else None
    ev = dict(z=z, gain=gain, se=se, se_task=se_task, sigma_run=float(sigma_run), n_p_blocks=int(n_p_blocks), delta=gain, n=n, kept=len(kept), h_pass=(hs / n if n else None), c_pass=(cs / n if n else None),
              gains=gains, losses=losses, c_cap_rate=cap, c_cap_rate_kept=c_cap_k, h_cap_rate_kept=h_cap_k,
              h_cap_rate_set=h_cap_whole, weighting='1/(q(1-q)), q=(c_sel+1)/(n_sel+2)',
              per_env={env: dict(gain=gain, se=se, z=z, n=n)})
    reasons = []
    if n < min_kept: reasons.append('insufficient_kept_tasks(%d<%d)' % (n, min_kept))
    if gain is None or not gain > 0: reasons.append('gain %s <= 0' % (None if gain is None else round(gain, 5)))
    if not ((z >= bar_z) if z_inclusive else (z > bar_z)): reasons.append('weighted_z %.3f %s %.2f' % (z, '<' if z_inclusive else '<=', bar_z))
    if cap is None or not cap < cap_max: reasons.append('cap_rate %s >= %.2f' % (None if cap is None else round(cap, 4), cap_max))
    return dict(decision=('red' if reasons else 'green'), reasons=reasons, evaluated=ev, sample_size=len(kept),
                batch_overlap=len(kept) - len(T), batch_pre_post=None, z_min=bar_z, z_inclusive=bool(z_inclusive), cap_max=cap_max,
                min_tasks=min_kept, scoring='ivw-v2')


def run_sigma(sel: dict, blocks: dict, kept) -> float | None:
    """Per-round estimate of the run-level sd of the weighted mean rate on the kept tasks, from the p_i blocks:
    sample variance of the block weighted means minus their mean within-block sampling variance (floored at 0)."""
    ms, vs = [], []
    for pt in blocks.values():
        num = W = v = 0.0
        for t in kept:
            sc = sel.get(t) if t in sel else sel.get(str(t)); c = pt.get(t) if t in pt else pt.get(str(t))
            if not (isinstance(sc, dict) and isinstance(c, dict) and c.get('n')): continue
            w = weight(sc); x = c['c'] / c['n']; num += w * x; W += w; v += w * w * x * (1 - x) / max(1, c['n'] - 1)
        if W: ms.append(num / W); vs.append(v / (W * W))
    if len(ms) < 2: return None
    m = sum(ms) / len(ms); sv = sum((x - m) ** 2 for x in ms) / (len(ms) - 1)
    return max(0.0, sv - sum(vs) / len(vs)) ** 0.5


SIGMA_POOLED = 'pooled-v1'


def round_excess_var(sel: dict, blocks: dict, kept):
    """One round's run-level evidence from its p_i blocks: between-block sample variance of the block weighted means minus the
    mean within-block sampling variance (NOT floored), the within-block variance and the degrees of freedom (blocks - 1)."""
    ms, vs = [], []
    for pt in blocks.values():
        num = W = v = 0.0
        for t in kept:
            sc = sel.get(t) if t in sel else sel.get(str(t)); c = pt.get(t) if t in pt else pt.get(str(t))
            if not (isinstance(sc, dict) and isinstance(c, dict) and c.get('n')): continue
            w = weight(sc); x = c['c'] / c['n']; num += w * x; W += w; v += w * w * x * (1 - x) / max(1, c['n'] - 1)
        if W: ms.append(num / W); vs.append(v / (W * W))
    if len(ms) < 2: return None
    m = sum(ms) / len(ms); sv = sum((x - m) ** 2 for x in ms) / (len(ms) - 1); wv = sum(vs) / len(vs)
    return dict(excess=sv - wv, within=wv, df=len(ms) - 1)


def pooled_sigma_run(root, rj, max_rounds=20, exclude=()):
    """Pooled run-level sd over the most recent `max_rounds` rounds with a finished baseline (all p_i blocks present):
    excess = df-weighted mean of round_excess_var; sigma = sqrt(max(0, excess)); ub95 = one-sided 95 % upper bound
    (chi-square, Wilson-Hilferty). Rounds are read from <root>/r*/ (BASELINE.private.json selection + kept, BLOCK-1..). -> dict."""
    import math
    rows = []
    for p in sorted(Path(root).glob('r2*'), key=lambda q: q.name, reverse=True):
        if p.name in exclude: continue
        b = rj(p / 'BASELINE.private.json'); r = rj(p / 'ROUND.private.json') or {}
        if not (isinstance(b, dict) and b.get('per_task') and b.get('kept')): continue
        nb = int(r.get('blocks') or 0); blocks = {}
        for bi in range(1, nb):
            blk = rj(p / ('BLOCK-%d.private.json' % bi))
            if isinstance(blk, dict) and blk.get('per_task'): blocks[bi] = blk['per_task']
        if nb < 3 or len(blocks) != nb - 1: continue
        ev = round_excess_var(b.get('selection') or {}, blocks, b['kept'])
        if ev is None: continue
        rows.append(dict(ev, round=p.name))
        if len(rows) >= int(max_rounds): break
    df = sum(x['df'] for x in rows)
    if not df: return dict(rounds=0, df=0, round_ids=[], excess_var=None, within_var=None, sigma_run_est=None, ub95=None)
    ex = sum(x['df'] * x['excess'] for x in rows) / df; wv = sum(x['df'] * x['within'] for x in rows) / df
    z = 1.645; chi_lo = df * (1 - 2 / (9 * df) - z * math.sqrt(2 / (9 * df))) ** 3
    ub = math.sqrt(max(0.0, (ex + wv) * df / chi_lo - wv)) if chi_lo > 0 else None
    return dict(rounds=len(rows), df=df, round_ids=[x['round'] for x in rows], excess_var=ex, within_var=wv,
                sigma_run_est=math.sqrt(max(0.0, ex)), ub95=ub, positive_rounds=sum(x['excess'] > 0 for x in rows))


def sigma_policy(root, rj, c: dict) -> dict:
    """Frozen sigma_run policy of a round being opened (stored in ROUND.private.json). value None -> old per-round rule."""
    floor = float(c.get('sigma_run') or 0.0)
    try:
        pz = pooled_sigma_run(root, rj, int(c.get('sigma_pool_rounds') or 20))
    except Exception as ex:
        pz = dict(rounds=0, error=str(ex)[:200])
    ok = pz.get('sigma_run_est') is not None and int(pz.get('rounds') or 0) >= int(c.get('sigma_pool_min_rounds') or 5)
    return dict(pz, policy=SIGMA_POOLED, floor=floor, value=(max(floor, float(pz['sigma_run_est'])) if ok else None),
                fallback=(None if ok else 'per-round max(floor, run_sigma): too few pooled rounds'), computed_at=time.time(),
                rule='sigma_run = max(floor, sqrt(max(0, df-weighted mean over the last %d finished rounds of (between-block variance of '
                     'the 3 head p_i block weighted means - mean within-block sampling variance)))), frozen when the round opens'
                     % int(c.get('sigma_pool_rounds') or 20))


def sigma_policy_name(r: dict) -> str:
    p = r.get('sigma_run_policy') if isinstance(r, dict) else None
    return p['policy'] if isinstance(p, dict) and p.get('value') is not None else 'per-round'


def turn_sigma_run(r: dict, c: dict, b: dict) -> float:
    """sigma_run of a turn: the round's frozen pooled value, else the old per-round rule max(floor, run_sigma estimate)."""
    p = r.get('sigma_run_policy') if isinstance(r, dict) else None
    if isinstance(p, dict) and p.get('value') is not None: return float(p['value'])
    return max(float(c['sigma_run']), float((b or {}).get('sigma_run_est') or 0.0))


def public_sigma(r: dict):
    p = r.get('sigma_run_policy') if isinstance(r, dict) else None
    if not isinstance(p, dict): return dict(policy='per-round', rule='max(floor, this round\'s 3-block estimate heldout.run_sigma)')
    return {k: p.get(k) for k in ('policy', 'value', 'floor', 'rounds', 'df', 'sigma_run_est', 'ub95', 'round_ids', 'fallback', 'rule')}


def settings_hash(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def round_id(seed: str, t: float) -> str:
    return 'r' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(t)) + '-' + hashlib.sha256(('%s:heldout-round-id' % seed).encode()).hexdigest()[:8]


def merge_cells(cells: list) -> dict:
    """Sum probe cells {n,c,w,u,b,x,tok,L} of the same task over baseline blocks."""
    out = dict(n=0, c=0, w=0, u=0, b=0, x=0, tok=0, L=[])
    for cl in cells:
        for k in ('n', 'c', 'w', 'u', 'b', 'x', 'tok'): out[k] += int(cl.get(k) or 0)
        out['L'] += list(cl.get('L') or [])
    return out


def merged_baseline(blocks: dict, indices) -> dict:
    """blocks: {b: {task: cell}} -> {task: merged cell} for tasks graded in EVERY block (a task missing in one block is dropped)."""
    bl = [blocks[k] for k in sorted(blocks, key=int)]
    out = {}
    for t in indices:
        cs = [b.get(t) if t in b else b.get(str(t)) for b in bl]
        if all(isinstance(c, dict) and c.get('n') for c in cs): out[int(t)] = merge_cells(cs)
    return out


def redact_decision(dec: dict) -> tuple[dict, dict]:
    """-> (public-safe DECISION, private extras). Open round: keep the verdict, z, cap rate and delta; move everything that
    reveals |S ∩ batch| or per-task data (n, h_pass/c_pass on T, gains/losses, batch overlap, pre/post on the batch) to private."""
    ev = dict(dec.get('evaluated') or {})
    priv = dict(evaluated=ev, batch_overlap=dec.get('batch_overlap'), batch_pre_post=dec.get('batch_pre_post'),
                h_cover=dec.get('h_cover'), sample_size=dec.get('sample_size'))
    pub = dict(dec)
    pub['evaluated'] = dict(z=ev.get('z'), c_cap_rate=ev.get('c_cap_rate'), delta=ev.get('delta'),
                            redacted='held-out round open: n, pass rates on the paired tasks, gains/losses and batch overlap are '
                                     'published with the round reveal (they would leak which tasks are in the secret set)')
    for k in ('batch_overlap', 'batch_pre_post', 'h_cover'): pub[k] = None
    return pub, priv


class Rounds:
    """Private round state under <gate state>/heldout (0700). CURRENT.private.json = the open round (or absent)."""
    def __init__(self, root: Path, wj, rj, log, cr):
        self.root = Path(root); self.wj, self.rj, self.log, self.cr = wj, rj, log, cr

    def d(self, rid): return self.root / rid
    def current(self):
        c = self.rj(self.root / 'CURRENT.private.json')
        return c if isinstance(c, dict) and not (self.d(c['id']) / 'CLOSED.json').exists() else None
    def get(self, rid): return self.rj(self.d(rid) / 'ROUND.private.json')

    def open(self, G, f, c, head, epoch, reason, prev=None):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(self.root, 0o700)
        seed = os.urandom(32).hex(); now = time.time()
        idx = G.draw_sample(seed, int(f['pool']), int(c['n']))
        rid = round_id(seed, now)
        r = dict(schema=SCHEMA, id=rid, seed=seed, indices=idx, n=len(idx), head=head, pool=int(f['pool']), env=f['env'],
                 commitment=G.sample_commitment(seed, idx), gate_version=G.VERSION, opened_at=now, opened_at_cr=self.cr(now),
                 opened_epoch=epoch, reason=reason, previous=(prev or {}).get('id'),
                 blocks=int(c['blocks']), k_block=int(c['k_block']), k_cand=int(c['k_cand']), bar_z=float(c['bar_z']),
                 z_inclusive=bool(c['z_inclusive']),
                 baseline_plan_seeds=[baseline_plan_seed(seed, b) for b in range(int(c['blocks']))])
        if c.get('sigma_run_policy') == 'pooled': r['sigma_run_policy'] = sigma_policy(self.root, self.rj, c)   # variance levers
        self.d(rid).mkdir(parents=True, exist_ok=True, mode=0o700)
        self.wj(self.d(rid) / 'ROUND.private.json', r, 0o600)
        self.wj(self.d(rid) / 'COMMIT.json', self.public_commit(r), 0o644)
        self.wj(self.root / 'CURRENT.private.json', dict(id=rid, head=head), 0o600)
        self.log(event='heldout_round_open', id=rid, head=head[:12], n=len(idx), epoch=epoch, reason=reason)
        return r

    @staticmethod
    def public_commit(r):
        return dict(schema=SCHEMA, round_id=r['id'], head=r['head'], set_size=r['n'], pool=r['pool'], env=r['env'],
                    commitment=r['commitment'], opened_at=r['opened_at'], opened_at_cr=r['opened_at_cr'], opened_epoch=r['opened_epoch'],
                    baseline=dict(blocks=r['blocks'], k_block=r['k_block'], selection='block 0: k_block on all of S; keep 0 < c < k_block',
                                  rollouts_per_task_p=(r['blocks'] - 1) * r['k_block'], candidate_seeds='block 1 request seeds (CRN)'),
                    k_cand=r['k_cand'], bar_z=r['bar_z'], z_inclusive=r['z_inclusive'], sigma_run=public_sigma(r),
                    rule='commitment = sha256(gate_version + "|" + seed_hex + "|" + comma-joined sorted task ids) '
                         '(gate_core.sample_commitment); the seed is os.urandom(32). Seed, task ids, the head baseline and every '
                         'candidate\'s per-task grades of the round are revealed when the round closes (GREEN, head change or '
                         'max_round_turns decided turns). Score: inverse-variance-weighted per-task gain vs the head p_i on the kept tasks; '
                         'GREEN iff gain > 0 and z > bar_z and estimated cap-hit < 0.20; z = gain / sqrt(se_task^2 + sigma_run^2 (1 + 1/3)), '
                         'sigma_run per the round\'s sigma_run field (pooled over recent rounds, frozen at open).')

    def close(self, rid, reason, epoch=None):
        p = self.d(rid) / 'CLOSED.json'
        if p.exists(): return False
        self.wj(p, dict(round_id=rid, closed_at=time.time(), closed_at_cr=self.cr(), reason=reason, closing_epoch=epoch), 0o644)
        cur = self.rj(self.root / 'CURRENT.private.json') or {}
        if cur.get('id') == rid:
            try: (self.root / 'CURRENT.private.json').unlink()
            except FileNotFoundError: pass
        self.log(event='heldout_round_closed', id=rid, reason=reason, epoch=epoch)
        return True

    def closed(self, rid): return (self.d(rid) / 'CLOSED.json').exists()

    def baseline(self, rid):
        b = self.rj(self.d(rid) / 'BASELINE.private.json')
        return b if isinstance(b, dict) and b.get('per_task') else None

    def turn_add(self, rid, rec):
        with open(self.d(rid) / 'TURNS.private.jsonl', 'a') as fh: fh.write(json.dumps(rec, sort_keys=True) + '\n')
        os.chmod(self.d(rid) / 'TURNS.private.jsonl', 0o600)

    def turns(self, rid):
        out = []
        try:
            for l in open(self.d(rid) / 'TURNS.private.jsonl'):
                try: out.append(json.loads(l))
                except Exception: pass
        except FileNotFoundError: pass
        return out

    def all_ids(self):
        return sorted(p.name for p in self.root.glob('r*') if (p / 'ROUND.private.json').exists()) if self.root.exists() else []
