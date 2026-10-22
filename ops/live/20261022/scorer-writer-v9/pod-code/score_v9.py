#!/usr/bin/env python3
"""SN120 v9 measured-impact scorer (SHADOW ONLY; nothing reads its output into weights):  score_v9.py --epoch E
Input : data/e<E>/docs.jsonl.gz (sealed-epoch miner training documents, fetched READ-ONLY by the Arbos v9 driver),
        data/e<E>/seed.json (public post-commit seed, rule sn120-v9-seed-rule-v1, derived on Arbos from the finalized chain).
Steps : base checkpoint -> reference drawn from the full offered pool by the public seed (gen_ref_v9.py, free GPUs only)
        -> 2-6/8 band filter + subject chunks from the seed -> chunk gradients + per-rollout dots (grad_docs_v9.py)
        -> v9agg: leave-task-out group scores, fixed 3-group units, cross-fitted out-of-sample max-of-3 null, per-unit p,
           window (W epochs) Stouffer + binomial gate at alpha/N_UID, V = max(0, sum z) for gated miners
        -> false-positive sanity checks (label-shuffle placebo, UID-shuffle) -> signed JSON (scorer key 749593a9...).
Output: out/score-e<E>.json (aggregates per miner/UID only; seed block + hash recorded for third-party reproduction)."""
import argparse, collections, glob, gzip, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from v9common import W, TS, VERSION, SEED_RULE, HS, load_seed, build_chunks, atomic_write, sign
import v9agg
PY = '/root/sampler-venv/bin/python'; NPLACEBO = 20

def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def free_gpus(maxn):
    if subprocess.run(['pgrep', '-f', r'claims/(run_real|run_gen|run_grad|gen_claims|grad_claims|grad_real|fd_|adv_)'], capture_output=True).returncode == 0: return []
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    busy = subprocess.run(['pgrep', '-f', r'(gen_ref|grad_docs)(_v9)?\.py'], capture_output=True, text=True).stdout.split()
    return [int(i) for i, m in (l.split(',') for l in q.strip().splitlines()) if int(m) < 2000][:maxn] if not busy else []
def wait_gpus(maxn, wait_s):
    t = time.time()
    while True:
        g = free_gpus(maxn)
        if g: return g
        if time.time() - t > wait_s: raise RuntimeError('no free GPU within %ds' % wait_s)
        time.sleep(30)
def base_for(E):
    c = [x for x in json.load(open(TS / 'data/index.json'))['checkpoints'] if str(x.get('opening_epoch')) == str(E)]
    if len(c) != 1: raise RuntimeError('epoch %s base checkpoint not unique in index (%d)' % (E, len(c)))
    return c[0]['checkpoint_id']
def run_shards(script, E, gpus, extra=()):
    procs = []
    for s, g in enumerate(gpus):
        lf = open(W / 'logs' / f'{Path(script).stem}-e{E}-{s}.log', 'w')
        procs.append(subprocess.Popen([PY, str(HERE / script), '--epoch', str(E), '--shard', str(s), '--nshards', str(len(gpus)), *extra],
                                      cwd=str(HERE), env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)), stdout=lf, stderr=subprocess.STDOUT))
    rc = [p.wait() for p in procs]
    if any(rc): raise RuntimeError('%s failed rc=%s (see logs)' % (script, rc))

def aggregate(E, R, chunks, dots_of, seed, window, alpha=0.01):
    """CPU part (also used by the dry run). Returns payload pieces."""
    fold = lambda m: HS(seed, 'fold', m) % 2
    groups = [dict(miner=r['miner_full'], uid=r.get('uid'), slot=int(r['slot']), task=int(r['index']),
                   rewards=[x['reward'] for x in r['rollouts']], dots=dots_of(r)) for r in R]
    rng = np.random.default_rng(HS(seed, 'null'))
    miners, null_info, nvalid = v9agg.score_epoch(groups, chunks, fold, rng)
    prev = []
    for e in range(E - window + 1, E):
        f = W / 'out' / f'score-e{e}.json'
        if f.exists(): prev.append(json.load(open(f))['payload']['epoch_units'])
    gate, summ = v9agg.gate_window(prev + [miners], alpha)
    summ['window_epochs_available'] = len(prev) + 1; summ['window_epochs'] = window
    # ---- false-positive sanity checks (epoch-only window so they are comparable across epochs)
    prng = np.random.default_rng(HS(seed, 'placebo')); pl = []
    for _ in range(NPLACEBO):
        sh = [dict(g, rewards=list(prng.permutation(g['rewards'])) if all(v is not None for v in g['rewards']) else g['rewards']) for g in groups]
        pm, _, _ = v9agg.score_epoch(sh, chunks, fold, prng); _, ps = v9agg.gate_window([pm], alpha)
        pl.append((ps['gated'], ps['flag_rate_stouffer_uncorrected'], ps['flag_rate_binom_uncorrected']))
    _, s1 = v9agg.gate_window([miners], alpha)
    units = [(m, u) for m, r in miners.items() for u in r['units']]; us = []
    for _ in range(NPLACEBO):
        perm = prng.permutation(len(units)); fake = collections.defaultdict(lambda: dict(uid=None, units=[]))
        owners = [m for m, _ in units]
        for k, j in enumerate(perm): fake[owners[k]]['units'].append(units[j][1])
        _, fs = v9agg.gate_window([fake], alpha); us.append(fs['gated'])
    fp = dict(label_shuffle=dict(reps=NPLACEBO, gated_mean=float(np.mean([a for a, _, _ in pl])), gated_max=int(max(a for a, _, _ in pl)),
                                 flag_rate_stouffer_uncorrected_mean=float(np.mean([b for _, b, _ in pl])),
                                 flag_rate_binom_uncorrected_mean=float(np.mean([c for _, _, c in pl]))),
              uid_shuffle=dict(reps=NPLACEBO, gated_mean=float(np.mean(us)), gated_max=int(max(us))),
              epoch_only=dict(gated=s1['gated'], flag_rate_stouffer_uncorrected=s1['flag_rate_stouffer_uncorrected'],
                              flag_rate_binom_uncorrected=s1['flag_rate_binom_uncorrected'], expected_uncorrected=alpha),
              unit_hit_rate=float(np.mean([u['hit'] for _, u in units])) if units else None,
              expected_unit_hit_rate=float(np.mean([u['q'] for _, u in units])) if units else None)
    return dict(epoch_units=miners, window_gate=gate, window_summary=summ, null=null_info, valid_groups=nvalid, false_positive_checks=fp)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', type=int, required=True); ap.add_argument('--max-gpus', type=int, default=4)
    ap.add_argument('--wait-gpu-seconds', type=int, default=4 * 3600); ap.add_argument('--window', type=int, default=4)
    a = ap.parse_args(); E = a.epoch; t0 = time.time(); Tm = {}
    for d in ('logs', 'out'): (W / d).mkdir(exist_ok=True)
    ED = W / 'data' / f'e{E}'; docs_path = ED / 'docs.jsonl.gz'; seedrec = load_seed(E); SEED = seedrec['seed']
    base = base_for(E); log('epoch', E, 'base', base[:16], 'seed block', seedrec['seed_block'])
    R_all = [json.loads(l) for l in gzip.open(docs_path, 'rt')]
    R = [r for r in R_all if r['built_on'] == base[:16] and r.get('miner_full')]
    log('docs', len(R_all), 'on base', len(R))
    rd = ED / 'ref'; rd.mkdir(parents=True, exist_ok=True); t = time.time()
    if not (rd / 'DONE').exists():
        gpus = wait_gpus(a.max_gpus, a.wait_gpu_seconds)
        for f in rd.glob('gen-*'): f.unlink()
        log('ref gen on GPUs', gpus); run_shards('gen_ref_v9.py', E, gpus, ['--base', base]); (rd / 'DONE').write_text('generated on %d gpus\n' % len(gpus))
    Tm['ref'] = time.time() - t
    sys.path.insert(0, str(TS)); from grad_docs_v9 import load_ref
    rb, meta = load_ref(E); assert rb == base, 'reference base mismatch'
    chunks, info, nsub = build_chunks(meta, SEED); K = len(chunks)
    have = {}
    def collect():
        for f in sorted(glob.glob(str(ED / 'dots-*.json'))):
            d = json.load(open(f))
            if d['base'] != base or d['chunks'] != chunks or d.get('seed') != SEED: continue
            for g in d['docs']: have[(g['miner'][:16], g['slot'])] = (g['lens'], g['dots'])
    def dots_of(r):
        h = have.get((r['miner'][:16], r['slot']))
        return h[1] if h and h[0] == [len(x['out']) for x in r['rollouts']] else None
    t = time.time(); rounds = 0
    while True:
        collect(); todo = [(r['miner'], r['slot']) for r in R if dots_of(r) is None]
        log('dots available', len(R) - len(todo), 'todo', len(todo))
        if not todo or rounds >= 2: break
        gpus = wait_gpus(a.max_gpus, a.wait_gpu_seconds); rounds += 1; atomic_write(ED / 'todo.json', json.dumps(todo))
        stamp = int(time.time())
        for f in ED.glob('dots-[0-9]*.json'): f.rename(ED / ('dots-r%d-%s' % (stamp, f.name.split('-')[-1])))
        log('dots on GPUs', gpus); run_shards('grad_docs_v9.py', E, gpus)
    Tm['dots'] = time.time() - t
    missing = [r for r in R if dots_of(r) is None]
    if missing: raise RuntimeError('%d documents still without dots' % len(missing))
    t = time.time(); agg = aggregate(E, R, chunks, dots_of, SEED, a.window); Tm['aggregate'] = time.time() - t
    payload = dict(version=VERSION, shadow_only=True, epoch=E, base_checkpoint=base, created_at=time.time(),
                   seed=seedrec, docs_sha256=sha_file(docs_path), docs_total=len(R_all), docs_on_base=len(R), miners=len(agg['epoch_units']),
                   reference=dict(rule=SEED_RULE, drawn_tasks=len({x['index'] for x in meta}), band_tasks=len(info), band='2-6 of 8 correct',
                                  chunks=K, chunk_tasks=4, informative_by_subject=nsub, subspace='layers 28-31 mlp gate/up/down', leave_task_out=True),
                   rule=dict(unit_groups=3, unit='slot // 3, partial units vs the same 3-group null', null='2-fold cross-fit by miner, max-of-3 empirical',
                             hit='p < q, q = P_null(M > med_M + 2 sd_M)', gate='Stouffer AND binomial at alpha/N_UID', alpha=0.01, window_epochs=a.window,
                             payment='V = max(0, sum_g z_g) for gated miners'),
                   timings={k: round(v, 1) for k, v in dict(Tm, total=time.time() - t0).items()}, **agg)
    doc = sign(payload); out = W / 'out' / f'score-e{E}.json'; atomic_write(out, json.dumps(doc, sort_keys=True))
    s = agg['window_summary']; log('WROTE', out, 'miners', len(agg['epoch_units']), 'gated', s['gated'], 'of', s['scored_uids'],
                                    'fp', json.dumps(agg['false_positive_checks']))

if __name__ == '__main__':
    main()
