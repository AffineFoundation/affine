#!/usr/bin/env python3
"""SN120 public gate pool v1 (Jacob 2026-11-01): MATH (original-math7496) + DAPO pool, minus every task that overlaps the private
perf-eval holdout. Overlap = sha256 of the raw problem text, OR sha256 of the normalised problem text (NFKC, lowercase, whitespace
collapsed), OR token-3-gram Jaccard >= 0.9 (the holdout's own dedupe rule). Overlapping tasks are dropped from the gate pool (the holdout
stays private and outside the gate pool).
usage: build_public_pool.py MATH_TASKS.json DAPO_POOL.jsonl HOLDOUT.private.json OUTDIR
Outputs (OUTDIR): gate-pool-v1.jsonl (public: id, env, problem, answer[, src_index, uid]), gate-pool-v1.manifest.json (public:
sha256 of every file, counts, rules, number of dropped tasks per env), overlap.private.json (0600: which ids overlapped which holdout ids)."""
import collections, hashlib, json, os, re, sys, unicodedata

def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def norm(s): return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', s).lower()).strip()
def toks(s): return re.findall(r'[a-z]+|\d+|[^\sa-z\d]', norm(s))
def sh3(s):
    t = toks(s); return {tuple(t[i:i + 3]) for i in range(max(0, len(t) - 2))}
def jacc(a, b): return len(a & b) / len(a | b) if a and b else 0.0

def main(math_p, dapo_p, hold_p, out):
    math = json.load(open(math_p)); dapo = [json.loads(l) for l in open(dapo_p)]
    hold = json.load(open(hold_p))['tasks']
    pool = [dict(id='math:%d' % i, env='affine_math', problem=t['data']['problem'], answer=str(t['data']['answer'])) for i, t in enumerate(math)]
    pool += [dict(id=r['id'], env='dapo', problem=r['problem'], answer=str(r['answer']), src_index=r.get('src_index'), uid=r.get('uid')) for r in dapo]
    H_raw = {sha(h['problem']): h['id'] for h in hold}; H_norm = {sha(norm(h['problem'])): h['id'] for h in hold}
    H_sh = [(h['id'], sh3(h['problem'])) for h in hold]; inv = collections.defaultdict(set)
    for k, (_, s) in enumerate(H_sh):
        for g in s: inv[g].add(k)
    hits = []
    for t in pool:
        why = None
        if sha(t['problem']) in H_raw: why = ('raw_sha256', H_raw[sha(t['problem'])], 1.0)
        elif sha(norm(t['problem'])) in H_norm: why = ('norm_sha256', H_norm[sha(norm(t['problem']))], 1.0)
        else:
            s = sh3(t['problem']); cnt = collections.Counter()
            for g in s:
                for k in inv.get(g, ()): cnt[k] += 1
            best = None
            for k, c in cnt.items():
                if c < 0.9 * len(s): continue
                j = jacc(s, H_sh[k][1])
                if j >= 0.9 and (best is None or j > best[2]): best = ('jaccard3>=0.9', H_sh[k][0], round(j, 4))
            why = best
        if why: hits.append(dict(id=t['id'], env=t['env'], rule=why[0], holdout_id=why[1], sim=why[2]))
    drop = {h['id'] for h in hits}
    kept = [t for t in pool if t['id'] not in drop]
    os.makedirs(out, exist_ok=True)
    pj = os.path.join(out, 'gate-pool-v1.jsonl')
    with open(pj, 'w') as f:
        for t in kept: f.write(json.dumps(t, ensure_ascii=False, sort_keys=True) + '\n')
    cnt = collections.Counter(t['env'] for t in kept); dc = collections.Counter(h['env'] for h in hits); rc = collections.Counter(h['rule'] for h in hits)
    man = dict(version='sn120-gate-pool-v1', files={'gate-pool-v1.jsonl': fsha(pj)},
               sources={'affine_math': dict(file='miner/data/original-math7496.tasks.json', sha256=fsha(math_p), tasks=len(math), id_rule='math:<list index>'),
                        'dapo': dict(file='(built by priors/dapo/build_dapo.py from BytedTsinghua-SIA/DAPO-Math-17k rev 65877096c24ffa7abc4e4fa5edb95cf3413a5674)',
                                     sha256_of_built_pool=fsha(dapo_p), tasks=len(dapo), id_rule='dapo:<position in the built pool>')},
               tasks=len(kept), tasks_per_env=dict(cnt), ids_sha256=sha('\n'.join(t['id'] for t in kept)),
               holdout_exclusion=dict(dropped=len(hits), dropped_per_env=dict(dc), dropped_per_rule=dict(rc),
                                      dropped_ids=sorted(drop, key=lambda x: (x.split(':')[0], int(x.split(':')[1]))),
                                      rule='a pool task is dropped if its problem matches a task of the private perf-eval holdout: sha256 of the raw '
                                           'problem text, sha256 of the normalised text (NFKC, lowercase, whitespace collapsed), or token-3-gram Jaccard >= 0.9. '
                                           'The holdout itself stays private (size + sha256 in https://affine.io/performance.json).'),
               line_format='one JSON object per line, sorted keys: id, env, problem, answer (+ src_index, uid for dapo); file order = math then dapo by id',
               prompt='[system: MANDATE, user: problem] for both envs (same as affine_math)', grader='the live affine_math grader (probe_worker.grade) for both envs')
    json.dump(man, open(os.path.join(out, 'gate-pool-v1.manifest.json'), 'w'), indent=1, sort_keys=True)
    pp = os.path.join(out, 'overlap.private.json')
    with open(os.open(pp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as f: json.dump(dict(hits=hits), f, indent=1)
    print(json.dumps(dict(tasks=len(kept), per_env=dict(cnt), dropped=len(hits), dropped_per_env=dict(dc), per_rule=dict(rc)), indent=1))

if __name__ == '__main__': main(*sys.argv[1:5])
