#!/usr/bin/env python3
"""CPU: build planted 'synthetic miner' batches for epoch E (train step; base = reference base).
Every planted rollout is either an existing scored rollout (real miner doc or regenerated group; dots exist)
or a NEW sequence whose dots plant_dots.py computes with the identical pipeline (same base, subspace, V).
Output: claims/data/planted/e<E>/spec.json (+ new.npz / new.json for sequences that need dots)."""
import gzip, glob, json, random, sys, hashlib, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import grader, grad_real
from grad_claims import build_chunks
E = sys.argv[1]; NB = 10; rng = random.Random(int(E) * 7919)
OUT = HERE / 'data' / 'planted' / f'e{E}'; OUT.mkdir(parents=True, exist_ok=True)
T = json.load(open(ROOT / 'data/original-math7496.tasks.json')); subj = lambda i: T[i]['data']['subject']
EOS = 248046
R = [json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt')]
RD = {(g['miner'], g['slot']): g for f in glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')) for g in json.load(open(f))['groups']}
RG = [g for f in glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-[0-9]*.json')) for g in json.load(open(f))['groups'] if g['valid']]
base, meta = grad_real.load_ref(E); chunks, info, _ = build_chunks(meta, E)
reft = {t for c in chunks for t in c['tasks']}
nb = collections.Counter(r['uid'] for r in R); G = int(np.median(list(nb.values())))
new_tok, new_meta = [], []
def new(index, tokens, why):
    new_tok.append(np.asarray(tokens, dtype=np.int32)); new_meta.append(dict(index=index, len=len(tokens), why=why)); return len(new_tok) - 1
rew = lambda c, L: grader.overlong_reward(c, L)
def real_group(r, cls=None):
    d = RD[(r['miner'], r['slot'])]; cls = cls or d['claimed_cls']
    return dict(index=r['index'], subject=subj(r['index']), rollouts=[dict(src='real', key=[r['miner'], r['slot'], j], reward=rew(1.0 if c == 'positive' else 0.0, L), len=L)
                                                                    for j, (c, L) in enumerate(zip(cls, d['lens']))])
pool = R[:]; rng.shuffle(pool); used = 0
def take(n):
    global used
    x = pool[used:used + n]; used += n; return x
batches = []
def add(t, groups): batches.append(dict(type=t, id=f'{t}-{sum(b["type"] == t for b in batches)}', groups=groups))
# 1 honest control: resampled real groups (random miners)
for b in range(NB): add('1_honest', [real_group(r) for r in take(G)])
# 2 random rewards
for b in range(NB):
    gs = []
    for r in take(G):
        while True:
            cls = ['positive' if rng.random() < 0.5 else 'negative' for _ in r['rollouts']]
            if 0 < cls.count('positive') < len(cls): break
        gs.append(real_group(r, cls))
    add('2_random_rewards', gs)
# 3 faked labels: 100% and 30% inverted
inv = lambda c: 'negative' if c == 'positive' else 'positive'
for b in range(NB): add('3a_faked_100pct', [real_group(r, [inv(c) for c in RD[(r['miner'], r['slot'])]['claimed_cls']]) for r in take(G)])
for b in range(NB): add('3b_faked_30pct', [real_group(r, [inv(c) if rng.random() < 0.3 else c for c in RD[(r['miner'], r['slot'])]['claimed_cls']]) for r in take(G)])
# 4 garbage completions claimed correct (truncated / shuffled tokens), real wrong completions kept at 0
for b in range(NB):
    gs = []
    for r in take(G):
        g = real_group(r)
        for j, (x, ro) in enumerate(zip(r['rollouts'], g['rollouts'])):
            if x['cls'] == 'positive':
                o = list(x['out'][:-1])
                if j % 2 == 0: garb = o[:max(8, len(o) // 5)] + [EOS]
                else: o2 = o[:]; rng.shuffle(o2); garb = o2 + [EOS]
                g['rollouts'][j] = dict(src='new', key=new(r['index'], garb, 'garbage'), reward=rew(1.0, len(garb)), len=len(garb))
        gs.append(g)
    add('4_garbage_claimed_correct', gs)
# reference-draw tasks (base samples, our labels); never informative chunks (nc>=7 or nc<=1 are excluded from chunks)
byref = collections.defaultdict(list)
for x in meta:
    if x['kind'] == 'ref': byref[x['index']].append(x)
def ref_group(i, xs, why):
    return dict(index=i, subject=subj(i), rollouts=[dict(src='new', key=new(i, x['tokens'], why), reward=rew(x['correct'] or 0.0, x['len']), len=x['len']) for x in xs])
nc = {i: sum(x['outcome'] == 'positive' for x in xs) for i, xs in byref.items()}
easy = [i for i in byref if nc[i] >= 7 and i not in reft]; rng.shuffle(easy)
tcor = collections.defaultdict(list); teach = dict(tasks=0, samples=0, tasks_with_correct=0)
if (OUT / 'teacher.json').exists():
    teach = json.load(open(OUT / 'teacher.json')); tz = np.load(OUT / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']
    for k, c in enumerate(teach['correct']): tcor[c['index']].append(tt[to[k]:to[k + 1]])
hard = [i for i in byref if nc[i] <= 1 and i not in reft and i not in tcor]; rng.shuffle(hard)
hard_all = [i for i in byref if nc[i] <= 1 and i not in reft]; rng.shuffle(hard_all)
hard = hard + [i for i in hard_all if i not in hard]   # prefer tasks not used by the teacher type
for b in range(min(NB, len(easy) // G)): add('5_easy_only', [ref_group(i, byref[i], 'easy') for i in easy[b * G:(b + 1) * G]])
for b in range(min(NB, len(hard) // G)): add('6_hard_only', [ref_group(i, byref[i], 'hard') for i in hard[b * G:(b + 1) * G]])
# 7 really good: base-model groups on miner-chosen (non-reference) tasks with 3-5 of 8 correct
good = [g for g in RG if 3 <= sum(r >= 0.5 for r in g['rewards']) <= 5 and g['index'] not in reft]; rng.shuffle(good)
seen = set(); good2 = []
for g in good:
    if g['index'] not in seen: seen.add(g['index']); good2.append(g)
for b in range(min(NB, len(good2) // G)):
    add('7_good_balanced', [dict(index=g['index'], subject=g['subject'], rollouts=[dict(src='regen', key=[g['uid'], g['index'], j], reward=g['rewards'][j], len=g['lens'][j]) for j in range(len(g['rewards']))])
                            for g in good2[b * G:(b + 1) * G]])
# 8 beyond-policy: correct solutions found by best-of-(8+40) base sampling on tasks the base solves <=1/8, + wrong base samples
tg = []
for i in sorted(tcor, key=lambda i: hashlib.sha256(f'{E}{i}'.encode()).hexdigest()):
    cors = tcor[i][:2]; wr = [x for x in byref[i] if x['outcome'] != 'positive'][:8 - len(cors)]
    ro = [dict(src='new', key=new(i, c, 'teacher'), reward=rew(1.0, len(c)), len=len(c)) for c in cors]
    ro += [dict(src='new', key=new(i, x['tokens'], 'teacher_wrong'), reward=rew(0.0, x['len']), len=x['len']) for x in wr]
    tg.append(dict(index=i, subject=subj(i), rollouts=ro))
for b in range(min(NB, len(tg) // G)): add('8_beyond_policy_bo48', tg[b * G:(b + 1) * G])
# 9 stale: real honest groups built 2 steps earlier (step E-2 docs, built on that step's base), scored at this base
S = [json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{int(E) - 2}.jsonl.gz', 'rt')]; rng.shuffle(S)
for b in range(NB):
    gs = []
    for r in S[b * G:(b + 1) * G]:
        gs.append(dict(index=r['index'], subject=subj(r['index']), rollouts=[dict(src='new', key=new(r['index'], x['out'], 'stale'), reward=rew(1.0 if x['cls'] == 'positive' else 0.0, len(x['out'])), len=len(x['out'])) for x in r['rollouts']]))
    add('9_stale_2steps', gs)
stale_from = collections.Counter(r['built_on'] for r in S).most_common(1)[0][0]
offs = np.cumsum([0] + [len(t) for t in new_tok])
np.savez(OUT / 'new.npz', tokens=np.concatenate(new_tok), offsets=offs)
json.dump(dict(epoch=E, base=base, meta=new_meta), open(OUT / 'new.json', 'w'))
json.dump(dict(epoch=E, base=base, groups_per_batch=G, stale_built_on=stale_from, teacher=dict(tasks=teach['tasks'], samples=teach['samples'], tasks_with_correct=teach['tasks_with_correct']),
               pools=dict(easy=len(easy), hard=len(hard_all), good=len(good2), teacher_groups=len(tg)), batches=batches), open(OUT / 'spec.json', 'w'))
print(E, 'G', G, collections.Counter(b['type'] for b in batches), 'new seqs', len(new_tok), 'tokens', int(offs[-1]), 'stale_from', stale_from, flush=True)
