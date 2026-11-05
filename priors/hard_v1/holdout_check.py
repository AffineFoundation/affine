"""Drop candidates overlapping the private perf-eval holdout (same rule as build_public_pool.py: raw sha256, normalized sha256,
token-3-gram Jaccard >= 0.9). Writes candidates.jsonl (renumbered hard:0..) + holdout_overlap.private.json (counts only printed)."""
import collections, hashlib, json, re, sys, unicodedata, os
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/private"
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def norm(s): return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', s).lower()).strip()
def toks(s): return re.findall(r'[a-z]+|\d+|[^\sa-z\d]', norm(s))
def sh3(s):
    t = toks(s); return {tuple(t[i:i + 3]) for i in range(max(0, len(t) - 2))}
def jacc(a, b): return len(a & b) / len(a | b) if a and b else 0.0
hold = json.load(open('/home/const/subnet120-rewrite/state/root-audits/20261031-performance-eval/holdout.private.json'))['tasks']
H_raw = {sha(h['problem']): h['id'] for h in hold}; H_norm = {sha(norm(h['problem'])): h['id'] for h in hold}
H_sh = [(h['id'], sh3(h['problem'])) for h in hold]; inv = collections.defaultdict(set)
for k, (_, s) in enumerate(H_sh):
    for g in s: inv[g].add(k)
rows = [json.loads(l) for l in open(D + '/candidates.prehold.jsonl')]; hits = []; keep = []
for t in rows:
    why = None
    if sha(t['problem']) in H_raw: why = ('raw_sha256', H_raw[sha(t['problem'])], 1.0)
    elif sha(norm(t['problem'])) in H_norm: why = ('norm_sha256', H_norm[sha(norm(t['problem']))], 1.0)
    else:
        s = sh3(t['problem']); cnt = collections.Counter()
        for g in s:
            for k in inv.get(g, ()): cnt[k] += 1
        for k, c in cnt.items():
            if c < 0.9 * len(s): continue
            j = jacc(s, H_sh[k][1])
            if j >= 0.9: why = ('jaccard3>=0.9', H_sh[k][0], round(j, 4)); break
    if why: hits.append(dict(src=t['src'], src_id=t['src_id'], rule=why[0], holdout_id=why[1], sim=why[2]))
    else: keep.append(t)
with open(D + '/candidates.jsonl', 'w') as f:
    for i, t in enumerate(keep):
        t.update(id='hard:%d' % i, idx=i); f.write(json.dumps(t, sort_keys=True, ensure_ascii=False) + '\n')
fd = os.open(D + '/holdout_overlap.private.json', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, 'w') as f: json.dump(dict(hits=hits), f, indent=1)
print(json.dumps(dict(holdout_tasks=len(hold), checked=len(rows), dropped=len(hits), by_src=collections.Counter(h['src'] for h in hits),
                      by_rule=collections.Counter(h['rule'] for h in hits), kept=len(keep),
                      by_src_kept=collections.Counter(t['src'] for t in keep),
                      sha256=hashlib.sha256(open(D + '/candidates.jsonl', 'rb').read()).hexdigest())))
