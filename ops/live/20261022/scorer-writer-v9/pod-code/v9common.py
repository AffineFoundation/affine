"""SN120 v9 measured-impact scorer: shared helpers (pod side). Aggregates only; never prints miner data.
Reference randomness comes ONLY from the public post-commit seed (seed_block.py on Arbos, rule sn120-v9-seed-rule-v1)."""
import base64, hashlib, json, os, random, subprocess
from pathlib import Path
W = Path('/root/sn120-scorer-v9'); TS = Path('/root/trusted-sampler'); V7 = Path('/root/sn120-scorer')
KEY = V7 / 'key' / 'scorer.seed'          # same dedicated scorer ed25519 key as v7c (pubkey 749593a9...)
VERSION = 'sn120-measured-impact-score-v9-shadow'
SEED_RULE = 'sn120-v9-seed-rule-v1'
def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def HS(seed, *a):
    """32-bit sub-seed derived from the public post-commit seed (hex) and labels."""
    return int(hashlib.sha256(':'.join(map(str, (seed,) + a)).encode()).hexdigest()[:8], 16) & 0x7fffffff
def check_seed(s, E):
    exp = hashlib.sha256(f"sn120-v9-ref-seed-v1:{s['epoch_id']}:{s['seed_block_hash']}".encode()).hexdigest()
    if s['rule'] != SEED_RULE or int(s['epoch']) != int(E) or s['seed'] != exp or s['seed_block'] != s['anchor_block'] + s['margin_blocks'] \
       or s['anchor_ts_ms'] < s['deadline'] * 1000:
        raise ValueError('seed record does not satisfy %s' % SEED_RULE)
    return s
def load_seed(E): return check_seed(json.load(open(W / 'data' / f'e{E}' / 'seed.json')), E)
def draw_reference(seed, pool, T, per_subject=200):
    """Stratified draw from the full offered pool, made only from the public post-commit seed."""
    rng = random.Random(HS(seed, 'draw')); subj = {}
    for i in pool: subj.setdefault(T[i]['data']['subject'], []).append(i)
    return sorted(t for s in sorted(subj) for t in rng.sample(sorted(subj[s]), min(per_subject, len(subj[s]))))
def build_chunks(meta, seed, per=4, nch=6):
    """2-6/8 band filter computed after the draw; subject chunks in two disjoint halves (pool chunks dropped: unused)."""
    by = {}
    for x in meta:
        if x['kind'] == 'ref': by.setdefault(x['index'], []).append(x)
    info = {}
    for i, xs in by.items():
        nc = sum(x['outcome'] == 'positive' for x in xs); nw = sum(x['outcome'] in ('negative', 'neutral') for x in xs)
        if 2 <= nc <= 6 and nw >= 1: info[i] = xs
    subj = {}
    for i in sorted(info): subj.setdefault(info[i][0]['subject'], []).append(i)
    rng = random.Random(HS(seed, 'chunks')); chunks = []
    for s in sorted(subj):
        ts = subj[s][:]; rng.shuffle(ts); hs = [ts[0::2], ts[1::2]]
        for h in (0, 1):
            for c in range(min(nch, len(hs[h]) // per)):
                chunks.append(dict(kind='subject', subject=s, half=h, c=c, tasks=hs[h][c * per:(c + 1) * per]))
    return chunks, info, {s: len(v) for s, v in subj.items()}
def atomic_write(path, data, mode=0o600):
    path = Path(path); tmp = path.with_name(path.name + '.tmp%d' % os.getpid())
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, 'wb') as f: f.write(data if isinstance(data, bytes) else data.encode()); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
def sign(payload):
    r = subprocess.run(['/root/scorer-signvenv/bin/python', str(V7 / 'common.py'), '--sign'], input=canonical(payload), capture_output=True, check=True)
    return dict(payload=payload, **json.loads(r.stdout))
