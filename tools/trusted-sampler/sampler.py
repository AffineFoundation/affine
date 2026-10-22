#!/usr/bin/env python3
"""Trusted paired sampler for SN120 (Affine MiMo GRPO mainnet run).

Detached from miners/trainer/validator/weights. For a consecutive checkpoint pair
(before = input of step e, after = output of step e) it draws its OWN random K tasks
from the full math pool (local CSPRNG seed, no coordination), samples m completions per
task from each checkpoint with identical settings/seeds, grades them locally with the
network's outcome rule (grader.py), and appends one JSONL record to
results/<sampler_id>.jsonl.

Modes:
  --pair A:B          score one pair (ids or prefixes)
  --backfill N        score the newest N consecutive pairs (oldest first), skipping pairs
                      this instance already recorded
  --watch             loop: score every new pair as it appears in the index (+ backfill)
Engines:
  --engine reload     (baseline) new vLLM engine per checkpoint
  --engine swap       (optimized) one resident engine, weights hot-swapped in place
"""
import argparse, hashlib, json, math, os, platform, random, secrets, socket, sys, time, uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')  # needed for the in-place weight swap RPC
sys.path.insert(0, str(HERE))
import fetch, grader

VERSION = 'trusted-paired-sampler-v1'


def now():
    return time.time()


class Timer(dict):
    def __call__(self, name):
        t = self
        class _C:
            def __enter__(s): s.t0 = time.time()
            def __exit__(s, *a): t[name] = t.get(name, 0.0) + time.time() - s.t0
        return _C()


def load_index(path):
    return json.loads(Path(path).read_text())


def find(idx, prefix):
    m = [c for c in idx['checkpoints'] if c['checkpoint_id'].startswith(prefix)]
    if len(m) != 1:
        raise SystemExit(f'checkpoint prefix {prefix!r} matched {len(m)}')
    return m[0]


def consecutive_pairs(idx):
    by = {c['checkpoint_id']: c for c in idx['checkpoints']}
    out = []
    for c in idx['checkpoints']:
        if c.get('input_checkpoint') in by and c['input_checkpoint'] != c['checkpoint_id']:
            out.append((by[c['input_checkpoint']], c))
    return out


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, a):
        self.a = a; self.llm = None; self.loaded = None

    def _new(self, path):
        from vllm import LLM
        self.llm = LLM(model=str(path), tokenizer=str(path), dtype='bfloat16', seed=0,
                       tensor_parallel_size=self.a.tp, max_model_len=12288,
                       gpu_memory_utilization=self.a.gpu_mem, enable_prefix_caching=True,
                       max_num_seqs=self.a.max_num_seqs, max_num_batched_tokens=self.a.max_batched_tokens,
                       generation_config='vllm', limit_mm_per_prompt={'image': 0, 'video': 0},
                       enforce_eager=self.a.enforce_eager)
        self.tokenizer = self.llm.get_tokenizer()

    def load(self, cp, path):
        if self.loaded == cp['checkpoint_id']:
            return 'resident'
        if self.llm is None or self.a.engine == 'reload':
            self.close(); self._new(path); how = 'cold'
        else:
            # Hot swap: stream the new safetensors into the resident model's parameters.
            def _swap(worker, p=str(path)):
                import glob
                from safetensors.torch import safe_open
                def it():
                    for f in sorted(glob.glob(p + '/*.safetensors')):
                        with safe_open(f, 'pt', device='cpu') as s:
                            for k in s.keys():
                                yield k, s.get_tensor(k)
                import torch
                model = worker.model_runner.model
                probe = next(iter(model.parameters()))
                before = float(probe.float().sum())
                loaded = model.load_weights(it())
                torch.cuda.synchronize()
                return dict(n=len(loaded) if loaded is not None else -1, probe_changed=float(probe.float().sum()) != before)
            n = self.llm.collective_rpc(_swap)
            self.llm.reset_prefix_cache(); how = dict(mode='swap', **n[0])
        self.loaded = cp['checkpoint_id']
        return how

    def close(self):
        if self.llm is not None:
            import gc, torch
            try:
                self.llm.llm_engine.engine_core.shutdown()
            except Exception:
                pass
            del self.llm; self.llm = None; gc.collect(); torch.cuda.empty_cache()
        self.loaded = None

    def generate(self, prompts, seeds, rt):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        reqs, params = [], []
        for ids, ss in zip(prompts, seeds):
            mx = min(self.a.max_tokens, rt['max_context_tokens'] - len(ids))
            for s in ss:
                reqs.append(TokensPrompt(prompt_token_ids=ids))
                params.append(SamplingParams(n=1, temperature=rt['temperature'], top_p=rt['top_p'],
                                             top_k=rt['top_k'], max_tokens=mx, seed=s,
                                             stop_token_ids=list(rt['eos_token_ids']),
                                             skip_special_tokens=False, detokenize=False))
        outs = self.llm.generate(reqs, params, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]; toks = list(c.token_ids)
            if c.finish_reason == 'stop' and (not toks or toks[-1] not in rt['eos_token_ids']) \
                    and c.stop_reason in rt['eos_token_ids']:
                toks.append(c.stop_reason)
            res.append(toks)
        m = len(seeds[0])
        return [res[i * m:(i + 1) * m] for i in range(len(prompts))]


# ---------------------------------------------------------------- scoring
def score_outputs(tok, outs, golds, vocab_size, pool):
    special = set(getattr(tok, 'all_special_ids', ()))
    jobs, meta = [], []
    for t, (samples, gold) in enumerate(zip(outs, golds)):
        for toks in samples:
            gt, reason = grader.project(tok, toks, special, vocab_size)
            meta.append((t, len(toks), reason)); jobs.append((gt, gold))
    results = list(pool.map(grader.classify, [j[0] for j in jobs], [j[1] for j in jobs], chunksize=8))
    per = [dict(n=0, correct=0.0, reward=0.0, pos=0, neg=0, neutral=0, indet=0, len=0) for _ in outs]
    for (t, L, reason), (kind, corr) in zip(meta, results):
        p = per[t]
        if kind == 'indeterminate':
            p['indet'] += 1; continue
        p['n'] += 1; p['correct'] += corr; p['reward'] += grader.overlong_reward(corr, L); p['len'] += L
        p[{'positive': 'pos', 'negative': 'neg', 'neutral': 'neutral'}[kind]] += 1
    for p in per:
        if p['n']:
            p['mean_correct'] = p['correct'] / p['n']; p['mean_reward'] = p['reward'] / p['n']
            p['mean_len'] = p['len'] / p['n']
        else:
            p['mean_correct'] = p['mean_reward'] = p['mean_len'] = None
    return per


def paired_stats(b, a):
    """Paired delta and SE over tasks; plus unpaired SE (independent task draws)."""
    d = [y - x for x, y in zip(b, a)]; K = len(d)
    if K < 2:
        return dict(K=K)
    mean = sum(d) / K
    var = lambda v: sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)
    se_p = math.sqrt(var(d) / K); se_u = math.sqrt(var(b) / K + var(a) / K)
    return dict(K=K, before=sum(b) / K, after=sum(a) / K, delta=mean, se_paired=se_p,
                se_unpaired=se_u, variance_ratio=(se_u / se_p) ** 2 if se_p > 0 else None)


# ---------------------------------------------------------------- main pass
SEED_SCHEME = 'seed = int(sha256(f"{rng_seed}:{task_index}:{j}{suffix}")[:8], 16) & 0x7fffffff; suffix="" for both sides with CRN, ":after" for the after side without CRN'


def sample_seed(rng_seed, task, j, suffix=''):
    return int(hashlib.sha256(f'{rng_seed}:{task}:{j}{suffix}'.encode()).hexdigest()[:8], 16) & 0x7fffffff


def new_draw(a, idx):
    """Each pair gets its own CSPRNG seed -> K uniform tasks (no replacement) + per-(task, j) sampling seeds.
    Common random numbers (default): the before and after generations of (task, j) use the SAME vLLM seed."""
    rng_seed = secrets.randbits(63); rng = random.Random(rng_seed)
    chosen = rng.sample(idx['pool']['indices'], a.K)
    sb = [[sample_seed(rng_seed, i, j) for j in range(a.m)] for i in chosen]
    sa = sb if a.crn else [[sample_seed(rng_seed, i, j, ':after') for j in range(a.m)] for i in chosen]
    return dict(rng_seed=rng_seed, chosen=chosen, seeds={'before': sb, 'after': sa},
                t_start=now(), timings=Timer(), loads={}, side={})


_TOK = {}
def tokenizer_for(path):
    if 'tok' not in _TOK:
        from transformers import AutoTokenizer
        _TOK['tok'] = AutoTokenizer.from_pretrained(str(path))
    return _TOK['tok']


def score_checkpoint(a, eng, idx, tasks, cp, items, pool, cache):
    """Generate+grade checkpoint cp on the union of task lists of several (draw, side) items in ONE batch."""
    rt = idx['checkpoints'][-1]['mimo_runtime']; vocab = idx['checkpoints'][-1]['vocab_size']
    T = Timer()
    with T('download'):
        path, _ = fetch.ensure(cp, a.cache)
    tok = tokenizer_for(path)
    def prompt(i):
        r = tasks[i]['data']
        p = tok.apply_chat_template([dict(role='system', content=r['system_prompt']), dict(role='user', content=r['prompt'])],
                                    tokenize=True, add_generation_prompt=True, enable_thinking=rt['enable_thinking'], return_dict=False)
        return list(p['input_ids'] if isinstance(p, dict) else p)
    key = lambda i, s: f"{cp['checkpoint_id']}:{i}:{','.join(map(str, s))}:{a.max_tokens}"
    work = []  # (item_no, j)
    for n, (d, sidename) in enumerate(items):
        d['side'][sidename] = [cache.get(key(i, s)) for i, s in zip(d['chosen'], d['seeds'][sidename])]
        work += [(n, j) for j, h in enumerate(d['side'][sidename]) if h is None]
    how = 'cached'
    if work:
        with T('load'):
            how = eng.load(cp, path)
        P = [prompt(items[n][0]['chosen'][j]) for n, j in work]
        S = [items[n][0]['seeds'][items[n][1]][j] for n, j in work]
        G = [tasks[items[n][0]['chosen'][j]]['data']['answer'] for n, j in work]
        with T('generate'):
            outs = eng.generate(P, S, rt)
        with T('verify'):
            per = score_outputs(eng.tokenizer, outs, G, vocab, pool)
        for (n, j), p in zip(work, per):
            d, sidename = items[n]; d['side'][sidename][j] = p
            cache.put(key(d['chosen'][j], d['seeds'][sidename][j]), p)
    for d, sidename in items:
        for k, v in T.items():
            d['timings'][f'{k}_{sidename}'] = d['timings'].get(f'{k}_{sidename}', 0) + v
        d['loads'][sidename] = how
        d.setdefault('shared', {})[sidename] = len(items)
    return T


def write_record(a, idx, before, after, d):
    side = d['side']; K = len(d['chosen'])
    keep = [j for j in range(K) if side['before'][j]['n'] and side['after'][j]['n']]
    stat = {k: paired_stats([side['before'][j][k] for j in keep], [side['after'][j][k] for j in keep])
            for k in ('mean_correct', 'mean_reward')}
    tsum = lambda s, k: sum(side[s][j][k] for j in range(K))
    rt = idx['checkpoints'][-1]['mimo_runtime']
    rec = dict(version=VERSION, sampler_id=a.sampler_id, host=socket.gethostname(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
               started_at=d['t_start'], finished_at=now(),
               before=dict(checkpoint_id=before['checkpoint_id'], optimizer_step=before['optimizer_step']),
               after=dict(checkpoint_id=after['checkpoint_id'], optimizer_step=after['optimizer_step'],
                          input_checkpoint=after['input_checkpoint']),
               checkpoint_auth='sha256(canonical(filemap))==id and every file sha256 matched before use',
               pool=dict(env_id='affine_math', size=len(idx['pool']['indices']), task_snapshot=idx['pool']['spec']['config']['task_snapshot']),
               rng_seed=d['rng_seed'], K=K, samples_per_task=a.m,
               generation=dict(engine='vllm', engine_mode=a.engine, chain=bool(a.chain), temperature=rt['temperature'], top_p=rt['top_p'],
                               top_k=rt['top_k'], max_tokens=a.max_tokens, max_context_tokens=rt['max_context_tokens'],
                               enable_thinking=rt['enable_thinking'], eos_token_ids=rt['eos_token_ids'],
                               dtype='bfloat16', common_random_numbers=bool(a.crn), seed_scheme=SEED_SCHEME),
               grader=dict(rule='mimo-final-channel-math-v1 + completed-boxed + math-verify-0.9.0 (local replica)', overlong=grader.OVERLONG),
               tasks=[dict(index=i, seeds_before=d['seeds']['before'][j], seeds_after=d['seeds']['after'][j],
                           before_correct=side['before'][j]['mean_correct'], after_correct=side['after'][j]['mean_correct'],
                           before_reward=side['before'][j]['mean_reward'], after_reward=side['after'][j]['mean_reward'],
                           before_n=side['before'][j]['n'], after_n=side['after'][j]['n'],
                           before_len=side['before'][j]['mean_len'], after_len=side['after'][j]['mean_len'])
                      for j, i in enumerate(d['chosen'])],
               outcome_counts={s: {k: tsum(s, k) for k in ('pos', 'neg', 'neutral', 'indet')} for s in side},
               tasks_used=len(keep), correctness=stat['mean_correct'], reward=stat['mean_reward'],
               delta=stat['mean_correct'].get('delta'), se=stat['mean_correct'].get('se_paired'),
               timings_s={k: round(v, 2) for k, v in d['timings'].items()}, loads=d['loads'],
               batch_shared_with=d.get('shared'), wall_s=round(now() - d['t_start'], 2))
    out = Path(a.results) / f'{a.sampler_id}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'a') as f:
        f.write(json.dumps(rec, separators=(',', ':')) + '\n'); f.flush(); os.fsync(f.fileno())
    c = rec['correctness']
    print(json.dumps(dict(pair=f"{before['checkpoint_id'][:8]}->{after['checkpoint_id'][:8]}", step=after['optimizer_step'], K=K, m=a.m,
                          delta=round(c['delta'], 4), se_paired=round(c['se_paired'], 4), se_unpaired=round(c['se_unpaired'], 4),
                          timings=rec['timings_s'], wall=rec['wall_s'])), flush=True)
    return rec


def run_pair(a, eng, idx, tasks, before, after, pool, cache):
    d = new_draw(a, idx)
    score_checkpoint(a, eng, idx, tasks, before, [(d, 'before')], pool, cache)
    score_checkpoint(a, eng, idx, tasks, after, [(d, 'after')], pool, cache)
    if a.engine == 'reload':
        eng.close()
    return write_record(a, idx, before, after, d)


def run_chain(a, eng, idx, tasks, cps, pool, cache, pending=None, keep_open=False):
    """Score consecutive pairs along cps, generating each checkpoint ONCE on (after-half of the previous pair)
    U (before-half of the next pair). Pairing stays inside this instance; each pair has its own random K.
    pending: a draw whose before-half was already scored on cps[0]. Returns the open draw if keep_open."""
    draws = [new_draw(a, idx) for _ in range(len(cps) - 1 + (1 if keep_open else 0))]
    if pending is not None:
        draws[0] = pending
    for k, cp in enumerate(cps):
        items = []
        if k >= 1:
            items.append((draws[k - 1], 'after'))
        if k < len(draws) and not (k == 0 and pending is not None):
            items.append((draws[k], 'before'))
        if items:
            score_checkpoint(a, eng, idx, tasks, cp, items, pool, cache)
        if k >= 1:
            write_record(a, idx, cps[k - 1], cp, draws[k - 1])
    return draws[-1] if keep_open else None


def grader_canon(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':')).encode()


class Cache:
    """Per-instance score cache keyed by (checkpoint, task, seeds, max_tokens)."""
    def __init__(self, path):
        self.path = Path(path) if path else None; self.d = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                k, v = json.loads(line); self.d[k] = v
    def get(self, k): return self.d.get(k)
    def put(self, k, v):
        self.d[k] = v
        if self.path:
            with open(self.path, 'a') as f: f.write(json.dumps([k, v]) + '\n')


def done_pairs(a):
    p = Path(a.results) / f'{a.sampler_id}.jsonl'; s = set()
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line); s.add((r['before']['checkpoint_id'], r['after']['checkpoint_id']))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', default=str(HERE / 'data/index.json'))
    ap.add_argument('--tasks', default=str(HERE / 'data/original-math7496.tasks.json'))
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--cache', default='/root/ckpt-cache')
    ap.add_argument('--sampler-id', default=None)
    ap.add_argument('-K', '--K', type=int, default=128)
    ap.add_argument('-m', '--m', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--engine', choices=['reload', 'swap'], default='swap')
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--gpu-mem', type=float, default=0.90)
    ap.add_argument('--max-num-seqs', type=int, default=1024)
    ap.add_argument('--max-batched-tokens', type=int, default=16384)
    ap.add_argument('--enforce-eager', action='store_true')
    ap.add_argument('--grade-workers', type=int, default=16)
    ap.add_argument('--score-cache', action='store_true', help='persist per-(ckpt,task,seed) scores for reuse')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--pair'); g.add_argument('--backfill', type=int); g.add_argument('--watch', action='store_true')
    ap.add_argument('--poll', type=float, default=120)
    ap.add_argument('--no-crn', dest='crn', action='store_false', help='independent seeds for the after side (for comparison only)')
    ap.add_argument('--chain', action='store_true', help='generate each checkpoint once for adjacent pairs (optimized)')
    ap.add_argument('--repeat', type=int, default=1, help='with --pair: score the pair this many times (fresh K each)')
    a = ap.parse_args()
    a.sampler_id = a.sampler_id or f"{socket.gethostname()}-g{os.environ.get('CUDA_VISIBLE_DEVICES', 'x')}-{uuid.uuid4().hex[:8]}"
    tasks = json.loads(Path(a.tasks).read_text())
    eng = Engine(a); pool = grader.pool(a.grade_workers)
    cache = Cache(Path(a.results) / f'.{a.sampler_id}.scorecache.jsonl' if a.score_cache else None)
    print(json.dumps(dict(sampler_id=a.sampler_id, K=a.K, m=a.m, engine=a.engine)), flush=True)
    if a.pair:
        idx = load_index(a.index); b, c = a.pair.split(':')
        for _ in range(a.repeat):
            if a.chain:
                run_chain(a, eng, idx, tasks, [find(idx, b), find(idx, c)], pool, cache)
            else:
                run_pair(a, eng, idx, tasks, find(idx, b), find(idx, c), pool, cache)
        return
    if a.backfill is not None:
        idx = load_index(a.index); pairs = consecutive_pairs(idx)[-a.backfill:]; done = done_pairs(a)
        if a.chain:
            # split into runs of consecutive pairs (b_k+1 == a_k) and chain each run
            runs, cur = [], []
            for b, c in pairs:
                if cur and cur[-1]['checkpoint_id'] != b['checkpoint_id']:
                    runs.append(cur); cur = []
                cur = cur or [b]; cur.append(c)
            if cur: runs.append(cur)
            for r in runs:
                run_chain(a, eng, idx, tasks, r, pool, cache)
        else:
            for b, c in pairs:
                if (b['checkpoint_id'], c['checkpoint_id']) not in done:
                    run_pair(a, eng, idx, tasks, b, c, pool, cache)
        return
    # watch: score the newest pair, keep the next pair's before-half pre-scored on the newest checkpoint
    pending, tip = None, None
    while True:
        idx = load_index(a.index); pairs = consecutive_pairs(idx); done = done_pairs(a)
        latest = pairs[-1][1] if pairs else None
        if latest is not None and (tip is None or latest['checkpoint_id'] != tip['checkpoint_id']):
            if a.chain and pending is not None and latest.get('input_checkpoint') == tip['checkpoint_id']:
                pending = run_chain(a, eng, idx, tasks, [tip, latest], pool, cache, pending=pending, keep_open=True)
            elif (pairs[-1][0]['checkpoint_id'], latest['checkpoint_id']) not in done:
                if a.chain:
                    pending = run_chain(a, eng, idx, tasks, list(pairs[-1]), pool, cache, keep_open=True)
                else:
                    run_pair(a, eng, idx, tasks, *pairs[-1], pool, cache)
            elif a.chain:
                pending = run_chain(a, eng, idx, tasks, [latest], pool, cache, keep_open=True)
            tip = latest
        time.sleep(a.poll)


if __name__ == '__main__':
    main()
