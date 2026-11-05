#!/usr/bin/env python3
"""Check your training gradient against the live trainer's (after Allan, catoneone/sn120-k1-fixes exp024/phase5 grad_probe.py).

Same loss, same rollouts, different forward/backward implementations; the gradient of each is saved per tensor (fp32) and
two runs are compared by cosine and norm ratio, overall and per 8-layer block.

  python3 kit/grad_probe.py run --head HEAD_DIR --batch batch.json --hotkey SS58 --mode kit     --out g_kit.safetensors
  python3 kit/grad_probe.py run --head HEAD_DIR --batch batch.json --hotkey SS58 --mode trainer --out g_trainer.safetensors
  python3 kit/grad_probe.py compare g_kit.safetensors g_trainer.safetensors [--out cmp.json]

Loss: L = sum_i A_i * mean_t log p_t (full-vocab log p of the sampled tokens; A_i = reward - group mean) over --n rollouts
with A != 0 (n/3 each from the short, middle and long third by length, fixed seed). One rollout per backward, gradients
summed in fp32, activation checkpointing on (as in the live trainer). Any public GREEN batch works
(https://affine.io/k1/turns/<epoch>/batch.json, hotkey in turn.json), with any head.

Modes:
  kit      what kit/train_step.py runs by default on CUDA (train_step.install_trainer_kernels + engine packed).
  trainer  the live trainer's rank forward, set up directly from miner/validator/trainer-v833m (v83_fast_kernels.install +
           packed_log_probs with v83_dp.LOGIT_CHUNK), independent of train_step.py.
  legacy   the pre-2026-11-05 kit path (engine padded, stock transformers forward: torch F.conv1d GDN short conv).
  fp32ref  fp32 weights, pure torch (torch_chunk_gated_delta_rule, F.conv1d, torch gated RMSNorm, sdpa): a slow reference.
What to expect on one H200 (24 rollouts, see README "Check your gradients"): kit vs trainer cos >= 0.99 and norm ratio ~1.0
in every block; any bf16 path vs fp32ref is lower (bf16 rounding); legacy is far off in layers 0-7. If your kit vs trainer
is far below 0.99, your environment (package versions, kernels, edited files) does not train like the gate.
"""
import argparse, collections, json, math, os, random, re, sys, time
from pathlib import Path
import kitpath  # noqa: F401  (sys.path + namespace stubs)


def pick(groups, n, seed=0):
    rows = []
    for g in groups:
        rs = [float(r['reward']) for r in g['rollouts']]; m = sum(rs) / len(rs)
        for r in g['rollouts']:
            a = float(r['reward']) - m
            if a != 0:
                rows.append(dict(prompt=list(g['prompt_tokens']), out=list(r['tokens']), adv=a))
    if len(rows) < n:
        raise SystemExit('only %d rollouts with a nonzero advantage (need --n %d)' % (len(rows), n))
    rows.sort(key=lambda r: len(r['out'])); k = len(rows) // 3; rng = random.Random(seed); out = []
    for seg in (rows[:k], rows[k:2 * k], rows[2 * k:]):
        out += rng.sample(seg, n // 3)
    return out


def run(a):
    import torch
    import ff_format, train_step, v8211_batched as B
    tok = ff_format.tokenizer(a.head)
    v = ff_format.validate(Path(a.batch).read_bytes(), a.hotkey, tok, 125, 1000, v2=True)
    rows = pick(v['groups'], a.n, a.seed)
    dev = torch.device('cuda'); fp32 = a.mode == 'fp32ref'
    model = train_step.load_model(a.head, dev, torch.float32 if fp32 else torch.bfloat16, True)
    rt = train_step.Runtime(model, 'grad-probe', dev, True, (248046, 248044), ff_format.MAX_CONTEXT)
    if a.mode == 'kit':
        info = train_step.install_trainer_kernels(model)
        logp = lambda P, O: B.batched_log_probs(rt, [(P, O)], gradients=True, engine='packed')[0]
    elif a.mode == 'trainer':
        import v83_fast_kernels as FK, v83_dp as DP
        info = FK.install(model)
        logp = lambda P, O: FK.packed_log_probs(rt, [(P, O)], gradients=True, logit_chunk=DP.LOGIT_CHUNK)[0]
    elif a.mode in ('legacy', 'fp32ref'):
        info = None
        if fp32:
            import v83_fast_kernels as FK
            from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
            for layer in model.model.language_model.layers:
                if layer.block_type == 'linear_attention':
                    m = layer.linear_attn; m.chunk_gated_delta_rule = Q.torch_chunk_gated_delta_rule; m.causal_conv1d_fn = None
                    m.norm.forward = (lambda h, g, _n=m.norm: FK.gated_rmsnorm_hf(_n, h, g))
        original = getattr(B.batched_log_probs, '_original', B.batched_log_probs)
        logp = lambda P, O: original(rt, [(P, O)], gradients=True, engine='padded')[0]
    else:
        raise SystemExit('--mode kit|trainer|legacy|fp32ref')
    train_step.enable_activation_checkpointing(model)
    model.eval()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    acc = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params}
    t0 = time.time(); log = []
    for i, r in enumerate(rows):
        lp = logp(r['prompt'], r['out'])
        (r['adv'] * lp.mean()).backward()
        for n, p in params:
            if p.grad is not None:
                acc[n] += p.grad.float(); p.grad = None
        log.append(dict(i=i, tokens=len(r['out']), adv=r['adv'], logp_mean=float(lp.mean())))
        print(json.dumps(log[-1]), flush=True)
    from safetensors.torch import save_file
    save_file({n: t.contiguous().cpu() for n, t in acc.items()}, a.out)
    Path(a.out + '.json').write_text(json.dumps(dict(mode=a.mode, kernels=info, head=str(a.head), batch=str(a.batch), n=a.n,
                                                     seed=a.seed, seconds=time.time() - t0, rows=log), indent=1))
    print(json.dumps(dict(done=a.mode, seconds=round(time.time() - t0), out=a.out)), flush=True)


def block(name):
    m = re.match(r'model\.language_model\.layers\.(\d+)\.', name)
    if not m:
        return 'top'
    b = int(m.group(1)) // 8 * 8
    return '%02d-%02d' % (b, b + 7)


def compare(a):
    from safetensors import safe_open
    A, Bf = safe_open(a.a, 'pt'), safe_open(a.b, 'pt')
    names = sorted(set(A.keys()) & set(Bf.keys()))
    if not names:
        raise SystemExit('no common tensors')
    grp = collections.defaultdict(lambda: [0.0, 0.0, 0.0]); worst = []
    for n in names:
        x, y = A.get_tensor(n).double().flatten(), Bf.get_tensor(n).double().flatten()
        d, nx, ny = float(x @ y), float(x @ x), float(y @ y)
        for k in ('all', block(n)):
            g = grp[k]; g[0] += d; g[1] += nx; g[2] += ny
        if nx and ny:
            worst.append((d / math.sqrt(nx * ny), n))
    res = {k: dict(cos=g[0] / math.sqrt(g[1] * g[2]) if g[1] and g[2] else None, ratio=math.sqrt(g[1] / g[2]) if g[2] else None)
           for k, g in sorted(grp.items())}
    out = dict(a=a.a, b=a.b, tensors=len(names), ratio='norm(a) / norm(b)', blocks=res,
               worst_tensors=[dict(name=n, cos=c) for c, n in sorted(worst)[:10]])
    for k, v in res.items():
        print('%-6s cos %.4f  norm ratio %.4f' % (k, v['cos'], v['ratio']))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--head', required=True); r.add_argument('--batch', required=True)
    r.add_argument('--hotkey', required=True); r.add_argument('--mode', required=True, choices=('kit', 'trainer', 'legacy', 'fp32ref'))
    r.add_argument('--n', type=int, default=24); r.add_argument('--seed', type=int, default=0); r.add_argument('--out', required=True)
    c = sub.add_parser('compare'); c.add_argument('a'); c.add_argument('b'); c.add_argument('--out')
    a = ap.parse_args(argv)
    return run(a) if a.cmd == 'run' else compare(a)


if __name__ == '__main__':
    main()
