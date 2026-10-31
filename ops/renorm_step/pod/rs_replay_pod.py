#!/usr/bin/env python3
"""RA/20261031-renorm-step-live 8-vs-1 check, SHADOW pod side: Allan's bf16_exp.py stream (fp32_renorm + normgrad, 1x, whole file as one
step = the live v8.3.3i recipe) on ONE GPU, from head 87fa7408 + its published fp32 master. Wraps Env.gradient only to also dump the
per-rollout details (loss, loss_tokens, support_outside_tokens, support_logq_mean). Plain disk /data/rs; no secrets.
usage: rs_replay_pod.py FF_FILE OUT_DIR SAVE_DIR"""
import importlib.util, json, sys
from pathlib import Path
ff, out, save = sys.argv[1:4]
H = '87fa740811e75e4c221258ce6cf2d8cb1db1519d69aa0a65b42cb5fddea5a499'
spec = importlib.util.spec_from_file_location('bf16_exp', '/data/allan/code/bf16_exp.py'); M = importlib.util.module_from_spec(spec)
sys.modules['bf16_exp'] = M; spec.loader.exec_module(M)
orig = M.Env.gradient
def gradient(self, actor, params, T, refdir):
    B = self.B; og = B._gradient_pass; cap = {}
    def gp(*a, **k):
        cap['details'] = a[9]; return og(*a, **k)
    B._gradient_pass = gp
    try:
        total, info = orig(self, actor, params, T, refdir)
    finally:
        B._gradient_pass = og
    keys = ('loss', 'loss_tokens', 'support_outside_tokens', 'support_logq_mean', 'task_id', 'nonce', 'tokens', 'advantage', 'k3_kl_mean', 'derail_index')
    rows = [dict(i=i, **{k: d.get(k) for k in keys}) for i, d in sorted(cap['details'].items())]
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'details.json').write_text(json.dumps(dict(info=info, rows=rows), default=str))
    return total, info
M.Env.gradient = gradient
sys.argv = ['bf16_exp.py', 'stream', '--source', '/data/src', '--refdir', str(Path(out).parent / 'refcache'), '--ref', '/data/models/base',
            '--arm', 'fp32_renorm', '--opt', 'normgrad', '--step-mult', '1.0', '--host', '/data/allan/code/replay/host_renorm',
            '--start', '/data/ckpt/' + H, '--start-master', '/data/ckpt/fp32master-' + H, '--ff', ff, '--ff-tok', '/data/ckpt/' + H,
            '--ff-tag', 'rs', '--ff-per-step', '0', '--ff-seed', '0', '--probe-turn', '-1', '--save', save, '--out', out]
M.main()
