"""End-to-end CPU mock of train_update_batched (v8.2.5) on real e75 groups with a tiny random model.
No real model, no GPU, no network, no publication. Validates the full loss/backward/report path."""
import json, sys, types, runpy
from pathlib import Path
sys.argv = ['x', 'release-v825-signed/trainer-host-v825']
g = runpy.run_path(str(Path(__file__).with_name('test_e75.py')), run_name='lib')  # builds groups (prints layout summary)
import torch
B, groups, cfg, eos = g['B'], g['groups'], g['cfg'], g['eos']
from subnet.mimo_grpo.optimizer import FP32Accumulator
from subnet.mimo_grpo.data import InputBinding
V = 1 + max(max(max(x.prompt), max(max(r.tokens) for r in x.trajectories)) for x in groups)
class Tiny(torch.nn.Module):
    def __init__(s, seed):
        super().__init__(); torch.manual_seed(seed)
        s.model = types.SimpleNamespace(); s.emb = torch.nn.Embedding(V, 4); s.lm_head = torch.nn.Linear(4, V, bias=False)
        s.model = lambda ids, **k: types.SimpleNamespace(last_hidden_state=torch.tanh(s.emb(ids)))
class RT:
    def __init__(s, seed, ck):
        s.model = Tiny(seed); s.device = 'cpu'; s.trainable = True; s.checkpoint_id = ck; s.vocab_size = V
        s.eos_token_ids = tuple(eos); s.max_context_tokens = 40000
    def guard(s): pass
    def _tokens(s, t): return list(t)
class Opt:
    def __init__(s, rt, ck, step):
        s.input_checkpoint = ck; s.global_step = step; s.config = cfg; s.failed = False
        s.parameters = [(n, p) for n, p in rt.model.named_parameters() if p.requires_grad]
    def zero_grad(s):
        for _, p in s.parameters: p.grad = None
    def step(s, buffers): s.global_step += 1; return dict(mock=True)
per = {}
dropped = [x for x in groups if sum(r.outcome != 'unresolved' for r in x.trajectories) < 2]
short = sorted([x for x in groups if x not in dropped and any(r.outcome == 'unresolved' for r in x.trajectories)],
               key=lambda x: max(len(r.tokens) for r in x.trajectories))[:3]
sub = tuple(dropped + short)
ck = 'a' * 64
import dataclasses
binding = types.SimpleNamespace(validate=lambda: None, checkpoint=ck, optimizer_step=76, run='r', epoch='e', contract_sha256='c' * 64)
out = {}
for flag in (False, True):
    B.DROP_UNRESOLVED = flag
    rt, ref = RT(1, ck), RT(2, cfg.kl_reference_checkpoint)
    rep = B.train_update_batched(rt, Opt(rt, ck, 76), sub, binding, config=cfg, reference_runtime=ref,
                                 engine='padded', token_budget=16384, max_rows=64, logit_chunk=512)
    out['drop_unresolved=%s' % flag] = {k: rep[k] for k in ('status', 'group_count', 'rollout_count', 'trained_rollouts', 'masked_rollouts',
        'loss_tokens', 'unresolved_trained_rollouts', 'objective_loss', 'preclip_gradient_norm', 'unresolved_filter')}
print(json.dumps(dict(e2e_subset_groups=len(sub), dropped_groups_in_subset=len(dropped), results=out), indent=1, default=str))
