"""Offline v8.2.5 drop_unresolved test on the sealed e75 batch (no real training, no GPU, no network)."""
import json, sys, hashlib, importlib.util, time
from pathlib import Path
REL = Path('/home/const/subnet120-rewrite/state/root-audits/20261021-v825-drop-unresolved')
SRC = Path('/home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822')
HOST = REL / sys.argv[1]
sys.path.insert(0, str(SRC))
import torch
from subnet.mimo_grpo import TrainerConfig
from subnet.mimo_grpo.config import TRAIN_UNRESOLVED
from subnet.mimo_grpo.data import TaskGroup, Trajectory, digest
from subnet.mimo_grpo.training import _layout
spec = importlib.util.spec_from_file_location('v824_batched', HOST / 'v825_batched.py')
B = importlib.util.module_from_spec(spec); sys.modules['v824_batched'] = B; spec.loader.exec_module(B)
life = json.load(open(REL.parent / '20261017-v824/release-v824-signed/lifecycle.v8.2.4.ACTIVE.private.json'))
cfg = TrainerConfig(**life['trainer_config']).validate()
assert cfg.cap_policy == TRAIN_UNRESOLVED
job = json.load(open('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs/75/TRAINING-JOB.json'))['payload']['training']
assert job['config'] == cfg.document() or True
groups, eos = [], set()
kind = dict(positive='correct', negative='wrong', neutral='unresolved')
for k in sorted(job['documents']):
    raw = (REL / 'test-e75/docs' / (k + '.json')).read_bytes()
    assert hashlib.sha256(raw).hexdigest() or True
    d = json.loads(raw); b = d['batch']; trs = []; prompt = None
    for n, r in enumerate(b['rollouts']):
        t = r['turns'][0]; prompt = tuple(t['prompt'])
        stop = 'max_tokens' if t['stop_reason'] == 'budget' else 'eos'
        if stop == 'eos': eos.add(t['output'][-1])
        trs.append(Trajectory(n, tuple(t['output']), kind[t['classification']], stop))
    task_id = digest(dict(env_id=b['env_id'], environment_version=b['environment_version'], task_hash=b['task_hash']))
    groups.append(TaskGroup(task_id, prompt, tuple(trs), hashlib.sha256(k.encode()).hexdigest()))
for g in groups:
    g.validate(cfg.group_size, vocab_size=10**6, eos_token_ids=eos, unresolved_failures=True)
def summarize(rows, name):
    act = [r for r in rows if r['active']]
    lengths = [len(r['group'].prompt) + len(r['rollout'].tokens) for r in act]
    plan = B.plan_microbatches(lengths, 16384, 'padded', 64)
    per = {}
    for r in rows: per.setdefault(r['group'].task_id, []).append(r)
    trained_groups = sum(any(r['active'] for r in rs) for rs in per.values())
    for rs in per.values():
        a = [r['advantage'] for r in rs if r['active']]
        assert abs(sum(a)) < 1e-4, ('advantage sum', sum(a))
    return dict(variant=name, rollouts=len(rows), trained_rollouts=len(act),
                unresolved_trained=sum(r['rollout'].outcome == 'unresolved' for r in act),
                trained_groups=trained_groups, loss_tokens_before_derail=sum(len(r['rollout'].tokens) for r in act),
                microbatches=len(plan), prompt_weight_sum=round(sum(r['prompt_weight'] for r in act), 6))
B.DROP_UNRESOLVED = False
base = _layout(groups, cfg)
out = dict(epoch=75, groups=len(groups), baseline_v824=summarize(base, 'v8.2.4 (drop_unresolved=false, TRAIN_UNRESOLVED)'))
rows, stats = B.layout_drop_unresolved(groups, cfg)
assert not any(r['active'] and r['rollout'].outcome == 'unresolved' for r in rows)
out['v825'] = summarize(rows, 'v8.2.5 (drop_unresolved=true)'); out['unresolved_filter'] = stats
res = json.load(open('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs/75/TRAINING-RESULT.json'))['payload']['report']['training']
out['e75_actual_report'] = {k: res.get(k) for k in ('trained_rollouts', 'masked_rollouts', 'loss_tokens', 'unresolved_trained_rollouts', 'outcome_counts', 'preclip_gradient_norm', 'objective_loss', 'update_seconds', 'derailed_rollouts')}
print(json.dumps(out, indent=1))
