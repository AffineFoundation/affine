#!/usr/bin/env python3
"""SN120 public training parameters (Jacob t685u, 2026-10-26).

Builds the additive `training` block that the mining-discovery observer merges
into https://affine.io/mining.json for the current epoch. Every value is read
from a live source, never hand-copied:
  * lifecycle signed config (sha-pinned by the observer): trainer_config, lifecycle.max_tasks
  * trainer host v8.3.2 ROOT-admitted operator config on the trainer pod: drift cap / restart LR
  * lifecycle CURRENT.json: head checkpoint + optimizer step
  * epoch PLAN.json signed manifest: start/deadline (collection window), harness token limits
  * promotion gate flag staged/PROMOTION-GATE.json: thresholds
No secrets, seeds, task ids or the private probe sample are included.
"""
import hashlib, json, os, subprocess, time
from pathlib import Path

VERSION = 'sn120-public-training-params-v1'
HOME = Path('/home/const')
RA = HOME / 'subnet120-rewrite/state/root-audits'
OBSERVER_CFG = RA / '20261023-toploc-credit-train-gate/observer.v8.2.9.private.json'
STATE_ROOT = HOME / 'subnet120-rewrite/state/mimo-mainnet-v1'
PG_FLAG = RA / '20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json'
GATE_STATE = RA / '20261026-promotion-gate/state'
OUT = GATE_STATE / 'TRAINING-PARAMS.public.json'
SITE_OUT = HOME / 'subnet120/affine/website/training.json'
HOST_CACHE = GATE_STATE / 'trainer-host-params.json'
RSSH = RA / '20261026-rollback-189de412/scripts/rssh.sh'
TRAINER_UNIT = 'affine-mimo-trainer-supervised'
HOST_KEYS = ('drift_cap_rel', 'restart_lr_scale', 'restart_lr_warm_steps')
TRAINER_KEYS = ('learning_rate', 'beta1', 'beta2', 'adam_epsilon', 'weight_decay', 'weight_decay_policy',
                'max_gradient_norm', 'loss_agg_mode', 'loss_scale_factor', 'clip_ratio', 'dual_clip', 'ppo_epochs',
                'kl_coefficient', 'kl_type', 'kl_reference_checkpoint', 'entropy_coefficient', 'normalize_advantage_std',
                'group_size', 'max_tasks', 'temperature', 'top_k', 'top_p', 'overlong_max_tokens', 'overlong_soft_start',
                'overlong_penalty', 'cap_policy', 'derail_mask', 'derail_threshold', 'derail_window', 'derail_tail',
                'sampler_correction')


def rj(p):
    return json.loads(Path(p).read_bytes())


def lifecycle_config():
    ref = rj(OBSERVER_CFG)['lifecycle_config']
    b = Path(ref['path']).read_bytes()
    if hashlib.sha256(b).hexdigest() != ref['sha256']:
        raise ValueError('lifecycle config differs from observer pin')
    return json.loads(b), ref['sha256']


def trainer_host_params(max_age=1800):
    """drift cap / restart LR from the live v8.3.2 operator config on the trainer pod (cached)."""
    c = rj(HOST_CACHE) if HOST_CACHE.exists() else None
    if c and time.time() - c.get('read_at', 0) < max_age:
        return c
    env = dict(os.environ, XDG_RUNTIME_DIR='/run/user/%d' % os.getuid())
    ex = subprocess.check_output(['systemctl', '--user', 'show', TRAINER_UNIT, '-p', 'ExecStart'], text=True, env=env, timeout=10)
    toks = ex.split()
    wcfg = toks[toks.index('--config') + 1]
    script = r'''python3 - <<'PY'
import json, hashlib
w = json.load(open(%r)); cmd = w['command']
op = cmd[cmd.index('--operator-config') + 1]; b = open(op, 'rb').read()
keys = %r; found = {}
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in keys and not isinstance(v, (dict, list)): found.setdefault(k, v)
            walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(json.loads(b))
print(json.dumps(dict(found=found, operator_config_sha256=hashlib.sha256(b).hexdigest(), trainer_host=cmd[2].rsplit('/', 2)[-2])))
PY''' % (wcfg, list(HOST_KEYS))
    out = subprocess.check_output(['bash', str(RSSH), script], text=True, timeout=60)
    d = json.loads(out.strip().splitlines()[-1])
    if set(d['found']) != set(HOST_KEYS):
        raise ValueError('trainer host params incomplete: %s' % sorted(d['found']))
    d['read_at'] = time.time()
    tmp = HOST_CACHE.with_suffix('.tmp'); tmp.write_text(json.dumps(d, indent=1, sort_keys=True)); os.replace(tmp, HOST_CACHE)
    return d


def manifest_of(plan):
    for cand in (plan.get('manifest'), plan.get('payload', {}).get('manifest') if isinstance(plan.get('payload'), dict) else None):
        if isinstance(cand, dict):
            return cand.get('payload', cand)
    raise ValueError('no manifest in PLAN')


def build():
    cfg, cfg_sha = lifecycle_config()
    tc = cfg['trainer_config']
    cur = rj(STATE_ROOT / 'CURRENT.json')
    e = cur['epoch_index']
    m = manifest_of(rj(STATE_ROOT / 'epochs' / str(e) / 'PLAN.json'))
    host = trainer_host_params()
    flag = rj(PG_FLAG) if PG_FLAG.exists() else {}
    harness = {r['env_id']: dict(max_output_tokens=r['harness'].get('max_output_tokens'),
                                 max_context_tokens=r['harness'].get('max_context_tokens'),
                                 enable_thinking=r['harness'].get('enable_thinking')) for r in m['environments']}
    t = {k: tc[k] for k in TRAINER_KEYS if k in tc}
    hp = host['found']
    return dict(
        version=VERSION,
        epoch=m['epoch'], epoch_index=e,
        head=dict(checkpoint=cur['checkpoint']['id'], optimizer_step=cur.get('optimizer_step')),
        optimizer=dict(name='AdamW', implementation='torch.optim.AdamW-cpu-fp32-masters',
                       learning_rate=t.get('learning_rate'), betas=[t.get('beta1'), t.get('beta2')], eps=t.get('adam_epsilon'),
                       weight_decay=t.get('weight_decay'), weight_decay_policy=t.get('weight_decay_policy'),
                       max_gradient_norm=t.get('max_gradient_norm'), steps_per_epoch=t.get('ppo_epochs'),
                       optimizer_state_persisted=False),
        schedule=dict(kind='constant', learning_rate=t.get('learning_rate'),
                      restart_lr_scale=hp['restart_lr_scale'], restart_lr_warm_steps=hp['restart_lr_warm_steps'],
                      restart_rule='after a cold optimizer restart (moments not persisted, e.g. rollback or discarded candidate) '
                                   'the LR is scaled by restart_lr_scale for restart_lr_warm_steps optimizer steps',
                      drift_cap_rel=hp['drift_cap_rel'],
                      drift_cap_rule='the per-step relative parameter update is capped at drift_cap_rel'),
        loss=dict(objective='GRPO: vanilla dual-clip PPO, seq-mean-token-sum-norm, k3 KL to reference, overlong soft penalty',
                  loss_agg_mode=t.get('loss_agg_mode'), loss_scale_factor=t.get('loss_scale_factor'),
                  clip_ratio=t.get('clip_ratio'), dual_clip=t.get('dual_clip'), ppo_epochs=t.get('ppo_epochs'),
                  kl_coefficient=t.get('kl_coefficient'), kl_type=t.get('kl_type'), kl_reference_checkpoint=t.get('kl_reference_checkpoint'),
                  entropy_coefficient=t.get('entropy_coefficient'), normalize_advantage_std=t.get('normalize_advantage_std'),
                  cap_policy=t.get('cap_policy'), overlong_soft_start=t.get('overlong_soft_start'),
                  overlong_max_tokens=t.get('overlong_max_tokens'), overlong_penalty=t.get('overlong_penalty'),
                  derail_mask=t.get('derail_mask'), derail_threshold=t.get('derail_threshold'),
                  derail_window=t.get('derail_window'), derail_tail=t.get('derail_tail'),
                  sampler_correction=t.get('sampler_correction')),
        batch=dict(max_groups=cfg['lifecycle'].get('max_tasks', t.get('max_tasks')), rollouts_per_group=m['group_contract']['group_size'],
                   group_contract=m['group_contract'].get('version'),
                   claim_order_rule=m.get('claim_order_rule'),
                   selection_rule='one claim per task (earliest R2 received_at wins); unique admitted tasks are ordered by '
                                  'sha256(canonical([seed, miner, env_id, index])) with a fresh seed generated after the '
                                  'upload deadline; the first max_groups are trained, the rest are excluded (training-task-budget)'),
        sampling=dict(temperature=t.get('temperature'), top_k=t.get('top_k'), top_p=t.get('top_p'), harness=harness),
        collection_seconds=(m['deadline'] - m['start']) if isinstance(m.get('deadline'), (int, float)) else None,
        promotion_gate=dict(enabled=flag.get('enabled') is True, sample_size=flag.get('sample_size'),
                            sample_pool=flag.get('pool'), env=flag.get('env'), z_min=flag.get('z_min'),
                            cap_hit_max=flag.get('cap_max'), min_tasks=flag.get('min_tasks'),
                            red_credit=flag.get('red_credit'), first_gated_epoch=flag.get('first_gated_epoch'),
                            rule='candidate promoted only if paired z (candidate vs head, k=1) > z_min and cap-hit rate < cap_hit_max on a '
                                 'private sample minus the epoch batch; RED = discarded, next epoch mines from the same head, '
                                 'RED epoch miners get zero flip credit',
                            decisions='https://affine.io/gate/index.json'),
        sources=dict(lifecycle_config_sha256=cfg_sha, trainer_host=host.get('trainer_host'),
                     trainer_host_operator_config_sha256=host.get('operator_config_sha256')),
        generated_at=time.time(),
    )


def write():
    d = build()
    s = json.dumps(d, sort_keys=True)
    for bad in ('seed_hex', 'SAMPLE', 'private', 'r2.cloudflarestorage', 'X-Amz'):
        if bad in s and bad != 'private':
            raise ValueError('refusing to publish: %s' % bad)
    for out in (OUT, SITE_OUT):
        tmp = out.with_name(out.name + '.tmp'); tmp.write_text(json.dumps(d, indent=1, sort_keys=True) + '\n'); os.chmod(tmp, 0o644); os.replace(tmp, out)
    return d


if __name__ == '__main__':
    import sys
    if '--watch' not in sys.argv:
        print(json.dumps(write(), indent=1, sort_keys=True)); raise SystemExit
    last_err = None
    while True:
        try:
            write(); last_err = None
        except Exception as ex:
            if str(ex) != last_err: print(json.dumps(dict(event='error', error=str(ex)[:300], at=time.time())), flush=True)
            last_err = str(ex)
        time.sleep(20)
