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
SGD_KEYS = ('sgd_lr', 'sgd_update_norm_cap', 'sgd_mode', 'sgd_freeze', 'sgd_tensor_rel_cap', 'sgd_fp32_master', 'sgd_direction', 'support_renorm')   # + v8.3.3i   # t712u K=1: trainer host v8.3.3 (one stateless SGD step)
K1_FLAG = RA / '20261023-toploc-credit-train-gate/staged/K1.json'
SCORE_ALL_FLAG = RA / '20261023-toploc-credit-train-gate/staged/SCORE-ALL-EPOCHS.json'  # Jacob t737u: scores update on pass and fail
K1_QUEUE = RA / '20261027-k1-per-uid/state/K1-QUEUE.public.json'
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
            if k in keys and (not isinstance(v, dict) or k == 'support_renorm'): found.setdefault(k, v)
            walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(json.loads(b))
print(json.dumps(dict(found=found, operator_config_sha256=hashlib.sha256(b).hexdigest(), trainer_host=cmd[2].rsplit('/', 2)[-2])))
PY''' % (wcfg, list(HOST_KEYS + SGD_KEYS))
    out = subprocess.check_output(['bash', str(RSSH), script], text=True, timeout=60)
    d = json.loads(out.strip().splitlines()[-1])
    if not (set(d['found']) == set(HOST_KEYS) or {'sgd_lr', 'sgd_update_norm_cap'} <= set(d['found']) <= set(SGD_KEYS)):
        raise ValueError('trainer host params incomplete: %s' % sorted(d['found']))
    d['read_at'] = time.time()
    tmp = HOST_CACHE.with_suffix('.tmp'); tmp.write_text(json.dumps(d, indent=1, sort_keys=True)); os.replace(tmp, HOST_CACHE)
    return d


def last_turn_training():
    """t764u: the loss aggregation and micro-batch plan the trainer actually used on the newest trained turn (trainer report)."""
    best = None
    for d in (STATE_ROOT / 'epochs').iterdir():
        if d.name.isdigit() and (d / 'TRAINING-RESULT.json').exists() and (best is None or int(d.name) > int(best.name)):
            best = d
    if best is None:
        return {}
    r = rj(best / 'TRAINING-RESULT.json'); r = r.get('payload', r)
    t = (r.get('report') or {}).get('training') or {}
    sgd = ((t.get('optimizer') or {}).get('sgd')) or {}
    return dict(epoch=int(best.name), loss_aggregation=t.get('loss_aggregation'), microbatch=t.get('microbatch'),
                runtime_versions=(r.get('report') or {}).get('runtime_versions'), support_renorm=t.get('support_renorm'),
                sgd_direction=sgd.get('sgd_direction'), raw_update_norm=sgd.get('raw_update_norm'),
                update_norm_master=(sgd.get('fp32_master') or {}).get('update_norm_master'))


TRANSPARENCY = dict(   # t764u: post-decision public artifacts (sidecar affine-k1-transparency)
    turns_index='https://affine.io/k1/turns/index.json',
    turn='https://affine.io/k1/turns/<epoch>/turn.json (after the gate decision: decision, the selected signed dispatch rows with '
         'native grades, the miner batch documents byte-exact under /k1/turns/<epoch>/docs/, trainer report: per-rollout loss/advantage/'
         'kept tokens, pre-clip norm, micro-batch plan, optimizer and per-tensor cap stats, candidate checkpoint id + file sha256 for GREEN and RED)',
    gate_reveal='https://affine.io/gate/reveal/<epoch>.json (GREEN epochs only; RED epochs are never revealed. From e222 the gate uses a STANDING private sample, revealed only after it is rotated out (24 h or a head re-probe); before e222 after each decision: sample seed, the 1000 sampled task ids, commitment check, '
                'plan seed, per-task head and candidate grades; verify with https://affine.io/gate/reveal/verify_gate_reveal.py)',
    step_recipe='https://affine.io/k1/spec/step-recipe.json',
    replay='https://affine.io/k1/replay/README.md (replay_k1.py + our own replay results)',
    dev_split='https://affine.io/k1/dev/README.md')


def heldout_se(h):
    # gate noise floor (Jacob pt900u, 2026-10-31): how the held-out SE is built, read from PROMOTION-GATE.json "heldout"
    return dict(formula='weighted_z = gain / SE ; SE = sqrt(se_task^2 + sigma_run^2 * (1 + 1/3))',
                se_task='empirical per-task standard error of the weighted gain: sqrt(sum w^2 g^2) / sum w',
                sigma_run='run-to-run noise of the head, measured each round from its 3 replicate head runs on the kept tasks '
                          '(between-run variance of the weighted mean minus within-run sampling variance); used as max(sigma_run_floor, measured)',
                sigma_run_floor=h.get('sigma_run', 0.022), sigma_run_history=h.get('sigma_run_history'))


def heldout_public():
    # held-out rounds (Jacob pt808u): the gate's public record of the open round; absent = held-out gate not active
    flag = rj(PG_FLAG) or {}
    h = flag.get('heldout') if isinstance(flag.get('heldout'), dict) else {}
    if h.get('enabled') is not True: return None
    try: cur = rj(HOME / 'subnet120/affine/website/gate/heldout/current.json')   # absent until the first round opens
    except Exception: cur = None
    keep = ('round_id', 'head', 'set_size', 'pool', 'env', 'commitment', 'opened_at', 'opened_at_cr', 'opened_epoch', 'baseline', 'k_cand',
            'bar_z', 'z_inclusive', 'status', 'rule')
    return dict(from_epoch=h.get('from_epoch'), bar_z=h.get('bar_z', 1.0), z_inclusive=h.get('z_inclusive', False), se=heldout_se(h),
                current_round=({k: cur.get(k) for k in keep} if isinstance(cur, dict) else None),
                rounds='https://affine.io/gate/heldout/index.json', current='https://affine.io/gate/heldout/current.json',
                rule=('one secret set of set_size tasks per head (a round); the head is scored once per round: a selection run (4 rollouts '
                      'per task on all of the set) locks the kept tasks (0 < solved < 4), then 12 rollouts per kept task give p_i; every '
                      'candidate of the round is scored on the kept tasks minus its own batch tasks (4 rollouts per task, the same request seeds '
                      'as the head, CRN) by inverse-variance-weighted gain over p_i and promoted only if gain > 0, z > bar_z and estimated '
                      'cap-hit < 0.20. Only the commitment is public while the round is open; a round is revealed only if a GREEN closed it, '
                      'and then only the GREEN turn (seed, task ids, the head baseline and the GREEN candidate per-task grades) at '
                      'https://affine.io/gate/heldout/<round>.json; RED turns and their probe results are never published'))


def manifest_of(plan):
    for cand in (plan.get('manifest'), plan.get('payload', {}).get('manifest') if isinstance(plan.get('payload'), dict) else None):
        if isinstance(cand, dict):
            return cand.get('payload', cand)
    raise ValueError('no manifest in PLAN')


# v8.2.9x (Jacob t796u, 2026-10-29): tunable per-miner batch cap. The lifecycle applies staged/BATCH-CAP.json at the K=1
# selection and records the value used for each opening in epochs/N/BATCH-CAP.json; this publishes that record.
BATCH_CAP_FILE = RA / '20261023-toploc-credit-train-gate/staged/BATCH-CAP.json'


def staged_batch_cap(ceiling):
    """Same bounds as the host (v829x_cap): invalid/absent -> 125 x 8 default."""
    dflt = dict(max_tasks_per_miner=min(125, ceiling), rollouts_per_task=8, max_rollouts_per_miner=min(125, ceiling) * 8, min_tasks=8)
    try:
        d = rj(BATCH_CAP_FILE)
        t, r, m, mn = (d.get(k) for k in ('max_tasks_per_miner', 'rollouts_per_task', 'max_rollouts_per_miner', 'min_tasks'))
        if not (all(type(x) is int for x in (t, r, m, mn)) and r == 8 and 8 <= t <= ceiling and 64 <= m <= 1000 and mn == 8):
            return dflt
        eff = min(t, m // r)
        return dict(max_tasks_per_miner=eff, rollouts_per_task=r, max_rollouts_per_miner=eff * r, min_tasks=mn) if eff >= 8 else dflt
    except Exception:
        return dflt


def batch_cap_public(e):
    rec = STATE_ROOT / 'epochs' / str(e) / 'BATCH-CAP.json'
    if not rec.exists():
        return None
    r = rj(rec); p = r.get('payload', r) if isinstance(r, dict) else {}
    eff = p.get('effective') or {}
    ceiling = p.get('signed_ceiling_max_tasks') or 125
    nxt = staged_batch_cap(ceiling)
    return dict(epoch=e, max_tasks_per_miner=eff.get('max_tasks_per_miner'), rollouts_per_task=eff.get('rollouts_per_task'),
                max_rollouts_per_miner=eff.get('max_rollouts_per_miner'), min_tasks=eff.get('min_tasks'),
                legacy_client_max_tasks=p.get('legacy_max_tasks'), signed_ceiling_max_tasks=ceiling,
                signed_upload_slots=p.get('signed_upload_slots'), next_seal=nxt,
                rule=('Tunable. A K=1 turn trains at most max_tasks_per_miner tasks x rollouts_per_task rollouts of the turn miner\'s batch '
                      '(over-cap batches are trimmed deterministically in submission order; never rejected), at least min_tasks tasks. '
                      'v8.2.3u clients (extended mode, requested by default, active from the next opening) may upload up to the signed ceiling; '
                      'v8.2.2 clients and --no-upgrade stay at legacy_client_max_tasks (9). The validator can change the cap without a new release, '
                      'between 8 and the signed ceiling; next_seal is the value the next seal will use. This block shows the live value.'),
                client='https://affine.io/releases/mimo-source-v8.2.3u/README.txt')


def build():
    return train_params_public(free_file(_build()))


def _build():
    cfg, cfg_sha = lifecycle_config()
    tc = cfg['trainer_config']
    try:
        lt = last_turn_training()
    except Exception:
        lt = {}
    lt_agg = lt.get('loss_aggregation') or {}
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
    sgd = 'sgd_lr' in hp
    k1f = rj(K1_FLAG) if K1_FLAG.exists() else {}
    saf = rj(SCORE_ALL_FLAG) if SCORE_ALL_FLAG.exists() else {}
    score_all = saf.get('enabled') is True
    score_all_from = saf.get('from_epoch', 169)
    k1_on = k1f.get('enabled') is True and type(k1f.get('from_epoch')) is int and e >= k1f['from_epoch']
    k1q = rj(K1_QUEUE) if K1_QUEUE.exists() else {}
    sup = last = None                                 # v8.3.3i support renorm (set below when live)
    if sgd and hp.get('sgd_mode') == 'fresh-adam':   # t751u: trainer host v8.3.3b, one AdamW step from zero state
        trc = hp.get('sgd_tensor_rel_cap')
        opt = dict(name='AdamW', implementation='fresh-adam-one-step-from-zero-state-v833b (bit-exact with torch.optim.AdamW(foreach=False), one step, zero moments)',
                   learning_rate=hp['sgd_lr'], learning_rate_rule='fixed by the validator (1e-6 x restart scale 0.5); miner-chosen LR is not open yet',
                   betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01, weight_decay_policy='applied to >= 2-D non-bias tensors only',
                   tensor_relative_cap=trc, global_update_norm_cap=hp['sgd_update_norm_cap'],
                   max_gradient_norm=t.get('max_gradient_norm'), steps_per_epoch=1, optimizer_state_persisted=False,
                   replay=('g = the fp32 gradient of the loss below over the turn batch (packed varlen micro-batches on 8 GPUs, fp32 per-rank accumulators '
                           'SUM-reduced over NCCL), clipped to global L2 norm max_gradient_norm. '
                           'Per tensor (fp32 from the bf16 head W_t): d_t = learning_rate * (g_t / (|g_t| + eps) + weight_decay * W_t * [W_t >= 2-D and not a bias]); '
                           'r_t = min(1, tensor_relative_cap * (1 - 1e-6) * ||W_t||_2 / ||d_t||_2) (fp64 norms); s = min(1, global_update_norm_cap / ||r*d||_2) '
                           '(not binding); W_t_new = bf16_round_nearest_even(fp32(W_t) - fp32(s * r_t) * d_t). With zero moments and bias correction '
                           'the betas cancel. Exactly one step per turn, nothing carried between turns, so W_new is determined by the published '
                           'head, the published batch (https://affine.io/k1/turns/<epoch>/turn.json) and these numbers up to GPU arithmetic: the '
                           'kernels (fla 0.6.0 Triton, flash attention) and the NCCL reduction order are not bit-deterministic, so an independent '
                           'replay matches closely but is not guaranteed bit-exact. Replay tool and our measured closeness: '
                           'https://affine.io/k1/replay/README.md'))
        sched = dict(kind='constant', learning_rate=hp['sgd_lr'], restart_rule='none (every turn is one step from zero optimizer state)',
                     drift_cap_rule='per-tensor relative cap tensor_relative_cap (same 2e-5 as the pooled AdamW drift cap)')
        if hp.get('sgd_direction') == 'normgrad':   # v8.3.3i (RA/20261031-renorm-step-live): direction step + support renorm, fp32 master kept
            opt.update(name='direction step (norm of the capped fresh-AdamW step, direction of the gradient)',
                       implementation='fresh-adamw-norm-gradient-direction-fp32-head-master-v833i (trainer host v8.3.3i)',
                       direction='normgrad', fp32_master=hp.get('sgd_fp32_master') is True, step_multiplier=1.0,
                       replay=('g = the fp32 gradient of the loss below over the turn batch (packed varlen micro-batches on 8 GPUs, fp32 per-rank '
                               'accumulators SUM-reduced over NCCL), clipped to global L2 norm max_gradient_norm. M_t = the head fp32 master of tensor t '
                               '(published on every GREEN, https://affine.io/k1/fp32-master/index.json). The step NORM is exactly the old capped fresh-AdamW '
                               'step: d_t = learning_rate * (g_t / (|g_t| + eps) + weight_decay * M_t * [M_t >= 2-D and not a bias]); '
                               'r_t = min(1, tensor_relative_cap * (1 - 1e-6) * ||M_t||_2 / ||d_t||_2) (fp64 norms); s = min(1, global_update_norm_cap / '
                               'sqrt(sum_t r_t^2 ||d_t||^2)) (not binding). Only the DIRECTION changes, from ~sign(g_t) to g_t / ||g_t|| per tensor: '
                               'Delta_t = fp32(s * r_t * ||d_t||_2 / ||g_t||_2) * g_t (0 if ||g_t|| = 0); M_C = M_t - Delta_t; candidate bf16 = '
                               'bf16_round_nearest_even(M_C). So each tensor moves by s * r_t * ||d_t|| (about 2e-5 of its norm on the ~242 rel-capped '
                               'tensors), sqrt(sum ||Delta_t||^2) = raw_update_norm (~0.022) and weight decay only sets the norm budget. Exactly one '
                               'step per turn, no moments carried. Replay: https://affine.io/k1/replay/README.md'))
            sup = hp.get('support_renorm') if isinstance(hp.get('support_renorm'), dict) else None
            last = lt.get('support_renorm') if lt.get('sgd_direction') == 'normgrad' else None
    elif sgd:
        mode = hp.get('sgd_mode', 'plain'); freeze = hp.get('sgd_freeze') or []
        dt = ('d_t = learning_rate * g_t' if mode == 'plain' else
              'd_t = learning_rate * ||W_t||_2 * g_t / ||g_t||_2 (per-tensor normalized: every updated tensor moves by the same relative '
              'size learning_rate before the cap)')
        opt = dict(name='SGD', implementation='sgd-stateless-bf16-replayable-v833', learning_rate=hp['sgd_lr'],
                   update_norm_cap=hp['sgd_update_norm_cap'], mode=mode, frozen_tensor_name_substrings=freeze,
                   momentum=0.0, weight_decay=0.0,
                   max_gradient_norm=t.get('max_gradient_norm'), steps_per_epoch=1, optimizer_state_persisted=False,
                   replay=('g = the fp32 gradient of the loss below over the turn batch, all-reduced over ranks, then clipped to global '
                           'L2 norm max_gradient_norm. A tensor whose name contains any frozen_tensor_name_substrings is not updated. For '
                           'every other tensor W_t (bf16): ' + dt + '; s = min(1, update_norm_cap / ||d||_2) over all updated tensors '
                           '(norms in float64); W_t_new = bf16_round_nearest_even(float32(W_t) - float32(s * coefficient_t) * g_t), '
                           'computed from the bf16 weights and rounded once. Exactly one step per turn, no optimizer state, no momentum, '
                           'no weight decay, so W_new is a pure function of the published head, the published batch and these numbers.'))
        sched = dict(kind='constant', learning_rate=hp['sgd_lr'], update_norm_cap=hp['sgd_update_norm_cap'],
                     restart_rule='none (stateless SGD: no restarts)', drift_cap_rule='none (replaced by the global update-norm cap)')
    else:
        opt = dict(name='AdamW', implementation='torch.optim.AdamW-cpu-fp32-masters',
                       learning_rate=t.get('learning_rate'), betas=[t.get('beta1'), t.get('beta2')], eps=t.get('adam_epsilon'),
                       weight_decay=t.get('weight_decay'), weight_decay_policy=t.get('weight_decay_policy'),
                       max_gradient_norm=t.get('max_gradient_norm'), steps_per_epoch=t.get('ppo_epochs'),
                       optimizer_state_persisted=False)
        sched = dict(kind='constant', learning_rate=t.get('learning_rate'),
                      restart_lr_scale=hp['restart_lr_scale'], restart_lr_warm_steps=hp['restart_lr_warm_steps'],
                      restart_rule='after a cold optimizer restart (moments not persisted, e.g. rollback or discarded candidate) '
                                   'the LR is scaled by restart_lr_scale for restart_lr_warm_steps optimizer steps',
                      drift_cap_rel=hp['drift_cap_rel'],
                      drift_cap_rule='the per-step relative parameter update is capped at drift_cap_rel')
    k1 = None
    if k1f.get('enabled') is True:
        k1 = dict(enabled=True, active_this_epoch=k1_on, from_epoch=k1f.get('from_epoch'),
                  rule=(('K=1 per-hotkey turns (Jacob t712u 2026-10-27; one turn per hotkey, permanently, t761u 2026-10-28, as teutonic.ai): '
                         'each epoch trains ONE hotkey\'s batch. Each HOTKEY gets exactly one turn, permanently (pass or fail), on any UID or registration '
                         'block. Turn order = registration block, then uid (UID 0 excluded); hotkeys that already had a turn (registrations_served / '
                         'served_hotkeys) leave the queue. At the seal the first unserved hotkey with a ready batch (>= min_tasks distinct admitted tasks) '
                         'takes the turn; others are skipped, not charged, and keep their place (no waiting). To try again, register a NEW hotkey. '
                         'Do not deregister a paying hotkey: pay follows the GREEN UID while its earning hotkey stays registered there. queue_next = '
                         'unserved hotkeys that had a ready batch at the last seal (forecast; the seal decides).' + (' ' + 'v8.2.9s (Jacob 2026-10-29): a REGISTRATION (uid, registration block) that has had a turn also never gets another, whatever hotkey is on it now: swap_hotkey keeps the uid and registration block, so a swapped-in hotkey on a served registration stays out of the queue. A genuine re-registration (new registration block) with a never-served hotkey is eligible.' if k1q.get('queue_rule') == 'hotkey-registration-once' else '')) if k1q.get('queue_rule') in ('hotkey-once', 'hotkey-registration-once') else
                        ('K=1 per-registration turns (Jacob t712u 2026-10-27; register-to-submit queue t758u 2026-10-28, like teutonic.ai one '
                         'submission per hotkey): each epoch trains ONE registration. A registration = (uid, hotkey, registration block) on the '
                         'metagraph and gets ONE turn, ever (pass or fail). Turn order = registration block, then uid (UID 0 excluded); '
                         'registrations that already had a turn leave the queue. At the seal the first unserved registration with a ready batch '
                         '(>= min_tasks distinct admitted tasks) takes the turn; others are skipped, not charged, and keep their place (no waiting). '
                         'To submit again, register again (new hotkey, or the same hotkey re-registered at a new block). queue_next = unserved '
                         'registrations that had a ready batch at the last seal (forecast; the seal decides).') if k1q.get('queue_rule') == 'registration-once' else
                        ('K=1 per-UID turns (Jacob t712u 2026-10-27): each epoch trains ONE UID. Turn order = fewest turns taken by this '
                        'uid+hotkey, then earliest registration block, then uid (UID 0 excluded). A turn counts pass or fail; a UID with '
                        'no ready batch (< min_tasks distinct admitted tasks) at the seal is skipped, not charged, and the next UID takes '
                        'the turn (no waiting). Re-registering with a new hotkey resets the count.')) + (' The turn batch = that UID\'s admitted '
                        'groups only, deterministically trimmed in submission order (received_at, slot) to max_rollouts. No TOPLOC, '
                        'score or reward checks on miner data and no bans. One stateless step (optimizer below) -> candidate -> promotion gate. '
                        'GREEN credit: 1/N to each of the last up to 10 distinct GREEN UIDs; RED/skipped earn nothing; '
                        'unearned weight burns to UID 0.'),
                  caps=dict(max_rollouts=k1f.get('max_rollouts', 1000), min_tasks=k1f.get('min_tasks', 8),
                            rollouts_per_task=[k1f.get('min_rollouts_per_task', 2), k1f.get('max_rollouts_per_task', 64)],
                            per_rollout_max_tokens='<= the gather (harness) max_output_tokens of the env',
                            current_protocol_note=('8 rollouts per task; per-miner task cap: see training.batch_cap (tunable, 125 x 8 = 1,000 '
                                                   'for v8.2.3u extended clients, 9 for v8.2.2 clients)') if batch_cap_public(e) else
                                                  ('this protocol still collects group_size rollouts per task and at most max_batches '
                                                   'tasks per miner per epoch (see the signed manifest), so a turn is <= max_batches x group_size rollouts')),
                  queue_next=(k1q.get('next') or [])[:8], queue_length=k1q.get('queue_length'),
                  queue_grants=k1q.get('grants') or [], queue_grants_rule=k1q.get('grants_rule'),
                  queue_rule=k1q.get('queue_rule', 'fewest-turns'), queue_rule_from_epoch=k1q.get('queue_rule_from_epoch'),
                  registrations_unserved=k1q.get('registrations_unserved'), registrations_served=k1q.get('served'),
                  hotkeys_unserved=k1q.get('hotkeys_unserved'), served_hotkeys=k1q.get('served_hotkeys'),
                  queue_readiness_epoch=k1q.get('readiness_epoch'),
                  queue_chain_block=k1q.get('chain_block'), probe_sample_size=flag.get('sample_size'),
                  probe=dict(samples_per_task=(flag.get('k1_probe') or {}).get('k', 1), seeded_common_random_numbers=(flag.get('k1_probe') or {}).get('seeded') is True,
                             from_epoch=(flag.get('k1_probe') or {}).get('from_epoch'),
                             rule='head and candidate scored on the same private held-out tasks (from E296 the turn batch is not excluded), each task sampled '
                                  'samples_per_task times; both sides use the same per-request sampling seeds (seed = sha256(plan_seed:env:index:j), one '
                                  'private plan seed per turn shared by head and candidate)'),
                  start_head=k1f.get('start_head'), learning_rate_choice='fixed by the validator (miner-chosen LR: later)',
                  turns_and_credit='https://affine.io/k1.json',
                  turn_batch_published=('after the gate decision the trained batch is public at https://affine.io/k1/turns/<epoch>/turn.json: '
                                        'the selected rows of the signed filtered dispatch (task index, native grades, document sha256) and the '
                                        'miner documents byte-exact under /k1/turns/<epoch>/docs/; task indices also in https://affine.io/k1.json '
                                        'turns[].task_indices. The R2 dispatch object itself is not publicly readable.'),
                  transparency=TRANSPARENCY)
    return dict(
        version=VERSION,
        epoch=m['epoch'], epoch_index=e,
        head=dict(checkpoint=cur['checkpoint']['id'], optimizer_step=cur.get('optimizer_step')),
        optimizer=opt,
        support_renorm=(dict(settings=sup, trainer='v8.3.3i', last_turn=last,
                             rule=('PG term uses log q of the sampled token renormalized over the trainer own sampler support (top_k candidates of '
                                   'its fp32 logits / temperature, then top_p on their softmax with cumsum(p) - p < top_p; the first is always kept; '
                                   'support computed without gradient). A sampled token outside that support is dropped from the PG term only (it '
                                   'leaves both numerator and denominator of that rollout PG token mean) and still counts in the k3 KL term and '
                                   'derail detection (full-vocab log p). Files are never rejected for it (format-only validation): the per-turn '
                                   'outside-support token fraction is published here (last_turn) and in turn.json training.support_renorm.'),
                             code='https://affine.io/k1/replay/host/v833i/v8211_batched.py support_logq (Allan catoneone/sn120-k1-fixes@230b111 patch/support_renorm.diff)')
                        if sup else None),
        schedule=sched,
        k1=k1,
        loss=dict(objective=('GRPO: vanilla dual-clip PPO, %s%s, k3 KL to reference, overlong soft penalty'
                             % (lt_agg.get('mode') or t.get('loss_agg_mode'),
                                (' x %s' % lt_agg.get('loss_scale')) if lt_agg.get('mode') == 'seq-mean-token-mean' else '')),
                  loss_agg_mode=lt_agg.get('loss_agg_mode') or t.get('loss_agg_mode'),
                  loss_scale=lt_agg.get('loss_scale'),
                  loss_scale_factor=(None if lt_agg.get('mode') == 'seq-mean-token-mean' else t.get('loss_scale_factor')),
                  signed_config_loss_agg_mode=t.get('loss_agg_mode'),
                  aggregation_source=('trainer host engine (loss_agg_mode, loss_scale) as reported by the trainer for epoch %s; the signed '
                                      'trainer_config still names %s, which the host overrides' % (lt.get('epoch'), t.get('loss_agg_mode'))
                                      if lt_agg.get('mode') == 'seq-mean-token-mean' else 'signed trainer_config'),
                  formula=('L = loss_scale * (1/N) * sum_i (1/(K_i + 1e-8)) * sum_{t<=K_i} [ pg_{i,t} + kl_coefficient * k3_{i,t} ], N = active '
                           'rollouts in the turn (all of them: unresolved rollouts stay in with reward 0), K_i = rollout i\'s kept response tokens. '
                           'pg = vanilla dual-clip PPO with OLD = current.detach() (one PPO epoch, ratio 1), so its gradient is -A_i * grad log pi. '
                           'k3 = exp(ref - cur) - (ref - cur) - 1 against the frozen kl_reference_checkpoint (ref - cur clamped to [-20, 20], k3 to '
                           '[-10, 10]). A_i = r_i - mean(r over the task\'s group), no std normalisation. r = 1[correct] - overlong_penalty * '
                           'clip((len - overlong_soft_start) / (overlong_max_tokens - overlong_soft_start), 0, 1). Derail mask: only for rollouts '
                           'stopped at max_tokens with negative advantage, K_i is cut at the derail index (derail_window / derail_threshold / '
                           'derail_tail; subnet/mimo_grpo/training.py derail_keep). Code: pinned verl core_algos agg_loss / '
                           'compute_policy_loss_vanilla / kl_penalty_forward (subnet/mimo_grpo/upstream.py) + https://affine.io/k1/replay/host/v8211_batched.py')
                  if lt_agg.get('mode') == 'seq-mean-token-mean' else None,
                  miner_logprobs='not used by the trainer: current log-probs are recomputed by the trainer from the head (OLD = current.detach()) '
                                 'and reference log-probs from kl_reference_checkpoint; uploaded logprobs only serve the capture protocol',
                  execution=dict(engine='packed varlen data-parallel (first-fit-decreasing micro-batches, LPT over ranks, fp32 accumulators '
                                        'SUM-reduced over NCCL), bf16 weights, fp32 logits/loss, layer activation checkpointing',
                                 microbatch=lt.get('microbatch'), runtime_versions=lt.get('runtime_versions'),
                                 kernels='fla 0.6.0 chunk_gated_delta_rule + causal_conv1d (Triton, varlen), torch varlen_attn (FlashAttention-2) '
                                         'for full attention; https://affine.io/k1/replay/host/v83_fast_kernels.py',
                                 deterministic=False,
                                 note='not bit-deterministic run to run (Triton/FA kernels, NCCL reduce order); see https://affine.io/k1/replay/README.md'),
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
                                  'upload deadline; the first max_groups are trained, the rest are excluded (training-task-budget)')
                   if not k1_on else ('K=1: only the turn UID\'s admitted groups are trained (see k1); the pooled max_groups budget '
                                      'and group-size rule do not apply to the turn batch'),
        sampling=dict(temperature=t.get('temperature'), top_k=t.get('top_k'), top_p=t.get('top_p'), harness=harness),
        collection_seconds=(m['deadline'] - m['start']) if isinstance(m.get('deadline'), (int, float)) else None,
        promotion_gate=dict(enabled=flag.get('enabled') is True, sample_size=flag.get('sample_size'),
                            sample_pool=flag.get('pool'), env=flag.get('env'), z_min=flag.get('z_min'),
                            cap_hit_max=flag.get('cap_max'), min_tasks=flag.get('min_tasks'),
                            z_inclusive=flag.get('z_inclusive') is True, z_rule_from_epoch=flag.get('z_rule_from_epoch'),
                            red_credit=('scored' if score_all else flag.get('red_credit')),
                            scores_update_on=('pass_and_fail' if score_all else 'pass_only'),
                            red_scored_from_epoch=(score_all_from if score_all else None),
                            first_gated_epoch=flag.get('first_gated_epoch'),
                            rule='(E168-E270 rule; from E271 see the K=1 text) candidate promoted only if paired z (candidate vs head, k=1) > z_min (>= z_min if z_inclusive) and cap-hit rate < cap_hit_max on a '
                                 'private sample minus the epoch batch (from epoch 168: z_min 0, strict = net must be positive, candidate '
                                 'pass rate > head pass rate); RED = discarded, next epoch mines from the same head, ' +
                                 ((('scores update on pass and fail: from epoch %s a RED epoch is scored like a GREEN one (head probe vs the '
                                    'discarded candidate probed on that epoch\'s own batch tasks); promotion unchanged' % score_all_from)
                                   if score_all else 'RED epoch miners get zero flip credit') if not k1_on else
                                  'K=1, held-out gate (live from E271): GREEN only if the weighted per-task gain over the head is positive and weighted_z > 1.00 (strict), estimated cap-hit < 0.20 and >= 100 kept tasks; head and candidate scored on the same tasks of a private held-out probe set (4 seeded samples per task, shared seeds); from E296 the turn batch is not excluded; a round is revealed only if a GREEN closed it, and then only the GREEN turn; RED turns are never revealed (https://affine.io/gate.json rule); GREEN -> promoted and the turn UID earns; '
                                  'RED = head kept, the turn earns nothing; no fixed-reference guard'),
                            decisions='https://affine.io/gate/index.json', heldout=heldout_public(),
                            reveal=('from e222 one STANDING private sample is reused across turns (the head probe is reused after RED; a promoted '
                                    'candidate\'s own probe becomes the new head probe); its commitment is published at first use and it is rotated '
                                    'to a fresh os.urandom(32) seed every 24 h or whenever the head must be re-probed. The seed, sampled task ids '
                                    'and per-task head/candidate grades are published at https://affine.io/gate/reveal/<epoch>.json for GREEN epochs '
                                    'only (index: https://affine.io/gate/reveal/index.json); RED epochs and their probe results are never published. '
                                    'From E271 the held-out gate reveals a round only if a GREEN closed it, and then only the GREEN turn')),
        batch_cap=batch_cap_public(e),
        sources=dict(lifecycle_config_sha256=cfg_sha, trainer_host=host.get('trainer_host'),
                     trainer_host_operator_config_sha256=host.get('operator_config_sha256')),
        generated_at=time.time(),
    )


FF_FLAG = RA / '20261023-toploc-credit-train-gate/staged/FF.json'


def free_file(d):
    """free-file protocol (Jacob pt898u 2026-10-30, RA/20261030-free-file): the miner uploads one batch file; the turn trains it
    with the miner's rewards as given. Rewrites the protocol-dependent texts; numbers stay generated from the live configs."""
    f = rj(FF_FLAG) if FF_FLAG.exists() else {}
    if f.get('enabled') is not True or d.get('epoch_index', 0) < f.get('from_epoch', 1 << 30):
        return d
    cut = f.get('legacy_cutoff_unix')
    d['batch_file'] = 'https://affine.io/mining.json -> batch_file (format, example, upload link, rules)'
    k1 = d.get('k1')
    if k1:
        k1['rule'] = ('One turn per hotkey and per registration (UID + registration block), permanently, pass or fail. Queue order = '
                      'registration block, then UID (UID 0 excluded); served hotkeys and registrations leave the queue. At each turn the '
                      'validator takes the first queue member whose batch file (batch_file.object) is present: it copies the file, checks the '
                      'format only (batch_file.rules) and trains it. A missing or invalid file = skipped, not charged, keeps its place; the next '
                      'member is tried. Until the old-format cutoff a member without a valid file but with a ready old-format batch (old client) '
                      'is trained the old way. One stateless step from the published head (optimizer below) -> candidate -> promotion gate. '
                      'GREEN credit: 1/N to each of the last up to 10 distinct GREEN UIDs; RED/skipped earn nothing; unearned weight burns to UID 0.')
        k1['caps'] = dict(max_prompts=(d.get('batch_cap') or {}).get('max_tasks_per_miner'),
                          max_rollouts=(d.get('batch_cap') or {}).get('max_rollouts_per_miner'),
                          rollouts_per_prompt='any number >= 1', max_output_tokens=8192, max_context_tokens=12288,
                          live_value='training.batch_cap')
        k1['turn_batch_published'] = ('after the gate decision: https://affine.io/k1/turns/<epoch>/turn.json. GREEN: the exact miner file '
                                      '(/k1/turns/<epoch>/batch.json) and the signed dispatch. RED: decision, counts and file sha256 only; '
                                      'the file stays private.')
        k1['probe']['rule'] = ('head and candidate on the round\'s private held-out set (the turn batch is NOT excluded: every kept task counts), '
                               'each task sampled samples_per_task times with shared per-request sampling seeds')
        k1['old_format_cutoff_unix'] = cut
    d['batch'] = ('free-file: the turn trains the turn miner\'s file only: one GRPO group per prompt, its rollouts as uploaded (any number), '
                  'up to training.batch_cap (whole prompts beyond the cap are dropped in file order)')
    bc = d.get('batch_cap')
    if bc:
        bc['rule'] = ('Tunable by the validator without a release (staged BATCH-CAP.json); applies at the next turn. A turn trains at most '
                      'max_tasks_per_miner prompts and max_rollouts_per_miner rollouts of the miner file; prompts beyond either cap are dropped '
                      'whole, in file order. No minimum for batch files (min_tasks applies to old-format batches until the cutoff only).')
        bc.pop('client', None)
        bc['rollouts_per_task'] = 'any (>= 1 per prompt); %s is the old-format group size' % bc.get('rollouts_per_task')
    lo = d.get('loss') or {}
    if lo.get('formula'):
        lo['formula'] = lo['formula'].replace(
            "A_i = r_i - mean(r over the task's group), no std normalisation. r = 1[correct] - overlong_penalty * "
            "clip((len - overlong_soft_start) / (overlong_max_tokens - overlong_soft_start), 0, 1).",
            "A_i = r_i - mean(r over the prompt's rollouts), no std normalisation (a prompt with one rollout: A_i = r_i). r = the miner's "
            "reward for the rollout - overlong_penalty * clip((len - overlong_soft_start) / (overlong_max_tokens - overlong_soft_start), 0, 1) "
            "(old-format batches until the cutoff: 1[correct] instead of the miner reward). Rollouts at exactly 8192 output tokens are "
            "length-capped (no EOS) and treated as unresolved for the derail mask.")
    lo['miner_logprobs'] = 'none: batch files carry text and rewards only; the trainer recomputes every log-prob from the head'
    pg = d.get('promotion_gate') or {}
    pg['rule'] = ('candidate promoted only if it beats the head on the private held-out set (heldout): z = gain/SE > bar_z (SE: heldout.se) and cap-hit < '
                  'cap_hit_max; the turn batch is NOT excluded from the probe (since the free-file cutover). GREEN -> next head, the turn earns; '
                  'RED -> head kept, candidate discarded, the turn earns nothing.')
    ho = pg.get('heldout')
    if isinstance(ho, dict) and isinstance(ho.get('rule'), str):
        ho['rule'] = ho['rule'].replace('on the kept tasks minus its own batch tasks', 'on all kept tasks (its batch is not excluded)')
    return d


TP_DIR = RA / '20261031-ff-train-params'
TP_FLAG = TP_DIR / 'state/FF-TRAIN.json'
TP_FIELDS = dict(lr=dict(default=5e-07, min=1.25e-07, max=1e-06), kl=dict(default=0.02, min=0.005, max=0.1))
TP_MATH = ('m = lr / 5e-7. Per tensor t (fp32 master W_t): d_t = 5e-7 * (g_t / (|g_t| + 1e-8) + 0.01 * W_t * [W_t >= 2-D and not a bias]); '
           'c_t = 2e-5 * (1 - 1e-6) * ||W_t||; n0_t = min(||d_t||, c_t); n_t = min(m * n0_t, c_t); s = min(1, 1.0 / ||n||); '
           'W_t_new = W_t - s * n_t * g_t / ||g_t|| (normgrad direction, trainer v8.3.3j). m = 1 is exactly the v8.3.3i step. lr below the '
           'default scales every tensor by m; above it only tensors under the per-tensor cap grow, up to c_t (hard ceiling). '
           'kl replaces kl_coefficient in L = loss_scale * (1/N) * sum_i (1/K_i) * sum_t [pg + kl * k3] for that turn only.')


def tp_recent(start, n=20):
    """GREEN turns only: the effective lr / kl the trainer reported (TRAINING-RESULT training.train_params)."""
    out = []
    if type(start) is not int: return out
    for d in sorted((p for p in (STATE_ROOT / 'epochs').iterdir() if p.name.isdigit() and int(p.name) >= start), key=lambda p: -int(p.name)):
        dec = rj(GATE_STATE / 'epochs' / d.name / 'DECISION.json') if (GATE_STATE / 'epochs' / d.name / 'DECISION.json').exists() else {}
        if dec.get('decision') != 'green' or not (d / 'TRAINING-RESULT.json').exists(): continue
        r = rj(d / 'TRAINING-RESULT.json'); r = r.get('payload', r)
        tp = ((r.get('report') or {}).get('training') or {}).get('train_params') or {}
        if 'lr' not in tp: continue
        out.append(dict(epoch=int(d.name), lr=tp.get('lr'), kl=tp.get('kl'), given=tp.get('given'), defaults_used=tp.get('defaults_used'),
                        turn='https://affine.io/k1/turns/%s/turn.json' % d.name))
        if len(out) >= n: break
    return out


def train_params_public(d):
    """Jacob 2026-10-31 07:08 CR (pt902u), RA/20261031-ff-train-params: optional miner lr / kl in affine-batch-v2 files."""
    f = rj(TP_FLAG) if TP_FLAG.exists() else {}
    if f.get('enabled') is not True:
        return d
    d['miner_train_params'] = dict(
        format='affine-batch-v2', field='train', optional=True, fields=TP_FIELDS, from_epoch=f.get('from_epoch'),
        rule=('An affine-batch-v2 file may carry "train": {"lr": x, "kl": y} (both optional; omitted = default). Bounds inclusive; a value '
              'outside them or not a number makes the file invalid (skipped, not charged). Never clamped. affine-batch-v1 files train '
              'with the defaults. Everything else (per-tensor cap, global cap, gradient clip, overlong penalty, head, one step, batch cap, '
              'queue, gate) is fixed.'),
        math=TP_MATH, example={'version': 'affine-batch-v2', 'hotkey': '5YourHotkeySs58Address', 'train': {'lr': 8e-07, 'kl': 0.01}, 'groups': ['...']},
        published='per turn, GREEN only: k1/turns/<epoch>/turn.json training.train_params and recent_green_turns below; RED stays private',
        recent_green_turns=tp_recent(f.get('from_epoch')))
    o = d.get('optimizer')
    if isinstance(o, dict):
        o['learning_rate_rule'] = ('default %s; the turn miner may set lr in [1.25e-7, 1e-6] in an affine-batch-v2 file '
                                   '(training.miner_train_params)' % o.get('learning_rate'))
    k = d.get('k1')
    if isinstance(k, dict):
        k['learning_rate_choice'] = 'miner-chosen per turn via affine-batch-v2 train.lr / train.kl (training.miner_train_params)'
    lo = d.get('loss')
    if isinstance(lo, dict):
        lo['kl_coefficient_rule'] = ('default %s; the turn miner may set kl in [0.005, 0.1] in an affine-batch-v2 file '
                                     '(training.miner_train_params)' % lo.get('kl_coefficient'))
    return d


def write():
    d = build()
    s = json.dumps(d, sort_keys=True)
    # guard the seed_hex KEY, not the word: the public held-out rule text mentions seed_hex (feed was stale from 18:54Z 2026-10-30)
    for bad in ('"seed_hex":', 'SAMPLE', 'private', 'r2.cloudflarestorage', 'X-Amz'):
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
