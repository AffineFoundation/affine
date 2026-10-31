"""ROOT-admitted v8.3 trainer host launcher over the unchanged worker closure.

v8.3.3h = v8.3.3g + free-file turns (Jacob pt898u 2026-10-30, RA/20261030-free-file): a signed 'sn120-ff-dispatch-v1'
  selection is prepared by ff_trainer.prepare (miner rewards used as given, any number of rollouts per group);
  v8211_batched._layout uses each rollout's reward and its group's own size (identical numbers for native groups).

v8.3.3j = v8.3.3i + per-turn training params (lr, kl) from a free-file affine-batch-v2 dispatch (Jacob 2026-10-31 pt902u,
  RA/20261031-ff-train-params). No engine key; a dispatch without 'train' or with the defaults runs the exact v8.3.3i step.
v8.3.3i = v8.3.3h + optional engine.sgd_direction normgrad + engine.support_renorm (Allan; Jacob 2026-10-31 07:00 CR).
v8.3.3f = v8.3.3 + optional engine.sgd_fp32_master (persistent fp32 head master, see FP32_KEYS). Jacob t831u 2026-10-30.
v8.3.3g = v8.3.3f + execution-only speedups (Jacob pt841u/t847u 2026-10-30, RA/20261030-trainer-speed): same update math and bytes;
  parallel+memoized checkpoint sha256 (v83_dp.install_fast_hash), helper ranks preload KL ref + actor at start, fp32 digests
  48 threads and bf16/fp32 digests concurrent, verified digests reused, fp32 step: finite check on the GPU copy and one direct
  D2H copy into the master, torch CPU threads 32 inside the step, fp32-master disk save deferred until the model is published.

v8.3 = v8.2.11 + a data-parallel packed execution of the SAME update (v83_dp.py), on W GPUs of one node:
the worker process is rank 0 (it owns the actor, the CPUAdamW optimizer, staging and publication exactly as in
v8.2.11); W-1 helper processes (one per extra GPU, started here, children of the worker) hold the frozen KL
reference and a per-job bit-exact copy of the actor and run their share of the packed micro-batches through the same
v8211_batched._reference_pass/_gradient_pass. FP32 gradients are SUM-reduced to rank 0, then ONE clip and ONE
CPUAdamW step (unchanged). Loss, normalization, rewards, cap policy, group budget, KL, lr: unchanged.
The fast training kernels (fla chunk_gated_delta_rule / causal_conv1d varlen, torch varlen flash attention) are
declared in engine.training_kernels and installed per layer instance for training forwards only (v83_kernels_policy).
Authorized by Jacob (owner) 2026-10-25 (pt619u; parity gate "at least as close to fp32 truth as prod" approved 15:13 CR).

v8.2.11 notes:

v8.2.11 = v8.2.10 + engine.loss_agg_mode ('config' | 'seq-mean-token-mean') + engine.loss_scale, authorized by
Jacob 2026-10-24 (t586u). v8211_batched.py replaces v825_batched.py (same update; aggregation selectable; adds
report.epoch_metrics). math_levels.json maps the math env task index to its MATH level for the cap-rate metric;
mimo_training_worker.prepare_signed_dispatch is wrapped only to record task_id -> (env_id, index) from the
authenticated receipts (return value unchanged). 'config' = byte-for-byte the v8.2.10 loss.

v8.2.10 = v8.2.5 + engine.optimizer_persistence ('none' | 'synchronous'), authorized by Jacob 2026-10-22
("saving the optimizer state should not be important can we remove this requirement"). 'none': the epoch
report never waits on writing the optimizer state (v8210_optimizer_offpath.py): the model is staged and
published exactly as before; the optimizer generation is a small local marker; the live optimizer stays in
process memory; cold recovery rebuilds fresh AdamW moments with the step kept when no state was persisted.
'synchronous' = exact v8.2.5 staging/recovery (pre-signed in-place rollback variant).

v8.2.5 = v8.2.4 + engine.drop_unresolved (bool): when true, unresolved rollouts are dropped from the
loss (see v825_batched.py). Rollback: drop_unresolved=false config or the v8.2.4 owned worker.

The frozen worker source closure (source_files / source_guard / registry) is not
modified on disk. This launcher verifies a ROOT admission over its own exact
configuration (host file digests + original-method source digests), installs the
v8.2.4 in-memory overrides, then runs `subnet.distributed_worker` unchanged:
  * padded micro-batched actor-as-OLD update (token_budget 16384, max_rows 64)
  * CPUAdamW.step -> pipelined (same torch AdamW, bitwise identical; pinned overlap)
  * fast staging (inline hashes seed the transaction memo) + model preupload overlap
  * FastR2ModelPublisher (v2 receipts: streamed sha256 + part Content-MD5 + HEAD)
  * v2-aware receipt checks (transaction._receipt, validate_publication, validate_result)
Host modules load under non-`subnet.` names so source_guard's loaded-module check holds.
"""
import sys
sys.dont_write_bytecode = True
import argparse, ast, hashlib, importlib.util, inspect, json, os, runpy, textwrap
from pathlib import Path

VERSION = 'affine-mimo-trainer-host-v833'
TRITON_CACHE = '/workspace/affine-mimo-trainer-prod/triton-cache-v831'
SCOPE = 'sn120-mimo-trainer-host-overrides-v8.3.3'
P = Path(__file__).resolve().parent
HOST_FILES = ('mimo_host_trainer_v83.py', 'ff_trainer.py', 'v83_dp.py', 'v83_fast_kernels.py', 'v83_kernels_policy.py', 'math_levels.json', 'v8210_optimizer_offpath.py', 'v8211_batched.py', 'v824_optimizer_fast.py', 'v824_staging_fast.py',
              'v824_publication_fast.py', 'patched/mimo_grpo.transaction.py', 'patched/mimo_grpo.publication.py',
              'patched/mimo_grpo.service.py', 'patched/mimo_training_worker.py')
SITES = ('mimo_grpo.transaction._receipt', 'mimo_grpo.transaction.Transaction.run',
         'mimo_grpo.publication.validate_publication', 'mimo_grpo.service.execute_transaction',
         'mimo_training_worker.validate_result', 'mimo_grpo.optimizer.CPUAdamW.step',
         'mimo_grpo.publication.R2ModelPublisher.__init__', 'mimo_grpo.recovery.restore_transaction_state',
         'mimo_training_worker.prepare_signed_dispatch')
PERSISTENCE = ('none', 'synchronous')
ENGINE_KEYS = {'batched_engine', 'token_budget', 'max_rows', 'fast_stage', 'stage_workers',
               'part_concurrency', 'shard_concurrency', 'optimizer_step_threads', 'drop_unresolved',
               'optimizer_persistence', 'loss_agg_mode', 'loss_scale',
               'dp_world', 'dp_port', 'dp_token_budget', 'dp_max_rows', 'training_kernels'}
# v8.3.2 optimizer drift control (Jacob 2026-10-26 13:06 CR). OPTIONAL engine keys; absent / None / 0 / 1.0 = OFF (v8.3.1 behaviour).
DRIFT_KEYS = {'drift_cap_rel', 'restart_lr_warm_steps', 'restart_lr_scale'}
DRIFT_FP32_MARGIN = 1e-3
DRIFT_OFF = dict(drift_cap_rel=None, restart_lr_warm_steps=0, restart_lr_scale=1.0)


def drift_settings(engine):
    """-> (drift_cap_rel or None, restart_lr_warm_steps, restart_lr_scale), validated; defaults are OFF."""
    cap = engine.get('drift_cap_rel', DRIFT_OFF['drift_cap_rel'])
    warm = engine.get('restart_lr_warm_steps', DRIFT_OFF['restart_lr_warm_steps'])
    scale = engine.get('restart_lr_scale', DRIFT_OFF['restart_lr_scale'])
    if cap is not None and not (type(cap) is float and 0 < cap <= 1):
        raise ValueError('drift_cap_rel must be None or a float in (0, 1]')
    if not (type(warm) is int and 0 <= warm <= 100):
        raise ValueError('restart_lr_warm_steps must be an int in [0, 100]')
    if not (type(scale) is float and 0 < scale <= 1):
        raise ValueError('restart_lr_scale must be a float in (0, 1]')
    return cap, warm, scale


def _fresh_moments(opt):
    """True if every AdamW state is absent or has all-zero moments (new optimizer or fresh-moments cold recovery)."""
    for name, _ in opt.parameters:
        state = opt.optimizer.state.get(opt.masters[name])
        if state and (bool(state['exp_avg'].any()) or bool(state['exp_avg_sq'].any())):
            return False
    return True


def controlled_step(opt, gradients, inner, cap, warm, scale, emit=None):
    """v8.3.2: ONE AdamW step (inner = the unchanged v8.3.1 pipelined step) with
    (a) restart LR: lr x scale for the first `warm` steps after fresh optimizer moments (detected once per optimizer object);
    (b) drift cap: after the step, per tensor scale the fp32 master update so ||dW_t||/||W_t|| <= cap, then one global factor so
        ||dW||/||W|| <= cap over all tensors; the bf16 parameters are re-projected from the capped masters.
    Runs only on the worker (rank 0, which owns the single CPU AdamW); helper ranks receive these exact bf16 weights by
    dist.broadcast(src=0) at the next job, so all ranks are identical by construction. Adam moments are not modified."""
    import math, time, torch
    t0 = time.monotonic()
    if not getattr(opt, '_v832_checked', False):
        opt._v832_fresh_at_start = _fresh_moments(opt)
        opt._v832_since_fresh = 0 if opt._v832_fresh_at_start else None
        opt._v832_checked = True
    since = opt._v832_since_fresh
    lr_factor = scale if (since is not None and since < warm) else 1.0
    old = {name: opt.masters[name].detach().clone() for name, _ in opt.parameters} if cap else None
    groups = opt.optimizer.param_groups
    saved = [g['lr'] for g in groups]
    try:
        if lr_factor != 1.0:
            for g in groups:
                g['lr'] = g['lr'] * lr_factor
        result = inner(opt, gradients)
    finally:
        for g, lr in zip(groups, saved):
            g['lr'] = lr
    info = dict(version='v832-drift-control-v1', optimizer_step=opt.global_step, fresh_moments_at_start=opt._v832_fresh_at_start,
                steps_since_fresh_before=since, lr_factor=lr_factor, effective_lr=saved[0] * lr_factor if saved else None,
                restart_lr_warm_steps=warm, restart_lr_scale=scale, drift_cap_rel=cap)
    if cap:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(max(previous_threads, getattr(opt, 'step_threads', 1)))
        try:
            result, info = _drift_cap(opt, result, info, old, cap)
        finally:
            torch.set_num_threads(previous_threads)
    if since is not None:
        opt._v832_since_fresh = since + 1
    info['seconds'] = time.monotonic() - t0
    result = dict(result, drift_control=info)
    (emit or (lambda row: print(json.dumps(row, sort_keys=True), flush=True)))(dict(component='trainer-host-v832', event='drift-control', **info))
    return result


def _drift_cap(opt, result, info, old, cap):
    import math, torch
    if True:
        f64 = torch.float64
        rows = []
        for name, _ in opt.parameters:
            o = old[name]; m = opt.masters[name].detach()
            dn = float(torch.linalg.vector_norm(m - o, dtype=f64)); wn = float(torch.linalg.vector_norm(o, dtype=f64))
            rows.append((name, dn, wn))
        W = math.sqrt(math.fsum(wn * wn for _, _, wn in rows))
        pre = math.sqrt(math.fsum(dn * dn for _, dn, _ in rows)) / W
        target = cap * (1.0 - DRIFT_FP32_MARGIN)   # fp32 rounding of old + f*(new-old) must not push a tensor past `cap`
        per = [min(1.0, target * wn / dn) if dn > 0 and wn > 0 else 1.0 for _, dn, wn in rows]
        mid = math.sqrt(math.fsum((s * dn) ** 2 for s, (_, dn, _) in zip(per, rows))) / W
        gscale = min(1.0, target / mid) if mid > 0 else 1.0
        # (sum dn_t^2 s_t^2)/(sum W_t^2) <= cap^2 once every tensor is capped, so the global factor only binds through fp32 rounding;
        # it is still applied and re-measured so the logged global bound is a measured fact, not an assumption.
        params = dict(opt.parameters)
        def apply(name, f):
            o = old[name]; m = opt.masters[name]; parameter = params[name]
            m.copy_(o + (opt._v832_new[name] - o) * f)
            parameter.copy_(m.detach().to(device=parameter.device, dtype=parameter.dtype))
            return float(torch.linalg.vector_norm(m.detach() - o, dtype=f64))
        factors = {}; post = {}; capped = 0
        with torch.no_grad():
            opt._v832_new = {}
            for s, (name, dn, wn) in zip(per, rows):
                f = s * gscale; factors[name] = f; post[name] = dn
                capped += s < 1.0
                if f < 1.0:
                    opt._v832_new[name] = opt.masters[name].detach().clone()
                    post[name] = apply(name, f)
                    for _ in range(4):                     # exact per-tensor bound despite fp32 rounding
                        if wn == 0 or post[name] / wn <= cap:
                            break
                        factors[name] *= cap / (post[name] / wn) * (1.0 - DRIFT_FP32_MARGIN)
                        post[name] = apply(name, factors[name])
            post_rel = math.sqrt(math.fsum(v * v for v in post.values())) / W
            repass = 1.0
            if post_rel > cap:
                repass = cap / post_rel * (1.0 - DRIFT_FP32_MARGIN)
                for name, dn, wn in rows:
                    if name not in opt._v832_new:
                        opt._v832_new[name] = opt.masters[name].detach().clone()
                    factors[name] *= repass
                    post[name] = apply(name, factors[name])
                post_rel = math.sqrt(math.fsum(v * v for v in post.values())) / W
            del opt._v832_new
            changed = 0
            for name, parameter in opt.parameters:
                changed += int((opt.masters[name].detach().to(parameter.dtype) != old[name].to(parameter.dtype)).sum())
                old[name] = None
        max_post = max((post[n] / wn for n, _, wn in rows if wn > 0), default=0.0)
        max_pre = max((dn / wn for _, dn, wn in rows if wn > 0), default=0.0)
        if post_rel > cap or max_post > cap:
            opt.failed = True
            raise ValueError('v8.3.2 drift cap bound not met')
        if not all(torch.isfinite(opt.masters[n]).all() for n, _ in opt.parameters):
            opt.failed = True
            raise ValueError('v8.3.2 drift cap produced a non-finite master')
        info.update(pre_cap_rel=pre, post_cap_rel=post_rel, pre_cap_dW=pre * W, post_cap_dW=post_rel * W, weight_norm=W,
                    tensors=len(rows), tensors_capped=int(capped), tensor_cap_scale_min=min(per), global_scale=gscale * repass,
                    min_factor=min(factors.values()), cap_fired=bool(capped or gscale * repass < 1.0),
                    max_tensor_rel_pre=max_pre, max_tensor_rel_post=max_post,
                    zero_norm_tensors=sum(1 for _, _, wn in rows if wn == 0),
                    projected_changed_elements_precap=result.get('projected_changed_elements'))
        result = dict(result, projected_changed_elements=changed)
    return result, info
# v8.3.3 K=1 plain SGD (Jacob t712u 2026-10-27 11:52 CR). OPTIONAL engine keys; absent = OFF (exact v8.3.2 behaviour).
# ON: ONE stateless SGD step per job, no Adam/momentum/weight decay, no drift cap, no restart LR. The update is computed from the
# bf16 parameters themselves (never from carried fp32 masters), so anyone holding the published bf16 head and the same gradient
# can replay it exactly:   g = clip(grad, max_gradient_norm)  (the unchanged trainer clip, fp32 all-reduced gradient)
#                          s = min(1, sgd_update_norm_cap / (sgd_lr * ||g||_2))      (python float64, ||g|| in float64)
#                          W_new = bf16_rne( fp32(W_bf16) - fp32(sgd_lr * s) * g )   (per tensor, fp32 arithmetic)
SGD_KEYS = {'sgd_lr', 'sgd_update_norm_cap', 'sgd_mode', 'sgd_freeze', 'sgd_tensor_rel_cap'}
SGD_MODES = ('plain', 'per-tensor-normalized', 'fresh-adam')
# sgd_mode 'fresh-adam' (Jacob 2026-10-27 14:40 CR): ONE torch.optim.AdamW step from zero state (no carried moments).
#   With m0 = v0 = 0 and bias correction, m_hat = g and v_hat = g^2 exactly, so (betas cancel)
#   d_t = lr * ( g_t / (|g_t| + FRESH_ADAM_EPS) + FRESH_ADAM_WD * W_t * [decayed_t] )     elementwise, ~ lr * sign(g)
#   decayed_t = tensor.dim() >= 2 and name does not end with 'bias'  (mcore-bias-and-1d-excluded-v1)
FRESH_ADAM_EPS = 1e-8
FRESH_ADAM_WD = 0.01
# sgd_tensor_rel_cap (optional, fresh-adam only): per tensor r_t = min(1, sgd_tensor_rel_cap * (1 - 1e-6) * ||W_t|| / ||d_t||)
#   (fp64 norms; today's prod drift cap 2e-5), applied before the global cap: W' = bf16_rne( fp32(W) - fp32(s * r_t) * d_t )
# sgd_mode 'plain':                 d_t = lr * g_t
# sgd_mode 'per-tensor-normalized': d_t = lr * ||W_t||_2 * g_t / ||g_t||_2     (lr is then the relative step per tensor)
# sgd_freeze: list of substrings; a tensor whose name contains any of them is not updated (d_t = 0).
# global cap: s = min(1, sgd_update_norm_cap / ||d||_2) over the non-frozen tensors (fp64 norms);
# W_new = bf16_rne( fp32(W_bf16) - fp32(s) * fp32(d_t) )   (computed from the bf16 weights, rounded once)


def sgd_settings(engine):
    """-> (lr, cap, mode, freeze) or (None, None, None, ()) when OFF. Validated; SGD excludes the v8.3.2 drift keys."""
    if not (set(engine) & SGD_KEYS):
        return None, None, None, ()
    lr, cap = engine.get('sgd_lr'), engine.get('sgd_update_norm_cap')
    mode, freeze = engine.get('sgd_mode', 'plain'), engine.get('sgd_freeze', [])
    if not (type(lr) is float and 0 < lr <= 1000.0):
        raise ValueError('sgd_lr must be a float in (0, 1000]')
    if not (type(cap) is float and 0 < cap <= 1e6):
        raise ValueError('sgd_update_norm_cap must be a float in (0, 1e6]')
    if mode not in SGD_MODES:
        raise ValueError('sgd_mode')
    if not (type(freeze) is list and all(type(x) is str and x for x in freeze) and len(freeze) <= 64):
        raise ValueError('sgd_freeze must be a list of non-empty strings')
    if set(engine) & DRIFT_KEYS:
        raise ValueError('SGD (v8.3.3) excludes the AdamW drift-control keys')
    trc = engine.get('sgd_tensor_rel_cap')
    if trc is not None and not (type(trc) is float and 0 < trc <= 1.0 and mode == 'fresh-adam'):
        raise ValueError('sgd_tensor_rel_cap must be a float in (0, 1] and needs sgd_mode fresh-adam')
    return lr, cap, mode, tuple(freeze)


def _fresh_adam_delta(name, w32, g, lr):
    import torch
    d = g / (g.abs() + torch.tensor(FRESH_ADAM_EPS, dtype=torch.float32, device=g.device))
    if w32.dim() >= 2 and not name.endswith('bias'):
        d = d + torch.tensor(FRESH_ADAM_WD, dtype=torch.float32, device=g.device) * w32
    return torch.tensor(lr, dtype=torch.float32, device=g.device) * d


# v8.3.3f fp32 head master (Jacob t821u-t823u/t829u/t831u 2026-10-30; Allan catoneone/sn120-k1-fixes `--arm fp32`). OPTIONAL engine key
# sgd_fp32_master (bool, fresh-adam only); absent / false = exact v8.3.3 behaviour (w32 = fp32(bf16 W)).
# ON: the step still starts from the CURRENT PUBLISHED HEAD, but w32 is the head's persistent fp32 master M_H instead of fp32(bf16 W_H):
#   d_t, r_t, s exactly as v8.3.3 fresh-adam with W_t := M_H,t (cap norms ||M_H,t|| fp64; fresh zero moments every step, nothing persisted)
#   M_C = M_H - fp32(s * r_t) * d_t   (fp32)        W_C = bf16_rne(M_C)   (the published bf16 candidate, unchanged format)
# M_H is used only if bf16_rne(M_H) == W_H bit-for-bit (self-verifying): in process memory after the previous step, else from the local
# store <FP32_STORE>/<sha256(bf16 W_H)>/ (written after every step), else bootstrap M_H = fp32(W_H) (exactly the v8.3.3 step).
# RED candidates' masters are never used (their bf16 never becomes the head) and are garbage-collected; on GREEN M_C becomes M_H.
# Report: optimizer.sgd.fp32_master = {source, input/output bf16 sha256, input/output master digest, timings}; master digest =
# sha256(canonical {tensor name: sha256(little-endian fp32 bytes)}). The host publishes M_C of every GREEN with that digest.
FP32_KEYS = {'sgd_fp32_master'}
FP32_STORE = '/workspace/affine-mimo-trainer-prod/fp32-master'
FP32_KEEP = 2            # newest complete entries kept besides the current head's entry (36 GB each)
FP32_SHARD_BYTES = 4 * 1024 ** 3
FP32_VERSION = 'v833f-fp32-head-master-v1'
FP32_SAVE_DEFER_MAX = 600   # v8.3.3g: seconds the background master save waits for the model publication to finish
import threading as _threading
_PUBLISHED = _threading.Event(); _PUBLISHED.set()


def fp32_settings(engine):
    on = engine.get('sgd_fp32_master', False)
    if type(on) is not bool:
        raise ValueError('sgd_fp32_master must be a bool')
    if on and engine.get('sgd_mode') != 'fresh-adam':
        raise ValueError('sgd_fp32_master needs sgd_mode fresh-adam')
    return on


# v8.3.3i (RA/20261031-renorm-step-live; Jacob 2026-10-31 07:00 CR; Allan catoneone/sn120-k1-fixes@230b111). OPTIONAL engine keys;
# absent = exact v8.3.3h behaviour.
#  sgd_direction 'normgrad' (fresh-adam + sgd_fp32_master only): every tensor keeps the norm of today's capped fresh-AdamW step
#     (d_t, r_t, s exactly as fresh-adam, computed from the fp32 head master M_H), only the direction changes from ~sign(g_t) to g_t/||g_t||:
#     Delta_t = fp32(s * r_t * ||d_t|| / ||g_t||) * g_t   (fp64 norms; Delta_t = 0 if ||g_t|| = 0);  M_C = M_H - Delta_t;  W_C = bf16_rne(M_C)
#     so ||Delta_t|| = s * r_t * ||d_t|| and sqrt(sum_t ||Delta_t||^2) = raw_update_norm (multiplier 1x, lr and caps unchanged).
#  support_renorm {top_k, top_p, temperature}: sets v8211_batched.SUPPORT_RENORM (Allan patch/support_renorm.diff, applied as written): the PG
#     term uses log q over the trainer's own top-k -> top-p support; sampled tokens outside it leave the PG token mean but stay in KL and
#     derail. Asserted equal to the job's signed sampling settings (temperature/top_k/top_p) at every update. Per-turn summary:
#     training.support_renorm (outside-support token fraction) in the trainer report.
DIR_KEYS = {'sgd_direction', 'support_renorm'}
SGD_DIRECTIONS = ('sign', 'normgrad')
SUPPORT_KEYS = ('top_k', 'top_p', 'temperature')


def direction_settings(engine):
    d = engine.get('sgd_direction', 'sign')
    if d not in SGD_DIRECTIONS:
        raise ValueError('sgd_direction must be sign or normgrad')
    if d == 'normgrad' and not (engine.get('sgd_mode') == 'fresh-adam' and engine.get('sgd_fp32_master') is True):
        raise ValueError('sgd_direction normgrad needs sgd_mode fresh-adam and sgd_fp32_master true')
    return d


def support_settings(engine):
    s = engine.get('support_renorm')
    if s is None:
        return None
    if not (type(s) is dict and set(s) == set(SUPPORT_KEYS) and type(s['top_k']) is int and 1 <= s['top_k'] <= 1024
            and type(s['top_p']) is float and 0 < s['top_p'] <= 1.0 and type(s['temperature']) is float and s['temperature'] > 0):
        raise ValueError('support_renorm must be {top_k: int, top_p: float, temperature: float}')
    return dict(top_k=s['top_k'], top_p=s['top_p'], temperature=s['temperature'])


def support_summary(trajectories, support):
    """Per-turn outside-support statistics from the per-rollout trainer details (support_outside_tokens over kept loss tokens)."""
    rows = [t for t in trajectories if t.get('support_outside_tokens') is not None]
    out = sum(int(t['support_outside_tokens']) for t in rows); tok = sum(int(t.get('loss_tokens') or 0) for t in rows)
    fr = [(t['support_outside_tokens'] / t['loss_tokens']) if t.get('loss_tokens') else 0.0 for t in rows]
    inside = tok - out
    lq = [t['support_logq_mean'] * (t['loss_tokens'] - t['support_outside_tokens']) for t in rows
          if t.get('support_logq_mean') is not None and t.get('loss_tokens')]
    return dict(version='v833i-support-renorm-v1', settings=dict(support), rollouts=len(rows), loss_tokens=tok, outside_tokens=out,
                outside_fraction=(out / tok) if tok else 0.0, rollouts_with_outside=sum(1 for t in rows if t['support_outside_tokens']),
                max_rollout_outside_fraction=max(fr) if fr else 0.0,
                rollouts_outside_over_0p5pct=sum(1 for f in fr if f > 0.005),
                inside_logq_mean=(sum(lq) / inside) if inside > 0 else None,
                rule='sampled tokens outside the replayed top_k -> top_p support are dropped from the PG token mean only (still in KL and derail); files are never rejected for it')


# v8.3.3j (RA/20261031-ff-train-params; Jacob 2026-10-31 07:08 CR pt902u): per-turn training params from a signed free-file dispatch
# ('train': {lr, kl, given, defaults}; ff_trainer.train_params). Defaults (lr == engine.sgd_lr, kl == config.kl_coefficient) -> the exact
# v8.3.3i code path (same config object, lr multiplier 1.0, same expressions).
#  kl: config.kl_coefficient := the turn's kl for this one update (all DP ranks: the job header carries the config); the loss is
#      L = loss_scale * (1/N) * sum_i (1/K_i) * sum_t [ pg_it + kl * k3_it ]; the signed config document is reported unchanged and
#      the applied value is in training.train_params.
#  lr: m = lr / engine.sgd_lr. With sgd_direction normgrad + sgd_tensor_rel_cap (required when m != 1):
#      n0_t = r_t * ||d_t||                                   the v8.3.3i per-tensor step norm (d_t, r_t at sgd_lr, before the global cap)
#      c_t  = sgd_tensor_rel_cap * (1 - 1e-6) * ||M_H,t||     the unchanged per-tensor ceiling (none if ||M_H,t|| = 0)
#      n_t  = min(m * n0_t, c_t)                              m <= 1: exactly m * n0_t; m > 1: grows only below the ceiling
#      s    = min(1, sgd_update_norm_cap / sqrt(sum_t n_t^2)) the unchanged global cap
#      Delta_t = fp32(s * n_t / ||g_t||) * g_t (fp64 scalars);  M_C = M_H - Delta_t;  W_C = bf16_rne(M_C)
#  training.optimizer.sgd keeps the default-lr (v8.3.3i) statistics; the turn's own step norms are in training.train_params.step.
TURN = dict(lr_multiplier=1.0, last=None)      # set only by train_update_v833j around one update
TURN_PARAMS = dict(params=None)                # set by prepare_signed_dispatch for the job being prepared


def _sha_tensor(t):
    import torch
    t = t.detach().contiguous()
    if t.device.type != 'cpu':
        t = t.cpu()
    raw = t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32) if t.dtype == torch.float32 else t
    return hashlib.sha256(memoryview(raw.numpy()).cast('B')).hexdigest()


def _digest_map(named, threads=48):
    from concurrent.futures import ThreadPoolExecutor
    order = sorted(named, key=lambda nt: -nt[1].numel())      # largest first (load balance); digest is order-free
    with ThreadPoolExecutor(threads) as ex:
        shas = list(ex.map(lambda nt: _sha_tensor(nt[1]), order))
    per = dict(zip([n for n, _ in order], shas))
    return hashlib.sha256(json.dumps(per, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), per


def _fp32_store_entries():
    root = Path(FP32_STORE)
    if not root.is_dir():
        return []
    rows = []
    for d in root.iterdir():
        if d.is_dir() and not d.name.startswith('.') and (d / 'meta.json').is_file():
            rows.append((d.stat().st_mtime, d))
    return sorted(rows)


def _fp32_gc(protect, emit):
    import shutil, time as _t
    try:
        entries = _fp32_store_entries()
        keep = {d.name for _, d in entries[-FP32_KEEP:]} | {p for p in protect if p}
        for _, d in entries:
            pin = d / 'PIN'
            if d.name in keep or (pin.exists() and _t.time() - pin.stat().st_mtime < 3 * 3600):
                continue
            shutil.rmtree(d, ignore_errors=True)
        for d in Path(FP32_STORE).iterdir():
            if d.name.startswith('.tmp-') and _t.time() - d.stat().st_mtime > 3600:
                shutil.rmtree(d, ignore_errors=True)
    except Exception as x:
        emit(dict(component='trainer-host-v833f', event='fp32-master-gc-error', error=repr(x)[:300]))


def _fp32_save(opt, entry, meta, protect, emit):
    """Background: write opt.masters (fp32, CPU) to <store>/<entry>/ as safetensors shards + meta.json (written last), then GC."""
    import os as _os, time as _t, torch
    from safetensors.torch import save_file
    t0 = _t.monotonic()
    root = Path(FP32_STORE); root.mkdir(parents=True, exist_ok=True, mode=0o700)
    final = root / entry
    try:
        if (final / 'meta.json').is_file():
            emit(dict(component='trainer-host-v833f', event='fp32-master-saved', entry=entry, present=True)); return
        tmp = root / ('.tmp-%s-%d' % (entry[:16], _os.getpid()))
        if tmp.exists():
            import shutil; shutil.rmtree(tmp)
        tmp.mkdir(mode=0o700)
        shards, cur, size = [], [], 0
        for name, _ in opt.parameters:
            b = opt.masters[name].numel() * 4
            if cur and size + b > FP32_SHARD_BYTES:
                shards.append(cur); cur, size = [], 0
            cur.append(name); size += b
        if cur:
            shards.append(cur)
        files, tensors = {}, {}
        for i, names in enumerate(shards):
            fn = 'fp32-master-%05d-of-%05d.safetensors' % (i + 1, len(shards))
            save_file({n: opt.masters[n].detach() for n in names}, str(tmp / fn), metadata=dict(format='pt', fp32_master=FP32_VERSION))
            h = hashlib.sha256()
            with open(tmp / fn, 'rb') as f:
                _os.fsync(f.fileno())
                for blk in iter(lambda: f.read(1 << 24), b''):
                    h.update(blk)
            files[fn] = dict(sha256=h.hexdigest(), bytes=(tmp / fn).stat().st_size)
            for n in names:
                tensors[n] = fn
        meta = dict(meta, files=files, tensor_file=tensors, shard_bytes=FP32_SHARD_BYTES, saved_at=_t.time(), save_seconds=_t.monotonic() - t0)
        (tmp / 'meta.json').write_text(json.dumps(meta, sort_keys=True, indent=1))
        _os.replace(tmp, final)
        emit(dict(component='trainer-host-v833f', event='fp32-master-saved', entry=entry, seconds=_t.monotonic() - t0,
                  bytes=sum(v['bytes'] for v in files.values())))
    except Exception as x:
        emit(dict(component='trainer-host-v833f', event='fp32-master-save-error', entry=entry, error=repr(x)[:300]))
    finally:
        _fp32_gc(protect, emit)


def _fp32_load(opt, live, entry, emit):
    """-> True if <store>/<entry> was loaded into opt.masters and verified (digest + bf16 projection == live parameters)."""
    import torch
    from safetensors.torch import load_file
    d = Path(FP32_STORE) / entry
    try:
        meta = json.loads((d / 'meta.json').read_text())
        if meta.get('bf16_sha256') != entry or meta.get('version') != FP32_VERSION:
            raise ValueError('meta')
        names = {n for n, _ in opt.parameters}
        if set(meta['tensor_file']) != names:
            raise ValueError('inventory')
        for fn in sorted(set(meta['tensor_file'].values())):
            data = load_file(str(d / fn), device='cpu')
            for n, v in data.items():
                m = opt.masters[n]
                if v.dtype != torch.float32 or v.shape != m.shape:
                    raise ValueError('tensor ' + n)
                with torch.no_grad():
                    m.copy_(v)
            del data
        dig, _ = _digest_map([(n, opt.masters[n]) for n, _ in opt.parameters])
        if dig != meta['master_digest']:
            raise ValueError('master digest')
        with torch.no_grad():
            for n, p in opt.parameters:
                if not torch.equal(opt.masters[n].detach().to(device=p.device).to(p.dtype), p.detach()):
                    raise ValueError('projection ' + n)
        return meta
    except Exception as x:
        emit(dict(component='trainer-host-v833f', event='fp32-master-load-rejected', entry=entry, error=repr(x)[:300]))
        return None


def _fp32_prepare(opt, live, emit):
    """Make opt.masters the verified fp32 master of the current (published head) parameters. -> info dict."""
    import time as _t, torch
    t0 = _t.monotonic()
    lock = getattr(opt, '_v833f_lock', None)
    if lock is None:
        import threading
        lock = opt._v833f_lock = threading.Lock()
    with lock:      # a previous background save of these masters must finish before they are reused / overwritten
        pass
    bf_in, _ = _digest_map(list(opt.parameters))
    t_fp = _t.monotonic() - t0
    source = None
    if getattr(opt, '_v833f_bf16', None) == bf_in:
        ok = True
        with torch.no_grad():
            for n, p in opt.parameters:
                if not torch.equal(opt.masters[n].detach().to(device=p.device).to(p.dtype), p.detach()):
                    ok = False; break
        if ok:
            source = 'memory'
    meta = None
    if source is None and (Path(FP32_STORE) / bf_in / 'meta.json').is_file():
        meta = _fp32_load(opt, live, bf_in, emit)
        if meta is not None:
            source = 'disk'
    if source is None:
        threads = torch.get_num_threads(); torch.set_num_threads(max(threads, 32))
        try:
            with torch.no_grad():
                for n, p in opt.parameters:
                    opt.masters[n].copy_(p.detach())      # bf16 -> fp32 is exact; same values as .to(cpu, float32)
        finally:
            torch.set_num_threads(threads)
        source = 'bootstrap-upcast'
    if source == 'memory' and getattr(opt, '_v833g_dig', None):
        dig_in = opt._v833g_dig                           # digest of these exact masters from the previous finish
    elif source == 'disk' and meta is not None:
        dig_in = meta['master_digest']                    # just recomputed and verified by _fp32_load
    else:
        dig_in, _ = _digest_map([(n, opt.masters[n]) for n, _ in opt.parameters])
    return dict(version=FP32_VERSION, source=source, input_bf16_sha256=bf_in, input_master_digest=dig_in,
                input_store_entry=bf_in if source == 'disk' else None, fingerprint_seconds=t_fp, prepare_seconds=_t.monotonic() - t0)


def _fp32_finish(opt, info, emit):
    """After the step: hash the candidate (bf16 + master), remember it in memory, save to the store in the background."""
    import threading, time as _t
    from concurrent.futures import ThreadPoolExecutor
    t0 = _t.monotonic()
    opt._v833g_dig = None
    with ThreadPoolExecutor(2) as ex:
        f_bf = ex.submit(_digest_map, list(opt.parameters))
        f_m = ex.submit(_digest_map, [(n, opt.masters[n]) for n, _ in opt.parameters])
        bf_out, _ = f_bf.result(); dig_out, per = f_m.result()
    opt._v833f_bf16 = bf_out; opt._v833g_dig = dig_out
    info.update(output_bf16_sha256=bf_out, output_master_digest=dig_out, output_store_entry=bf_out,
                master_tensors=len(per), master_bytes=sum(opt.masters[n].numel() * 4 for n, _ in opt.parameters),
                finish_seconds=_t.monotonic() - t0, store=FP32_STORE)
    meta = dict(version=FP32_VERSION, bf16_sha256=bf_out, master_digest=dig_out, tensor_sha256=per,
                input_bf16_sha256=info['input_bf16_sha256'], input_master_digest=info['input_master_digest'],
                source=info['source'], optimizer_step=opt.global_step)
    lock = opt._v833f_lock
    lock_ready = threading.Event()
    _PUBLISHED.clear()
    def runner():
        lock.acquire()
        lock_ready.set()
        try:
            _PUBLISHED.wait(FP32_SAVE_DEFER_MAX)        # v8.3.3g: keep the 36 GB write off model staging/publication IO
            _fp32_save(opt, bf_out, meta, (info['input_bf16_sha256'], bf_out), emit)
        finally:
            lock.release()
    threading.Thread(target=runner, name='v833f-fp32-save', daemon=False).start()
    lock_ready.wait(30)
    return info


def fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit, t0, tensor_rel_cap=None, fp32_master=False, direction='sign'):
    """v8.3.3 fresh-adam: W' = bf16_rne( fp32(W) - fp32(s) * d ),  s = min(1, cap / ||d||_2) (fp64 norm over live tensors).
    v8.3.3f (fp32_master=True): fp32(W) is replaced by the head's verified fp32 master M_H and M_C = M_H - fp32(s) * d is kept."""
    import math, time, torch
    emit_ = emit or (lambda row: print(json.dumps(row, sort_keys=True), flush=True))
    _threads = torch.get_num_threads(); torch.set_num_threads(max(_threads, 32))
    try:
        return _fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit_, t0, tensor_rel_cap, fp32_master, direction)
    finally:
        torch.set_num_threads(_threads)


def _fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit_, t0, tensor_rel_cap, fp32_master, direction='sign'):
    import math, time, torch
    normgrad = direction == 'normgrad'
    if normgrad and not fp32_master:
        raise ValueError('normgrad direction needs the fp32 master')
    fpi = _fp32_prepare(opt, live, emit_) if fp32_master else None
    if fp32_master:
        W32 = lambda name, parameter: opt.masters[name].detach().to(device=parameter.device)
    else:
        W32 = lambda name, parameter: parameter.detach().float()
    with torch.no_grad():
        dn2 = []; r = {}; qd = {}; gn2 = {}; wnd = {}
        for name, parameter in live:
            g = gradients[name].detach().to(device=parameter.device); w32 = W32(name, parameter)
            q = float(_fresh_adam_delta(name, w32, g, lr).double().pow(2).sum())
            if normgrad:
                qd[name] = q; gn2[name] = float(g.double().pow(2).sum())
            rt = 1.0
            if tensor_rel_cap is not None and q > 0:
                wn = math.sqrt(float(w32.double().pow(2).sum())); wnd[name] = wn
                rt = min(1.0, tensor_rel_cap * (1 - 1e-6) * wn / math.sqrt(q)) if wn > 0 else 1.0
            r[name] = rt; dn2.append(q * rt * rt)
            del w32, g
    raw = math.sqrt(math.fsum(dn2)); scale = min(1.0, cap / raw) if raw > 0 else 1.0
    lr_mult = TURN['lr_multiplier']; coef_j = None          # v8.3.3j
    n0 = {name: r[name] * math.sqrt(qd[name]) for name in r} if normgrad else {}
    capped_frac = (math.sqrt(math.fsum(n0[k] ** 2 for k in n0 if r[k] < 1.0)) / raw) if (normgrad and raw > 0) else None
    if lr_mult != 1.0:
        if not (normgrad and tensor_rel_cap is not None and fp32_master):
            raise ValueError('v8.3.3j: a turn lr needs the normgrad fp32-master step with the per-tensor cap')
        nj = {}
        for name in r:
            mn = lr_mult * n0[name]
            nj[name] = min(mn, tensor_rel_cap * (1 - 1e-6) * wnd[name]) if wnd.get(name, 0.0) > 0 else mn
        raw_j = math.sqrt(math.fsum(v * v for v in nj.values())); scale_j = min(1.0, cap / raw_j) if raw_j > 0 else 1.0
        coef_j = {name: (scale_j * nj[name] / math.sqrt(gn2[name])) if gn2[name] > 0 else 0.0 for name in r}
        TURN['last'] = dict(path='scaled', lr_multiplier=lr_mult, raw_update_norm=raw_j, cap_scale=scale_j, cap_fired=scale_j < 1.0,
                            tensors_at_ceiling=sum(1 for k in nj if nj[k] < lr_mult * n0[k]), tensors=len(nj),
                            raw_update_norm_default=raw, cap_scale_default=scale, tensors_rel_capped_default=sum(v < 1.0 for v in r.values()),
                            capped_norm_fraction_default=capped_frac)
    else:
        TURN['last'] = dict(path='v833i-exact', lr_multiplier=1.0, raw_update_norm=raw, cap_scale=scale, cap_fired=scale < 1.0,
                            tensors=len(r), raw_update_norm_default=raw, cap_scale_default=scale,
                            tensors_rel_capped_default=sum(v < 1.0 for v in r.values()), capped_norm_fraction_default=capped_frac)
    changed = 0; dsq = []; wsq = []; msq = []
    try:
        with torch.no_grad():
            for name, parameter in live:
                g = gradients[name].detach().to(device=parameter.device)
                w32 = W32(name, parameter)
                if not fp32_master:      # exact v8.3.3 expression
                    new = (w32 - torch.tensor(scale * r[name], dtype=torch.float32, device=parameter.device) * _fresh_adam_delta(name, w32, g, lr)).to(parameter.dtype)
                    changed += int((new != parameter).sum())
                    dsq.append(float((new.float() - w32).double().pow(2).sum())); wsq.append(float(w32.double().pow(2).sum()))
                    parameter.copy_(new)
                    opt.masters[name].copy_(new.detach().to(device=opt.masters[name].device, dtype=opt.masters[name].dtype))
                else:                    # same fp32 arithmetic, in place (M_C = M_H - fp32(s*r_t) * d_t elementwise; no extra full-size temp)
                    wsq.append(float(w32.double().pow(2).sum()))
                    if normgrad:         # v8.3.3i: same per-tensor norm s * r_t * ||d_t||, direction g_t / ||g_t||
                        c = ((scale * r[name] * math.sqrt(qd[name]) / math.sqrt(gn2[name])) if gn2[name] > 0 else 0.0) if coef_j is None else coef_j[name]
                        d = g * torch.tensor(c, dtype=torch.float32, device=parameter.device)
                    else:
                        d = _fresh_adam_delta(name, w32, g, lr)
                        d.mul_(torch.tensor(scale * r[name], dtype=torch.float32, device=parameter.device))
                    msq.append(float(d.double().pow(2).sum()))
                    w32.sub_(d); del d
                    new = w32.to(parameter.dtype)
                    changed += int((new != parameter).sum())
                    dsq.append(float((new.float() - parameter.detach().float()).double().pow(2).sum()))
                    parameter.copy_(new)
                    if not torch.isfinite(w32).all():         # same values as the master after the exact copy below
                        raise ValueError('non-finite fresh-adam update: ' + name)
                    opt.masters[name].copy_(w32)               # one direct D2H copy (v8.3.3g)
                if not fp32_master and not torch.isfinite(opt.masters[name]).all():
                    raise ValueError('non-finite fresh-adam update: ' + name)
                del w32, new, g
        opt.global_step += 1
    except BaseException:
        opt.failed = True
        opt._v833f_bf16 = None; opt._v833g_dig = None
        raise
    finally:
        opt.zero_grad()
    W = math.sqrt(math.fsum(wsq)); dW = math.sqrt(math.fsum(dsq))
    info = dict(version='v833-fresh-adam-v1' if not fp32_master else ('v833f-fresh-adam-fp32-master-v1' if not normgrad else 'v833i-normgrad-fp32-master-v1'), optimizer_step=opt.global_step, sgd_lr=lr, sgd_update_norm_cap=cap, sgd_mode='fresh-adam',
                adam_eps=FRESH_ADAM_EPS, weight_decay=FRESH_ADAM_WD, sgd_tensor_rel_cap=tensor_rel_cap,
                tensors_rel_capped=sum(v < 1.0 for v in r.values()), sgd_freeze=list(freeze), frozen_tensors=len(opt.parameters) - len(live),
                sgd_direction=direction, zero_gradient_tensors=(sum(1 for v in gn2.values() if v == 0) if normgrad else None),
                raw_update_norm=raw, cap_scale=scale, cap_fired=scale < 1.0, update_norm_bf16=dW, weight_norm_live=W,
                update_rel_bf16=dW / W if W else None, projected_changed_elements=changed, seconds=time.monotonic() - t0)
    if fp32_master:
        dM = math.sqrt(math.fsum(msq))
        try:
            fpi = _fp32_finish(opt, fpi, emit_)
        except Exception as x:
            opt._v833f_bf16 = None
            fpi = dict(fpi, finish_error=repr(x)[:300])
        info['fp32_master'] = dict(fpi, update_norm_master=dM, update_rel_master=dM / W if W else None)
        if TURN['last'] is not None:
            TURN['last']['update_norm_master'] = dM
        info['seconds'] = time.monotonic() - t0
    emit_(dict(component='trainer-host-v833', event='sgd-step', **info))
    return dict(optimizer_step=opt.global_step, implementation='adamw-fresh-state-bf16-replayable-v833' if not fp32_master else ('adamw-fresh-state-fp32-head-master-v833f' if not normgrad else 'fresh-adamw-norm-gradient-direction-fp32-head-master-v833i'),
                foreach=False, fused=False,
                lr=lr, betas=[0.9, 0.999], eps=FRESH_ADAM_EPS, weight_decay=FRESH_ADAM_WD, state_dtype=None,
                master_dtype='bfloat16-projected' if not fp32_master else 'float32-head-master',
                weight_decay_policy='mcore-bias-and-1d-excluded-v1', gradient_accumulation_dtype='float32', projected_changed_elements=changed,
                momentum=0.0, optimizer_state='none (fresh zero moments every step)', sgd=info)


def sgd_step(opt, gradients, lr, cap, mode='plain', freeze=(), emit=None, tensor_rel_cap=None, fp32_master=False, direction='sign'):
    """v8.3.3: one stateless SGD step (see SGD_KEYS). Runs on rank 0; helper ranks receive these exact bf16 weights by
    dist.broadcast(src=0) at the next job (unchanged v8.3 DP hand-off)."""
    import math, time, torch
    t0 = time.monotonic()
    if opt.failed or set(gradients) != set(opt.masters):
        raise ValueError('healthy optimizer and exact gradient inventory required')
    for name, parameter in opt.parameters:
        g = gradients[name]
        if g.dtype != torch.float32 or g.shape != parameter.shape or not torch.isfinite(g).all():
            raise ValueError('finite FP32 accumulated gradients required: ' + name)
    live = [(n, p) for n, p in opt.parameters if not any(f in n for f in freeze)]
    gsq = {n: math.fsum([float(gradients[n].detach().double().pow(2).sum())]) for n, _ in live}
    gn = math.sqrt(math.fsum(gsq.values()))
    if mode == 'fresh-adam':
        return fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit, t0, tensor_rel_cap, fp32_master, direction)
    if mode == 'plain':
        coef = {n: lr for n, _ in live}
    else:
        coef = {}
        for n, p in live:
            wn = math.sqrt(float(p.detach().double().pow(2).sum())); g_n = math.sqrt(gsq[n])
            coef[n] = (lr * wn / g_n) if g_n > 0 else 0.0
    raw = math.sqrt(math.fsum(coef[n] * coef[n] * gsq[n] for n, _ in live))
    scale = min(1.0, cap / raw) if raw > 0 else 1.0
    changed = 0; dsq = []; wsq = []
    try:
        with torch.no_grad():
            for name, parameter in live:
                g = gradients[name].detach().to(device=parameter.device)
                w32 = parameter.detach().float()
                new = (w32 - torch.tensor(coef[name] * scale, dtype=torch.float32, device=parameter.device) * g).to(parameter.dtype)
                changed += int((new != parameter).sum())
                dsq.append(float((new.float() - w32).double().pow(2).sum())); wsq.append(float(w32.double().pow(2).sum()))
                parameter.copy_(new)
                opt.masters[name].copy_(new.detach().to(device=opt.masters[name].device, dtype=opt.masters[name].dtype))
                if not torch.isfinite(opt.masters[name]).all():
                    raise ValueError('non-finite SGD update: ' + name)
        opt.global_step += 1
    except BaseException:
        opt.failed = True
        raise
    finally:
        opt.zero_grad()
    W = math.sqrt(math.fsum(wsq)); dW = math.sqrt(math.fsum(dsq))
    info = dict(version='v833-sgd-v2', optimizer_step=opt.global_step, sgd_lr=lr, sgd_update_norm_cap=cap, sgd_mode=mode,
                sgd_freeze=list(freeze), frozen_tensors=len(opt.parameters) - len(live), gradient_norm_clipped_live=gn,
                raw_update_norm=raw, cap_scale=scale, cap_fired=scale < 1.0, update_norm_bf16=dW, weight_norm_live=W,
                update_rel_bf16=dW / W if W else None, projected_changed_elements=changed, seconds=time.monotonic() - t0)
    (emit or (lambda row: print(json.dumps(row, sort_keys=True), flush=True)))(dict(component='trainer-host-v833', event='sgd-step', **info))
    return dict(optimizer_step=opt.global_step, implementation='sgd-stateless-bf16-replayable-v833', foreach=False, fused=False,
                lr=lr, betas=None, eps=None, weight_decay=0.0, state_dtype=None, master_dtype='bfloat16-projected',
                weight_decay_policy='none', gradient_accumulation_dtype='float32', projected_changed_elements=changed,
                momentum=0.0, optimizer_state='none', sgd=info)


LOSS_AGG_MODES = ('config', 'seq-mean-token-mean')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method_sha(obj):
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def private_json(path):
    path = Path(path)
    st = path.lstat()
    if not path.is_file() or path.is_symlink() or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise ValueError('private regular operator configuration required')
    return json.loads(path.read_bytes())


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, P / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def patched_def(rel, qualname):
    """Exact source of one def from the admitted patched copy."""
    tree = ast.parse((P / 'patched' / rel).read_text())
    parts = qualname.split('.')
    nodes = tree.body
    for i, part in enumerate(parts):
        hits = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
        if len(hits) != 1:
            raise ValueError('exact patched definition: ' + qualname)
        node = hits[0]; nodes = node.body
    lines = (P / 'patched' / rel).read_text().splitlines(keepends=True)
    return textwrap.dedent(''.join(lines[node.lineno - 1:node.end_lineno]))


def patched_assign(rel, names):
    text = (P / 'patched' / rel).read_text(); out = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', None) in names:
            out.append(ast.get_source_segment(text, node))
    if len(out) != len(names):
        raise ValueError('exact patched constants')
    return '\n'.join(out) + '\n'


def define(module, source, name, cls=None):
    """Compile in a copy of the module globals (methods) or the module itself (functions)."""
    if cls is None:
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        return module.__dict__[name]
    # Methods compile against the live module globals (later global patches stay visible);
    # the temporary top-level name is removed again.
    if name in module.__dict__:
        raise ValueError('method name collides with a module global: ' + name)
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    function = module.__dict__.pop(name)
    original = getattr(cls, name)
    function.__qualname__, function.__module__ = original.__qualname__, original.__module__
    function.__defaults__, function.__kwdefaults__ = original.__defaults__, original.__kwdefaults__
    return function


def checked(config_path):
    config = private_json(config_path)
    need = lambda c, m: (_ for _ in ()).throw(ValueError(m)) if not c else None
    need(set(config) == {'version', 'scope', 'authority', 'source_root', 'host_files', 'original_methods',
                         'engine', 'admission'} and config['version'] == VERSION and config['scope'] == SCOPE,
         'exact v8.3 trainer host configuration')
    need(set(config['host_files']) == set(HOST_FILES) and set(config['original_methods']) == set(SITES)
         and ENGINE_KEYS <= set(config['engine']) <= ENGINE_KEYS | DRIFT_KEYS | SGD_KEYS | FP32_KEYS | DIR_KEYS and type(config['engine']['drop_unresolved']) is bool
         and config['engine']['optimizer_persistence'] in PERSISTENCE and config['engine']['fast_stage'] is True
         and config['engine']['loss_agg_mode'] in LOSS_AGG_MODES and type(config['engine']['loss_scale']) is float
         and 0 < config['engine']['loss_scale'] <= 1024
         and (config['engine']['loss_agg_mode'] != 'config' or config['engine']['loss_scale'] == 1.0)
         and config['engine']['batched_engine'] == 'packed'
         and type(config['engine']['dp_world']) is int and 1 <= config['engine']['dp_world'] <= 8
         and type(config['engine']['dp_port']) is int and 1024 < config['engine']['dp_port'] < 65536
         and type(config['engine']['dp_token_budget']) is int and 8192 <= config['engine']['dp_token_budget'] <= 65536
         and type(config['engine']['dp_max_rows']) is int and 1 <= config['engine']['dp_max_rows'] <= 1024
         and type(config['engine']['training_kernels']) is dict,
         'exact v8.3 host inventory')
    drift_settings(config['engine']); sgd_settings(config['engine']); fp32_settings(config['engine'])
    direction_settings(config['engine']); support_settings(config['engine'])
    for name, digest_ in config['host_files'].items():
        path = P / name
        need(path.is_file() and not path.is_symlink() and file_sha(path) == digest_, 'exact admitted host file ' + name)
    root = Path(config['source_root']).resolve()
    need(root.is_dir() and Path.cwd().resolve() == root, 'worker cwd is the admitted closure root')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from subnet.backend_jobs import signed
    from subnet.storage import canonical, sha
    proposal = {k: v for k, v in config.items() if k != 'admission'}
    expected = dict(version=VERSION, configuration_sha256=sha(canonical(proposal)), source_unchanged=True, scope=SCOPE)
    need(signed(config['admission'], config['authority']) == expected, 'exact ROOT trainer host admission')
    return config


def install(config, start_dp=False):
    import subnet.mimo_grpo.transaction as transaction
    import subnet.mimo_grpo.publication as publication
    import subnet.mimo_grpo.service as service
    import subnet.mimo_grpo.optimizer as optimizer
    import subnet.mimo_grpo.handover as handover
    import subnet.mimo_grpo.recovery as recovery
    import subnet.mimo_training_worker as worker
    import subnet.mimo_grpo as mimo_grpo
    import subnet.mimo_grpo.prepared as prepared
    pins = config['original_methods']
    objects = {
        'mimo_grpo.transaction._receipt': transaction._receipt,
        'mimo_grpo.transaction.Transaction.run': transaction.Transaction.run,
        'mimo_grpo.publication.validate_publication': publication.validate_publication,
        'mimo_grpo.service.execute_transaction': service.execute_transaction,
        'mimo_training_worker.validate_result': worker.validate_result,
        'mimo_grpo.optimizer.CPUAdamW.step': optimizer.CPUAdamW.step,
        'mimo_grpo.publication.R2ModelPublisher.__init__': publication.R2ModelPublisher.__init__,
        'mimo_grpo.recovery.restore_transaction_state': recovery.restore_transaction_state,
        'mimo_training_worker.prepare_signed_dispatch': worker.prepare_signed_dispatch,
    }
    for site, obj in objects.items():
        if method_sha(obj) != pins[site]:
            raise ValueError('exact original trainer method required: ' + site)
    if worker.R2ModelPublisher is not publication.R2ModelPublisher or worker.CPUAdamW is not optimizer.CPUAdamW \
            or worker.execute_transaction is not service.execute_transaction \
            or handover.validate_publication is not publication.validate_publication \
            or recovery._receipt is not transaction._receipt \
            or worker.restore_transaction_state is not recovery.restore_transaction_state \
            or worker.prepare_signed_dispatch is not prepared.prepare_signed_dispatch:
        raise ValueError('exact original trainer bindings required')
    engine = config['engine']
    # prepare (all compiled before any install)
    sources = dict(
        receipt=patched_def('mimo_grpo.transaction.py', '_receipt'),
        run=patched_def('mimo_grpo.transaction.py', 'Transaction.run'),
        receipt_consts=patched_assign('mimo_grpo.transaction.py', {'V1_RECEIPT', 'V2_RECEIPT'}),
        pub_consts=patched_assign('mimo_grpo.publication.py', {'VERSION_V2', 'UPLOAD_VERIFICATION_V2'}),
        verified=patched_def('mimo_grpo.publication.py', 'publication_verified'),
        validate_publication=patched_def('mimo_grpo.publication.py', 'validate_publication'),
        execute=patched_def('mimo_grpo.service.py', 'execute_transaction'),
        validate_result=patched_def('mimo_training_worker.py', 'validate_result'))
    for value in sources.values():
        compile(value, '<v824>', 'exec')
    # imported by the patched service as `v824_batched`; v8.2.5 module registered under that name
    batched = load('v824_batched', 'v8211_batched.py')
    batched.DROP_UNRESOLVED = engine['drop_unresolved']
    batched.LOSS_AGG_MODE = engine['loss_agg_mode']
    batched.LOSS_SCALE = engine['loss_scale']
    support = support_settings(engine)                 # v8.3.3i
    batched.SUPPORT_RENORM = support
    levels = json.loads((P / 'math_levels.json').read_bytes())
    if levels.get('version') != 'affine-math-task-levels-v1' or not all(
            type(v) is int and 1 <= v <= 5 for v in levels['levels'].values()):
        raise ValueError('exact math level table')
    batched.MATH_LEVELS = dict(levels['levels'])
    batched.loss_aggregation(__import__('subnet.mimo_grpo.config', fromlist=['TrainerConfig']).TrainerConfig(
        loss_agg_mode='seq-mean-token-sum-norm', loss_scale_factor=8192))  # fail fast on a bad mode/scale
    staging = load('v824_staging_fast', 'v824_staging_fast.py')
    fast_opt = load('v824_optimizer_fast', 'v824_optimizer_fast.py')
    fast_pub = load('v824_publication_fast', 'v824_publication_fast.py')
    offpath = load('v8210_optimizer_offpath', 'v8210_optimizer_offpath.py')
    # install
    exec(compile(sources['receipt_consts'], transaction.__file__, 'exec'), transaction.__dict__)
    recovery._receipt = define(transaction, sources['receipt'], '_receipt')   # recovery imported it by name
    transaction.Transaction.run = define(transaction, sources['run'], 'run', transaction.Transaction)
    exec(compile(sources['pub_consts'], publication.__file__, 'exec'), publication.__dict__)
    define(publication, sources['verified'], 'publication_verified')
    handover.validate_publication = define(publication, sources['validate_publication'], 'validate_publication')
    patched_execute = define(service, sources['execute'], 'execute_transaction')
    define(worker, sources['validate_result'], 'validate_result')

    def execute_transaction(*args, **kwargs):
        kwargs.update(batched_engine=engine['batched_engine'], token_budget=engine['token_budget'],
                      max_rows=engine['max_rows'], fast_stage=engine['fast_stage'],
                      stage_workers=engine['stage_workers'])
        return patched_execute(*args, **kwargs)
    worker.execute_transaction = execute_transaction

    original_prepare = worker.prepare_signed_dispatch

    ff = load('ff_trainer', 'ff_trainer.py')   # v8.3.3h free-file turns

    def prepare_signed_dispatch(*args, **kwargs):
        dispatch = args[1] if len(args) > 1 else kwargs['dispatch']
        TURN_PARAMS['params'] = None
        if ff.is_ff(dispatch):        # v8.3.3h: signed sn120-ff-dispatch-v1 -> miner rewards as given
            batched.TASK_INDEX.clear()
            out = ff.prepare(*args, **kwargs)
            TURN_PARAMS['params'] = ff.train_params(out['dispatch'], out['binding'])     # v8.3.3j (None: no 'train')
            return out
        prepared_ = original_prepare(*args, **kwargs)
        try:   # diagnostics only: never alters or blocks the authenticated preparation
            index = {g.task_id: (r['task']['env_id'], r['task']['index'])
                     for g, r in zip(prepared_['groups'], prepared_['receipts'], strict=True)}
        except Exception:
            index = {}
        batched.TASK_INDEX.clear()
        batched.TASK_INDEX.update(index)
        return prepared_
    prepare_signed_dispatch.__qualname__ = original_prepare.__qualname__
    worker.prepare_signed_dispatch = prepare_signed_dispatch

    class R2ModelPublisherV824(fast_pub.FastR2ModelPublisher):
        def __init__(self, bucket, signer, **kwargs):
            kwargs.setdefault('shard_concurrency', engine['shard_concurrency'])
            super().__init__(bucket, signer, part_concurrency=engine['part_concurrency'], **kwargs)

        def __call__(self, request):        # v8.3.3g: signal the deferred fp32-master save (result unchanged)
            try:
                return super().__call__(request)
            finally:
                _PUBLISHED.set()
    worker.R2ModelPublisher = R2ModelPublisherV824

    pipelined = fast_opt.PipelinedCPUAdamW.step
    threads = engine['optimizer_step_threads']
    drift_cap, restart_warm, restart_scale = drift_settings(engine)
    sgd_lr, sgd_cap, sgd_mode, sgd_freeze = sgd_settings(engine)
    sgd_trc = engine.get('sgd_tensor_rel_cap')
    sgd_fp32 = fp32_settings(engine)
    sgd_dir = direction_settings(engine)               # v8.3.3i

    def step(self, gradients):
        if sgd_lr is not None:
            return sgd_step(self, gradients, sgd_lr, sgd_cap, sgd_mode, sgd_freeze, tensor_rel_cap=sgd_trc, fp32_master=sgd_fp32, direction=sgd_dir)   # v8.3.3 K=1: stateless SGD (no AdamW state touched)
        if not hasattr(self, '_pinned'):
            import torch
            self.step_threads = threads
            peak = max(p.numel() for _, p in self.parameters)
            # pinned host staging only when CUDA exists (production); plain host buffers otherwise
            self._pinned = [torch.empty(peak, dtype=torch.float32, pin_memory=torch.cuda.is_available()) for _ in range(2)]
            self._copy_stream = None
        if drift_cap is None and (restart_warm == 0 or restart_scale == 1.0):
            return pipelined(self, gradients)      # flags OFF: exactly the v8.3.1 step
        return controlled_step(self, gradients, pipelined, drift_cap, restart_warm, restart_scale)
    step.__qualname__ = optimizer.CPUAdamW.step.__qualname__
    optimizer.CPUAdamW.step = step
    if engine['optimizer_persistence'] == 'none':
        # The patched service imports `stage_for_publication_fast` from v824_staging_fast at call time.
        staging.stage_for_publication_fast = offpath.stage_for_publication_model_only
        restore = offpath.make_restore(recovery.restore_transaction_state)
        restore.__qualname__ = recovery.restore_transaction_state.__qualname__
        recovery.restore_transaction_state = restore
        worker.restore_transaction_state = restore     # imported by name from subnet.mimo_grpo
        mimo_grpo.restore_transaction_state = restore
    # ---- v8.3: declared training kernels + data-parallel packed execution of the same update
    # v8.3.1: private Triton kernel cache per rank on local disk (no cross-process compile race, no FUSE rename; /dev/shm is noexec)
    os.environ['TRITON_CACHE_DIR'] = TRITON_CACHE + '/rank0'
    kernels_policy = load('v83_kernels_policy', 'v83_kernels_policy.py')
    kernels = kernels_policy.allow(engine['training_kernels'])
    fast_kernels = load('v83_fast_kernels', 'v83_fast_kernels.py')
    dp = load('v83_dp', 'v83_dp.py')
    dp.BUDGET, dp.MAX_ROWS = engine['dp_token_budget'], engine['dp_max_rows']
    dp.install_fast_hash()     # v8.3.3g
    helpers = []
    if start_dp:
        import subprocess
        world, port = engine['dp_world'], engine['dp_port']
        for r in range(1, world):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(r), MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                       TRITON_CACHE_DIR=TRITON_CACHE + '/rank%d' % r)
            helpers.append(subprocess.Popen([sys.executable, '-B', str(P / 'v83_dp.py'), str(r), str(world), str(port),
                                             str(P), str(Path(config['source_root']).resolve())],
                                            env=env, cwd=str(Path(config['source_root']).resolve())))
        dp.init(world, port)
        dp.STATE['helpers'] = helpers
    dp_update = dp.make_train_update(batched, fast_kernels)

    def train_update_v833i(runtime, optimizer, groups, binding, *, config, **kwargs):
        if support is not None:     # Allan: the support settings must equal the job's (manifest) sampling settings
            got = dict(top_k=getattr(config, 'top_k', None), top_p=getattr(config, 'top_p', None), temperature=getattr(config, 'temperature', None))
            if got != support:
                raise ValueError('support_renorm %r != job sampling settings %r' % (support, got))
            if batched.SUPPORT_RENORM != support:
                raise ValueError('v8211_batched.SUPPORT_RENORM changed')
        out = dp_update(runtime, optimizer, groups, binding, config=config, **kwargs)
        if support is not None and isinstance(out, dict) and out.get('status') == 'updated':
            out['support_renorm'] = support_summary(out.get('trajectories') or [], support)
        return out
    train_update_v833i._original = dp_update._original
    import dataclasses as _dc

    def train_update_v833j(runtime, optimizer, groups, binding, *, config, **kwargs):
        """v8.3.3j: apply the prepared free-file turn's lr / kl to this one update (defaults -> exact v8.3.3i)."""
        tp = TURN_PARAMS['params']
        TURN['lr_multiplier'] = 1.0; TURN['last'] = None
        if tp is None:
            return train_update_v833i(runtime, optimizer, groups, binding, config=config, **kwargs)
        if (binding.epoch, binding.checkpoint) != (tp['epoch'], tp['checkpoint']):
            raise ValueError('v8.3.3j: turn params belong to another job')
        if tp['defaults'] != dict(lr=sgd_lr, kl=float(config.kl_coefficient)):
            raise ValueError('v8.3.3j: dispatch defaults %r != trainer defaults lr %r kl %r' % (tp['defaults'], sgd_lr, config.kl_coefficient))
        lr, kl = tp['lr'], tp['kl']
        m = 1.0 if lr == sgd_lr else lr / sgd_lr
        if m != 1.0 and not (sgd_mode == 'fresh-adam' and sgd_fp32 and sgd_dir == 'normgrad' and sgd_trc is not None):
            raise ValueError('v8.3.3j: a turn lr needs fresh-adam + fp32 master + normgrad + sgd_tensor_rel_cap')
        cfg = config if kl == config.kl_coefficient else _dc.replace(config, kl_coefficient=kl).validate()
        saved = optimizer.config
        TURN['lr_multiplier'] = m
        try:
            if cfg is not config:
                optimizer.config = cfg
            out = train_update_v833i(runtime, optimizer, groups, binding, config=cfg, **kwargs)
        finally:
            optimizer.config = saved; TURN['lr_multiplier'] = 1.0
        if isinstance(out, dict):
            if cfg is not config and 'config' in out:
                out['config'] = config.document()           # the signed config document, unchanged (validate_result)
            if out.get('status') == 'updated':
                out['train_params'] = dict(
                    version='v833j-turn-train-params-v1', lr=lr, kl=kl, given=list(tp['given']), defaults=dict(tp['defaults']),
                    defaults_used=(m == 1.0 and cfg is config), lr_multiplier=m, kl_applied=float(cfg.kl_coefficient),
                    step=TURN['last'],
                    rule=('kl replaces kl_coefficient in L = loss_scale * (1/N) * sum_i (1/K_i) * sum_t [pg + kl * k3]; '
                          'lr: m = lr / sgd_lr, n_t = min(m * r_t * ||d_t||, sgd_tensor_rel_cap * (1 - 1e-6) * ||M_H,t||), '
                          's = min(1, sgd_update_norm_cap / ||n||), Delta_t = s * n_t * g_t / ||g_t||; m = 1 -> the v8.3.3i step exactly'))
        TURN['last'] = None
        return out
    train_update_v833j._original = dp_update._original
    batched.train_update_batched = train_update_v833j
    return dict(component='trainer-host-v83', dp_world=engine['dp_world'] if start_dp else 0, training_kernels=kernels,
                engine=engine, drop_unresolved=batched.DROP_UNRESOLVED,
                loss_agg_mode=batched.LOSS_AGG_MODE, loss_scale=batched.LOSS_SCALE, math_levels=len(batched.MATH_LEVELS),
                drop_policy=batched.DROP_POLICY, optimizer_persistence=engine['optimizer_persistence'],
                drift_control=dict(drift_cap_rel=drift_cap, restart_lr_warm_steps=restart_warm, restart_lr_scale=restart_scale),
                sgd=dict(sgd_lr=sgd_lr, sgd_update_norm_cap=sgd_cap, sgd_mode=sgd_mode, sgd_freeze=list(sgd_freeze), sgd_tensor_rel_cap=sgd_trc, fp32_master=sgd_fp32, direction=sgd_dir, on=sgd_lr is not None),
                support_renorm=support, turn_train_params='v833j: free-file dispatch train {lr, kl}',
                sites=sorted(SITES))


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit('usage: mimo_host_trainer_v83.py --operator-config PATH [--check] -- <distributed_worker args>')
    split = argv.index('--')
    p = argparse.ArgumentParser(); p.add_argument('--operator-config', required=True); p.add_argument('--check', action='store_true')
    args = p.parse_args(argv[:split]); worker_args = argv[split + 1:]
    config = checked(args.operator_config)
    result = install(config, start_dp=not args.check)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.check:
        return 0
    sys.argv = [str(Path(config['source_root']) / 'subnet' / 'distributed_worker.py')] + worker_args
    runpy.run_module('subnet.distributed_worker', run_name='__main__', alter_sys=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
