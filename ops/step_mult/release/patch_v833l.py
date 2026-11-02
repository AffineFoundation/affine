#!/usr/bin/env python3
"""RA/20261102-step-mult: trainer host v8.3.3k -> v8.3.3l (optional per-turn knobs, all defaulting to today's step; Jacob 2026-11-02 11:19 CR).
usage: patch_v833l.py SRC_DIR(host-v833k stage) OUT_DIR KNOBSPEC.py"""
import shutil, sys
from pathlib import Path
S, O, KS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).read_text()
if O.exists(): shutil.rmtree(O)
shutil.copytree(S, O, symlinks=True)
def edit(n, pairs):
    p = O / n; s = p.read_text()
    for o, nw in pairs:
        c = s.count(o)
        if c != 1: raise SystemExit('%s: anchor count %d: %r' % (n, c, o[:100]))
        s = s.replace(o, nw)
    p.write_text(s)
SPEC = KS.split('KNOB_SPEC = {', 1)[1]; SPEC = 'KNOB_SPEC = {' + SPEC
# ---------------------------------------------------------------- mimo_host_trainer_v83.py
H_DOC_O = "  per-tensor cap are scaled together; one gradient, one step, same trainer time. k = 1 (or no step_mult) runs the exact v8.3.3j step.\n"
H_DOC_N = H_DOC_O + ("v8.3.3l = v8.3.3k + optional per-turn knobs from the dispatch's train.knobs (RA/20261102-step-mult, Jacob 2026-11-02 11:19 CR):\n"
  "  per-group lr scales (attn, mlp, embed, lm_head, early / late layers), adam_eps, weight_decay, clamp_mode + global_norm_cap (step),\n"
  "  loss_agg, adv_center, adv_clip (loss). Every knob defaults to today's value; a turn whose knobs are all default runs the exact\n"
  "  v8.3.3k code path (TURN['opt'] is None and v8211_batched.KNOBS is None).\n")
TURN_O = "TURN = dict(lr_multiplier=1.0, step_mult=1, last=None)      # set only by train_update_v833j around one update\n"
TURN_N = ("TURN = dict(lr_multiplier=1.0, step_mult=1, opt=None, last=None)      # set only by train_update_v833j around one update\n"
  "# v8.3.3l knobs (RA/20261102-step-mult). TURN['opt'] (None = the v8.3.3k step exactly) = the step knobs of one turn:\n"
  "#   e_t = lr * (g_t / (|g_t| + adam_eps) + weight_decay * W_t * [decayed_t]);  sq_t = ||e_t||;  a_t = lr scale of t's group x early/late scale\n"
  "#   per_tensor / both: n_t = k * min(m * a_t * min(sq_t, c_t), c_t);  global_norm: n_t = k * m * a_t * sq_t (no per-tensor cap)\n"
  "#   s = min(1, G / ||n||), G = sgd_update_norm_cap (per_tensor) or min(sgd_update_norm_cap, global_norm_cap) (global_norm / both)\n"
  "#   Delta_t = fp32(s * n_t / ||g_t||) * g_t  (direction unchanged: g_t / ||g_t||). a_t = 0 freezes t.\n"
  + SPEC + "\n\n"
  "def knob_group(name):\n"
  "    if name.startswith('lm_head'): return 'lm_head'\n"
  "    if 'embed_tokens' in name: return 'embed'\n"
  "    if '.self_attn.' in name or '.linear_attn.' in name: return 'attn'\n"
  "    if '.mlp.' in name: return 'mlp'\n"
  "    return 'other'\n\n\n"
  "def knob_layer(name):\n"
  "    p = name.split('.layers.', 1)\n"
  "    if len(p) != 2: return None\n"
  "    i = p[1].split('.', 1)[0]\n"
  "    return int(i) if i.isdigit() else None\n\n\n"
  "def knob_scale(name, opt, half):\n"
  "    g = knob_group(name); a = opt['lr_scale_' + g] if g != 'other' else 1.0\n"
  "    li = knob_layer(name)\n"
  "    if li is not None: a = a * (opt['lr_scale_early'] if li < half else opt['lr_scale_late'])\n"
  "    return a\n")
STEP_O = ("        TURN['last'].update(path=TURN['last']['path'] + '+step_mult', step_mult=k_mult, raw_update_norm_k1=TURN['last']['raw_update_norm'],\n"
          "                            raw_update_norm=raw_k, cap_scale=scale_k, cap_fired=scale_k < 1.0, tensor_rel_cap_effective=tensor_rel_cap * k_mult)\n")
STEP_N = STEP_O + (
"    ok = TURN.get('opt')                                   # v8.3.3l step knobs (None -> nothing below runs)\n"
"    if ok is not None:\n"
"        if not (normgrad and tensor_rel_cap is not None and fp32_master):\n"
"            raise ValueError('v8.3.3l: step knobs need the normgrad fp32-master step with the per-tensor cap')\n"
"        eps_l, wd_l, mode_l = ok['adam_eps'], ok['weight_decay'], ok['clamp_mode']\n"
"        q_l = qd; wn_l = dict(wnd)\n"
"        if eps_l != FRESH_ADAM_EPS or wd_l != FRESH_ADAM_WD:\n"
"            q_l = {}\n"
"            with torch.no_grad():\n"
"                for name, parameter in live:\n"
"                    g = gradients[name].detach().to(device=parameter.device); w32 = W32(name, parameter)\n"
"                    e = g / (g.abs() + torch.tensor(eps_l, dtype=torch.float32, device=g.device))\n"
"                    if w32.dim() >= 2 and not name.endswith('bias'):\n"
"                        e = e + torch.tensor(wd_l, dtype=torch.float32, device=g.device) * w32\n"
"                    e = torch.tensor(lr, dtype=torch.float32, device=g.device) * e\n"
"                    q_l[name] = float(e.double().pow(2).sum())\n"
"                    if name not in wn_l: wn_l[name] = math.sqrt(float(w32.double().pow(2).sum()))\n"
"                    del w32, g, e\n"
"        layers = [knob_layer(n) for n in r]; half = (max([x for x in layers if x is not None] or [-1]) + 1) // 2\n"
"        n_l = {}; at_cap = 0; grp = {}\n"
"        for name in r:\n"
"            a = knob_scale(name, ok, half); sq = math.sqrt(q_l[name])\n"
"            c = tensor_rel_cap * (1 - 1e-6) * wn_l[name] if wn_l.get(name, 0.0) > 0 else None\n"
"            if mode_l in ('per_tensor', 'both') and c is not None:\n"
"                v = lr_mult * a * min(sq, c); at_cap += v >= c; v = min(v, c)\n"
"            else:\n"
"                v = lr_mult * a * sq\n"
"            n_l[name] = k_mult * v; gk = knob_group(name); grp[gk] = grp.get(gk, 0.0) + n_l[name] ** 2\n"
"        G = cap if mode_l == 'per_tensor' else min(cap, ok['global_norm_cap'])\n"
"        raw_l = math.sqrt(math.fsum(v * v for v in n_l.values())); scale_l = min(1.0, G / raw_l) if raw_l > 0 else 1.0\n"
"        coef_j = {name: (scale_l * n_l[name] / math.sqrt(gn2[name])) if gn2[name] > 0 else 0.0 for name in r}\n"
"        TURN['last'].update(path=TURN['last']['path'] + '+knobs', knobs_step=dict(ok), raw_update_norm_pre_knobs=TURN['last']['raw_update_norm'],\n"
"                            raw_update_norm=raw_l, cap_scale=scale_l, cap_fired=scale_l < 1.0, global_cap_effective=G,\n"
"                            tensors_at_ceiling_knobs=int(at_cap), frozen_tensors=sum(1 for v in n_l.values() if v == 0.0),\n"
"                            group_step_norms={k2: math.sqrt(v2) * scale_l for k2, v2 in sorted(grp.items())})\n")
W_RESET_O = "        TURN['lr_multiplier'] = 1.0; TURN['step_mult'] = 1; TURN['last'] = None\n        if tp is None:\n"
W_RESET_N = "        TURN['lr_multiplier'] = 1.0; TURN['step_mult'] = 1; TURN['opt'] = None; TURN['last'] = None; batched.KNOBS = None\n        if tp is None:\n"
W_CHK_O = "        cfg = config if kl == config.kl_coefficient else _dc.replace(config, kl_coefficient=kl).validate()\n        saved = optimizer.config\n        TURN['lr_multiplier'] = m; TURN['step_mult'] = k_mult\n"
W_CHK_N = ("        kn = tp.get('knobs') or dict(KNOB_DEFAULTS)          # v8.3.3l\n"
"        if type(kn) is not dict or set(kn) != set(KNOB_SPEC):\n"
"            raise ValueError('v8.3.3l: knobs keys %r' % (sorted(kn) if type(kn) is dict else kn,))\n"
"        for kk, vv in kn.items():\n"
"            if not (vv is None and KNOB_SPEC[kk][2] is None) and knob_value(kk, vv) != vv:\n"
"                raise ValueError('v8.3.3l: knob %s not normalized' % kk)\n"
"        OPT_KEYS = ('lr_scale_attn', 'lr_scale_mlp', 'lr_scale_embed', 'lr_scale_lm_head', 'lr_scale_early', 'lr_scale_late',\n"
"                    'adam_eps', 'weight_decay', 'clamp_mode', 'global_norm_cap')\n"
"        LOSS_KEYS = ('loss_agg', 'adv_center', 'adv_clip')\n"
"        opt_on = kn['clamp_mode'] != 'per_tensor' or any(kn[x] != KNOB_DEFAULTS[x] for x in OPT_KEYS if x != 'global_norm_cap')\n"
"        loss_on = any(kn[x] != KNOB_DEFAULTS[x] for x in LOSS_KEYS)\n"
"        if opt_on and not (sgd_mode == 'fresh-adam' and sgd_fp32 and sgd_dir == 'normgrad' and sgd_trc is not None):\n"
"            raise ValueError('v8.3.3l: step knobs need fresh-adam + fp32 master + normgrad + sgd_tensor_rel_cap')\n"
"        if loss_on and batched.DROP_UNRESOLVED:\n"
"            raise ValueError('v8.3.3l: loss knobs need drop_unresolved off')\n"
+ W_CHK_O +
"        TURN['opt'] = {x: kn[x] for x in OPT_KEYS} if opt_on else None\n"
"        batched.KNOBS = {x: kn[x] for x in LOSS_KEYS} if loss_on else None\n")
W_FIN_O = "            optimizer.config = saved; TURN['lr_multiplier'] = 1.0; TURN['step_mult'] = 1\n"
W_FIN_N = "            optimizer.config = saved; TURN['lr_multiplier'] = 1.0; TURN['step_mult'] = 1; TURN['opt'] = None; batched.KNOBS = None\n"
W_REP_O = ("                    defaults_used=(m == 1.0 and cfg is config and k_mult == 1), lr_multiplier=m, kl_applied=float(cfg.kl_coefficient),\n"
           "                    step_mult=k_mult,\n")
W_REP_N = ("                    defaults_used=(m == 1.0 and cfg is config and k_mult == 1 and not opt_on and not loss_on), lr_multiplier=m, kl_applied=float(cfg.kl_coefficient),\n"
           "                    step_mult=k_mult, knobs=dict(kn), knobs_applied=dict(step=opt_on, loss=loss_on),\n")
RULE_O = "'step_mult k (v8.3.3k): n_t := k * n_t before the global cap (lr and per-tensor cap scaled together), k = 1 -> unchanged'))\n"
RULE_N = ("'step_mult k (v8.3.3k): n_t := k * n_t before the global cap (lr and per-tensor cap scaled together), k = 1 -> unchanged; '\n"
          "                          'knobs (v8.3.3l): see the TURN comment / mining.json batch_file.train; all default -> unchanged'))\n")
edit('mimo_host_trainer_v83.py', [(H_DOC_O, H_DOC_N), (TURN_O, TURN_N), (STEP_O, STEP_N), (W_RESET_O, W_RESET_N), (W_CHK_O, W_CHK_N),
    (W_FIN_O, W_FIN_N), (W_REP_O, W_REP_N), (RULE_O, RULE_N),
    ("turn_train_params='v833k: free-file dispatch train {lr, kl, step_mult}'", "turn_train_params='v833l: free-file dispatch train {lr, kl, step_mult, knobs}'")])
# ---------------------------------------------------------------- v8211_batched.py
LAYOUT_O = ("            rows.append(dict(group=group, rollout=rollout, advantage=float(advantage), active=active,\n"
            "                             keep=len(rollout.tokens), derail_index=None,\n"
            "                             reward=rollout_reward(config, rollout)))\n    _prompt_weights(rows)\n    return rows\n")
LAYOUT_N = ("            rows.append(dict(group=group, rollout=rollout, advantage=float(advantage), active=active,\n"
            "                             keep=len(rollout.tokens), derail_index=None,\n"
            "                             reward=rollout_reward(config, rollout)))\n"
            "    if KNOBS is not None:          # v8.3.3l per-turn loss knobs (None -> the exact previous rows)\n"
            "        _apply_knobs(rows)\n"
            "    _prompt_weights(rows)\n    return rows\n\n\n"
            "KNOBS = None   # v8.3.3l: dict(loss_agg, adv_center, adv_clip) for one turn, set by the host around one update (+ broadcast to DP helpers)\n\n\n"
            "def _apply_knobs(rows):\n"
            "    \"\"\"adv_center: 'mean' (upstream GRPO mean baseline, no std; a 1-rollout group keeps A = r), 'none' (A = r),\n"
            "    'group_norm' (upstream GRPO with std normalisation, eps 1e-6; 1-rollout group A = r). adv_clip c: A = clamp(A, -c, c).\n"
            "    loss_agg 'token_mean': each active rollout's loss x N * len_i / sum_j len_j (N active rollouts, len = rollout tokens),\n"
            "    i.e. (1/sum len) * sum_i sum_t instead of (1/N) * sum_i (1/len_i) * sum_t.\"\"\"\n"
            "    k = KNOBS; c = k.get('adv_center', 'mean'); clip = k.get('adv_clip'); agg = k.get('loss_agg', 'seq_mean')\n"
            "    if c not in ('mean', 'none', 'group_norm') or agg not in ('seq_mean', 'token_mean'):\n"
            "        raise ValueError('loss knobs %r' % (k,))\n"
            "    if c != 'mean':\n"
            "        by = {}\n"
            "        for i, r in enumerate(rows):\n"
            "            by.setdefault(id(r['group']), []).append(i)\n"
            "        for idx in by.values():\n"
            "            if c == 'none':\n"
            "                for i in idx: rows[i]['advantage'] = float(rows[i]['reward'])\n"
            "            else:\n"
            "                rewards = torch.tensor([[rows[i]['reward']] for i in idx], dtype=torch.float32)\n"
            "                vals, _ = reference().compute_grpo_outcome_advantage(rewards, torch.ones_like(rewards),\n"
            "                    np.array([rows[idx[0]]['group'].task_id] * len(idx)), norm_adv_by_std_in_grpo=True)\n"
            "                for i, v in zip(idx, vals[:, 0], strict=True): rows[i]['advantage'] = float(v)\n"
            "    if clip is not None:\n"
            "        for r in rows: r['advantage'] = max(-float(clip), min(float(clip), r['advantage']))\n"
            "    if agg == 'token_mean':\n"
            "        act = [r for r in rows if r['active']]; total = sum(len(r['rollout'].tokens) for r in act)\n"
            "        for r in act: r['agg_w'] = len(act) * len(r['rollout'].tokens) / total\n")
DROP_O = '    group baseline is computed over resolved rollouts only; groups with < 2 resolved are dropped."""\n'
DROP_N = DROP_O + "    if KNOBS is not None:\n        raise ValueError('v8.3.3l loss knobs need drop_unresolved off')\n"
GP_O = "            if loss_scale != 1.0:\n                loss = loss * loss_scale\n"
GP_N = GP_O + "            if row.get('agg_w') is not None:           # v8.3.3l loss_agg token_mean (absent -> unchanged)\n                loss = loss * row['agg_w']\n"
edit('v8211_batched.py', [(LAYOUT_O, LAYOUT_N), (DROP_O, DROP_N), (GP_O, GP_N)])
# ---------------------------------------------------------------- v83_dp.py
edit('v83_dp.py', [("mode=B.LOSS_AGG_MODE, scale=B.LOSS_SCALE, support=B.SUPPORT_RENORM,\n",
                    "mode=B.LOSS_AGG_MODE, scale=B.LOSS_SCALE, support=B.SUPPORT_RENORM, knobs=getattr(B, 'KNOBS', None),\n"),
                   ("        B.SUPPORT_RENORM = job.get('support')\n", "        B.SUPPORT_RENORM = job.get('support')\n        B.KNOBS = job.get('knobs')                       # v8.3.3l\n")])
# ---------------------------------------------------------------- ff_trainer.py
edit('ff_trainer.py', [
  ("def train_params(selection, binding):\n", SPEC + "\n\ndef train_params(selection, binding):\n"),
  ("    need(type(t) is dict and set(t) - {'step_mult'} == {'lr', 'kl', 'given', 'defaults'}, 'train keys')\n",
   "    need(type(t) is dict and set(t) - {'step_mult', 'knobs'} == {'lr', 'kl', 'given', 'defaults'}, 'train keys')\n"
   "    kn = t.get('knobs', dict(KNOB_DEFAULTS))                # v8.3.3l\n"
   "    need(type(kn) is dict and set(kn) == set(KNOB_SPEC), 'train.knobs keys')\n"
   "    for kk, vv in kn.items():\n"
   "        if vv is None and KNOB_SPEC[kk][2] is None: continue\n"
   "        try: ok = knob_value(kk, vv) == vv and type(vv) is (float if KNOB_SPEC[kk][0] == 'number' else str)\n"
   "        except ValueError: ok = False\n"
   "        need(ok, 'train.knobs.%s' % kk)\n"),
  ("    need(type(t['given']) is list and set(t['given']) <= set(TRAIN_BOUNDS) | {'step_mult'} and",
   "    need(type(t['given']) is list and set(t['given']) <= set(TRAIN_BOUNDS) | {'step_mult'} | set(KNOB_SPEC) and"),
  ("    return dict(lr=t['lr'], kl=t['kl'], step_mult=km, given=list(t['given']), defaults=dict(lr=d['lr'], kl=d['kl']),",
   "    return dict(lr=t['lr'], kl=t['kl'], step_mult=km, knobs=dict(kn), given=list(t['given']), defaults=dict(lr=d['lr'], kl=d['kl']),")])
print('ok')
