#!/usr/bin/env python3
"""Apply ONE SN120 turn step to a batch file locally, with the live trainer code (v8.3.3l).

What runs here is the validator's own code, imported from miner/validator/ (verbatim copies, sha256 in
miner/validator/MANIFEST.sha256):
  * format check + tokenization: batch/ff_format.validate (the lifecycle's check; affine-batch-v1 and v2)
  * groups: ff_trainer.RewardedTrajectory / FreeGroup / label (exactly as ff_trainer.prepare builds them)
  * loss + gradient: v8211_batched.train_update_batched (rewards + overlong penalty, GRPO advantage, derail mask,
    support renorm top-20 / top-p 0.95, seq-mean-token-mean x 0.5635, k3 KL to the reference, fp32 accumulation, clip 1.0)
  * step: engine CPUAdamW.step routed to mimo_host_trainer_v83.sgd_step exactly as the host install() does
    (fresh-adam, per-tensor cap 2e-5, global cap 1.0, fp32 master, direction normgrad) + the v8.3.3j lr multiplier / kl + the v8.3.3l step_mult and knobs.
Differences from the live run (execution only, same math): one process and engine 'padded' (live: 8 GPUs, engine
'packed' with fla/FlashAttention varlen kernels, NCCL SUM); gradients therefore differ at the bf16-ulp level
(see ops/renorm_step/results/EIGHT-VS-ONE-e354.json for the size of that difference). The fp32 master: pass the
published one (fetch_fp32_master.py --out HEAD_DIR-fp32-master; affine.io/k1/fp32-master/<head>.json). It is used
automatically when HEAD_DIR-fp32-master/FP32-MASTER.json names this head, or pass --master DIR. Without it the step
bootstraps M_H = fp32(bf16 head), which drifts from the gate's candidate (the live trainer steps the stored master).
The candidate's own fp32 master is written to CANDIDATE_DIR/fp32-master/ (a subdirectory, so vLLM / from_pretrained
never load it as model weights and the checkpoint id is unchanged).
--activation-checkpointing (opt-in, default off): recompute each decoder layer in the backward pass instead of keeping its
activations (torch.utils.checkpoint, use_reentrant=False). Same math, more compute, much less memory: use it when one
GPU runs out of memory on a full batch file (e.g. 1x H200 with 1,000 rollouts).

usage: train_step.py --head HEAD_DIR --ref REF_DIR --batch batch.json --hotkey SS58 --out CANDIDATE_DIR
       [--master FP32_MASTER_DIR | --no-master] [--activation-checkpointing] [--device cuda|cpu] [--report report.json]
       [--max-tasks 125 --max-rollouts 1000]
"""
import argparse, dataclasses, hashlib, json, math, sys, time
from pathlib import Path
import kitpath  # noqa: F401  (sys.path + namespace stubs)

CFG = json.loads((kitpath.HOST / 'trainer-config.public.json').read_text())


def descriptor_id(path):
    """Checkpoint id = sha256(canonical JSON {file name: sha256}) over the checkpoint files (mimo_runtime.validate_descriptor)."""
    files = {}
    for p in sorted(Path(path).iterdir()):
        if p.is_file() and not p.name.startswith('.') and (p.suffix in ('.json', '.safetensors', '.jinja')):
            files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Runtime:
    """The attributes v8211_batched uses from the validator's MiMo runtime."""
    def __init__(self, model, checkpoint_id, device, trainable, eos, max_context):
        self.model, self.checkpoint_id, self.device, self.trainable = model, checkpoint_id, device, trainable
        cfg = getattr(model.config, 'text_config', None) or model.config
        self.vocab_size = int(cfg.vocab_size); self.eos_token_ids = tuple(eos); self.max_context_tokens = max_context
    def guard(self): pass
    def _tokens(self, x): return list(x)


def load_model(path, device, dtype, trainable):
    import torch
    from transformers import AutoModelForCausalLM
    try:
        from transformers import AutoModelForImageTextToText as AV
    except Exception:
        AV = None
    cfg = json.loads((Path(path) / 'config.json').read_text())
    cls = AV if (AV is not None and 'ConditionalGeneration' in ''.join(cfg.get('architectures') or [])) else AutoModelForCausalLM
    model = cls.from_pretrained(str(path), dtype=dtype).to(device)
    names = [n for n, _ in model.named_parameters()]
    prefixes = ('model.language_model.', 'lm_head.') if any(n.startswith('model.language_model.') for n in names) else ('',)
    for n, p in model.named_parameters():
        p.requires_grad_(trainable and n.startswith(prefixes) and 'mtp' not in n)
    return model


def enable_activation_checkpointing(model):
    """Opt-in: wrap every decoder layer's forward in torch.utils.checkpoint (non-reentrant) while gradients are on.
    The trainer forward always runs with use_cache=False and no dropout (eval mode), so recompute gives the same values."""
    import torch
    from torch.utils.checkpoint import checkpoint
    lm = getattr(model.model, 'language_model', model.model)
    layers = getattr(lm, 'layers', None)
    if layers is None:
        raise SystemExit('--activation-checkpointing: no decoder layers found')
    for layer in layers:
        def fwd(*args, _f=layer.forward, **kw):
            if torch.is_grad_enabled() and not kw.get('use_cache') and kw.get('past_key_values') is None:
                return checkpoint(_f, *args, use_reentrant=False, **kw)
            return _f(*args, **kw)
        layer.forward = fwd
    return len(layers)


def find_master(head_dir, head_id):
    """HEAD_DIR-fp32-master (fetch_fp32_master.py --out) when it was verified for this exact head."""
    d = Path(str(Path(head_dir)).rstrip('/') + '-fp32-master'); tag = d / 'FP32-MASTER.json'
    try:
        return str(d) if json.loads(tag.read_text()).get('checkpoint') == head_id else None
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--head', required=True); ap.add_argument('--ref', required=True); ap.add_argument('--batch', required=True)
    ap.add_argument('--hotkey', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--master', help='published fp32 master dir of the head (default: HEAD_DIR-fp32-master when verified)')
    ap.add_argument('--no-master', action='store_true', help='ignore any master; bootstrap M_H = fp32(bf16 head)')
    ap.add_argument('--activation-checkpointing', action='store_true',
                    help='opt-in (default off): recompute decoder layers in backward to fit one GPU; same math, slower')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--report')
    ap.add_argument('--max-tasks', type=int, default=125); ap.add_argument('--max-rollouts', type=int, default=1000)
    ap.add_argument('--token-budget', type=int, default=16384)
    a = ap.parse_args(argv)
    import torch
    import ff_format, ff_trainer, v8211_batched as B, mimo_host_trainer_v83 as H
    from subnet.mimo_grpo.config import TrainerConfig
    from subnet.mimo_grpo.data import InputBinding, digest
    from subnet.mimo_grpo import optimizer as engine_opt

    E = CFG['engine']
    t0 = time.time()
    # 1. the lifecycle's format check + tokenization (same caps as the live turn; v2 accepted)
    tok = ff_format.tokenizer(a.head)
    raw = Path(a.batch).read_bytes()
    v = ff_format.validate(raw, a.hotkey, tok, a.max_tasks, a.max_rollouts, v2=True)
    s = v['summary']; train = s.get('train') or dict(ff_format.TRAIN_DEFAULTS)
    lr, kl = float(train['lr']), float(train['kl']); k_mult = int(train.get('step_mult', 1))
    kn = {x: train.get(x, H.KNOB_DEFAULTS[x]) for x in H.KNOB_SPEC}          # v8.3.3l knobs, as train_update_v833j applies them
    OPT_KEYS = ('lr_scale_attn', 'lr_scale_mlp', 'lr_scale_embed', 'lr_scale_lm_head', 'lr_scale_early', 'lr_scale_late',
                'adam_eps', 'weight_decay', 'clamp_mode', 'global_norm_cap')
    LOSS_KEYS = ('loss_agg', 'adv_center', 'adv_clip')
    opt_on = kn['clamp_mode'] != 'per_tensor' or any(kn[x] != H.KNOB_DEFAULTS[x] for x in OPT_KEYS if x != 'global_norm_cap')
    loss_on = any(kn[x] != H.KNOB_DEFAULTS[x] for x in LOSS_KEYS)
    # 2. groups exactly as ff_trainer.prepare builds them (document -> RewardedTrajectory / FreeGroup)
    groups = []
    for i, g in enumerate(v['groups']):
        trajs = tuple(ff_trainer.RewardedTrajectory(nonce=n, tokens=tuple(r['tokens']), outcome=ff_trainer.label(r['stop'], r['reward']),
                                                    stop=r['stop'], reward=float(r['reward'])) for n, r in enumerate(g['rollouts']))
        psha = hashlib.sha256(g['prompt'].encode()).hexdigest()
        doc = dict(version=ff_trainer.DOCUMENT, group_index=i, hotkey=a.hotkey, prompt_sha256=psha, prompt_tokens=g['prompt_tokens'],
                   rollouts=g['rollouts'])
        grp = ff_trainer.FreeGroup(task_id=digest(dict(ff=ff_trainer.VERSION, epoch='local-sim', hotkey=a.hotkey, group_index=i,
                                                       prompt_sha256=psha)), prompt=tuple(g['prompt_tokens']), trajectories=trajs,
                                   admission_sha256=digest(doc))
        groups.append(grp)
    # 3. config: the signed GRPO config; the turn's kl replaces kl_coefficient (train_update_v833j)
    base_cfg = TrainerConfig(**CFG['trainer_config']['settings']).validate()
    config = base_cfg if kl == base_cfg.kl_coefficient else dataclasses.replace(base_cfg, kl_coefficient=kl).validate()
    m = 1.0 if lr == E['sgd_lr'] else lr / E['sgd_lr']
    # 4. the live engine settings
    B.DROP_UNRESOLVED = E['drop_unresolved']; B.LOSS_AGG_MODE = E['loss_agg_mode']; B.LOSS_SCALE = E['loss_scale']
    B.SUPPORT_RENORM = dict(E['support_renorm'])
    eos = (248046, 248044)
    head_id = descriptor_id(a.head)
    dev = torch.device(a.device)
    model = load_model(a.head, dev, torch.bfloat16, True)
    if a.activation_checkpointing:
        print(json.dumps(dict(activation_checkpointing=enable_activation_checkpointing(model))), flush=True)
    master = None if a.no_master else (a.master or find_master(a.head, head_id))
    if master:
        tag = Path(master) / 'FP32-MASTER.json'
        if tag.exists() and json.loads(tag.read_text()).get('checkpoint') != head_id:
            raise SystemExit(f'{master} is the fp32 master of another head (FP32-MASTER.json), not {head_id}')
    print(json.dumps(dict(fp32_master=master or 'bootstrap fp32(bf16 head)')), flush=True)
    ref = load_model(a.ref, dev, torch.bfloat16, False).eval()
    runtime = Runtime(model, head_id, dev, True, eos, ff_format.MAX_CONTEXT)
    ref_rt = Runtime(ref, config.kl_reference_checkpoint, dev, False, eos, ff_format.MAX_CONTEXT)
    for grp in groups:
        grp.validate(vocab_size=runtime.vocab_size, eos_token_ids=set(eos))
    # 5. optimizer: engine CPUAdamW with .step routed to the host's stateless step (as install() does)
    opt = engine_opt.CPUAdamW(model.named_parameters(), config, input_checkpoint=head_id)
    rows = []
    def step(self, gradients):
        return H.sgd_step(self, gradients, E['sgd_lr'], E['sgd_update_norm_cap'], E['sgd_mode'], tuple(E['sgd_freeze']),
                          emit=rows.append, tensor_rel_cap=E['sgd_tensor_rel_cap'], fp32_master=E['sgd_fp32_master'],
                          direction=E['sgd_direction'])
    engine_opt.CPUAdamW.step = step
    H.FP32_STORE = str(Path(a.out) / '.fp32-store-unused')
    H._fp32_save = lambda *x, **k: None            # the live trainer persists the master to its pod disk; not needed here
    if master:                                     # published head master -> the trainer's 'memory' path (verified bf16(M) == W)
        from safetensors.torch import load_file
        got = 0; params = dict(model.named_parameters())
        for f in sorted(Path(master).glob('*.safetensors')):
            for n, t in load_file(str(f)).items():
                if n in opt.masters:
                    opt.masters[n].data.copy_(t.float()); got += 1
                    if not torch.equal(opt.masters[n].to(params[n].device, torch.bfloat16), params[n].data):
                        raise SystemExit(f'bf16(master) != head weight for {n}: the master is not this head\'s')
        if got != len(opt.masters): raise SystemExit('fp32 master covers %d of %d trainable tensors' % (got, len(opt.masters)))
        opt._v833f_bf16, _ = H._digest_map(list(opt.parameters))
    binding = InputBinding('local-sim', 'local-sim', head_id, '0' * 64, opt.global_step)
    H.TURN['lr_multiplier'] = m; H.TURN['step_mult'] = k_mult; H.TURN['last'] = None
    H.TURN['opt'] = {x: kn[x] for x in OPT_KEYS} if opt_on else None
    B.KNOBS = {x: kn[x] for x in LOSS_KEYS} if loss_on else None
    try:
        out = B.train_update_batched(runtime, opt, groups, binding, config=config, reference_runtime=ref_rt, engine='padded',
                                     token_budget=a.token_budget, max_rows=E['max_rows'])
    finally:
        H._PUBLISHED.set()
    out['train_params'] = dict(lr=lr, kl=kl, step_mult=k_mult, knobs=kn, knobs_applied=dict(step=opt_on, loss=loss_on), lr_multiplier=m, kl_applied=float(config.kl_coefficient), given=s.get('train_given'),
                               step=H.TURN['last'])
    if out.get('status') == 'updated':
        out['support_renorm'] = H.support_summary(out.get('trajectories') or [], B.SUPPORT_RENORM)
    H.TURN['lr_multiplier'] = 1.0; H.TURN['step_mult'] = 1; H.TURN['opt'] = None; B.KNOBS = None
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    for extra in ('chat_template.jinja', 'generation_config.json'):
        p = Path(a.head) / extra
        if p.exists() and not (Path(a.out) / extra).exists(): (Path(a.out) / extra).write_bytes(p.read_bytes())
    from safetensors.torch import save_file
    (Path(a.out) / 'fp32-master').mkdir(exist_ok=True)   # subdirectory: never picked up as model weights
    save_file({n: t.detach().contiguous() for n, t in opt.masters.items()}, str(Path(a.out) / 'fp32-master' / 'fp32-master.safetensors'))
    rep = dict(head=head_id, fp32_master=master, activation_checkpointing=a.activation_checkpointing, file_summary=s, wall_seconds=time.time() - t0, sgd_events=rows,
               report={k: out[k] for k in out if k not in ('trajectories', 'admitted_groups')},
               trajectories=out.get('trajectories'))
    txt = json.dumps(rep, indent=1, default=str)
    if a.report: Path(a.report).write_text(txt)
    o = rep['report']; sgd = (o.get('optimizer') or {}).get('sgd') or {}
    print(json.dumps(dict(status=o.get('status'), loss=o.get('objective_loss'), preclip_gradient_norm=o.get('preclip_gradient_norm'),
                          raw_update_norm=sgd.get('raw_update_norm'), update_norm_master=(sgd.get('fp32_master') or {}).get('update_norm_master'),
                          tensors_rel_capped=sgd.get('tensors_rel_capped'), lr=lr, kl=kl, step_mult=k_mult, knobs={x: v for x, v in kn.items() if v != H.KNOB_DEFAULTS[x]}, out=a.out)))
    return rep


if __name__ == '__main__':
    main()
