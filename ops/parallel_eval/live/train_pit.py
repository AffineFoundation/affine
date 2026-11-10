"""PIT test1 (per-item train+eval, SHADOW/TEST ONLY): one native GRPO step on the box, no coordinator, no signer seed.
Inputs under /workspace/pst/in (signed TRAINING-JOB manifest + signed TRAIN-DISPATCH-FF + docs + head + fp32 master in FP32_STORE).
Fail closed: AFFINE_FP32_EXPECTED_MASTER_HASH must be set; no bootstrap override is allowed; the trainer's own _fp32_prepare checks it.

Non-sharded path (B300/B200/H200): always ``host.checked(ROOT-ADMITTED.private.json)`` against
host-v834e — same call and admitted host-file digests as 27701370.

Sharded path (host-v834f): separate ROOT-admitted package. Until Jacob ROOT-signs that
admission, host-v834f is absent on the box and this script stays on host-v834e.
``z3.activate(host)`` (never ``None``) so ``wrap_host_install`` runs; helpers start
through ``v835_zero3.py``; every rank FSDP-wraps together before the optimizer.
"""
import json, os, sys, time, importlib.util
from pathlib import Path
T = time.time()
W = Path('/workspace/affine-mimo-trainer-prod'); SRC = W / 'source-final-v823u'


def _select_host() -> Path:
    """host-v834e by default; host-v834f only when sharded mode is authorized.

    Fail closed:
      * host-v834e missing when non-sharded path is required;
      * ``PIT_ZERO3`` set without ``allow_sharded_gpus`` (PROMOTION-GATE).
    """
    v834e = W / 'host-v834e'
    v834f = W / 'host-v834f'
    force = os.environ.get('PIT_ZERO3', '').strip().lower() in ('1', 'true', 'yes', 'on')
    allow = os.environ.get('PIT_ALLOW_SHARDED_GPUS', '').strip().lower() in (
        '1', 'true', 'yes', 'on',
    )
    if force and not allow:
        raise SystemExit(
            'PIT_ZERO3 set but allow_sharded_gpus/PIT_ALLOW_SHARDED_GPUS is false '
            '(fail closed — refuse host-v834f without PROMOTION-GATE)'
        )
    low = False
    if not force:
        try:
            mem = os.environ.get('PIT_PER_GPU_MEM_GIB') or os.environ.get('CUDA_VISIBLE_MEM_GIB')
            if mem is not None:
                low = float(mem) < 100.0
        except ValueError:
            low = False
    want_sharded = bool(force or low) and allow
    if want_sharded:
        if not v834f.is_dir():
            raise SystemExit('host-v834f required for sharded mode but missing (fail closed)')
        return v834f
    if not v834e.is_dir():
        raise SystemExit('host-v834e missing (fail closed)')
    return v834e


HOST = _select_host()
IN = Path('/workspace/pst/in').resolve(); OUT = Path('/workspace/pst/out').resolve()   # /workspace is a symlink on Lium nodes: runtime requires an ordinary absolute dir
def ev(event, **kw): print(json.dumps(dict(event=event, seconds=round(time.time() - T, 1), **kw), default=str), flush=True)
exp = os.environ.get('AFFINE_FP32_EXPECTED_MASTER_HASH') or ''
assert len(exp) == 64, 'expected master hash required (fail closed)'
assert not os.environ.get('AFFINE_FP32_BOOTSTRAP_OVERRIDE'), 'ROOT bootstrap override forbidden in PIT'
os.chdir(SRC); sys.path.insert(0, str(SRC)); sys.path.insert(0, str(HOST))
spec = importlib.util.spec_from_file_location('mimo_host_trainer_v83', HOST / 'mimo_host_trainer_v83.py')
host = importlib.util.module_from_spec(spec); sys.modules[spec.name] = host; spec.loader.exec_module(host)
assert host.FP32_BOOTSTRAP_OVERRIDE is None
# Fail closed: ROOT-admitted host-file digests + fp32 settings + ROOT signature.
cfg = host.checked(HOST / 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json')
ev('root-admission-checked', engine_fp32=cfg['engine'].get('sgd_fp32_master'), host=str(HOST))
rd = lambda n: json.loads((IN / n).read_text())
head = {k: v for k, v in rd('head.descriptor.json').items() if k in ('id', 'files')}
manifest = rd('manifest.json'); dispatch = rd('dispatch.json'); tcfg = rd('train_config.json'); step = int(rd('meta.json')['optimizer_step'])
assert dispatch['signature'] and dispatch['signer'] != 'SHADOW-UNSIGNED'
# ZeRO-3: pass host so wrap_host_install runs (helpers via v835_zero3 entry).
z3 = None
if HOST.name == 'host-v834f' and (HOST / 'v835_zero3.py').is_file():
    zspec = importlib.util.spec_from_file_location('v835_zero3', HOST / 'v835_zero3.py')
    z3 = importlib.util.module_from_spec(zspec); sys.modules['v835_zero3'] = z3; zspec.loader.exec_module(z3)
    pol = z3.activate(host)
    if pol.get('zero3'):
        eng = z3.apply_low_vram_token_budget(cfg['engine'], pol.get('per_gpu_mem_gib'))
        cfg = dict(cfg, engine=eng)
    ev('zero3-policy', **{k: pol[k] for k in pol})
info = host.install(cfg, start_dp=True); ev('native-installed', dp_world=info.get('dp_world') if isinstance(info, dict) else None,
                                           zero3=(info or {}).get('zero3') if isinstance(info, dict) else None)
import torch
from subnet.mimo_runtime import MimoRuntime
from subnet.mimo_grpo.config import TrainerConfig
from subnet.mimo_grpo.optimizer import CPUAdamW
import subnet.mimo_training_worker as worker
config = TrainerConfig(**tcfg['settings']); assert config.document() == tcfg
runtime = MimoRuntime.load(str(IN / 'actor' / head['id']), head, manifest['payload']['mimo_runtime'], device='cuda', trainable=True); ev('actor-loaded')
world = int(cfg['engine'].get('dp_world') or 8)
token_budget = int(cfg['engine'].get('dp_token_budget') or cfg['engine'].get('token_budget') or 69632)
if z3 is not None and (z3.zero3_enabled() or z3.needs_zero3()):
    try:
        pref = z3.preflight_for_runtime(runtime, world=world, token_budget=token_budget)
        ev('vram-preflight-ok', **{k: pref[k] for k in pref})
    except z3.VramPreflightError as ex:
        ev('vram-preflight-refuse', error=str(ex)[:300])
        raise
    # Every rank wraps together (helpers handle op=fsdp-init). Never wrap rank-0 alone.
    runtime = z3.prepare_wrapped_model(runtime, local_rank=0)
    ev('fsdp-wrapped', sharding='FULL_SHARD', world=world)
    named = z3.canonical_named_parameters(runtime.model)
else:
    named = list(runtime.model.named_parameters())
opt = CPUAdamW(named, config, input_checkpoint=head['id']); opt.global_step = step; ev('optimizer-built', step=step)
prepared = worker.prepare_signed_dispatch(manifest, dispatch, authority=cfg['authority'], documents_root=str(IN / 'docs'), runtime=runtime, optimizer_step=step)
groups, binding = prepared['groups'], prepared['binding']
mr = max(len(t.tokens) for g in groups for t in g.trajectories); mc = max(len(g.prompt) + len(t.tokens) for g in groups for t in g.trajectories)
ev('signed-dispatch-prepared', groups=len(groups), rollouts=sum(len(g.trajectories) for g in groups), max_response_tokens=mr, max_context_tokens=mc)
e = cfg['engine']
# Use the (possibly cut) engine token_budget — not a stale pre-cut value.
report = sys.modules['v824_batched'].train_update_batched(
    runtime, opt, groups, binding, config=config, reference_runtime=None,
    engine=e['batched_engine'], token_budget=e['token_budget'], max_rows=e['max_rows'],
)
ev('native-grpo-complete', status=report.get('status'), optimizer_step=report.get('optimizer_step'), trained_rollouts=report.get('trained_rollouts'), trained_tokens=report.get('trained_tokens'),
   preparation_seconds=report.get('preparation_seconds'), optimizer_seconds=report.get('optimizer_seconds'), preclip=report.get('preclip_gradient_norm'), loss=report.get('objective_loss'))
OUT.mkdir(parents=True, exist_ok=True)
# save_model is patched under ZeRO-3 to emit host-v834e canonical keys.
off = sys.modules['v8210_optimizer_offpath']; staged = off.stage_for_publication_model_only(runtime, opt, report, str(OUT), workers=e['stage_workers']); staged = staged[0] if isinstance(staged, tuple) else staged
(OUT / 'descriptor.json').write_text(json.dumps(dict(id=staged['output_checkpoint'], files=staged['checkpoint_files'])))
(OUT / 'report.private.json').write_text(json.dumps(report, default=str))
ev('staged', output_checkpoint=staged['output_checkpoint'])
host._PUBLISHED.set()
with opt._v833f_lock: pass
fp = report['optimizer']['sgd']['fp32_master']
assert fp['source'] in ('disk', 'memory'), 'fp32 source %s' % fp['source']
assert fp['input_master_digest'] == exp, 'input master digest mismatch'
ev('done', checkpoint=staged['output_checkpoint'], fp32=dict(source=fp['source'], input_master_digest=fp['input_master_digest'], output_master_digest=fp.get('output_master_digest'),
   input_bf16=fp.get('input_bf16_sha256'), output_bf16=fp.get('output_bf16_sha256')), torch_max_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0,
   host=HOST.name, token_budget=e.get('token_budget'), dp_token_budget=e.get('dp_token_budget'))
try: sys.modules['v83_dp']._bcast(dict(op='stop'))
except Exception: pass
os._exit(0)
