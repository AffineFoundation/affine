"""PIT test1 (per-item train+eval, SHADOW/TEST ONLY): one native GRPO step on the box, no coordinator, no signer seed.
Inputs under /workspace/pst/in (signed TRAINING-JOB manifest + signed TRAIN-DISPATCH-FF + docs + head + fp32 master in FP32_STORE).
Fail closed: AFFINE_FP32_EXPECTED_MASTER_HASH must be set; no bootstrap override is allowed; the trainer's own _fp32_prepare checks it.

host-v834f: on <100 GiB GPUs enables ZeRO-3 (FSDP FULL_SHARD + CPU fp32 accum) shipped in the trainer-host
bundle (v835_zero3.py). Preflight refuses when projected per-GPU peak exceeds VRAM.
"""
import hashlib, json, os, sys, time, importlib.util
from pathlib import Path
T = time.time()
W = Path('/workspace/affine-mimo-trainer-prod'); SRC = W / 'source-final-v823u'
# Prefer host-v834f (ZeRO-3); fall back to host-v834e for older ships.
HOST = W / 'host-v834f' if (W / 'host-v834f').is_dir() else W / 'host-v834e'
IN = Path('/workspace/pst/in').resolve(); OUT = Path('/workspace/pst/out').resolve()   # /workspace is a symlink on Lium nodes: runtime requires an ordinary absolute dir
def ev(event, **kw): print(json.dumps(dict(event=event, seconds=round(time.time() - T, 1), **kw), default=str), flush=True)
exp = os.environ.get('AFFINE_FP32_EXPECTED_MASTER_HASH') or ''
assert len(exp) == 64, 'expected master hash required (fail closed)'
assert not os.environ.get('AFFINE_FP32_BOOTSTRAP_OVERRIDE'), 'ROOT bootstrap override forbidden in PIT'
os.chdir(SRC); sys.path.insert(0, str(SRC)); sys.path.insert(0, str(HOST))
spec = importlib.util.spec_from_file_location('mimo_host_trainer_v83', HOST / 'mimo_host_trainer_v83.py')
host = importlib.util.module_from_spec(spec); sys.modules[spec.name] = host; spec.loader.exec_module(host)
assert host.FP32_BOOTSTRAP_OVERRIDE is None
# PIT ships host-v834f overlays; verify on-disk host file digests when public config is present,
# but do not require a fresh ROOT admission signature for the shadow/test path.
cfg_path = HOST / 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json'
pub = HOST / 'trainer-config.public.json'
pub_doc = json.loads(pub.read_text()) if pub.is_file() else {}
if pub_doc:
    digs = pub_doc.get('host_files_sha256') or {}
    for rel, expect in digs.items():
        p = HOST / rel
        if not p.is_file():
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == expect, 'host file digest mismatch %s' % rel
    ev('host-public-digests-ok', host_version=pub_doc.get('host_version'), n=len(digs))
cfg = host.private_json(cfg_path) if cfg_path.is_file() else host.checked(cfg_path)
# Prefer public engine overlays (dp_token_budget etc.) when the shipped public config is newer.
if isinstance(pub_doc.get('engine'), dict):
    eng = dict(cfg['engine'])
    for k in ('dp_token_budget', 'dp_max_rows', 'dp_world', 'dp_port', 'token_budget', 'max_rows',
              'sgd_fp32_master', 'sgd_mode', 'sgd_lr', 'sgd_update_norm_cap', 'sgd_direction',
              'sgd_tensor_rel_cap', 'batched_engine', 'training_kernels', 'loss_agg_mode', 'loss_scale',
              'drop_unresolved', 'fast_stage', 'stage_workers', 'optimizer_persistence',
              'optimizer_step_threads', 'part_concurrency', 'shard_concurrency', 'sgd_freeze'):
        if k in pub_doc['engine']:
            eng[k] = pub_doc['engine'][k]
    cfg = dict(cfg, engine=eng)
ev('root-admission-checked', engine_fp32=cfg['engine'].get('sgd_fp32_master'), host=str(HOST))
rd = lambda n: json.loads((IN / n).read_text())
head = {k: v for k, v in rd('head.descriptor.json').items() if k in ('id', 'files')}
manifest = rd('manifest.json'); dispatch = rd('dispatch.json'); tcfg = rd('train_config.json'); step = int(rd('meta.json')['optimizer_step'])
assert dispatch['signature'] and dispatch['signer'] != 'SHADOW-UNSIGNED'
# Activate ZeRO-3 env before install so helpers inherit PIT_ZERO3.
z3 = None
if (HOST / 'v835_zero3.py').is_file():
    zspec = importlib.util.spec_from_file_location('v835_zero3', HOST / 'v835_zero3.py')
    z3 = importlib.util.module_from_spec(zspec); sys.modules['v835_zero3'] = z3; zspec.loader.exec_module(z3)
    pol = z3.activate(None)
    ev('zero3-policy', **{k: pol[k] for k in pol})
info = host.install(cfg, start_dp=True); ev('native-installed', dp_world=info.get('dp_world') if isinstance(info, dict) else None, zero3=(info or {}).get('zero3') if isinstance(info, dict) else None)
import torch
from subnet.mimo_runtime import MimoRuntime
from subnet.mimo_grpo.config import TrainerConfig
from subnet.mimo_grpo.optimizer import CPUAdamW
import subnet.mimo_training_worker as worker
config = TrainerConfig(**tcfg['settings']); assert config.document() == tcfg
runtime = MimoRuntime.load(str(IN / 'actor' / head['id']), head, manifest['payload']['mimo_runtime'], device='cuda', trainable=True); ev('actor-loaded')
# VRAM preflight + FSDP wrap BEFORE the optimizer captures parameter identities.
world = int(cfg['engine'].get('dp_world') or 8)
token_budget = int(cfg['engine'].get('dp_token_budget') or cfg['engine'].get('token_budget') or 69632)
if z3 is not None:
    z3.patch_fp32_accumulator()
    try:
        pref = z3.preflight_for_runtime(runtime, world=world, token_budget=token_budget)
        ev('vram-preflight-ok', **{k: pref[k] for k in pref})
    except z3.VramPreflightError as ex:
        ev('vram-preflight-refuse', error=str(ex)[:300])
        raise
    if z3.zero3_enabled() or z3.needs_zero3():
        runtime.model = z3.wrap_fsdp(runtime.model, local_rank=0)
        ev('fsdp-wrapped', sharding='FULL_SHARD')
opt = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=head['id']); opt.global_step = step; ev('optimizer-built', step=step)
prepared = worker.prepare_signed_dispatch(manifest, dispatch, authority=cfg['authority'], documents_root=str(IN / 'docs'), runtime=runtime, optimizer_step=step)
groups, binding = prepared['groups'], prepared['binding']
mr = max(len(t.tokens) for g in groups for t in g.trajectories); mc = max(len(g.prompt) + len(t.tokens) for g in groups for t in g.trajectories)
ev('signed-dispatch-prepared', groups=len(groups), rollouts=sum(len(g.trajectories) for g in groups), max_response_tokens=mr, max_context_tokens=mc)
e = cfg['engine']
report = sys.modules['v824_batched'].train_update_batched(runtime, opt, groups, binding, config=config, reference_runtime=None, engine=e['batched_engine'], token_budget=e['token_budget'], max_rows=e['max_rows'])
ev('native-grpo-complete', status=report.get('status'), optimizer_step=report.get('optimizer_step'), trained_rollouts=report.get('trained_rollouts'), trained_tokens=report.get('trained_tokens'),
   preparation_seconds=report.get('preparation_seconds'), optimizer_seconds=report.get('optimizer_seconds'), preclip=report.get('preclip_gradient_norm'), loss=report.get('objective_loss'))
OUT.mkdir(parents=True, exist_ok=True)
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
   input_bf16=fp.get('input_bf16_sha256'), output_bf16=fp.get('output_bf16_sha256')), torch_max_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
   host_version='host-v834f' if HOST.name == 'host-v834f' else HOST.name)
try: sys.modules['v83_dp']._bcast(dict(op='stop'))
except Exception: pass
os._exit(0)
