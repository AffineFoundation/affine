"""CPU test of the v8.2.10 host layer over the exact source closure (no GPU, no network, no services).
Runs the real patched service.execute_transaction (through the installed worker wrapper), the real
Transaction journal, retention and the (wrapped) cold-recovery function with a tiny bf16 model.
usage: test_v8210_cpu.py <host_dir> <source_root> <mode none|synchronous> <tmp> [legacy_dir]"""
import sys, json, os, shutil
sys.dont_write_bytecode = True
from pathlib import Path
from dataclasses import dataclass, asdict
HOST, SRC, MODE, TMP = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
LEGACY = Path(sys.argv[5]) if len(sys.argv) > 5 else None
os.chdir(SRC); sys.path.insert(0, str(SRC))
import importlib.util
spec = importlib.util.spec_from_file_location('mimo_host_trainer_v8210', HOST / 'mimo_host_trainer_v8210.py')
L = importlib.util.module_from_spec(spec); sys.modules['mimo_host_trainer_v8210'] = L; spec.loader.exec_module(L)
import subnet.mimo_grpo.transaction as T, subnet.mimo_grpo.publication as PB, subnet.mimo_grpo.service as SV
import subnet.mimo_grpo.optimizer as OP, subnet.mimo_grpo.recovery as RC, subnet.mimo_training_worker as W
objs = {'mimo_grpo.transaction._receipt': T._receipt, 'mimo_grpo.transaction.Transaction.run': T.Transaction.run,
        'mimo_grpo.publication.validate_publication': PB.validate_publication,
        'mimo_grpo.service.execute_transaction': SV.execute_transaction,
        'mimo_training_worker.validate_result': W.validate_result, 'mimo_grpo.optimizer.CPUAdamW.step': OP.CPUAdamW.step,
        'mimo_grpo.publication.R2ModelPublisher.__init__': PB.R2ModelPublisher.__init__,
        'mimo_grpo.recovery.restore_transaction_state': RC.restore_transaction_state}
engine = dict(batched_engine='padded', token_budget=16384, max_rows=64, fast_stage=True, stage_workers=4,
              part_concurrency=16, shard_concurrency=8, optimizer_step_threads=4, drop_unresolved=True,
              optimizer_persistence=MODE)
cfg = dict(original_methods={k: L.method_sha(v) for k, v in objs.items()}, engine=engine)
print(json.dumps(L.install(cfg)), flush=True)
import torch
from safetensors.torch import save_file, load_file
from subnet.mimo_grpo.config import TrainerConfig
from subnet.mimo_grpo.data import digest, InputBinding
from subnet.mimo_grpo.transaction import TRAINER, Job
import v824_batched
config = TrainerConfig()

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); g = torch.Generator().manual_seed(0)
        self.w = torch.nn.Parameter(torch.randn(64, 48, generator=g).to(torch.bfloat16))
        self.b = torch.nn.Parameter(torch.randn(48, generator=g).to(torch.bfloat16))

class Runtime:
    def __init__(self, checkpoint, model=None):
        self.checkpoint_id, self.model, self.closed = checkpoint, model or Model(), False
    def save_model(self, directory):
        directory.mkdir(parents=True)
        (directory / 'config.json').write_text('{"tiny": true}')
        save_file({n: p.detach().contiguous() for n, p in self.model.named_parameters()}, str(directory / 'model.safetensors'))
    def guard(self): pass
    def acknowledge_model_publication(self, output, files): self.checkpoint_id = output
    def close(self): self.closed = True

def load_runtime(directory, descriptor):
    m = Model(); data = load_file(str(directory / 'model.safetensors'))
    with torch.no_grad():
        for n, p in m.named_parameters(): p.copy_(data[n])
    files = {p.name: OP._hash_file(p) for p in sorted(directory.iterdir()) if p.is_file()}
    assert digest(files) == descriptor['id'] and files == descriptor['files']
    return Runtime(descriptor['id'], m)

@dataclass(frozen=True)
class Group:
    task: str

def fake_train(runtime, optimizer, groups, binding, *, config, guard, record, reference_runtime, engine, token_budget, max_rows):
    g = torch.Generator().manual_seed(optimizer.global_step + 7)
    grads = {n: torch.randn(p.shape, generator=g) * 1e-2 for n, p in optimizer.parameters}
    info = optimizer.step(grads)
    return dict(version=TRAINER, status='updated', input=asdict(binding), input_checkpoint=binding.checkpoint,
                optimizer_step=optimizer.global_step, config=config.document(), optimizer=info)
v824_batched.train_update_batched = fake_train

published = []
def publish(request):
    published.append(request)
    assert request['optimizer_state_uploaded'] is False and all(not n.startswith('optimizer') for n in request['model_files'])
    return dict(job_id=request['job_id'], output_checkpoint=request['output_checkpoint'], model_full_get_verified=False,
                model_upload_verification='streamed-sha256+part-content-md5+head-length', optimizer_state_uploaded=False,
                receipt_sha256='a' * 64)

TMP.mkdir(parents=True, exist_ok=True); os.chmod(TMP, 0o700)
history = TMP / 'run' / 'history'; history.mkdir(parents=True, mode=0o700); os.chmod(TMP / 'run', 0o700)
ck0 = 'c' * 64
runtime = Runtime(ck0)
optimizer = OP.CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=ck0)
results = []
checkpoint, step = ck0, 0
import time
for epoch in range(3):
    binding = InputBinding(run='run', epoch=f'e{epoch}', checkpoint=checkpoint, contract_sha256='d' * 64, optimizer_step=step)
    folder = history / ('update-' + digest(dict(epoch=binding.epoch, checkpoint=binding.checkpoint)))
    t = time.time()
    r = W.execute_transaction(folder, runtime, optimizer, [Group(f't{epoch}{i}') for i in range(2)], binding,
        manifest_sha256='e' * 64, source_closure_sha256='f' * 64, assert_trainer_lease=lambda: None,
        publish_model=publish, config=config)
    opt_files = sorted(p.name for p in (folder / 'optimizer').iterdir())
    exp_avg_nonzero = all(bool(optimizer.optimizer.state[m]['exp_avg'].abs().sum() > 0) for m in optimizer.masters.values())
    results.append(dict(epoch=epoch, state=r['state'], step=optimizer.global_step, output=r['output_checkpoint'][:12],
                        optimizer_files=len(opt_files), marker=opt_files == ['NOT-PERSISTED.json'],
                        in_memory_moments_nonzero=exp_avg_nonzero, seconds=round(time.time() - t, 3)))
    assert r['state'] == 'published' and optimizer.global_step == step + 1 and runtime.checkpoint_id == r['output_checkpoint']
    staged = json.loads((folder / 'STAGED.json').read_text())
    assert set(staged) == {'version', 'input_checkpoint', 'output_checkpoint', 'checkpoint_files', 'optimizer_step', 'report_sha256',
                           'optimizer_descriptor_sha256', 'publish_files', 'local_state_files', 'retain_committed_generations',
                           'max_local_optimizer_generations'}
    if MODE == 'none':
        assert opt_files == ['NOT-PERSISTED.json'] and list(staged['local_state_files']) == ['optimizer/NOT-PERSISTED.json']
    else:
        assert 'descriptor.json' in opt_files and len(opt_files) > 2
    assert exp_avg_nonzero
    checkpoint, step = r['output_checkpoint'], optimizer.global_step
    last_job, last_folder = r['job'], folder
# retention: older generations lost their optimizer dirs (marker or shards), current kept
retained = {p.name[:15]: (p / 'optimizer').exists() for p in history.iterdir()}
assert sum(retained.values()) == 1, retained
print(json.dumps(dict(mode=MODE, epochs=results, retained_optimizer_dirs=retained)), flush=True)
# same model bytes regardless of mode (determinism check across the two processes)
print(json.dumps(dict(mode=MODE, final_output_checkpoint=checkpoint)), flush=True)
# cold recovery of the current generation through the worker-bound function
job = dict(last_job); job['prepared_documents_sha256'] = tuple(job['prepared_documents_sha256']); job = Job(**job)
live = {n: (optimizer.masters[n].clone(), optimizer.optimizer.state[optimizer.masters[n]]['exp_avg'].clone()) for n in optimizer.masters}
rt, op, val = W.restore_transaction_state(last_folder, expected_job=job, load_runtime=load_runtime, config=config)
masters_match = all(torch.equal(op.masters[n], live[n][0].to(torch.bfloat16).float()) for n in op.masters)
moments = {n: float(op.optimizer.state[op.masters[n]]['exp_avg'].abs().sum()) for n in op.masters}
steps = {int(op.optimizer.state[m]['step']) for m in op.masters.values()}
info = dict(mode=MODE, restored_step=op.global_step, state_steps=sorted(steps), restored_checkpoint=rt.checkpoint_id[:12],
            masters_equal_bf16_upcast=masters_match, restored_exp_avg_abs_sum=moments)
if MODE == 'none':
    assert op.global_step == step and steps == {step} and all(v == 0 for v in moments.values()) and masters_match
else:
    assert op.global_step == step and all(torch.equal(op.optimizer.state[op.masters[n]]['exp_avg'], live[n][1]) for n in op.masters)
# the restored optimizer can take the next step
g = {n: torch.randn(p.shape) * 1e-2 for n, p in op.parameters}
op.step(g); info['post_restore_step'] = op.global_step
assert op.global_step == step + 1
print(json.dumps(info), flush=True)
if MODE == 'synchronous':
    # keep a legacy (full-state) generation for the 'none' process to restore through the wrapper
    dest = TMP.parent / 'legacy-generation'
    shutil.copytree(last_folder, dest); (dest / 'JOB.json').write_text(json.dumps(last_job))
    torch.save({n: live[n][1] for n in live}, str(TMP.parent / 'legacy-exp_avg.pt'))
if MODE == 'none' and LEGACY is not None:
    job = json.loads((LEGACY / 'JOB.json').read_text()); job['prepared_documents_sha256'] = tuple(job['prepared_documents_sha256'])
    os.chmod(LEGACY, 0o700)
    rt, op, val = W.restore_transaction_state(LEGACY, expected_job=Job(**job), load_runtime=load_runtime, config=config)
    ref = torch.load(str(LEGACY.parent / 'legacy-exp_avg.pt'))
    exact = all(torch.equal(op.optimizer.state[op.masters[n]]['exp_avg'], ref[n]) for n in op.masters)
    assert exact and op.global_step == 3
    print(json.dumps(dict(mode=MODE, legacy_generation_restored_exactly_via_wrapper=exact, step=op.global_step)), flush=True)
print('ALL-CHECKS-PASSED', MODE, flush=True)
