"""Persistent FP32 CPU AdamW with bounded parameter-at-a-time transfers.

Uses PyTorch's actual AdamW implementation (foreach=False), not a re-derived
optimizer. BF16 actor tensors are projections of persistent FP32 master tensors.
One logical step advances each parameter once. No optimizer state is published
to the model bucket. The enclosing transaction owns local retention and leases.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import torch

from .config import TrainerConfig, MCORE_DECAY
from .data import digest, sha


def _hash_file(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def _write_json(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


class FP32Accumulator:
    def __init__(self, named_parameters):
        self.parameters = tuple((name, p) for name, p in named_parameters if p.requires_grad)
        if not self.parameters or len({name for name, _ in self.parameters}) != len(self.parameters):
            raise ValueError('unique nonempty trainable parameter inventory')
        if len({id(p) for _, p in self.parameters}) != len(self.parameters):
            raise ValueError('duplicate trainable parameter object')
        self.buffers = {name: torch.zeros_like(p, dtype=torch.float32) for name, p in self.parameters}
        self.microsteps = 0
        self.clipped = False

    def capture(self):
        if self.clipped:
            raise ValueError('gradient accumulator already clipped')
        for name, parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None or gradient.is_sparse or gradient.shape != parameter.shape:
                raise ValueError('missing/dense gradient inventory: ' + name)
        with torch.no_grad():
            for name, parameter in self.parameters:
                self.buffers[name].add_(parameter.grad)
                parameter.grad = None
        self.microsteps += 1

    def clip(self, maximum):
        if self.clipped or not self.microsteps:
            raise ValueError('one clip after nonempty accumulation')
        # The norm is computed before clipping and reported, never an after-clip
        # norm mislabeled as the signal needed to diagnose instability.
        norms = [torch.linalg.vector_norm(g, dtype=torch.float32) for g in self.buffers.values()]
        total = torch.linalg.vector_norm(torch.stack(norms), dtype=torch.float32)
        if not torch.isfinite(total):
            raise ValueError('nonfinite accumulated gradient norm')
        scale = torch.clamp(maximum / (total + 1e-6), max=1.)
        with torch.no_grad():
            for gradient in self.buffers.values():
                gradient.mul_(scale)
        self.clipped = True
        return float(total)


class CPUAdamW:
    def __init__(self, named_parameters, config: TrainerConfig, *, input_checkpoint: str):
        config.validate()
        if not sha(input_checkpoint):
            raise ValueError('immutable input checkpoint required')
        self.config = config
        self.input_checkpoint = input_checkpoint
        self.parameters = tuple((n, p) for n, p in named_parameters if p.requires_grad)
        if not self.parameters or len({n for n, _ in self.parameters}) != len(self.parameters):
            raise ValueError('unique nonempty trainable inventory')
        if len({id(p) for _, p in self.parameters}) != len(self.parameters):
            raise ValueError('duplicate parameter objects')
        if any(p.dtype not in (torch.bfloat16, torch.float32) for _, p in self.parameters):
            raise ValueError('BF16 or FP32 actor parameters required')
        self.inventory = [{'name': n, 'shape': list(p.shape), 'numel': p.numel()} for n, p in self.parameters]
        self.masters = {n: torch.nn.Parameter(p.detach().to(device='cpu', dtype=torch.float32, copy=True)) for n, p in self.parameters}
        # NVIDIA Megatron Core _get_param_groups default: no decay for *.bias
        # or one-dimensional tensors. Embeddings remain decayed by default.
        # The policy is in the signed objective and durable state descriptor.
        self.decay_inventory = [dict(name=n, weight_decay=(0. if config.weight_decay_policy == MCORE_DECAY
            and (n.endswith('.bias') or p.ndim == 1) else config.weight_decay)) for n,p in self.parameters]
        grouped = {}
        for row in self.decay_inventory:
            grouped.setdefault(row['weight_decay'], []).append(self.masters[row['name']])
        parameter_groups = [dict(params=parameters, weight_decay=decay)for decay,parameters in grouped.items()]
        self.optimizer = torch.optim.AdamW(parameter_groups, lr=config.learning_rate,
            betas=(config.beta1, config.beta2), eps=config.adam_epsilon,
            weight_decay=config.weight_decay, foreach=False, fused=False)
        self.global_step = 0
        self.failed = False

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for _, parameter in self.parameters:
            parameter.grad = None

    def resource_requirements(self):
        count = sum(p.numel() for _, p in self.parameters)
        peak = max(p.numel() for _, p in self.parameters)
        return dict(cpu_persistent_bytes=count * 12, cpu_step_scratch_bound_bytes=peak * 20,
                    gpu_fp32_accumulator_bytes=count * 4, local_generation_payload_bytes=count * 12)

    def step(self, gradients):
        if self.failed or set(gradients) != set(self.masters):
            raise ValueError('healthy optimizer and exact gradient inventory required')
        # Validate everything BEFORE changing any parameter or moment.
        for name, parameter in self.parameters:
            g = gradients[name]
            if g.dtype != torch.float32 or g.shape != parameter.shape or not torch.isfinite(g).all():
                raise ValueError('finite FP32 accumulated gradients required: ' + name)
        before = self.global_step
        self.zero_grad()
        changed = 0
        try:
            # AdamW skips parameters with grad=None. This gives the same update
            # and per-parameter step as one full optimizer.step, while the host
            # holds at most one transferred FP32 gradient at a time.
            for name, parameter in self.parameters:
                master = self.masters[name]
                master.grad = gradients[name].detach().to('cpu', copy=True)
                self.optimizer.step()
                master.grad = None
                state = self.optimizer.state[master]
                if int(state['step']) != before + 1 or not torch.isfinite(master).all():
                    raise ValueError('optimizer state counter/finite update failed')
                projection = master.detach().to(device=parameter.device, dtype=parameter.dtype)
                changed += int((projection != parameter).sum())
                with torch.no_grad():
                    parameter.copy_(projection)
            self.global_step += 1
        except BaseException:
            # A partially updated in-memory actor is poisoned until recovery
            # from the previous local committed optimizer generation.
            self.failed = True
            raise
        finally:
            self.zero_grad()
        return dict(optimizer_step=self.global_step, implementation='torch.optim.AdamW',
            foreach=False, fused=False, lr=self.config.learning_rate,
            betas=[self.config.beta1, self.config.beta2], eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay, state_dtype='float32', master_dtype='float32',
            weight_decay_policy=self.config.weight_decay_policy,
            parameter_decay_inventory_sha256=digest(self.decay_inventory),
            decay_parameter_count=sum(row['weight_decay'] != 0 for row in self.decay_inventory),
            no_decay_parameter_count=sum(row['weight_decay'] == 0 for row in self.decay_inventory),
            gradient_accumulation_dtype='float32', projected_changed_elements=changed)

    def save_local(self, directory, *, output_checkpoint, shard_bytes=256 * 1024**2, artifact_hasher=None):
        """Write a new locally durable generation without a full state replica."""
        if self.failed or not sha(output_checkpoint) or shard_bytes < 4096:
            raise ValueError('valid committed candidate and bounded shard size')
        if artifact_hasher is not None and not callable(artifact_hasher):
            raise ValueError('original transaction artifact hasher required')
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        chunks = []
        elements = shard_bytes // 4
        for index, (name, _) in enumerate(self.parameters):
            master = self.masters[name]
            state = self.optimizer.state[master]
            if self.global_step and int(state['step']) != self.global_step:
                raise ValueError('all parameter moments at the same optimizer step')
            for slot in ('master', 'exp_avg', 'exp_avg_sq'):
                tensor = master.detach() if slot == 'master' else state.get(slot)
                if tensor is None:
                    tensor = torch.zeros_like(master)
                flat = tensor.reshape(-1)
                for start in range(0, flat.numel(), elements):
                    value = flat[start:start + elements].contiguous()
                    filename = f'{index:04d}-{slot}-{start:012d}.safetensors'
                    from safetensors.torch import save_file
                    save_file({'value': value}, str(root / filename))
                    with (root / filename).open('rb') as stream:
                        os.fsync(stream.fileno())
                    chunks.append(dict(file=filename, parameter=name, slot=slot, offset=start,
                        elements=value.numel(), bytes=(root / filename).stat().st_size,
                        sha256=(_hash_file(root / filename) if artifact_hasher is None
                                else artifact_hasher(root / filename)['sha256'])))
        descriptor = dict(version='affine-mimo-local-adamw-v1', optimizer_step=self.global_step,
            input_checkpoint=self.input_checkpoint, checkpoint=output_checkpoint,
            inventory=self.inventory, config=self.config.document(), chunks=chunks)
        _write_json(root / 'descriptor.json', descriptor)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return descriptor

    def restore_local(self, directory, *, descriptor_sha256, checkpoint):
        """Restore only an authenticated local descriptor; never unsafe pickle."""
        try:
            return self._restore_local(directory, descriptor_sha256=descriptor_sha256, checkpoint=checkpoint)
        except BaseException:
            # A late corrupt shard or model-projection failure may occur after
            # masters were partially filled. Never let that object step or be
            # retried; recovery must construct a fresh optimizer.
            self.failed = True
            raise

    def _restore_local(self, directory, *, descriptor_sha256, checkpoint):
        if self.global_step != 0 or self.optimizer.state or self.failed:
            raise ValueError('restore requires newly constructed optimizer')
        root = Path(directory)
        descriptor = json.loads((root / 'descriptor.json').read_text())
        if (digest(descriptor) != descriptor_sha256 or descriptor['checkpoint'] != checkpoint or
                descriptor['inventory'] != self.inventory or descriptor['config'] != self.config.document() or
                descriptor['version'] != 'affine-mimo-local-adamw-v1'):
            raise ValueError('authenticated checkpoint/config/state inventory mismatch')
        step = descriptor['optimizer_step']
        if type(step) is not int or step < 0:
            raise ValueError('optimizer step')
        slots = {}
        for name, master in self.masters.items():
            slots[(name, 'master')] = master.detach()
            for slot in ('exp_avg', 'exp_avg_sq'):
                slots[(name, slot)] = torch.empty_like(master)
        offsets = {key: 0 for key in slots}
        from safetensors.torch import load_file
        for chunk in descriptor['chunks']:
            filename = chunk['file']
            if Path(filename).name != filename or (root / filename).is_symlink():
                raise ValueError('local optimizer shard path')
            if (root / filename).stat().st_size != chunk['bytes'] or _hash_file(root / filename) != chunk['sha256']:
                raise ValueError('optimizer shard content mismatch')
            key = (chunk['parameter'], chunk['slot'])
            if key not in slots or chunk['offset'] != offsets[key]:
                raise ValueError('ordered nonoverlapping optimizer shards')
            data = load_file(str(root / filename), device='cpu')
            if set(data) != {'value'}:
                raise ValueError('optimizer shard tensor inventory')
            value = data['value']
            if value.dtype != torch.float32 or value.ndim != 1 or value.numel() != chunk['elements'] or not torch.isfinite(value).all():
                raise ValueError('finite FP32 optimizer shard')
            if key[1] == 'exp_avg_sq' and (value < 0).any():
                raise ValueError('nonnegative second moments')
            start = offsets[key]
            slots[key].reshape(-1)[start:start + value.numel()].copy_(value)
            offsets[key] += value.numel()
        if any(offsets[key] != tensor.numel() for key, tensor in slots.items()):
            raise ValueError('complete optimizer state required')
        for name, parameter in self.parameters:
            master = self.masters[name]
            if not torch.equal(master.detach().to(parameter.device, parameter.dtype), parameter.detach()):
                raise ValueError('master projection differs from authenticated input model')
            self.optimizer.state[master] = dict(step=torch.tensor(float(step)),
                exp_avg=slots[(name, 'exp_avg')], exp_avg_sq=slots[(name, 'exp_avg_sq')])
        self.global_step = step
        self.input_checkpoint = checkpoint
        return descriptor
