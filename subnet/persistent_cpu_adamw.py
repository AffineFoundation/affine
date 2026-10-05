"""Prospective FP32 CPU optimizer state for exact BF16 inference weights.

No active policy selects this module. The caller authenticates genesis/parent
descriptors and fully verifies training pairs before constructing this object.
CPU tensors keep sub-BF16 updates across epochs; GPU parameters are projections.
"""
import hashlib
import math
import copy
import threading
from contextlib import contextmanager

from .storage import canonical

POLICY = 'bf16-cpu-fp32-master-task-normalized-persistent-v4'
HYPERPARAMETERS = dict(lr=1e-5, betas=[.9, .999], eps=1e-8, weight_decay=.01,
                     max_grad_norm=1., preference_beta=.1)
GENESIS_VERSION = 'explicit-fp32-master-genesis-v1'


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def checkpoint_id(value):
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('exact checkpoint/state SHA256 required')
    return value


def parameter_inventory(named_parameters):
    parameters = list(named_parameters)
    names = [name for name, _ in parameters]
    if (not parameters or len(parameters) > 4096 or len(set(names)) != len(names) or
            len({id(p) for _, p in parameters}) != len(parameters) or
            any(not isinstance(n, str) or not n or len(n) > 1024 for n in names)):
        raise ValueError('nonempty unique parameter names required')
    inventory = []
    for name, parameter in parameters:
        shape = list(parameter.shape)
        if parameter.numel() < 1:
            raise ValueError('nonempty parameter tensor required')
        inventory.append(dict(name=name, shape=shape, numel=parameter.numel()))
    return parameters, inventory


def genesis(inventory, input_checkpoint):
    """Caller must sign this explicit start/reset authorization in a new job."""
    return dict(version=GENESIS_VERSION, policy=POLICY,
                hyperparameters=copy.deepcopy(HYPERPARAMETERS),
                parameters_sha256=sha(inventory),
                input_checkpoint=checkpoint_id(input_checkpoint),
                explicit_optimizer_genesis=True)


def finite(torch, tensor, *, nonnegative=False):
    # Bounded checking avoids an additional full-size boolean tensor.
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), 1_000_000):
        part = flat[start:start + 1_000_000]
        if not bool(torch.isfinite(part).all()):
            raise ValueError('nonfinite persistent training state/gradient')
        if nonnegative and not bool((part >= 0).all()):
            raise ValueError('negative second optimizer moment')


class PersistentCPUAdamW:
    def __init__(self, named_parameters, input_checkpoint, *,
                 approved_genesis=None, approved_genesis_sha256=None,
                 restored=None, resource_admission=None):
        import torch
        self.parameters, self.inventory = parameter_inventory(named_parameters)
        if any(parameter.dtype != torch.bfloat16 for _, parameter in self.parameters):
            raise ValueError('exact BF16 inference parameter profile')
        self.input_checkpoint = checkpoint_id(input_checkpoint)
        self.hyperparameters = copy.deepcopy(HYPERPARAMETERS)
        self.rows = {}; self.global_step = 0
        self._state_lock = threading.RLock(); self._publishing = False
        self.parent_state_sha256 = None; self.genesis_sha256 = None
        if (restored is None) == (approved_genesis is None):
            raise ValueError('one authenticated parent state or explicit genesis required')
        if restored is not None:
            descriptor, rows, approved_parent_sha256 = restored
            from .persistent_training_state import validate_descriptor
            validate_descriptor(descriptor, approved_parent_sha256,
                                self.input_checkpoint, self.inventory)
            self.parent_state_sha256 = approved_parent_sha256
            self.genesis_sha256 = descriptor['genesis_sha256']
            self.global_step = descriptor['optimizer_steps']
            self.rows = rows
            if set(rows) != {r['name'] for r in self.inventory}:
                raise ValueError('restored parameter state names')
        else:
            expected = genesis(self.inventory, self.input_checkpoint)
            if (canonical(approved_genesis) != canonical(expected) or
                    sha(approved_genesis) != approved_genesis_sha256):
                raise ValueError('signed explicit genesis binding')
            from .persistent_training_state import resource_plan, admit_resources
            if not resource_admission or resource_admission.get('admitted') is not True:
                raise ValueError('actual RAM/disk admission before genesis allocation')
            plan = resource_plan(self.inventory,
                bf16_export_bytes=resource_admission['bf16_export_bytes'],
                transfer_bytes=resource_admission['bounded_transfer_bytes'],
                disk_reserve_bytes=resource_admission['disk_reserve_bytes'],
                ram_reserve_bytes=resource_admission['ram_reserve_bytes'],
                concurrency=resource_admission.get('state_transfer_concurrency',1))
            if any(resource_admission.get(k) != v for k, v in plan.items()):
                raise ValueError('genesis resource/inventory binding')
            admit_resources(resource_admission['workspace'], plan)
            self.genesis_sha256 = approved_genesis_sha256
            for name, parameter in self.parameters:
                self.rows[name] = dict(master=parameter.detach().to('cpu', torch.float32).clone(),
                    exp_avg=torch.zeros_like(parameter, device='cpu', dtype=torch.float32),
                    exp_avg_sq=torch.zeros_like(parameter, device='cpu', dtype=torch.float32),
                    step=0)
        for name, parameter in self.parameters:
            if parameter.dtype != torch.bfloat16:
                raise ValueError('exact BF16 inference parameter profile')
            row = self.rows[name]
            if set(row) != {'master', 'exp_avg', 'exp_avg_sq', 'step'}:
                raise ValueError('optimizer row fields')
            if type(row['step']) is not int or row['step'] != self.global_step:
                raise ValueError('global/per-parameter optimizer counter binding')
            for slot in ('master', 'exp_avg', 'exp_avg_sq'):
                value = row[slot]
                if (value.device.type != 'cpu' or value.dtype != torch.float32 or
                        list(value.shape) != list(parameter.shape) or not value.is_contiguous()):
                    raise ValueError('CPU FP32 parameter state shape/dtype')
                finite(torch, value, nonnegative=slot == 'exp_avg_sq')
            if not torch.equal(row['master'].to(torch.bfloat16), parameter.detach().cpu()):
                raise ValueError('master projection does not equal signed BF16 input model')

    def zero_grad(self):
        for _, parameter in self.parameters:
            parameter.grad = None

    @contextmanager
    def freeze_for_publication(self):
        with self._state_lock:
            if self._publishing:
                raise ValueError('refuse overlapping state publications')
            self._publishing = True
            try:
                yield
            finally:
                self._publishing = False

    def step(self):
        with self._state_lock:
            if self._publishing:
                raise ValueError('refuse optimizer update during state publication')
            return self._step_impl()

    def _step_impl(self):
        """All gradients must already be clipped by the signed trainer rule."""
        import torch
        if canonical(self.hyperparameters) != canonical(HYPERPARAMETERS):
            raise ValueError('immutable optimizer hyperparameters')
        if self.global_step >= 2**31 - 1:
            raise ValueError('optimizer counter limit')
        for _, parameter in self.parameters:
            if parameter.grad is None or parameter.grad.is_sparse:
                raise ValueError('every full-model parameter requires a dense gradient')
            finite(torch, parameter.grad)
        step = self.global_step + 1
        h = self.hyperparameters; beta1, beta2 = h['betas']
        diagnostics = []
        with torch.no_grad():
            for name, parameter in self.parameters:
                row = self.rows[name]
                # One parameter's gradient and denominator at a time. No full
                # FP32 model gradient replica is retained on the CPU or GPU.
                gradient = parameter.grad.detach().to('cpu', torch.float32)
                before_master = row['master'].clone()
                before_bf16 = parameter.detach().cpu().clone()
                row['exp_avg'].lerp_(gradient, 1 - beta1)
                row['exp_avg_sq'].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                finite(torch, row['exp_avg'])
                finite(torch, row['exp_avg_sq'], nonnegative=True)
                denominator = row['exp_avg_sq'].sqrt()
                denominator.div_(math.sqrt(1 - beta2**step)).add_(h['eps'])
                row['master'].mul_(1 - h['lr'] * h['weight_decay'])
                row['master'].addcdiv_(row['exp_avg'], denominator,
                                      value=-h['lr'] / (1 - beta1**step))
                finite(torch, row['master'])
                projected = row['master'].to(torch.bfloat16)
                delta = row['master'] - before_master
                diagnostics.append(dict(name=name, elements=parameter.numel(),
                    master_changed_elements=int((row['master'] != before_master).sum()),
                    bf16_changed_elements=int((projected != before_bf16).sum()),
                    master_delta_l2=float(torch.linalg.vector_norm(delta.reshape(-1))),
                    master_delta_max_abs=float(delta.abs().max())))
                parameter.copy_(projected.to(device=parameter.device))
                row['step'] = step
                del gradient, denominator, before_master, before_bf16, projected, delta
        self.global_step = step
        self.last_update = dict(optimizer_step=step, parameters=diagnostics,
            master_changed_elements=sum(r['master_changed_elements'] for r in diagnostics),
            bf16_changed_elements=sum(r['bf16_changed_elements'] for r in diagnostics),
            optimizer_state_dtype='torch.float32', master_dtype='torch.float32',
            inference_dtype='torch.bfloat16')
        return self.last_update
