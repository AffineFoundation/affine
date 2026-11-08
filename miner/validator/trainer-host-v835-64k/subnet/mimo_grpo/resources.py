"""Observed, cgroup-aware admission before full-model state allocation."""
from pathlib import Path
import shutil


def admit_staging(runtime, workspace, *, reserve_bytes=8 * 1024**3):
    """Check one in-flight state/model replacement before mutating the actor."""
    count = sum(p.numel() for p in runtime.model.parameters() if p.requires_grad)
    model_bytes = sum(p.numel() * p.element_size() for p in runtime.model.parameters())
    required = 12 * count + model_bytes + 256 * 1024**2 + reserve_bytes
    free = shutil.disk_usage(workspace).free
    if free < required:
        raise ValueError('in-flight local optimizer/model disk capacity: required=' + str(required) + ', free=' + str(free))
    return dict(additional_disk_required_bytes=required, observed_free_disk_bytes=free)


def admit_fresh_optimizer(runtime, workspace, *, reserve_bytes=8 * 1024**3,
                          extra_model_copies=0):
    """Account for one new local optimizer generation and model export.

    Existing generations are already reflected in free disk. CPU masters and
    both moments are reserved before allocation, including bounded Adam scratch.
    This is resource admission, not a claim that backward kernels are qualified.
    """
    from ..persistent_training_state import available_ram_bytes
    parameters = tuple(p for p in runtime.model.parameters() if p.requires_grad)
    if not parameters or type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError('nonempty actor and explicit nonnegative reserve')
    if type(extra_model_copies) is not int or extra_model_copies < 0:
        raise ValueError('explicit diagnostic model copy count')
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError('ordinary trainer workspace')
    count = sum(p.numel() for p in parameters)
    largest = max(p.numel() for p in parameters)
    model_bytes = sum(p.numel() * p.element_size() for p in runtime.model.parameters())
    # Tokenizer, config and safetensor header allowance is separate from tensors.
    export_bytes = model_bytes + 256 * 1024**2
    required_ram = 12 * count + 20 * largest + reserve_bytes
    required_disk = 12 * count + export_bytes * (1 + extra_model_copies) + reserve_bytes
    available_ram, available_disk = available_ram_bytes(), shutil.disk_usage(root).free
    result = dict(trainable_parameters=count, model_parameter_bytes=model_bytes,
        cpu_additional_required_bytes=required_ram, observed_available_ram_bytes=available_ram,
        additional_disk_required_bytes=required_disk, observed_free_disk_bytes=available_disk,
        fp32_accumulator_bytes=4 * count, local_optimizer_generation_bytes=12 * count,
        reserve_bytes=reserve_bytes, extra_model_copies=extra_model_copies,
        retained_committed_generations=1, max_local_optimizer_generations=2,
        forward_backward_capacity_qualified=False)
    if available_ram < required_ram:
        raise ValueError('insufficient cgroup-aware RAM for CPU FP32 optimizer: ' + str(result))
    if available_disk < required_disk:
        raise ValueError('insufficient disk for original state plus in-flight publication: ' + str(result))
    if parameters[0].device.type == 'cuda':
        import torch
        available_gpu, _ = torch.cuda.mem_get_info(parameters[0].device)
        result['observed_free_gpu_bytes'] = available_gpu
        if available_gpu < 4 * count + reserve_bytes:
            raise ValueError('insufficient GPU memory for FP32 accumulation reserve: ' + str(result))
    return result
