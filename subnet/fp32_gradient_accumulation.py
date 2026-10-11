"""Prospective training-only FP32 accumulation; NOT enabled in production."""
import math
import json
import sys

METHOD = 'fp32-task-gradient-accumulation-v1'


def admit_capacity(torch, named_parameters):
    """Preflight the extra buffers plus bounded backward workspace headroom.

    This is capacity admission, not a replacement for GPU qualification of the
    exact model/runtime. No allocation or training state mutation occurs here.
    """
    parameters = list(named_parameters)
    devices = {p.device for _, p in parameters}
    if len(devices) != 1 or not all(p.is_cuda for _, p in parameters):
        raise ValueError('one qualified CUDA device for FP32 accumulation')
    extra_bytes = sum(p.numel() * 4 for _, p in parameters)
    parameter_bytes = sum(p.numel() * p.element_size() for _, p in parameters)
    reserve_bytes = max(32 * 1024**3, 2 * parameter_bytes)
    device = next(iter(devices))
    required_bytes = extra_bytes + reserve_bytes
    def snapshot(stage):
        free, total = torch.cuda.mem_get_info(device)
        stats = torch.cuda.memory_stats(device)
        row = dict(version='reference-boundary-CUDA-cache-admission-v1', stage=stage,
            device=str(device), extra_buffer_bytes=extra_bytes,
            backward_reserve_bytes=reserve_bytes, required_free_bytes=required_bytes,
            free_bytes=free, total_bytes=total,
            allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device),
            inactive_split_bytes=stats.get('inactive_split_bytes.all.current'),
            allocated_peak_bytes=torch.cuda.max_memory_allocated(device),
            reserved_peak_bytes=torch.cuda.max_memory_reserved(device))
        # The ordinary worker's stderr is its original captured log. Emit before
        # cache release and before rejection, so failure retains real values.
        print(json.dumps(row, sort_keys=True, separators=(',', ':')), file=sys.stderr, flush=True)
        return row
    # Called after reference forwards return CPU scalars, before grad buffers.
    # Only this process's unused allocator cache is released. Live tensors,
    # model/gradient values and the existing capacity floor remain intact.
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        before = snapshot('before-unused-cache-release')
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        after = snapshot('after-unused-cache-release')
    if after['free_bytes'] < required_bytes:
        raise ValueError('insufficient GPU capacity for FP32 accumulation and backward reserve')
    return dict(method=METHOD, extra_buffer_bytes=extra_bytes,
                backward_reserve_bytes=reserve_bytes, free_bytes=after['free_bytes'],
                total_bytes=after['total_bytes'], admitted=True,
                cache_admission=dict(version='reference-boundary-CUDA-cache-admission-v1',
                    before=before, after=after, unused_cache_release_calls=1,
                    existing_floor_unchanged=True))



class FP32GradientAccumulator:
    def __init__(self, named_parameters):
        import torch
        self.torch = torch
        self.parameters = list(named_parameters)
        names = [name for name, _ in self.parameters]
        if not names or len(set(names)) != len(names):
            raise ValueError('unique named parameters required')
        if len({id(p) for _, p in self.parameters}) != len(names):
            raise ValueError('distinct parameter objects required')
        if any(not p.requires_grad or p.dtype not in (torch.bfloat16, torch.float32)
               for _, p in self.parameters):
            raise ValueError('trainable BF16 or FP32 parameters required')
        if len({p.device for _, p in self.parameters}) != 1:
            raise ValueError('one device required for global gradient norm')
        self.buffers = {name: torch.zeros_like(p, dtype=torch.float32,
                        memory_format=torch.preserve_format)
                        for name, p in self.parameters}
        self.microsteps = 0
        self.clipped = False
        self.failed = False

    @property
    def extra_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.buffers.values())

    def capture(self):
        """Call once after each weighted backward; clear BF16 micro-gradients."""
        if self.failed or self.clipped:
            raise ValueError('accumulator is not collecting')
        gradients = []
        # Validate the full inventory before changing any accumulated tensor.
        for name, parameter in self.parameters:
            gradient = parameter.grad
            if (gradient is None or gradient.is_sparse or gradient.shape != parameter.shape
                    or gradient.dtype != parameter.dtype or gradient.device != parameter.device):
                raise ValueError('complete dense parameter gradient required')
            gradients.append((name, parameter, gradient))
        try:
            with self.torch.no_grad():
                for name, parameter, gradient in gradients:
                    # TensorIterator promotes the BF16 addend into the FP32
                    # destination without retaining a second full-model copy.
                    self.buffers[name].add_(gradient)
                    parameter.grad = None
            self.microsteps += 1
        except Exception:
            self.failed = True
            raise

    def clip(self, maximum):
        """Compute and clip the global norm in FP32, returning the pre-clip norm."""
        if self.failed or self.clipped or not self.microsteps:
            raise ValueError('nonempty unconsumed accumulation required')
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError('positive finite clipping bound required')
        if any(p.grad is not None for _, p in self.parameters):
            raise ValueError('uncaptured gradient at clipping boundary')
        torch = self.torch
        with torch.no_grad():
            norms = [torch.linalg.vector_norm(value, dtype=torch.float32)
                     for value in self.buffers.values()]
            total = torch.linalg.vector_norm(torch.stack(norms), dtype=torch.float32)
            if not bool(torch.isfinite(total)):
                self.failed = True
                raise ValueError('nonfinite accumulated gradient norm')
            scale = torch.clamp(maximum / (total + 1e-6), max=1.0)
            for value in self.buffers.values():
                value.mul_(scale)
        self.clipped = True
        return float(total)

    def gradients(self):
        if not self.clipped or self.failed:
            raise ValueError('clipped complete gradients required')
        # The optimizer must consume these FP32 buffers directly. Reassigning
        # them to a BF16 parameter.grad would round away the improvement.
        return dict(self.buffers)
