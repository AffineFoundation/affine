"""Isolated learning-rate ablations; never change a deployed trainer or protocol.

The parent is restored using the existing, immutable optimizer contract. Only
the learning rate of its next disposable research update differs. No optimizer
state from these arms may be exported or used as a production continuation.
"""
import ast
import copy
import inspect
import math
import textwrap


class FP32GradientAccumulator:
    """Sum microbatch gradients in FP32, projecting once for the parent AdamW."""
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.buffers = {}
        self.handles = []
        if not self.parameters or len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError('distinct nonempty gradient parameters')

    def __enter__(self):
        import torch
        def collect(parameter):
            gradient = parameter.grad
            if gradient is None or gradient.is_sparse:
                raise ValueError('dense full-model gradient required')
            with torch.no_grad():
                key = id(parameter)
                if key not in self.buffers:
                    self.buffers[key] = gradient.detach().to(torch.float32).clone()
                else:
                    self.buffers[key].add_(gradient.detach())
                parameter.grad = None
        self.handles = [p.register_post_accumulate_grad_hook(collect) for p in self.parameters]
        return self

    def flush(self):
        import torch
        if set(self.buffers) != {id(p) for p in self.parameters}:
            raise ValueError('incomplete full-model accumulated gradients')
        with torch.no_grad():
            for parameter in self.parameters:
                value = self.buffers.pop(id(parameter))
                if not bool(torch.isfinite(value).all()):
                    raise ValueError('nonfinite accumulated gradient')
                parameter.grad = value.to(dtype=parameter.dtype)

    def __exit__(self, *unused):
        for handle in self.handles: handle.remove()
        self.handles.clear()
        self.buffers.clear()


def research_optimizer(base_class, learning_rate):
    if (type(learning_rate) not in (int, float) or
            not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-5):
        raise ValueError('bounded research learning rate')
    # Preserve the actual parent implementation and all of its state/finite
    # checks. Change exactly its local step arithmetic, not the authenticated
    # hyperparameters used to restore the parent descriptor.
    method = ast.parse(textwrap.dedent(inspect.getsource(base_class._step_impl)))
    matches = 0
    for node in ast.walk(method):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'h' and
                ast.dump(node.value) == ast.dump(ast.parse('self.hyperparameters', mode='eval').body)):
            node.value = ast.parse('dict(self.hyperparameters, lr=self.research_learning_rate)', mode='eval').body
            matches += 1
    if matches != 1:
        raise ValueError('research adapter requires exactly one original step hyperparameter binding')
    namespace = dict(base_class._step_impl.__globals__)
    exec(compile(ast.fix_missing_locations(method), '<isolated-research-AdamW>', 'exec'), namespace)

    class DisposableResearchAdamW(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.research_learning_rate = float(learning_rate)

    DisposableResearchAdamW._step_impl = namespace['_step_impl']
    return DisposableResearchAdamW


def train_arm(training_module, runtime, pairs, output, *, learning_rate,
              fp32_gradient_accumulation=False, **kwargs):
    """Run in an isolated process. Restore the same parent separately per arm."""
    original = training_module.PersistentCPUAdamW
    training_module.PersistentCPUAdamW = research_optimizer(original, learning_rate)
    accumulate = training_module.accumulate_tasks
    try:
        if fp32_gradient_accumulation:
            with FP32GradientAccumulator(runtime.model.parameters()) as accumulator:
                def precise_accumulation(*args, **kw):
                    observations = accumulate(*args, **kw)
                    accumulator.flush()
                    return observations
                training_module.accumulate_tasks = precise_accumulation
                destination, optimizer, diagnostics = training_module.train_epoch(runtime, pairs, output, **kwargs)
        else:
            destination, optimizer, diagnostics = training_module.train_epoch(runtime, pairs, output, **kwargs)
    finally:
        training_module.PersistentCPUAdamW = original
        training_module.accumulate_tasks = accumulate
    diagnostics = copy.deepcopy(diagnostics)
    for update in diagnostics['updates']:
        update['parent_hyperparameters'] = copy.deepcopy(update['hyperparameters'])
        update['hyperparameters']['lr'] = float(learning_rate)
        update['optimizer_lifecycle'] = 'disposable-isolated-learning-rate-ablation'
    diagnostics.update(research_only=True, production_mutations=False,
        optimizer_state_durable=False, state_publication_required=False,
        effective_learning_rate=float(learning_rate), heldout_gain_claimed=False,
        microbatch_gradient_accumulation_dtype='float32' if fp32_gradient_accumulation else 'bfloat16',
        research_parent_state_sha256=optimizer.parent_state_sha256)
    # Do not expose changed research moments as a resumable optimizer under the
    # parent's unchanged descriptor. Only the inference model is an output.
    optimizer.rows.clear()
    return destination, diagnostics
