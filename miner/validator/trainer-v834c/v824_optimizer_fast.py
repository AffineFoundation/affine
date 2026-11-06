"""v8.2.4 candidate: same torch.optim.AdamW (foreach=False, fused=False) math as
CPUAdamW, bitwise-identical update, with (a) a scoped intra-op thread count for
the CPU step and (b) the next gradient's device->host copy overlapped with the
current parameter's CPU step. Elementwise AdamW has no reductions, so the result
does not depend on the thread count (verified bitwise on E42: 2/8/14/16 threads).
"""
import torch
from subnet.mimo_grpo.optimizer import CPUAdamW, digest


class PipelinedCPUAdamW(CPUAdamW):
    def __init__(self, named_parameters, config, *, input_checkpoint, step_threads=14):
        super().__init__(named_parameters, config, input_checkpoint=input_checkpoint)
        if type(step_threads) is not int or not 1 <= step_threads <= 64:
            raise ValueError('bounded optimizer step threads')
        self.step_threads = step_threads
        peak = max(p.numel() for _, p in self.parameters)
        # Two pinned staging buffers (double buffer); 2 x largest tensor x 4 B.
        self._pinned = [torch.empty(peak, dtype=torch.float32, pin_memory=True) for _ in range(2)]
        self._copy_stream = None

    def step(self, gradients):
        if self.failed or set(gradients) != set(self.masters):
            raise ValueError('healthy optimizer and exact gradient inventory required')
        for name, parameter in self.parameters:
            g = gradients[name]
            if g.dtype != torch.float32 or g.shape != parameter.shape or not torch.isfinite(g).all():
                raise ValueError('finite FP32 accumulated gradients required: ' + name)
        before = self.global_step
        self.zero_grad()
        previous_threads = torch.get_num_threads()
        on_cuda = any(gradients[n].is_cuda for n, _ in self.parameters)
        if on_cuda and self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        changed = 0
        events = [None, None]

        def launch(index):
            name, parameter = self.parameters[index]
            g = gradients[name]
            slot = index % 2
            view = self._pinned[slot][:g.numel()].view(g.shape)
            if on_cuda:
                self._copy_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._copy_stream):
                    view.copy_(g.detach(), non_blocking=True)
                    event = torch.cuda.Event(); event.record(self._copy_stream)
                events[slot] = event
            else:
                view.copy_(g.detach())
                events[slot] = None
            return view
        try:
            torch.set_num_threads(self.step_threads)
            pending = launch(0)
            for index, (name, parameter) in enumerate(self.parameters):
                slot = index % 2
                if events[slot] is not None:
                    events[slot].synchronize()
                master = self.masters[name]
                # Private copy: the pinned slot is reused two parameters later.
                master.grad = pending.clone()
                if index + 1 < len(self.parameters):
                    pending = launch(index + 1)
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
            self.failed = True
            raise
        finally:
            torch.set_num_threads(previous_threads)
            self.zero_grad()
        return dict(optimizer_step=self.global_step, implementation='torch.optim.AdamW',
            foreach=False, fused=False, lr=self.config.learning_rate,
            betas=[self.config.beta1, self.config.beta2], eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay, state_dtype='float32', master_dtype='float32',
            weight_decay_policy=self.config.weight_decay_policy,
            parameter_decay_inventory_sha256=digest(self.decay_inventory),
            decay_parameter_count=sum(row['weight_decay'] != 0 for row in self.decay_inventory),
            no_decay_parameter_count=sum(row['weight_decay'] == 0 for row in self.decay_inventory),
            gradient_accumulation_dtype='float32', projected_changed_elements=changed,
            step_threads=self.step_threads, transfer='pinned-double-buffer-overlap')
