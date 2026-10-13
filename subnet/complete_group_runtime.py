"""Native math/session facade for the authenticated MiMo compute backend.

One loaded backend is shared across tasks at a checkpoint. This facade neither
downloads checkpoints nor deletes them per audit; the host owns that lifetime.
"""
import copy
import math
import numpy as np

from . import harness as policy
from .environments import EnvironmentSpec, create_session
from .complete_group_distribution import eos_ids
from .complete_group_sampling import need


def completed_answer(text):
    """Existing math task contract: one balanced, nonempty final boxed answer."""
    if type(text) is not str or text.count('\\boxed{') != 1:
        return False
    start = text.index('\\boxed{') + len('\\boxed{')
    depth = 1
    for at in range(start, len(text)):
        if text[at] == '{':
            depth += 1
        elif text[at] == '}':
            depth -= 1
            if depth == 0:
                return bool(text[start:at].strip()) and not text[at+1:].strip()
    return False


def outcome(result, text, stop):
    """A grading/infra exception must propagate, never become a negative."""
    kind = result.get('classification')
    need(kind in ('positive', 'negative', 'neutral') and result.get('done') is True,
         'complete native classification required')
    reward = result.get('reward')
    need(type(reward) in (int, float) and math.isfinite(reward)
         and reward == float(kind == 'positive'), 'binary native math outcome')
    if stop != 'eos' or not completed_answer(text):
        return 'neutral', 0.
    return kind, float(kind == 'positive')


class MimoGroupRuntime:
    def __init__(self, checkpoint, manifest, environment, harness, *, miner, backend=None):
        from .mimo_runtime import MimoRuntime, validate_runtime_spec
        from .forced_sampling import bind_runtime
        need(type(miner) is str, 'authenticated miner required before model work')
        self.manifest = copy.deepcopy(manifest)
        spec = validate_runtime_spec(manifest['mimo_runtime'])
        descriptor = {k: manifest['checkpoint'][k] for k in ('id', 'files')}
        if backend is None:
            backend = MimoRuntime.load(checkpoint, descriptor, spec,
                                      device=manifest['backend_profile']['device'], trainable=False)
        else:
            backend.guard()
            need(backend.checkpoint_id == descriptor['id'] and backend.spec == spec and not backend.trainable,
                 'leased backend exact checkpoint/runtime inference binding')
        self.backend = backend
        self.model, self.tokenizer = self.backend.model, self.backend.tokenizer
        self.eos_token_ids = tuple(self.backend.eos_token_ids)
        self.runtime_revision = manifest['model_runtime_revision']
        self.configure(environment, harness)
        need(spec.enable_thinking == self.harness['enable_thinking']
             and spec.max_context_tokens == self.harness['max_context_tokens']
             and spec.temperature == self.harness['temperature'] and spec.top_p == self.harness['top_p']
             and spec.top_k == self.harness['top_k'], 'runtime/harness exact sampling and template settings')
        from toploc import build_proofs_base64
        from .proofs import verify_mapped_proofs
        from functools import partial
        self.build_proofs = build_proofs_base64
        self.verify_proofs = partial(verify_mapped_proofs, num_threads=2)
        bind_runtime(self, manifest, miner)

    def configure(self, environment, harness):
        self.spec = EnvironmentSpec.from_dict(environment)
        self.env_config = copy.deepcopy(environment)
        self.harness = policy.normalize(harness)
        need(self.spec.max_turns == 1 and self.harness['max_output_tokens'] <= self.spec.max_output_tokens,
             'single-turn native math horizon')
        return self

    def for_environment(self, environment, harness=None):
        result = copy.copy(self).configure(environment, harness)
        from .forced_sampling import bind_runtime
        return bind_runtime(result, self.manifest, self.sampling_context['miner'])

    def next_token_logits(self, *args, **kwargs):
        return self.backend.next_token_logits(*args, **kwargs)

    def compute(self, prompt, output):
        return self.backend.compute(prompt, output)

    def prompt(self, messages, tools=()):
        need(not tools, 'initial MiMo math scope is text-only')
        return self.backend.prepare_prompt(messages, tools)

    def rollout(self, index, seed):
        from .forced_sampling import sample, receipt
        from .complete_group_sampling import validate_runtime
        validate_runtime(self, self.sampling_context, index=index)
        env_seed = int(self.spec.config.get('seed', 0))
        session = create_session(self.spec)
        try:
            initial = session.reset(index, env_seed)
            prompt = self.prompt(initial['messages'], initial.get('tools', []))
            output = sample(self, prompt, seed, 0, index, initial['task_hash'])
            stop = 'eos' if output[-1] in self.eos_token_ids else 'budget'
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            acts, full = self.compute(prompt, output)
            need(full.ndim == 2 and full.shape[0] == len(output), 'actual selected T1 framing')
            arrays = full[np.arange(len(output)), output].astype(np.float32).reshape(-1, 1)
            proofs = self.build_proofs(acts, decode_batching_size=16, topk=128)
            from .proofs import validate_framing
            validate_framing(proofs, 1+math.ceil(len(output)/16), 128)
            from .mimo_answer_projection import grade
            native, kind, reward = grade(session, self.manifest, self.harness, self.tokenizer, output, stop)
            if stop == 'budget':
                # The collector preserves this native claim, then projects the
                # capped row to neutral. A cap never fills the wrong quota.
                kind, reward = native['classification'], float(native['reward'])
            turn = dict(prompt=prompt, output=output, text=text, proofs=proofs,
                observations=native['observations'], done=True, reward=reward, classification=kind)
            row = dict(schema=2, env_id=self.spec.id, environment_version=self.spec.version,
                index=index, sample_index=index, seed=seed, env_seed=env_seed, task_hash=initial['task_hash'],
                reward=reward, classification=kind, turns=[turn], sampling=receipt(self.sampling_context, seed))
            return row, [arrays]
        finally:
            session.close()

    def verify(self, rollout, arrays):
        from .complete_group_verification import verify_rollout
        verify_rollout(self, rollout, arrays)
        return True
