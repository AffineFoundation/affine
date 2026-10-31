"""Execute exact functions from the licensed, content-pinned MiMo source.

Only registration decorators/import plumbing are omitted. Function AST bodies,
including clipping, metrics and prompt normalization, are unchanged. This avoids
pulling in the unrelated Ray/Megatron/agent stack merely to evaluate the loss.
"""
import ast
from collections import defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

CORE_SHA = '812512d3569ed22ff972d4e419235d314b06eb5c8503f929fdc1010dee2f556b'
FUNCTIONAL_SHA = 'cb37d045dec2161d637e2633032cc97c97334e5547cea62dace42630c9683259'


class AlgoConfig:
    """Type marker required by upstream's explicit actor/config assertion."""


class ActorConfig(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


def _load(path, digest, names, namespace):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('pinned upstream source changed: ' + str(path))
    tree = ast.parse(raw, filename=str(path))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    if {node.name for node in nodes} != set(names):
        raise ValueError('missing exact upstream functions')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), 'exec'), namespace)


@lru_cache(maxsize=1)
def reference():
    root = Path(__file__).resolve().parents[2] / 'vendor' / 'mimo_verl_pinned'
    namespace = dict(torch=torch, np=np, defaultdict=defaultdict, AlgoConfig=AlgoConfig)
    functional = dict(torch=torch)
    _load(root / 'verl/utils/torch_functional.py', FUNCTIONAL_SHA, ('masked_sum', 'masked_mean'), functional)
    namespace['verl_F'] = SimpleNamespace(**{k: functional[k] for k in ('masked_sum', 'masked_mean')})
    names = ('compute_grpo_outcome_advantage', 'compute_prompt_loss_weights', 'agg_loss',
             'compute_policy_loss_vanilla', 'kl_penalty_forward')
    _load(root / 'verl/trainer/ppo/core_algos.py', CORE_SHA, names, namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def batch_info(config, prompt_weights=None, global_batch_size=None):
    """Global (not microbatch-local) normalizers for the pinned agg_loss."""
    if config.loss_agg_mode == 'prompt-mean':
        return dict(prompt_loss_weights=prompt_weights, dp_size=1)
    if type(global_batch_size) is not int or global_batch_size < 1:
        raise ValueError('constant normalization needs the global active rollout count')
    return dict(global_batch_size=global_batch_size, loss_scale_factor=config.loss_scale_factor, dp_size=1)


def policy_loss(old, current, advantages, mask, prompt_weights, config, *, global_batch_size=None):
    actor = ActorConfig(clip_ratio=config.clip_ratio, clip_ratio_low=config.clip_ratio,
        clip_ratio_high=config.clip_ratio, clip_ratio_c=config.dual_clip,
        global_batch_info=batch_info(config, prompt_weights, global_batch_size))
    return reference().compute_policy_loss_vanilla(old, current, advantages, mask,
        loss_agg_mode=config.loss_agg_mode, config=actor)


def aggregate(loss_mat, mask, prompt_weights, config, *, global_batch_size=None):
    """Same pinned aggregation as the policy loss, for KL/corrected terms."""
    return reference().agg_loss(loss_mat=loss_mat, loss_mask=mask, loss_agg_mode=config.loss_agg_mode,
                                **batch_info(config, prompt_weights, global_batch_size))


def k3_kl(current, reference_logprob):
    """Pinned upstream k3 estimator: exp(ref-cur) - (ref-cur) - 1, clamped."""
    return reference().kl_penalty_forward(current, reference_logprob, 'low_var_kl')
