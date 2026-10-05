"""Explicit signed GPU contracts; hardware support is not qualification evidence."""
from copy import deepcopy
import json

LEGACY_REVISION = 'cuda-bf16-eager-sm86-v1'
HOPPER_REVISION = 'cuda-bf16-eager-sm90-v1'
HOPPER_FP32_REVISION = 'cuda-fp32-eager-sm90-v1'
NUMERICAL_POLICY = dict(logprob_atol=1e-5, logprob_rtol=0,
    toploc_exp_mismatches=0, toploc_mant_err_mean=0, toploc_mant_err_median=0)

def profile(revision):
    if revision not in (LEGACY_REVISION, HOPPER_REVISION, HOPPER_FP32_REVISION):
        raise ValueError('GPU profile or numerical policy: unknown runtime revision')
    return (revision, dict(device='cuda', dtype='float32' if revision==HOPPER_FP32_REVISION else 'bfloat16', attention='eager',
        sm=[8,6] if revision == LEGACY_REVISION else [9,0], tf32=False,
        deterministic_algorithms=True, cublas_workspace_config=':4096:8',
        native_toploc_threads=2, torch_threads=2), deepcopy(NUMERICAL_POLICY))

def resolve(manifest):
    resolved = profile(manifest.get('model_runtime_revision'))
    def exact(a,b):
        try:return json.dumps(a,sort_keys=True,allow_nan=False,separators=(',',':'))==json.dumps(b,sort_keys=True,allow_nan=False,separators=(',',':'))
        except (TypeError,ValueError):return False
    if not exact(manifest.get('backend_profile'),resolved[1]) or not exact(manifest.get('numerical_policy'),resolved[2]):
        raise ValueError('GPU profile or numerical policy')
    return resolved

def for_config(config):
    return profile(config.get('model_runtime_revision', LEGACY_REVISION))

def execution_profile(manifest,role):
    """Signed prospective BF16 learner override; inference profile stays FP32."""
    generation=resolve(manifest)
    override=manifest.get('training_runtime')
    if override is None:return generation
    from .persistent_cpu_adamw import POLICY
    expected=dict(version='separate-bf16-persistent-training-runtime-v1',model_runtime_revision=HOPPER_REVISION,
        backend_profile=profile(HOPPER_REVISION)[1],numerical_policy=profile(HOPPER_REVISION)[2])
    if (manifest.get('training_input_policy')!='committed-unaudited-training-v1' or manifest.get('training_policy')!=POLICY or
        generation[0]!=HOPPER_FP32_REVISION or override!=expected):
        raise ValueError('explicit separate BF16 persistent training runtime')
    return profile(HOPPER_REVISION) if role=='train' else generation
