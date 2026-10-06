"""Signed public draws and strict sampler replay; no historical execution claim."""
import hashlib
import inspect
import json
import secrets
from pathlib import Path

VERSION = 'forced-inverse-cdf-replay-v1'
FIELDS = {'version', 'randomness', 'max_attempts', 'verification', 'generation'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate(value):
    from .fast_prefill_audit import VERSION as FAST,SUPPORT_VERSION as SUPPORT,calibration
    expected=(FIELDS|{'calibration','support_adjudication'}if isinstance(value,dict)and value.get('version')==SUPPORT else FIELDS|{'calibration'}if isinstance(value,dict)and value.get('version')==FAST else FIELDS)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('forced sampling contract fields')
    if value['version']in(FAST,SUPPORT):
        if value['version']==SUPPORT and value['support_adjudication']!='exact-cached-replay-v1':raise ValueError('explicit cached support adjudication')
        if value['verification']!='prefill-cdf-calibrated' or value['generation']!='cached-eager-inverse-cdf':raise ValueError('fast sampling contract version')
        calibration(value['calibration'])
    elif value['version'] != VERSION or value['verification'] != 'exact-token-replay' or value['generation'] != 'uncached-eager-inverse-cdf':
        raise ValueError('forced sampling contract version')
    random = value['randomness']
    if not isinstance(random, str) or len(random) != 64 or any(c not in '0123456789abcdef' for c in random):
        raise ValueError('forced sampling public randomness')
    if type(value['max_attempts']) is not int or not 2 <= value['max_attempts'] <= 128:
        raise ValueError('forced sampling attempt budget')
    return dict(value)


def new_contract(config):
    from .fast_prefill_audit import VERSION as FAST,SUPPORT_VERSION as SUPPORT
    if isinstance(config,dict)and config.get('version')in(FAST,SUPPORT):
        expected={'version','max_attempts','calibration'}|({'support_adjudication'}if config['version']==SUPPORT else set())
        if set(config)!=expected:raise ValueError('fast sampling opening configuration')
        return validate(dict(config,randomness=secrets.token_hex(32),verification='prefill-cdf-calibrated',generation='cached-eager-inverse-cdf'))
    if not isinstance(config, dict) or set(config) != {'version', 'max_attempts'}:
        raise ValueError('forced sampling opening configuration')
    return validate(dict(config, randomness=secrets.token_hex(32),
                         verification='exact-token-replay', generation='uncached-eager-inverse-cdf'))


def validate_harness(config,contract=None):
    from .harness import normalize
    c = normalize(config)
    if c['policy'] != 'autoregressive' or c.get('turn_overrides'):
        raise ValueError('forced sampling requires unmodified autoregressive policy')
    # Cached generation is deliberately not an alternative sampler in v1.
    if c['version'] == 'text-tools-long-kv-v3':
        raise ValueError('forced sampling v1 requires uncached harness')
    return c


def binding(manifest):
    value = manifest.get('sampling_contract')
    if value is None:
        if 'sampling_source_hash' in manifest:
            raise ValueError('sampling contract missing')
        return None
    validate(value)
    if manifest.get('sampling_source_hash') != source_hash():
        raise ValueError('trusted sampling source mismatch')
    epoch, checkpoint = manifest.get('epoch'), manifest.get('checkpoint', {}).get('id')
    if not isinstance(epoch, str) or not epoch or not isinstance(checkpoint, str) or len(checkpoint) != 64:
        raise ValueError('sampling epoch/checkpoint binding')
    return {'contract': json.loads(canonical(value)), 'epoch': epoch, 'checkpoint': checkpoint}


def bind_runtime(runtime, manifest):
    context = binding(manifest)
    if context is not None:
        validate_harness(runtime.harness)
    if context is not None and context['contract']['version']!=VERSION:
        from .fast_prefill_audit import bind
        runtime.fast_sampling_calibration=bind(manifest,runtime.harness)
        runtime.fast_sampling_manifest=manifest
    else:runtime.fast_sampling_calibration=None
    runtime.sampling_context = context
    return runtime


def receipt(context, attempt):
    validate_attempt(context, attempt)
    return {'version': context['contract']['version'], 'binding_sha256': hashlib.sha256(canonical(context)).hexdigest(), 'attempt': attempt}


def validate_attempt(context, attempt):
    if type(attempt) is not int or not 0 <= attempt < context['contract']['max_attempts']:
        raise ValueError('forced sampling attempt out of range')


def uniform(context, env_id, task_hash, index, attempt, turn, position):
    validate_attempt(context, attempt)
    if any(type(i) is not int or i < 0 for i in (index, turn, position)):
        raise ValueError('forced sampling position')
    message = dict(context, environment=env_id, task_hash=task_hash, index=index,
                   attempt=attempt, turn=turn, position=position)
    # 53 significant bits keep the public draw precisely representable in float64.
    integer = int.from_bytes(hashlib.sha256(canonical(message)).digest()[:8], 'big') >> 11
    return integer / 2**53


def pick(logits, u, temperature, top_p):
    import torch
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1:
        values, indices = torch.sort(probabilities, descending=True, stable=True)
        values = torch.where(values.cumsum(0) - values >= top_p, 0., values)
        probabilities = torch.zeros_like(probabilities).scatter(0, indices, values)
        probabilities = probabilities / probabilities.sum()
    if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
        raise RuntimeError('nonfinite sampling distribution; infrastructure failure')
    # float64 interval arithmetic retains the full public uniform, rather than
    # rounding a near-one draw to 1 in float32 and selecting a zero-mass tail.
    probabilities = probabilities.double()
    probabilities = probabilities / probabilities.sum()
    cumulative = probabilities.cumsum(0)
    cumulative[-1] = 1.
    selected = int(torch.searchsorted(cumulative, torch.tensor(u, dtype=torch.float64, device=cumulative.device), right=True))
    if selected >= len(probabilities) or not probabilities[selected] > 0:
        raise RuntimeError('invalid sampling interval; infrastructure failure')
    return selected


def sample(runtime, prompt, attempt, turn, index, task_hash):
    import torch
    context = runtime.sampling_context
    if context['contract']['version']!=VERSION:
        from .fast_prefill_audit import cached_sample
        return cached_sample(runtime,prompt,attempt,turn,index,task_hash)
    validate_attempt(context, attempt)
    config = validate_harness(runtime.harness)
    device = next(runtime.model.parameters()).device
    output = []
    stop = runtime.tokenizer.eos_token_id
    supports_last = 'logits_to_keep' in inspect.signature(runtime.model.forward).parameters
    with torch.inference_mode():
        for position in range(config['max_output_tokens']):
            kwargs = {'use_cache': False}
            if supports_last:
                kwargs['logits_to_keep'] = 1
            result = runtime.model(torch.tensor([prompt + output], device=device), **kwargs)
            u = uniform(context, runtime.spec.id, task_hash, index, attempt, turn, position)
            token = pick(result.logits[0, -1], u, config['temperature'], config['top_p'])
            output.append(token)
            del result
            if token == stop:
                break
    return output


def assurance(manifest):
    context = binding(manifest)
    if context is None:
        return {'sampling_required': False, 'scope': 'model-computation-and-environment-only'}
    return {'sampling_required': True, 'scope': 'fully-audited-rollouts-only',
            'version': context['contract']['version'], 'binding_sha256': hashlib.sha256(canonical(context)).hexdigest(),
            'verification': context['contract']['verification'], 'historical_execution_proven': False}


def require_report(manifest, report):
    """Refuse unbound audit attestations before publication or reward credit.

    This checks the authenticated worker's assertion, not a second inference
    proof. Legacy epochs retain their original signed audit contract.
    """
    context = binding(manifest)
    if context is None:
        return
    if report.get('sampling_assurance') != assurance(manifest):
        raise ValueError('audit sampling assurance missing or mismatched')
    for batch in report.get('accepted', []):
        for rollout in batch.get('rollouts', []):
            attempt = rollout.get('seed')
            if rollout.get('sampling') != receipt(context, attempt):
                raise ValueError('credited rollout sampling receipt mismatch')
