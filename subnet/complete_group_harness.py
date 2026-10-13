"""Prospective complete-group horizon, bound to an executed ROOT qualification.

No model loader, context extension, RoPE mutation, or qualification generation
belongs here. The existing trusted loader still authenticates checkpoint files.
"""
import base64
from copy import deepcopy
import hashlib
import json

VERSION = 'text-tools-complete-group-long-v1'
QUALIFICATION = 'complete-group-long-context-qualification-v1'
SAMPLER = 'forced-inverse-cdf-complete-group-v1'
AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
MAX_OUTPUT = 4096
MAX_CONTEXT = 8192
PROFILE_KEY = 'complete_group_long_context'
FIELDS = {'version', 'harness_version', 'max_output_tokens', 'max_context_tokens',
          'checkpoint_config_sha256', 'effective_config_sha256', 'model_runtime_revision',
          'backend_profile_sha256', 'runtime_profile_sha256', 'eos_token_id',
          'vocab_size', 'checks', 'report_sha256'}
CHECKS = {'cached_generation', 'full_prefill', 'token_bounds', 'no_runtime_config_mutation'}
MIMO_VERSION = 'text-tools-mimo-complete-group-v1'


def normalize_mimo(value):
    need(type(value) is dict and set(value) == {'version', 'policy', 'max_output_tokens',
         'max_context_tokens', 'temperature', 'top_p', 'top_k', 'enable_thinking'},
         'exact MiMo complete-group harness fields')
    need(value['version'] == MIMO_VERSION and value['policy'] == 'autoregressive', 'MiMo autoregressive harness')
    need(type(value['max_output_tokens']) is int and 1 <= value['max_output_tokens'] <= 8192,
         'bounded MiMo output horizon')
    need(type(value['max_context_tokens']) is int and value['max_output_tokens'] < value['max_context_tokens'] <= 32768,
         'separate bounded MiMo context horizon')
    need(type(value['temperature']) in (int, float) and value['temperature'] == 1.
         and type(value['top_p']) in (int, float) and value['top_p'] == .95
         and type(value['top_k']) is int and value['top_k'] == 20,
         'signed MiMo reference sampler T1 top-p .95 top-k20')
    need(type(value['enable_thinking']) is bool, 'explicit MiMo thinking mode')
    return deepcopy(value)


def render_mimo(tokenizer, messages, tools=(), config=None):
    h = normalize_mimo(config)
    need(not tools, 'initial MiMo math harness is text-only')
    result = tokenizer.apply_chat_template([dict(m) for m in messages], tokenize=True,
        add_generation_prompt=True, enable_thinking=h['enable_thinking'], return_dict=False)
    return list(result['input_ids']) if hasattr(result, 'keys') else list(result)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def model_config_digest(config):
    """Paths are not computation; preserve every other serialized config field."""
    value = deepcopy(config.to_dict())
    value.pop('_name_or_path', None)
    return digest(value)


def normalize(value):
    need(set(value) == {'version', 'policy', 'max_output_tokens', 'max_context_tokens',
                       'temperature', 'top_p'}, 'exact complete-group long harness fields')
    need(value['version'] == VERSION and value['policy'] == 'autoregressive', 'group long autoregressive harness')
    need(type(value['max_output_tokens']) is int and 1 <= value['max_output_tokens'] <= MAX_OUTPUT,
         'complete-group output budget')
    need(type(value['max_context_tokens']) is int and value['max_context_tokens'] == MAX_CONTEXT,
         'explicit qualified group context budget')
    return value


def require_sampler(config, contract):
    if config.get('version') == VERSION:
        need(type(contract) is dict and contract.get('version') == SAMPLER,
             'long harness requires complete-group sampler')


def unbound_sample(*args, **kwargs):
    raise ValueError('long harness requires authenticated complete-group runtime dispatcher')


def bind_manifest(manifest, environments):
    policies = [(h or {}) for row in environments for h in (
        row['harness']['by_index'].values() if isinstance(row['harness'], dict) and row['harness'].get('version') == 'indexed-harness-v1'
        else [row['harness']])]
    selected = [h for h in policies if h.get('version') == VERSION]
    runtime = manifest.get('runtime_profile', {})
    if not selected:
        need(PROFILE_KEY not in runtime, 'long qualification without long harness')
        return None
    require_sampler(selected[0], manifest.get('sampling_contract'))
    for policy in selected:
        normalize(policy)
    envelope = runtime.get(PROFILE_KEY)
    need(type(envelope) is dict and set(envelope) == {'payload', 'signature', 'signer'},
         'explicit signed long-context qualification required')
    need(envelope['signer'] == AUTHORITY, 'trusted long-context qualification authority')
    from nacl.signing import VerifyKey
    try:
        VerifyKey(bytes.fromhex(AUTHORITY)).verify(canonical(envelope['payload']),
            base64.b64decode(envelope['signature'], validate=True))
    except Exception as exc:
        raise ValueError('long-context qualification signature') from exc
    q = envelope['payload']
    need(type(q) is dict and set(q) == FIELDS, 'exact long-context qualification fields')
    need(q['version'] == QUALIFICATION and q['harness_version'] == VERSION,
         'qualified long-context version')
    need(type(q['max_output_tokens']) is int and q['max_output_tokens'] == MAX_OUTPUT
         and type(q['max_context_tokens']) is int and q['max_context_tokens'] == MAX_CONTEXT,
         'qualified exact group horizon')
    for name in ('checkpoint_config_sha256', 'effective_config_sha256', 'backend_profile_sha256',
                 'runtime_profile_sha256', 'report_sha256'):
        need(type(q[name]) is str and len(q[name]) == 64 and set(q[name]) <= set('0123456789abcdef'),
             'qualified lowercase SHA256 '+name)
    need(type(q['checks']) is dict and set(q['checks']) == CHECKS and all(v is True for v in q['checks'].values()), 'executed long-context qualification checks')
    need(type(q['vocab_size']) is int and q['vocab_size'] > 1 and type(q['eos_token_id']) is int
         and 0 <= q['eos_token_id'] < q['vocab_size'], 'qualified tokenizer geometry')
    files = manifest['checkpoint'].get('files', {})
    need(files.get('config.json') == q['checkpoint_config_sha256'], 'qualified checkpoint config pin')
    need(q['model_runtime_revision'] == manifest['model_runtime_revision']
         and q['backend_profile_sha256'] == digest(manifest['backend_profile']), 'qualified backend runtime pin')
    rest = {k: v for k, v in runtime.items() if k != PROFILE_KEY}
    need(q['runtime_profile_sha256'] == digest(rest), 'qualified runtime profile pin')
    transport = manifest.get('group_transport_policy')
    need(type(transport) is dict and transport.get('version') == 'complete-group-bounded-selected-v1',
         'signed complete-group transport geometry required')
    for field in ('max_context_tokens', 'eos_token_id', 'vocab_size'):
        need(type(transport.get(field)) is int and transport[field] == q[field], 'transport/qualification '+field)
    return deepcopy(q)


def validate_runtime(runtime, context):
    h = runtime.harness
    if h.get('version') != VERSION:
        return
    require_sampler(h, context['contract'])
    q = context.get('long_context_contract')
    need(type(q) is dict, 'bound long-context qualification required')
    need(model_config_digest(runtime.model.config) == q['effective_config_sha256'], 'loaded effective config differs')
    available = getattr(runtime.model.config, 'max_position_embeddings', None)
    need(type(available) is int and available >= MAX_CONTEXT, 'insufficient native model context; publish qualified config first')
    need(getattr(runtime, 'runtime_revision', None) == q['model_runtime_revision'], 'loaded runtime revision differs')
    need(getattr(runtime.model.config, '_attn_implementation', None) == 'eager', 'qualified eager sampling backend')
    need(getattr(runtime.tokenizer, 'eos_token_id', None) == q['eos_token_id']
         and len(runtime.tokenizer) == q['vocab_size'], 'loaded tokenizer geometry differs')
    head = getattr(runtime.model.config, 'vocab_size', None)
    need(type(head) is int and head >= q['vocab_size'], 'model head cannot cover qualified tokenizer')


def validate_prompt(context, harness, prompt):
    if harness.get('version') != VERSION:
        return
    q = context.get('long_context_contract')
    need(type(q) is dict, 'bound long-context qualification required')
    need(len(prompt) + harness['max_output_tokens'] <= q['max_context_tokens'], 'full prompt plus horizon exceeds context')
    # Reuse the existing framing validator without selecting its SM86/32K runtime.
    from .long_context_runtime import validate_tokens
    validate_tokens(prompt, [q['eos_token_id']], q['vocab_size'], context=q['max_context_tokens'])
