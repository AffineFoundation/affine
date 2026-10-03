"""Versioned policy/rendering boundary, independent of environment implementation.

Candidates are an explicitly curated research policy. Autoregressive policy is
unconstrained target-model sampling. Neither receives hidden environment state.
"""
import hashlib
import json
import math
from pathlib import Path
import torch

DEFAULT = {'version': 'text-tools-v1', 'policy': 'autoregressive', 'max_output_tokens': 64,
           'temperature': .7, 'top_p': 1.0}


def normalize(config=None):
    value = dict(DEFAULT); value.update(config or {})
    if value['version'] not in HARNESS_REGISTRY or value['policy'] not in ('autoregressive', 'candidates', 'visible-copy-candidates', 'public-mrcr-shell-candidates'):
        raise ValueError('unsupported harness or policy')
    limit=2048 if value['version'] in ('text-tools-long-v2','text-tools-format-long-v1','text-tools-long-kv-v3') else 512
    if type(value['max_output_tokens']) is not int or not 1 <= value['max_output_tokens'] <= limit:
        raise ValueError('generation token budget')
    if not math.isfinite(value['temperature']) or not 0 < value['temperature'] <= 4:
        raise ValueError('temperature')
    if not 0 < value['top_p'] <= 1:
        raise ValueError('top_p')
    if value['policy'] == 'public-mrcr-shell-candidates':
        from .native_mrcr_public_policy import REVISION
        digest=hashlib.sha256((Path(__file__).parent/'native_mrcr_public_policy.py').read_bytes()).hexdigest()
        if value.get('public_policy_revision')!=REVISION or value.get('public_policy_sha256')!=digest:
            raise ValueError('public MRCR policy pin')
    if value['version']=='text-tools-format-long-v1':
        instruction=value.get('response_format_instruction')
        if not isinstance(instruction,str) or not instruction.strip() or len(instruction)>512:
            raise ValueError('bounded signed response format instruction')
    elif 'response_format_instruction' in value:
        raise ValueError('response format instruction requires its versioned harness')
    if 'generation_kv_cache' in value or 'sampling_mode' in value:
        raise ValueError('sampling optimization must use its explicit signed harness version')
    if value['version']=='text-tools-long-kv-v3' and value['policy']!='autoregressive':
        raise ValueError('KV harness only supports explicit autoregressive generation')
    history_fields={'history_prefix_messages','history_window_messages'}
    if value['version']=='text-tools-window-v1':
        prefix=value.setdefault('history_prefix_messages',2)
        window=value.setdefault('history_window_messages',2)
        if (type(prefix) is not int or not 1<=prefix<=8 or
                type(window) is not int or not 1<=window<=32):
            raise ValueError('signed history window bounds')
    elif history_fields & set(value):
        raise ValueError('history window requires its versioned harness')
    if value['policy'] == 'candidates':
        candidates = value.get('candidates', [])
        if not 2 <= len(candidates) <= 256 or any(not isinstance(t, str) or not t or len(t) > 1000 for t in candidates):
            raise ValueError('curated candidates')
    if value['policy'] == 'visible-copy-candidates':
        tags = value.get('input_tags', ['<text>', '</text>'])
        if not isinstance(tags,list) or len(tags)!=2 or any(not isinstance(t,str) or not t for t in tags):
            raise ValueError('visible copy delimiters')
        value['input_tags']=tags
        value.setdefault('output_tags',['<answer>','</answer>'])
        if len(value['output_tags'])!=2 or any(not isinstance(t,str) for t in value['output_tags']):
            raise ValueError('visible output delimiters')
    overrides=value.get('turn_overrides',{})
    if not isinstance(overrides,dict) or len(overrides)>32:
        raise ValueError('per-turn policy budget')
    for key,override in overrides.items():
        if not isinstance(key,str) or not key.isdigit() or not 0<=int(key)<32 or not isinstance(override,dict):
            raise ValueError('per-turn policy schema')
        if set(override)-{'policy','candidates','input_tags','output_tags','temperature','top_p','max_output_tokens'}:
            raise ValueError('unsupported per-turn policy field')
        merged=dict(value,**override);merged.pop('turn_overrides',None)
        checked=normalize(merged)
        if checked['max_output_tokens']>value['max_output_tokens']:
            raise ValueError('per-turn policy exceeds signed output budget')
    return value


def turn_config(config,turn_index):
    value=normalize(config)
    if type(turn_index) is not int or turn_index<0:
        raise ValueError('turn index')
    override=value.get('turn_overrides',{}).get(str(turn_index),{})
    value=dict(value,**override);value.pop('turn_overrides',None)
    return normalize(value)


def mrcr_candidates(messages):
    from .native_mrcr_public_policy import shell_command,parse_public_question
    questions=[m['content'].split('\n\n',1)[0].strip() for m in messages
        if m.get('role')=='user' and isinstance(m.get('content'),str) and m['content'].startswith('Prepend ')]
    if len(questions)!=1:raise ValueError('exact public MRCR question required')
    prefix,_,_=parse_public_question(questions[0])
    wrong_prefix=('1' if prefix[0]=='0' else '0')+prefix[1:]
    wrong_question=questions[0].replace(prefix,wrong_prefix,1)
    commands=[shell_command(questions[0]),shell_command(wrong_question)]
    return [json.dumps({'tool_call':{'name':'bash','arguments':{'command':c}}},separators=(',',':')) for c in commands]


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).parent/'native_mrcr_public_policy.py').read_bytes()+(Path(__file__).parent/'sample_harness.py').read_bytes()).hexdigest()


def _chat_render(tokenizer, messages, tools=(), config=None):
    # Explicit text tool protocol works for tokenizers without a native tool
    # template. Observations remain ordinary chat messages and are replayed.
    messages = [dict(m) for m in messages]
    if tools:
        description = ('Available tools: ' + json.dumps(list(tools), sort_keys=True, separators=(',', ':')) +
                       '\nCall a tool with exactly {"tool_call":{"name":"NAME","arguments":{...}}}.' +
                       ' Otherwise answer normally. Tool results follow as user messages.')
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] += '\n' + description
        else:
            messages.insert(0, {'role': 'system', 'content': description})
    value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return list(value['input_ids']) if hasattr(value, 'keys') else list(value)


def _text_action(text):
    result = {'text': text, 'tool_calls': []}
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return result
    if isinstance(value, dict) and set(value) == {'tool_call'}:
        call = value['tool_call']
        if isinstance(call, dict) and set(call) == {'name', 'arguments'} and isinstance(call['name'], str) and isinstance(call['arguments'], dict):
            result['tool_calls'] = [call]
    return result


def _sample(model, tokenizer, prompt, seed, config, messages=()):
    config = normalize(config)
    rng = torch.Generator().manual_seed(seed)
    if config['policy'] == 'public-mrcr-shell-candidates':
        config=dict(config,policy='candidates',candidates=mrcr_candidates(messages))
    if config['policy'] == 'visible-copy-candidates':
        opening,closing=config['input_tags']
        visible='\n'.join(str(m.get('content','')) for m in messages if m['role']=='user')
        start=visible.rfind(opening)
        if start<0 or closing not in visible[start+len(opening):]:
            raise ValueError('visible copy span missing')
        text=visible[start+len(opening):].split(closing,1)[0]
        before,after=config['output_tags']
        config=dict(config,policy='candidates',candidates=[before+text+after,before+text+'!'+after])
    if config['policy'] == 'candidates':
        # Score the complete candidate sequence under the target model. This
        # supports arbitrary text/tool choices, not an environment-specific digit.
        tokens = [tokenizer.encode(text, add_special_tokens=False) for text in config['candidates']]
        if any(not ids or len(ids) > config['max_output_tokens'] for ids in tokens):
            raise ValueError('candidate token budget')
        scores = []
        with torch.inference_mode():
            for ids in tokens:
                logits = model(torch.tensor([prompt+ids]), use_cache=False).logits[0, len(prompt)-1:len(prompt)+len(ids)-1]
                logprobs = torch.log_softmax(logits.float(), -1)
                scores.append(logprobs.gather(1, torch.tensor(ids)[:, None]).sum())
            distribution = torch.softmax(torch.stack(scores)/config['temperature'], -1)
        selected = int(torch.multinomial(distribution, 1, generator=rng))
        return tokens[selected]
    output = []
    eos = tokenizer.eos_token_id
    with torch.inference_mode():
        for _ in range(config['max_output_tokens']):
            logits = model(torch.tensor([prompt+output]), use_cache=False).logits[0, -1].float()/config['temperature']
            probs = torch.softmax(logits, -1)
            if config['top_p'] < 1:
                sorted_probs, indices = probs.sort(descending=True)
                mask = sorted_probs.cumsum(0)-sorted_probs > config['top_p']
                sorted_probs[mask] = 0
                probs = torch.zeros_like(probs).scatter(0, indices, sorted_probs)
                probs = probs/probs.sum()
            token = int(torch.multinomial(probs, 1, generator=rng))
            output.append(token)
            if token == eos:
                break
    return output


def _text_observations(messages, config):
    """Portable text-tools-v1 observation rendering; other wire protocols reject."""
    normalize(config)
    result=[]
    for message in messages:
        if not isinstance(message,dict) or message.get('role') not in ('system','user','assistant','tool') or not isinstance(message.get('content'),str):
            raise ValueError('unsupported observation wire type')
        if message['role']=='tool':
            result.append(dict(role='user',content='Tool result: '+message['content']))
        else:
            result.append(dict(role=message['role'],content=message['content']))
    return result


def plain_render(messages,tools=()):
    """Distinct text harness for models without a chat template."""
    blocks=[]
    if tools:
        blocks.append("SYSTEM: Available tools: "+json.dumps(list(tools),sort_keys=True,separators=(',',':'))+
            '\nCall with exactly {"tool_call":{"name":"NAME","arguments":{...}}}.')
    for message in messages:
        if message.get('role') not in ('system','user','assistant') or not isinstance(message.get('content'),str):
            raise ValueError('plain harness unsupported message')
        blocks.append(message['role'].upper()+': '+message['content'])
    return '\n\n'.join(blocks)+'\n\nASSISTANT: '


def _plain_render(tokenizer,messages,tools=(),config=None):
    return tokenizer.encode(plain_render(messages,tools),add_special_tokens=False)


def _window_chat_render(tokenizer,messages,tools=(),config=None):
    """Bound visible history; the complete trajectory remains replay-verified.

    Preserve the signed number of initial task messages and the most recent
    complete messages. Nothing truncates an observation or rewrites its text.
    An individually oversized task/observation still fails the model budget.
    """
    config=normalize(config)
    prefix=config['history_prefix_messages'];window=config['history_window_messages']
    retained=list(messages[:prefix])+list(messages[prefix:][-window:])
    return _chat_render(tokenizer,retained,tools,config)


def _format_chat_render(tokenizer,messages,tools=(),config=None):
    """Render an explicit signed format instruction without rewriting responses."""
    config=normalize(config)
    copied=[dict(m) for m in messages]
    last=next((i for i in range(len(copied)-1,-1,-1) if copied[i].get('role')=='user'),None)
    if last is None or not isinstance(copied[last].get('content'),str):
        raise ValueError('response format requires public text user message')
    copied[last]['content']+='\n\nResponse format: '+config['response_format_instruction']
    return _chat_render(tokenizer,copied,tools,config)


def _kv_sample(model,tokenizer,prompt,seed,config,messages=()):
    from .cached_sampling import sample as cached_sample
    return cached_sample(model,prompt,seed=seed,max_output_tokens=config['max_output_tokens'],temperature=config['temperature'],top_p=config['top_p'],eos_token_id=tokenizer.eos_token_id,mode='kv-last-logits-v1')[0]


# Every version owns its render/action/observation/sample boundary. Core model
# computation does not branch on environment or harness names.
HARNESS_REGISTRY={
    'text-tools-long-kv-v3': {'render':_chat_render,'action':_text_action,'observations':_text_observations,'sample':_kv_sample},
    'text-tools-format-long-v1': {'render':_format_chat_render,'action':_text_action,'observations':_text_observations,'sample':_sample},
    'text-tools-long-v2': {'render':_chat_render,'action':_text_action,'observations':_text_observations,'sample':_sample},
    'text-tools-v1': {'render':_chat_render,'action':_text_action,'observations':_text_observations,'sample':_sample},
    'plain-transcript-v1': {'render':_plain_render,'action':_text_action,'observations':_text_observations,'sample':_sample},
    'text-tools-window-v1': {'render':_window_chat_render,'action':_text_action,'observations':_text_observations,'sample':_sample},
}


def render(tokenizer,messages,tools=(),config=None):
    config=normalize(config)
    return HARNESS_REGISTRY[config['version']]['render'](tokenizer,messages,tools,config)


def action(text,config=None):
    config=normalize(config)
    return HARNESS_REGISTRY[config['version']]['action'](text)


def observations(messages,config=None):
    config=normalize(config)
    return HARNESS_REGISTRY[config['version']]['observations'](messages,config)


def sample(model,tokenizer,prompt,seed,config,messages=(),turn_index=0):
    config=turn_config(config,turn_index)
    return HARNESS_REGISTRY[config['version']]['sample'](model,tokenizer,prompt,seed,config,messages)
