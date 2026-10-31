"""SN120 miner batch file, format "affine-batch-v1" (free-file protocol, Jacob 2026-10-30 pt898u).

A miner's batch is ONE JSON object:
    {"version": "affine-batch-v1",
     "hotkey": "<the miner's ss58 hotkey>",
     "groups": [{"prompt": "<plain text>",
                 "rollouts": [{"output": "<completion text>", "reward": <number>}, ...]}, ...]}

Only the format is checked. Rewards are used exactly as given (no regrade). Rules:
  * exact keys, UTF-8 JSON, file <= MAX_FILE_BYTES; prompts distinct and non-empty; >= 1 rollout per group;
    rewards finite numbers (not booleans).
  * groups beyond the live batch cap (BATCH-CAP.json: tasks and rollouts) are dropped whole, in file order;
    everything kept must be valid, otherwise the whole file is invalid.
  * prompt = the text as a single user message in the model's chat template with the generation prompt
    (tokenizer.apply_chat_template(..., add_generation_prompt=True)); output = tokenizer.encode(output,
    add_special_tokens=False). Neither may contain a chat control token (<|im_start|>, <|im_end|>, <|endoftext|>).
  * EOS: an output of <= MAX_OUTPUT-1 tokens gets <|im_end|> appended (stop "eos"); exactly MAX_OUTPUT tokens is
    a length-capped rollout (stop "max_tokens", no EOS); longer is invalid. prompt + output <= MAX_CONTEXT tokens.

Format "affine-batch-v2" (Jacob 2026-10-31 pt902u, RA/20261031-ff-train-params) = v1 + ONE optional top-level key
    "train": {"lr": <number>, "kl": <number>}      (either field may be omitted; an omitted field takes its default)
  * lr in [1.25e-7, 1e-6] (default 5e-7), kl in [0.005, 0.1] (default 0.02), bounds inclusive; a value out of range,
    non-numeric (incl. booleans/null/strings), non-finite, or any other key inside "train" makes the whole file invalid
    (no clamping). "train" may be {} (all defaults). v1 files are unchanged and still accepted (they train with the
    defaults); "train" in a v1 file is invalid (v1 keys stay exact). v2 is accepted only while the validator enables it
    (FF-TRAIN.json); otherwise a v2 file is invalid.
"""
import hashlib, json, math
from pathlib import Path

FORMAT = 'affine-batch-v1'
FORMAT_V2 = 'affine-batch-v2'
FORMATS = (FORMAT, FORMAT_V2)
TRAIN_DEFAULTS = {'lr': 5e-07, 'kl': 0.02}
TRAIN_BOUNDS = {'lr': (1.25e-07, 1e-06), 'kl': (0.005, 0.1)}
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_OUTPUT = 8192          # output tokens incl. the appended EOS
MAX_CONTEXT = 12288        # prompt + output tokens
EOS = 248046               # <|im_end|>
CONTROL = frozenset({248044, 248045, 248046})   # <|endoftext|> <|im_start|> <|im_end|>


class Invalid(ValueError):
    pass


def need(ok, why):
    if not ok:
        raise Invalid(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def _reject_constant(name):
    raise Invalid('non-finite number ' + name)


def train_params(value):
    """The v2 "train" object -> (effective {lr, kl} with defaults, the keys the file set). Raises Invalid; never clamps."""
    need(isinstance(value, dict), 'train must be an object {"lr": number, "kl": number} (either may be omitted)')
    extra = sorted(set(value) - set(TRAIN_BOUNDS))
    need(not extra, 'train keys must be among lr, kl (got %s)' % ', '.join(map(repr, extra))[:100])
    out = dict(TRAIN_DEFAULTS)
    for k, (lo, hi) in TRAIN_BOUNDS.items():
        if k in value:
            v = value[k]
            need(type(v) in (int, float) and math.isfinite(v) and lo <= v <= hi,
                 'train.%s must be a number in [%r, %r] (got %s)' % (k, lo, hi, json.dumps(v)[:40]))
            out[k] = float(v)
    return out, sorted(k for k in TRAIN_BOUNDS if k in value)


def parse_full(raw, hotkey, v2=True):
    """parse() + the format version and the training params: -> (groups, version, train, train_given)."""
    need(isinstance(raw, (bytes, bytearray)) and 0 < len(raw) <= MAX_FILE_BYTES, 'file must be 1..%d bytes' % MAX_FILE_BYTES)
    try:
        value = json.loads(bytes(raw).decode('utf-8'), parse_constant=_reject_constant)
    except Invalid:
        raise
    except Exception as e:
        raise Invalid('not UTF-8 JSON: ' + type(e).__name__)
    need(isinstance(value, dict), 'top level keys must be exactly version, hotkey, groups')
    version = value.get('version')
    if version == FORMAT_V2 and v2:
        need({'version', 'hotkey', 'groups'} <= set(value) <= {'version', 'hotkey', 'groups', 'train'},
             'top level keys must be version, hotkey, groups and optionally train (%s)' % FORMAT_V2)
        train, given = train_params(value['train']) if 'train' in value else (dict(TRAIN_DEFAULTS), [])
    else:
        need(set(value) == {'version', 'hotkey', 'groups'},
             'top level keys must be exactly version, hotkey, groups' + (' (%s; "train" needs %s)' % (FORMAT, FORMAT_V2) if v2 and 'train' in value else ''))
        need(version == FORMAT, 'version must be ' + (' or '.join(FORMATS) if v2 else FORMAT))
        train, given = dict(TRAIN_DEFAULTS), []
    return _groups(value, hotkey), version, train, given


def parse(raw, hotkey):
    """Structural checks only (no tokenizer). Returns the list of groups as (prompt, [(output, reward)])."""
    need(isinstance(raw, (bytes, bytearray)) and 0 < len(raw) <= MAX_FILE_BYTES, 'file must be 1..%d bytes' % MAX_FILE_BYTES)
    try:
        value = json.loads(bytes(raw).decode('utf-8'), parse_constant=_reject_constant)
    except Invalid:
        raise
    except Exception as e:
        raise Invalid('not UTF-8 JSON: ' + type(e).__name__)
    need(isinstance(value, dict) and set(value) == {'version', 'hotkey', 'groups'}, 'top level keys must be exactly version, hotkey, groups')
    need(value['version'] == FORMAT, 'version must be ' + FORMAT)
    return _groups(value, hotkey)


def _groups(value, hotkey):
    need(value['hotkey'] == hotkey, 'hotkey does not match the object owner')
    groups = value['groups']
    need(isinstance(groups, list) and groups, 'groups must be a non-empty list')
    out, seen = [], set()
    for i, g in enumerate(groups):
        need(isinstance(g, dict) and set(g) == {'prompt', 'rollouts'}, 'groups[%d] keys must be exactly prompt, rollouts' % i)
        p, rs = g['prompt'], g['rollouts']
        need(isinstance(p, str) and p.strip(), 'groups[%d].prompt must be a non-empty string' % i)
        need(p not in seen, 'groups[%d].prompt duplicates an earlier prompt' % i); seen.add(p)
        need(isinstance(rs, list) and rs, 'groups[%d].rollouts must be a non-empty list' % i)
        rows = []
        for j, r in enumerate(rs):
            need(isinstance(r, dict) and set(r) == {'output', 'reward'}, 'groups[%d].rollouts[%d] keys must be exactly output, reward' % (i, j))
            o, w = r['output'], r['reward']
            need(isinstance(o, str), 'groups[%d].rollouts[%d].output must be a string' % (i, j))
            need(type(w) in (int, float) and math.isfinite(w), 'groups[%d].rollouts[%d].reward must be a finite number' % (i, j))
            rows.append((o, float(w)))
        out.append((p, rows))
    return out


def cap(groups, max_tasks, max_rollouts):
    """Keep whole groups in file order while both caps hold. Returns (kept, dropped_groups)."""
    kept, n = [], 0
    for g in groups:
        if len(kept) == max_tasks or n + len(g[1]) > max_rollouts:
            break
        kept.append(g); n += len(g[1])
    need(kept, 'the first group alone exceeds the rollout cap')
    return kept, len(groups) - len(kept)


def tokenizer(path, pins=None):
    """The pinned head tokenizer (files under `path`; optional {file: sha256} pins)."""
    path = Path(path)
    for name, digest in (pins or {}).items():
        need(sha((path / name).read_bytes()) == digest, 'tokenizer file pin ' + name)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def render(tok, prompt):
    ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True)
    return list(ids['input_ids'] if hasattr(ids, 'keys') else ids)


def tokenize(tok, groups):
    """[(prompt, [(output, reward)])] -> [{'prompt': text, 'prompt_tokens': [...], 'rollouts': [{'tokens','reward','stop'}]}]."""
    outputs = [o for _, rs in groups for o, _ in rs]
    encoded = tok(outputs, add_special_tokens=False)['input_ids'] if outputs else []
    k, result = 0, []
    for i, (p, rs) in enumerate(groups):
        need(not CONTROL & set(tok.encode(p, add_special_tokens=False)), 'groups[%d].prompt contains a chat control token' % i)
        pt = render(tok, p)
        rows = []
        for j, (_, w) in enumerate(rs):
            ot = list(encoded[k]); k += 1
            need(not CONTROL & set(ot), 'groups[%d].rollouts[%d].output contains a chat control token' % (i, j))
            need(len(ot) <= MAX_OUTPUT, 'groups[%d].rollouts[%d].output is %d tokens > %d' % (i, j, len(ot), MAX_OUTPUT))
            stop = 'max_tokens' if len(ot) == MAX_OUTPUT else 'eos'
            if stop == 'eos':
                ot.append(EOS)
            need(len(pt) + len(ot) <= MAX_CONTEXT, 'groups[%d].rollouts[%d]: prompt+output %d tokens > %d' % (i, j, len(pt) + len(ot), MAX_CONTEXT))
            rows.append(dict(tokens=ot, reward=w, stop=stop))
        result.append(dict(prompt=p, prompt_tokens=pt, rollouts=rows))
    return result


def validate(raw, hotkey, tok, max_tasks, max_rollouts, v2=False):
    """Full check. Returns dict(groups=[tokenized], summary) or raises Invalid. v2: accept affine-batch-v2 (train params)."""
    parsed, version, train, given = parse_full(raw, hotkey, v2)
    groups, dropped = cap(parsed, max_tasks, max_rollouts)
    out = tokenize(tok, groups)
    return dict(groups=out, summary=dict(file_sha256=sha(bytes(raw)), bytes=len(raw), groups=len(out), format=version,
                                         train=train, train_given=given,
                                         rollouts=sum(len(g['rollouts']) for g in out), dropped_groups=dropped,
                                         capped_rollouts=sum(r['stop'] == 'max_tokens' for g in out for r in g['rollouts'])))


def check(raw, hotkey, tok, max_tasks, max_rollouts, v2=False):
    """validate() as a verdict: (True, summary, groups) or (False, reason, None)."""
    try:
        v = validate(raw, hotkey, tok, max_tasks, max_rollouts, v2)
        return True, v['summary'], v['groups']
    except Invalid as e:
        return False, str(e), None
