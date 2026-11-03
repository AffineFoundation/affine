"""Validate a synthetic 12K batch file with the LIVE ff_format.py (sha printed): outputs of 12287 (eos), 12288 (capped), 8192 (legacy capped), 12289 (invalid)."""
import sys, json, hashlib, importlib.util
F, TOK, HK = sys.argv[1], sys.argv[2], '5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY'
s = importlib.util.spec_from_file_location('ff_format', F); ff = importlib.util.module_from_spec(s); s.loader.exec_module(ff)
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(TOK)
unit = ' alpha'; assert len(t.encode(unit * 50, add_special_tokens=False)) == 50
def out(n): return unit * n
def run(ns):
    G = [dict(prompt='Compute %d + %d. Put the answer in \\boxed{}.' % (i, i), rollouts=[dict(output=out(n), reward=1.0)]) for i, n in enumerate(ns)]
    raw = json.dumps(dict(version='affine-batch-v1', hotkey=HK, groups=G)).encode()
    try:
        groups = ff.parse(raw, HK); tg = ff.tokenize(ff.tokenizer(TOK), groups)
        return 'accepted', tg
    except Exception as e: return 'rejected: %s' % e, None
print(json.dumps(dict(ff_format_sha256=hashlib.sha256(open(F, 'rb').read()).hexdigest(), MAX_OUTPUT=ff.MAX_OUTPUT, MAX_CONTEXT=ff.MAX_CONTEXT)))
r, tg = run([12287, 12288, 8192, 5000]); print('12287/12288/8192/5000:', r)
if tg is not None:
    print(json.dumps([[len(x) for x in [g]] and str(g)[:0] for g in []]))
    try: print([(r_.get('stop') if isinstance(r_, dict) else getattr(r_, 'stop', None)) for g in tg for r_ in (g['rollouts'] if isinstance(g, dict) else g.rollouts)])
    except Exception as e: print('stops: n/a', type(tg), str(tg)[:200])
print('12289:', run([12289])[0])
