"""The gate probe's prompt, sampling settings and grader, importable (no GPU needed for grading).

Lines below the marker are copied VERBATIM from ops/performance_eval/perf_eval.py (lines 13-96), the public reproduction
of the probe pod's live grader (ops/probe_pod/probe_worker.py grade(): subnet.mimo_answer_projection final-channel
projection + verify.py semantics with math-verify 0.9.0, then the lenient fix allan-lenient-v1-786cdb9 whose source is
allan_src.py, byte-identical to probe_worker.py _ALLAN_SRC). tests/test_kit_cpu.py checks the copy is verbatim.

  init(head_dir)                      load the head tokenizer + lenient grader (call once per process)
  render(problem) -> prompt token ids the gate's prompt: [system: SYSTEM, user: problem], enable_thinking=True
  grade(out_ids, stop, gold, problem) -> (kind, how); kind 'positive' = correct, 'negative' = wrong,
                                         'neutral' = unresolved (capped or no single final \\boxed), 'indeterminate'
  H                                   the gate harness: temperature 1.0, top_p 0.95, top_k 20, 8192 output / 12288 context
  seed_of(P, set, task, j)            per-request seed rule (the gate uses sha256(P:env:task:j)[:12] with env='affine_math')
"""
import hashlib, json, os
from pathlib import Path
HERE = str(Path(__file__).resolve().parent.parent / 'validator' / 'probe')   # allan_src.py lives there


def init(head_dir):
    _init(str(head_dir))


# ---- verbatim from ops/performance_eval/perf_eval.py lines 13-96 ----
EOS = (248046, 248044); OPEN_T, CLOSE_T = 248068, 248069
SYSTEM = ("Solve the math problem. Reason step by step in plain text, then end your response with the final answer in `\\boxed{}`. "
          "Emit exactly one `\\boxed{}` block and nothing after it.")
H = dict(enable_thinking=True, max_context_tokens=12288, max_output_tokens=8192, temperature=1.0, top_k=20, top_p=0.95)
_G = {}
def _init(ckpt_dir):
    os.nice(5)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt_dir); _G['tok'] = tok
    assert tok.convert_tokens_to_ids('<think>') == OPEN_T and tok.convert_tokens_to_ids('</think>') == CLOSE_T
    _G['special'] = set(getattr(tok, 'all_special_ids', ()))
    ns = {'__name__': 'allan_lenient_v1'}
    exec(compile('import re\nfrom math_verify import parse, verify\n' + open(os.path.join(HERE, 'allan_src.py')).read(), 'allan_lenient_v1', 'exec'), ns)
    _G['allan'] = ns
def _boxed_final(text):   # subnet/mimo_answer_projection._boxed_final (verbatim logic)
    if text.count('\\boxed{') != 1: return None
    begin = text.index('\\boxed{'); start = begin + len('\\boxed{'); depth = 1
    for at in range(start, len(text)):
        if text[at] == '{': depth += 1
        elif text[at] == '}':
            depth -= 1
            if depth == 0:
                if not text[start:at].strip(): return None
                tail = text[at + 1:].strip(); prefix = text[:begin].rstrip()
                if tail:
                    opening = {'\\]': '\\[', '\\)': '\\(', '$$': '$$', '$': '$'}.get(tail)
                    if opening is None or not prefix.endswith(opening): return None
                elif prefix.endswith(('\\[', '\\(', '$')): return None
                return text[begin:at + 1]
    return None
def project(out):   # subnet/mimo_answer_projection.project, enable_thinking=True, stop == 'eos'
    tok = _G['tok']
    if not out or out[-1] not in EOS: return None
    body = list(out[:-1])
    if any(t in _G['special'] or t in EOS for t in body): return None
    if body.count(OPEN_T) != 1 or body.count(CLOSE_T) != 1: return None
    a = body.index(OPEN_T); b = body.index(CLOSE_T)
    if a >= b or tok.decode(body[:a], skip_special_tokens=False).strip(): return None
    decoded = [tok.decode(ids, skip_special_tokens=False) for ids in (body[:a], body[a + 1:b], body[b + 1:])]
    if any('<think>' in t or '</think>' in t for t in decoded): return None
    return _boxed_final(decoded[-1].strip())
def _last_boxed(text):   # verify.py last_boxed
    found = None; O = "\\boxed{"; start = text.find(O)
    while start != -1:
        depth = 0
        for i in range(start + len(O) - 1, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0: found = text[start:i + 1]; break
        start = text.find(O, start + len(O))
    return found
def native(gold, reply):   # verify.py semantics: True/False; raises -> indeterminate (verify.py exit != 0)
    from math_verify import parse, verify
    g = parse("\\boxed{" + gold + "}", raise_on_error=True)
    if not g: raise RuntimeError('grader_reference_unparseable')
    pred = _last_boxed(reply)
    if pred is None: return False
    return bool(verify(g, parse(pred, raise_on_error=True), raise_on_error=True))
def grade(out, stop, gold, prompt):
    """-> (kind, how). kind in positive/negative/neutral/indeterminate (probe_worker._grade for live-math)."""
    try:
        if stop != 'eos': return 'neutral', 'capped'
        box = project(out)
        try: ok = native(gold, box or '')
        except Exception as x: return 'indeterminate', type(x).__name__
        kind = 'neutral' if box is None else ('positive' if ok else 'negative'); how = 'live'
        if kind != 'positive':
            try:
                ns = _G['allan']; text = _G['tok'].decode(out[:-1], skip_special_tokens=False)
                gbox, viol, pfix = ns['lenient_project'](text)
                if gbox is not None:
                    ok2, efix = ns['lenient_equal'](gold, gbox, prompt)
                    if ok2: kind = 'positive'; how = '+'.join(x for x in (pfix, efix) if x) or 'live'
            except Exception: pass
        return kind, how
    except Exception as x:
        return 'indeterminate', 'err ' + type(x).__name__
def render(prompt):
    tok = _G['tok']
    ids = tok.apply_chat_template([dict(role='system', content=SYSTEM), dict(role='user', content=prompt)], tokenize=True,
                                  return_dict=False, add_generation_prompt=True, enable_thinking=True)
    return list(ids['input_ids'] if isinstance(ids, dict) else ids)
def seed_of(P, s, tid, j): return int(hashlib.sha256(f"{P}:{s}:{tid}:{j}".encode()).hexdigest()[:12], 16)
