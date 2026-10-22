"""Local, pure-python replica of the SN120 MiMo math outcome rule (no network verifiers).

Mirrors, verbatim in logic:
  * subnet/mimo_answer_projection.py  (mimo-final-channel-math-v1: exactly one <think>..</think>,
    no control tokens, final channel must hold exactly one balanced \\boxed{} and nothing after)
  * subnet/complete_group_runtime.py::outcome/completed_answer (budget stop or bad answer -> neutral)
  * subnet/vendor/legacy/.../affine_math_v1/verify.py (math-verify 0.9.0 parse/verify of the LAST
    \\boxed{} vs '\\boxed{'+gold+'}')
  * overlong soft penalty from the live v8.2.4 training config (mimo_grpo/config.py):
    reward = float(correct) - overlong_penalty * clip((len - soft_start)/(max_tokens - soft_start), 0, 1)
Classification: positive (correct), negative (completed wrong), neutral (unresolved/budget/format),
indeterminate (grader error/timeout; excluded from means, never scored as 0).
"""
from concurrent.futures import ProcessPoolExecutor
import os

EOS = (248046, 248044)
THINK_OPEN, THINK_CLOSE = 248068, 248069
OPEN = '\\boxed{'
OVERLONG = dict(soft_start=6000, max_tokens=8192, penalty=0.5)


def _boxed_final(text):
    if text.count('\\boxed{') != 1:
        return None
    begin = text.index('\\boxed{'); start = begin + len('\\boxed{'); depth = 1
    for at in range(start, len(text)):
        if text[at] == '{':
            depth += 1
        elif text[at] == '}':
            depth -= 1
            if depth == 0:
                if not text[start:at].strip():
                    return None
                tail = text[at + 1:].strip(); prefix = text[:begin].rstrip()
                if tail:
                    opening = {'\\]': '\\[', '\\)': '\\(', '$$': '$$', '$': '$'}.get(tail)
                    if opening is None or not prefix.endswith(opening):
                        return None
                elif prefix.endswith(('\\[', '\\(', '$')):
                    return None
                return text[begin:at + 1]
    return None


def completed_answer(text):
    if type(text) is not str or text.count('\\boxed{') != 1:
        return False
    start = text.index('\\boxed{') + len('\\boxed{'); depth = 1
    for at in range(start, len(text)):
        if text[at] == '{':
            depth += 1
        elif text[at] == '}':
            depth -= 1
            if depth == 0:
                return bool(text[start:at].strip()) and not text[at + 1:].strip()
    return False


def project(tokenizer, output, special_ids, vocab_size):
    """Return (grading_text or None, reason)."""
    if not output or output[-1] not in EOS:
        return None, 'not-completed-eos'
    body = list(output[:-1])
    if any(t in special_ids or t in EOS or t >= vocab_size for t in body):
        return None, 'unexpected-control-token'
    if body.count(THINK_OPEN) != 1 or body.count(THINK_CLOSE) != 1:
        return None, 'thinking-marker-count'
    a = body.index(THINK_OPEN); b = body.index(THINK_CLOSE)
    if a >= b or tokenizer.decode(body[:a], skip_special_tokens=False).strip():
        return None, 'thinking-marker-order'
    chunks = [body[:a], body[a + 1:b], body[b + 1:]]
    decoded = [tokenizer.decode(ids, skip_special_tokens=False) for ids in chunks]
    if any('<think>' in t or '</think>' in t for t in decoded):
        return None, 'noncanonical-thinking-marker'
    boxed = _boxed_final(decoded[-1].strip())
    if boxed is None:
        return None, 'missing-complete-final-answer'
    return boxed, 'completed-final-answer'


def _last_boxed(text):
    found = None; start = text.find(OPEN)
    while start != -1:
        depth = 0
        for i in range(start + len(OPEN) - 1, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    found = text[start:i + 1]; break
        start = text.find(OPEN, start + len(OPEN))
    return found


def math_verify(gold, reply):
    """verify.py core. Returns 1.0 / 0.0, or None (indeterminate)."""
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException
    pred = _last_boxed(reply)
    try:
        g = parse(OPEN + gold + '}', raise_on_error=True)
        if not g:
            return None
        if pred is None:
            return 0.0
        p = parse(pred, raise_on_error=True)
        return 1.0 if verify(g, p, raise_on_error=True) else 0.0
    except TimeoutException:
        return None
    except Exception:
        return None


def overlong_reward(correct, length):
    o = OVERLONG
    frac = min(max((length - o['soft_start']) / (o['max_tokens'] - o['soft_start']), 0.0), 1.0)
    return float(correct) - o['penalty'] * frac


def classify(grading_text, gold):
    """Pool worker: (classification, correct) for one sample. grading_text None => neutral."""
    if grading_text is None:
        return 'neutral', 0.0
    v = math_verify(gold, grading_text)
    if v is None:
        return 'indeterminate', None
    if not completed_answer(grading_text):
        return 'neutral', 0.0
    return ('positive', 1.0) if v == 1.0 else ('negative', 0.0)


def pool(workers=None):
    return ProcessPoolExecutor(max_workers=workers or min(16, os.cpu_count() or 4))
