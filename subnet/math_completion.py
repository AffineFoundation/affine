"""Prospective MATH outcome semantics; historical specs retain their grading.

An exhausted output budget is a stopping condition, not a mathematical label.
Only an explicit, complete final answer is eligible for binary outcome grading.
This module checks answer framing, not mathematical correctness or sampling.
"""
import hashlib
from pathlib import Path

FIELD = 'math_outcome_policy'
VERSION = 'completed-boxed-math-outcome-v1'
ENVIRONMENT_VERSION = 'prime-v1-2-completed-math'
OPEN = '\\boxed{'


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def enabled(spec):
    if hasattr(spec, 'to_dict'):
        spec = spec.to_dict()
    value = spec.get('config', {}).get(FIELD)
    if value is None:
        return False
    if (value != VERSION or spec.get('id') != 'affine_math'
            or spec.get('adapter', 'prime_v1') != 'prime_v1'
            or spec.get('max_turns') != 1
            or spec.get('version') != ENVIRONMENT_VERSION):
        raise ValueError('explicit completed-answer MATH environment contract')
    return True


def final_box(text):
    """Return the last answer only if its latest boxed expression is complete.

    A dangling later box cannot silently fall back to an earlier answer. EOS,
    response length and Unicode density are deliberately not correctness tests.
    The native mathematical grader remains responsible for interpreting the body.
    """
    if not isinstance(text, str):
        raise ValueError('decoded mathematical response text required')
    at = text.rfind(OPEN)
    if at < 0:
        return None
    start = at + len(OPEN)
    depth = 1
    for position in range(start, len(text)):
        character = text[position]
        if character == '{':
            depth += 1
        elif character == '}':
            depth -= 1
            if depth == 0:
                body = text[start:position]
                return body if body.strip() else None
    return None


def unresolved(spec, text):
    return enabled(spec) and final_box(text) is None
