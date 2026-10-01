"""Prospective public-schema proposals with common wrong-type mutations.

This never reads a grader. Native grading must separately establish whether a
mutation actually fails a particular schema. Candidate token probability is
measured before treating this proposal policy as useful for model mining.
"""
import json
from .public_pydantic import proposals as original_proposals
REVISION = 'public-pydantic-same-key-wrong-type-v2'


def proposals(messages):
    positive = original_proposals(messages)[0]
    opening = '```json\n'
    closing = '\n```'
    if not positive.startswith(opening) or not positive.endswith(closing):
        raise ValueError('bounded public JSON proposal framing')
    value = json.loads(positive[len(opening):-len(closing)])
    if not isinstance(value, dict) or not value:
        raise ValueError('nonempty public root object')
    # Keep the publicly declared key and original insertion order. Deliberately
    # change its type; the operator's native control, not this generator, decides
    # whether Pydantic coercion or Optional semantics permits the mutation.
    key = next(iter(value))
    first = value[key]
    value[key] = [] if not isinstance(first, list) else {}
    negative = opening + json.dumps(value, separators=(',', ':'), ensure_ascii=False) + closing
    return [positive, negative]
