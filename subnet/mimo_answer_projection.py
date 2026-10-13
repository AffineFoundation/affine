"""Versioned native math grading for MiMo's explicit thinking channel.

Only the final assistant channel supplies an answer to the environment. The
original prompt/output tokens remain unchanged for sampling proofs and training.
This is an outcome rule, not a claim that the reasoning is correct.
"""
from copy import deepcopy

POLICY_KEY = 'mimo_answer_projection'
VERSION = 'mimo-final-channel-math-v1'
POLICY = dict(version=VERSION, thinking_open_token_id=248068,
              thinking_close_token_id=248069)


def policy(manifest):
    value = manifest.get(POLICY_KEY)
    if value is None:
        return None  # Historical manifests retain their original outcome rule.
    from .complete_group_distribution import MIMO_VERSION
    if (type(value) is not dict or value != POLICY
            or any(type(value.get(k)) is not int for k in ('thinking_open_token_id', 'thinking_close_token_id'))
            or manifest.get('sampling_contract', {}).get('version') != MIMO_VERSION):
        raise ValueError('exact versioned MiMo final-channel grading policy')
    return deepcopy(value)


def _boxed_final(text):
    """Extract exactly one balanced final box, accepting paired math wrappers."""
    if text.count('\\boxed{') != 1:
        return None
    begin = text.index('\\boxed{')
    start = begin + len('\\boxed{')
    depth = 1
    for at in range(start, len(text)):
        if text[at] == '{':
            depth += 1
        elif text[at] == '}':
            depth -= 1
            if depth == 0:
                if not text[start:at].strip():
                    return None
                tail = text[at+1:].strip()
                prefix = text[:begin].rstrip()
                if tail:
                    opening = {'\\]': '\\[', '\\)': '\\(', '$$': '$$', '$': '$'}.get(tail)
                    if opening is None or not prefix.endswith(opening):
                        return None
                elif prefix.endswith(('\\[', '\\(', '$')):
                    return None  # Do not accept an unclosed display wrapper.
                return text[begin:at+1]
    return None


def project(manifest, harness, tokenizer, output, stop):
    """Return a final-channel grading view, never rewritten trajectory tokens."""
    selected = policy(manifest)
    if selected is None:
        raise ValueError('explicit MiMo final-channel policy required')
    if (harness.get('version') != 'text-tools-mimo-complete-group-v1'
            or type(harness.get('enable_thinking')) is not bool):
        raise ValueError('exact MiMo thinking harness required')
    opening = selected['thinking_open_token_id']; closing = selected['thinking_close_token_id']
    if (tokenizer.convert_tokens_to_ids('<think>') != opening
            or tokenizer.convert_tokens_to_ids('</think>') != closing):
        raise ValueError('checkpoint tokenizer thinking tokens differ from signed policy')
    from .complete_group_distribution import MIMO_EOS
    def unresolved(reason):
        return dict(complete=False, reason=reason, final_text='', grading_text='')
    if stop != 'eos' or not output or output[-1] not in MIMO_EOS:
        return unresolved('not-completed-eos')
    body = list(output[:-1])
    special = set(getattr(tokenizer, 'all_special_ids', ()))
    if any(t in special or t in MIMO_EOS for t in body):
        return unresolved('unexpected-control-token')
    if harness['enable_thinking']:
        if body.count(opening) != 1 or body.count(closing) != 1:
            return unresolved('thinking-marker-count')
        a = body.index(opening); b = body.index(closing)
        if a >= b or tokenizer.decode(body[:a], skip_special_tokens=False).strip():
            return unresolved('thinking-marker-order')
        chunks = [body[:a], body[a+1:b], body[b+1:]]
    else:
        # The pinned no-thinking chat template already emits <think></think>.
        if opening in body or closing in body:
            return unresolved('unexpected-thinking-channel')
        chunks = [body]
    decoded = [tokenizer.decode(ids, skip_special_tokens=False) for ids in chunks]
    if any('<think>' in text or '</think>' in text for text in decoded):
        return unresolved('noncanonical-thinking-marker')
    final = decoded[-1].strip()
    boxed = _boxed_final(final)
    if boxed is None:
        return unresolved('missing-complete-final-answer')
    return dict(complete=True, reason='completed-final-answer', final_text=final,
                grading_text=boxed)


def grade(session, manifest, harness, tokenizer, output, stop):
    """One native-grader path shared by miner, independent grader and audit."""
    from . import harness as action_policy
    from .complete_group_runtime import outcome
    text = tokenizer.decode(output, skip_special_tokens=True)
    if policy(manifest) is None:
        native = session.step(action_policy.action(text, harness))
        kind, reward = outcome(native, text, stop)
    else:
        projected = project(manifest, harness, tokenizer, output, stop)
        native = session.step(action_policy.action(projected['grading_text'], harness))
        kind, reward = outcome(native, projected['grading_text'], stop)
        if not projected['complete']:
            kind, reward = 'neutral', 0.
    return native, kind, reward
