"""Project structurally admitted documents and trusted native grading to GRPO.

The native results argument must come from the pinned native grader, never from
the miner's document. This helper deliberately does not run sampling audits.
"""
from .data import TaskGroup, Trajectory, digest


def project_document(document, *, expected_document_sha256, binding, trusted_prompt,
                     native_results, vocab_size, eos_token_ids, group_size=8):
    binding.validate()
    if digest(document) != expected_document_sha256:
        raise ValueError('original committed document content required')
    if (document.get('version') != 'complete-group-training-document-v1' or
            document.get('epoch') != binding.epoch or document.get('checkpoint') != binding.checkpoint):
        raise ValueError('committed document bound to training input')
    batch = document['batch']
    if len(batch['rollouts']) != group_size or len(native_results) != group_size:
        raise ValueError('native result for every original nonce required')
    result = []
    for nonce, (rollout, native) in enumerate(zip(batch['rollouts'], native_results, strict=True)):
        if rollout.get('seed') != nonce or type(rollout.get('seed')) is not int:
            raise ValueError('canonical original attempt order')
        if type(native.get('nonce')) is not int or native['nonce'] != nonce:
            raise ValueError('native result must retain original nonce order')
        if len(rollout['turns']) != 1:
            raise ValueError('single-turn math training projection')
        turn = rollout['turns'][0]
        if native.get('output_sha256') != digest(turn['output']):
            raise ValueError('native output digest differs from original commitment')
        if (tuple(native['prompt']) != tuple(trusted_prompt) or native['prompt'] != turn['prompt'] or
                native['output'] != turn['output'] or native['task_hash'] != batch['task_hash'] or
                rollout['task_hash'] != batch['task_hash']):
            raise ValueError('native task/prompt/output projection differs from commitment')
        kind = native['classification']
        if kind not in ('positive', 'negative', 'neutral'):
            raise ValueError('grader error is not a training outcome')
        reward = 1. if kind == 'positive' else 0.
        if native['reward'] != reward or native['done'] is not True:
            raise ValueError('native completed/unresolved reward semantics')
        if any(turn.get(k) != native[k] for k in ('classification', 'reward', 'stop_reason', 'done', 'observations')):
            raise ValueError('native outcome claim mismatch')
        if rollout.get('classification') != kind or rollout.get('reward') != reward:
            raise ValueError('rollout native outcome claim mismatch')
        if native['stop_reason'] == 'budget':
            if kind != 'neutral':
                raise ValueError('budget caps must be unresolved')
            stop = 'max_tokens'
        elif native['stop_reason'] == 'eos':
            stop = 'eos'
        else:
            raise ValueError('explicit native termination required')
        outcome = dict(positive='correct', negative='wrong', neutral='unresolved')[kind]
        result.append(Trajectory(nonce, tuple(native['output']), outcome, stop))
    receipt = dict(version='affine-mimo-native-training-projection-v1',
        document_sha256=expected_document_sha256, checkpoint=binding.checkpoint,
        epoch=binding.epoch, contract_sha256=binding.contract_sha256,
        trusted_prompt=list(trusted_prompt), native_results=native_results,
        task=dict(env_id=batch['env_id'], environment_version=batch['environment_version'],
                  index=batch['index'], task_hash=batch['task_hash']))
    task_id = digest({k: receipt['task'][k] for k in ('env_id', 'environment_version', 'task_hash')})
    group = TaskGroup(task_id, tuple(trusted_prompt), tuple(result), digest(receipt))
    group.validate(group_size, vocab_size=vocab_size, eos_token_ids=set(eos_token_ids))
    return group, receipt
