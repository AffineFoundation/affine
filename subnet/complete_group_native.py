"""CPU native task authority and grading, independent of model-proof audits.

The caller supplies an authenticated manifest and a checkpoint-authenticated
tokenizer. Miner JSON cannot select a source, task snapshot, prompt or grader.
"""
from . import harness as policy
from .protocol import entry, harness_for
from .environments import EnvironmentSpec, create_session
from .complete_group_admission import TaskAuthority, structural_batch
from .complete_group_sampling import digest, need
from .complete_group_distribution import eos_ids, geometry
from .complete_group_runtime import outcome


def _resolved(manifest, env_id, index, tokenizer):
    definition = entry(manifest, env_id)
    h = harness_for(definition, index)
    spec = EnvironmentSpec.from_dict(definition['spec'])
    need(spec.max_turns == 1 and h['max_output_tokens'] <= spec.max_output_tokens, 'single-turn native task')
    g = geometry(manifest['group_transport_policy']); stops = eos_ids(g)
    need(len(tokenizer) == g['vocab_size'] and tokenizer.eos_token_id in stops, 'trusted tokenizer geometry')
    if h['version'] == 'text-tools-mimo-complete-group-v1':
        from .mimo_runtime import validate_runtime_spec
        import hashlib
        runtime = validate_runtime_spec(manifest['mimo_runtime'])
        need(type(tokenizer.chat_template) is str
             and hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == runtime.chat_template_sha256
             and runtime.enable_thinking == h['enable_thinking'], 'trusted tokenizer template and thinking mode')
    return definition, spec, h, g


def resolve_task(manifest, env_id, index, tokenizer):
    definition, spec, h, g = _resolved(manifest, env_id, index, tokenizer)
    session = create_session(spec)
    try:
        initial = session.reset(index, int(spec.config.get('seed', 0)))
        prompt = policy.render(tokenizer, initial['messages'], initial.get('tools', []), h)
    finally:
        session.close()
    need(prompt and all(type(t) is int and 0 <= t < g['vocab_size'] for t in prompt)
         and len(prompt) + h['max_output_tokens'] <= g['max_context_tokens'], 'canonical full prompt and horizon')
    stops = eos_ids(g)
    return TaskAuthority(manifest['epoch'], manifest['checkpoint']['id'], env_id, spec.version,
        index, initial['task_hash'], tuple(prompt), tokenizer.eos_token_id, g['vocab_size'],
        stops if 'eos_token_ids' in g else ())


def grade_batch(batch, manifest, tokenizer, *, miner):
    """Return trusted outcomes plus native receipts without sampling verification.

    A grader exception defers the group to the host; it never produces reward0.
    Claimed outcome disagreement rejects it; it cannot be relabelled silently.
    """
    from .audit_policy import InvalidSample
    def submitted(condition, message):
        if not condition:
            raise InvalidSample(message)
    task = resolve_task(manifest, batch['env_id'], batch['index'], tokenizer)
    structural_batch(batch, manifest, miner, task)
    definition, spec, h, g = _resolved(manifest, batch['env_id'], batch['index'], tokenizer)
    results = []
    for row in batch['rollouts']:
        turn = row['turns'][0]; output = turn['output']
        text = tokenizer.decode(output, skip_special_tokens=True)
        submitted(text == turn['text'], 'native text must match exact output tokens')
        session = create_session(spec)
        try:
            initial = session.reset(task.index, int(spec.config.get('seed', 0)))
            prompt = policy.render(tokenizer, initial['messages'], initial.get('tools', []), h)
            need(initial['task_hash'] == task.task_hash and tuple(prompt) == task.prompt,
                 'stable native task and prompt across original attempts')
            from .mimo_answer_projection import grade
            native, kind, reward = grade(session, manifest, h, tokenizer, output, turn['stop_reason'])
        finally:
            session.close()
        submitted(row['classification'] == turn['classification'] == kind and row['reward'] == turn['reward'] == reward,
             'native grading differs from submitted outcome')
        submitted(turn['observations'] == native['observations'], 'native observations differ')
        if turn['stop_reason'] == 'budget':
            submitted(row['native_classification'] == turn['native_classification'] == native['classification'],
                 'original capped native outcome differs')
        results.append(dict(nonce=row['seed'], classification=kind, reward=reward,
            stop_reason=turn['stop_reason'], output_sha256=digest(output), native_classification=native['classification'],
            prompt=list(task.prompt), output=list(output), task_hash=task.task_hash,
            done=True, observations=native['observations']))
    return dict(version='complete-group-native-grading-v1', batch_sha256=digest(batch),
        manifest_sha256=digest(manifest), miner=miner, task_hash=task.task_hash,
        prompt=list(task.prompt), results=results, sampling_verified=False)
