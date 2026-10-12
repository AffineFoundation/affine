"""Separate TRAIN diagnostic contract; never a held-out or miner-score policy."""
import base64
import hashlib
import json
import math
from pathlib import Path
import statistics

VERSION = 'fresh-training-evaluation-v1'
COHORT_VERSION = 'fixed-approved-training128-v1'
PLAN_VERSION = 'fresh-approved-training128-plan-v1'
HARNESS = dict(version='text-tools-long-kv-v3', policy='autoregressive',
               max_output_tokens=2048, temperature=.7, top_p=1.)
SIZE = 128


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def signed(envelope, authority):
    from nacl.signing import VerifyKey
    if set(envelope) != {'payload', 'signer', 'signature'} or envelope['signer'] != authority:
        raise ValueError('exact diagnostic authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),
                                             base64.b64decode(envelope['signature'], validate=True))
    return envelope['payload']


def need(condition, reason):
    if not condition: raise ValueError(reason)


def hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def seed_for(index, question_hash):
    body = {'domain': 'affine-periodic-training128-paired-draw-20261012-v1',
            'index': index, 'normalized_question_sha256': question_hash}
    return int.from_bytes(hashlib.sha256(canonical(body)).digest()[:4], 'big') & 0x7fffffff


def validate_cohort(cohort, approved_manifest=None):
    need(cohort['version'] == COHORT_VERSION and cohort['task_count'] == SIZE, 'fixed TRAIN128 contract')
    need(cohort['selection_uses_outcomes'] is False and cohort['confirmation'] is False, 'TRAIN diagnostic only')
    tasks = cohort['tasks']
    need(len(tasks) == SIZE, 'complete fixed TRAIN128')
    need(len({t['index'] for t in tasks}) == len({t['normalized_question_sha256'] for t in tasks}) == SIZE,
         'distinct task and normalized-question identities')
    need(cohort['tasks_sha256'] == digest(tasks), 'immutable TRAIN identities and seeds')
    for t in tasks:
        need(t['env_id'] == 'affine_math' and type(t['index']) is int and t['index'] >= 0,
             'original native MATH identity')
        need(hex64(t['normalized_question_sha256']) and hex64(t['registry_row_sha256']), 'task metadata hashes')
        need(t['seed'] == seed_for(t['index'], t['normalized_question_sha256']), 'fixed paired draw')
    if approved_manifest is not None:
        definitions = [d for d in approved_manifest['environments'] if d['env_id'] == 'affine_math']
        need(len(definitions) == 1, 'one approved MATH registry')
        indices = {t['index'] for t in tasks}
        need(indices <= set(definitions[0]['indices']), 'cohort subset of approved mining membership')
        heldout = approved_manifest.get('heldout_indices', [])
        if isinstance(heldout, dict): heldout = heldout.get('affine_math', [])
        need(not indices & set(heldout), 'TRAIN/heldout disjointness')
        need(digest(approved_manifest) == cohort['approved_manifest_payload_sha256'], 'exact approved mining manifest')
    return tasks


def scientific_profile(plan):
    m = plan['manifest']['payload']
    return dict(max_output_tokens=2048, max_context_tokens=8192, temperature=.7, top_p=1.,
                harness=HARNESS, runtime_versions=plan['runtime_versions'],
                model_runtime_revision=m['model_runtime_revision'], backend_profile=m['backend_profile'],
                numerical_policy=m['numerical_policy'],
                scientific_files_sha256=digest(plan['scientific_files']),
                source_files_sha256=digest(plan['source_files']),
                environment_sha256=digest(m['environments'][0]['spec']))


def validate_plan(plan, authority):
    need(plan['version'] == PLAN_VERSION and plan['task_count'] == SIZE, 'separate TRAIN evaluation plan')
    need(plan['production_mutation'] is False and plan['optimizer_allocation'] is False,
         'nonblocking inference only')
    cohort = signed(plan['training_cohort'], authority)
    approved = signed(plan['approved_mining_manifest'], authority)
    tasks = validate_cohort(cohort, approved)
    signed(plan['manifest'], authority)
    need(plan['cohort_sha256'] == digest(cohort), 'exact TRAIN cohort identity')
    need(len(plan['suites']) == 4, 'four fixed32 batches')
    flat = []
    for suite in plan['suites']:
        need(suite['env_id'] == 'affine_math' and suite['harness'] == HARNESS, 'unmixed EV3 cap2048 profile')
        need(len(suite['indices']) == len(suite['seeds']) == 32, 'complete fixed32 suite')
        flat.extend(zip(suite['indices'], suite['seeds']))
    need(flat == [(t['index'], t['seed']) for t in tasks], 'exact TRAIN order and seed mapping')
    need(plan['profile'] == scientific_profile(plan) and plan['profile_sha256'] == digest(plan['profile']),
         'exact shared numerical profile')
    need(plan['seeds_sha256'] == digest([(t['index'], t['seed']) for t in tasks]), 'paired seed identity')
    need(plan['phase'] in ('baseline', 'latest') and type(plan['optimizer_step']) is int,
         'declared checkpoint role')
    need(plan['phase'] != 'baseline' or plan['optimizer_step'] == 0, 'original baseline step zero')
    need(plan['reference_kind'] == 'original-upstream-comparison' and
         hex64(plan['actual_run_genesis_checkpoint']), 'reference coordinate separate from actual run genesis')
    need(plan['checkpoint']['id'] == digest(plan['checkpoint']['files']), 'complete checkpoint file map')
    return cohort


def choose_latest(latest_step, previous_step, *, first_step=4, step_gap=4):
    """Latest durable checkpoint after at least four new updates; coalesce backlog."""
    need(type(latest_step) is int and latest_step >= 0 and step_gap == 4, 'four-update TRAIN cadence')
    return latest_step >= first_step and (previous_step is None or latest_step >= previous_step + step_gap)


def failed_rollout_classification(native, error):
    """Native neutral means unresolved; arbitrary grader/runtime failures remain errors."""
    if (native and isinstance(error, ValueError) and str(error) == 'binary native math outcome'
            and native[-1].get('classification') in ('neutral', 'unresolved')
            and native[-1].get('done') is True and native[-1].get('reward') == 0):
        return 'unresolved'
    return 'error'


def summarize(plan, rows, *, exposure=None):
    tasks = plan['training_cohort']['payload']['tasks']
    expected = {(t['env_id'], t['index'], t['seed']) for t in tasks}
    by = {(r['env_id'], r['index'], r['seed']): r for r in rows}
    need(len(rows) == len(by) == SIZE and set(by) == expected, 'all original128 records including errors')
    for r in rows:
        need(r['checkpoint'] == plan['checkpoint']['id'] and r['cohort_sha256'] == plan['cohort_sha256'],
             'exact original checkpoint/cohort on every record')
        need(r.get('profile_sha256') == plan['profile_sha256'] and
             r.get('generation_origin') == 'fresh-policy-generation', 'actual fresh scientific profile')
        need(r['classification'] in ('positive', 'negative', 'unresolved', 'error'), 'explicit terminal classification')
        need(r['reward'] == (1 if r['classification'] == 'positive' else 0), 'native binary reward')
        need(r.get('verified') is False and r.get('proof_verification_performed') is False,
             'operator diagnostic never miner verification evidence')
        if r['classification'] != 'error':
            need(r.get('native_graded') is True and len(r['raw_turns']) == 1,
                 'one native graded MATH generation')
        for n, turn in enumerate(r['raw_turns']):
            need(turn['seed'] == r['seed'] + n and turn['prompt_tokens'] and
                 all(type(i) is int and i >= 0 for i in turn['prompt_tokens'] + turn['output_tokens']) and
                 len(turn['output_tokens']) <= 2048, 'actual paired draw tokens and seed')
        need(type(r['cap_hit']) is bool and type(r['terminal_eos']) is bool, 'recorded stopping semantics')
        need(r['cap_hit'] == any(len(t['output_tokens']) >= 2048 for t in r['raw_turns']), 'actual cap accounting')
    completed = [r for r in rows if r['classification'] != 'error']
    lengths = [sum(len(t['output_tokens']) for t in r['raw_turns']) for r in completed]
    counts = {k: sum(r['classification'] == k for r in rows) for k in ('positive', 'negative', 'unresolved', 'error')}
    if exposure is None:
        exposure = plan.get('exposure')
    if exposure is None:
        exposure = dict(scope='current-training-run-through-checkpoint', trained_tasks=None,
                        untrained_tasks=None, unknown_tasks=SIZE, task_exposures=None)
    known = exposure['trained_tasks']
    if known is not None:
        need(type(known) is int and 0 <= known <= SIZE and
             known + exposure['untrained_tasks'] + exposure['unknown_tasks'] == SIZE, 'exposure accounting')
    else:
        need(exposure['untrained_tasks'] is None and exposure['unknown_tasks'] == SIZE, 'unknown exposure stays unknown')
    def group(values):
        return dict(count=len(values), solved=sum(r['classification'] == 'positive' for r in values),
                    wrong=sum(r['classification'] == 'negative' for r in values),
                    unresolved=sum(r['classification'] == 'unresolved' for r in values),
                    errors=sum(r['classification'] == 'error' for r in values))
    task_counts = plan.get('exposure_task_counts')
    groups = dict(trained=[], untrained=[], unknown=[])
    for r in rows:
        count = None if task_counts is None else task_counts.get(str(r['index']))
        need(count is None or type(count) is int and count >= 0, 'per-task exposure count')
        groups['unknown' if count is None else 'trained' if count else 'untrained'].append(r)
    grouped = {name: group(values) for name, values in groups.items()}
    need(grouped['unknown']['count'] == exposure['unknown_tasks'], 'aggregate/task exposure consistency')
    if known is not None:
        need(grouped['trained']['count'] == known and grouped['untrained']['count'] == exposure['untrained_tasks'],
             'aggregate/task trained partition')
    initial = plan.get('fixed_initial_trained_mask')
    fixed = None
    if initial is not None:
        need(set(initial['indices']) <= {t['index'] for t in tasks} and
             len(initial['indices']) == len(set(initial['indices'])), 'fixed initial exposure subset')
        fixed = dict(group([r for r in rows if r['index'] in set(initial['indices'])]),
                     mask_sha256=digest(initial), checkpoint=initial['checkpoint'],
                     optimizer_step=initial['optimizer_step'])
    return dict(version=VERSION, series='training-fresh', label='Training set — fresh generations',
                run_id=plan['run_id'], training_run_id=plan['training_run_id'], env_id='affine_math',
                cohort_sha256=plan['cohort_sha256'], profile_sha256=plan['profile_sha256'],
                profile=plan['profile'], seeds_sha256=plan['seeds_sha256'],
                checkpoint=plan['checkpoint']['id'], optimizer_step=plan['optimizer_step'],
                epoch=plan['epoch'], phase=plan['phase'], generation_origin='fresh-policy-generation',
                reference_kind=plan['reference_kind'], actual_run_genesis_checkpoint=plan['actual_run_genesis_checkpoint'],
                requested=SIZE, completed=len(completed), solved=counts['positive'], wrong=counts['negative'],
                unresolved=counts['unresolved'], errors=counts['error'],
                cap_hits=sum(bool(r['cap_hit']) for r in completed),
                terminal_eos=sum(bool(r['terminal_eos']) for r in completed),
                mean_output_tokens=statistics.mean(lengths) if lengths else None,
                median_output_tokens=statistics.median(lengths) if lengths else None,
                exposure=exposure, exposure_groups=grouped, fixed_initial_trained=fixed,
                exposure_subsets_descriptive=True, confirmation=False, production_barrier=False)


def paired_comparability(baseline_plan, baseline_rows, plan, rows):
    for name in ('cohort_sha256', 'profile_sha256', 'seeds_sha256', 'training_run_id'):
        need(baseline_plan[name] == plan[name], 'baseline/current scientific identity mismatch')
    summarize(baseline_plan, baseline_rows); summarize(plan, rows)
    b = {(r['env_id'], r['index'], r['seed']): r for r in baseline_rows}
    failures = []
    for r in rows:
        key = r['env_id'], r['index'], r['seed']; before = b[key]
        if before['classification'] == 'error' or r['classification'] == 'error':
            failures.append(dict(index=r['index'], reason='infrastructure_error')); continue
        if (before.get('task_hash') != r.get('task_hash') or not before.get('task_hash') or
            [t['prompt_tokens'] for t in before['raw_turns']] != [t['prompt_tokens'] for t in r['raw_turns']]):
            failures.append(dict(index=r['index'], reason='task_or_actual_prompt_tokens_mismatch'))
    return dict(comparable=not failures, denominator=SIZE, failures=failures)
