"""Signed, eval-only MiMo sampling; no miner quotas, proofs or upload rights.

The complete held-out suite lives in the authenticated opening. Evaluation uses
the same checkpoint, template, sampler, output horizon and final-channel grader
as mining, but its own reproducible random draws and independent task authority.
No training/mining allowlist is modified to make a held-out task accessible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from .storage import canonical

VERSION = 'mimo-heldout-evaluation-v1'
DRAW_POLICY = 'mimo-evaluation-common-random-draws-v1'
MAX_OUTPUT_TOKENS = 1_250_000  # Bounds a token-only queue report below 12 MiB.


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate_sets(manifest):
    from .protocol import entries
    from .harness import normalize
    from .mimo_runtime import validate_runtime_spec
    from .mimo_answer_projection import policy
    from .complete_group_distribution import MIMO_HARNESS
    rows = manifest.get('evaluation_sets', [])
    need(type(rows) is list and len(rows) <= 16, 'bounded signed evaluation suites')
    if not rows:
        return {}
    definitions = {row['env_id']: row for row in entries(manifest)}
    spec = validate_runtime_spec(manifest['mimo_runtime'])
    need(policy(manifest) is not None, 'evaluation requires explicit final-channel grading')
    result = {}
    for row in rows:
        need(type(row) is dict and set(row) == {'name','env_id','indices','seed','repeats'},
             'exact signed evaluation set fields')
        name = row['name']
        need(type(name) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name)
             and name not in result, 'unique safe evaluation set name')
        need(row['env_id'] in definitions, 'evaluation taskset must be in the signed registry')
        definition = definitions[row['env_id']]
        h = normalize(definition['harness'])
        need(definition['spec']['max_turns'] == 1 and
             h['max_output_tokens'] <= definition['spec']['max_output_tokens'],
             'evaluation inherits the signed single-turn native horizon')
        need(h['version'] == MIMO_HARNESS and h['max_context_tokens'] == spec.max_context_tokens
             and (h['temperature'],h['top_p'],h['top_k'],h['enable_thinking']) ==
             (spec.temperature,spec.top_p,spec.top_k,spec.enable_thinking),
             'evaluation inherits the exact non-indexed MiMo harness')
        indices = row['indices']
        need(type(indices) is list and 1 <= len(indices) <= 750 and
             all(type(i) is int and 0 <= i < definition['spec']['num_samples'] for i in indices)
             and len(set(indices)) == len(indices), 'unique bounded held-out indices')
        # Full registries remain authoritative even if a worker gets a mining
        # subset. Indices in unrelated tasksets do not share an identity.
        mining = set()
        for other in definitions.values():
            if other['spec'] == definition['spec'] or other['env_id'] == row['env_id']:
                mining.update(other['indices'])
                mining.update(manifest.get('sample_harness_registry', {}).get(other['env_id'], {}).get('indices', []))
        need(not set(indices) & mining, 'held-out tasks overlap the mining population')
        need(type(row['seed']) is int and 0 <= row['seed'] < 2**63 and
             type(row['repeats']) is int and 1 <= row['repeats'] <= 16,
             'explicit bounded evaluation seed and repeats')
        need(len(indices)*row['repeats']*h['max_output_tokens'] <= MAX_OUTPUT_TOKENS,
             'evaluation report budget exceeded; publish smaller explicit suites')
        frozen = dict(version=VERSION, suite=row, environment=definition['spec'],
                      harness=h, runtime=manifest['mimo_runtime'],
                      answer_projection=manifest['mimo_answer_projection'], draw_policy=DRAW_POLICY)
        result[name] = dict(suite=row, definition=definition, harness=h, dataset_id=digest(frozen))
    return result


def validate_job(job, manifest):
    value = job.get('evaluation')
    need(type(value) is dict and set(value) == {'version','run_id','set_name'} and
         value['version'] == VERSION and value['run_id'] == manifest.get('run_id'),
         'explicit evaluation job and original run binding')
    suites = validate_sets(manifest)
    need(value['set_name'] in suites, 'evaluation must select an original signed held-out suite')
    need(not any(key in job for key in ('capability','miner_identity_file','mining_subset','submissions','training')),
         'evaluation has no submission or training capabilities')
    return suites[value['set_name']]


def draw(dataset_id, seed, index, repeat, position):
    """Identical draws across checkpoints, independent of mutable epoch names."""
    raw = hashlib.sha256(canonical([DRAW_POLICY,dataset_id,seed,index,repeat,position])).digest()
    return (int.from_bytes(raw[:8], 'big') >> 11) / 2**53


def _summary(rows):
    counts = {kind: sum(row['classification'] == kind for row in rows)
              for kind in ('positive','negative','neutral','indeterminate')}
    return dict(requested_count=len(rows), count=len(rows)-counts['indeterminate'],
        successes=counts['positive'], completed_wrong=counts['negative'], unresolved=counts['neutral'],
        error_count=counts['indeterminate'],
        mean_reward=counts['positive']/len(rows) if rows and not counts['indeterminate'] else None,
        status='error' if counts['indeterminate'] else 'complete')


def evaluate(manifest, selection, runtime, *, guard=lambda: None):
    """Evaluate every requested attempt; no outcome-based selection or retries."""
    import torch
    from .environments import EnvironmentSpec, create_session
    from .complete_group_distribution import distribution, pick
    from .complete_group_sampling import validate_geometry, token_geometry
    from .mimo_answer_projection import grade
    suite, definition, h = selection['suite'], selection['definition'], selection['harness']
    need(runtime.checkpoint_id == manifest['checkpoint']['id'] and
         runtime.spec.to_dict() == manifest['mimo_runtime'] and
         tuple(runtime.eos_token_ids) == tuple(manifest['mimo_runtime']['eos_token_ids']),
         'actual evaluation model/runtime/geometry differ from the signed opening')
    # The checkpoint's LM head can contain padding beyond the tokenizer's
    # support. Use the same authenticated support as mining, not padded logits.
    geometry = validate_geometry(runtime, {'token_geometry':
        token_geometry(manifest['group_transport_policy'])})
    support = geometry['vocab_size']
    runtime.guard()
    spec = EnvironmentSpec.from_dict(definition['spec'])
    rows = []
    for index in suite['indices']:
        for repeat in range(suite['repeats']):
            guard()
            row = dict(index=index,repeat=repeat,task_hash=None,prompt_sha256=None,
                       output=[],output_length=0,stop_reason=None,classification='indeterminate',
                       reward=None,error=None)
            session, cache, stage = None, None, 'native-reset'
            started = time.monotonic()
            try:
                session = create_session(spec)
                initial = session.reset(index, int(spec.config.get('seed', 0)))
                row['task_hash'] = initial['task_hash']
                stage = 'canonical-prompt'
                need(not initial.get('tools'), 'initial MiMo math evaluation is text-only')
                prompt = runtime.prepare_prompt(initial['messages'], initial.get('tools', []))
                need(prompt and len(prompt)+h['max_output_tokens'] <= runtime.max_context_tokens,
                     'held-out prompt and full output horizon exceed signed context')
                row['prompt_sha256'] = digest(prompt)
                stage = 'generation'
                with torch.inference_mode():
                    for position in range(h['max_output_tokens']):
                        guard()
                        logits, cache = runtime.next_token_logits(prompt if not row['output'] else [row['output'][-1]],
                                                                  past_key_values=cache,use_cache=True)
                        need(logits.ndim == 1 and logits.shape[0] == runtime.vocab_size,
                             'evaluation logits disagree with pinned model head')
                        token = pick(distribution(logits[:support],h['temperature'],h['top_p'],h['top_k']),
                            draw(selection['dataset_id'],suite['seed'],index,repeat,position))
                        row['output'].append(token)
                        if token in runtime.eos_token_ids:
                            break
                row['stop_reason'] = 'eos' if row['output'][-1] in runtime.eos_token_ids else 'budget'
                stage = 'native-grade'
                guard()
                _, kind, reward = grade(session,manifest,h,runtime.tokenizer,row['output'],row['stop_reason'])
                row.update(classification=kind,reward=reward)
            except Exception as error:
                # An expired/lost host lease aborts the job, rather than turning
                # the rest of the suite into invented task failures.
                guard()
                row.update(classification='indeterminate',reward=None,
                           error=dict(stage=stage,type=type(error).__name__))
            finally:
                cache = None
                if session is not None:
                    session.close()
            row['output_length'] = len(row['output'])
            row['seconds'] = time.monotonic()-started
            rows.append(row)
    guard()
    runtime.guard()
    return dict(version=VERSION,run_id=manifest['run_id'],epoch=manifest['epoch'],
        checkpoint=manifest['checkpoint']['id'],manifest_sha256=digest(manifest),
        runtime_sha256=digest(manifest['mimo_runtime']),dataset_id=selection['dataset_id'],
        suite=suite,draw_policy=DRAW_POLICY,harness=h,answer_projection=manifest['mimo_answer_projection'],
        rows=rows,**_summary(rows),payable=False,weight_submission=False,sampling_audit_performed=False)


def validate_result(result, job, manifest):
    selection = validate_job(job,manifest)
    expected = dict(version=VERSION,run_id=manifest['run_id'],epoch=manifest['epoch'],
        checkpoint=manifest['checkpoint']['id'],manifest_sha256=digest(manifest),
        runtime_sha256=digest(manifest['mimo_runtime']),dataset_id=selection['dataset_id'],
        suite=selection['suite'],draw_policy=DRAW_POLICY,harness=selection['harness'],
        answer_projection=manifest['mimo_answer_projection'],payable=False,
        weight_submission=False,sampling_audit_performed=False)
    need(all(result.get(k) == v for k,v in expected.items()), 'evaluation report source/taskset/runtime/checkpoint binding')
    rows = result.get('rows'); suite = selection['suite']; h = selection['harness']
    wanted = [(i,r) for i in suite['indices'] for r in range(suite['repeats'])]
    need(type(rows) is list and len(rows)==len(wanted), 'every original held-out attempt needs a result')
    stops = set(manifest['mimo_runtime']['eos_token_ids']); support=manifest['group_transport_policy']['vocab_size']
    for row,(index,repeat) in zip(rows,wanted,strict=True):
        need(type(row) is dict and type(row.get('index')) is int and type(row.get('repeat')) is int and
             (row['index'],row['repeat']) == (index,repeat), 'ordered held-out task coverage')
        tokens = row.get('output')
        need(type(tokens) is list and len(tokens)<=h['max_output_tokens'] and
             all(type(t) is int and 0<=t<support for t in tokens) and row.get('output_length')==len(tokens),
             'bounded original evaluation output tokens')
        kind=row.get('classification')
        need(kind in ('positive','negative','neutral','indeterminate'), 'explicit evaluation outcome')
        if kind == 'indeterminate':
            need(row.get('reward') is None and type(row.get('error')) is dict, 'infrastructure failure is not reward zero')
        else:
            need(tokens and row.get('reward') == float(kind=='positive') and row.get('error') is None,
                 'native evaluation reward semantics')
            if row.get('stop_reason')=='budget':
                need(len(tokens)==h['max_output_tokens'] and not stops.intersection(tokens) and kind=='neutral',
                     'budget caps remain unresolved')
            else:
                need(row.get('stop_reason')=='eos' and tokens[-1] in stops and not stops.intersection(tokens[:-1]),
                     'completed evaluation stops at first approved EOS')
    need(all(result.get(k)==v for k,v in _summary(rows).items()), 'evaluation summary must reflect all requested tasks')


def main():
    """Run the same authenticated worker path, including source/checkpoint pins."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job');parser.add_argument('--authority',required=True)
    parser.add_argument('--workspace',required=True);parser.add_argument('--checkpoint-cache')
    args=parser.parse_args()
    from .backend_jobs import execute, signed
    envelope=json.loads(Path(args.job).read_bytes())
    need(signed(envelope,args.authority).get('role')=='evaluate','evaluation-only CLI')
    result=execute(envelope,args.authority,args.workspace,cache=args.checkpoint_cache)
    print(json.dumps(dict(status=result['evaluation']['status'],count=result['evaluation']['count'],
                          successes=result['evaluation']['successes'])))


if __name__=='__main__':
    main()
