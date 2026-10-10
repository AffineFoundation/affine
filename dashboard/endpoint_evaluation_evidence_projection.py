"""Allowlist only the existing g/g-plus16 endpoint fixed128 study archives.

No dispatch, model, tokenizer, network, or discovery of other private experiments.
Finalization is authenticated by the caller before supplying checkpoint identities.
"""
import base64
import hashlib
import json
import math
from pathlib import Path
import re

from nacl.signing import VerifyKey

AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
RELATIVE = Path('state/root-audits/20261010-live-retained-nll1-candidate/production-nll16-representative-v1/evaluation-operator-v1/state')
ARMS = ('g', 'g-plus16')
COHORTS = ('original128', 'exposed_test128')
PROGRAM = '2a90db3e1457420fb23ba0e92e5ec7cf8abe0751075b4ca14266c1b33430a144'
SAMPLER = '181ba77fc35ae307e31ac6fb4f2dd5f200170a1429a4ca659b438cd8d53bc469'
OWNED = 'd17078c1e9fd3e51d5eda535194a809ddd3ad22b4963fe2c74edfb650a41f37c'
HARNESS = dict(version='text-tools-long-kv-v3', policy='autoregressive',
               max_output_tokens=2048, temperature=.7, top_p=1.)
HASH = re.compile(r'[0-9a-f]{64}\Z')
MAX_FILE = 32*1024**2
MAX_TASK = 2*1024**2


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(x):
    return sha(canonical(x))


def authenticate(document):
    if (type(document) is not dict or set(document) != {'payload','signature','signer'}
            or document['signer'] != AUTHORITY):
        raise ValueError('endpoint_archive_authority')
    VerifyKey(bytes.fromhex(AUTHORITY)).verify(canonical(document['payload']),
        base64.b64decode(document['signature'], validate=True))
    return document['payload']


def read(path, maximum=MAX_FILE):
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError('endpoint_regular_canonical_file')
    with path.open('rb') as f:
        raw = f.read(maximum+1)
    if len(raw) > maximum:
        raise ValueError('endpoint_artifact_bound')
    return raw


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def tokens(value, limit):
    if (type(value) is not list or not 1 <= len(value) <= limit
            or any(type(t) is not int or not 0 <= t < 2**31 for t in value)):
        raise ValueError('endpoint_token_ids')
    return value


def native_result(value):
    label, reward = value.get('classification'), value.get('reward')
    if (label not in ('positive','negative','neutral','unresolved') or type(value.get('done')) is not bool
            or not finite(reward) or reward != (1 if label == 'positive' else 0)):
        raise ValueError('endpoint_native_result')
    return dict(classification=label, reward=reward, done=value['done'])


def project_task(task, *, index, seed, checkpoint, cohort_sha256):
    if (task.get('index') != index or task.get('seed') != seed or task.get('checkpoint') != checkpoint
            or task.get('cohort_sha256') != cohort_sha256 or task.get('env_id') != 'affine_math'
            or task.get('verified') is not False or task.get('proof_verification_performed') is not False
            or type(task.get('cap_hit')) is not bool or type(task.get('budget_exhausted')) is not bool
            or not finite(task.get('elapsed_seconds')) or task['elapsed_seconds'] < 0):
        raise ValueError('endpoint_task_identity')
    outer = task.get('classification')
    if outer not in ('positive','negative','unresolved','error') or not finite(task.get('reward')):
        raise ValueError('endpoint_outer_result')
    if task['reward'] != (1 if outer == 'positive' else 0):
        raise ValueError('endpoint_outer_reward')
    raw, native = task.get('raw_turns'), task.get('native_results')
    if type(raw) is not list or len(raw) > 1 or type(native) is not list or len(native) > 1:
        raise ValueError('endpoint_one_turn_math')
    native = [native_result(v) for v in native]
    if task.get('native_graded') is not bool(native):
        raise ValueError('endpoint_native_presence')
    if outer in ('positive','negative') and (not native or native[-1]['classification'] != outer or not native[-1]['done']):
        raise ValueError('endpoint_native_outer_agreement')
    if outer == 'unresolved' and (not native or native[-1]['classification'] != 'unresolved'):
        raise ValueError('endpoint_unresolved_binding')
    turns=[]
    for turn in raw:
        output=tokens(turn.get('output_tokens'),2048); prompt=tokens(turn.get('prompt_tokens'),8192)
        if (turn.get('seed') != seed or type(turn.get('text')) is not str
                or len(turn['text'].encode()) > MAX_TASK or len(prompt)+len(output)>8192):
            raise ValueError('endpoint_raw_turn')
        reached=len(output)==2048
        if task['cap_hit'] != reached or task['budget_exhausted'] and not reached:
            raise ValueError('endpoint_recorded_cap_consistency')
        # The exact pinned wrapper records budget_exhausted only when the final
        # generated token is not tokenizer EOS. The pinned sampler has no other
        # successful early stop. No stop claim is made for absent raw output.
        stop='budget' if task['budget_exhausted'] else 'eos'
        turns.append(dict(prompt_token_ids=prompt,output_token_ids=output,output_text=turn['text'],
            prompt_length=len(prompt),output_length=len(output),seed=seed,max_output_tokens=2048,
            reached_token_budget=reached,stop_reason=stop,
            stop_reason_basis='authenticated_pinned_program_budget_flag_and_sampler'))
    if not raw and (native or task['cap_hit'] or task['budget_exhausted']):
        raise ValueError('endpoint_missing_generation_consistency')
    native_label=native[-1]['classification'] if native else None
    incomplete=native_label in ('neutral','unresolved')
    infrastructure=(outer=='error' and not incomplete)
    result=dict(index=index,seed=seed,outer_classification=outer,outer_reward=task['reward'],
        raw_native_results=native,native_classification=native_label,
        unresolved_or_incomplete=incomplete,infrastructure_error=infrastructure,
        native_graded=bool(native),proof_verification_performed=False,
        raw_turns=turns,output_length=sum(t['output_length'] for t in turns) if turns else None,
        cap_hit=task['cap_hit'],budget_exhausted=task['budget_exhausted'],elapsed_seconds=task['elapsed_seconds'])
    if 'task_hash' in task:
        if type(task['task_hash']) is not str or not HASH.fullmatch(task['task_hash']):
            raise ValueError('endpoint_task_hash')
        result['task_hash']=task['task_hash']
    return result


def project_phase(directory, arm, cohort, finalized_checkpoints):
    if arm not in ARMS or cohort not in COHORTS:
        raise ValueError('only_existing_fixed_study_endpoint_scope')
    archive_raw=read(directory/'archive.json'); archive=authenticate(json.loads(archive_raw))
    if (archive.get('version') != 'private-learning-evidence-archive-v1'
            or archive.get('label') != 'evaluate-'+cohort or archive.get('full_readback_verified') is not True
            or not finite(archive.get('at'))):
        raise ValueError('endpoint_archive_receipt')
    def captured(name, limit=MAX_FILE):
        raw=read(directory/'captured'/name,limit); pin=archive['objects'][name]
        if (pin.get('full_readback_verified') is not True or type(pin.get('bytes')) is not int
                or pin['bytes'] != len(raw) or pin.get('sha256') != sha(raw)):
            raise ValueError('endpoint_original_full_readback')
        return raw
    plan_raw=captured('plan.json'); envelope=json.loads(plan_raw); plan=authenticate(envelope)
    checkpoint=plan['checkpoint']['id']
    # Open/future checkpoints are excluded before reading any task outputs.
    if checkpoint not in finalized_checkpoints:
        return None
    if (plan.get('version') != 'fresh-run-disjoint-heldout128-v1' or plan.get('endpoint') != arm
            or plan.get('exposed_cohort') != cohort or plan.get('task_count') != 128
            or plan.get('production_mutation') is not False or plan.get('optimizer_allocation') is not False
            or plan.get('program_sha256') != PROGRAM or not HASH.fullmatch(checkpoint)
            or digest(plan['checkpoint']['files']) != checkpoint
            or not finite(plan.get('created_at')) or not finite(plan.get('expires_at'))
            or not 0 < plan['expires_at']-plan['created_at'] <= 7200):
        raise ValueError('endpoint_exact_plan')
    for name,h in (('cached_sampling',SAMPLER),('owned_cached_evaluation',OWNED)):
        key='subnet/'+name+'.py'
        if plan['source_files'].get(key) != h or plan['scientific_files'].get(key) != h:
            raise ValueError('endpoint_original_sampler_bytes')
    manifest=authenticate(plan['manifest']); definitions=manifest['environments']
    definition=next(d for d in definitions if d['env_id']=='affine_math')
    if definition['spec']['max_turns'] != 1 or len(plan['suites']) != 4:
        raise ValueError('endpoint_fixed_single_turn_cohort')
    ordered=[]; cohort_hashes=[]
    for suite in plan['suites']:
        indices,seeds=suite['indices'],suite['seeds']
        if (suite['env_id'] != 'affine_math' or suite['harness'] != HARNESS
                or len(indices) != 32 or len(seeds) != 32
                or any(type(i) is not int or i<0 for i in indices+seeds)):
            raise ValueError('endpoint_exact32_suites')
        frozen=dict(version='owned-cached-native-evaluation-v1',env_id=suite['env_id'],
            environment=definition['spec'],harness=HARNESS,indices=indices,seeds=seeds,
            model_runtime_revision=manifest['model_runtime_revision'],backend_profile=manifest['backend_profile'],
            source_files=plan['scientific_files'])
        h=digest(frozen);cohort_hashes.append(h);ordered.extend((i,s,h) for i,s in zip(indices,seeds))
    indices=[i for i,_,_ in ordered]
    if (len(set(indices)) != 128 or cohort_hashes != plan['cohort_sha256']
            or set(indices)&set(definition['indices']) or set(indices)&set(plan['excluded_diagnostic_indices'])
            or not set(indices)<=set(plan['reserved_indices'])):
        raise ValueError('endpoint_original_cohort_binding')
    fingerprint=json.loads(captured('loaded-model-fingerprint.json'))
    plan_sha=digest(envelope)
    if (fingerprint.get('matched') is not True or fingerprint.get('checkpoint') != checkpoint
            or fingerprint.get('plan_sha256') != plan_sha
            or fingerprint['actual_loaded'] != fingerprint['expected_loaded']
            or not plan['created_at'] <= fingerprint['at'] < plan['expires_at']):
        raise ValueError('endpoint_loaded_model_fingerprint')
    summary=json.loads(captured('output/result.json'))
    if (summary.get('version') != plan['version'] or summary.get('plan_sha256') != plan_sha
            or summary.get('checkpoint') != checkpoint or summary.get('tasks') != 128
            or summary.get('cohort_sha256') != cohort_hashes or summary.get('production_mutation') is not False
            or summary.get('optimizer_allocation') is not False
            or not plan['created_at'] <= summary['completed_at'] < plan['expires_at']
            or summary['completed_at'] > archive['at']):
        raise ValueError('endpoint_original_summary')
    expected={f'task-{i}.json' for i in indices}
    archived={n.removeprefix('output/') for n in archive['objects'] if n.startswith('output/task-')}
    if set(summary['raw_artifacts']) != expected or archived != expected:
        raise ValueError('endpoint_all128_artifact_inventory')
    rows=[]
    for i,s,h in ordered:
        name=f'task-{i}.json';raw=captured('output/'+name,MAX_TASK)
        if sha(raw) != summary['raw_artifacts'][name]:
            raise ValueError('endpoint_summary_task_byte_binding')
        row=project_task(json.loads(raw),index=i,seed=s,checkpoint=checkpoint,cohort_sha256=h)
        row['artifact_sha256']=sha(raw);rows.append(row)
    counts={label:sum(r['outer_classification']==label for r in rows) for label in ('positive','negative','unresolved','error')}
    if (summary['correct'] != counts['positive'] or summary['errors'] != counts['error']
            or summary['unresolved'] != counts['unresolved'] or summary['accuracy'] != counts['positive']/128
            or summary['cap_hits'] != sum(r['cap_hit'] for r in rows)
            or summary['budget_exhausted'] != sum(r['budget_exhausted'] for r in rows)):
        raise ValueError('endpoint_original_summary_counts')
    return dict(suite='fixed-study-endpoint-heldout128',arm=arm,cohort=cohort,checkpoint=checkpoint,
        study='retained-Adam-g26-to-g42-fixed16',comparison_status='active_fixed_study_no_improvement_claim',
        historical_independence_proven=False,requested_count=128,tasks=rows,
        native_unresolved_or_incomplete_count=sum(r['unresolved_or_incomplete'] for r in rows),
        infrastructure_error_count=sum(r['infrastructure_error'] for r in rows),
        original_wrapper_counts=counts,harness=HARNESS,completed_at=summary['completed_at'],
        plan_sha256=plan_sha,archive_sha256=sha(archive_raw),summary_sha256=sha(captured('output/result.json')),
        scientific_files_sha256=digest(plan['scientific_files']),cohort_sha256=cohort_hashes,
        loaded_fingerprint_sha256=sha(captured('loaded-model-fingerprint.json')),
        proof_verification_performed=False,model_calls_by_exporter=0,
        limits=['Native neutral remains incomplete even when the original wrapper labels it error.',
                'Endpoint cohort and2048-token cap differ from ordinary fixed32/heldout128 diagnostics.',
                'Checkpoint association does not assert that an endpoint evaluation occurred during that epoch.'])


def collect_endpoint_evidence(source_root, finalized):
    """Return (epoch->phase rows, safe availability issues) for exact known paths."""
    by_epoch={epoch:[] for epoch in finalized};issues=[]
    checkpoints={v[k] for v in finalized.values() for k in ('input_checkpoint','output_checkpoint')}
    root=Path(source_root).resolve()/RELATIVE
    for arm in ARMS:
        for cohort in COHORTS:
            directory=root/arm/'phases'/('evaluate-'+cohort)
            if not (directory/'archive.json').is_file():
                issues.append(dict(arm=arm,cohort=cohort,status='complete_archive_not_available'))
                continue
            try:
                row=project_phase(directory,arm,cohort,checkpoints)
                if row is None:
                    issues.append(dict(arm=arm,cohort=cohort,status='checkpoint_not_finalized'))
                    continue
            except Exception:
                issues.append(dict(arm=arm,cohort=cohort,status='unavailable_or_failed_authentication'))
                continue
            for epoch,final in finalized.items():
                association=[]
                if row['checkpoint']==final['input_checkpoint']:association.append('input_checkpoint')
                if row['checkpoint']==final['output_checkpoint']:association.append('post_update')
                if association:by_epoch[epoch].append(dict(row,checkpoint_association=association))
    return by_epoch,issues
