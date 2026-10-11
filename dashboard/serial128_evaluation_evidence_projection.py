"""Read-only projection of completed original serial128 math diagnostics.

Only the existing signed monitor and its original fixed cohort are admitted.
No dispatch, model imports, network, private-experiment discovery or token decode.
"""
import json
from pathlib import Path
import re
from dashboard import endpoint_evaluation_evidence_projection as e

MONITOR = Path('/tmp/affine-serial-heldout128-monitor-20261009-v1')
STUDY = Path('/tmp/affine-fresh-run-heldout128-20261009-v1')
PROGRAM = '0952423ef6f5f5016b82e37c5c069f41e7567f9b89f0ec847e41084c65e2dd9b'
BASE = '6493a901bd009f0800d5eed97d19aee78947cc586afeb27abfb2c72032ad1924'
FIXED = ('suites','cohort_sha256','source_files','runtime_versions','manifest','scientific_files','program_sha256')
JOB = re.compile(r'step-([0-9]+)-([0-9a-f]{16})\Z')


def document(path):
    return e.authenticate(json.loads(e.read(Path(path))))


def configuration():
    path=MONITOR/'config.ROOT-SIGNED.private.json';c=document(path)
    expected=dict(version='serial-heldout128-monitor-v1',local_root=str(MONITOR/'state'),
        study_root=str(STUDY),base_plan=str(STUDY/'base-plan.ROOT-SIGNED.private.json'),
        base_archive=str(STUDY/'base-archive.ROOT-SIGNED.private.json'),
        base_raw_directory=str(STUDY/'base-captured-output'),evaluate_program=str(STUDY/'evaluate.py'))
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('exact existing serial monitor scope')
    # Never copy this operator config: it contains credential-path locators.
    base_raw=e.read(Path(c['base_plan']));base=e.authenticate(json.loads(base_raw))
    if (c['pinned_files'].get(c['base_plan'])!=e.sha(base_raw) or base['checkpoint']['id']!=BASE
        or c['pinned_files'].get(c['evaluate_program'])!=PROGRAM):raise ValueError('original serial anchor and program')
    return c,base,e.sha(e.read(path))


def plan_scope(plan,base):
    if (plan.get('version')!='fresh-run-disjoint-heldout128-v1' or plan.get('task_count')!=128
        or plan.get('production_mutation')is not False or plan.get('optimizer_allocation')is not False
        or plan.get('program_sha256')!=PROGRAM or any(k not in base or k not in plan or plan[k]!=base[k] for k in FIXED)
        or not e.HASH.fullmatch(plan['checkpoint']['id']) or e.digest(plan['checkpoint']['files'])!=plan['checkpoint']['id']
        or not e.finite(plan.get('created_at')) or not e.finite(plan.get('expires_at'))
        or not 0<plan['expires_at']-plan['created_at']<=7200):raise ValueError('exact original serial128 plan')
    for name,h in [('cached_sampling',e.SAMPLER),('owned_cached_evaluation',e.OWNED)]:
        key='subnet/'+name+'.py'
        if plan['source_files'].get(key)!=h or plan['scientific_files'].get(key)!=h:raise ValueError('serial exact sampler source')
    manifest=e.authenticate(plan['manifest']);definition=next(x for x in manifest['environments']if x['env_id']=='affine_math')
    if definition['spec']['max_turns']!=1 or len(plan['suites'])!=4:raise ValueError('serial single-turn original128')
    ordered=[];hashes=[]
    for suite in plan['suites']:
        indices,seeds=suite['indices'],suite['seeds']
        if (suite['env_id']!='affine_math' or suite['harness']!=e.HARNESS or len(indices)!=32 or len(seeds)!=32
            or any(type(v)is not int or v<0 for v in indices+seeds)):raise ValueError('serial exact original suites')
        h=e.digest(dict(version='owned-cached-native-evaluation-v1',env_id=suite['env_id'],environment=definition['spec'],harness=e.HARNESS,
            indices=indices,seeds=seeds,model_runtime_revision=manifest['model_runtime_revision'],backend_profile=manifest['backend_profile'],source_files=plan['scientific_files']))
        hashes.append(h);ordered.extend((i,s,h)for i,s in zip(indices,seeds))
    indices={i for i,_,_ in ordered}
    if (len(indices)!=128 or hashes!=plan['cohort_sha256'] or indices&set(definition['indices'])
        or indices&set(plan['excluded_diagnostic_indices']) or not indices<=set(plan['reserved_indices'])):raise ValueError('serial fixed disjoint cohort')
    return ordered


def project_original(plan_path,archive_path,captured,base,finalized_checkpoints,*,step,assignment_path=None):
    plan_raw=e.read(plan_path);envelope=json.loads(plan_raw);plan=e.authenticate(envelope);checkpoint=plan['checkpoint']['id']
    # This precedes archive/task-body reads, including any unfinished/open run.
    if checkpoint not in finalized_checkpoints:return None
    ordered=plan_scope(plan,base);archive_raw=e.read(archive_path);archive=e.authenticate(json.loads(archive_raw));plan_sha=e.digest(envelope)
    if (archive.get('version')!='prospective-heldout128-durable-readback-v1' or archive.get('full_readback_verified')is not True
        or archive.get('evaluation_lease_free_at_capture')is not True or archive.get('production_mutation')is not False
        or archive.get('optimizer_allocation')is not False or archive.get('checkpoint')!=checkpoint
        or archive.get('task_count')!=128 or archive.get('plan_sha256')!=plan_sha or not e.finite(archive.get('at'))):raise ValueError('serial complete signed original archive')
    pins=archive['archive'];plan_names=[n for n in ('plan.json','plan.ROOT-SIGNED.private.json')if n in pins]
    if len(plan_names)!=1 or pins[plan_names[0]]['sha256']!=e.sha(plan_raw) or pins[plan_names[0]]['size']!=len(plan_raw):raise ValueError('serial archive original plan bytes')
    if pins.get('evaluate.py',{}).get('sha256')!=PROGRAM:raise ValueError('serial archived exact program')
    assignment_sha=None
    if assignment_path is not None:
        raw=e.read(assignment_path);assignment=e.authenticate(json.loads(raw));assignment_sha=e.sha(raw)
        if assignment.get('step')!=step or assignment.get('checkpoint')!=checkpoint:raise ValueError('serial original assignment checkpoint')
    def captured_raw(name,maximum=e.MAX_TASK):
        raw=e.read(captured/name,maximum);pin=pins[name]
        if pin['sha256']!=e.sha(raw) or pin['size']!=len(raw):raise ValueError('serial full retained artifact SHA')
        return raw
    result_raw=captured_raw('result.json',e.MAX_FILE);summary=json.loads(result_raw)
    if (archive.get('result_sha256')!=e.sha(result_raw) or summary.get('version')!=plan['version']
        or summary.get('plan_sha256')!=plan_sha or summary.get('checkpoint')!=checkpoint or summary.get('tasks')!=128
        or summary.get('cohort_sha256')!=plan['cohort_sha256'] or summary.get('production_mutation')is not False
        or summary.get('optimizer_allocation')is not False or not e.finite(summary.get('completed_at'))
        or not plan['created_at']<=summary['completed_at']<plan['expires_at'] or summary['completed_at']>archive['at']):raise ValueError('serial original terminal summary')
    names={f'task-{i}.json'for i,_,_ in ordered}
    if set(summary['raw_artifacts'])!=names or {n for n in pins if n.startswith('task-')}!=names:raise ValueError('serial complete128 artifact denominator')
    rows=[]
    for index,seed,cohort in ordered:
        name=f'task-{index}.json';raw=captured_raw(name)
        if e.sha(raw)!=summary['raw_artifacts'][name]:raise ValueError('serial task summary SHA')
        row=e.project_task(json.loads(raw),index=index,seed=seed,checkpoint=checkpoint,cohort_sha256=cohort);row['artifact_sha256']=e.sha(raw);rows.append(row)
    counts={label:sum(r['outer_classification']==label for r in rows)for label in ('positive','negative','unresolved','error')}
    if (summary['correct']!=counts['positive'] or summary['errors']!=counts['error'] or summary['unresolved']!=counts['unresolved']
        or summary['accuracy']!=counts['positive']/128 or summary['cap_hits']!=sum(r['cap_hit']for r in rows)
        or summary['budget_exhausted']!=sum(r['budget_exhausted']for r in rows)):raise ValueError('serial exact original summary counts')
    return dict(suite='serial-heldout128',study='existing-fp32-serial-diagnostic',step=step,checkpoint=checkpoint,requested_count=128,tasks=rows,
        harness=e.HARNESS,completed_at=summary['completed_at'],output_text_status='retained',original_wrapper_counts=counts,
        native_unresolved_or_incomplete_count=sum(r['unresolved_or_incomplete']for r in rows),infrastructure_error_count=sum(r['infrastructure_error']for r in rows),
        plan_sha256=e.sha(plan_raw),archive_sha256=e.sha(archive_raw),summary_sha256=e.sha(result_raw),assignment_sha256=assignment_sha,
        cohort_sha256=plan['cohort_sha256'],scientific_files_sha256=e.digest(plan['scientific_files']),program_sha256=PROGRAM,
        proof_verification_performed=False,model_calls_by_exporter=0,historical_independence_proven=False,
        limits=['Exposed repeated serial diagnostic; this publication does not establish convergence.',
            'Baseline6493 was already trained; it is not pristine upstream2173 or the separate fixed-study g26 baseline.',
            'Original raw output and pinned sampler budget flags establish stopping metadata; no tokenizer/model replay is performed.',
            'Association to an epoch checkpoint does not imply the evaluation ran during that epoch.'])


def collect_serial128_evidence(source_root,finalized):
    result={epoch:[]for epoch in finalized};issues=[]
    if not(MONITOR/'config.ROOT-SIGNED.private.json').is_file():return result,[dict(kind='serial128',status='original_monitor_not_retained')]
    try:c,base,config_sha=configuration()
    except Exception:return result,[dict(kind='serial128',status='monitor_scope_unavailable_or_failed_authentication')]
    checkpoints={r[k]for r in finalized.values()for k in ('input_checkpoint','output_checkpoint')}
    sources=[(0,Path(c['base_plan']),Path(c['base_archive']),Path(c['base_raw_directory']),None)]
    legacy=c.get('completed_legacy_model',{});expected_plan=STUDY/'learned-step1-plan.ROOT-SIGNED.private.json';expected_archive=STUDY/'learned-step1-archive.ROOT-SIGNED.private.json'
    if legacy.get('plan')==str(expected_plan)and legacy.get('archive')==str(expected_archive):
        sources.append((1,expected_plan,expected_archive,STUDY/'learned-step1-captured-output',None))
    jobs=sorted((MONITOR/'state/jobs').glob('step-*'))
    if len(jobs)>4096:return result,[dict(kind='serial128',status='original_job_inventory_bound')]
    for directory in jobs:
        match=JOB.fullmatch(directory.name)
        if not match or directory.is_symlink()or not directory.is_dir():continue
        if(directory/'archive.json').is_file():sources.append((int(match[1]),directory/'plan.json',directory/'archive.json',directory/'captured',directory/'assignment.json'))
    seen=set()
    for step,plan,archive,captured,assignment in sources:
        if not archive.is_file():continue
        try:
            row=project_original(plan,archive,captured,base,checkpoints,step=step,assignment_path=assignment)
            if row is None:continue
            if row['plan_sha256']in seen:continue
            seen.add(row['plan_sha256']);row['monitor_config_sha256']=config_sha
        except Exception:
            issues.append(dict(kind='serial128',step=step,status='unavailable_or_failed_authentication'));continue
        for epoch,final in finalized.items():
            association=[name for key,name in [('input_checkpoint','input_checkpoint'),('output_checkpoint','post_update')]if row['checkpoint']==final[key]]
            if association:result[epoch].append(dict(row,checkpoint_association=association))
    return result,issues
