"""Pure projection of retained pre-cached fixed32 evidence for epochs 14–18.

The signed public summary authenticates aggregate results and task identities.
Original reports/terminal files are unsigned; that limitation stays explicit.
No network, queue dispatch, models, pointers-as-execution, or other experiment scan.
"""
import hashlib
import json
import math
from pathlib import Path
import re

from dashboard.learner_projection import AUTHORITY, authenticated, canonical

BASE = Path('state/live-math-launch-preparation-v1/distributed-preparation/live-controller-v1')
CACHE = Path('state/root-audits/20261010-finalized-evidence-publication/legacy-evaluation-cache')
INDICES = [6903,3689,47,3166,5601,4899,2214,7211,4292,437,2893,6498,5356,2077,6873,4012,
           6812,4327,335,3087,6665,4971,1849,6925,4414,433,3083,5656,5447,2235,7363,3574]
HARNESS = dict(max_output_tokens=1024,policy='autoregressive',temperature=.7,top_p=1.,version='text-tools-long-v2')
SOURCES = {'6a85c31011b7a1ca23d745e9868d7b01dd0e92e9fe74c997e8222482f76393ca',
           '7459c28cbe11b0999b41644aea2aa672d40eb44faa94ccb261e176d8f71b0d46'}
EPOCH = re.compile(r'nonpayable-live-reward-math-v1--[0-9]+-(14|15|16|17|18)\Z')
HASH = re.compile(r'[0-9a-f]{64}\Z')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError('regular canonical legacy evidence')
    with path.open('rb') as stream:raw=stream.read(8*1024**2+1)
    if len(raw)>8*1024**2:raise ValueError('bounded legacy evidence')
    return json.loads(raw)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def job_binding(envelope):
    job=authenticated(envelope,AUTHORITY);manifest=authenticated(job['manifest'],AUTHORITY)
    epoch=manifest['epoch'];checkpoint=manifest['checkpoint']['id']
    if (not EPOCH.fullmatch(epoch) or not HASH.fullmatch(checkpoint) or job['role']!='evaluate'
            or not re.fullmatch(re.escape(epoch)+r'-eval-(before|after)-[0-9a-f]{8}',job['job_id'])
            or manifest['source_bundle']['sha256'] not in SOURCES
            or job['heldout'] != [dict(env_id='affine_math',harness=HARNESS,indices=INDICES,
                                     seeds=[20261002+i*1000 for i in INDICES])]
            or not finite(job['created_at']) or not finite(job['expires_at'])
            or not job['created_at']<job['expires_at']):
        raise ValueError('exact legacy fixed32 signed job')
    return job,manifest


def project_completed(envelope, job_envelope, report):
    summary=authenticated(envelope,AUTHORITY);job,manifest=job_binding(job_envelope)
    checkpoint=manifest['checkpoint']['id'];epoch=manifest['epoch'];phase=job['job_id'].split('-eval-',1)[1].split('-',1)[0]
    if (summary.get('version')!='independent-checkpoints-v1' or summary.get('status')!='complete'
            or summary.get('epoch')!=epoch or summary.get('checkpoint')!=checkpoint
            or summary.get('phase')!=phase or len(summary.get('records',[]))!=1):
        raise ValueError('signed original legacy evaluation summary')
    record=summary['records'][0]
    if (report.get('role')!='evaluate' or report.get('job_id')!=job['job_id']
            or report.get('job_sha256')!=digest(job) or report.get('epoch')!=epoch
            or report.get('checkpoint')!=checkpoint or report.get('source_files')!=job['source_files']
            or report.get('runtime_versions')!=job['runtime_versions'] or report.get('chain_transactions') is not False
            or not finite(report.get('completed_at'))
            or not job['created_at']<=report['completed_at']<job['expires_at']
            or record.get('remote_job_id')!=job['job_id'] or record.get('checkpoint')!=checkpoint
            or record.get('epoch_id')!=epoch or record.get('env_id')!='affine_math'
            or record.get('experiment_id')!='live-original-math94ae-fixed32-v1'
            or record.get('harness_config')!=HARNESS or record.get('heldout_indices')!=INDICES
            or record.get('timestamp')!=report['completed_at'] or summary['completed_at']!=report['completed_at']
            or record.get('payable') is not False or record.get('weight_submission') is not False):
        raise ValueError('original report and signed summary identity binding')
    suite=job['heldout'][0];definition=next(e for e in manifest['environments'] if e['env_id']=='affine_math')
    frozen=dict(env_id='affine_math',environment=definition['spec'],harness=HARNESS,
        indices=INDICES,seeds=suite['seeds'],model_runtime_revision=manifest['model_runtime_revision'],
        backend_profile=manifest['backend_profile'],runtime_versions=report['runtime_versions'],
        harness_source_hash=manifest['harness_source_hash'],source_files={n:report['source_files'][n] for n in
        ('subnet/model.py','subnet/gpu_runtime.py','subnet/environments.py','subnet/harness.py','subnet/proofs.py')})
    if record.get('dataset_id')!=digest(frozen) or record.get('taskset_hash')!=digest(frozen):
        raise ValueError('signed original fixed32 runtime cohort')
    rows=report.get('heldout',[]);failures=report.get('heldout_failures',[]);tasks=[];seen=set()
    wanted=set(zip(INDICES,suite['seeds']))
    for row in rows:
        key=(row['index'],row['seed']);label=row.get('classification')
        if (key not in wanted or key in seen or row.get('env_id')!='affine_math'
                or label not in ('positive','negative','neutral','unresolved')
                or not finite(row.get('reward')) or row['reward']!=(1 if label=='positive' else 0)
                or not HASH.fullmatch(row.get('task_hash','')) or type(row.get('verified')) is not bool):
            raise ValueError('legacy task identity and recorded verdict')
        seen.add(key)
        tasks.append(dict(index=key[0],seed=key[1],task_hash=row['task_hash'],verdict=label,reward=row['reward'],
            original_report_verified_flag=row['verified'],individual_result_authenticated=False,
            output_length=None,output_token_ids=None,output_text=None,stop_reason=None,turns=[],
            token_metadata_status='not_retained_in_original_report',max_output_tokens=1024))
    for failure in failures:
        key=(failure['index'],failure['seed'])
        if key not in wanted or key in seen or failure.get('env_id')!='affine_math':
            raise ValueError('legacy failed task identity')
        seen.add(key);tasks.append(dict(index=key[0],seed=key[1],verdict='evaluation_error',reward=None,
            individual_result_authenticated=False,output_length=None,output_token_ids=None,output_text=None,
            stop_reason=None,turns=[],error_message='not_published'))
    successes=sum(row['classification']=='positive' for row in rows)
    if (seen!=wanted or record.get('requested_count')!=32 or record.get('attempted_count')!=32
            or record.get('completed_count')!=len(rows) or record.get('count')!=len(rows)
            or record.get('successes')!=successes or record.get('task_hashes')!=[r['task_hash'] for r in rows]
            or record.get('fixed_task_ids')!=[r['task_hash'] for r in rows]
            or record.get('evaluation_failures')!=failures
            or record.get('status')!=('error' if failures else 'complete')
            or record.get('mean_reward')!=(None if failures else sum(r['reward'] for r in rows)/32)):
        raise ValueError('signed aggregate and complete original task denominator')
    tasks.sort(key=lambda row:INDICES.index(row['index']))
    return dict(suite='fixed32-legacy',checkpoint=checkpoint,original_epoch=epoch,job_id=job['job_id'],
        completed_at=report['completed_at'],requested_count=32,tasks=tasks,harness=HARNESS,
        original_phase=phase,summary_sha256=digest(envelope),job_sha256=digest(job),
        original_report_sha256=digest(report),source_sha256=manifest['source_bundle']['sha256'],
        cohort_sha256=digest(frozen),signed_summary_authenticated=True,individual_results_authenticated=False,
        original_report_provenance='unsigned_original_report_bound_to_signed_job_and_signed_aggregate',
        output_text_status='not_retained_in_original_report',output_length_status='not_retained_in_original_report',
        stop_reason_status='not_retained_in_original_report',
        limits=['The original public signature binds aggregate counts and task hashes, not individual verdict assignments.',
                'Checkpoint association reuses an original execution and does not create a new evaluation.'])


def project_failure(job_envelope, terminal):
    job,manifest=job_binding(job_envelope)
    if (terminal.get('job_id')!=job['job_id'] or terminal.get('phase')!='failed'
            or type(terminal.get('exit_code')) is not int or terminal['exit_code']==0
            or not finite(terminal.get('started_at')) or not finite(terminal.get('finished_at'))
            or not job['created_at']<=terminal['started_at']<=terminal['finished_at']):
        raise ValueError('retained original failed-job observation')
    return dict(suite='fixed32-legacy',checkpoint=manifest['checkpoint']['id'],original_epoch=manifest['epoch'],
        job_id=job['job_id'],requested_count=32,tasks=[],completed_at=None,observed_finished_at=terminal['finished_at'],
        status='infrastructure_failure_observed',infrastructure_error=True,task_outcomes_retained=False,
        observed_exit_code=terminal['exit_code'],harness=HARNESS,job_sha256=digest(job),
        terminal_sha256=digest(terminal),terminal_authenticated=False,
        provenance='unsigned_original_terminal_observation_bound_to_signed_job',
        limits=['A failed job does not establish task-level verdicts, token lengths, or EOS.'])


def collect_legacy_evaluation_evidence(source_root,finalized):
    root=Path(source_root).resolve();state=root/BASE/'controller-state';cache=root/CACHE
    selected={e:v for e,v in finalized.items() if EPOCH.fullmatch(e)}
    by_epoch={e:[] for e in selected};issues=[];collected=[]
    for epoch in sorted(selected):
        for phase in ('before','after'):
            public=cache/(epoch+'-evaluation-'+phase+'.json')
            try:
                if public.exists():
                    envelope=read(public);summary=authenticated(envelope,AUTHORITY)
                    job_id=summary['records'][0]['remote_job_id']
                    if not re.fullmatch(re.escape(epoch+'-eval-'+phase)+r'-[0-9a-f]{8}',job_id):
                        raise ValueError('safe scoped original job identity')
                    collected.append(project_completed(envelope,read(state/'roles'/(job_id+'-job.json')),
                        read(state/'roles'/(job_id+'-report.json'))))
                else:
                    failures=sorted((state/'roles').glob(epoch+'-eval-'+phase+'-*-failure.json'))
                    if len(failures)>8:raise ValueError('bounded original failure attempts')
                    for failure in failures:
                        job_id=failure.name.removesuffix('-failure.json')
                        collected.append(project_failure(read(state/'roles'/(job_id+'-job.json')),read(failure)))
                    if not failures:issues.append(dict(epoch_id=epoch,phase=phase,status='no_original_result_for_epoch_phase'))
            except (ValueError,KeyError,TypeError,OSError):
                issues.append(dict(epoch_id=epoch,phase=phase,status='unavailable_or_failed_authentication'))
    for epoch,final in selected.items():
        for row in collected:
            association=[]
            if row['checkpoint']==final['input_checkpoint']:association.append('input_checkpoint')
            if row['checkpoint']==final['output_checkpoint']:association.append('post_update')
            if association:by_epoch[epoch].append(dict(row,checkpoint_association=association))
    return by_epoch,issues
