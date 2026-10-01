"""Audit finalized continuous GPU epochs without running models or chain writes.

The operator authority is the trust anchor. Remote execution reports are
operator-collected evidence, not a cryptographic proof of GPU execution.
"""
import argparse
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

from subnet.backend_jobs import canonical, file_map, signed
from subnet.batches import unpack
from subnet.scoring import score
from subnet.storage import Bucket, Identity
from ops.check_service_evidence import direct_read_routes
from ops.check_epoch_evidence import require

# This published worker was reviewed to consume pairs cyclically in audit order.
# Later workers must report explicit per-update attribution; no silent downgrade.
LEGACY_CYCLIC_WORKER='01d458b9f39312ddebc750482963c038f85aba335f75b51be2739adaee016d8a'

def authenticated_training_pairs(manifest, job, train, metrics, fresh, authority):
    """Reconstruct attribution; historical pairs never enter miner scoring."""
    envelope=job.get('replay')
    if envelope is None:
        require('replay_training' not in train and 'replay_training' not in metrics and
                metrics.get('replay_inputs_sha256') is None, 'unexpected unsigned replay claim')
        return fresh
    from subnet.replay_training import admitted,merge_pairs,REVISION
    from subnet.verified_replay_pool import digest
    _,selection=admitted(manifest,envelope,authority)
    definitions={row['env_id']:row for row in manifest['environments']}
    historical=[]
    for entry in selection['selected']:
        target=digest(dict(environment_id=entry['environment_id'],
            environment_index=entry['environment_index'],task_hash=entry['task_hash'],
            positive_rollout_sha256=digest(entry['positive']),
            negative_rollout_sha256=digest(entry['negative'])))
        require(target==entry['target_sha256'], 'replay target content binding')
        historical.append((definitions[entry['environment_id']],entry['positive'],entry['negative']))
    pairs,targets=merge_pairs(fresh,historical,envelope['reuse_counts'])
    checks=[dict(env_id=e['environment_id'],index=e['environment_index'],
        target_sha256=e['target_sha256'],current_checkpoint=manifest['checkpoint']['id'],
        historical_probabilities_used_as_reference=False,
        fresh_current_numerical_native_verification=True) for e in selection['selected']
        if e['target_sha256'] in targets]
    expected=dict(revision=REVISION,checks=checks,pool_sha256=selection['pool_sha256'],
        proposed_reuse_increments={target:1 for target in targets},optimizer_performed=False)
    require(train.get('replay_training')==expected and metrics.get('replay_training')==expected
        and metrics.get('replay_inputs_sha256')==digest(envelope), 'signed replay execution binding')
    require(metrics['steps']>=len(pairs), 'every replay family consumed')
    return pairs

LONG_CONTEXT_REVISION='cuda-bf16-sdpa-flash-sm86-selective-head-common-v1'
LONG_CONTEXT_PROFILE=dict(device='cuda',dtype='bfloat16',attention='sdpa-flash-only',
    sm=[8,6],tf32=False,deterministic_algorithms=True,cublas_workspace_config=':4096:8',
    native_toploc_threads=2,torch_threads=2,max_context=32768,
    output_head='output-prediction-rows-full-vocabulary',candidate_score_reduction='numpy-float32-sum')
LONG_CONTEXT_NUMERICAL=dict(logprob_atol=1e-5,logprob_rtol=0,toploc_exp_mismatches=0,
    toploc_mant_err_mean=0,toploc_mant_err_median=0)


def unpack_authenticated_epoch(body, manifest):
    """Caller must first authenticate the signed public manifest and receipt.

    Historical epochs retain the 100 MB reader. Enlargement requires the exact
    reviewed long-context profile; an arbitrary signed size declaration is insufficient.
    """
    if manifest.get('model_runtime_revision') == LONG_CONTEXT_REVISION:
        require(canonical(manifest.get('backend_profile')) == canonical(LONG_CONTEXT_PROFILE) and
            canonical(manifest.get('numerical_policy')) == canonical(LONG_CONTEXT_NUMERICAL) and
            manifest.get('transport_policy') == 'direct-r2-v1' and
            canonical(manifest.get('artifact_policy')) == canonical(dict(compressed_bytes=250_000_000,raw_bytes=500_000_000)),
            'GPU qualified long-context artifact policy')
        return unpack(body, max_upload=250_000_000)
    require(not manifest.get('artifact_policy'), 'GPU unknown enlarged artifact policy')
    return unpack(body)


def check_source_bundle(body, descriptor, expected):
    require(len(body) == descriptor['size'] and len(body) <= 32*1024**2 and
            hashlib.sha256(body).hexdigest() == descriptor['sha256'], 'GPU worker archive bytes')
    observed={}; total=0
    with tarfile.open(fileobj=io.BytesIO(body),mode='r:gz') as archive:
        for member in archive:
            total+=member.size
            require(total <= 256*1024**2, 'GPU worker archive expanded budget')
            name=member.name[2:] if member.name.startswith('./') else member.name
            if name not in expected:continue
            require(member.isfile() and not member.issym() and member.size <= 1024**2 and
                    name not in observed, 'GPU worker archive source entry')
            with archive.extractfile(member) as stream:
                value=stream.read(1024**2+1)
            require(len(value) == member.size, 'GPU worker source size')
            observed[name]=hashlib.sha256(value).hexdigest()
    require(observed == expected, 'GPU worker reproducible source inventory')
    return dict(sha256=descriptor['sha256'], bytes=len(body), source_files=len(observed))


def checked_job(state, identifier, authority):
    job = signed(json.loads((state/'roles'/f'{identifier}-job.json').read_text()), authority)
    manifest = signed(job['manifest'], authority)
    report = json.loads((state/'roles'/f'{identifier}-report.json').read_text())
    expected = dict(job_id=identifier, role=job['role'], operator=authority,
        checkpoint=manifest['checkpoint']['id'], epoch=manifest['epoch'],
        source_files=job['source_files'], runtime_versions=job['runtime_versions'],
        backend_profile=manifest['backend_profile'], numerical_policy=manifest['numerical_policy'],
        job_sha256=hashlib.sha256(canonical(job)).hexdigest(), success=True, chain_transactions=False)
    require(all(report.get(k) == v for k, v in expected.items()), 'GPU job/report binding')
    require(job['created_at'] <= report['completed_at'] < job['expires_at'], 'GPU job expiry')
    require(file_map(manifest['checkpoint']['files']) == manifest['checkpoint']['id'], 'GPU input checkpoint')
    return job, manifest, report


def check_aborted_training(status,manifest,accepted_count):
    admission=status.get('status')=='aborted_training_admission'
    require(status.get('status') in ('aborted_evaluation','aborted_training_admission') and
        status.get('epoch')==manifest['epoch'] and
        status.get('checkpoint')==status.get('next_checkpoint')==manifest['checkpoint']['id'] and
        status.get('optimizer_ran') is False and type(status.get('steps')) is int and status['steps']==0 and
        status.get('payable') is False and status.get('chain_transactions') is False and
        type(status.get('fully_audited_batches')) is int and status['fully_audited_batches']==accepted_count and
        isinstance(status.get('failed_jobs'),list) and bool(status['failed_jobs']) and
        all(isinstance(j,str) and j.startswith(manifest['epoch']+('-train-' if admission else '-eval-before-')) for j in status['failed_jobs']),
        'GPU aborted epoch unchanged checkpoint/no training binding')
    if admission:
        require(status.get('before_evaluation_complete') is True and
            status.get('remote_model_or_optimizer_launch') is False and
            status.get('admission_failure')=='missing_signed_fixed_reference_training_policy',
            'GPU training admission abort before model launch')


def check_empty_closed(status,manifest,scores,accepted_count):
    require(status.get('status')=='closed_no_accepted_batches' and
        status.get('epoch')==manifest['epoch'] and status.get('checkpoint')==manifest['checkpoint']['id'] and
        status.get('payable') is False and type(accepted_count) is int and accepted_count==0 and
        type(scores.get('total')) is int and scores['total']==0 and scores.get('points')=={} and scores.get('weights')=={},
        'GPU empty epoch unchanged checkpoint and zero verified reward')


def epoch_pending(abort, empty_status, metrics_exist, after_count, environment_count):
    return abort is None and (after_count != environment_count or
        (empty_status is None and not metrics_exist))


def bind_empty_recovery(empty, completed):
    verified={r['epoch']:r for r in completed}
    for row in empty:
        successor=verified.get(row['next_epoch'])
        row['recovery_verified']=bool(successor and successor['steps']>0 and
            sum(successor['points'].values())>0 and successor['checkpoint']!=row['checkpoint'])
        row['recovery_checkpoint']=successor['checkpoint'] if row['recovery_verified'] else None


def inspect(state, bucket, evaluations):
    authority = Identity(bytes.fromhex((state/'authority.seed').read_text().strip())).id
    def public(key):
        return signed(json.loads(bucket.get(key)), authority)
    manifests = sorted((json.loads(p.read_text()) for p in state.glob('*-manifest.json')
                        if not p.name.endswith('-audit-manifest.json')), key=lambda m: m['start'])
    completed, pending, aborted, empty, source_checks = [], [], [], [], {}
    for manifest in manifests:
        epoch = manifest['epoch']; prefix = f'public/{epoch}/'
        metrics_path = state/f'{epoch}-training-metrics.json'
        after_paths = list(evaluations.glob(f'{epoch}-eval-after-*.json'))
        abort_paths=[p for p in (state/f'{epoch}-aborted-evaluation.json',
            state/f'{epoch}-aborted-training-admission.json') if p.exists()]
        require(len(abort_paths)<=1, 'GPU conflicting abort statuses')
        abort=signed(json.loads(abort_paths[0].read_text()),authority) if abort_paths else None
        empty_path=state/f'{epoch}-empty-closed.json'
        empty_status=json.loads(empty_path.read_text()) if empty_path.exists() else None
        require(not (abort is not None and empty_status is not None), 'GPU conflicting empty/abort statuses')
        if epoch_pending(abort,empty_status,metrics_path.exists(),len(after_paths),len(manifest['environments'])):
            pending.append(epoch); continue
        require(public(prefix+'manifest.json') == manifest, 'GPU public manifest')
        require(manifest['payable'] is False and direct_read_routes(manifest), 'GPU nonpayable/direct policy')
        scores = public(prefix+'scores.json'); challenge = public(prefix+'audit-challenge.json')
        require(scores == json.loads((state/f'{epoch}-scores.json').read_text()), 'GPU local/public scores')
        require(scores['checkpoint'] == manifest['checkpoint']['id'] and scores['payable'] is False,
                'GPU score checkpoint/payout binding')
        require(scores['finalized_at'] >= manifest['deadline'] and
                challenge['generated_after_freeze_at'] >= manifest['deadline'] and
                challenge['receipts'] == scores['receipts'], 'GPU frozen challenge')
        reports = {}; frozen = []
        for miner, receipt in scores['receipts'].items():
            require(manifest['start'] <= receipt['received_at'] < manifest['deadline'], 'GPU upload deadline')
            body = bucket.get(receipt['frozen_key'])
            require(len(body) == receipt['size'] and hashlib.sha256(body).hexdigest() == receipt['sha256'],
                    'GPU frozen bytes')
            audit = public(prefix+f'audits/{miner}.json')
            _, audit_manifest, remote = checked_job(state, audit['remote_job_id'], authority)
            require(audit_manifest == dict(manifest, audit_seed=challenge['seed'],
                audit_frozen_receipts=scores['receipts']), 'GPU verifier challenge binding')
            raw = remote['audits'][0]
            require(all(audit.get(k) == v for k, v in raw.items()) and
                    audit['submission_sha256'] == receipt['sha256'], 'GPU independent audit report')
            batches = [b for b, arrays in unpack_authenticated_epoch(body, manifest)]
            accepted = [batches[o['batch']] for o in audit['outcomes'] if o.get('valid')]
            require(accepted == audit['accepted'] and all(o.get('fully_audited')
                    for o in audit['outcomes'] if o.get('valid')), 'GPU accepted frozen batch binding')
            reports[miner] = audit
            frozen.append(dict(miner=miner, bytes=len(body), sha256=receipt['sha256'], accepted=len(accepted)))
        calculated = score(reports)
        require(all(scores[k] == calculated[k] for k in calculated), 'GPU recomputed scores')
        if empty_status is not None:
            if manifest.get('operator_test_policy') is not None:
                from subnet.empty_epoch_policy import dispatch_allowed
                require(not dispatch_allowed(manifest), 'GPU signed controlled empty policy')
                require(not list((state/'roles').glob(epoch+'-mine-*-job.json')),
                    'GPU controlled empty window cannot dispatch a miner')
            require(not metrics_path.exists(), 'GPU empty epoch cannot claim optimizer metrics')
            check_empty_closed(empty_status,manifest,scores,sum(len(r['accepted']) for r in reports.values()))
            proposed=json.loads((state/f'{epoch}-proposed-weights.json').read_text())
            require(proposed.get('weights')=={} and proposed.get('payable') is False and
                proposed.get('chain_transactions') is False, 'GPU empty epoch no proposed payout')
            if after_paths:
                require(len(after_paths)==len(manifest['environments']), 'GPU incomplete empty-epoch evaluation')
                for definition in manifest['environments']:
                    env=definition['env_id']
                    records=[json.loads((evaluations/f'{epoch}-eval-{phase}-{env}.json').read_text())
                        for phase in ('before','after')]
                    require(all(records[0][k]==records[1][k] for k in
                        ('dataset_id','task_hashes','runtime_profile','harness_config')), 'GPU empty epoch comparable evaluations')
                    for record in records:
                        eval_job,eval_manifest,remote=checked_job(state,record['remote_job_id'],authority)
                        values=[v for v in remote['heldout'] if v['env_id']==env]
                        plan=next(s for s in eval_job['heldout'] if s['env_id']==env)
                        require(record['checkpoint']==eval_manifest['checkpoint']['id']==manifest['checkpoint']['id'] and
                            record['status']=='complete' and not record['evaluation_failures'] and
                            record['count']==record['requested_count']==len(values)>0 and
                            plan['indices']==record['heldout_indices'] and plan['harness']==record['harness_config'] and
                            sorted(zip(plan['indices'],plan['seeds']))==sorted((v['index'],v['seed']) for v in values) and
                            record['task_hashes']==[v['task_hash'] for v in values] and
                            record['mean_reward']==sum(v['reward'] for v in values)/len(values),
                            'GPU empty epoch evaluations use unchanged checkpoint')
            following=next((m for m in manifests if m['start']>=manifest['deadline'] and m['epoch']!=epoch),None)
            if following:
                require(public(f'public/{following["epoch"]}/manifest.json')==following and
                    following['checkpoint']['id']==manifest['checkpoint']['id'] and following['payable'] is False,
                    'GPU empty epoch unchanged signed checkpoint handover')
            empty.append(dict(epoch=epoch,status=empty_status['status'],checkpoint=manifest['checkpoint']['id'],
                accepted_batches=0,steps=0,completed_training_epoch=False,frozen=frozen,
                next_epoch=following['epoch'] if following else None))
            continue
        if abort is not None:
            require(not metrics_path.exists(), 'GPU aborted epoch cannot also claim optimizer metrics')
            require(not after_paths, 'GPU untrained abort cannot claim successor evaluations')
            check_aborted_training(abort,manifest,sum(len(r['accepted']) for r in reports.values()))
            if abort['status']=='aborted_training_admission':
                for identifier in abort['failed_jobs']:
                    failed_job=signed(json.loads((state/'roles'/f'{identifier}-job.json').read_text()),authority)
                    require(failed_job['job_id']==identifier and failed_job['role']=='train' and
                        signed(failed_job['manifest'],authority)==manifest,
                        'GPU rejected training job signed epoch binding')
                for definition in manifest['environments']:
                    env=definition['env_id']
                    record=json.loads((evaluations/f'{epoch}-eval-before-{env}.json').read_text())
                    eval_job, eval_manifest, remote=checked_job(state,record['remote_job_id'],authority)
                    values=[v for v in remote['heldout'] if v['env_id']==env]
                    plan=next(s for s in eval_job['heldout'] if s['env_id']==env)
                    require(record['checkpoint']==eval_manifest['checkpoint']['id']==manifest['checkpoint']['id'] and
                        record['status']=='complete' and not record['evaluation_failures'] and
                        record['count']==record['requested_count']==len(values)>0 and
                        plan['indices']==record['heldout_indices'] and plan['harness']==record['harness_config'] and
                        sorted(zip(plan['indices'],plan['seeds']))==sorted((v['index'],v['seed']) for v in values) and
                        record['task_hashes']==[v['task_hash'] for v in values] and
                        record['mean_reward']==sum(v['reward'] for v in values)/len(values),
                        'GPU admission abort retains complete authenticated baseline evaluation')
            require(public(prefix+'training.json')==abort and
                json.loads((state/f'{epoch}-training.json').read_text())==abort,
                'GPU aborted evaluation signed public status')
            aborted.append(dict(epoch=epoch,status=abort['status'],frozen=frozen,
                accepted_batches=abort['fully_audited_batches'],checkpoint=abort['checkpoint'],
                steps=0,completed_training_epoch=False))
            continue
        metrics = json.loads(metrics_path.read_text())
        require(public(prefix+'training.json') == metrics, 'GPU training publication')
        train_job, training_manifest, train = checked_job(state, metrics['remote_job_id'], authority)
        bundle=manifest['source_bundle']
        source_key=(bundle['sha256'],hashlib.sha256(canonical(train_job['source_files'])).hexdigest())
        if source_key not in source_checks:
            source_checks[source_key]=check_source_bundle(bucket.get(bundle['key']),bundle,train_job['source_files'])
        require(training_manifest == manifest and metrics['steps'] > 0 and
                metrics['full_model_finetune'] is True and metrics['weights_changed'] is True and
                metrics['checkpoint'] != manifest['checkpoint']['id'], 'GPU real full-model update')
        require(train['new_checkpoint']['id'] == metrics['checkpoint'] and
                train['training']['steps'] == metrics['steps'] and
                train['training']['updates'] == metrics['updates'], 'GPU training result binding')
        expected_audits = {r['submission_sha256']:r for r in reports.values() if r['accepted']}
        require({s['sha256'] for s in train_job['submissions']} == set(expected_audits) and
                {a['submission_sha256'] for a in train['audits']} == set(expected_audits),
                'GPU training frozen submissions')
        for audit in train['audits']:
            require(all(expected_audits[audit['submission_sha256']].get(k) == v
                        for k,v in audit.items()), 'GPU training independently verified pairs')
        training_pairs=[]
        for audit in train['audits']:
            for batch in audit['accepted']:
                positives=[r for r in batch['rollouts'] if r['classification']=='positive']
                negatives=[r for r in batch['rollouts'] if r['classification']=='negative']
                training_pairs.extend((batch,p,n) for p,n in zip(positives,negatives))
        training_pairs=authenticated_training_pairs(manifest,train_job,train,metrics,
            training_pairs,authority)
        require(training_pairs and len(metrics['updates'])==metrics['steps'], 'GPU optimizer update count')
        optimized=[]
        for step,update in enumerate(metrics['updates']):
            batch,pos,neg=training_pairs[step%len(training_pairs)]
            expected=dict(attribution_revision='verified-pair-v1',optimizer_step=step+1,
                env_id=batch['env_id'],index=batch['index'],
                positive_rollout_sha256=hashlib.sha256(canonical(pos)).hexdigest(),
                negative_rollout_sha256=hashlib.sha256(canonical(neg)).hexdigest())
            if 'attribution_revision' in update:
                require(all(update.get(k)==v for k,v in expected.items()), 'GPU explicit training attribution')
            else:
                require(train_job['source_files']['subnet/backend_jobs.py']==LEGACY_CYCLIC_WORKER,
                        'GPU unknown worker missing training attribution')
            optimized.append(dict(expected,attribution_evidence='explicit-report' if
                'attribution_revision' in update else 'derived-from-pinned-cyclic-worker'))
        descriptor = public(metrics['new_checkpoint']['descriptor_key'])
        require(descriptor['id'] == metrics['checkpoint'] == file_map(descriptor['files']) and
                descriptor['files'] == metrics['new_checkpoint']['files'], 'GPU signed output checkpoint')
        publication = json.loads((state/f'{epoch}-checkpoint-publication.json').read_text())
        require(publication['checkpoint'] == descriptor['id'] and publication['operator_independent_hashes']
                and {k:v['sha256'] for k,v in publication['objects'].items()} == descriptor['files'],
                'GPU operator-streamed published bytes')
        pairs = []
        for definition in manifest['environments']:
            env = definition['env_id']
            records = [json.loads((evaluations/f'{epoch}-eval-{phase}-{env}.json').read_text())
                       for phase in ('before', 'after')]
            before, after = records
            require(before['checkpoint'] == manifest['checkpoint']['id'] and
                    after['checkpoint'] == metrics['checkpoint'], 'GPU heldout checkpoint')
            require(all(before[k] == after[k] for k in ('dataset_id','task_hashes','runtime_profile','harness_config')),
                    'GPU comparable heldouts')
            for record in records:
                eval_job, eval_manifest, remote = checked_job(state, record['remote_job_id'], authority)
                values = [v for v in remote['heldout'] if v['env_id'] == env]
                plan = next(s for s in eval_job['heldout'] if s['env_id'] == env)
                require(sorted(zip(plan['indices'],plan['seeds'])) ==
                        sorted((v['index'],v['seed']) for v in values) and
                        plan['indices'] == record['heldout_indices'] and
                        plan['harness'] == record['harness_config'], 'GPU fixed heldout plan')
                require(record['status'] == 'complete' and not record['evaluation_failures'] and
                        record['count'] == record['requested_count'] == len(values) and
                        record['task_hashes'] == [v['task_hash'] for v in values] and
                        record['mean_reward'] == sum(v['reward'] for v in values)/len(values) and
                        eval_manifest['checkpoint']['id'] == record['checkpoint'], 'GPU actual heldout report')
            pairs.append(dict(env_id=env, before=before['mean_reward'], after=after['mean_reward'], count=before['count']))
        following = next((m for m in manifests if m['start'] >= manifest['deadline'] and m['epoch'] != epoch), None)
        if following:
            require(public(f'public/{following["epoch"]}/manifest.json') == following and
                    following['checkpoint']['id'] == metrics['checkpoint'], 'GPU next epoch handover')
        completed.append(dict(epoch=epoch, frozen=frozen, steps=metrics['steps'], checkpoint=metrics['checkpoint'],
            points=scores['points'], weights=scores['weights'], heldout_pairs=pairs,
            worker_source=source_checks[source_key],
            optimized_pairs=optimized,
            next_epoch=following['epoch'] if following else None))
    bind_empty_recovery(empty,completed)
    return dict(timestamp=time.time(), success=True, epochs=completed, pending_epochs=pending, aborted_epochs=aborted,empty_epochs=empty,
        authority=authority, chain_write_operations=0, fresh_model_execution_in_this_check=False,
        remote_reports_are_operator_collected=True, published_bytes_evidence='operator_stream_hashes', goal_complete=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, default=Path('state/gpu-continuous'))
    p.add_argument('--bucket-config', type=Path, default=Path('state/r2-direct.json'))
    p.add_argument('--evaluations', type=Path, default=Path('state/evaluations'))
    a = p.parse_args(); result = inspect(a.state, Bucket(json.loads(a.bucket_config.read_text())), a.evaluations)
    output = a.state/'root-continuous-independent-evidence.json'
    output.write_bytes(canonical(result)); output.chmod(0o600)
    print(json.dumps({'epochs_verified':len(result['epochs']), 'pending_epochs':result['pending_epochs'],
        'aborted_epochs':len(result['aborted_epochs']),'empty_epochs':len(result['empty_epochs'])}))


if __name__ == '__main__':
    main()
